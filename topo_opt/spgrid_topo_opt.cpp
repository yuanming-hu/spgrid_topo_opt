#include <taichi/geometry/mesh.h>
#include <taichi/testing.h>
#include "spgrid_topo_opt.h"
#include "hex_fem.h"

TC_NAMESPACE_BEGIN

TC_INTERFACE_DEF(Objective, "objective");
TC_INTERFACE_DEF(Optimizer, "optimizer");

void SPGridTopologyOptimization3D::initialize(const Config &config) {
  Base::initialize(config);

  fixed_cell_density_ = config.get("fixed_cell_density", 1.0_f);

  last_iter = -1;

  this->solver_state_ptr = nullptr;
  this->config = config;

  block_size =
      Vector3i(1 << ChannelMask::block_xbits, 1 << ChannelMask::block_ybits,
               1 << ChannelMask::block_zbits);

  fix_cells_near_force = config.get("fix_cells_near_force", false);
  fix_cells_at_dirichlet = config.get("fix_cells_at_dirichlet", false);

  auto res = config.get<Vector3i>("res");
  TC_P(res);
  dx = 1.0_f / res[0];
  inv_dx = 1.0_f / dx;

  auto spgrid_res = res + Vectori(padding * 2 + 16);

  set_grid(std::make_unique<TopOptGrid>(spgrid_res));
  grid->verbose = config.get("verbose_snapshot", false);

  volume_fraction = config.get("volume_fraction", 0.1_f);
  minimum_density = config.get("minimum_density", 0.00_f);
  minimum_stiffness = config.get("minimum_stiffness", 1e-9_f);
  if (config.get("progressive_penalty", 0)) {
    if (minimum_stiffness != 0) {
      TC_ERROR(
          "nonzero progressive panlty is allowed only when minimum_stiffness = "
          "0.");
    }
  }
  TC_ERROR_IF(
      minimum_stiffness == 0 && minimum_density == 0,
      "minimum_stiffness and minimum_density cannot be zero simultaneously.");
  TC_ASSERT(volume_fraction > minimum_density);
  TC_WARN_IF(volume_fraction < minimum_density * 5,
             "Volume fraction is too low for this minimum_density. Ideally it "
             "should be 5x larger.");

  if (config.get("use_youngs", true)) {
    material =
        Material(config.get<real>("E", 1.0e6_f), config.get<real>("nu", 0.3_f));
  } else {
    material.set_lame(config.get<real>("mat_lambda"),
                      config.get<real>("mat_mu"));
  }

  initialize_fem_kernel();

  optimizer = create_instance_unique_ctor<Optimizer>("oc", Dict("opt", this));
  if (!config.get("exclude_minimal_compliance", false)) {
    objectives.emplace_back(create_instance_unique_ctor<Objective>(
        "minimal_compliance", (Dict("opt", this))));
    active_objective = objectives[0].get();
  }
}

void Opt::initialize_fem_kernel() {
  Config cfg = config;
  cfg.set("material", &material);
  cfg.set("res", Vectori(16));
  cfg.set("solver_type", config.get<std::string>("solver_type", "gmg"));
  cfg.set("cg_tolerance", config.get<real>("cg_tolerance", 1e-5_f));
  cfg.set("cg_min_iterations", config.get<int>("cg_min_iterations", 1));
  cfg.set("cg_max_iterations", config.get<int>("cg_max_iterations", 100000));
  cfg.set("verbose", false);
  cfg.set("penalty", 3);
  cfg.set("num_threads", config.get("num_threads", -1));
  cfg.set("print_residuals", config.get<bool>("print_residuals", false));
  cfg.set("e_min", config.get<real>("e_min", 1e-6_f));
  cfg.set("use_preconditioner", config.get<bool>("use_preconditioner", true));
  fem = std::make_unique<HexFEMSolver3D>();
  fem->initialize(cfg);
}

void SPGridTopologyOptimization3D::update_thin_page_map() {
  auto active_threshold = config.get<real>("active_threshold", 1e-6_f);
  // Compute block density bounds
  auto sparse_density = grid->density();
  auto flags = grid->flags();
  TC_INFO("# blocks (before pruning) = {}", fat_page_map->Get_Blocks().second);
  thin_page_map->Clear();
  block_meta.resize(fat_page_map->Get_Blocks().second);
  parallel_for_each_block(*fat_page_map, [&](int b, uint64 block_offset,
                                             const Vector3i &base_coord) {
    BlockMeta m;
    bool has_force = false;
    Region region(Vectori(0), block_size);
    for (auto ind : region) {
      Vectori coord = base_coord + ind.get_ipos();
      real d = sparse_density(coord);
      m.density_min = std::min(m.density_min, d);
      m.density_max = std::max(m.density_max, d);
      m.density_sum += d;
      has_force |= flags(coord).get_has_force();
    }
    if (m.density_max >= minimum_density + active_threshold || has_force) {
      thin_page_map->Set_Page(block_offset);
    }
    block_meta[b] = m;
  });
  thin_page_map->Update_Block_Offsets();
  TC_INFO("# blocks (after pruning) = {}", thin_page_map->Get_Blocks().second);
  real fraction_to_keep = config.get("fraction_to_keep", 1.0_f);
  if (fraction_to_keep != 1.0_f) {
    TC_INFO("Selecting top {:.2f}% blocks", fraction_to_keep * 100.0_f);

    int blocks_to_keep =
        int(fraction_to_keep * container_page_map->Get_Blocks().second);
    if (blocks_to_keep < (int)thin_page_map->Get_Blocks().second) {
      TC_INFO("fraction_to_keep constraint already satisfied ({:.2f}%)",
              100.0_f * thin_page_map->Get_Blocks().second /
                  container_page_map->Get_Blocks().second);
    } else {
      std::vector<real> density_sums;
      for (auto &m : block_meta) {
        density_sums.push_back(m.density_sum);
      }
      std::nth_element(std::begin(density_sums),
                       std::begin(density_sums) + blocks_to_keep - 1,
                       std::end(density_sums));
      real threshold = density_sums[blocks_to_keep - 1];

      thin_page_map->Clear();  // Cleared, but block offsets still not updated.
      parallel_for_each_block(*thin_page_map, [&](int b, uint64 block_offset,
                                                  const Vector3i &base_coord) {
        if (block_meta[b].density_sum > threshold) {
          thin_page_map->Set_Page(block_offset);
        }
      });
      thin_page_map->Update_Block_Offsets();
      TC_INFO("{} blocks remained (target {})",
              thin_page_map->Get_Blocks().second, blocks_to_keep);
    }
  }
}

// Both fat and fatter page maps will be updated
void SPGridTopologyOptimization3D::update_fat_page_map() {
  auto active_threshold = config.get<real>("active_threshold", 1e-6_f);
  auto fat_margin = config.get<int>("fat_margin", 1);

  auto sparse_flags = grid->flags();
  auto sparse_density = grid->density();
  auto const blocks = thin_page_map->Get_Blocks();
  fat_page_map->Clear();
  fatter_page_map->Clear();

  std::vector<Vectori> offsets = {Vectori(-1, -1, -1), Vectori(-1, 1, -1),
                                  Vectori(-1, -1, 1),  Vectori(-1, 1, 1),
                                  Vectori(1, -1, -1),  Vectori(1, 1, -1),
                                  Vectori(1, -1, 1),   Vectori(1, 1, 1)};

  for (uint32 i = 0; i < blocks.second; i++) {
    auto base_coord = Vectori(ChannelMask::LinearToCoord(blocks.first[i]));
    Region region(Vectori(0), block_size);
    for (auto ind : region) {
      Vectori coord = base_coord + ind.get_ipos();
      real d = sparse_density(coord);
      if (d > active_threshold) {
        for (auto &offset : offsets) {
          auto neighbour_coord = coord + fat_margin * offset;
          auto neighbour_offset = ChannelMask::Linear_Offset(neighbour_coord);
          if (container_page_map->Test_Page(neighbour_offset)) {
            fat_page_map->Set_Page(neighbour_offset);
          }
        }
      }
    }
  }
  fat_page_map->Update_Block_Offsets();

  auto const fat_blocks = fat_page_map->Get_Blocks();
  for (uint32 i = 0; i < fat_blocks.second; i++) {
    auto base_coord = Vectori(ChannelMask::LinearToCoord(fat_blocks.first[i]));
    for (auto ind : Region(Vectori(0), Vectori(2))) {
      auto neighbour_coord = base_coord + ind.get_ipos() * block_size;
      auto neighbour_offset = ChannelMask::Linear_Offset(neighbour_coord);
      if (container_page_map->Test_Page(neighbour_offset)) {
        fatter_page_map->Set_Page(neighbour_offset);
      }
    }
  }
  fatter_page_map->Update_Block_Offsets();

  TC_INFO("# thin blocks  = {} ({:.2f}%)", thin_page_map->Get_Blocks().second,
          100.0_f * thin_page_map->Get_Blocks().second /
              container_page_map->Get_Blocks().second);
  TC_INFO("# fat blocks   = {} ({:.2f}%)", fat_page_map->Get_Blocks().second,
          100.0_f * fat_page_map->Get_Blocks().second /
              container_page_map->Get_Blocks().second);
  TC_INFO("# fatter blocks = {} ({:.2f}%)",
          fatter_page_map->Get_Blocks().second,
          100.0_f * fatter_page_map->Get_Blocks().second /
              container_page_map->Get_Blocks().second);

  TC_INFO("# container blocks = {}", container_page_map->Get_Blocks().second);

  const auto fatter_blocks = fatter_page_map->Get_Blocks();

  int active_voxel_counter = 0;
  for (uint32 i = 0; i < fatter_blocks.second; i++) {
    auto base_coord =
        Vectori(ChannelMask::LinearToCoord(fatter_blocks.first[i]));
    Region region(Vectori(0), block_size);
    for (auto ind : region) {
      Vectori coord = base_coord + ind.get_ipos();
      if (!sparse_flags(coord).get_inside_container()) {
        continue;
      }
      real d = sparse_density(coord);
      if (d > active_threshold) {
        active_voxel_counter += 1;
      }
    }
  }
  TC_INFO("# active voxels = {} ({:.2f}M)", active_voxel_counter,
          active_voxel_counter * 1e-6_f);
  TC_INFO("#    {:.2f}% of fat_page_map",
          100.0_f * active_voxel_counter / fat_page_map->Get_Blocks().second /
              block_size.prod());
  TC_INFO("#    {:.2f}% of fatter_page_map",
          100.0_f * active_voxel_counter /
              fatter_page_map->Get_Blocks().second / block_size.prod());
}

void SPGridTopologyOptimization3D::populate_grid(
    const std::function<real(Vector3)> &sdf,
    const std::string &dirichlet,
    const std::string &mirror) {
  // TC_ASSERT_INFO(mirror == "", "Mirroring is no longer supported.");
  domain_func = sdf;
  total_container_voxels = 0;
  auto sparse_density = grid->density();
  auto sparse_flags = grid->flags();

  bool mirroring[dim];
  for (int i = 0; i < dim; i++) {
    mirroring[i] = mirror.find(char('x' + i)) != std::string::npos;
    if (mirroring[i])
      TC_INFO("Mirroring domain in {} direction", char('X' + i));
  }

  Vectori res = config.get<Vectori>("res");
  Vectori center(res / Vectori(2) + Vectori(padding));

  container_page_map->Clear();
  bool use_wireframe = config.get<bool>("wireframe", false);
  if (use_wireframe) {
    TC_INFO("Populating using wireframe");
  }
  int wireframe_grid_size = config.get<int>("wireframe_grid_size", 32);
  int wireframe_thickness = config.get<int>("wireframe_thickness", 4);
  int num_hard_voxels = 0;
  container_bounds[0] = Vectori(1000000000);
  container_bounds[1] = Vectori(-1000000000);
  // get_cell_region is not yet usable here
  TC_INFO("Populating domain");
  int64 counter = 0;
  // TODO: speed up this part?
  Region region;
  if (config.has_key("bounding_box0")) {
    region = Region(
        Vectori(padding) +
            ((config.get<Vector>("bounding_box0") + Vector(0.5_f)) * inv_dx)
                .template cast<int>(),
        Vectori(padding) +
            ((config.get<Vector>("bounding_box1") + Vector(0.5_f)) * inv_dx)
                .template cast<int>());
  } else {
    region = Region(Vectori(padding), Vectori(padding) + res);
  }

  for (auto &ind : region) {
    // cell-centered
    counter++;
    if (counter % 500000000 == 0) {
      TC_INFO("{} {:.2f}% voxels populated", counter,
              100.0_f * counter / res.template cast<float64>().prod());
    }
    Vector pos = normalize_pos(ind.get_pos());
    bool inside = true;
    for (int i = 0; i < dim; i++) {
      if (mirroring[i] && ind.get_ipos()[i] > center[i]) {
        inside = false;
      }
    }
    if (!inside)
      continue;
    if (sdf(pos) <= 1e-6_f) {
      /*
      if (seed_from_previous) {
        auto old_ind =
            (ind.get_ipos() + Vector3i(-padding)) / Vectori(upsample_ratio) +
            Vectori(padding);
        Region offset(Vectori(-1), Vectori(1));
        uint32 old_flag = 0;
        real old_density = 0;
        for (auto o : offset) {
          old_flag = old_flag | sparse_flags_old(old_ind + o.get_ipos());
          old_density = std::max(old_density,
                                 sparse_density_old(old_ind + o.get_ipos()));
        }
        // 0.06 is a threshold for deletion
        if (!old_flag || old_density < 0.06) {
          continue;
        }
      } // End seed from previous
      */

      real density = minimum_density;
      if (!use_wireframe) {
        // Uniform initialization
        num_hard_voxels += 1;
        density = get_volume_fraction(0);
      } else {
        const int A = wireframe_thickness, B = wireframe_grid_size;
        if (int(ind.i / block_size.x % B < A) +
                int(ind.j / block_size.y % B < A) +
                int(ind.k / block_size.z % B < A) >=
            2) {
          num_hard_voxels += 1;
          density = 1.0_f;
        }
      }

      sparse_density(ind.get_ipos()) = density;

      for (int i = 0; i < dim; i++) {
        container_bounds[0][i] =
            std::min(container_bounds[0][i], ind.get_ipos()[i]);
        container_bounds[1][i] =
            std::max(container_bounds[1][i], ind.get_ipos()[i]);
      }

      sparse_flags(ind.get_ipos()).set_inside_container(true);
      if (!dirichlet.empty())
        add_cell_boundary(ind.get_ipos(), dirichlet);
      Vectori extend = Vectori(2);
      Region reg(Vectori(0), extend);
      for (int i = 0; i < dim; i++) {
        if (mirroring[i] && ind.get_ipos()[i] == center[i]) {
          extend += Vectori::axis(i);
        }
      }
      for (auto &d : reg) {
        container_page_map->Set_Page(
            ChannelMask::Linear_Offset(ind.get_ipos() + d.get_ipos()));
      }
      for (int i = 0; i < dim; i++) {
        if (mirroring[i] && ind.get_ipos()[i] == center[i]) {
          auto dirichlet_cell = ind.get_ipos() + Vectori::axis(i);
          sparse_density(dirichlet_cell) = density;
          sparse_flags(dirichlet_cell).set_inside_container(1);
          add_cell_boundary(dirichlet_cell, std::string(1, char('x' + i)));
          for (auto &d : Region(Vectori(0), Vectori(2))) {
            container_page_map->Set_Page(
                ChannelMask::Linear_Offset(dirichlet_cell + d.get_ipos()));
          }
        }
      }
      total_container_voxels += 1;
    }
  }
  container_bounds[1] += Vectori(1);
  container_page_map->Update_Block_Offsets();
  TC_P(get_container_size());
  TC_INFO("Ambient grid voxels: {}", get_container_size().prod());
  TC_INFO("\nActive voxels: {} ({:.2f}M)\nHard   voxels: {} ({:.2f}M, {:.2f}%)",
          total_container_voxels, total_container_voxels * 1e-6_f,
          num_hard_voxels, num_hard_voxels * 1e-6_f,
          100.0_f * num_hard_voxels / total_container_voxels);
  TopOptGrid::copy_page_map(container_page_map, thin_page_map);
  TopOptGrid::copy_page_map(container_page_map, fat_page_map);
  TopOptGrid::copy_page_map(container_page_map, fatter_page_map);
  TC_ASSERT(container_page_map->Get_Blocks().second ==
            fat_page_map->Get_Blocks().second);
}

void SPGridTopologyOptimization3D::populate_grid(const Config &config) {
  domain_config = config;
  auto domain_type = config.get_string("domain_type");
  TC_INFO("Populating grid as type [{}]", domain_type);
  std::string mirror = config.get<std::string>("mirror", "");
  std::string uniform_dirichlet_bc = config.get<std::string>("uniform_bc", "");
  if (uniform_dirichlet_bc != "") {
    TC_INFO("Setting Dirichlet BCs on each cell (axis = {})",
            uniform_dirichlet_bc);
  }
  DomainFunc sdf;
  using namespace fem_interface;
  FEMInputs param;
  int scaling = 1;
  if (domain_type == "dense") {
    sdf = [](const Vector &pos) { return -1; };
  } else if (domain_type == "cylinder") {
    real radius = config.get("radius", 0.4_f);
    real thickness = config.get("thickness", 0.03_f);
    real height = config.get("height", 1_f);
    auto half_height = height / 2;
    sdf = [radius, thickness, half_height](const Vector &pos) {
      return std::max(
          abs(pos.y) - half_height,
          abs(length(Vector2(pos.x, pos.z)) - radius) - thickness * 0.5_f);
    };
  } else if (domain_type == "wheel") {
    sdf = [](const Vector &pos) {
      auto pos2d = Vector2(std::hypot(pos.x, pos.y), pos.z);
      real distance_to_tip = length(pos2d - Vector2(0.45_f, 0));
      return std::min(distance_to_tip - 0.03,
                      std::abs(pos2d.x) + std::abs(pos2d.y) * 4.5 - 0.45);
    };
  } else if (domain_type == "box") {
    Vector size = config.get("size", Vector(0.3_f));
    sdf = [size](const Vector &pos) { return (pos.abs() - size).max(); };
  } else if (domain_type == "texture") {
    std::shared_ptr<Texture> tex =
        AssetManager::get_asset<Texture>(config.get<int>("tex_id"));
    sdf = [tex](const Vector &pos) {
      // Note: texture has range 0 ~ 1, while topOpt uses -0.5 ~ 0.5
      real d = tex->sample(pos + Vector(0.5_f)).x;
      return -d + 0.5;
    };
  } else if (domain_type == "sphere") {
    real inner_radius = config.get("inner_radius", 0.4_f);
    real outer_radius = config.get("outer_radius", 0.5_f);
    if (config.has_key("upper_only")) {
      TC_ERROR("upper_only is no longer used");
    }
    bool lower_upper = config.get("lower_only", false);
    sdf = [inner_radius, outer_radius, lower_upper](const Vector &pos) {
      real to_center = length(pos);
      real to_surface = max(to_center - outer_radius, inner_radius - to_center);
      if (lower_upper) {
        return max(to_surface, pos.y);
      } else
        return to_surface;
    };
  } else if (domain_type == "field") {
    Vectori field_res;
    field_res = config.get<Vectori>("field_res");
    scaling = config.get<int>("scaling");
    Array3D<uint8> field;
    field.initialize(field_res);
    read_from_binary_file(param, config.get<std::string>("field_fn"));
    TC_P(param.density.blocks.size());
    using scalar_block_size = FEMInputs::ScalarGrid::block_size;
    for (auto &b : param.density.blocks) {
      for (int i = 0; i < scalar_block_size::x; i++) {
        for (int j = 0; j < scalar_block_size::y; j++) {
          for (int k = 0; k < scalar_block_size::z; k++) {
            Vectori coord(b.base_coordinates[0] + i, b.base_coordinates[1] + j,
                          b.base_coordinates[2] + k);
            TC_ASSERT(field.inside(coord));
            field[coord] = 1;
          }
        }
      }
    }
    sdf = [field, field_res, this, scaling](const Vector &pos) -> real {
      // inverse of (ind - Vector(padding)) * dx - Vector(0.5_f);
      Vectori ipos = ((pos + Vector(0.5_f)) * inv_dx / Vector(scaling))
                         .template cast<int>() +
                     Vectori(padding);  // flooring
      if (field[ipos] > 0) {
        return -0.1_f;
      } else {
        return 0.1_f;
      }
    };
  } else {
    TC_ERROR("Unknown domain type: {}", domain_type);
  }
  populate_grid(sdf, uniform_dirichlet_bc, mirror);

  if (domain_type == "field") {
    auto sparse_density = grid->density();
    TC_INFO("Setting initial density...");
    using scalar_block_size = FEMInputs::ScalarGrid::block_size;
    int64 total_voxel_count = 0;
    int64 active_voxel_count = 0;
    real threshold = 0.03_f;
    TC_P(threshold);
    for (auto &b : param.density.blocks) {
      for (int i = 0; i < scalar_block_size::x; i++) {
        for (int j = 0; j < scalar_block_size::y; j++) {
          for (int k = 0; k < scalar_block_size::z; k++) {
            Vectori coord(b.base_coordinates[0] + i, b.base_coordinates[1] + j,
                          b.base_coordinates[2] + k);
            real d = b.get(i, j, k);
            // d = std::pow((d - 1e-6_f) / (1 - 1e-6_f), 1.0_f / 3.0_f);
            d = std::pow(d, 1.0_f / 3.0_f);
            for (auto &ind : Region(Vectori(0), Vectori(scaling))) {
              total_voxel_count += 1;
              sparse_density(coord * scaling + ind.get_ipos()) = d;
              if (d > threshold) {
                active_voxel_count += 1;
              }
            }
          }
        }
      }
    }
    TC_P(threshold);
    TC_P(total_voxel_count);
    TC_P(active_voxel_count);
    TC_P(1.0_f * active_voxel_count / total_voxel_count);
  }
}

void Opt::smooth_dc() const {
  auto sparse_dc0 = grid->dc0();
  auto sparse_dc1 = grid->dc1();
  auto sparse_density = grid->density();
  auto sparse_flags = grid->flags();

  const auto &blocks = fat_page_map->Get_Blocks();

  real filter_size = config.get("filter_size", 1.5_f);
  int half_expand = (int)std::floor(filter_size);

  tbb::parallel_for(0u, blocks.second, [&](int b) {
    auto offset = blocks.first[b];
    Vector3i base_coord(ChannelMask::LinearToCoord(offset));

    Region region(Vectori(0), block_size);
    for (auto ind : region) {
      Vectori coord = base_coord + ind.get_ipos();
      real sum = 0.0_f;
      real weight_sum = 0.0_f;

      for (int dx = -half_expand; dx <= half_expand; dx++) {
        for (int dy = -half_expand; dy <= half_expand; dy++) {
          for (int dz = -half_expand; dz <= half_expand; dz++) {
            auto offset = Vectori(dx, dy, dz);
            if (sparse_flags(offset + coord).get_inside_container()) {
              real weight =
                  max(0.0_f, filter_size - std::sqrt((real)length2(offset)));
              sum += weight * sparse_dc0(offset + coord) *
                     sparse_density(offset + coord);
              weight_sum += weight;
            }
          }
        }
      }
      sparse_dc1(coord) = sum / (weight_sum * (sparse_density(coord) + 1e-3_f));
    }
  });
  // Copy dc1 to dc0
  tbb::parallel_for(0u, blocks.second, [&](int b) {
    auto offset = blocks.first[b];
    for (unsigned int e = 0; e < ChannelMask::elements_per_block;
         ++e, offset += sizeof(real)) {
      std::swap(sparse_dc0(offset), sparse_dc1(offset));
    }
  });
}

struct DensityField {
  Vector3i res;
  std::vector<float64> densities;
  TC_IO_DEF(res, densities);
};

std::string SPGridTopologyOptimization3D::general_action(const Config &param) {
  general_actions.push_back(param);
  auto action = param.get<std::string>("action");
  TC_MEMORY_USAGE("Before " + action);
  DeferedExecution _([&]() { TC_MEMORY_USAGE("After " + action); });
  TC_MEMORY_USAGE("After " + action);
  if (action == "output") {
    output(param.get<std::string>("file_name"));
  } else if (action == "iterate") {
    int iter = param.get<int>("iter");
    real objective = iterate(iter);
    last_iter = iter;
    return fmt::format("{:.30f}", objective);
  } else if (action == "upsample") {
    upsample();
  } else if (action == "print_laplacian") {
    real sum = 0;
    auto density = grid->density();
    for (auto ind : get_cell_region()) {
      auto lap = density(ind.get_ipos()) * 6 -
                 density(ind.get_ipos() + Vector3i(0, 0, 1)) -
                 density(ind.get_ipos() + Vector3i(0, 0, -1)) -
                 density(ind.get_ipos() + Vector3i(0, 1, 0)) -
                 density(ind.get_ipos() + Vector3i(0, -1, 0)) -
                 density(ind.get_ipos() + Vector3i(1, 0, 0)) -
                 density(ind.get_ipos() + Vector3i(-1, 0, 0));
      sum += pow<2>(lap);
    }
    TC_P(sqrt(sum));
  } else if (action == "add_objective") {
    objectives.emplace_back(create_instance_unique_ctor<Objective>(
        param.get<std::string>("name"), Dict("opt", this)));
    active_objective = objectives.back().get();
    active_objective->weight = param.get<real>("weight");
  } else if (action == "select_active_objective") {
    int id = param.get<int>("id");
    if (id == (int)objectives.size()) {
      objectives.emplace_back(create_instance_unique_ctor<Objective>(
          "minimal_compliance", Dict("opt", this)));
    }
    TC_ASSERT(id < (int)objectives.size())
    active_objective = objectives[id].get();
  } else if (action == "objective") {
    compute_objective(param.get("objective_threshold", 0.5_f));
  } else if (action == "voxel_connectivity_filtering") {
    // This removes unconnected voxels outside the container
    filter_isolated_voxels(0);
  } else if (action == "make_shell_mesh") {
    make_shell_mesh(param.get<std::string>("mesh_fn"),
                    param.get<real>("maximum_distance"));
  } else if (action == "make_shell") {
    // Note: not param but config here. 'fixed_cell_density' is stored in
    // topOpt::config
    real fixed_cell_density = config.get<real>("fixed_cell_density");
    // Mark all cells OUTSIDE this texture to be a constant density
    std::shared_ptr<Texture> tex =
        AssetManager::get_asset<Texture>(param.get<int>("tex_id"));
    auto flags = grid->flags();
    auto density = grid->density();
    auto blocks = grid->container_page_map->Get_Blocks();
    int counter = 0;
    for (uint64 b = 0; b < blocks.second; b++) {
      auto offset = blocks.first[b];
      for (unsigned int e = 0; e < ChannelMask::elements_per_block;
           ++e, offset += sizeof(real)) {
        if (flags(offset).get_inside_container()) {
          Vector pos = i2f(Vector(0.5_f) +
                           Vectori(ChannelMask::LinearToCoord(offset))
                               .template cast<real>()) *
                       1.03_f;
          real d = -tex->sample(pos + Vector(0.5_f)).x + 0.5_f;
          if (d > 0) {
            flags(offset).set_fixed_density(true);
            density(offset) = fixed_cell_density;
            counter += 1;
          }
        }
      }
    }
    TC_INFO("{} ({:.2f}%) voxels set to be the shell.", counter,
            100.0_f * counter / total_container_voxels);
  } else if (action == "save_state") {
    save_state(param.get<std::string>("filename"));
  } else if (action == "load_state") {
    load_state(param.get<std::string>("filename"));
    return fmt::format("{}", last_iter);
  } else if (action == "override") {
    std::string key = param.get<std::string>("key");
    std::string val = param.get<std::string>("val");
    if (!config.has_key(key)) {
      TC_WARN("key '{}' does not exist in config.", key);
    } else {
      TC_INFO("original value of '{}' was '{}'", key,
              config.get<std::string>(key));
    }
    config.set<std::string>(key, val);
    TC_WARN("config['{}'] set to {}", key, val);
  } else if (action == "add_dirichlet_bc") {
    add_cell_boundary(
        param.get<Vector>("center"), param.get<real>("radius", 0.03_f),
        param.get<std::string>("axis"), param.get<Vector>("value", Vector(0)));
  } else if (action == "add_box_dirichlet_bc") {
    std::string axis_to_fix = param.get<std::string>("axis_to_fix");
    Vector bounds[2];
    bounds[0] = param.get<Vector>("bound0");
    bounds[1] = param.get<Vector>("bound1");
    Vectori ipos0 = ((bounds[0] + Vector(0.5_f)) * inv_dx - Vector(0.5_f))
                        .template cast<int>() +
                    Vectori(padding);
    Vectori ipos1 = ((bounds[1] + Vector(0.5_f)) * inv_dx + Vector(0.5_f))
                        .template cast<int>() +
                    Vectori(padding);
    for (int i = 0; i < dim; i++) {
      ipos0[i] =
          clamp(ipos0[i], container_bounds[0][i], container_bounds[1][i]);
      ipos1[i] =
          clamp(ipos1[i], container_bounds[0][i], container_bounds[1][i]);
    }
    TC_ASSERT(ipos0 <= ipos1);
    // TC_ASSERT(container_bounds[0] <= ipos0);
    // TC_ASSERT(ipos1 <= container_bounds[1]);
    // Enumerate cells
    bool failed_bc = false;
    int counter = 0;
    auto value = param.get("value", Vector(0));
    for (auto c : Region(Vectori(ipos0), Vectori(ipos1))) {
      if (grid->density()(c.get_ipos()) == 0) {
        continue;
      }
      Vector cell_center_pos = i2f(c.get_pos());
      bool failed = false;
      if (bounds[0] <= cell_center_pos && cell_center_pos <= bounds[1]) {
        bool ret = add_cell_boundary(c.get_ipos(), "xyz", value);
        failed = failed || !ret;
      }
      if (failed) {
        TC_WARN("There are failed cells during adding Box Dirichlet BC.");
      }
    }
    if (failed_bc) {
      TC_WARN("Tried to add Dirichlet BC to 'air' nodes.");
    }
    TC_TRACE("Dirichelt BC added to {} nodes", counter);
  } else if (action == "add_box_load") {
    Region region(f2i(param.get<Vector>("bound0")),
                  f2i(param.get<Vector>("bound1")));
    auto f = param.get<Vector>("force");
    for (auto &ind : region) {
      auto i = ind.get_ipos();
      if (node_flag(i)) {
        add_force(i, f);
      }
    }
  } else if (action == "add_box_target_deformation") {
    Region region(f2i(param.get<Vector>("bound0")),
                  f2i(param.get<Vector>("bound1")));
    auto f = param.get<Vector>("displacement");
    auto weight = param.get<real>("weight", 1.0_f);
    for (auto &ind : region) {
      auto i = ind.get_ipos();
      if (node_flag(i)) {
        active_objective->targets.push_back(Objective::Target{i, f, weight});
      }
    }
  } else if (action == "add_load") {
    Vectori size(std::ceil(param.get<real>("size") * inv_dx));
    TC_ASSERT(size >= Vectori(1));
    bool success = add_force(param.get<Vector>("center"),
                             param.get<Vector>("force"), size);
    if (!success) {
      TC_WARN(
          "There are forces added to nodes without active neighbouring "
          "cells; these forces on node will be ignored.");
    }
  } else if (action == "add_plane_load") {
    add_plane_force(param.get<Vector>("force"), param.get<int>("axis"),
                    param.get<int>("extreme"), param.get<Vector>("bound1"),
                    param.get<Vector>("bound2"));
  } else if (action == "add_precise_plane_force_bridge") {
    add_precise_plane_force_bridge();
  } else if (action == "add_pressure_load") {
    int version = param.get("version", 1);
    TC_INFO("Wing boundary condition version {}", version);
    int counter = 0;
    for (int i = container_bounds[0][0]; i < container_bounds[1][0]; i++) {
      for (int k = container_bounds[0][2]; k < container_bounds[1][2]; k++) {
        constexpr int INF = 1000000000;
        int lower = INF, upper = -INF;
        for (int j = container_bounds[0][1]; j < container_bounds[1][1]; j++) {
          if (node_flag(Vector3i(i, j, k))) {
            lower = std::min(lower, j);
            upper = std::max(upper, j);
          }
        }
        auto pos = i2f(Vector(i, lower, k));
        if (lower != INF && -0.35 < pos.x && pos.x < 0.35) {
          counter += 1;
          add_force(Vector3i(i, upper, k), Vector(0, -0.5, 0), Vectori(1));
          if (version == 1) {
            // Add cell Dirichlet
            auto cell_idx = Vectori(i, lower + 1, k);
            if (grid->density()(cell_idx) > 0) {
              Region node_region = Region(Vectori(0), Vectori(2));
              for (auto n : node_region) {
                // For each node
                // Duplicate node BCs will be deleted
                // and conflict node BCs will be detected before feeding into
                // solver
                Vectori node = cell_idx + n.get_ipos();
                for (int t = 0; t < dim; t++) {
                  add_node_boundary(node, t, 0);
                }
              }
            }
          } else {
            // Version 2, both top and bottom sides have force
            add_force(Vector3i(i, lower, k), Vector(0, 1, 0), Vectori(1));
          }
        }
      }
    }
    TC_INFO("pressure load to {} nodes", counter);
  } else if (action == "add_plane_dirichlet_bc") {
    add_plane_dirichlet_bc(
        param.get<std::string>("axis_to_fix"), param.get<int>("axis_to_search"),
        param.get<int>("extreme"), param.get<Vector>("value"));
  } else if (action == "populate_grid") {
    populate_grid(param);
  } else if (action == "wireframe") {
    convert_to_wireframe();
  } else if (action == "add_mesh_normal_force") {
    add_mesh_normal_force(
        param.get<std::string>("mesh_fn"), param.get<real>("magnitude"),
        param.get<Vector3>("center"), param.get<real>("falloff"),
        param.get<real>("maximum_distance"),
        param.get("override", Vector(0.0_f)), param.get("no_z", false));
  } else if (action == "load_density_from_fem") {
    load_density_from_fem(param.get<std::string>("fn"));
  } else if (action == "threshold_and_kill") {
    threshold_and_kill(param.get<real>("threshold"));
  } else if (action == "island_test") {
    auto density = grid->density();
    for (auto &ind : get_node_region()) {
      Vector p = i2f(ind.get_ipos());
      bool inside = false;
      if (p.x < -0.25) {
        inside = true;
      }
      if (p.x > 0.2 && -0.2 < std::min(p.y, p.z) && max(p.y, p.z) < 0.2) {
        inside = true;
      }
      density(ind.get_ipos()) = inside ? 1.0_f : minimum_density;
    }

  } else if (action == "set_step_limit") {
    config.set("step_limit", param.get<real>("value"));
  } else if (action == "threshold_density") {
    threshold_density();
  } else if (action == "threshold_and_compute_objective") {
    real vol = total_container_voxels * volume_fraction;
    if (param.has_key("threshold")) {
      auto threshold = param.get<real>("threshold");
      vol = threshold_density_trail(threshold, true);
    } else {
      threshold_density();
    }
    std::map<std::string, real> evaluated;
    real objective = 0;
    {
      TC_PROFILER("compute derivative");
      for (int i = 0; i < (int)objectives.size(); i++) {
        auto objective_name =
            fmt::format("[{}]{}", i, objectives[i]->get_name());
        TC_PROFILER(objective_name);
        auto obj = objectives[i]->compute(Dict("iter", 1));
        TC_TRACE("  {} = {}", objective_name, obj);
        evaluated[objective_name] = obj;
        objective += obj;
      }
    }
    TC_WARN("Total objective: {}", objective);
    for (auto kv : evaluated) {
      TC_WARN("   {}: {}", kv.first, kv.second);
    }
    return fmt::format("{:30f}, {:30f}", objective, vol);
  } else if (action == "get_block_counts") {
    grid->update_block_offsets();
    auto n_thin = thin_page_map->Get_Blocks().second;
    auto n_fat = fat_page_map->Get_Blocks().second;
    auto n_fatter = fatter_page_map->Get_Blocks().second;
    auto n_cont = container_page_map->Get_Blocks().second;
    return fmt::format("{} {} {} {}", n_thin, n_fat, n_fatter, n_cont);
  } else if (action == "save_density") {
    auto fn = param.get<std::string>("fn");
    DensityField f;
    f.res = container_bounds[1] - container_bounds[0];
    auto d = this->grid->density();
    for (auto ind : get_cell_region()) {
      f.densities.push_back(d(ind.get_ipos()));
    }
    TC_ASSERT(f.densities.size() == f.res.prod());
    write_to_binary_file(f, fn);
  } else if (action == "load_density") {
    auto fn = param.get<std::string>("fn");
    DensityField f;
    read_from_binary_file(f, fn);
    Vector3i actual_res = container_bounds[1] - container_bounds[0];
    int scale = (actual_res.z - 2) / (f.res.z - 2);
    TC_P(actual_res);
    TC_P(f.res);
    TC_ERROR_UNLESS(
        (f.res - Vector3i(0, 0, 2)) * scale == actual_res - Vector3i(0, 0, 2),
        "Resolution mismatch.");
    TC_P(scale);
    TC_P(actual_res);
    TC_P(f.res);
    real total_density =
        std::accumulate(f.densities.begin(), f.densities.end(), 0.0_f);
    TC_P(total_density);
    auto d = this->grid->density();
    for (auto ind : get_cell_region()) {
      auto i = ((ind.get_ipos() - Vector3i(container_bounds[0]) +
                 Vector3i(0, 0, scale - 1)) /
                scale);
      auto index = i.x * f.res.y * f.res.z + i.y * f.res.z + i.z;
      TC_ASSERT(0 <= index && index < (int)f.densities.size());
      d(ind.get_ipos()) = f.densities[index];
    }
  } else if (action == "add_volume_load") {
    std::shared_ptr<Texture> tex =
        AssetManager::get_asset<Texture>(param.get<int>("tex"));
    auto f = param.get<Vector>("force");
    int counter = 0;
    for (auto &ind : get_node_region()) {
      auto pos = i2f(ind.get_ipos());
      real d = -tex->sample(pos + Vector(0.5_f)).x + 0.5;
      if (d < 0) {
        counter += 1;
        add_force(ind.get_ipos(), f, Vectori(1));
      }
    }
    TC_P(counter);
  } else if (action == "fix_airfoil") {
    auto flags = this->grid->flags();
    for (auto &ind : get_cell_region()) {
      auto pos = i2f(ind.get_ipos());
      if (domain_func(pos) < 0 &&
          flags(ind.get_ipos()).get_inside_container() &&
          length(Vector2(pos.x, pos.y) - Vector2(-0.1765_f, -0.004_f)) <
              0.0691_f) {
        add_cell_boundary(ind.get_ipos());
      }
    }
  } else if (action == "set_up_wheel") {
    for (auto &ind : get_node_region()) {
      auto pos = i2f(ind.get_ipos());
      if (domain_func(pos) < 0 && length(Vector2(pos.x, pos.y)) > 0.42_f) {
        add_force(ind.get_ipos(), Vector(-pos.y, pos.x, 0));
        // add_force(ind.get_ipos(), Vector(pos.x, pos.y, 0));
        // add_force(ind.get_ipos(), Vector(0, 0, 0));
      }
    }
  } else if (action == "set_up_wheel_shear") {
    for (auto &ind : get_node_region()) {
      auto pos = i2f(ind.get_ipos());
      if (domain_func(pos) < 0 && length(Vector2(pos.x, pos.y)) > 0.42_f) {
        // add_force(ind.get_ipos(), Vector(pos.x, pos.y, 0));
        add_force(ind.get_ipos(), Vector(0, 0, 1));
      }
    }
  } else if (action == "set_cylinder_density") {
    auto sparse_density = grid->density();
    for (auto &ind : get_cell_region()) {
      auto pos = i2f(ind.get_ipos());
      if (domain_func(pos) < 0 && length(Vector2(pos.y, pos.z)) < 0.2_f) {
        sparse_density(ind.get_ipos()) = 0.5;
      } else {
        sparse_density(ind.get_ipos()) = minimum_density;
      }
    }
  } else if (action == "set_arch_density") {
    auto sparse_density = grid->density();
    for (auto &ind : get_cell_region()) {
      auto pos = i2f(ind.get_ipos());
      if (domain_func(pos) < 0 && pos.y < -0.46) {
        sparse_density(ind.get_ipos()) = 0.5;
      } else {
        sparse_density(ind.get_ipos()) = 1e-2;
      }
    }
  } else {
    TC_ERROR("Unknown action: {}", action);
  }
  return "";
}

bool SPGridTopologyOptimization3D::fem_solve(
    int iter,
    bool pure_objective,
    const std::vector<fem_interface::ForceOnNode> &forces,
    const typename HexFEMSolver<dim>::BoundaryCondition &boundary_conditions,
    bool write_to_u) {
  using namespace fem_interface;
  TC_MEMORY_USAGE("Begin FEM Solve");
  FEMInterface interface;
  auto &param = interface.param;
  TC_P(get_penalty(iter));

  // TC_P(container_bounds);
  // Cell size, instead of node size
  for (int i = 0; i < 3; i++)
    param.resolution[i] = container_bounds[1][i] + padding * 6;

  param.global_mu = material.mu;
  param.global_lambda = material.lambda;
  param.use_density_only = true;
  param.dx = dx;

  param.krylov.tolerance = config.get("cg_tolerance", 1e-4_f);
  param.krylov.max_iterations = config.get("cg_max_iterations", 50);
  param.krylov.print_residuals = config.get("print_residuals", true);
  param.krylov.restart_iterations = config.get("restart_iterations", 0);
  TC_ERROR_IF(config.has_key("global_smoothing_iter"),
              "global_smoothing_iter is obselete.");
  param.defect_correction_cg_iter() =
      config.get("defect_correction_cg_iter", 3);
  param.penalty() = get_penalty(iter);
  param.minimum_stiffness() = minimum_stiffness;

  param.defect_correction_iter() = config.get<int>("defect_correction_iter", 0);

  // Do not do warm starting for a pure objective solve
  param.keep_state() =
      (int)(config.get<bool>("warm_starting", false) && !pure_objective);
  param.forced_reuse() = (int)config.get<bool>("forced_reuse", true);

  // Bool's solver type option
  param.solver_type() = config.get<int>("solver_type", 1);

  // Multigrid Parameters

  if (config.has_key("mg_level")) {
    param.mg_level = config.get<int>("mg_level");
  } else {
    int current_n = get_container_size().max();
    param.mg_level = 1;
    while (current_n > config.get("mg_bottom_size", 64)) {
      current_n /= 2;
      param.mg_level += 1;
    }
  }
  TC_P(param.mg_level);
  if (param.mg_level == 2) {
    TC_WARN(
        "mg_level = 2 is no longer supported by the solver. Setting it to 3.");
    param.mg_level = 3;
  }

  if (!pure_objective)
    param.set_solver_state_ptr(solver_state_ptr);
  param.explicit_mg_level = config.get("explicit_mg_level", 0);

  if (param.explicit_mg_level >= param.mg_level) {
    TC_WARN("Clamping explicit_mg_level ({}) to {}!", param.explicit_mg_level,
            param.mg_level - 1);
    param.explicit_mg_level = param.mg_level - 1;
  }
  param.pre_and_post_smoothing_iter = config.get("smoothing_iters", 1);
  param.boundary_smoothing() = config.get("boundary_smoothing_iters", 3);
  param.bottom_smoothing_iter = 600;
  param.jacobi_damping = config.get("jacobi_damping", 0.4_f);

  using scalar_block_size = FEMInputs::ScalarGrid::block_size;

  for (int i = 0; i < 3; i++) {
    param.mu.resolution[i] = param.resolution[i];
    param.lambda.resolution[i] = param.resolution[i];
    param.density.resolution[i] = param.resolution[i];
  }

  auto sparse_density = grid->density();
  auto sparse_flags = grid->flags();

  Vector3i block_size(sparse_density.geometry.block_xsize,
                      sparse_density.geometry.block_ysize,
                      sparse_density.geometry.block_zsize);

  const auto &blocks = fatter_page_map->Get_Blocks();

  double bounds[2] = {1e30f, -1e30f};

  bool has_nan = false;

  for (SPGrid_Block_Iterator<ChannelMask> iterator(blocks); iterator.Valid();
       iterator.Next_Block()) {
    Vector3i base_coord;

    for (int v = 0; v < dim; ++v)
      base_coord[v] = iterator.Index()[v];

    Region region(Vectori(0), block_size);

    // whether the block is only for nodes, even if there are no cells
    bool only_for_nodes = !fat_page_map->Test_Page(iterator.Offset());

    TC_ASSERT(block_size.x % scalar_block_size::x == 0);
    TC_ASSERT(block_size.y % scalar_block_size::y == 0);
    TC_ASSERT(block_size.z % scalar_block_size::z == 0);

    for (int i = base_coord[0]; i < base_coord[0] + block_size.x;
         i += scalar_block_size::x) {
      for (int j = base_coord[1]; j < base_coord[1] + block_size.y;
           j += scalar_block_size::y) {
        for (int k = base_coord[2]; k < base_coord[2] + block_size.z;
             k += scalar_block_size::z) {
          // FEMInputs::ScalarGrid::Block mu_block;
          // FEMInputs::ScalarGrid::Block lambda_block;
          FEMInputs::ScalarGrid::Block density_block;

          // mu_block.base_coordinates[0] = i;
          // lambda_block.base_coordinates[0] = i;
          density_block.base_coordinates[0] = i;

          // mu_block.base_coordinates[1] = j;
          // lambda_block.base_coordinates[1] = j;
          density_block.base_coordinates[1] = j;

          // mu_block.base_coordinates[2] = k;
          // lambda_block.base_coordinates[2] = k;
          density_block.base_coordinates[2] = k;

          for (int ii = 0; ii < scalar_block_size::x; ii++) {
            for (int jj = 0; jj < scalar_block_size::y; jj++) {
              for (int kk = 0; kk < scalar_block_size::z; kk++) {
                float64 scale = 0;
                if (sparse_flags(i + ii, j + jj, k + kk)
                        .get_inside_container() &&
                    !only_for_nodes) {
                  real d = sparse_density(i + ii, j + jj, k + kk);
                  // Note: we can not feed zero density to the solver as it will
                  // consider the cell to be empty.
                  scale = std::max(d, std::numeric_limits<real>::min());

                  // scale = (1 - minimum_stiffness) * std::pow(d, penalty) +
                  //        minimum_stiffness;
                  auto offset =
                      ChannelMask::Linear_Offset(i + ii, j + jj, k + kk);
                  bool check =
                      fatter_page_map->Test_Page(
                          ChannelMask::Packed_Offset<0, 0, 0>(offset)) &&
                      fatter_page_map->Test_Page(
                          ChannelMask::Packed_Offset<0, 0, 1>(offset)) &&
                      fatter_page_map->Test_Page(
                          ChannelMask::Packed_Offset<0, 1, 0>(offset)) &&
                      fatter_page_map->Test_Page(
                          ChannelMask::Packed_Offset<0, 1, 1>(offset)) &&
                      fatter_page_map->Test_Page(
                          ChannelMask::Packed_Offset<1, 0, 0>(offset)) &&
                      fatter_page_map->Test_Page(
                          ChannelMask::Packed_Offset<1, 0, 1>(offset)) &&
                      fatter_page_map->Test_Page(
                          ChannelMask::Packed_Offset<1, 1, 0>(offset)) &&
                      fatter_page_map->Test_Page(
                          ChannelMask::Packed_Offset<1, 1, 1>(offset));
                  TC_ASSERT(check);
                }
                // mu_block.get(ii, jj, kk) = scale * material.mu;
                // lambda_block.get(ii, jj, kk) = scale * material.lambda;

                density_block.get(ii, jj, kk) = scale;
                bounds[0] = std::min(bounds[0], scale);
                bounds[1] = std::max(bounds[1], scale);
                has_nan = has_nan || (scale != scale);
              }
            }
          }
          // param.mu.blocks.push_back(mu_block);
          // param.lambda.blocks.push_back(lambda_block);
          param.density.blocks.push_back(density_block);
        }
      }
    }
  }

  TC_P(bounds);
  TC_ERROR_IF(has_nan, "density field contains nan!");

  /*
  TC_P(param.density.blocks.size());
  TC_P(block_size.x);
  TC_P(block_size.y);
  TC_P(block_size.z);
  */

  std::map<uint64, real> dirichlet_bc[3];
  for (auto &bc : boundary_conditions) {
    DirichletOnNode dirichlet;
    dirichlet.coord[0] = bc.node[0];
    dirichlet.coord[1] = bc.node[1];
    dirichlet.coord[2] = bc.node[2];
    dirichlet.axis = bc.axis;
    dirichlet.value = bc.val;
    auto ind = ChannelMask::Linear_Offset(bc.node);
    if (dirichlet_bc[bc.axis].find(ind) != dirichlet_bc[bc.axis].end()) {
      // Duplicate
      TC_ASSERT(dirichlet_bc[bc.axis][ind] == dirichlet.value);
    } else {
      // Insert element
      dirichlet_bc[bc.axis][ind] = dirichlet.value;
      param.dirichlet_nodes.push_back(dirichlet);
    }
  }

  param.forces = forces;
  for (auto &f : param.forces) {
    TC_ASSERT(node_flag(Vectori(f.coord[0], f.coord[1], f.coord[2])));
  }

  TC_MEMORY_USAGE("FEM Input Generated");
  param.caller_method = "taichi_topo_opt";
  interface.preserve_output(param.density.blocks.size());
  TC_MEMORY_USAGE("FEM Output Preserved");

  TC_INFO("Invoking FEM Solver");

  std::string fem_fn;
  if (pure_objective) {
    fem_fn = fmt::format("{}/fem_obj/{:05d}.tcb.zip", working_directory, iter);
  } else {
    fem_fn = fmt::format("{}/fem/{:05d}.tcb.zip", working_directory, iter);
  }
  {
    TC_MEMORY_USAGE("Binary Serializer Created");
    BinaryOutputSerializer bser;
    bser.initialize();
    param.io(bser);
    TC_MEMORY_USAGE("Binary Serialization Done");
    bser.finalize();
    TC_MEMORY_USAGE("Binary Serialization Finalized");
    bser.write_to_file(fem_fn);
  }
  TC_MEMORY_USAGE("Binary Serialization Written");
  auto residual = grid->residual();

  param.min_fraction() = 0;  // config.get<real>("current_min_fraction");
  std::string solver = config.get<std::string>("solver", "");
  TC_MEMORY_USAGE("Solver Started");
  TC_PROFILE("solver", interface.run());
  TC_MEMORY_USAGE("Solver Returned");
  TC_INFO("FEM Solver returned");

  using vector_block_size = FEMInputs::ScalarGrid::block_size;

  auto sparse_u0 = grid->u0();
  auto sparse_u1 = grid->u1();
  auto sparse_u2 = grid->u2();

  bounds[0] = 1e30f;
  bounds[1] = -1e30f;

  has_nan = false;

  for (auto &block : interface.outputs.displacements.blocks) {
    int i = block.base_coordinates[0];
    int j = block.base_coordinates[1];
    int k = block.base_coordinates[2];
    for (int ii = 0; ii < vector_block_size::x; ii++) {
      for (int jj = 0; jj < vector_block_size::y; jj++) {
        for (int kk = 0; kk < vector_block_size::z; kk++) {
          if (write_to_u) {
            sparse_u0(i + ii, j + jj, k + kk) = block.get(ii, jj, kk)[0];
            sparse_u1(i + ii, j + jj, k + kk) = block.get(ii, jj, kk)[1];
            sparse_u2(i + ii, j + jj, k + kk) = block.get(ii, jj, kk)[2];
            residual(i + ii, j + jj, k + kk) = block.get(ii, jj, kk)[3];
          } else {
            auto sparse_v0 = grid->v0();
            auto sparse_v1 = grid->v1();
            auto sparse_v2 = grid->v2();
            sparse_v0(i + ii, j + jj, k + kk) = block.get(ii, jj, kk)[0];
            sparse_v1(i + ii, j + jj, k + kk) = block.get(ii, jj, kk)[1];
            sparse_v2(i + ii, j + jj, k + kk) = block.get(ii, jj, kk)[2];
          }

          for (int r = 0; r < dim; r++) {
            bounds[0] = std::min(bounds[0], block.get(ii, jj, kk)[r]);
            bounds[1] = std::max(bounds[1], block.get(ii, jj, kk)[r]);
            has_nan = has_nan ||
                      (block.get(ii, jj, kk)[r] != block.get(ii, jj, kk)[r]);
          }
        }
      }
    }
  }
  TC_MEMORY_USAGE("Displacements fetched");
  TC_P(bounds);
  TC_WARN_UNLESS(interface.outputs.success, "FEM solve has failed!");
  TC_WARN_UNLESS(!has_nan, "FEM solution contains nan!");

  // No states for pure_objective solve
  if (!pure_objective) {
    this->solver_state_ptr = interface.outputs.get_solver_state_ptr();
  }
  bool success = !has_nan && interface.outputs.success;
  TC_MEMORY_USAGE("End FEM Solve");
  return success;
}

// returns: objective
real Opt::iterate(int iter) {
  TC_MEMORY_USAGE("Begin Iteration");
  Profiler _("TopoOpt");
  if (iter >= config.get<int>("grid_update_start", 5)) {
    Profiler _("update page map");
    TC_PROFILE("update thin_page_map", update_thin_page_map());
    if (config.get("connectivity_filtering", true)) {
      TC_PROFILE("filter_isolated", filter_isolated_blocks(iter));
    }
    TC_PROFILE("update_fat_page_map", update_fat_page_map());
  }
  TC_MEMORY_USAGE("Page Map Updated");

  real objective = 0;
  // Clear gradients
  clear_gradients();

  TC_MEMORY_USAGE("Gradients cleared");

  std::map<std::string, real> evaluated;
  {
    TC_PROFILER("compute derivative");
    for (int i = 0; i < (int)objectives.size(); i++) {
      auto objective_name = fmt::format("[{}]{}", i, objectives[i]->get_name());
      TC_PROFILER(objective_name);
      auto obj = objectives[i]->compute(Dict("iter", iter));
      TC_TRACE("  {} = {}", objective_name, obj);
      evaluated[objective_name] = obj;
      objective += obj;
    }
  }
  TC_WARN("Total objective: {}", objective);
  for (auto kv : evaluated) {
    TC_WARN("   {}: {}", kv.first, kv.second);
  }
  TC_MEMORY_USAGE("Objectives computed");
  TC_PROFILE("smooth_dc", smooth_dc());
  TC_MEMORY_USAGE("Gradient smoothed");
  TC_PROFILE("optimize_density", optimizer->optimize(Dict("iter", iter)));
  TC_MEMORY_USAGE("Density updated");
  TC_MEMORY_USAGE("End Iteration");
  return objective;
}

void Opt::threshold_and_kill(real threshold) {
  auto density = grid->density();
  auto flags = grid->flags();
  for (auto ind : get_cell_region()) {
    if (density(ind.get_ipos()) < threshold) {
      flags(ind.get_ipos()).set_inside_container(false);
    }
  }
  filter_isolated_voxels(0);
  update_thin_page_map();
  update_fat_page_map();
}

void Opt::load_density_from_fem(const std::string &filename) {
  TC_INFO("Loading from fem file {}", filename);
  fem_interface::FEMInputs param;
  read_from_binary_file(param, filename);

  // Clear density field
  auto blocks = container_page_map->Get_Blocks();
  auto density = grid->density();
  auto flags = grid->flags();

  using scalar_block_size = fem_interface::FEMInputs::ScalarGrid::block_size;
  auto input_block_size = Vector3i(scalar_block_size::x, scalar_block_size::y,
                                   scalar_block_size::z);

  for (uint i = 0; i < blocks.second; i++) {
    auto offset = blocks.first[i];
    for (unsigned int e = 0; e < ChannelMask::elements_per_block;
         ++e, offset += sizeof(real)) {
      density(offset) = 0;
    }
  }

  bool map_back = false;
  if (param.penalty() == 0) {
    map_back = true;
  }
  // Load from fem blocks
  int non_air_cell = 0;
  for (auto &block : param.density.blocks) {
    auto base_coord = Vector3i::from_array(block.base_coordinates);
    for (auto ind : Region(Vector3i(0), input_block_size)) {
      auto coord = base_coord + ind.get_ipos();
      // Convert back
      real data = block.get(ind.i, ind.j, ind.k);
      if (data == 0)
        continue;
      if (!flags(coord).get_inside_container()) {
        non_air_cell += 1;
        continue;
      }
      if (map_back) {
        data = std::pow(data - 1e-6_f, 0.33333333_f) * (1 - 1e-6_f);
      }
      TC_ASSERT_INFO(-1e-6_f <= data && data <= 1 + 1e-6_f,
                     fmt::format("abnormal data {}", data));
      data = clamp(data, 0.0_f, 1.0_f);
      TC_ASSERT(flags(coord).get_inside_container());
      density(coord) = data;
    }
  }
  update_thin_page_map();
  update_fat_page_map();
  if (non_air_cell) {
    TC_WARN("{} non-air cell outside container detected", non_air_cell);
  }
  TC_INFO("loaded.");
}

void Opt::filter_isolated_blocks(int iter) {
  // BFS
  // We only consider 6 connectivity instead of 26

  // this should operate on thin_page_map
  std::vector<int64> component_sizes;
  PageMap visited(*grid->grid);
  std::deque<uint64> Q;
  int64 total_size = 0;

  auto blocks = thin_page_map->Get_Blocks();
  std::unordered_map<uint64, int> color;

  auto BFS = [&](uint64 offset) {
    int count = 0;
    auto visit = [&](uint64 neighbour) {
      if (thin_page_map->Test_Page(neighbour) &&
          !visited.Test_Page(neighbour)) {
        Q.push_back(neighbour);
        visited.Set_Page(neighbour);
        color[neighbour] = (int)component_sizes.size();
        count += 1;
      }
    };

    Q.resize(0);
    visit(offset);

    while (!Q.empty()) {
      auto u = Q.front();
      Q.pop_front();
      visit(
          ChannelMask::template Packed_Offset<+(1 << ChannelMask::block_xbits),
                                              0, 0>(u));
      visit(
          ChannelMask::template Packed_Offset<-(1 << ChannelMask::block_xbits),
                                              0, 0>(u));
      visit(ChannelMask::template Packed_Offset<
            0, +(1 << ChannelMask::block_ybits), 0>(u));
      visit(ChannelMask::template Packed_Offset<
            0, -(1 << ChannelMask::block_ybits), 0>(u));
      visit(ChannelMask::template Packed_Offset<
            0, 0, +(1 << ChannelMask::block_zbits)>(u));
      visit(ChannelMask::template Packed_Offset<
            0, 0, -(1 << ChannelMask::block_zbits)>(u));
    }
    component_sizes.push_back(count);
    total_size += count;
  };

  for (uint b = 0; b < blocks.second; b++) {
    auto offset = blocks.first[b];
    if (!visited.Test_Page(offset)) {
      BFS(offset);
    }
  }

  TC_P(total_size);
  TC_P(blocks.second);
  TC_ASSERT(total_size == blocks.second);
  // TODO: Is is still possible, that the blocks are connected while the voxels
  // are not
  TC_ASSERT(!component_sizes.empty());
  if (component_sizes.size() == 1) {
    TC_INFO("Only one connected component detected.");
    return;
  }

  TC_WARN("There are more than one block components detected");

  int larger_than_30_count = 0;
  int larger_than_30_index = -1;
  for (int i = 0; i < (int64)component_sizes.size(); i++) {
    int64 s = component_sizes[i];
    real fraction = 1.0_f * s / total_size;
    TC_WARN("  Component size: {:9d} ({:5.2f}%)", s, 100.0_f * fraction);
    if (fraction > 0.3) {
      larger_than_30_count += 1;
      larger_than_30_index = i;
      if (larger_than_30_count > 1) {
        TC_ERROR(
            "There are more than two connected components with fraction > 30%. "
            "Something may be wrong.");
      }
    }
  }

  TC_ERROR_IF(
      larger_than_30_count == 0,
      "All compoments have fraction < 30%. How can this be so fragmental?");

  // larger_than_30_count must be 1 now.

  // Filtering
  for (uint b = 0; b < blocks.second; b++) {
    auto offset = blocks.first[b];

    if (color[offset] != larger_than_30_index) {
      thin_page_map->Unset_Page(offset);
    }
  }
  thin_page_map->Update_Block_Offsets();

  auto left = thin_page_map->Get_Blocks().second;
  TC_WARN("{} ({:.2f}%) blocks remained after connectivity filtering.", left,
          100.0_f * left / total_size);
}

void Opt::filter_isolated_voxels(int iter) {
  // BFS
  // We only consider 6 connectivity instead of 26

  // this should operate on thin_page_map
  std::vector<int64> component_sizes;
  PageMap visited(*grid->grid);
  std::deque<uint64> Q;
  int64 total_size = 0;

  auto blocks = thin_page_map->Get_Blocks();
  auto flags = grid->flags();
  auto density = grid->density();
  auto color = grid->color();

  auto BFS = [&](uint64 offset) {
    int64 count = 0;
    auto visit = [&](uint64 neighbour) {
      if (thin_page_map->Test_Page(neighbour) &&
          flags(neighbour).get_inside_container() &&
          !flags(neighbour).get_visited()) {
        Q.push_back(neighbour);
        visited.Set_Page(neighbour);
        flags(neighbour).set_visited(true);
        color(neighbour) = (int)component_sizes.size();
        count += 1;
      }
    };

    Q.resize(0);
    visit(offset);

    while (!Q.empty()) {
      auto u = Q.front();
      Q.pop_front();
      visit(ChannelMask::template Packed_Offset<1, 0, 0>(u));
      visit(ChannelMask::template Packed_Offset<-1, 0, 0>(u));
      visit(ChannelMask::template Packed_Offset<0, 1, 0>(u));
      visit(ChannelMask::template Packed_Offset<0, -1, 0>(u));
      visit(ChannelMask::template Packed_Offset<0, 0, +1>(u));
      visit(ChannelMask::template Packed_Offset<0, 0, -1>(u));
    }
    component_sizes.push_back(count);
    total_size += count;
  };

  for (uint b = 0; b < blocks.second; b++) {
    auto offset = blocks.first[b];
    for (unsigned int e = 0; e < ChannelMask::elements_per_block;
         ++e, offset += sizeof(real)) {
      flags(offset).set_visited(false);
    }
  }

  for (uint b = 0; b < blocks.second; b++) {
    auto offset = blocks.first[b];
    for (unsigned int e = 0; e < ChannelMask::elements_per_block;
         ++e, offset += sizeof(real)) {
      // This works because float32 and int32 has the same size
      if (flags(offset).get_inside_container() &&
          !flags(offset).get_visited()) {
        TC_INFO("  Visiting connected component {}", component_sizes.size());
        BFS(offset);
      }
    }
  }

  TC_ASSERT(!component_sizes.empty());
  if (component_sizes.size() == 1) {
    TC_INFO("Only one connected component detected.");
    return;
  }

  TC_WARN("There are more than one voxel components detected");

  int larger_than_30_count = 0;
  int larger_than_30_index = -1;
  for (uint i = 0; i < component_sizes.size(); i++) {
    int64 s = component_sizes[i];
    real fraction = 1.0_f * s / total_size;
    TC_WARN("  Component size: {:9d} ({:5.2f}%)", s, 100.0_f * fraction);
    if (fraction > 0.3) {
      larger_than_30_count += 1;
      larger_than_30_index = i;
      if (larger_than_30_count > 1) {
        TC_ERROR(
            "There are more than two connected components with fraction > 30%. "
            "Something may be wrong.");
      }
    }
  }

  TC_ERROR_IF(
      larger_than_30_count == 0,
      "All compoments have fraction < 30%. How can this be so fragmental?");

  // larger_than_30_count must be 1 now.

  int64 left = 0;

  // Filtering
  for (uint b = 0; b < blocks.second; b++) {
    auto offset = blocks.first[b];
    for (uint e = 0; e < ChannelMask::elements_per_block;
         ++e, offset += sizeof(real)) {
      if (flags(offset).get_inside_container()) {
        if (color(offset) != larger_than_30_index) {
          // Delete it
          flags(offset).set_inside_container(false);
          density(offset) = 0;
        } else {
          left += 1;
        }
      }
    }
  }
  thin_page_map->Update_Block_Offsets();

  TC_WARN(
      "{} ({:.2f}%) voxels remained after connectivity filtering. Filtered "
      "voxels are eliminated permenantly (marked as outside container)",
      left, 100.0_f * left / total_size);
}

void Opt::make_shell_mesh(const std::string &mesh_fn, real maximum_distance) {
  ElementMesh<3> mesh;
  Config cfg;
  cfg.set("mesh_fn", mesh_fn);
  mesh.initialize(cfg);
  // Force has a linear falloff
  // Scatter normal to nodes
  auto flags = grid->flags();
  std::unordered_map<uint64, Vector3> normals;
  auto density = grid->density();
  real fixed_cell_density = config.get("fixed_cell_density", 1.0_f);
  int count = 0;
  for (auto e : mesh.elements) {
    Vector3 bounding_box_f[2];
    bounding_box_f[0] = Vector3(1e5);
    bounding_box_f[1] = Vector3(-1e5);
    for (int i = 0; i < 3; i++) {
      for (int k = 0; k < 3; k++) {
        bounding_box_f[0][k] = std::min(bounding_box_f[0][k], e.v[i][k]);
        bounding_box_f[1][k] = std::max(bounding_box_f[1][k], e.v[i][k]);
      }
    }
    Vector3i bounding_box[2];
    bounding_box[0] = f2i(bounding_box_f[0] - Vector(maximum_distance));
    bounding_box[1] =
        f2i(bounding_box_f[1] + Vector(maximum_distance)) + Vector3i(1);

    for (auto ind : Region(bounding_box[0], bounding_box[1])) {
      Vector pos = i2f(ind.get_ipos());
      if (!flags(ind.get_ipos()).get_inside_container()) {
        continue;
      }
      real dist = abs(distance_to_triangle(pos, e));
      if (dist > maximum_distance) {
        continue;
      }
      flags(ind.get_ipos()).set_fixed_density(true);
      density(ind.get_ipos()) = fixed_cell_density;
    }
  }
  TC_INFO("Making shell to {} cells", count);
}

// strength = -1 means plane wing force
void Opt::add_mesh_normal_force(const std::string &mesh_fn,
                                real strength,
                                Vector3 center,
                                real falloff,
                                real maximum_distance,
                                Vector3 override,
                                bool no_z) {
  ElementMesh<3> mesh;
  Config cfg;
  cfg.set("mesh_fn", mesh_fn);
  mesh.initialize(cfg);
  // Force has a linear falloff
  // Scatter normal to nodes
  auto flags = grid->flags();
  std::unordered_map<uint64, Vector3> normals;
  for (auto e : mesh.elements) {
    Vector3 bounding_box_f[2];
    bounding_box_f[0] = Vector3(1e5);
    bounding_box_f[1] = Vector3(-1e5);
    for (int i = 0; i < 3; i++) {
      for (int k = 0; k < 3; k++) {
        bounding_box_f[0][k] = std::min(bounding_box_f[0][k], e.v[i][k]);
        bounding_box_f[1][k] = std::max(bounding_box_f[1][k], e.v[i][k]);
      }
    }
    Vector3i bounding_box[2];
    bounding_box[0] = f2i(bounding_box_f[0] - Vector(maximum_distance));
    bounding_box[1] =
        f2i(bounding_box_f[1] + Vector(maximum_distance)) + Vector3i(1);

    for (auto ind : Region(bounding_box[0], bounding_box[1])) {
      Vector pos = i2f(ind.get_ipos());
      real weight = std::max(0.0_f, 1 - length(pos - center) / falloff);
      if (weight <= 0 || !flags(ind.get_ipos()).get_inside_container()) {
        continue;
      }
      real dist = abs(distance_to_triangle(pos, e));
      if (dist > maximum_distance) {
        continue;
      }
      auto offset = ChannelMask::Linear_Offset(ind.get_ipos());
      normals[offset] += e.get_normal();
    }
  }
  for (auto &kv : normals) {
    auto node = Vector3i(ChannelMask::LinearToCoord(kv.first));
    auto pos = i2f(node);
    real weight = std::max(0.0_f, 1 - length(pos - center) / falloff);
    Vector3 n = kv.second;
    if (length2(n) < 1e-6_f) {
      continue;
    }
    n = normalized(n);
    if (length(override)) {
      n = override;
    }
    if (strength != -1) {
      if (!(no_z && std::abs(n.z) >= 0.99_f)) {
        add_force(node, n * strength * weight);
      }
    } else {
      // Wing force
      real y = n.y;
      if (abs(y) > 0.05_f) {
        // Avoid singularity
        if (y > 0) {  // Upper surface
          add_force(node, -n * y * weight);
        } else {  // Lower surface
          add_force(node, -n * (0.4_f / y) * weight);
        }
      }
    }
  }
  TC_INFO("Adding force to {} nodes", normals.size());
}

real Opt::threshold_density_trail(real threshold, bool apply) {
  real effective_voxels = 0;
  auto blocks = fat_page_map->Get_Blocks();
  auto density = this->grid->density();
  auto flags = this->grid->flags();
  for (uint64 b = 0; b < blocks.second; b++) {
    auto offset = blocks.first[b];
    for (unsigned int e = 0; e < ChannelMask::elements_per_block;
         ++e, offset += sizeof(real)) {
      if (density(offset) >= threshold || flags(offset).get_fixed_density()) {
        effective_voxels += 1;
        if (apply) {
          density(offset) = 1;
        }
      } else {
        if (apply) {
          density(offset) = 0;
        }
      }
    }
  }
  return effective_voxels;
}

void Opt::threshold_density() {
  real lo = 0, hi = 1;
  // Binary search for a threshold
  real target_voxels = (volume_fraction * this->total_container_voxels);
  while (lo + 1e-8_f < hi) {
    real mid = (lo + hi) * 0.5_f;
    auto effective_voxels = this->threshold_density_trail(mid, false);
    TC_WARN(" t={}, v={}, target={}", mid, effective_voxels, target_voxels);
    if (effective_voxels > target_voxels) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  real threshold = lo;
  TC_WARN("Final threshold = {}", threshold);
  threshold_density_trail(threshold, true);
}

float64 Opt::compute_objective(real threshold) {
  grid_ = std::make_unique<TopOptGrid>(*grid);
  // grid is now the grid to threshold density
  // grid_ is the backup
  update_thin_page_map();
  auto density = grid->density();
  auto flags = grid->flags();
  auto blocks = fat_page_map->Get_Blocks();
  for (uint64 b = 0; b < blocks.second; b++) {
    auto offset = blocks.first[b];
    for (unsigned int e = 0; e < ChannelMask::elements_per_block;
         ++e, offset += sizeof(real)) {
      if (density(offset) >= threshold) {
        density(offset) = 0.5_f;
      } else {
        density(offset) = 0;
        flags(offset).set_inside_container(false);
      }
    }
  }
  filter_isolated_voxels(0);
  update_thin_page_map();
  update_fat_page_map();

  TC_NOT_IMPLEMENTED
  // fem_solve(last_iter, true);
  float64 compliance = 0;
  // float64 compliance = objectives[0]->compute(Dict("iter", 0));
  // delete the threshold grid
  set_grid(std::move(grid_));
  grid_ = nullptr;
  return compliance;
}

TC_IMPLEMENTATION(Simulation3D,
                  SPGridTopologyOptimization3D,
                  "spgrid_topo_opt");

class GripperTexture final : public Texture {
 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
  }

  virtual Vector4 sample(const Vector3 &coord_) const override {
    auto coord = coord_ - Vector3(0.5_f);
    bool inside = true;
    if (std::abs(coord.z) > 0.1) {
      inside = false;
    }
    if (std::abs(coord.x) < 0.3 && coord.y > 0) {
      inside = false;
    }
    return inside ? Vector4(1) : Vector4(0);
  }
};

TC_IMPLEMENTATION(Texture, GripperTexture, "gripper");

TC_NAMESPACE_END
