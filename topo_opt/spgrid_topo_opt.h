#pragma once

#include <taichi/util.h>
#include <taichi/math.h>
#include <taichi/common/bit.h>
#include <taichi/io/io.h>
#include <taichi/system/threading.h>
#include <taichi/system/memory.h>
#include <taichi/visualization/particle_visualization.h>
#include <taichi/common/task.h>
#include <taichi/dynamics/simulation.h>
#include <taichi/visualization/pakua.h>
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include <SPGrid/Tools/SPGrid_Clear.h>
#include <mutex>
#include <taichi/system/profiler.h>
#include <pybind11/pybind11.h>
#include "fem_interface.h"
#include "hex_fem.h"
#include "objective/objective.h"
#include "optimizer/optimizer.h"

TC_NAMESPACE_BEGIN

using namespace SPGrid;

static_assert(sizeof(real) == 8,
              "Please compile Taichi and TopOpt in double precision.");

struct TopOptFlags : public bit::Bits<64> {
  using Base = bit::Bits<64>;
  // Is this cell inside the container?
  TC_BIT_FIELD(bool, inside_container, 0);
  // Does this cell has forces near it?
  TC_BIT_FIELD(bool, has_force, 1);
  // Fix density to be 1?
  TC_BIT_FIELD(bool, fixed_density, 2);
  // For BFS?
  TC_BIT_FIELD(bool, visited, 3);

  TC_BIT_FIELD(bool, has_dirichlet, 4);
};

// SOA
struct NodeStruct {
  // 0 - 7
  TopOptFlags flags;
  float64 density;
  union {
    int64 color;
    float64 residual;
  };
  float64 dc0;
  float64 dc1;
  float64 u0;
  float64 u1;
  float64 u2;
#if defined(TC_TARGET_DEFORMATION)
  // 8 - 15
  float64 v0;
  float64 v1;
  float64 v2;
  float64 t0;
  float64 t1;
  float64 t2;
  float64 t3;
  float64 t4;
#endif
};

static_assert(bit::is_power_of_two(sizeof(NodeStruct)),
              "NodeStruct size is not POT");

struct BlockMeta {
  real density_min;
  real density_max;
  real density_sum;
  TC_IO_DEF(density_min, density_max);

  BlockMeta() {
    density_max = 0;
    density_min = 1;
    density_sum = 0;
  }
};

#if defined(TC_TARGET_DEFORMATION)
static constexpr int log2_size = 13;
#else
static constexpr int log2_size = 12;
#endif
static constexpr int padding = 8;
static constexpr int page_size = 1 << log2_size;

struct TopOptGrid {
  struct Block {
    uint64_t offset;
    uint8 data[page_size];
    TC_IO_DEF(offset, data);
  };

  int version = 1;
  Vector3i res;
  Dict extras;
  bool verbose = false;  // write everything, or just density and flag?

  using Grid = SPGrid_Allocator<NodeStruct, 3, log2_size>;
  // Note: make sure the channels have the same size!
  using Mask = typename decltype(
      std::declval<Grid>().Get_Array(&NodeStruct::density))::MASK;
  using PageMap = SPGrid_Page_Map<log2_size>;

  std::unique_ptr<Grid> grid;
  // for all blocks in the container
  std::unique_ptr<PageMap> container_page_map;
  // for all active blocks
  std::unique_ptr<PageMap> thin_page_map;
  // for all active blocks with surroundings
  std::unique_ptr<PageMap> fat_page_map;
  // for all active blocks with surroundings, for NODEs.
  // Fatter than fat_page_map, which is for CELLs.
  std::unique_ptr<PageMap> fatter_page_map;

  TopOptGrid() {
    extras.set("write_whole", true);
  }

  TopOptGrid(Vector3i res) : res(res) {
    grid = std::make_unique<Grid>(res[0], res[1], res[2]);
    container_page_map = std::make_unique<PageMap>(*grid);
    thin_page_map = std::make_unique<PageMap>(*grid);
    fat_page_map = std::make_unique<PageMap>(*grid);
    fatter_page_map = std::make_unique<PageMap>(*grid);
  }

  TopOptGrid(TopOptGrid &o) : TopOptGrid(o.res) {
    copy_page_map(o.container_page_map.get(), this->container_page_map.get());
    copy_page_map(o.thin_page_map.get(), this->thin_page_map.get());
    copy_page_map(o.fat_page_map.get(), this->fat_page_map.get());
    copy_page_map(o.fatter_page_map.get(), this->fatter_page_map.get());
    auto blocks = container_page_map->Get_Blocks();
    auto array = grid->Get_Array();
    auto o_array = o.grid->Get_Array();
    for (uint b = 0; b < blocks.second; b++) {
      auto offset = blocks.first[b];
      std::memcpy((void *)&array(offset), (void *)&o_array(offset), page_size);
    }
  }

#define FIELD(x)                            \
  TC_FORCE_INLINE auto x() {                \
    return grid->Get_Array(&NodeStruct::x); \
  }

  FIELD(density);
  FIELD(color);
  FIELD(residual);
  FIELD(flags);
  FIELD(dc0);
  FIELD(dc1);
  FIELD(u0);
  FIELD(u1);
  FIELD(u2);
#if !defined(TC_TARGET_DEFORMATION)
#undef FIELD
#define FIELD(x)                                              \
  TC_FORCE_INLINE auto x() {                                  \
    TC_ERROR("Please recompile with TC_TARGET_DEFORMATION."); \
    return grid->Get_Array(&NodeStruct::dc1);                 \
  }
#endif
  FIELD(v0);
  FIELD(v1);
  FIELD(v2);
  FIELD(t0);
  FIELD(t1);
  FIELD(t2);
  FIELD(t3);
  FIELD(t4);

#undef FIELD

  using ChannelMask = typename decltype(
      std::declval<Grid>().Get_Array(&NodeStruct::density))::MASK;

  void update_block_offsets() {
    container_page_map->Update_Block_Offsets();
    thin_page_map->Update_Block_Offsets();
    fat_page_map->Update_Block_Offsets();
    fatter_page_map->Update_Block_Offsets();
  }

  bool get_thin() const {
    return extras.get("thin", false);
  }

  void set_thin(bool val) {
    extras.set("thin", val);
  }

  TC_IO_DECL {
    TC_IO(version, res, extras, verbose);

    auto write_page_map = [&](std::unique_ptr<PageMap> &pm) {
      pm->Update_Block_Offsets();
      auto blocks = pm->Get_Blocks();
      std::vector<uint64_t> offsets(blocks.first, blocks.first + blocks.second);
      TC_IO(offsets);
    };

    auto read_page_map = [&](std::unique_ptr<PageMap> &pm) {
      pm = std::make_unique<PageMap>(*grid);
      std::vector<uint64_t> offsets;
      TC_IO(offsets);
      pm->Clear();
      for (auto &o : offsets) {
        pm->Set_Page(o);
      }
      pm->Update_Block_Offsets();
    };

    TopOptGrid *p = const_cast<TopOptGrid *>(this);

    PageMap *page_map_io;
    if (TC_SERIALIZER_IS(BinaryOutputSerializer)) {
      write_page_map(p->container_page_map);
      write_page_map(p->thin_page_map);
      write_page_map(p->fat_page_map);
      write_page_map(p->fatter_page_map);

      if (get_thin()) {
        page_map_io = p->thin_page_map.get();
      } else {
        page_map_io = p->container_page_map.get();
      }

      // Output
      auto bptr = page_map_io->Get_Blocks();
      if (verbose) {
        auto array = grid->Get_Array();
        std::vector<Block> blocks;
        blocks.reserve(bptr.second);
        for (size_t i = 0; i < bptr.second; i++) {
          Block blk;
          blk.offset = bptr.first[i];
          std::memmove(blk.data, (void *)&array(blk.offset), page_size);
          blocks.push_back(blk);
        }
        TC_IO(blocks);
      } else {
        auto density = p->density();
        auto flags = p->flags();
        std::vector<real> data_densities;
        std::vector<uint32> data_flags;
        for (size_t i = 0; i < bptr.second; i++) {
          auto offset = bptr.first[i];
          for (unsigned int e = 0; e < ChannelMask::elements_per_block;
               ++e, offset += sizeof(real)) {
            data_densities.push_back(density(offset));
            data_flags.push_back(flags(offset).get());
          }
        }
        TC_IO(data_densities);
        TC_IO(data_flags);
      }
    } else if (TC_SERIALIZER_IS(BinaryInputSerializer)) {
      // Input, initialize
      p->grid = std::make_unique<Grid>(res[0], res[1], res[2]);
      read_page_map(p->container_page_map);
      read_page_map(p->thin_page_map);
      read_page_map(p->fat_page_map);
      read_page_map(p->fatter_page_map);

      // Note this must be done after read_page_map.
      if (get_thin()) {
        page_map_io = p->thin_page_map.get();
      } else {
        page_map_io = p->container_page_map.get();
      }

      if (verbose) {
        auto array = grid->Get_Array();
        std::vector<Block> blocks;
        TC_IO(blocks);
        for (auto &blk : blocks) {
          std::memmove((void *)&array(blk.offset), blk.data, page_size);
        }
      } else {
        auto bptr = page_map_io->Get_Blocks();
        auto density = p->density();
        auto flags = p->flags();

        std::vector<real> data_densities;
        std::vector<uint32> data_flags;
        TC_IO(data_densities);
        TC_IO(data_flags);
        int t = 0;
        for (size_t i = 0; i < bptr.second; i++) {
          auto offset = bptr.first[i];
          for (unsigned int e = 0; e < ChannelMask::elements_per_block;
               ++e, offset += sizeof(real)) {
            density(offset) = data_densities[t];
            flags(offset).set(data_flags[t]);
            t += 1;
          }
        }
      }
    } else {
      TC_NOT_IMPLEMENTED;
    }
  }

  static void copy_page_map(PageMap *a, PageMap *b) {
    b->Clear();
    auto blocks = a->Get_Blocks();
    for (uint i = 0; i < blocks.second; i++) {
      b->Set_Page(blocks.first[i]);
    }
    b->Update_Block_Offsets();
  }
};

class SPGridTopologyOptimization3D : public Simulation<3> {
 public:
  using Base = Simulation<3>;
  constexpr static int dim = 3;
  using Vector = VectorND<dim, real>;
  using Vectori = VectorND<dim, int>;
  template <typename T>
  using Array = ArrayND<dim, T>;
  using Region = RegionND<dim>;
  using Index = IndexND<dim>;
  using SparseGrid = SPGrid_Allocator<NodeStruct, 3, log2_size>;
  using ChannelMask = TopOptGrid::ChannelMask;
  using PageMap = TopOptGrid::PageMap;
  using DomainFunc = std::function<real(const Vector &)>;

  // ***************************************************************************
  // States
  Config config;
  Vectori container_bounds[2];
  int total_container_voxels;
  real dx, inv_dx;
  bool fix_cells_near_force;
  bool fix_cells_at_dirichlet;
  real volume_fraction;
  real minimum_density, minimum_stiffness;
  Material material;
  Vector3i block_size;
  int last_iter;
  std::unique_ptr<TopOptGrid> grid;
  std::vector<BlockMeta> block_meta;
  std::vector<Dict> general_actions;
  std::map<std::string, std::vector<uint8>> extras;
  // Forces
  Dict domain_config;
  std::vector<std::unique_ptr<Objective>> objectives;
  Objective *active_objective;
  std::unique_ptr<Optimizer> optimizer;

  DomainFunc domain_func;

  TC_IO_DECL {
    TC_IO(config);
    TC_IO(container_bounds);
    TC_IO(total_container_voxels);
    TC_IO(dx, inv_dx);
    TC_IO(volume_fraction);
    TC_IO(minimum_density, minimum_stiffness);
    TC_IO(material);
    TC_IO(block_size);
    TC_IO(last_iter)
    TC_IO(grid);
    TC_IO(block_meta);
    TC_IO(general_actions);
    TC_IO(domain_config);
    TC_IO(extras);
    if (TC_SERIALIZER_IS(BinaryInputSerializer)) {
      const_cast<SPGridTopologyOptimization3D *>(this)->wrangle_page_maps();
      const_cast<SPGridTopologyOptimization3D *>(this)->initialize_fem_kernel();
      const_cast<SPGridTopologyOptimization3D *>(this)->fix_cells_near_force =
          config.get("fix_cells_near_force", false);
      const_cast<SPGridTopologyOptimization3D *>(this)->fixed_cell_density_ =
          config.get<real>("fixed_cell_density", 1.0_f);
    }
    // TODO: what happened here?
    // TC_IO(objectives);
    // TC_IO(optimizer);
    // TC_IO(active_objective);
  }

  // ***************************************************************************
  // No need to remember
  std::unique_ptr<TopOptGrid> grid_;
  void *solver_state_ptr = nullptr;
  std::unique_ptr<HexFEMSolver<dim>> fem;
  real fixed_cell_density_;

  // Below are just handy shortcuts - should point to the page maps in "grid"
  PageMap *container_page_map;
  PageMap *thin_page_map;
  PageMap *fat_page_map;
  PageMap *fatter_page_map;

  SPGridTopologyOptimization3D() : Base() {
  }

  void initialize(const Config &config) override;

  Region get_cell_region() {
    return Region(container_bounds[0], container_bounds[1]);
  }

  Region get_node_region() {
    return Region(container_bounds[0], container_bounds[1] + Vectori(1),
                  Vector(0.0_f));
  }

  void set_grid(std::unique_ptr<TopOptGrid> new_grid) {
    grid = std::move(new_grid);
    wrangle_page_maps();
  }

  void wrangle_page_maps() {
    container_page_map = grid->container_page_map.get();
    thin_page_map = grid->thin_page_map.get();
    fat_page_map = grid->fat_page_map.get();
    fatter_page_map = grid->fatter_page_map.get();
  }

  void initialize_fem_kernel();

  Vectori get_container_size() const {
    return container_bounds[1] - container_bounds[0];
  }

  // Prone blocks with all minimum density
  // Fat page map -> new thin page map
  void update_thin_page_map();

  // Fat page map -> new thin page map
  void update_fat_page_map();

  // Integer to float
  TC_FORCE_INLINE auto i2f(Vector ind) {
    return (ind - Vector(padding)) * (real)dx - Vector(0.5_f);
  }

  // Integer to float
  TC_FORCE_INLINE auto i2f(Vectori ind) {
    return (ind - Vectori(padding)).template cast<real>() * (real)dx -
           Vector(0.5_f);
  }

  TC_FORCE_INLINE auto f2i(Vector pos) {
    return ((pos + Vector(0.5)) * (real)inv_dx).template cast<int>() +
           Vectori(padding);
  }

  // TODO: remove normalize_pos
  TC_FORCE_INLINE auto normalize_pos(Vector ind) {
    return i2f(ind);
  }

  real get_volume_fraction(int iter) {
    int steps = config.get("progressive_vol_frac", 0);
    if (steps == 0) {
      return volume_fraction;
    } else {
      return lerp(std::min(1.0_f / steps * iter, 1.0_f), 1.0_f,
                  volume_fraction);
    }
  }

  void populate_grid(const std::function<real(Vector3)> &sdf,
                     const std::string &dirichlet,
                     const std::string &mirror);

  bool add_force(Vector pos, Vector f, Vectori size = Vectori(1)) {
    Vectori ipos =
        ((pos + Vector(0.5)) * inv_dx).template cast<int>() + Vectori(padding);
    return add_force(ipos, f, size);
  }

  // extreme = +/-1
  void add_plane_force(Vector f,
                       int axis,
                       int extreme,
                       Vector bound1,
                       Vector bound2) {
    int node_count = 0;
    TC_ASSERT(extreme == 1 || extreme == -1);
    int val = container_bounds[(extreme + 1) / 2][axis];
    for (auto &ind : get_node_region()) {
      Vector p = normalize_pos(ind.get_pos());
      if (node_flag(ind.get_ipos()) && ind.get_ipos()[axis] == val &&
          bound1 <= p && p <= bound2) {
        node_count += 1;
      }
    }
    TC_TRACE("Adding force to {} nodes.", node_count);
    for (auto &ind : get_node_region()) {
      Vector p = normalize_pos(ind.get_pos());
      if (node_flag(ind.get_ipos()) && ind.get_ipos()[axis] == val &&
          bound1 <= p && p <= bound2) {
        add_force(ind.get_ipos(), f * (1.0_f / node_count), Vectori(1));
      }
    }
  }

  void add_precise_plane_force_bridge() {
    int axis = 1, extreme = -1;
    int cell_count = 0;
    Vector3 f(0, -1, 0);
    TC_ASSERT(extreme == -1);
    int val = container_bounds[(extreme + 1) / 2][axis];
    auto flag = grid->flags();
    for (auto &ind : get_cell_region()) {
      if (flag(ind.get_ipos()).get_inside_container() &&
          ind.get_ipos()[axis] == val) {
        cell_count += 1;
      }
    }
    TC_TRACE("Adding force to {} cells (faces).", cell_count);
    auto offsets = {Vector3i(0, 0, 0), Vector3i(-1, 0, 0), Vector3i(0, 0, -1),
                    Vector3i(-1, 0, -1)};
    auto count = 0;
    auto total_scale = 0.0_f;
    for (auto &ind : get_node_region()) {
      int neighbouring_cells = 0;
      for (auto o : offsets) {
        if (flag(ind.get_ipos() + o).get_inside_container() &&
            ind.get_ipos()[axis] == val) {
          neighbouring_cells += 1;
        }
      }
      if (neighbouring_cells > 0) {
        real scale = 1.0_f / 4.0_f / cell_count * neighbouring_cells;
        add_force(ind.get_ipos(), f * scale, Vectori(1));
        total_scale += scale;
      }
    }
    TC_P(total_scale);
  }

  bool node_flag(Vectori ipos) const {
    auto sparse_flags = grid->flags();
    bool has_neighbour = false;
    Region region(Vectori(-1), Vectori(1));
    for (auto &ind : region) {
      if (sparse_flags(ind.get_ipos() + ipos).get_inside_container()) {
        has_neighbour = true;
      }
    }
    return has_neighbour;
  }

  // extreme = +/-1
  void add_plane_dirichlet_bc(const std::string &fix_to_zero,
                              int axis,
                              int extreme = 1,
                              Vector value = Vector(0)) {
    // TODO: corner case?
    auto sparse_flags = grid->flags();
    TC_ASSERT(extreme == 1 || extreme == -1);
    int val =
        container_bounds[(extreme + 1) / 2][axis] - (extreme == 1 ? 1 : 0);
    for (auto &ind : get_cell_region()) {
      if (sparse_flags(ind.get_ipos()).get_inside_container() &&
          ind.get_ipos()[axis] == val) {
        add_cell_boundary(ind.get_ipos(), fix_to_zero, value);
      }
    }
  }

  bool add_force(Vectori ipos, Vector f, Vectori size = Vectori(1)) {
    if (size == Vectori(1)) {
      auto sparse_flags = grid->flags();
      auto sparse_density = grid->density();
      fem_interface::ForceOnNode force;
      for (int i = 0; i < dim; i++) {
        force.coord[i] = ipos[i];
      }
      for (auto &ind : Region(-Vectori(1), Vectori(1))) {
        auto offset = ChannelMask::Linear_Offset(ipos + ind.get_ipos());
        if (sparse_flags(offset).get_inside_container()) {
          sparse_flags(offset).set_has_force(true);
          if (fix_cells_near_force) {
            sparse_flags(offset).set_fixed_density(true);
            sparse_density(offset) = fixed_cell_density_;
          }
        }
      }
      if (!node_flag(ipos) || sparse_flags(ipos).get_has_dirichlet()) {
        if (sparse_flags(ipos).get_has_dirichlet()) {
          static bool warned = false;
          if (!warned) {
            TC_WARN("Adding force to Dirichlet nodes!");
            warned = true;
          }
        }
        return false;
      }
      force.f[0] = f[0];
      force.f[1] = f[1];
      force.f[2] = f[2];
      active_objective->forces.push_back(force);
      return true;
    } else {
      Region region(-size / Vectori(2), size - size / Vectori(2));
      real scale = 1.0_f / size.prod();
      TC_TRACE("Adding force to {} nodes", size.prod());
      bool success = true;
      for (auto &ind : region) {
        bool ret = add_force(ipos + ind.get_ipos(), f * scale, Vectori(1));
        success = success && ret;
      }
      return success;
    }
  }

  // Single cell, must be active and inside the container
  // For each node
  // Duplicate node BCs will be deleted
  // and conflict node BCs will be detected before feeding into solver
  bool add_cell_boundary(Vectori ipos,
                         const std::string &axis = "xyz",
                         Vector value = Vector(0)) {
    bool failed_bc = false;
    Region node_region = Region(Vectori(0), Vectori(2));
    auto flags = grid->flags();
    auto density = grid->density();
    TC_ASSERT(flags(ipos).get_inside_container());
    // TC_ASSERT(density(ipos) > 0);
    if (fix_cells_at_dirichlet) {
      flags(ipos).set_fixed_density(true);
      density(ipos) = 1;
    }
    for (auto n : node_region) {
      Vectori node = ipos + n.get_ipos();
      for (int i = 0; i < dim; i++) {
        if (axis.find('x' + char(i)) != std::string::npos) {
          if (add_node_boundary(node, i, value[i])) {
            flags(node).set_has_dirichlet(true);
          } else {
            failed_bc = true;
          }
        }
      }
    }
    TC_ERROR_IF(failed_bc, "Dirichlet must be added to active voxel");
    return !failed_bc;
  }

  void add_cell_boundary(Vector pos,
                         real radius,
                         std::string axis = "xyz",
                         Vector val = Vector(0)) {
    Vectori ipos = ((pos + Vector(0.5_f)) * inv_dx).template cast<int>() +
                   Vectori(padding);
    int radius_i = std::max(static_cast<int>(std::round(radius * inv_dx)), 1);
    TC_P(radius_i);
    // Enumerate cells
    bool failed_bc = false;
    int voxels_fixed = 0;
    for (auto c : Region(Vectori(-radius_i + 1), Vectori(radius_i))) {
      Vectori cell_idx = ipos + c.get_ipos();
      Vector cell_center_pos =
          (ipos.template cast<real>() + c.get_pos() - Vector(padding)) * dx -
          Vector(0.5_f);
      if (length(cell_center_pos - pos) > radius + dx) {
        continue;
      }
      if (container_bounds[0] <= cell_idx && cell_idx < container_bounds[1] &&
          grid->density()(cell_idx) > 0) {
        voxels_fixed += 1;
        add_cell_boundary(ipos + c.get_ipos(), axis);
      } else {
        failed_bc = true;
      }
    }
    if (failed_bc) {
      TC_WARN("Tried to add Dirichlet BC to 'air' nodes.");
    }
    TC_INFO("Adding Dirichlet BC to {} voxels", voxels_fixed);
  }

  bool node_has_neighbouring_cell(Vectori ipos) const {
    auto sparse_flags = grid->flags();
    bool has_neighbour = false;
    Region region(Vectori(-1), Vectori(1));
    for (auto &ind : region) {
      if (sparse_flags(ind.get_ipos() + ipos).get_inside_container()) {
        has_neighbour = true;
      }
    }
    return has_neighbour;
  }

  bool add_node_boundary(Vectori ipos, int axis, real val) {
    bool has_neighbour = node_has_neighbouring_cell(ipos);
    if (has_neighbour) {
      active_objective->boundary_condition.push_back({ipos, axis, val});
      return true;
    } else {
      return false;
    }
  }

  void populate_grid(const Config &config);

  void convert_to_wireframe() {
    TC_ERROR("");
    auto sparse_density = grid->density();
    auto sparse_flags = grid->flags();

    int counter = 0;
    int total = 0;
    int wireframe_grid_size = config.get<int>("wireframe_grid_size");
    int wireframe_thickness = config.get<int>("wireframe_thickness");
    for (auto &ind : get_cell_region()) {
      if (!sparse_flags(ind.get_ipos()).get_inside_container()) {
        continue;
      }
      total += 1;
      auto &d = sparse_density(ind.get_ipos());
      const int A = wireframe_thickness, B = wireframe_grid_size;
      if (int(ind.i % B < A) + int(ind.j % B < A) + int(ind.k % B < A) >= 2) {
        d = 1.0;
        counter += 1;
      } else {
        d = minimum_density;
      }
    }
    TC_P(counter);
    TC_P(total);
    real fraction = 1.0 * counter / total;
    TC_P(fraction);
  }

  void smooth_dc() const;

  template <typename T>
  void parallel_for_each_block(const std::unique_ptr<PageMap> &page_map,
                               const T &target) {
    parallel_for_each_block(*page_map, target);
  }

  template <typename T>
  void parallel_for_each_block(PageMap *page_map, const T &target) {
    parallel_for_each_block(*page_map, target);
  }

  template <typename T>
  void parallel_for_each_block(PageMap &page_map, const T &target) {
    const auto &blocks = page_map.Get_Blocks();
    ThreadedTaskManager::run((int)blocks.second, this->num_threads, [&](int b) {
      Vector3i base_coord;
      auto ind = ChannelMask::LinearToCoord(blocks.first[b]);
      for (int v = 0; v < dim; ++v)
        base_coord[v] = ind[v];
      target(b, blocks.first[b], base_coord);
    });
  }

  // returns: max change
  real iterate(int iter);

  void output(const std::string &file_name) {
    TC_NOT_IMPLEMENTED;
    return;
    bool verbose = grid->verbose;
    bool thin = grid->get_thin();
    grid->verbose = false;
    grid->set_thin(true);
    write_to_binary_file(*grid, file_name);
    grid->set_thin(thin);
    grid->verbose = verbose;
  }

  std::string general_action(const Config &param) override;

  bool fem_solve(
      int iter,
      bool pure_objective,
      const std::vector<fem_interface::ForceOnNode> &forces,
      const typename HexFEMSolver<dim>::BoundaryCondition &boundary_conditions,
      bool write_to_u = true);  // write to v otherwise

  void upsample() {
    TC_NOT_IMPLEMENTED
    /*
    n = n * upsample_ratio;
    current_dx = current_dx * (1.0_f / upsample_ratio);

    // Reinitialize
    seed_from_previous = true;
    std::swap(spgrid, spgrid_);
    initialize_grid();
    */
  }

  void update_page_mask() {
    /*
    auto blocks = page_map->Get_Blocks();
    page_map_->Clear();
    for (int b = 0; b < (int)blocks.second; b++) {
      auto base_offset = blocks.first[b];
      auto x = 1 << ChannelMask::block_xbits;
      auto y = 1 << ChannelMask::block_ybits;
      auto z = 1 << ChannelMask::block_zbits;
      auto c = ChannelMask::LinearToCoord(base_offset);
      for (int i = -1 + (c[0] == 0); i < 2; i++) {
        for (int j = -1 + (c[1] == 0); j < 2; j++) {
          for (int k = -1 + (c[2] == 0); k < 2; k++) {
            page_map_->Set_Page(ChannelMask::Packed_Add(
                base_offset, ChannelMask::Linear_Offset(x * i, y * j, z * k)));
          }
        }
      }
    }
    page_map_->Update_Block_Offsets();
    page_map = page_map_;
    */
  }

  void load_state(const std::string &filename) {
    read_from_binary_file(*this, filename);
  }

  void save_state(const std::string &filename) {
    write_to_binary_file(*this, filename);
  }

  void load_density_from_fem(const std::string &filename);

  void threshold_and_kill(real threshold);

  real get_penalty(int iter) {
    real target_penalty = config.get("penalty", 3.0_f);
    real progressive = config.get("progressive_penalty", 0);
    real progressive_step = config.get("progressive_penalty_step", 1);
    if (progressive == 0) {
      return target_penalty;
    } else {
      real r = min(
          (iter / progressive_step * progressive_step) * 1.0_f / progressive,
          1.0_f);
      return lerp(r, 1.0_f, target_penalty);
    }
  }

  void filter_isolated_voxels(int iter);

  void filter_isolated_blocks(int iter);

  float64 compute_objective(real threshold);

  void add_mesh_normal_force(const std::string &mesh_fn,
                             real strength,
                             Vector3 center,
                             real falloff,
                             real max_distance,
                             Vector3 override,
                             bool no_z);
  void make_shell_mesh(const std::string &mesh_fn, real max_distance);

  void clear_gradients() {
    auto sparse_der = grid->dc0();
    parallel_for_each_block(*fat_page_map, [&](int b, uint64 block_offset,
                                               const Vector3i &base_coord) {
      Region region(Vectori(0), block_size);
      for (auto ind : region) {
        sparse_der(base_coord + ind.get_ipos()) = 0;
      }
    });
  }

  void threshold_density();

  real threshold_density_trail(real threshold, bool apply);
};

using Opt = SPGridTopologyOptimization3D;

TC_NAMESPACE_END
