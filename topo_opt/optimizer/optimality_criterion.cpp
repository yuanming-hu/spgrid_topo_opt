#include "../spgrid_topo_opt.h"

TC_NAMESPACE_BEGIN

class OptimalityCriterion : public Optimizer {
 public:
  using Base = Optimizer;

  TC_IO_DECL {
    TC_IO(config);
  }

  OptimalityCriterion() : Optimizer() {
  }

  OptimalityCriterion(const Dict &config) : Optimizer(config) {
  }

  virtual real optimize(const Dict &param) override {
    int iter = param.get<int>("iter");
    auto &opt = config.get<SPGridTopologyOptimization3D &>("opt");
    auto &grid = opt.grid;
    real step_limit = opt.config.get("step_limit", 0.2_f);

    auto sparse_density = grid->density();
    auto sparse_flags = grid->flags();
    auto sparse_dc = grid->dc0();

    auto blocks = opt.fat_page_map->Get_Blocks();

    bool exclude_fixed_cells = opt.config.get("exclude_fixed_cells", true);

    // TODO: when adding force, this value should be used instead of 1
    real fixed_cell_density = opt.config.get("fixed_cell_density", 1.0_f);

    int64 fixed_density_voxels = 0;

    real neg_dc_min = 1e30;
    real neg_dc_max = -1e30;
    bool smart_oc = opt.config.get<bool>("smart_oc", false);
    if (smart_oc) {
      for (int b = 0; b < blocks.second; b++) {
        auto offset = blocks.first[b];
        for (unsigned int e = 0; e < Opt::ChannelMask::elements_per_block;
             ++e, offset += sizeof(real)) {
          // This works because float32 and int32 has the same size
          if (sparse_flags(offset).get_inside_container()) {
            neg_dc_min = std::min(neg_dc_min, -sparse_dc(offset));
            neg_dc_max = std::max(neg_dc_max, -sparse_dc(offset));
          }
        }
      }
    }

    if (exclude_fixed_cells) {
      fixed_density_voxels = tbb::parallel_reduce(
          tbb::blocked_range<int>(0, blocks.second), 0.0_f,
          [&](const tbb::blocked_range<int> &r, int64 sum) {
            for (int b = r.begin(); b < r.end(); b++) {
              auto offset = blocks.first[b];
              for (unsigned int e = 0; e < Opt::ChannelMask::elements_per_block;
                   ++e, offset += sizeof(real)) {
                // This works because float32 and int32 has the same size
                if (sparse_flags(offset).get_inside_container()) {
                  sum += int(sparse_flags(offset).get_fixed_density());
                }
              }
            }
            return sum;
          },
          std::plus<int64>());
      TC_P(fixed_density_voxels);
    }

    auto trail = [&](real mid, bool apply) {
      std::mutex mut;
      real change = 0.0_f;
      real sum = tbb::parallel_reduce(
          tbb::blocked_range<int>(0, blocks.second), 0.0_f,
          [&](const tbb::blocked_range<int> &r, real sum) {
            real max_change = 0;
            for (int b = r.begin(); b < r.end(); b++) {
              auto offset = blocks.first[b];
              for (unsigned int e = 0; e < Opt::ChannelMask::elements_per_block;
                   ++e, offset += sizeof(real)) {
                if (sparse_flags(offset).get_inside_container()) {
                  real old_density = sparse_density(offset);

                  float64 new_density = clamp(
                      old_density *
                          std::sqrt(
                              std::max(0.0_f64, (-sparse_dc(offset) -
                                                 std::min(0.0_f, neg_dc_min))) /
                              float64(mid)),
                      old_density - step_limit, old_density + step_limit);
                  new_density =
                      clamp(new_density, opt.minimum_density, 1.0_f64);

                  if (sparse_flags(offset).get_fixed_density()) {
                    new_density = fixed_cell_density;
                  }

                  TC_ASSERT(new_density == new_density);
                  if (!(exclude_fixed_cells &&
                        sparse_flags(offset).get_fixed_density())) {
                    sum += new_density;
                  }

                  if (apply) {
                    sparse_density(offset) = new_density;
                  }

                  max_change =
                      std::max(max_change, std::abs(new_density - old_density));
                }
              }
            }
            mut.lock();
            change = std::max(change, max_change);
            mut.unlock();
            return sum;
          },
          std::plus<real>());
      return std::make_pair(sum, change);
    };

    float64 mid = 0;
    float64 lower = 1e-20_f, upper = 1e20_f64;
    real sum = 0;
    while (lower * (1 + 1e-7_f) < upper) {
      mid = 0.5 * (lower + upper);
      sum = trail(mid, false).first;
      TC_ASSERT(sum == sum);
      if (sum > opt.get_volume_fraction(iter) *
                    (opt.total_container_voxels - fixed_density_voxels)) {
        lower = mid;
      } else {
        upper = mid;
      }
    }
    TC_WARN("OC Final Volume: {}, Lagrangian Multiplier: {}", sum, mid);
    TC_WARN(" fixed_density_cells: {}", fixed_density_voxels);
    auto final_ret = trail(mid, true);
    return final_ret.second;
  }

  TC_NAME("oc");
};

TC_IMPLEMENTATION_NEW(Optimizer, OptimalityCriterion);

TC_NAMESPACE_END
