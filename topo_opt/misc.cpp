#include <taichi/util.h>
#include <taichi/math.h>
#include <taichi/io/io.h>
#include <taichi/system/threading.h>
#include <taichi/visualization/particle_visualization.h>
#include <taichi/common/task.h>
#include <taichi/system/profiler.h>
#include <taichi/io/ply_writer.h>
#include "fem_interface.h"
#include "hex_fem.h"

TC_NAMESPACE_BEGIN

auto fem_solve = [](const std::vector<std::string> &parameters) {
  using namespace fem_interface;

  TC_ASSERT(parameters.size() >= 1);
  std::string file_name = parameters[0];

  FEMInterface interface;
  FEMInputs &param = interface.param;
  BinaryInputSerializer reader;
  reader.initialize(file_name);
  reader(param);
  interface.preserve_output(param.density.blocks.size());
  interface.param.set_solver_state_ptr(nullptr);

  for (int i = 1; i < (int)parameters.size(); i++) {
    auto p = parameters[i];
    auto s = p.find("=");
    if (s != std::string::npos) {
      auto field = std::string(p.begin(), p.begin() + s);
      auto value = std::string(p.begin() + s + 1, p.end());
      TC_TRACE("Setting {} = {}", field, value);
      if (field == "cg_max_iterations") {
        param.krylov.max_iterations = std::stoi(value);
      } else if (field == "minimum_stiffness") {
        param.minimum_stiffness() = std::stod(value);
      } else {
        TC_ERROR("Unknown field");
      }
    }
  }

  {
    Time::Timer _("fem_solve");
    interface.run();
  }
};

TC_REGISTER_TASK(fem_solve);

auto convert_fem_solve = [](const std::vector<std::string> &parameters) {
  using namespace fem_interface;

  TC_ASSERT(parameters.size() >= 1);
  std::string file_name = parameters[0];

  FEMInterface interface;
  FEMInputs &param = interface.param;
  BinaryInputSerializer reader;
  reader.initialize(file_name);
  reader(param);

  double min_density = 1, max_density = 0;
  bool has_nan = false;
  bool has_inf = false;
  for (auto &block : param.density.blocks) {
    for (auto d : block.data) {
      if (d != 0) {
        min_density = std::min(min_density, d);
        max_density = std::max(max_density, d);
        has_nan = std::isnan(d) || has_nan;
        has_inf = std::isinf(d) || has_inf;
      }
    }
  }
  TC_P(min_density);
  TC_P(max_density);
  if (has_nan) {
    TC_WARN("nan detected");
  }
  if (has_inf) {
    TC_WARN("inf detected");
  }
  if (!has_nan && !has_inf) {
    TC_INFO("Good. No nan nor inf detected.");
  }

  if (parameters.size() <= 1) {
    param.density.blocks.clear();
  }

  TextSerializer ser2;
  ser2("FEM Solve Parameters", param);
  ser2.write_to_file("human_readable.txt");
};

TC_REGISTER_TASK(convert_fem_solve);

auto test_exception = []() {
  try {
    throw std::runtime_error("something happened");
  } catch (const std::exception &e) {
    fmt::print(e.what());
  }
};

TC_REGISTER_TASK(test_exception);

class FilterDensity : public Task {
 public:
  bool active[4048][4024][2024];
  uint32 color[4048][4024][2024];

  Vector3i res;

  TC_FORCE_INLINE bool get(int x, int y, int z) const {
    if (x < 0 || y < 0 || z < 0) {
      return false;
    }
    if (x >= res.x || y >= res.y || z >= res.z) {
      return false;
    }
    return active[x][y][z];
  }

  TC_FORCE_INLINE bool get(Vector3i i) const {
    return get(i.x, i.y, i.z);
  }

  TC_FORCE_INLINE int get_color(int x, int y, int z) const {
    if (x < 0 || y < 0 || z < 0) {
      return false;
    }
    if (x >= res.x || y >= res.y || z >= res.z) {
      return false;
    }
    return color[x][y][z];
  }

  TC_FORCE_INLINE int get_color(Vector3i i) const {
    return get_color(i.x, i.y, i.z);
  }

  const Vector3i offsets[6]{
      Vector3i(0, 0, 1),  Vector3i(0, 0, -1), Vector3i(0, 1, 0),
      Vector3i(0, -1, 0), Vector3i(1, 0, 0),  Vector3i(-1, 0, 0),
  };

  void filter() {  // BFS
    // We only consider 6 connectivity instead of 26
    // this should operate on thin_page_map
    std::vector<int64> component_sizes;

    // color == -1 -> unvisited
    memset(color, -1, sizeof(color));

    std::deque<Vector3i> Q;
    int64 total_size = 0;

    auto BFS = [&](Vector3i u) {
      int64 count = 0;
      auto visit = [&](Vector3i neighbour) {
        if (get(neighbour) && get_color(neighbour) == -1) {
          Q.push_back(neighbour);
          color[neighbour.x][neighbour.y][neighbour.z] =
              (int)component_sizes.size();
          count += 1;
        }
      };

      Q.resize(0);

      visit(u);
      while (!Q.empty()) {
        auto v = Q.front();
        Q.pop_front();
        for (auto o : offsets) {
          visit(v + o);
        }
      }
      component_sizes.push_back(count);
      total_size += count;
    };

    for (auto &ind : Region3D(Vector3i(0), res)) {
      Vector3i i = ind.get_ipos();
      if (get_color(i) == -1 && get(i)) {
        TC_INFO("  Visiting connected component {}", component_sizes.size());
        BFS(i);
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
              "There are more than two connected components with fraction > "
              "30%. "
              "Something may be wrong.");
        }
      }
    }

    TC_ERROR_IF(
        larger_than_30_count == 0,
        "All compoments have fraction < 30%. How can this be so fragmented?");

    // larger_than_30_count must be 1 now.

    int64 left = 0;

    // Filtering
    for (auto &ind : Region3D(Vector3i(0), res)) {
      Vector3i i = ind.get_ipos();
      if (get(i)) {
        if (get_color(i) != larger_than_30_index) {
          active[i.x][i.y][i.z] = false;
        } else {
          left += 1;
        }
      }
    }
    TC_WARN(
        "{} ({:.2f}%) voxels remained after connectivity filtering. Filtered "
        "voxels are eliminated permenantly (marked as outside container)",
        left, 100.0_f * left / total_size);
  }

  std::string run(const std::vector<std::string> &parameters) override {
    using namespace fem_interface;

    memset(active, 0, sizeof(active));

    TC_ASSERT(parameters.size() >= 2);
    std::string file_name = parameters[0];

    real threshold = std::atof(parameters[1].c_str());

    FEMInterface interface;
    FEMInputs &param = interface.param;
    BinaryInputSerializer reader;
    reader.initialize(file_name);
    reader(param);
    interface.preserve_output(param.density.blocks.size());
    interface.param.set_solver_state_ptr(nullptr);

    // Setting up density field
    Vector3i min_coord(1000000000);
    Vector3i max_coord(0);
    using scalar_block_size = FEMInputs::ScalarGrid::block_size;
    auto block_size = Vector3i(scalar_block_size::x, scalar_block_size::y,
                               scalar_block_size::z);
    for (auto block : param.density.blocks) {
      for (auto ind : RegionND<3>(Vector3i(0), block_size)) {
        for (int i = 0; i < 3; i++) {
          auto d = block.get(ind.i, ind.j, ind.k);
          if (d > threshold) {
            min_coord[i] = std::min(
                min_coord[i], block.base_coordinates[i] + ind.get_ipos()[i]);
            max_coord[i] =
                std::max(max_coord[i],
                         block.base_coordinates[i] + ind.get_ipos()[i] + 1);
            active[ind.i + block.base_coordinates[0]]
                  [ind.j + block.base_coordinates[1]]
                  [ind.k + block.base_coordinates[2]] = true;
          }
        }
      }
    }
    res = max_coord;
    TC_P(res);

    filter();

    for (auto &block : param.density.blocks) {
      for (auto ind : RegionND<3>(Vector3i(0), block_size)) {
        for (int i = 0; i < 3; i++) {
          auto &d = block.get(ind.i, ind.j, ind.k);
          if (!active[ind.i + block.base_coordinates[0]]
                     [ind.j + block.base_coordinates[1]]
                     [ind.k + block.base_coordinates[2]] ||
              d < threshold) {
            d = 0;
          }
        }
      }
    }
    TC_INFO("Writing...");
    write_to_binary_file(param, "filtered.tcb");
    return "";
  }
};

TC_IMPLEMENTATION(Task, FilterDensity, "filter_density");

auto plane_wing_mesh = []() {
  std::FILE *f = fopen("vertices.txt", "r");
  int n;
  fscanf(f, "%d", &n);
  std::vector<real> X, Y;
  for (int i = 0; i < n; i++) {
    real x;
    fscanf(f, "%lf", &x);
    X.push_back(x);
  }
  for (int i = 0; i < n; i++) {
    real y;
    fscanf(f, "%lf", &y);
    Y.push_back(y);
  }
  std::vector<Vector2> p;
  Vector2 center(0);
  for (int i = 0; i < n; i++) {
    p.push_back(Vector2(X[i], Y[i]));
  }
  p.push_back(p[0]);
  for (int i = 0; i < n; i++) {
    center += p[i];
  }
  PLYWriter writer("/tmp/wing.ply");
  Vector2 o1(0.34, -0.4);
  Vector2 o2(0.4, 0.4);
  int N = 20;
  auto surface = [&](Vector2 a, Vector2 b) {
    for (int j = 0; j < N; j++) {
      auto p = lerp(1.0_f * j / N, o1, o2);
      auto q = lerp(1.0_f * (j + 1) / N, o1, o2);
      writer.add_face({Vector3(a * p.x, p.y), Vector3(b * p.x, p.y),
                       Vector3(b * q.x, q.y), Vector3(a * q.x, q.y)});
    }
  };
  for (int i = 0; i < n; i++) {
    surface(p[i], p[i + 1]);
  }
  center *= 1.0_f / n;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < N; j++) {
      auto add_face = [&](Vector2 o) {
        auto p1 = lerp(1.0_f * j / N, center, p[i]);
        auto q1 = lerp(1.0_f * j / N, center, p[i + 1]);
        auto p2 = lerp(1.0_f * (j + 1) / N, center, p[i]);
        auto q2 = lerp(1.0_f * (j + 1) / N, center, p[i + 1]);
        writer.add_face({Vector3(p1 * o.x, o.y), Vector3(q1 * o.x, o.y),
                         Vector3(q2 * o.x, o.y), Vector3(p2 * o.x, o.y)});
      };
      add_face(o1);
      add_face(o2);
    }
  }
  fclose(f);
};

TC_REGISTER_TASK(plane_wing_mesh);

TC_NAMESPACE_END
