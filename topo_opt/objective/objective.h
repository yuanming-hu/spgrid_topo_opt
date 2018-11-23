#include <taichi/common/util.h>
#include "../fem_interface.h"

TC_NAMESPACE_BEGIN

class Objective : public Unit {
  static constexpr int dim = 3;

 public:
  using Vectori = TVector<int, 3>;
  using Vector = TVector<real, 3>;

  struct Target {
    Vectori node_coord;
    Vector deformation;
    real weight;
  };

  std::vector<Target> targets;
  Dict config;
  using ForceOnNode = fem_interface::ForceOnNode;
  std::vector<ForceOnNode> forces;
  real weight;
  // Dirichlet BCs
  typename HexFEMSolver<dim>::BoundaryCondition boundary_condition;

  TC_IO_DEF(config);

  virtual float64 compute(const Dict &param) {
    TC_NOT_IMPLEMENTED;
    return 0.0_f64;
  }

  Objective() : Unit() {
    this->weight = 1.0_f;
  }

  Objective(const Dict &config) : Unit() {
    this->config = config;
    this->weight = 1.0_f;
  }

  virtual std::string get_name() const override {
    return "topOpt_objective";
  }

  TC_FORCE_INLINE void add_force(const ForceOnNode &f) {
    forces.push_back(f);
  }
};

TC_INTERFACE(Objective);

TC_NAMESPACE_END
