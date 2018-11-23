#pragma once

#include <taichi/util.h>
#include <unordered_map>

TC_NAMESPACE_BEGIN

template <int dim>
constexpr int ke_size() {
  return pow<dim>(2) * dim;
}

class Material {
 public:
  float64 E;       // Young's modulus
  float64 nu;      // Poisson's ratio
  float64 lambda;  // 1st Lame's param
  float64 mu;      // 2nd Lame's param

  Material(float64 E, float64 nu) : E(E), nu(nu) {
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    mu = E / (2 * (1 + nu));
  }

  void set_lame(float64 lambda, float64 mu) {
    this->mu = mu;
    this->lambda = lambda;
    this->E = mu * (3 * lambda + 2 * mu) / (lambda + mu);
    this->nu = 0.5_f * lambda / (lambda + mu);
  }

  Material() {
    E = nu = mu = lambda = 0.0_f;
  }

  TC_IO_DECL {
    TC_IO(E);
    TC_IO(nu);
    TC_IO(lambda);
    TC_IO(mu);
  };
};

template <int dim>
std::vector<float64> get_Ke(Material material);

// Note: this is a dummy solver. See the 'solver' folder for the
// high-performance solver.
template <int dim>
class HexFEMSolver : public Unit {
 public:
  using Vector = VectorND<dim, real>;
  using Vectori = VectorND<dim, int>;
  template <typename T>
  using Array = ArrayND<dim, T>;
  using Region = RegionND<dim>;
  using Index = IndexND<dim>;

  struct BoundaryConditionNode {
    Vectori node;
    int axis;
    real val;
  };

  using BoundaryCondition = std::vector<BoundaryConditionNode>;

  Material material;
  std::vector<float64> ke;

  void initialize(const Config &config) override {
    material = *config.get_ptr<Material>("material");

    ke = get_Ke<dim>(material);

    assert_info(ke.size() == pow<2>(ke_size<dim>()), "Incorrect Ke size");
    for (int i = 0; i < ke_size<dim>(); i++) {
      for (int j = 0; j < ke_size<dim>(); j++) {
        assert_info(Ke(i, j) == Ke(j, i), "Asymmetric!");
      }
    }
  }

  float64 Ke(int row, int column) const {
    return ke[row * ke_size<dim>() + column];
  }

  constexpr int get_index(Index ind, int d) const {
    int ret = 0;
    for (int i = 0; i < dim; i++) {
      ret += (1 << i) * (ind[dim - 1 - i]);
    }
    return ret * dim + d;
  }
};

using HexFEMSolver2D = HexFEMSolver<2>;
using HexFEMSolver3D = HexFEMSolver<3>;

TC_NAMESPACE_END
