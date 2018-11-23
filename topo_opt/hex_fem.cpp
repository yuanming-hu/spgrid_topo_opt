#include <taichi/common/interface.h>
#include <pybind11/pybind11.h>
#include "hex_fem.h"
#include "../solver/CPU_Kernels/Stiffness_Matrices.h"

TC_NAMESPACE_BEGIN

template <>
std::vector<float64> get_Ke<2>(Material material) {
  static constexpr int indices[8][8] = {
      {0, 1, 2, 3, 4, 5, 6, 7}, {1, 0, 7, 6, 5, 4, 3, 2},
      {2, 7, 0, 5, 6, 3, 4, 1}, {3, 6, 5, 0, 7, 2, 1, 4},
      {4, 5, 6, 7, 0, 1, 2, 3}, {5, 4, 3, 2, 1, 0, 7, 6},
      {6, 3, 4, 1, 2, 7, 0, 5}, {7, 2, 1, 4, 3, 6, 5, 0}};

  float64 E = material.E;
  float64 nu = material.nu;

  constexpr int dim = 2;

  std::vector<float64> ke_entries = {1.0f / 2.0f - nu / 6.0f,
                                     1 / 8.0f + nu / 8.0f,
                                     -1 / 4.0f - nu / 12.0f,
                                     -1 / 8.0f + 3 * nu / 8.0f,
                                     -1 / 4.0f + nu / 12.0f,
                                     -1 / 8.0f - nu / 8.0f,
                                     nu / 6.0f,
                                     1 / 8.0f - 3.0f * nu / 8.0f};

  for (int i = 0; i < ke_size<2>(); i++) {
    ke_entries[i] *= E / (1.0f - pow<2>(nu));
  }

  int map[] = {0, 3, 1, 2};

  auto M = [&](int i, int p) { return map[i] * dim + p; };

  std::vector<float64> Ke;
  for (int i = 0; i < pow<dim>(2); i++) {
    for (int p = 0; p < dim; p++) {
      for (int j = 0; j < pow<dim>(2); j++) {
        for (int q = 0; q < dim; q++) {
          Ke.push_back(ke_entries[indices[M(i, p)][M(j, q)]]);
        }
      }
    }
  }
  return Ke;
}

template <>
std::vector<float64> get_Ke<3>(Material material) {
  constexpr int dim = 3;

  constexpr int N = ke_size<dim>();

  auto get = [&](int i, int j, int k, int l) {
    TC_ASSERT(0 <= i && i < 8);
    TC_ASSERT(0 <= j && j < 8);
    TC_ASSERT(0 <= k && k < 3);
    TC_ASSERT(0 <= l && l < 3);
    return K_mu<real>[i][j][k][l] * material.mu +
           K_la<real>[i][j][k][l] * material.lambda;
  };

  std::vector<float64> Ke;
  for (int i = 0; i < pow<dim>(2); i++) {
    for (int p = 0; p < dim; p++) {
      for (int j = 0; j < pow<dim>(2); j++) {
        for (int q = 0; q < dim; q++) {
          Ke.push_back(get(i, j, p, q));
        }
      }
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      assert_info(Ke[i * N + j] == Ke[j * N + i], "Asymmetric!");
    }
  }
  return Ke;
}

TC_NAMESPACE_END
