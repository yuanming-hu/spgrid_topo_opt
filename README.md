<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/topopt/bird-beak.gif">

# Narrow-Band Topology Optimization on SPGrid
## **[[Paper]](http://taichi.graphics/wp-content/uploads/2018/10/narrowband_topopt.pdf) [[Video]](https://www.youtube.com/watch?v=H2OxHdQEQCQ)**

**Narrow-Band Topology Optimization on a Sparsely Populated Grid**, ACM Transactions on Graphics (SIGGRAPH Asia 2018).

By
[Haixiang Liu (University of Wisconsin-Madison)](http://pages.cs.wisc.edu/~cslhxac/),

[Yuanming Hu (MIT CSAIL)](http://taichi.graphics/me/),

[Bo Zhu (Dartmouth College)](http://www.dartmouth.edu/~boolzhu/),

[Wojciech Matusik (MIT CSAIL)](http://people.csail.mit.edu/wojciech/),

[Eftychios Sifakis (University of Wisconsin-Madison)](http://pages.cs.wisc.edu/~sifakis/).



<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/topopt/bridge-density.gif" width="800px"><img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/topopt/bridge-zoom.gif"  width="800px">


## Installation (Tested on Ubuntu 16.04/18.04/Arch Linux. Windows/OS X not supported.)
 - Install [`taichi (legacy branch)`](https://taichi.readthedocs.io/en/latest/installation.html#ubuntu-arch-linux-and-mac-os-x) first and put this in the `projects` folder.
 - Build the FEM solver: `cd solver && make` (Note: this needs Intel `icc` and `mkl`. Please install if you don't have it.)
 - Set the following environment variables according to your machine.
 For example,
 ```
 export TC_MKL_PATH=/opt/intel/compilers_and_libraries_2018.0.128/linux/mkl/lib/intel64_lin/
 export CUDA_ARCH=61 # or 0 if there is no CUDA
 export TC_USE_DOUBLE=1 #
 ```
`TC_MKL_PATH` is for `libmkl_rt.so`.
 
 - `ti build` in shell and wait for the build to finish.
 - Pick a script under the `scripts` folder and run with `python3`. Some scripts have extremely high resolution (see Table 2 in our paper). If you run out of memory, you may want to use a smaller `n`.

## TopoOpt.__init__
 - `res`: resolution say `(64, 64, 64)`
 - `volume_fraction`: default=`0.1`
 - `use_youngs`: Use Young's modulus and Poisson's ratio for material description
   * `E`: `1e6`
   * `nu`: `0.3`
 - `penalty`: SIMP penalty, default=`3`
 - `minimum_density`: minimum_density in topology optimization, default=`0`
 - `minimum_stiffness`: default=`1e-9`
 - `grid_update_start`: default=`5`
 - `fraction_to_keep`: default=`1.0`, keep only this fraction of SPGrid blocks during optimization. The blocks with highest density sum will be selected.
 - `wireframe`: default=`False`
 - `wireframe_grid_size`: default=`32`
 - `wireframe_thickness`: default=`4`
 - `fix_cells_near_force`: default=`False`
 - `fix_cells_at_dirichlet`: default=`False`
 - `progressive_vol_frac`: default=`0`
 (Solver parameters:)
 - `cg_tolerance`: default `1e-4`
 - `active_threshold`: default `1e-6`
 - `cg_max_iterations`: default=`50`
 - `defect_correction_iter`: default=`10`
 - `verbose_snapshot`: default=`False`
 - `defect_correction_cg_iter`: default=`3`
 - `boundary_smoothing_iters`: default=`3`
 - `smoothing_iters`: default=`1` (interior smoothing iterations)
 - `mg_bottom_size`: default=`64`
 - `mg_level`: default=adaptively choose a value s.t. the coarsest level has resolution <= `mg_bottom_size`
 - `explicit_mg_level`: default=`1`
 - `restart_iterations`: default=`0`
 - `print_residuals`: default=`False`
 - `jacobi_damping`: default=`0.4` (not useful when using GS. Do we need a parameter to switch between GS and jacobi?)
 - `connectivity_filtering`: default=`True`
 - `objective_threshold`: default=`0.5`
 - `exclude_fixed_cells`: default=`True`
 - `fixed_cell_density`: default=`1`
 - `step_limit`: default=`0.2`
 - `exclude_minimal_compliance`: default=`false`

## TopoOpt.populate_grid
 - `domain_type`: a container used to populate the SPGrid.
  * box: ...
  * cylinder: ...
  * sphere:
    - inner_radius=0.4
    - outer_radius=0.5
    - upper_only=False
    
 - `uniform_bc`: Set Dirichlet on all nodes? A string of axis (e.g. `'xy'`). default = `""`

## Set Boundary Conditions
 - `add_dirichlet_bc(center, radius=0.03, axis='xyz', value=(0, 0, 0))`
 - `add_load(center, force)`
 - `add_plane_load(force, axis_to_search=0/1/2, extreme=+1/-1, bound1=(-0.5, -0.5, -0.5), bound2=(0.5, 0.5, 0.5))`
 - `add_plane_dirichlet_bc(axis_to_fix, axis_to_search, extreme)`

## Utilities
 - Solve a single `tcb` (e.g. `00002.tcb`)
 ```
 ti run fem_solve 00002.tcb
 ```
 - Convert binary `tcb` to human-readable formats (for inspecting boundary conditions, solver parameters e.t.c.)
 ```
 # Without density field
 ti run convert_fem_solve 00002.tcb
 # With density field (can be huge)
 ti run convert_fem_solve 00002.tcb --with-density
 ```

## Visualization
```
ti vd [mode] [file/folder name]
```
`mode` can be `fem` (default, contains BCs, solver parameters and density field only) or `snapshots` (basically everything).
`vd` stands for `visualize_density`.

This can be called directly at the output folder (say, `bridge_v6_r1024`), or the `fem`, or `snapshots` folder (`bridge_v6_r1024/snapshots`), or a single tcb file (`bridge_v6_r1024/fem/00001.tcb`).

Use `H` and `L` to switch between frames.
Use `J` and `K` to change threshold in voxel rendering.
Transparent volume rendering is also supported, press `T` (for a while, like 1 second) and `R` to switch.

From snapshots (generated with `verbose_snapshot=True`), press `C` to switch channels between `density`, `sensitivity`, `smoothed_sensitivity`, and `displacement`.


For symmetric domain, press `1`, `2`, `3` to mirror `x`, `y`, `z` axes.


# Bibtex
Please cite our [paper](http://taichi.graphics/wp-content/uploads/2018/10/narrowband_topopt.pdf) if you use this code for your research: 
```
@article{liu2018narrow,
  title={Narrow-Band Topology Optimization on a Sparsely Populated Grid},
  author={Liu, Haixiang and Hu, Yuanming and Zhu, Bo and Matusik, Wojciech and Sifakis, Eftychios},
  journal={ACM Transactions on Graphics (TOG)},
  volume={37},
  number={6},
  year={2018},
  publisher={ACM}
}
```

