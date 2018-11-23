from topo_opt import TopoOpt
import taichi as tc

version = 8
wireframe = False
narrow_band = False
plane_force = True
volume_fraction=0.08

use_mirror = True

# Initialize
n = 1800
#tc.core.set_core_trigger_gdb_when_crash(True);
opt = TopoOpt(res=(n, n, n), version=version, volume_fraction=volume_fraction,
              grid_update_start=5 if narrow_band else 1000000,
              progressive_vol_frac=5, cg_tolerance=1e-3,
              minimum_stiffness=0, minimum_density=1e-2,
              fix_cells_at_dirichlet=False, fix_cells_near_force=True, connectivity_filtering=True, adaptive_min_fraction=False, verbose_snapshot=False)

x, y, z = 0.1, 0.1, 0.4
if use_mirror:
  mirror = 'xz'
else:
  mirror = ''

opt.populate_grid(domain_type='box', size=(x, y, z), mirror=mirror)

# Set up BCs
opt.add_plane_dirichlet_bc(axis_to_fix="xyz", axis_to_search=2, extreme=-1)
if not use_mirror:
  opt.add_plane_dirichlet_bc(axis_to_fix="xyz", axis_to_search=2, extreme=1)

eps = 0.02

if plane_force:
  s = z - eps
  opt.add_plane_load(force=(0, -1e4, 0), axis=1, extreme=-1, bound1=(-s, -s, -s), bound2=(s, s, s))
else:
  opt.add_load(center=(0.0, -y + 0.01, 0.0), force=(0, -1e6, 0))

# Optimize
opt.run()
