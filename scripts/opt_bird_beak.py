from topo_opt import TopoOpt
import taichi as tc

# Change log
# v1 Initial commit
# v2 correct BC and plane wing force

import crash_report
crash_report.enable()

version = 1
narrow_band = True
volume_fraction = 0.1 # Initialize
n = 3000
opt = TopoOpt(res=(n, n, n), version=version, volume_fraction=volume_fraction,
              grid_update_start=5 if narrow_band else 1000000,
              fix_cells_near_force=True, fixed_cell_density=0.1)

s = 1.0
tex = tc.Texture(
  'mesh',
  translate=(0.5, 0.5, 0.5),
  scale=(s, s, s),
  adaptive=False,
  filename='projects/topo_opt/data/beak.obj')

opt.populate_grid(domain_type='texture', tex_id=tex.id, mirror='z')
opt.general_action(action='voxel_connectivity_filtering')

opt.general_action(action='add_box_dirichlet_bc', axis_to_fix='xyz', bound0=(0.4, -0.48, -0.48), bound1=(0.48, 0.48, 0.48))
# try plane force
opt.general_action(action='add_mesh_normal_force', mesh_fn='projects/topo_opt/data/beak.obj', magnitude=-1, center=(0.5, 0, 0), falloff=1000, maximum_distance=0.015, override=(0, 0, 0))

# Optimize
opt.run()
