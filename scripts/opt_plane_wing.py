from topo_opt import TopoOpt
import taichi as tc

import crash_report
crash_report.enable()

version = 10
narrow_band = True
volume_fraction = 0.2
# Initialize
n = 2000
opt = TopoOpt(res=(n, n, n), version=version, volume_fraction=volume_fraction,
              grid_update_start=2 if narrow_band else 1000000,
              progressive_vol_frac=0, fixed_cell_density=0.21, cg_max_iterations=100)


s = 1
tex = tc.Texture(
    'mesh',
    translate=(0.5, 0.5, 0.5),
    scale=(s, s, s),
    adaptive=False,
    filename='projects/topo_opt/data/wing_2.obj')

s *= 0.95
tex_shell = tc.Texture(
    'mesh',
    translate=(0.5, 0.5, 0.5),
    scale=(s, s, s),
    adaptive=False,
    filename='projects/topo_opt/data/wing_2.obj')

opt.populate_grid(domain_type='texture', tex_id=tex.id)
opt.general_action(action='make_shell', tex_id=tex_shell.id)
opt.general_action(action='voxel_connectivity_filtering')

opt.general_action(action='add_box_dirichlet_bc', axis_to_fix='xyz', bound0=(-0.48, -0.48, -0.48), bound1=(-0.4, 0.48, 0.48))
opt.general_action(action='add_mesh_normal_force', mesh_fn='projects/topo_opt/data/wing_2.obj', magnitude=-1, center=(0.21, 0.26, 0), falloff=3000, maximum_distance=0.005)

# Optimize
opt.run()

