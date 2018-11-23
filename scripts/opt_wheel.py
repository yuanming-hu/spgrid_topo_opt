from topo_opt import TopoOpt
import taichi as tc

version = 1
narrow_band = True
volume_fraction = 0.01
# Initialize
n = 1024
opt = TopoOpt(res=(n, n, n), version=version, volume_fraction=volume_fraction, grid_update_start=5 if narrow_band else 1000000,
              cg_tolerance=1e-2, cg_max_iterations=10, progressive_volume_fraction=5, fix_cells_near_force=True, fixed_cell_density=0.2,
              mg_level=5)

#opt.populate_grid(domain_type='cylinder', radius=0.25, thickness=0.5, height=0.05)
opt.populate_grid(domain_type='wheel', radius=0.25, thickness=0.5, height=0.05, mirror='')

# Set up BCs
opt.add_dirichlet_bc((0, 0, -0.1), radius=0.05, axis='xyz', value=(0, 0, 0))
opt.add_dirichlet_bc((0, 0, 0.1), radius=0.05, axis='xyz', value=(0, 0, 0))
opt.set_up_wheel()

opt.add_objective(name="minimal_compliance", weight=0.1)
opt.add_dirichlet_bc((0, 0, -0.1), radius=0.05, axis='xyz', value=(0, 0, 0))
opt.add_dirichlet_bc((0, 0, 0.1), radius=0.05, axis='xyz', value=(0, 0, 0))
opt.set_up_wheel_shear()

# Optimize
opt.run()

