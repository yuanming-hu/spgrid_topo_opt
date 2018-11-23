from topo_opt import TopoOpt

# Change log
# v1 This is designed to match the Mitchell Sphere in the TopOpt-Petsc paper.
# v2 Use four loads. Does it make a difference?
# Original name: opt_michell_sphere.py

# Initialize
version = 1
n = 256
opt = TopoOpt(res=(n, n, n), volume_fraction=0.05, mg_level=3, version=version, grid_update_start=10,
              fix_cells_at_dirichelt=False, fix_cells_near_force=True, check_log_file=False, cg_max_iterations=100,
              progressive_vol_frac=50, cg_tolerance=1e-6)

outer_radius = 0.5
inner_radius = 0.48

opt.populate_grid(domain_type='sphere', inner_radius=inner_radius, outer_radius=outer_radius)

delta = outer_radius - inner_radius
center_radius = (inner_radius + outer_radius) * 0.5

# The radius of the Dirichlet region is 10% of the whole sphere
opt.add_dirichlet_bc(center=(-center_radius, 0.0, 0.0), radius=delta * 12)

d = delta
opt.add_load(center=(outer_radius - delta * 0.5,  d,  0), force=(0,  0,  8), size=0.04)
opt.add_load(center=(outer_radius - delta * 0.5, -d,  0), force=(0,  0, -8), size=0.04)
opt.add_load(center=(outer_radius - delta * 0.5,  0,  d), force=(0, -8,  0), size=0.04)
opt.add_load(center=(outer_radius - delta * 0.5,  0, -d), force=(0,  8,  0), size=0.04)

# Optimize
opt.run()
