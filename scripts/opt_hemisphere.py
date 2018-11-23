from topo_opt import TopoOpt

# Initialize
dense_container = False
version = 6
n = 4096
opt = TopoOpt(res=(n, n, n), volume_fraction=0.04, mg_level=7, version=version, cg_tolerance=1e-2, grid_update_start=10, fix_cells_near_force=True, fix_cells_at_dirichlet=True)

outer_radius = 0.50
inner_radius = 0.48

opt.populate_grid(domain_type='sphere', inner_radius=inner_radius, outer_radius=outer_radius, lower_only=True, mirror='xz')

shell_thickness = outer_radius - inner_radius
shell_center = (outer_radius + inner_radius) / 2.0
dirichlet_size = shell_thickness * 4.0
opt.add_dirichlet_bc(center=(0, -shell_center, 0.0), radius = dirichlet_size)

opt.add_plane_load(force=(0, -1, 0), axis=1, extreme=1)

# Optimize
opt.run()
