from topo_opt import TopoOpt

import crash_report
crash_report.enable()

version = 3
narrow_band = True
plane_force = True
volume_fraction=0.07
# Initialize
n = 32
opt = TopoOpt(res=(n, n, n), version=version, volume_fraction=volume_fraction, grid_update_start=5 if narrow_band else 1000000,
              cg_max_iteraetions=30)

x, y, z = 0.48, 0.48, 0.48
opt.populate_grid(domain_type='dense')

# Set up BCs
opt.add_plane_dirichlet_bc(axis_to_fix="xyz", axis_to_search = 1, extreme=1)

opt.add_plane_load(force=(1, 0, 0), axis=1, extreme=-1)

# Optimize
opt.run()
