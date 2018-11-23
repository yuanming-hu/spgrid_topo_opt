import sys
import os
import shutil
from taichi.dynamics import Simulation

import taichi.core as tc_core
import taichi as tc
import argparse
from taichi.misc.util import get_unique_task_id

class TopoOpt(Simulation):
  def __init__(self, **kwargs):
    res = kwargs['res']
    self.script_name = sys.argv[0].split('.')[0]
    assert(self.script_name.startswith('opt_'))
    self.script_name = self.script_name[4:]
    self.snapshot_period = kwargs.get('snapshot_period', 0)
    script_fn = os.path.join(os.getcwd(), sys.argv[0])

    suffix = ''

    self.version = kwargs.get('version', 0)
    if 'version' in kwargs:
      suffix += '_v{:0d}'.format(int(self.version))

    self.wireframe = kwargs.get('wireframe', False)
    if 'wireframe' in kwargs:
      if 'wireframe_grid_size' not in kwargs:
        kwargs['wireframe_grid_size'] = 10
      if 'wireframe_thickness' not in kwargs:
        kwargs['wireframe_thickness'] = 3
      if self.wireframe:
        suffix += '_wf{}g{}t{}'.format(int(self.wireframe), kwargs['wireframe_grid_size'], kwargs['wireframe_thickness'])
      else:
        suffix += '_wf{}'.format(int(self.wireframe))

    suffix += '_r{:04d}'.format(res[0])

    parser = argparse.ArgumentParser(description='Topology Optimization.')
    parser.add_argument('options', metavar='Option', type=str, nargs='*',
                        help='An option to override')
    parser.add_argument('-c', type=str, help='iteration to start from')

    args = parser.parse_args()

    if args.c is not None:
      suffix += '_continue'
    
    self.task_id = get_unique_task_id()
    self.suffix = suffix + kwargs.get('suffix', '')

    self.working_directory = os.path.join(tc.get_output_directory(), 'topo_opt', self.script_name, self.task_id + '_' + self.suffix)
    kwargs['working_directory'] = self.working_directory
    self.snapshot_directory = os.path.join(self.working_directory, 'snapshots')
    self.fem_directory = os.path.join(self.working_directory, 'fem')
    self.fem_obj_directory = os.path.join(self.working_directory, 'fem_obj')

    os.makedirs(self.snapshot_directory, exist_ok=True)
    os.makedirs(self.fem_directory, exist_ok=True)
    os.makedirs(self.fem_obj_directory, exist_ok=True)
    self.max_iterations = kwargs.get('max_iterations', 1000)
    
    self.log_fn = os.path.join(self.working_directory, 'log.txt')
    tc.start_memory_monitoring(os.path.join(self.working_directory, 'memory_usage.txt'), interval=0.1)
    tc.duplicate_stdout_to_file(self.log_fn)
    tc.redirect_print_to_log()
    tc.trace("log_fn = {}", self.log_fn)

    with open(script_fn) as f:
      script_content = f.read()

    shutil.copy(sys.argv[0], self.working_directory + "/")
    tc.info("Script backuped")
    tc.info("Script content:\n********************\n\n{}\n*******************", script_content)
    print(args)
    super().__init__(name='spgrid_topo_opt', **kwargs)

    if args.c is not None:
      # Restart (continue)
      print(args.options)
      print(args.c)

      last_iter = self.general_action(action='load_state',
                                      #filename=self.get_snapshot_file_name(args.c))
                                      filename=args.c)
      for o in args.options:
        o = o.split('=')
        assert(len(o) == 2)
        self.override_parameter(o[0], o[1])

      self.i_start = int(last_iter) + 1
      tc.info("\n*** Restarting from iter {}", self.i_start)
      self.run()
      exit()

    # Start from scratch
    self.i_start = 0

    tc.trace("log duplicated")
    if kwargs.get('check_log_file', True):
      assert(os.path.exists(self.log_fn))

  def output(self, iter):
    pass
    #self.general_action(action='output', file_name=self.get_frame_file_name(iter))

  def get_snapshot_file_name(self, iter):
    return "{}/{:05}.tcb.zip".format(self.snapshot_directory, iter)

  def iterate(self, i):
    tc.trace("Starting Iteration {}...".format(i))
    #if i > 10:
    #  self.general_action(action="set_step_limit", value=0.0)
    objective = float(self.general_action("iterate", iter=i))
    tc.trace("\n**** Task {}".format(self.task_id))
    #tc.trace("\n**** Iteration {} finished.\n*** (Maximum change = {:6.3})".format(i, changed))
    tc.trace("\n**** Iteration {} finished.\n*** (objective = {:6.3f})".format(i, objective))
    # self.output(i + 1)
    tc.core.print_profile_info()

    if self.snapshot_period != 0 and i % self.snapshot_period == 0:
      self.general_action('save_state',
                          filename=self.get_snapshot_file_name(i))
    return objective

  def run(self):
    # self.output(0)
    objectives = []
    blklog = open("{}/blocks.log".format(self.working_directory), "w")

    for i in range(self.i_start, self.max_iterations):
      blklog.write(self.get_block_counts() + '\n')
      blklog.flush()

      #if i % 5 == 1 and i > 20:
      #  tc.info("Computing objective")
      #  self.general_action(action='objective')
      obj = float(self.iterate(i))
      objectives.append(obj)
      if i > 10 and len(objectives) >= 4:
        r = abs((objectives[-1] + objectives[-2] - objectives[-3] - objectives[-4]) / (objectives[-1] + objectives[-2]))
        tc.trace("r = {:4.2f}%", r * 100)
        if r < 5e-3:
          tc.trace("*************** Should stop now,  Final objective: {}", objectives[-1])
      #print("debug, exiting")
      #break

    blklog.close()

  def dump(self):
    self.general_action(
        name='write_density', fn='density_%d.bin' % (self.res[0] + 1))

  def add_dirichlet_bc(self, center, radius=0.05, axis='xyz', value=(0, 0, 0)):
    assert isinstance(axis, str)
    for ch in axis:
      assert ch in 'xyz'
    self.general_action('add_dirichlet_bc', center=center, value=value, radius=radius, axis=axis)

  def add_plane_dirichlet_bc(self, axis_to_fix, axis_to_search, extreme, value=(0, 0, 0), bound1=(-0.51, -0.51, -0.51), bound2=(0.51, 0.51, 0.51)):
    assert isinstance(axis_to_fix, str)
    for ch in axis_to_fix:
      assert ch in 'xyz'
    self.general_action('add_plane_dirichlet_bc', axis_to_fix=axis_to_fix, axis_to_search=axis_to_search, extreme=extreme, value=value)

  # One node only by default
  def add_load(self, center, force, size=1e-6):
    self.general_action('add_load', center=center, force=force, size=size)

  def add_plane_load(self, force, axis_to_search=None, axis=None, extreme=1, bound1=(-0.51, -0.51, -0.51), bound2=(0.51, 0.51, 0.51)):
    if axis_to_search is None:
      assert axis is not None
      axis_to_search = axis
    self.general_action(action='add_plane_load', force=force, axis=axis_to_search, extreme=extreme, bound1=bound1, bound2=bound2)

  def populate_grid(self, domain_type, **kwargs):
    kwargs = tc.visual.asset_manager.asset_ptr_to_id(kwargs)
    self.general_action(action='populate_grid', domain_type=domain_type, **kwargs)

  def override_parameter(self, key, val):
    self.general_action(action='override', key=key, val=val)

  def load_density_from_fem(self, fn):
    self.general_action(action='load_density_from_fem', fn=fn)

  def get_block_counts(self):
    return self.general_action(action='get_block_counts')

