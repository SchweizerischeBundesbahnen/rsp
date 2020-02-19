from typing import Callable
from typing import NamedTuple
from typing import Optional

from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.route_dag.generators.route_dag_generator_schedule import RouteDAGConstraintsDict

SchedulingExperimentResult = NamedTuple('SchedulingExperimentResult',
                                        [('total_reward', int),
                                         ('solve_time', float),
                                         ('optimization_costs', float),
                                         ('build_problem_time', float),
                                         ('trainruns_dict', TrainrunDict),
                                         ('nb_conflicts', int),
                                         ('route_dag_constraints', Optional[RouteDAGConstraintsDict])
                                         ])

# test_id: int, solver_name: str, i_step: int
SolveProblemRenderCallback = Callable[[int, str, int], None]
