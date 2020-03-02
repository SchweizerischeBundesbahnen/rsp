from typing import Callable
from typing import NamedTuple
from typing import Optional

from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.route_dag.route_dag import RouteDAGConstraintsDict
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.utils.data_types import ExperimentMalfunction

SchedulingExperimentResult = NamedTuple('SchedulingExperimentResult',
                                        [('total_reward', int),
                                         ('solve_time', float),
                                         ('optimization_costs', float),
                                         ('build_problem_time', float),
                                         ('trainruns_dict', TrainrunDict),
                                         ('nb_conflicts', int),
                                         ('route_dag_constraints', Optional[RouteDAGConstraintsDict])
                                         ])
SchedulingExperimentResult.__doc__ = """
    Parameters
    ----------
    total_reward: int
    solve_time: float
    optimization_costs: float
    build_problem_time: float
    trainruns_dict: TrainrunDict
    nb_conflicts: int
    route_dag_constraints: Optional[RouteDAGConstraintsDict]
"""

# test_id: int, solver_name: str, i_step: int
SolveProblemRenderCallback = Callable[[int, str, int], None]

ScheduleAndMalfunction = NamedTuple('ScheduleAndMalfunction', [
    ('schedule_problem_description', ScheduleProblemDescription),
    ('schedule_experiment_result', SchedulingExperimentResult),
    ('experiment_malfunction', ExperimentMalfunction)
])
