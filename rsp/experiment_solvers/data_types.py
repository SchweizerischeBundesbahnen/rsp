from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Set

from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.experiment_solvers.global_switches import COMPATIBILITY_MODE
from rsp.route_dag.route_dag import RouteDAGConstraintsDict
from rsp.route_dag.route_dag import ScheduleProblemDescription

ExperimentMalfunction = NamedTuple('ExperimentMalfunction', [
    ('time_step', int),
    ('agent_id', int),
    ('malfunction_duration', int)
])

SchedulingExperimentResult = NamedTuple('SchedulingExperimentResult',
                                        [('total_reward', int),
                                         ('solve_time', float),
                                         ('optimization_costs', float),
                                         ('build_problem_time', float),
                                         ('trainruns_dict', TrainrunDict),
                                         ('nb_conflicts', int),
                                         ('route_dag_constraints', Optional[RouteDAGConstraintsDict]),
                                         ('solver_statistics', Dict),
                                         ('solver_result', Set[str]),
                                         ('solver_configuration', Dict),
                                         ('solver_seed', int),
                                         ('solver_program', Optional[List[str]])
                                         ])
if COMPATIBILITY_MODE:
    SchedulingExperimentResult.__new__.__defaults__ = (None,) * len(SchedulingExperimentResult._fields)
else:
    # backwards compatibility and space reduction: solver_program is optional
    SchedulingExperimentResult.__new__.__defaults__ = (None,)
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


def schedule_experiment_results_equals_modulo_solve_time(s1: SchedulingExperimentResult,
                                                         s2: SchedulingExperimentResult):
    """Tests whether two `ScheduleExperimentResults' are the equal except for
    solve_time."""
    for index, slot in enumerate(s1._fields):
        if slot in ['solve_time', 'build_problem_time', 'solver_statistics']:
            continue
        elif s1[index] != s2[index]:
            return False
    return True


# test_id: int, solver_name: str, i_step: int
SolveProblemRenderCallback = Callable[[int, str, int], None]

ScheduleAndMalfunction = NamedTuple('ScheduleAndMalfunction', [
    ('schedule_problem_description', ScheduleProblemDescription),
    ('schedule_experiment_result', SchedulingExperimentResult),
    ('experiment_malfunction', ExperimentMalfunction)
])
