from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Set

import networkx as nx
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.schedule_problem_description.data_types_and_utils import RouteDAGConstraintsDict
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription

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


def fake_solver_statistics(elapsed_time):
    return {
        "summary": {
            "times": {
                "total": elapsed_time,
                "solve": elapsed_time,
            },
            "costs": [-1]
        },
        "solving": {
            "solvers": {
                "choices": -1,
                "conflicts": -1,
            }
        },
        "user_accu": {
            "DifferenceLogic": {
                "Thread": []
            }
        },
        "user_step": {
            "DifferenceLogic": {
                "Thread": []
            }
        }
    }


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

Schedule = NamedTuple('Schedule', [
    ('schedule_problem_description', ScheduleProblemDescription),
    ('schedule_experiment_result', SchedulingExperimentResult),
])

# TODO SIM-661 we should separate grid generation from agent placement, speed generation and topo_dict extraction (shortest paths);
#  however, we do not pass the city information out of FLATland to place agents.
Infrastructure = NamedTuple('Infrastructure', [
    ('topo_dict', Dict[int, nx.DiGraph]),
    ('minimum_travel_time_dict', Dict[int, int]),
    ('max_episode_steps', int)
])
