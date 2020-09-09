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
