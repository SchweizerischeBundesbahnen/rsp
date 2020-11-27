from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Set

from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.scheduling.scheduling_problem import RouteDAGConstraintsDict
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription

SchedulingExperimentResult = NamedTuple(
    "SchedulingExperimentResult",
    [
        ("total_reward", int),
        ("solve_time", float),
        ("optimization_costs", float),
        ("build_problem_time", float),
        ("trainruns_dict", TrainrunDict),
        ("nb_conflicts", int),
        ("route_dag_constraints", Optional[RouteDAGConstraintsDict]),
        ("solver_statistics", Dict),
        ("solver_result", Set[str]),
        ("solver_configuration", Dict),
        ("solver_seed", int),
        ("solver_program", Optional[List[str]]),
    ],
)

Schedule = NamedTuple("Schedule", [("schedule_problem_description", ScheduleProblemDescription), ("schedule_experiment_result", SchedulingExperimentResult)])
