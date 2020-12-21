import os
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Set
from typing import Tuple

import sys
sys.version

from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.global_data_configuration import EXPERIMENT_INFRA_SUBDIRECTORY_NAME
from rsp.global_data_configuration import EXPERIMENT_SCHEDULE_SUBDIRECTORY_NAME
from rsp.scheduling.scheduling_problem import RouteDAGConstraintsDict
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ScheduleParameters
from rsp.utils.pickle_helper import _pickle_dump
from rsp.utils.pickle_helper import _pickle_load

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


def save_schedule(
    schedule: Schedule, schedule_parameters: ScheduleParameters, base_directory: str,
):
    """Persist `Schedule` and `ScheduleParameters` to a file.

    Parameters
    ----------
    schedule_parameters
    schedule
    base_directory
    """
    folder = os.path.join(
        base_directory,
        EXPERIMENT_INFRA_SUBDIRECTORY_NAME,
        f"{schedule_parameters.infra_id:03d}",
        EXPERIMENT_SCHEDULE_SUBDIRECTORY_NAME,
        f"{schedule_parameters.schedule_id:03d}",
    )
    _pickle_dump(obj=schedule, folder=folder, file_name="schedule.pkl")
    _pickle_dump(obj=schedule_parameters, folder=folder, file_name="schedule_parameters.pkl")


def exists_schedule(base_directory: str, infra_id: int, schedule_id: int) -> bool:
    """Does a persisted `Schedule` exist?"""
    file_name = os.path.join(
        base_directory, EXPERIMENT_INFRA_SUBDIRECTORY_NAME, f"{infra_id:03d}", EXPERIMENT_SCHEDULE_SUBDIRECTORY_NAME, f"{schedule_id:03d}", f"schedule.pkl"
    )
    return os.path.isfile(file_name)


def load_schedule(base_directory: str, infra_id: int, schedule_id: int = 0) -> Tuple[Schedule, ScheduleParameters]:
    """Load a persisted `Schedule` from a file.
    Parameters
    ----------
    schedule_id
    base_directory
    infra_id


    Returns
    -------
    """
    folder = os.path.join(base_directory, EXPERIMENT_INFRA_SUBDIRECTORY_NAME, f"{infra_id:03d}", EXPERIMENT_SCHEDULE_SUBDIRECTORY_NAME, f"{schedule_id:03d}")
    schedule = _pickle_load(folder=folder, file_name="schedule.pkl")
    schedule_parameters = _pickle_load(folder=folder, file_name="schedule_parameters.pkl")
    return schedule, schedule_parameters
