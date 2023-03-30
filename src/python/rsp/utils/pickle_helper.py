import _pickle
import os
import pickle
from typing import Any
from typing import Optional

from rsp.utils.file_utils import check_create_folder
from rsp.utils.rsp_logger import rsp_logger

MODULE_RENAME_MAPPING = {
    # -> rsp.scheduling
    ("rsp.schedule_problem_description.data_types_and_utils", "TopoDict"): ("rsp.scheduling.scheduling_problem", "TopoDict"),
    ("rsp.schedule_problem_description.data_types_and_utils", "AgentPaths"): ("rsp.scheduling.scheduling_problem", "AgentPaths"),
    ("rsp.schedule_problem_description.data_types_and_utils", "AgentsPathsDict"): ("rsp.scheduling.scheduling_problem", "AgentsPathsDict"),
    ("rsp.schedule_problem_description.data_types_and_utils", "RouteDAGConstraints"): ("rsp.scheduling.scheduling_problem", "RouteDAGConstraints"),
    ("rsp.schedule_problem_description.data_types_and_utils", "RouteDAGConstraintsDict"): ("rsp.scheduling.scheduling_problem", "RouteDAGConstraintsDict"),
    ("rsp.schedule_problem_description.data_types_and_utils", "RouteDagEdge"): ("rsp.scheduling.scheduling_problem", "RouteDagEdge"),
    ("rsp.schedule_problem_description.data_types_and_utils", "RouteSectionPenalties"): ("rsp.scheduling.scheduling_problem", "RouteSectionPenalties"),
    ("rsp.schedule_problem_description.data_types_and_utils", "WaypointPenalties"): ("rsp.scheduling.scheduling_problem", "WaypointPenalties"),
    ("rsp.schedule_problem_description.data_types_and_utils", "RouteSectionPenaltiesDict"): ("rsp.scheduling.scheduling_problem", "RouteSectionPenaltiesDict"),
    ("rsp.experiment_solvers.data_types", "Schedule"): ("rsp.scheduling.schedule", "Schedule"),
    ("rsp.experiment_solvers.data_types", "SchedulingExperimentResult"): ("rsp.scheduling.schedule", "SchedulingExperimentResult"),
    ("rsp.schedule_problem_description.data_types_and_utils", "ScheduleProblemDescription"): (
        "rsp.scheduling.scheduling_problem",
        "ScheduleProblemDescription",
    ),
    # -> rsp.step_01_agenda_expansion
    ("rsp.utils.global_constants", "GlobalConstants"): ("rsp.step_01_agenda_expansion.global_constants", "GlobalConstants"),
    ("rsp.utils.data_types", "ExperimentParameters"): ("rsp.step_01_agenda_expansion.experiment_parameters_and_ranges", "ExperimentParameters"),
    ("rsp.utils.data_types", "ExperimentAgenda"): ("rsp.step_01_agenda_expansion.experiment_parameters_and_ranges", "ExperimentAgenda"),
    ("rsp.utils.data_types", "ParameterRanges"): ("rsp.step_01_agenda_expansion.experiment_parameters_and_ranges", "ParameterRanges"),
    ("rsp.utils.data_types", "ParameterRangesAndSpeedData"): ("rsp.step_01_agenda_expansion.experiment_parameters_and_ranges", "ParameterRangesAndSpeedData"),
    ("rsp.utils.data_types", "InfrastructureParameters"): ("rsp.step_01_agenda_expansion.experiment_parameters_and_ranges", "InfrastructureParameters"),
    ("rsp.utils.data_types", "InfrastructureParametersRange"): (
        "rsp.step_01_agenda_expansion.experiment_parameters_and_ranges",
        "InfrastructureParametersRange",
    ),
    ("rsp.utils.data_types", "ScheduleParameters"): ("rsp.step_01_agenda_expansion.experiment_parameters_and_ranges", "ScheduleParameters"),
    ("rsp.utils.data_types", "ScheduleParametersRange"): ("rsp.step_01_agenda_expansion.experiment_parameters_and_ranges", "ScheduleParametersRange"),
    ("rsp.utils.data_types", "ReScheduleParameters"): ("rsp.step_01_agenda_expansion.experiment_parameters_and_ranges", "ReScheduleParameters"),
    ("rsp.utils.data_types", "ReScheduleParametersRange"): ("rsp.step_01_agenda_expansion.experiment_parameters_and_ranges", "ReScheduleParametersRange"),
    ("rsp.step_01_planning.experiment_parameters_and_ranges", "ExperimentParameters"): (
        "rsp.step_01_agenda_expansion.experiment_parameters_and_ranges",
        "ExperimentParameters",
    ),
    ("rsp.step_01_planning.experiment_parameters_and_ranges", "ExperimentAgenda"): (
        "rsp.step_01_agenda_expansion.experiment_parameters_and_ranges",
        "ExperimentAgenda",
    ),
    ("rsp.step_01_planning.experiment_parameters_and_ranges", "ParameterRanges"): (
        "rsp.step_01_agenda_expansion.experiment_parameters_and_ranges",
        "ParameterRanges",
    ),
    ("rsp.step_01_planning.experiment_parameters_and_ranges", "ParameterRangesAndSpeedData"): (
        "rsp.step_01_agenda_expansion.experiment_parameters_and_ranges",
        "ParameterRangesAndSpeedData",
    ),
    ("rsp.step_01_planning.experiment_parameters_and_ranges", "InfrastructureParameters"): (
        "rsp.step_01_agenda_expansion.experiment_parameters_and_ranges",
        "InfrastructureParameters",
    ),
    ("rsp.step_01_planning.experiment_parameters_and_ranges", "InfrastructureParametersRange"): (
        "rsp.step_01_agenda_expansion.experiment_parameters_and_ranges",
        "InfrastructureParametersRange",
    ),
    ("rsp.step_01_planning.experiment_parameters_and_ranges", "ScheduleParameters"): (
        "rsp.step_01_agenda_expansion.experiment_parameters_and_ranges",
        "ScheduleParameters",
    ),
    ("rsp.step_01_planning.experiment_parameters_and_ranges", "ScheduleParametersRange"): (
        "rsp.step_01_agenda_expansion.experiment_parameters_and_ranges",
        "ScheduleParametersRange",
    ),
    ("rsp.step_01_planning.experiment_parameters_and_ranges", "ReScheduleParameters"): (
        "rsp.step_01_agenda_expansion.experiment_parameters_and_ranges",
        "ReScheduleParameters",
    ),
    ("rsp.step_01_planning.experiment_parameters_and_ranges", "ReScheduleParametersRange"): (
        "rsp.step_01_agenda_expansion.experiment_parameters_and_ranges",
        "ReScheduleParametersRange",
    ),
    # -> rsp.step_02_infrastructure_generation
    ("rsp.experiment_solvers.data_types", "Infrastructure"): ("rsp.step_02_infrastructure_generation.infrastructure", "Infrastructure"),
    ("rsp.step_01_planning.data_types", "Infrastructure"): ("rsp.step_02_infrastructure_generation.infrastructure", "Infrastructure"),
    # -> rsp.step_05_experiment_run
    ("rsp.step_03_run.experiment_results", "ExperimentResults"): ("rsp.step_05_experiment_run.experiment_results", "ExperimentResults"),
    ("rsp.step_02_setup.data_types", "ExperimentMalfunction"): ("rsp.step_05_experiment_run.experiment_malfunction", "ExperimentMalfunction"),
    ("rsp.step_02_setup.data_types", "Infrastructure"): ("rsp.step_02_infrastructure_generation.infrastructure", "Infrastructure"),
}


# https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            renamed_module, renamed_name = MODULE_RENAME_MAPPING.get((module, name), (module, name))
            return super(RenameUnpickler, self).find_class(renamed_module, renamed_name)
        except TypeError as e:
            rsp_logger.error(f"renamed_module={renamed_module} for module={module}, renamed_name={renamed_name} for name={name}, {e}")
            raise e


def _pickle_dump(obj: Any, file_name: str, folder: Optional[str] = None):
    file_path = file_name
    if folder is not None:
        file_path = os.path.join(folder, file_name)
        check_create_folder(folder)
    else:
        check_create_folder(os.path.dirname(file_name))
    with open(file_path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _pickle_load(file_name: str, folder: Optional[str] = None):
    file_path = file_name
    if folder is not None:
        file_path = os.path.join(folder, file_name)
    with open(file_path, "rb") as handle:
        try:
            return RenameUnpickler(handle).load()
        except (_pickle.UnpicklingError, ModuleNotFoundError) as e:
            rsp_logger.error(f"Failed unpickling {file_path}")
            rsp_logger.error(e, exc_info=True)
            raise e
