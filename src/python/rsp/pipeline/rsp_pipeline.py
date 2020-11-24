import os
from shutil import copyfile
from typing import Optional

from rsp.scheduling.asp.asp_data_types import ASPHeuristics
from rsp.step_01_planning.experiment_parameters_and_ranges import InfrastructureParametersRange
from rsp.step_01_planning.experiment_parameters_and_ranges import ReScheduleParametersRange
from rsp.step_01_planning.experiment_parameters_and_ranges import ScheduleParametersRange
from rsp.step_03_run.experiments import AVAILABLE_CPUS
from rsp.step_03_run.experiments import create_experiment_folder_name
from rsp.step_03_run.experiments import create_infrastructure_and_schedule_from_ranges
from rsp.utils.file_utils import check_create_folder
from rsp.utils.global_constants import get_defaults
from rsp.utils.global_data_configuration import BASELINE_DATA_FOLDER
from rsp.utils.global_data_configuration import INFRAS_AND_SCHEDULES_FOLDER
from utils.rsp_pipline_offline import list_from_base_directory_and_run_experiment_agenda


def run_agenda(
    base_directory: str,
    reschedule_parameters_range: ReScheduleParametersRange,
    experiment_output_base_directory: Optional[str] = None,
    experiment_filter=None,
    parallel_compute: int = 5,
    csv_only: bool = False,
):
    experiments_per_grid_element = 1
    experiment_name_prefix = os.path.basename(base_directory) + "_"
    # baseline with defaults
    _, experiment_output_base_directory = list_from_base_directory_and_run_experiment_agenda(
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_base_directory,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=("%sbaseline" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        experiment_filter=experiment_filter,
        csv_only=csv_only,
        global_constants=get_defaults(),
    )
    # effect of SEQ heuristic (SIM-167)
    experiment_output_directory_with_seq = experiment_output_base_directory.replace("baseline", "with_SEQ")
    check_create_folder(experiment_output_directory_with_seq)
    copyfile(experiment_output_base_directory + "/experiment_agenda.pkl", experiment_output_directory_with_seq + "/experiment_agenda.pkl")
    list_from_base_directory_and_run_experiment_agenda(
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_directory_with_seq,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=("%swith_SEQ" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        experiment_filter=experiment_filter,
        csv_only=csv_only,
        global_constants=get_defaults(reschedule_heuristics=[ASPHeuristics.HEURISTIC_SEQ]),
        online_unrestricted_only=True,
    )
    # effect of delay model resolution with 2, 5, 10 (SIM-542)
    experiment_output_directory_with_delay_model_resolution_2 = experiment_output_base_directory.replace("baseline", "with_delay_model_resolution_2")
    check_create_folder(experiment_output_directory_with_delay_model_resolution_2)
    copyfile(
        experiment_output_base_directory + "/experiment_agenda.pkl", experiment_output_directory_with_delay_model_resolution_2 + "/experiment_agenda.pkl",
    )
    list_from_base_directory_and_run_experiment_agenda(
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_directory_with_delay_model_resolution_2,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=("%swith_delay_model_resolution_2" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        experiment_filter=experiment_filter,
        csv_only=csv_only,
        global_constants=get_defaults(delay_model_resolution=2),
        online_unrestricted_only=True,
    )
    experiment_output_directory_with_delay_model_resolution_5 = experiment_output_base_directory.replace("baseline", "with_delay_model_resolution_5")
    check_create_folder(experiment_output_directory_with_delay_model_resolution_5)

    copyfile(
        experiment_output_base_directory + "/experiment_agenda.pkl", experiment_output_directory_with_delay_model_resolution_5 + "/experiment_agenda.pkl",
    )
    list_from_base_directory_and_run_experiment_agenda(
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_directory_with_delay_model_resolution_5,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=("%swith_delay_model_resolution_5" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        experiment_filter=experiment_filter,
        csv_only=csv_only,
        global_constants=get_defaults(delay_model_resolution=5),
        online_unrestricted_only=True,
    )
    experiment_output_directory_with_delay_model_resolution_10 = experiment_output_base_directory.replace("baseline", "with_delay_model_resolution_10")
    check_create_folder(experiment_output_directory_with_delay_model_resolution_10)
    copyfile(
        experiment_output_base_directory + "/experiment_agenda.pkl", experiment_output_directory_with_delay_model_resolution_10 + "/experiment_agenda.pkl",
    )
    list_from_base_directory_and_run_experiment_agenda(
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_directory_with_delay_model_resolution_10,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=("%swith_delay_model_resolution_10" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        experiment_filter=experiment_filter,
        csv_only=csv_only,
        global_constants=get_defaults(delay_model_resolution=10),
        online_unrestricted_only=True,
    )
    # effect of --propagate (SIM-543)
    experiment_output_directory_without_propagate_partial = experiment_output_base_directory.replace("baseline", "without_propagate_partial")
    check_create_folder(experiment_output_directory_without_propagate_partial)
    copyfile(
        experiment_output_base_directory + "/experiment_agenda.pkl", experiment_output_directory_without_propagate_partial + "/experiment_agenda.pkl",
    )
    list_from_base_directory_and_run_experiment_agenda(
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_directory_without_propagate_partial,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=("%swithout_propagate_partial" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        experiment_filter=experiment_filter,
        csv_only=csv_only,
        global_constants=get_defaults(dl_propagate_partial=False),
        online_unrestricted_only=True,
    )
    return experiment_output_base_directory


def generate_infras_and_schedules(
    infra_parameters_range: InfrastructureParametersRange,
    schedule_parameters_range: ScheduleParametersRange,
    base_directory: Optional[str] = None,
    parallel_compute: int = 5,
    speed_data=None,
    grid_mode: bool = False,
):
    if speed_data is None:
        speed_data = {
            1.0: 0.25,  # Fast passenger train
            1.0 / 2.0: 0.25,  # Fast freight train
            1.0 / 3.0: 0.25,  # Slow commuter train
            1.0 / 4.0: 0.25,  # Slow freight train
        }
    if base_directory is None:
        base_directory = create_experiment_folder_name("h1")
        check_create_folder(base_directory)

    create_infrastructure_and_schedule_from_ranges(
        base_directory=base_directory,
        infrastructure_parameters_range=infra_parameters_range,
        schedule_parameters_range=schedule_parameters_range,
        grid_mode=grid_mode,
        speed_data=speed_data,
        run_experiments_parallel=parallel_compute,
    )
    return base_directory


def rsp_pipeline(
    infra_parameters_range: InfrastructureParametersRange,
    schedule_parameters_range: ScheduleParametersRange,
    reschedule_parameters_range: ReScheduleParametersRange,
    base_directory=INFRAS_AND_SCHEDULES_FOLDER,
    experiment_output_base_directory=BASELINE_DATA_FOLDER,
    experiment_filter=None,
    speed_data=None,
    grid_mode: bool = False,
):
    parallel_compute = AVAILABLE_CPUS // 2
    generate_infras_and_schedules(
        infra_parameters_range=infra_parameters_range,
        schedule_parameters_range=schedule_parameters_range,
        base_directory=base_directory,
        parallel_compute=parallel_compute,
        speed_data=speed_data,
        grid_mode=grid_mode,
    )
    return run_agenda(
        base_directory=base_directory,
        reschedule_parameters_range=reschedule_parameters_range,
        # incremental re-start after interruption
        experiment_output_base_directory=experiment_output_base_directory,
        experiment_filter=experiment_filter,
        parallel_compute=parallel_compute,
        csv_only=True,
    )
