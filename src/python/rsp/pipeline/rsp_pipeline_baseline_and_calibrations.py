import os
from shutil import copyfile
from typing import Optional

from rsp.pipeline.rsp_pipeline import rsp_pipeline
from rsp.scheduling.asp.asp_data_types import ASPHeuristics
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import InfrastructureParametersRange
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ReScheduleParametersRange
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ScheduleParametersRange
from rsp.step_01_agenda_expansion.global_constants import get_defaults
from rsp.step_05_experiment_run.experiment_run import AVAILABLE_CPUS
from rsp.utils.file_utils import check_create_folder


def rsp_pipeline_baseline_and_calibrations(
    base_directory: str,
    infra_parameters_range: InfrastructureParametersRange,
    schedule_parameters_range: ScheduleParametersRange,
    reschedule_parameters_range: ReScheduleParametersRange,
    # if passed, run incrementally
    experiment_output_base_directory: Optional[str] = None,
    experiment_filter=None,
    speed_data=None,
    grid_mode: bool = False,
    parallel_compute=AVAILABLE_CPUS // 2,
    experiments_per_grid_element=1,
    csv_only: bool = False,
):
    """Run the same rsp pipeline multiple times with different
    `GlobalConstants`.

    Parameters
    ----------
    base_directory
    infra_parameters_range
    schedule_parameters_range
    reschedule_parameters_range
    experiment_output_base_directory
    experiment_filter
    speed_data
    grid_mode
    parallel_compute
    experiments_per_grid_element
    csv_only

    Returns
    -------
    """
    experiment_name_prefix = os.path.basename(base_directory) + "_"
    # baseline with defaults
    experiment_output_base_directory, _ = rsp_pipeline(
        infra_parameters_range=infra_parameters_range,
        schedule_parameters_range=schedule_parameters_range,
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_base_directory,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=("%sbaseline" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        grid_mode=grid_mode,
        speed_data=speed_data,
        experiment_filter=experiment_filter,
        csv_only=csv_only,
        global_constants=get_defaults(),
    )
    # effect of SEQ heuristic (SIM-167)
    experiment_output_directory_with_seq = experiment_output_base_directory.replace("baseline", "with_SEQ")
    check_create_folder(experiment_output_directory_with_seq)
    copyfile(experiment_output_base_directory + "/experiment_agenda.pkl", experiment_output_directory_with_seq + "/experiment_agenda.pkl")
    rsp_pipeline(
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_directory_with_seq,
        infra_parameters_range=infra_parameters_range,
        schedule_parameters_range=schedule_parameters_range,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=("%swith_SEQ" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        grid_mode=grid_mode,
        speed_data=speed_data,
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
    rsp_pipeline(
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_directory_with_delay_model_resolution_2,
        reschedule_parameters_range=reschedule_parameters_range,
        infra_parameters_range=infra_parameters_range,
        schedule_parameters_range=schedule_parameters_range,
        experiment_name=("%swith_delay_model_resolution_2" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        grid_mode=grid_mode,
        speed_data=speed_data,
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
    rsp_pipeline(
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_directory_with_delay_model_resolution_5,
        infra_parameters_range=infra_parameters_range,
        schedule_parameters_range=schedule_parameters_range,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=("%swith_delay_model_resolution_5" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        grid_mode=grid_mode,
        speed_data=speed_data,
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
    rsp_pipeline(
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_directory_with_delay_model_resolution_10,
        infra_parameters_range=infra_parameters_range,
        schedule_parameters_range=schedule_parameters_range,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=("%swith_delay_model_resolution_10" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        grid_mode=grid_mode,
        speed_data=speed_data,
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
    rsp_pipeline(
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_directory_without_propagate_partial,
        infra_parameters_range=infra_parameters_range,
        schedule_parameters_range=schedule_parameters_range,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=("%swithout_propagate_partial" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        grid_mode=grid_mode,
        speed_data=speed_data,
        experiment_filter=experiment_filter,
        csv_only=csv_only,
        global_constants=get_defaults(dl_propagate_partial=False),
        online_unrestricted_only=True,
    )
    return experiment_output_base_directory
