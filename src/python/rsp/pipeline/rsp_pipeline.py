import os
from typing import List
from typing import Optional

from rsp.step_01_agenda_expansion.agenda_expansion import create_experiment_agenda_from_infrastructure_and_schedule_ranges
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import InfrastructureParametersRange
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ReScheduleParametersRange
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ScheduleParametersRange
from rsp.step_01_agenda_expansion.global_constants import GlobalConstants
from rsp.step_05_experiment_run.experiment_run import AVAILABLE_CPUS
from rsp.step_05_experiment_run.experiment_run import create_experiment_folder_name
from rsp.step_05_experiment_run.experiment_run import create_infrastructure_and_schedule_from_ranges
from rsp.step_05_experiment_run.experiment_run import list_infrastructure_and_schedule_params_from_base_directory
from rsp.step_05_experiment_run.experiment_run import run_experiment_agenda
from rsp.step_05_experiment_run.experiment_run import save_experiment_agenda_and_hash_to_file
from rsp.step_06_analysis.data_analysis_all_in_one import hypothesis_one_data_analysis
from rsp.utils.file_utils import check_create_folder
from rsp.utils.json_file_dumper import dump_object_as_human_readable_json


def generate_infras_and_schedules(
    infra_parameters_range: InfrastructureParametersRange,
    schedule_parameters_range: ScheduleParametersRange,
    base_directory: Optional[str] = None,
    parallel_compute: int = 5,
    speed_data=None,
    grid_mode: bool = False,
):
    """Generate infrastructure and schedules for the given ranges.

    Parameters
    ----------
    infra_parameters_range
    schedule_parameters_range
    base_directory
    parallel_compute
    speed_data
    grid_mode

    Returns
    -------
    """
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
    experiment_name: str,
    experiment_base_directory=None,
    experiment_output_directory=None,
    experiment_filter=None,
    speed_data=None,
    grid_mode: bool = False,
    parallel_compute=AVAILABLE_CPUS // 2,
    experiments_per_grid_element=1,
    csv_only: bool = False,
    global_constants: GlobalConstants = None,
    online_unrestricted_only: bool = False,
    run_analysis: bool = False,
    qualitative_analysis_experiment_ids: List[int] = None,
):
    """Run all steps 1..7 of the RSP pipeline: 1 Agenda Expansion 2
    Infrastructure Generation 3 Schedule Generation 4 Agenda Run 5 Experiment
    Run 6 Agenda and Experiment Analysis.

    Parameters
    ----------
    infra_parameters_range
    schedule_parameters_range
    reschedule_parameters_range
    experiment_name
    experiment_base_directory
    experiment_output_directory
    experiment_filter
    speed_data
    grid_mode
    parallel_compute
    experiments_per_grid_element
    csv_only
    global_constants
    online_unrestricted_only
    run_analysis
    qualitative_analysis_experiment_ids

    Returns
    -------
    """
    # steps 1--3
    generate_infras_and_schedules(
        infra_parameters_range=infra_parameters_range,
        schedule_parameters_range=schedule_parameters_range,
        base_directory=experiment_base_directory,
        parallel_compute=parallel_compute,
        speed_data=speed_data,
        grid_mode=grid_mode,
    )
    infra_parameters_list, infra_schedule_dict = list_infrastructure_and_schedule_params_from_base_directory(base_directory=experiment_base_directory)
    experiment_agenda = create_experiment_agenda_from_infrastructure_and_schedule_ranges(
        experiment_name=experiment_name,
        reschedule_parameters_range=reschedule_parameters_range,
        infra_parameters_list=infra_parameters_list,
        infra_schedule_dict=infra_schedule_dict,
        experiments_per_grid_element=experiments_per_grid_element,
        global_constants=global_constants,
    )
    if experiment_output_directory is None:
        experiment_output_directory = f"{experiment_base_directory}/" + create_experiment_folder_name(experiment_agenda.experiment_name)
        check_create_folder(experiment_output_directory)
    save_experiment_agenda_and_hash_to_file(output_base_folder=experiment_output_directory, experiment_agenda=experiment_agenda)
    dump_object_as_human_readable_json(obj=infra_parameters_range, file_name=os.path.join(experiment_output_directory, "infrastructure_parameters_range.json"))
    dump_object_as_human_readable_json(obj=schedule_parameters_range, file_name=os.path.join(experiment_output_directory, "schedule_parameters_range.json"))
    dump_object_as_human_readable_json(obj=reschedule_parameters_range, file_name=os.path.join(experiment_output_directory, "reschedule_parameters_range.json"))
    dump_object_as_human_readable_json(obj=experiment_agenda, file_name=os.path.join(experiment_output_directory, "experiment_agenda.json"))

    # steps 4--5
    run_experiment_agenda(
        experiment_agenda=experiment_agenda,
        experiment_base_directory=experiment_base_directory,
        experiment_output_directory=experiment_output_directory,
        filter_experiment_agenda=experiment_filter,
        csv_only=csv_only,
        online_unrestricted_only=online_unrestricted_only,
    )
    # 7 analysis
    if run_analysis:
        hypothesis_one_data_analysis(
            experiment_output_directory=experiment_output_directory, qualitative_analysis_experiment_ids=qualitative_analysis_experiment_ids,
        )
    return experiment_output_directory, experiment_agenda
