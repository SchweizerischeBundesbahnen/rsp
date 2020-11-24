import os
from typing import List
from typing import Optional

from rsp.step_01_planning.agenda_expansion import create_experiment_agenda_from_infrastructure_and_schedule_ranges
from rsp.step_01_planning.experiment_parameters_and_ranges import InfrastructureParametersRange
from rsp.step_01_planning.experiment_parameters_and_ranges import ReScheduleParametersRange
from rsp.step_01_planning.experiment_parameters_and_ranges import ScheduleParametersRange
from rsp.step_03_run.experiments import AVAILABLE_CPUS
from rsp.step_03_run.experiments import create_experiment_folder_name
from rsp.step_03_run.experiments import create_infrastructure_and_schedule_from_ranges
from rsp.step_03_run.experiments import list_infrastructure_and_schedule_params_from_base_directory
from rsp.step_03_run.experiments import run_experiment_agenda
from rsp.step_03_run.experiments import save_experiment_agenda_and_hash_to_file
from rsp.step_04_analysis.data_analysis_all_in_one import hypothesis_one_data_analysis
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
    global_constants=None,
    online_unrestricted_only: bool = False,
    run_analysis: bool = False,
    qualitative_analysis_experiment_ids: List[int] = None,
):
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

    run_experiment_agenda(
        experiment_agenda=experiment_agenda,
        experiment_base_directory=experiment_base_directory,
        experiment_output_directory=experiment_output_directory,
        filter_experiment_agenda=experiment_filter,
        csv_only=csv_only,
        online_unrestricted_only=online_unrestricted_only,
    )
    # C. Experiment Analysis
    if run_analysis:
        hypothesis_one_data_analysis(
            experiment_output_directory=experiment_output_directory, analysis_2d=True, qualitative_analysis_experiment_ids=qualitative_analysis_experiment_ids,
        )
    return experiment_output_directory, experiment_agenda
