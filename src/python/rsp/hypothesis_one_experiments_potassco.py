import os
from typing import Optional

from rsp.hypothesis_one_pipeline_all_in_one import list_from_base_directory_and_run_experiment_agenda
from rsp.scheduling.asp.asp_data_types import ASPHeuristics
from rsp.step_01_planning.experiment_parameters_and_ranges import ExperimentParameters
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


def run_potassco_agenda(
    base_directory: str, experiment_output_base_directory: Optional[str] = None, experiment_filter=None, parallel_compute: int = 5, csv_only: bool = False
):
    reschedule_parameters_range = ReScheduleParametersRange(
        earliest_malfunction=[30, 30, 1],
        malfunction_duration=[50, 50, 1],
        # take all agents (200 is larger than largest number of agents)
        malfunction_agent_id=[0, 200, 200],
        number_of_shortest_paths_per_agent=[10, 10, 1],
        max_window_size_from_earliest=[60, 60, 1],
        asp_seed_value=[99, 99, 1],
        # route change is penalized the same as 30 seconds delay
        weight_route_change=[30, 30, 1],
        weight_lateness_seconds=[1, 1, 1],
    )

    experiments_per_grid_element = 1
    experiment_name_prefix = os.path.basename(base_directory) + "_"
    # baseline with defaults
    list_from_base_directory_and_run_experiment_agenda(
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
    list_from_base_directory_and_run_experiment_agenda(
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_base_directory,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=("%swith_SEQ" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        experiment_filter=experiment_filter,
        csv_only=csv_only,
        global_constants=get_defaults(reschedule_heuristics=[ASPHeuristics.HEURISTIC_SEQ]),
    )
    # effect of delay model resolution with 2, 5, 10 (SIM-542)
    list_from_base_directory_and_run_experiment_agenda(
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_base_directory,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=("%swith_delay_model_resolution_2" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        experiment_filter=experiment_filter,
        csv_only=csv_only,
        global_constants=get_defaults(delay_model_resolution=2),
    )
    list_from_base_directory_and_run_experiment_agenda(
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_base_directory,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=("%swith_delay_model_resolution_5" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        experiment_filter=experiment_filter,
        csv_only=csv_only,
        global_constants=get_defaults(delay_model_resolution=5),
    )
    list_from_base_directory_and_run_experiment_agenda(
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_base_directory,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=("%swith_delay_model_resolution_10" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        experiment_filter=experiment_filter,
        csv_only=csv_only,
        global_constants=get_defaults(delay_model_resolution=10),
    )
    # effect of --propagate (SIM-543)
    list_from_base_directory_and_run_experiment_agenda(
        experiment_base_directory=base_directory,
        experiment_output_directory=experiment_output_base_directory,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=("%swithout_propagate_partial" % experiment_name_prefix),
        parallel_compute=parallel_compute,
        experiments_per_grid_element=experiments_per_grid_element,
        experiment_filter=experiment_filter,
        csv_only=csv_only,
        global_constants=get_defaults(dl_propagate_partial=False),
    )


def generate_potassco_infras_and_schedules(base_directory: Optional[str] = None, parallel_compute: int = 5):
    if base_directory is None:
        base_directory = create_experiment_folder_name("h1")
        check_create_folder(base_directory)

    infra_parameters_range = InfrastructureParametersRange(
        number_of_agents=[64, 64, 1],
        width=[40, 40, 1],
        height=[40, 40, 1],
        flatland_seed_value=[10, 10, 1],
        max_num_cities=[4, 4, 1],
        max_rail_in_city=[3, 3, 1],
        max_rail_between_cities=[1, 1, 1],
        number_of_shortest_paths_per_agent=[10, 10, 1],
    )
    schedule_parameters_range = ScheduleParametersRange(asp_seed_value=[1, 104, 30], number_of_shortest_paths_per_agent_schedule=[1, 1, 1],)

    create_infrastructure_and_schedule_from_ranges(
        base_directory=base_directory,
        infrastructure_parameters_range=infra_parameters_range,
        schedule_parameters_range=schedule_parameters_range,
        speed_data={
            1.0: 0.25,  # Fast passenger train
            1.0 / 2.0: 0.25,  # Fast freight train
            1.0 / 3.0: 0.25,  # Slow commuter train
            1.0 / 4.0: 0.25,
        },  # Slow freight train
        grid_mode=False,
        run_experiments_parallel=parallel_compute,
    )
    return base_directory


def experiment_filter_first_ten_of_each_schedule(experiment: ExperimentParameters):
    return experiment.re_schedule_parameters.malfunction_agent_id < 20


if __name__ == "__main__":
    parallel_compute = AVAILABLE_CPUS // 2
    generate_potassco_infras_and_schedules(base_directory=INFRAS_AND_SCHEDULES_FOLDER, parallel_compute=parallel_compute)
    run_potassco_agenda(
        base_directory=INFRAS_AND_SCHEDULES_FOLDER,
        # incremental re-start after interruption
         experiment_filter=experiment_filter_first_ten_of_each_schedule,
        parallel_compute=parallel_compute,
        csv_only=False,
    )
