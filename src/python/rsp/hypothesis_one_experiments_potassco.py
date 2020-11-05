import os
from shutil import copyfile
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


INFRA_PARAMETERS_RANGE = InfrastructureParametersRange(
    number_of_agents=[64, 64, 1],
    width=[60, 60, 1],
    height=[60, 60, 1],
    flatland_seed_value=[10, 10, 1],
    max_num_cities=[4, 16, 12],
    max_rail_in_city=[3, 3, 1],
    max_rail_between_cities=[1, 1, 1],
    number_of_shortest_paths_per_agent=[10, 10, 1],
)
SCHEDULE_PARAMETERS_RANGE = ScheduleParametersRange(asp_seed_value=[1, 104, 1], number_of_shortest_paths_per_agent_schedule=[1, 1, 1],)
RESCHEDULE_PARAMETERS_RANGE = ReScheduleParametersRange(
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


def run_potassco_agenda(
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


def generate_potassco_infras_and_schedules(
    infra_parameters_range: InfrastructureParametersRange,
    schedule_parameters_range: ScheduleParametersRange,
    base_directory: Optional[str] = None,
    parallel_compute: int = 5,
):
    if base_directory is None:
        base_directory = create_experiment_folder_name("h1")
        check_create_folder(base_directory)

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
    return experiment.malfunction.agent_id <= 20


def hypothesis_one_experiments_potassco(
    infra_parameters_range: InfrastructureParametersRange,
    schedule_parameters_range: ScheduleParametersRange,
    reschedule_parameters_range: ReScheduleParametersRange,
    base_directory=INFRAS_AND_SCHEDULES_FOLDER,
    experiment_output_base_directory=BASELINE_DATA_FOLDER,
    experiment_filter=None,
):
    parallel_compute = AVAILABLE_CPUS // 2
    generate_potassco_infras_and_schedules(
        infra_parameters_range=infra_parameters_range,
        schedule_parameters_range=schedule_parameters_range,
        base_directory=base_directory,
        parallel_compute=parallel_compute,
    )
    return run_potassco_agenda(
        base_directory=base_directory,
        reschedule_parameters_range=reschedule_parameters_range,
        # incremental re-start after interruption
        experiment_output_base_directory=experiment_output_base_directory,
        experiment_filter=experiment_filter,
        parallel_compute=parallel_compute,
        csv_only=False,
    )


if __name__ == "__main__":
    hypothesis_one_experiments_potassco(
        infra_parameters_range=INFRA_PARAMETERS_RANGE,
        schedule_parameters_range=SCHEDULE_PARAMETERS_RANGE,
        reschedule_parameters_range=RESCHEDULE_PARAMETERS_RANGE,
        base_directory="DENSITY_VARIATION_EXPERIMENTS_FIXED",
        experiment_output_base_directory="DENSITY_VARIATION_EXPERIMENTS_FIXED/DENSITY_VARIATION_EXPERIMENTS_FIXED_baseline_2020_11_05T09_58_04",
        experiment_filter=experiment_filter_first_ten_of_each_schedule,
    )
