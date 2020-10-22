import os
import re
from typing import Optional

from rsp.hypothesis_one_pipeline_all_in_one import list_from_base_directory_and_run_experiment_agenda
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import InfrastructureParametersRange
from rsp.utils.data_types import ReScheduleParametersRange
from rsp.utils.data_types import ScheduleParametersRange
from rsp.utils.experiments import AVAILABLE_CPUS
from rsp.utils.experiments import create_experiment_folder_name
from rsp.utils.experiments import create_infrastructure_and_schedule_from_ranges
from rsp.utils.file_utils import check_create_folder
from rsp.utils.global_data_configuration import BASELINE_DATA_FOLDER
from rsp.utils.global_data_configuration import INFRAS_AND_SCHEDULES_FOLDER


# TODO SIM-719 pass arguments instead of hacky file editing
def enable_seq(enable=True):
    off = "RESCHEDULE_HEURISTICS = []"
    on = "RESCHEDULE_HEURISTICS = [ASPHeuristics.HEURISTIC_SEQ]"
    file_name = "rsp/utils/global_constants.py"
    with open(file_name, "r") as fh:
        output_str = fh.read().replace(off if enable else on, on if enable else off)
    with open(file_name, "w") as output:
        output.write(output_str)


# TODO SIM-719 pass arguments instead of hacky file editing
def set_delay_model_resolution(resolution=1):
    file_name = "rsp/utils/global_constants.py"
    with open(file_name, "r") as fh:
        regex = re.compile("DELAY_MODEL_RESOLUTION = .*")
        output_str = regex.sub(f"DELAY_MODEL_RESOLUTION = {resolution}", fh.read())
    with open(file_name, "w") as output:
        output.write(output_str)


# TODO SIM-719 pass arguments instead of hacky file editing
def enable_propagate_partial(enable: bool = True):
    file_name = "rsp/utils/global_constants.py"
    with open(file_name, "r") as fh:
        regex = re.compile("DL_PROPAGATE_PARTIAL = .*")
        output_str = regex.sub(f"DL_PROPAGATE_PARTIAL = True" if enable else f"DL_PROPAGATE_PARTIAL = False", fh.read())
    with open(file_name, "w") as output:
        output.write(output_str)


def set_defaults():
    enable_seq(False)
    set_delay_model_resolution(1)
    enable_propagate_partial(enable=True)


def run_potassco_agenda(
        base_directory: str,
        experiment_output_base_directory: Optional[str] = None,
        experiment_filter=None,
        parallel_compute: int = 5
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
        weight_lateness_seconds=[1, 1, 1]
    )

    try:
        experiments_per_grid_element = 1
        experiment_name_prefix = os.path.basename(base_directory) + "_"
        parallel_compute = parallel_compute
        # baseline with defaults
        set_defaults()
        list_from_base_directory_and_run_experiment_agenda(
            experiment_base_directory=base_directory,
            experiment_output_directory=experiment_output_base_directory,
            reschedule_parameters_range=reschedule_parameters_range,
            experiment_name=('%sbaseline' % experiment_name_prefix),
            parallel_compute=parallel_compute,
            experiments_per_grid_element=experiments_per_grid_element,
            experiment_filter=experiment_filter
        )
        # effect of SEQ heuristic (SIM-167)
        set_defaults()
        enable_seq(True)

        list_from_base_directory_and_run_experiment_agenda(
            experiment_base_directory=base_directory,
            experiment_output_directory=experiment_output_base_directory,
            reschedule_parameters_range=reschedule_parameters_range,
            experiment_name=('%swith_SEQ' % experiment_name_prefix),
            parallel_compute=parallel_compute,
            experiments_per_grid_element=experiments_per_grid_element

        )
        # effect of delay model resolution (SIM-542)
        set_defaults()
        set_delay_model_resolution(2)
        list_from_base_directory_and_run_experiment_agenda(
            experiment_base_directory=base_directory,
            experiment_output_directory=experiment_output_base_directory,
            reschedule_parameters_range=reschedule_parameters_range,
            experiment_name=('%swith_delay_model_resolution_2' % experiment_name_prefix),
            parallel_compute=parallel_compute,
            experiments_per_grid_element=experiments_per_grid_element
        )
        set_defaults()
        set_delay_model_resolution(5)
        list_from_base_directory_and_run_experiment_agenda(
            experiment_base_directory=base_directory,
            experiment_output_directory=experiment_output_base_directory,
            reschedule_parameters_range=reschedule_parameters_range,
            experiment_name=('%swith_delay_model_resolution_5' % experiment_name_prefix),
            parallel_compute=parallel_compute,
            experiments_per_grid_element=experiments_per_grid_element
        )
        set_defaults()
        set_delay_model_resolution(10)
        list_from_base_directory_and_run_experiment_agenda(
            experiment_base_directory=base_directory,
            experiment_output_directory=experiment_output_base_directory,
            reschedule_parameters_range=reschedule_parameters_range,
            experiment_name=('%swith_delay_model_resolution_10' % experiment_name_prefix),
            parallel_compute=parallel_compute,
            experiments_per_grid_element=experiments_per_grid_element
        )
        # # effect of --propagate (SIM-543)
        set_defaults()
        enable_propagate_partial(enable=False)
        list_from_base_directory_and_run_experiment_agenda(
            experiment_base_directory=base_directory,
            experiment_output_directory=experiment_output_base_directory,
            reschedule_parameters_range=reschedule_parameters_range,
            experiment_name=('%swithout_propagate_partial' % experiment_name_prefix),
            parallel_compute=parallel_compute,
            experiments_per_grid_element=experiments_per_grid_element
        )
    finally:
        set_defaults()


def generate_potassco_infras_and_schedules(
        base_directory: Optional[str] = None,
        parallel_compute: int = 5
):
    if base_directory is None:
        base_directory = create_experiment_folder_name("h1")
        check_create_folder(base_directory)

    infra_parameters_range = InfrastructureParametersRange(
        number_of_agents=[80, 150, 8],
        width=[110, 110, 1],
        height=[110, 110, 1],
        flatland_seed_value=[12, 12, 1],
        max_num_cities=[10, 10, 1],
        max_rail_in_city=[3, 3, 1],
        max_rail_between_cities=[1, 1, 1],
        number_of_shortest_paths_per_agent=[10, 10, 1]
    )
    schedule_parameters_range = ScheduleParametersRange(
        asp_seed_value=[94, 104, 10],
        number_of_shortest_paths_per_agent_schedule=[1, 1, 1],
    )

    create_infrastructure_and_schedule_from_ranges(
        base_directory=base_directory,
        infrastructure_parameters_range=infra_parameters_range,
        schedule_parameters_range=schedule_parameters_range,
        speed_data={1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25},  # Slow freight train
        grid_mode=False,
        run_experiments_parallel=parallel_compute
    )
    return base_directory


def experiment_filter_first_ten_of_each_schedule(experiment: ExperimentParameters):
    return experiment.malfunction_agent_id < 10 and experiment.experiment_id > 2000


if __name__ == '__main__':
    parallel_compute = AVAILABLE_CPUS // 2
    generate_potassco_infras_and_schedules(
        base_directory=INFRAS_AND_SCHEDULES_FOLDER,
        parallel_compute=parallel_compute
    )
    run_potassco_agenda(
        base_directory=INFRAS_AND_SCHEDULES_FOLDER,
        # incremental re-start after interruption
        experiment_output_base_directory=BASELINE_DATA_FOLDER,
        experiment_filter=experiment_filter_first_ten_of_each_schedule,
        parallel_compute=parallel_compute
    )
