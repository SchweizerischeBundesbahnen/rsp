import os

import numpy as np

from rsp.global_data_configuration import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.global_data_configuration import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ExperimentAgenda
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ExperimentParameters
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import InfrastructureParameters
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ReScheduleParameters
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ScheduleParameters
from rsp.step_01_agenda_expansion.global_constants import get_defaults
from rsp.step_05_experiment_run.experiment_results_analysis_load_and_save import load_and_expand_experiment_results_from_data_folder
from rsp.step_05_experiment_run.experiment_run import create_experiment_folder_name
from rsp.step_05_experiment_run.experiment_run import delete_experiment_folder
from rsp.step_05_experiment_run.experiment_run import run_experiment_agenda
from rsp.step_06_analysis.malfunction_analysis.disturbance_propagation import extract_time_windows_and_transmission_chains
from rsp.step_06_analysis.malfunction_analysis.disturbance_propagation import plot_transmission_chains_time_window


def test_hypothesis_two():
    """Run hypothesis two."""
    experiment_base_directory = "./tests/03_pipeline_tests/mini_toy_example"

    experiment_agenda = ExperimentAgenda(
        experiment_name="test_hypothesis_two",
        global_constants=get_defaults(),
        experiments=[
            ExperimentParameters(
                experiment_id=0,
                grid_id=0,
                infra_id_schedule_id=0,
                infra_parameters=InfrastructureParameters(
                    infra_id=0,
                    width=18,
                    height=18,
                    number_of_agents=2,
                    flatland_seed_value=12,
                    max_num_cities=2,
                    grid_mode=True,
                    max_rail_between_cities=1,
                    max_rail_in_city=2,
                    speed_data={
                        1.0: 0.25,  # Fast passenger train
                        1.0 / 2.0: 0.25,  # Fast freight train
                        1.0 / 3.0: 0.25,  # Slow commuter train
                        1.0 / 4.0: 0.25,
                    },  # Slow freight train
                    number_of_shortest_paths_per_agent=10,
                ),
                schedule_parameters=ScheduleParameters(infra_id=0, schedule_id=0, asp_seed_value=94, number_of_shortest_paths_per_agent_schedule=1),
                re_schedule_parameters=ReScheduleParameters(
                    earliest_malfunction=5,
                    malfunction_duration=20,
                    malfunction_agent_id=0,
                    weight_route_change=1,
                    weight_lateness_seconds=1,
                    max_window_size_from_earliest=np.inf,
                    number_of_shortest_paths_per_agent=10,
                    asp_seed_value=94,
                ),
            )
        ],
    )

    experiment_folder_name = "target/" + create_experiment_folder_name(experiment_agenda.experiment_name)
    try:
        experiment_output_directory = run_experiment_agenda(
            experiment_agenda=experiment_agenda, experiment_base_directory=experiment_base_directory, experiment_output_directory=experiment_folder_name
        )

        experiment_results_list, _ = load_and_expand_experiment_results_from_data_folder(
            experiment_data_folder_name=os.path.join(experiment_output_directory, EXPERIMENT_DATA_SUBDIRECTORY_NAME), experiment_ids=[0]
        )
        experiment_result = experiment_results_list[0]
        transmission_chains_time_window = extract_time_windows_and_transmission_chains(experiment_result=experiment_result)
        plot_transmission_chains_time_window(
            experiment_result=experiment_result,
            transmission_chains_time_window=transmission_chains_time_window,
            output_folder=os.path.join(experiment_output_directory, EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME),
        )
    finally:
        delete_experiment_folder(experiment_folder_name)
