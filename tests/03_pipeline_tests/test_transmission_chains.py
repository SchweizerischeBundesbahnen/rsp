import os

import numpy as np
from rsp.step_01_planning.experiment_parameters_and_ranges import ExperimentAgenda
from rsp.step_01_planning.experiment_parameters_and_ranges import ParameterRanges
from rsp.step_01_planning.experiment_parameters_and_ranges import ParameterRangesAndSpeedData
from rsp.step_03_run.experiments import create_experiment_agenda_from_parameter_ranges_and_speed_data
from rsp.step_03_run.experiments import create_experiment_folder_name
from rsp.step_03_run.experiments import delete_experiment_folder
from rsp.step_03_run.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.step_03_run.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.step_03_run.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.step_03_run.experiments import run_experiment_agenda
from rsp.step_04_analysis.malfunction_analysis.disturbance_propagation import extract_time_windows_and_transmission_chains
from rsp.step_04_analysis.malfunction_analysis.disturbance_propagation import plot_transmission_chains_time_window


def get_agenda_pipeline_params_001_simple_setting() -> ParameterRangesAndSpeedData:
    parameter_ranges = ParameterRanges(
        agent_range=[2, 2, 1],
        size_range=[18, 18, 1],
        in_city_rail_range=[2, 2, 1],
        out_city_rail_range=[1, 1, 1],
        city_range=[2, 2, 1],
        earliest_malfunction=[5, 5, 1],
        malfunction_duration=[20, 20, 1],
        number_of_shortest_paths_per_agent=[10, 10, 1],
        max_window_size_from_earliest=[np.inf, np.inf, 1],
        asp_seed_value=[94, 94, 1],
        # route change is penalized the same as 60 seconds delay
        weight_route_change=[60, 60, 1],
        weight_lateness_seconds=[1, 1, 1],
    )
    # Define the desired speed profiles
    speed_data = {
        1.0: 0.25,  # Fast passenger train
        1.0 / 2.0: 0.25,  # Fast freight train
        1.0 / 3.0: 0.25,  # Slow commuter train
        1.0 / 4.0: 0.25,
    }  # Slow freight train
    return ParameterRangesAndSpeedData(parameter_ranges=parameter_ranges, speed_data=speed_data)


def test_hypothesis_two():
    """Run hypothesis two."""
    experiment_base_directory = "./tests/03_pipeline_tests/mini_toy_example"

    experiment_agenda: ExperimentAgenda = create_experiment_agenda_from_parameter_ranges_and_speed_data(
        experiment_name="test_hypothesis_two", parameter_ranges_and_speed_data=get_agenda_pipeline_params_001_simple_setting()
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
