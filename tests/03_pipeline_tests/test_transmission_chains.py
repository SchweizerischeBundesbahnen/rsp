import os
from typing import List

from rsp.hypothesis_one_pipeline_all_in_one import get_agenda_pipeline_params_001_simple_setting
from rsp.hypothesis_two_encounter_graph import extract_time_windows_and_transmission_chains
from rsp.hypothesis_two_encounter_graph import plot_transmission_chains_time_window
from rsp.step_01_planning.experiment_parameters_and_ranges import ExperimentAgenda
from rsp.step_03_run.experiment_results_analysis import ExperimentResultsAnalysis
from rsp.step_03_run.experiments import create_experiment_agenda_from_parameter_ranges_and_speed_data
from rsp.step_03_run.experiments import create_experiment_folder_name
from rsp.step_03_run.experiments import delete_experiment_folder
from rsp.step_03_run.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.step_03_run.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.step_03_run.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.step_03_run.experiments import run_experiment_agenda


def test_hypothesis_two():
    """Run hypothesis two."""
    experiment_base_directory = "./tests/03_pipeline_tests/mini_toy_example"

    experiment_agenda: ExperimentAgenda = create_experiment_agenda_from_parameter_ranges_and_speed_data(
        experiment_name="test_hypothesis_two", parameter_ranges_and_speed_data=get_agenda_pipeline_params_001_simple_setting()
    )
    experiment_folder_name = create_experiment_folder_name(experiment_agenda.experiment_name)
    try:
        experiment_output_directory = run_experiment_agenda(
            experiment_agenda=experiment_agenda, experiment_base_directory=experiment_base_directory, experiment_output_directory=experiment_folder_name
        )

        experiment_results_list: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_data_folder(
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
