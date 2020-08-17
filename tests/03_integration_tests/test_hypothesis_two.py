from typing import List

from rsp.hypothesis_two_encounter_graph import extract_time_windows_and_transmission_chains
from rsp.hypothesis_two_encounter_graph import plot_transmission_chains_time_window
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.experiments import EXPERIMENT_AGENDA_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.utils.experiments import load_experiment_result_without_expanding
from rsp.utils.experiments import load_malfunction
from rsp.utils.experiments import load_schedule


def test_hypothesis_two(re_save: bool = False):
    """Run hypothesis two."""
    experiment_base_directory = './tests/03_integration_tests/mini_toy_example'
    experiment_agenda_directory = f'{experiment_base_directory}/{EXPERIMENT_AGENDA_SUBDIRECTORY_NAME}'
    experiment_data_directory = f'{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}'
    experiment_id = 0

    # used if module path used in pickle has changed
    # use with wrapper file https://stackoverflow.com/questions/13398462/unpickling-python-objects-with-a-changed-module-path
    if re_save:
        load_experiment_result_without_expanding(experiment_data_folder_name=experiment_data_directory, experiment_id=experiment_id, re_save=True)
        load_schedule(base_directory=experiment_agenda_directory, infra_id=experiment_id, re_save=True)
        load_malfunction(base_directory=experiment_agenda_directory, experiment_id=experiment_id, re_save=True)

    experiment_results_list: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_data_folder(
        experiment_data_folder_name=experiment_data_directory,
        experiment_ids=[0])
    experiment_result = experiment_results_list[0]
    transmission_chains_time_window = extract_time_windows_and_transmission_chains(experiment_result=experiment_result)
    plot_transmission_chains_time_window(experiment_result, transmission_chains_time_window)
