from typing import List

from rsp.encounter_graph.encounter_graph_visualization import plot_encounter_graphs_for_experiment_result
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.utils.file_utils import check_create_folder


def hypothesis_two_encounter_graph_undirected(experiment_base_directory: str,
                                              experiment_ids: List[int] = None):
    """This method computes the encounter graphs of the specified experiments.
    Within this first approach, the distance measure within the encounter
    graphs is undirected.

    Parameters
    ----------
    experiment_base_directory
    experiment_ids
    """
    experiment_analysis_directory = f'{experiment_base_directory}/{EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME}/'

    experiment_results_list: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_data_folder(
        experiment_data_folder_name=experiment_base_directory,
        experiment_ids=experiment_ids)

    for i in list(range(len(experiment_results_list))):
        experiment_output_folder = f"{experiment_analysis_directory}/experiment_{experiment_ids[i]:04d}_analysis"
        encounter_graph_folder = f"{experiment_output_folder}/encounter_graphs"

        # Check and create the folders
        check_create_folder(experiment_output_folder)
        check_create_folder(encounter_graph_folder)

        experiment_result = experiment_results_list[i]

        plot_encounter_graphs_for_experiment_result(experiment_result=experiment_result,
                                                    encounter_graph_folder=encounter_graph_folder)


if __name__ == '__main__':
    hypothesis_two_encounter_graph_undirected(experiment_base_directory='./exp_hypothesis_one_2020_03_04T19_19_00',
                                              experiment_ids=[12, 13])
