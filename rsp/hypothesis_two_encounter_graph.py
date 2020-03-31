import os
from typing import List

import numpy as np

from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.encounter_graph import compute_undirected_distance_matrix
from rsp.utils.encounter_graph import plot_encounter_graph_undirected
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.utils.file_utils import check_create_folder
from rsp.utils.flatland_replay_utils import convert_trainrundict_to_entering_positions_for_all_timesteps


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
        trainrun_dict_full = experiment_result.solution_full
        trainrun_dict_full_after_malfunction = experiment_result.solution_full_after_malfunction

        train_schedule_dict_full = convert_trainrundict_to_entering_positions_for_all_timesteps(trainrun_dict_full)
        train_schedule_dict_full_after_malfunction = convert_trainrundict_to_entering_positions_for_all_timesteps(
            trainrun_dict_full_after_malfunction)

        distance_matrix_full, additional_info = compute_undirected_distance_matrix(trainrun_dict_full,
                                                                                   train_schedule_dict_full)
        distance_matrix_full_after_malfunction, additional_info_after_malfunction = compute_undirected_distance_matrix(
            trainrun_dict_full_after_malfunction,
            train_schedule_dict_full_after_malfunction)
        distance_matrix_diff = np.abs(distance_matrix_full_after_malfunction - distance_matrix_full)

        pos = plot_encounter_graph_undirected(
            distance_matrix=distance_matrix_full,
            title="encounter graph initial schedule",
            file_name=os.path.join(encounter_graph_folder, f"encounter_graph_initial_schedule.png")
        )

        plot_encounter_graph_undirected(
            distance_matrix=distance_matrix_full_after_malfunction,
            title="encounter graph schedule after malfunction",
            file_name=os.path.join(encounter_graph_folder, f"encounter_graph_schedule_after_malfunction.png"),
            pos=pos)

        plot_encounter_graph_undirected(
            distance_matrix=distance_matrix_diff,
            title="encounter graph difference",
            file_name=os.path.join(encounter_graph_folder, f"encounter_graph_difference.png"),
            pos=pos)


if __name__ == '__main__':
    hypothesis_two_encounter_graph_undirected(experiment_base_directory='./exp_hypothesis_one_2020_03_04T19_19_00',
                                              experiment_ids=[12, 13])
