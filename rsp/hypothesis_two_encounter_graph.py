from typing import List

import numpy as np

from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.encounter_graph import compute_undirected_distance_matrix
from rsp.utils.encounter_graph import plot_encounter_graph_undirected
from rsp.utils.experiments import load_and_expand_experiment_results_from_folder
from rsp.utils.flatland_replay_utils import convert_trainrundict_to_entering_positions_for_all_timesteps


def compute_encounter_graph_undirected():
    data_folder = './../rsp/exp_hypothesis_one_2020_03_04T19_19_00'
    encounter_graph_exp_name = "encounter_graphs_undirected"

    import os
    directory = "{}/{}/".format(data_folder, encounter_graph_exp_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    experiment_results_list: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_folder(
        experiment_folder_name=data_folder,
        experiment_ids=[12])
    print("results loaded")

    # todo: compare with wegzeit diagram
    # todo: test with directed graphs and other distance measure

    for exp_id in list(range(len(experiment_results_list))):
        experiment_result = experiment_results_list[exp_id]
        trainrun_dict_full = experiment_result.solution_full
        print(trainrun_dict_full)
        trainrun_dict_full_after_malfunction = experiment_result.solution_full_after_malfunction

        train_schedule_dict_full = convert_trainrundict_to_entering_positions_for_all_timesteps(trainrun_dict_full)
        train_schedule_dict_full_after_malfunction = convert_trainrundict_to_entering_positions_for_all_timesteps(
            trainrun_dict_full_after_malfunction)

        distance_matrix_full = compute_undirected_distance_matrix(trainrun_dict_full, train_schedule_dict_full)
        distance_matrix_full_after_malfunction = compute_undirected_distance_matrix(
            trainrun_dict_full_after_malfunction,
            train_schedule_dict_full_after_malfunction)
        distance_matrix_diff = np.abs(distance_matrix_full_after_malfunction - distance_matrix_full)

        file_name_base = "{}/{}/experiment_{}_".format(
            data_folder,
            encounter_graph_exp_name,
            experiment_result.experiment_id)

        pos = plot_encounter_graph_undirected(
            distance_matrix=distance_matrix_full,
            title="encounter graph initial schedule",
            file_name=file_name_base+"encounter_graph_initial_schedule.png")

        pos = plot_encounter_graph_undirected(
            distance_matrix=distance_matrix_full_after_malfunction,
            title="encounter graph schedule after malfunction",
            file_name=file_name_base + "encounter_graph_schedule_after_malfunction.png",
            pos=pos)

        pos = plot_encounter_graph_undirected(
            distance_matrix=distance_matrix_diff,
            title="encounter graph difference",
            file_name=file_name_base + "encounter_graph_difference.png",
            pos=pos)


if __name__ == '__main__':
    compute_encounter_graph_undirected()
