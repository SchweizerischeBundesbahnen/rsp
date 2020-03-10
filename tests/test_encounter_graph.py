from typing import List

import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.experiment_solvers.experiment_solver import asp_schedule_wrapper
from rsp.route_dag.generators.route_dag_generator_schedule import schedule_problem_description_from_rail_env
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.encounter_graph import compute_undirected_distance_matrix
from rsp.utils.encounter_graph import plot_encounter_graph_undirected
from rsp.utils.experiments import create_env_pair_for_experiment
from rsp.utils.experiments import load_and_expand_experiment_results_from_folder
from rsp.utils.flatland_replay_utils import convert_trainrundict_to_entering_positions_for_all_timesteps


def create_simple_env_and_schedule():
    test_parameters = ExperimentParameters(experiment_id=0, grid_id=0,
                                           number_of_agents=10, width=30,
                                           height=30, flatland_seed_value=12, asp_seed_value=94, max_num_cities=20,
                                           grid_mode=False,
                                           max_rail_between_cities=2, max_rail_in_city=6, earliest_malfunction=20,
                                           malfunction_duration=20, speed_data={1.0: 0.25, 0.5: 0.25, 0.3333333333333333: 0.25, 0.25: 0.25},
                                           number_of_shortest_paths_per_agent=10,
                                           weight_route_change=60,
                                           weight_lateness_seconds=1,
                                           max_window_size_from_earliest=np.inf)
    static_env, dynamic_env = create_env_pair_for_experiment(params=test_parameters)

    tc_schedule_problem = schedule_problem_description_from_rail_env(static_env, 10)
    schedule_result = asp_schedule_wrapper(schedule_problem_description=tc_schedule_problem, static_rail_env=static_env)
    schedule_trainruns: TrainrunDict = schedule_result.trainruns_dict
    return schedule_result, schedule_trainruns


def test_simple_env_encounter_graph():
    schedule_result, schedule_trainruns = create_simple_env_and_schedule()
    train_schedule_dict = convert_trainrundict_to_entering_positions_for_all_timesteps(schedule_trainruns)
    distance_matrix = compute_undirected_distance_matrix(schedule_trainruns, train_schedule_dict)
    plot_encounter_graph_undirected(distance_matrix)

    print("distance matrix")
    print(distance_matrix)


def test_encounter_graph_undirected():
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
    # todo: test with directed graphs and distance measure

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

        edge_weights_full, pos = plot_encounter_graph_undirected(
            distance_matrix=distance_matrix_full,
            title="encounter graph initial schedule",
            file_name=file_name_base+"encounter_graph_initial_schedule.png")

        edge_weights_full_after_malfunction, pos = plot_encounter_graph_undirected(
            distance_matrix=distance_matrix_full_after_malfunction,
            title="encounter graph schedule after malfunction",
            file_name=file_name_base + "encounter_graph_schedule_after_malfunction.png",
            pos=pos)

        edge_weights_diff, pos = plot_encounter_graph_undirected(
            distance_matrix=distance_matrix_diff,
            title="encounter graph difference",
            file_name=file_name_base + "encounter_graph_difference.png",
            pos=pos)
