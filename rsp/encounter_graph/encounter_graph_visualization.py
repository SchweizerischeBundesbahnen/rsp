import os
from typing import List
from typing import Optional

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from rsp.encounter_graph.encounter_graph import compute_undirected_distance_matrix
from rsp.route_dag.route_dag import ScheduleProblemEnum
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.flatland_replay_utils import convert_trainrundict_to_entering_positions_for_all_timesteps


def _plot_encounter_graph_undirected(
        distance_matrix: np.ndarray,
        title: str, file_name: Optional[str],
        pos: dict = None,
        highlights: dict = None,
        fixed_positions: Optional[dict] = None):
    """This method plots the encounter graph and the heatmap of the distance
    matrix into one file.

    Parameters
    ----------
    distance_matrix
    title
    file_name
    pos

    Returns
    -------
    """
    dt = [('weight', float)]
    distance_matrix_as_weight = np.copy(distance_matrix)
    distance_matrix_as_weight.dtype = dt
    graph = nx.from_numpy_matrix(distance_matrix_as_weight)

    # position of nodes
    if pos is None:
        # Position nodes using Fruchterman-Reingold force-directed algorithm
        # https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html
        if fixed_positions is not None:
            fixed_nodes = fixed_positions.keys()
            pos = nx.spring_layout(graph, seed=42, pos=fixed_positions, fixed=fixed_nodes)
        else:
            pos = nx.spring_layout(graph, seed=42)

    # Color the nodes
    node_color = ['b' for i in range(graph.number_of_nodes())]
    if highlights is not None:
        for node_idx in highlights:
            if highlights[node_idx]:
                node_color[node_idx] = 'r'

    fig = plt.figure(figsize=(18, 12), dpi=80)
    fig.suptitle(title, fontsize=16)
    plt.subplot(121)

    # draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color=node_color)

    # draw edges with corresponding weights
    higlighted_edge = []
    for edge in graph.edges(data=True):
        if fixed_positions is not None:
            if fixed_positions.get(edge[0]) is not None or fixed_positions.get(edge[1]) is not None:
                higlighted_edge.append(edge)
            else:
                nx.draw_networkx_edges(graph, pos, edgelist=[edge], width=edge[2]['weight'] * 5.)
        else:
            nx.draw_networkx_edges(graph, pos, edgelist=[edge], width=edge[2]['weight'] * 5.)

    # Highlight the edges and draw them on top
    for edge in higlighted_edge:
        nx.draw_networkx_edges(graph, pos, edgelist=[edge], edge_color='r', width=edge[2]['weight'] * 5.)

    # draw labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family='sans-serif')

    # visualize distance matrix as heat plot
    plt.subplot(122)

    plt.imshow(distance_matrix, cmap='hot', interpolation='nearest')

    if file_name is not None:
        fig.savefig(file_name)
        plt.close(fig)

    return pos


def plot_encounter_graphs_for_experiment_result(
        experiment_result: ExperimentResultsAnalysis,
        pos: Optional[dict] = None,
        highlighted_nodes: Optional[dict] = None,
        encounter_graph_folder: Optional[str] = None,
        metric_function: Optional = None,
        fixed_positions: Optional[dict] = None):
    """

    :param experiment_result:
    :param pos:
    :param highlighted_nodes:
    :param encounter_graph_folder:
    :param metric_function:
    :return:
    """
    trainrun_dict_full = experiment_result.solution_full
    trainrun_dict_full_after_malfunction = experiment_result.solution_full_after_malfunction
    train_schedule_dict_full = convert_trainrundict_to_entering_positions_for_all_timesteps(trainrun_dict_full)
    train_schedule_dict_full_after_malfunction = convert_trainrundict_to_entering_positions_for_all_timesteps(
        trainrun_dict_full_after_malfunction)
    distance_matrix_full, additional_info = compute_undirected_distance_matrix(trainrun_dict_full,
                                                                               train_schedule_dict_full,
                                                                               metric_function=metric_function)
    distance_matrix_full_after_malfunction, additional_info_after_malfunction = compute_undirected_distance_matrix(
        trainrun_dict_full_after_malfunction,
        train_schedule_dict_full_after_malfunction, metric_function=metric_function)
    distance_matrix_diff = np.abs(distance_matrix_full_after_malfunction - distance_matrix_full)

    titles = {
        ScheduleProblemEnum.PROBLEM_SCHEDULE: "encounter graph initial schedule (S0)",
        ScheduleProblemEnum.PROBLEM_RSP_FULL: "encounter graph re-schedule full after malfunction (S)",
        ScheduleProblemEnum.PROBLEM_RSP_DELTA: "encounter graph re-schedule delta after malfunction (S')"
    }
    if encounter_graph_folder is not None:
        file_names = {
            ScheduleProblemEnum.PROBLEM_SCHEDULE:
                os.path.join(encounter_graph_folder, f"encounter_graph_initial_schedule.pdf"),
            ScheduleProblemEnum.PROBLEM_RSP_FULL:
                os.path.join(encounter_graph_folder, f"encounter_graph_schedule_after_malfunction.pdf"),
            ScheduleProblemEnum.PROBLEM_RSP_DELTA:
                os.path.join(encounter_graph_folder, f"encounter_graph_schedule_after_malfunction.pdf")
        }
    distance_matrices = {
        ScheduleProblemEnum.PROBLEM_SCHEDULE: distance_matrix_full,
        ScheduleProblemEnum.PROBLEM_RSP_FULL: distance_matrix_full_after_malfunction,
        ScheduleProblemEnum.PROBLEM_RSP_DELTA: distance_matrix_diff
    }
    schedule_problems_to_visualize: List[ScheduleProblemEnum] = [
        ScheduleProblemEnum.PROBLEM_SCHEDULE,
        ScheduleProblemEnum.PROBLEM_RSP_FULL,
        ScheduleProblemEnum.PROBLEM_RSP_DELTA
    ]
    for schedule_problem_to_visualize in schedule_problems_to_visualize:
        _ = _plot_encounter_graph_undirected(
            distance_matrix=distance_matrices[schedule_problem_to_visualize],
            title=titles[schedule_problem_to_visualize],
            file_name=(file_names[schedule_problem_to_visualize] if encounter_graph_folder is not None else None),
            pos=pos,
            highlights=highlighted_nodes,
            fixed_positions=fixed_positions
        )

    return (distance_matrix_full, distance_matrix_full_after_malfunction, distance_matrix_diff)
