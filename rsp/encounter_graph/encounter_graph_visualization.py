import os
from typing import List
from typing import Optional
from typing import Tuple

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from rsp.encounter_graph.encounter_graph import compute_symmetric_distance_matrix
from rsp.route_dag.route_dag import ScheduleProblemEnum
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.flatland_replay_utils import convert_trainrundict_to_entering_positions_for_all_timesteps


def _plot_encounter_graph_directed(weights_matrix: np.ndarray,
                                   title: str,
                                   file_name: Optional[str] = None,
                                   pos: Optional[dict] = None,
                                   highlights: Optional[dict] = None):
    """This method plots the encounter graph and the heatmap of the distance
    matrix into one file.

    Parameters
    ----------
    weights_matrix
        matrix of weights to be rendered as encounter graph
    title
        title of plot
    file_name
        string of filename if saving is required
    pos [Optional]
        fixed positions of nodes in encountergraph
    highlights [Optional]
        dict containing the nodes that need to be highlighted

    Returns
    -------
        dict containing the positions of the nodes
    """
    dt = [('weight', float)]
    distance_matrix_as_weight = np.copy(weights_matrix)
    distance_matrix_as_weight.dtype = dt

    graph = nx.from_numpy_array(distance_matrix_as_weight, create_using=nx.DiGraph)
    print(f"nb edges={len(graph.edges)}, nodes={graph.number_of_nodes()}, "
          f"expected nb of edges={graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2} "
          "(in diff matrix, <= is ok since the zeros are those without change)")




    # # position of nodes
    # if pos is None:
    #     # Position nodes using Fruchterman-Reingold force-directed algorithm
    #     # https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html
    #     pos = nx.spring_layout(graph, seed=42)
    # else:
    #     fixed_nodes = pos.keys()
    #     pos = nx.spring_layout(graph, seed=42, pos=pos, fixed=fixed_nodes)

    # Color the nodes
    node_color = ['lightblue' for i in range(graph.number_of_nodes())]
    if highlights is not None:
        for node_idx in highlights:
            if highlights[node_idx]:
                node_color[node_idx] = 'r'

    fig = plt.figure(figsize=(18, 12), dpi=80)
    fig.suptitle(title, fontsize=16)
    plt.subplot(121)

    # draw nodes
    plt.gca().invert_yaxis()
    nx.draw_networkx_nodes(graph, pos, node_color=node_color)

    # draw edges with corresponding weights
    for edge_with_data in graph.edges(data=True):
        edge_weight = edge_with_data[2]['weight']
        edge = edge_with_data[:2]
        # TODO could we use colors instead? same as from heatmap below?
        nx.draw_networkx_edges(graph, pos, edgelist=[edge], width=edge_weight)

    # draw labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family='sans-serif')

    # visualize distance matrix as heat plot
    plt.subplot(122)

    plt.imshow(weights_matrix, cmap='hot', interpolation='nearest')


    if file_name is not None:
        fig.savefig(file_name)
        plt.close(fig)
    else:
        plt.show()

    return pos


def _plot_encounter_graph_undirected(distance_matrix: np.ndarray,
                                     title: str,
                                     file_name: Optional[str],
                                     pos: Optional[dict] = None,
                                     highlights: Optional[dict] = None):
    """This method plots the encounter graph and the heatmap of the distance
    matrix into one file.

    Parameters
    ----------
    distance_matrix
        matrix to be rendered as encounter graph
    title
        title of plot
    file_name
        string of filename if saving is required
    pos [Optional]
        fixed positions of nodes in encountergraph
    highlights [Optional]
        dict containing the nodes that need to be highlighted

    Returns
    -------
        dict containing the positions of the nodes
    """
    dt = [('weight', float)]
    distance_matrix_as_weight = np.copy(distance_matrix)
    distance_matrix_as_weight.dtype = dt

    graph = nx.from_numpy_array(distance_matrix_as_weight)
    print(f"nb edges={len(graph.edges)}, nodes={graph.number_of_nodes()}, "
          f"expected nb of edges={graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2} "
          "(in diff matrix, <= is ok since the zeros are those without change)")

    # position of nodes
    if pos is None:
        # Position nodes using Fruchterman-Reingold force-directed algorithm
        # https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html
        pos = nx.spring_layout(graph, seed=42)
    else:
        fixed_nodes = pos.keys()
        pos = nx.spring_layout(graph, seed=42, pos=pos, fixed=fixed_nodes)

    # Color the nodes
    node_color = ['lightblue' for i in range(graph.number_of_nodes())]
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
    for edge_with_data in graph.edges(data=True):
        edge_weight = edge_with_data[2]['weight']
        edge = edge_with_data[:2]
        # TODO could we use colors instead? same as from heatmap below?
        nx.draw_networkx_edges(graph, pos, edgelist=[edge], width=edge_weight)

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
        debug_pair: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Parameters
    ----------

    experiment_result
        Experiment Data to be used to generate encountergraphs
    pos
        Fixed positions for the nodes in the encountergraph
    highlighted_nodes
        Dict containing nodes that need to be highlighted
    encounter_graph_folder
        Folder to store encounter graphs
    metric_function
        Custom metric function to determine distance between nodes
    debug_pair
    Returns
    -------
    """
    print("plot_encounter_graphs_for_experiment_result")
    trainrun_dict_full = experiment_result.solution_full
    trainrun_dict_full_after_malfunction = experiment_result.solution_full_after_malfunction
    train_schedule_dict_full = convert_trainrundict_to_entering_positions_for_all_timesteps(trainrun_dict_full)
    train_schedule_dict_full_after_malfunction = convert_trainrundict_to_entering_positions_for_all_timesteps(
        trainrun_dict_full_after_malfunction)
    print("schedule: compute_undirected_distance_matrix")
    distance_matrix_full = compute_symmetric_distance_matrix(
        trainrun_dict=trainrun_dict_full,
        schedule_problem_description=experiment_result.problem_full,
        train_schedule_dict=train_schedule_dict_full,
        metric_function=metric_function)
    print(distance_matrix_full)
    if debug_pair is not None:
        print(distance_matrix_full[debug_pair[0], debug_pair[1]])
    print("re-schedule: compute_undirected_distance_matrix")

    distance_matrix_full_after_malfunction = compute_symmetric_distance_matrix(
        trainrun_dict=trainrun_dict_full_after_malfunction,
        schedule_problem_description=experiment_result.problem_full_after_malfunction,
        train_schedule_dict=train_schedule_dict_full_after_malfunction,
        metric_function=metric_function)
    print(distance_matrix_full_after_malfunction)
    if debug_pair is not None:
        print(distance_matrix_full_after_malfunction[debug_pair[0], debug_pair[1]])
    distance_matrix_diff = np.abs(distance_matrix_full_after_malfunction - distance_matrix_full)
    print(distance_matrix_diff)
    if debug_pair is not None:
        print(distance_matrix_diff[debug_pair[0], debug_pair[1]])

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
        pos = _plot_encounter_graph_undirected(distance_matrix=distance_matrices[schedule_problem_to_visualize],
                                               title=titles[schedule_problem_to_visualize], file_name=(
                file_names[schedule_problem_to_visualize] if encounter_graph_folder is not None else None), pos=pos,
                                               highlights=highlighted_nodes)

    return distance_matrix_full, distance_matrix_full_after_malfunction, distance_matrix_diff
