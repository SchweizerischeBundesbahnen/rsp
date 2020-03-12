import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from matplotlib import pyplot as plt

from rsp.utils.data_types import TrainSchedule
from rsp.utils.data_types import TrainScheduleDict


def undirected_distance_between_trains(train_schedule_0: TrainSchedule, train_run_0: Trainrun,
                                       train_schedule_1: TrainSchedule, train_run_1: Trainrun) -> (float, int):
    train_0_start_time = train_run_0[0].scheduled_at
    train_0_end_time = train_run_0[-1].scheduled_at
    train_1_start_time = train_run_1[0].scheduled_at
    train_1_end_time = train_run_1[-1].scheduled_at

    # if the time window of the two trains do not overlap -> no relationship between trains
    if train_0_end_time < train_1_start_time or train_1_end_time < train_0_start_time:
        return 0, 0, 0, 0

    # some timesteps overlap -> find out which ones
    if train_0_start_time <= train_1_start_time:
        start_time_step = train_1_start_time
    else:
        start_time_step = train_0_start_time

    if train_0_end_time <= train_1_end_time:
        end_time_step = train_0_end_time
    else:
        end_time_step = train_1_end_time

    # get positions in time window
    train_0_positions = [waypoint.position for i, waypoint in train_schedule_0.items() if
                         (start_time_step <= i <= end_time_step)]
    train_1_positions = [waypoint.position for i, waypoint in train_schedule_1.items() if
                         (start_time_step <= i <= end_time_step)]

    # compute distances of positions in time window
    distances_in_time_window = np.zeros((1, len(train_0_positions)))
    for i, pos in enumerate(train_0_positions):
        distances_in_time_window[0, i] = np.sqrt(
            np.sum(np.square([pos[0] - train_1_positions[i][0], pos[1] - train_1_positions[i][1]])))

    # first heuristic -> get the smallest distance
    dist_between_trains = np.min(distances_in_time_window)
    index_min_dist = np.argmin(distances_in_time_window)
    time_of_min = start_time_step + index_min_dist
    train_0_position_at_min = train_0_positions[int(index_min_dist)]
    train_1_position_at_min = train_1_positions[int(index_min_dist)]

    return 1. / dist_between_trains, time_of_min, train_0_position_at_min, train_1_position_at_min


def compute_undirected_distance_matrix(trainrun_dict: TrainrunDict,
                                       train_schedule_dict: TrainScheduleDict) -> np.ndarray:
    number_of_trains = len(trainrun_dict)
    distance_matrix = np.zeros((number_of_trains, number_of_trains))

    additional_info = {}
    for row, train_schedule_row in train_schedule_dict.items():
        for column, train_schedule_column in train_schedule_dict.items():
            if column > row:
                train_run_row = trainrun_dict.get(row)
                train_run_column = trainrun_dict.get(column)
                distance, time_of_min, train_0_position_at_min, train_1_position_at_min = \
                    undirected_distance_between_trains(
                        train_schedule_row, train_run_row,
                        train_schedule_column, train_run_column)
                distance_matrix[row, column] = distance
                distance_matrix[column, row] = distance
                additional_info[(row, column)] = (
                    distance, time_of_min, train_0_position_at_min, train_1_position_at_min)
    return distance_matrix, additional_info


def plot_encounter_graph_undirected(distance_matrix: np.ndarray, title: str, file_name: str, pos: dict = None):
    dt = [('weight', float)]
    distance_matrix_as_weight = np.matrix(distance_matrix, dtype=dt)
    graph = nx.from_numpy_matrix(distance_matrix_as_weight)

    # position of nodes
    if pos is None:
        # Position nodes using Fruchterman-Reingold force-directed algorithm
        # https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html
        pos = nx.spring_layout(graph, seed=42)

    fig = plt.figure(figsize=(18, 12), dpi=80)
    fig.suptitle(title, fontsize=16)
    plt.subplot(121)

    # draw nodes
    nx.draw_networkx_nodes(graph, pos)

    # draw edges with corresponding weights
    for edge in graph.edges(data=True):
        nx.draw_networkx_edges(graph, pos, edgelist=[edge], width=edge[2]['weight'] * 5.)

    # draw labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family='sans-serif')

    # visualize distance matrix as heat plot
    plt.subplot(122)
    plt.imshow(distance_matrix, cmap='hot', interpolation='nearest')
    fig.savefig(file_name)
    plt.close(fig)

    return pos
