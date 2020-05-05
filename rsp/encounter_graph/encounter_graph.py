from typing import Dict

import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.utils.data_types import TrainSchedule
from rsp.utils.data_types import TrainScheduleDict
from rsp.utils.data_types import UndirectedEncounterGraphDistance


def undirected_distance_between_trains(train_schedule_0: TrainSchedule, train_run_0: Trainrun,
                                       train_schedule_1: TrainSchedule,
                                       train_run_1: Trainrun) -> UndirectedEncounterGraphDistance:
    """computes the Euclidian distance between two trains. It computes the
    Euclidian distance at each time step between the position of the two trains
    at this time step.

    Parameters
    ----------
    train_schedule_0
    train_run_0
    train_schedule_1
    train_run_1

    Returns
    -------
    UndirectedEncounterGraphDistance
        contains the data related to the undirected encounter graph distance
    """
    train_0_start_time = train_run_0[0].scheduled_at
    train_0_end_time = train_run_0[-1].scheduled_at
    train_1_start_time = train_run_1[0].scheduled_at
    train_1_end_time = train_run_1[-1].scheduled_at

    # if the time window of the two trains do not overlap -> no relationship between trains
    if train_0_end_time < train_1_start_time or train_1_end_time < train_0_start_time:
        return UndirectedEncounterGraphDistance(inverted_distance=0,
                                                time_of_min=0,
                                                train_0_position_at_min=0,
                                                train_1_position_at_min=0)

    # some timesteps overlap -> find out which ones
    start_time_step = max(train_0_start_time, train_1_start_time)
    end_time_step = min(train_0_end_time, train_1_end_time)

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

    distance = UndirectedEncounterGraphDistance(inverted_distance=(1. / dist_between_trains),
                                                time_of_min=(start_time_step + index_min_dist),
                                                train_0_position_at_min=train_0_positions[int(index_min_dist)],
                                                train_1_position_at_min=train_1_positions[int(index_min_dist)])

    return distance


def compute_undirected_distance_matrix(trainrun_dict: TrainrunDict,
                                       train_schedule_dict: TrainScheduleDict,
                                       metric_function=None) -> (np.ndarray,
                                                                 Dict):
    """This method computes the distance matrix for a complete TrainrunDict ->
    each distance between each pair of trains is computed.

    Parameters
    ----------
    trainrun_dict
        Dictionary containing all the trainruns
    train_schedule_dict
        Dictionary containing the schedules (Visited times and cells of all trains
    metric_function
        Metric function to be used to compute the distance matrix

    Returns
    -------
    distance_matrix
        the distance matrix as a symmetric matrix each entry corresponds to a pair of trains
    additional_info
        a dictionary with additional info like the time step at which the minimal distance happened and the location of
        the trains
    """
    if metric_function is None:
        metric_function = undirected_distance_between_trains
    number_of_trains = len(trainrun_dict)
    distance_matrix = np.zeros((number_of_trains, number_of_trains))

    additional_info = {}
    for row, train_schedule_row in train_schedule_dict.items():
        for column, train_schedule_column in train_schedule_dict.items():
            if column > row:
                train_run_row = trainrun_dict.get(row)
                train_run_column = trainrun_dict.get(column)
                undirected_distance = metric_function(
                    train_schedule_row, train_run_row,
                    train_schedule_column, train_run_column)
                distance_matrix[row, column] = undirected_distance.inverted_distance
                distance_matrix[column, row] = undirected_distance.inverted_distance
                additional_info[(row, column)] = undirected_distance
    return distance_matrix, additional_info
