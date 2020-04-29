from typing import Dict

import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.route_dag.route_dag import ScheduleProblemDescription, RouteDAGConstraints
from rsp.utils.data_types import TrainSchedule
from rsp.utils.data_types import TrainScheduleDict
from rsp.utils.data_types import UndirectedEncounterGraphDistance



def undirected_distance_between_trains(train_schedule_0: TrainSchedule,
                                       train_run_0: Trainrun,
                                       constraints_0: RouteDAGConstraints,
                                       train_schedule_1: TrainSchedule,
                                       train_run_1: Trainrun,
                                       constraints_1: RouteDAGConstraints
                                       ) -> UndirectedEncounterGraphDistance:
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

    train_0_earliest = {wp: np.inf for wp in train_0_positions}
    train_0_latest = {wp: -np.inf for wp in train_0_positions}
    train_1_earliest = {wp: np.inf for wp in train_1_positions}
    train_1_latest = {wp: -np.inf for wp in train_1_positions}

    for wp, scheduled_at in constraints_0.freeze_earliest.items():
        cell = wp.position
        if cell in train_0_earliest:
            train_0_earliest[cell] = min(train_0_earliest[cell], scheduled_at)
    for wp, scheduled_at in constraints_0.freeze_latest.items():
        cell = wp.position
        if cell in train_0_latest:
            train_0_latest[cell] = max(train_0_latest[cell], scheduled_at)

    for wp, scheduled_at in constraints_1.freeze_earliest.items():
        cell = wp.position
        if cell in train_1_earliest:
            train_1_earliest[cell] = min(train_1_earliest[cell], scheduled_at)
    for wp, scheduled_at in constraints_1.freeze_latest.items():
        cell = wp.position
        if cell in train_1_latest:
            train_1_latest[cell] = max(train_1_latest[cell], scheduled_at)
    # print(train_0_earliest)
    # print(train_0_latest)
    # print(train_1_earliest)
    # print(train_1_latest)

    # compute distances of positions in time window
    overlap_lengths = [overlaps((train_0_earliest[waypoint.position], train_0_latest[waypoint.position]), (train_1_earliest[waypoint.position], train_1_latest[waypoint.position]))
                       for waypoint, earliest_0 in train_0_earliest if waypoint in train_1_earliest.keys()]
    # print(f"overlap_lengths={overlap_lengths}")
    overlap_intervals = [
        overlap_interval((train_0_earliest[cell], train_0_latest[cell]), (train_1_earliest[cell], train_1_latest[cell]))
        for cell, earliest_0 in train_0_earliest.items() if cell in train_1_earliest.keys()]
    print(f"overlap_intervals={overlap_intervals}")
    overlap_starts = [interval[0] for interval in overlap_intervals if interval is not None]
    # print(f"overlap_starts={overlap_starts}")
    if len(overlap_starts) == 0:
        return UndirectedEncounterGraphDistance(inverted_distance=0,
                                                time_of_min=0,
                                                train_0_position_at_min=0,
                                                train_1_position_at_min=0)

    # heuristic: sum of time window overlaps of resources in schedule
    index_min_dist = np.min(overlap_starts)
    dist_between_trains = np.sum(overlap_lengths)
    distance = UndirectedEncounterGraphDistance(inverted_distance=(1. / dist_between_trains),
                                                time_of_min=(start_time_step + index_min_dist),
                                                train_0_position_at_min=train_0_positions[min(int(index_min_dist), len(train_0_positions)-1)],
                                                train_1_position_at_min=train_1_positions[min(int(index_min_dist), len(train_1_positions)-1)])

    return distance


def overlap_interval(interval1, interval2):
    """
    Given [0, 4] and [1, 10] returns [1, 4]
    """
    if interval2[0] <= interval1[0] <= interval2[1]:
        start = interval1[0]
    elif interval1[0] <= interval2[0] <= interval1[1]:
        start = interval2[0]
    else:
        return None

    if interval2[0] <= interval1[1] <= interval2[1]:
        end = interval1[1]
    elif interval1[0] <= interval2[1] <= interval1[1]:
        end = interval2[1]
    else:
        return None

    if start > -np.inf and end < np.inf:

        return (start, end)
    return None


def overlaps(a, b):
    """
    Return the amount of overlap, in bp
    between a and b.
    If >0, the number of bp of overlap
    If 0,  they are book-ended.
    If <0, the distance in bp between them
    """

    return min(a[1], b[1]) - max(a[0], b[0])


def compute_undirected_distance_matrix(trainrun_dict: TrainrunDict,
                                       schedule_problem_description: ScheduleProblemDescription,
                                       train_schedule_dict: TrainScheduleDict,

                                       ) -> (np.ndarray, Dict):
    """This method computes the distance matrix for a complete TrainrunDict ->
    each distance between each pair of trains is computed.

    Parameters
    ----------
    trainrun_dict
    train_schedule_dict

    Returns
    -------
    distance_matrix
        the distance matrix as a symmetric matrix each entry corresponds to a pair of trains
    additional_info
        a dictionary with additional info like the time step at which the minimal distance happened and the location of
        the trains
    """
    print("blabla")

    number_of_trains = len(trainrun_dict)
    distance_matrix = np.zeros((number_of_trains, number_of_trains))

    additional_info = {}
    for row, train_schedule_row in train_schedule_dict.items():
        for column, train_schedule_column in train_schedule_dict.items():
            if column > row:
                train_run_row = trainrun_dict.get(row)
                train_run_column = trainrun_dict.get(column)
                undirected_distance = undirected_distance_between_trains(
                    train_schedule_row, train_run_row, schedule_problem_description.route_dag_constraints_dict[row],
                    train_schedule_column, train_run_column, schedule_problem_description.route_dag_constraints_dict[column])
                print(f"{column} - {row}: {undirected_distance.inverted_distance}")
                distance_matrix[row, column] = undirected_distance.inverted_distance
                distance_matrix[column, row] = undirected_distance.inverted_distance
                additional_info[(row, column)] = undirected_distance
    return distance_matrix, additional_info
