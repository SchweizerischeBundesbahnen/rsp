from typing import Callable
from typing import Dict
from typing import List

import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.route_dag.route_dag import RouteDAGConstraints
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.utils.data_types import TrainSchedule
from rsp.utils.data_types import TrainScheduleDict
from rsp.utils.data_types import UndirectedEncounterGraphDistance

MetricFunction = Callable[[TrainSchedule,
                           Trainrun,
                           RouteDAGConstraints,
                           TrainSchedule,
                           Trainrun,
                           RouteDAGConstraints,
                           bool], UndirectedEncounterGraphDistance]


def undirected_distance_between_trains(train_schedule_0: TrainSchedule,
                                       train_run_0: Trainrun,
                                       constraints_0: RouteDAGConstraints,
                                       train_schedule_1: TrainSchedule,
                                       train_run_1: Trainrun,
                                       constraints_1: RouteDAGConstraints,
                                       debug: bool = False
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
    constraints_1
        not used
    constraints_0
        not used
    debug


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


def undirected_distance_between_trains_sum_of_time_window_overlaps(
        train_schedule_0: TrainSchedule,
        train_run_0: Trainrun,
        constraints_0: RouteDAGConstraints,
        train_schedule_1: TrainSchedule,
        train_run_1: Trainrun,
        constraints_1: RouteDAGConstraints,
        debug: bool = False
) -> UndirectedEncounterGraphDistance:
    """Computes the sum of time window overlaps of all common resources.
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

    # get positions in time window
    # TODO wann kann waypoint None sein?
    train_0_positions = [waypoint.position for i, waypoint in train_schedule_0.items()
                         if waypoint is not None]
    train_1_positions = [waypoint.position for i, waypoint in train_schedule_1.items()
                         if waypoint is not None]

    train_0_earliest, train_0_latest = _extract_earliest_latest_dict(constraints_0, train_0_positions)
    train_1_earliest, train_1_latest = _extract_earliest_latest_dict(constraints_1, train_1_positions)

    if debug:
        print("positions")
        print(train_0_positions)
        print(train_1_positions)
        print("earliest")
        print(train_0_earliest)
        print(train_0_latest)
        print("latest")
        print(train_1_earliest)
        print(train_1_latest)

    # compute distances of positions in time window
    intervals = [((train_0_earliest[cell], train_0_latest[cell]), (train_1_earliest[cell], train_1_latest[cell]))
                 for cell, earliest_0 in train_0_earliest.items() if cell in train_1_earliest.keys()]

    overlap_lengths = [abs(overlaps(*interval)) for interval in intervals]
    if debug:
        print(f"intervals={intervals}")
        print(f"overlap_lengths={overlap_lengths}")

    overlap_intervals = [
        overlap_interval((train_0_earliest[cell], train_0_latest[cell]), (train_1_earliest[cell], train_1_latest[cell]))
        for cell, earliest_0 in train_0_earliest.items() if cell in train_1_earliest.keys()]
    if debug:
        print(f"overlap_intervals={overlap_intervals}")
    overlap_starts = [interval[0] for interval in overlap_intervals if interval is not None]
    if debug:
        print(f"overlap_starts={overlap_starts}")
    if len(overlap_starts) == 0:
        return UndirectedEncounterGraphDistance(inverted_distance=0,
                                                time_of_min=0,
                                                train_0_position_at_min=0,
                                                train_1_position_at_min=0)

    # heuristic: sum of time window overlaps of resources in schedule
    index_min_dist = np.min(overlap_starts)
    proximity_between_trains = np.sum(overlap_lengths)

    # TODO what about time_of_min?
    start_time_step = max(train_0_start_time, train_1_start_time)
    distance = UndirectedEncounterGraphDistance(inverted_distance=proximity_between_trains,
                                                time_of_min=(start_time_step + index_min_dist),
                                                train_0_position_at_min=train_0_positions[min(int(index_min_dist), len(train_0_positions) - 1)],
                                                train_1_position_at_min=train_1_positions[min(int(index_min_dist), len(train_1_positions) - 1)])

    if debug:
        print(f"proximity_between_trains={proximity_between_trains}")

    return distance


def _extract_earliest_latest_dict(constraints_0, train_0_positions):
    train_earliest = {wp: np.inf for wp in train_0_positions}
    train_latest = {wp: -np.inf for wp in train_0_positions}
    for wp, scheduled_at in constraints_0.freeze_earliest.items():
        cell = wp.position
        if cell in train_earliest:
            train_earliest[cell] = min(train_earliest[cell], scheduled_at)
    for wp, scheduled_at in constraints_0.freeze_latest.items():
        cell = wp.position
        if cell in train_latest:
            train_latest[cell] = max(train_latest[cell], scheduled_at)
    return train_earliest, train_latest


def overlap_interval(interval1, interval2):
    """Given [0, 4] and [1, 10] returns [1, 4]"""

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
    """Return the amount of overlap, in bp between a and b.

    If >0, the number of bp of overlap If 0,  they are book-ended. If
    <0, the distance in bp between them
    """

    return min(a[1], b[1]) - max(a[0], b[0])


def compute_undirected_distance_matrix(trainrun_dict: TrainrunDict,
                                       schedule_problem_description: ScheduleProblemDescription,
                                       train_schedule_dict: TrainScheduleDict,
                                       metric_function: MetricFunction = None,
                                       ) -> (np.ndarray, Dict):
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
    proximity_matrix = np.zeros((number_of_trains, number_of_trains))

    for row, train_schedule_row in train_schedule_dict.items():
        for column, train_schedule_column in train_schedule_dict.items():
            if column > row:
                train_run_row = trainrun_dict.get(row)
                train_run_column = trainrun_dict.get(column)

                # TODO remove debugging code
                debug = False
                if row == 15 and column == 44:
                    print(f"{row} - {column}")
                    debug = True
                undirected_distance: UndirectedEncounterGraphDistance = metric_function(
                    train_schedule_0=train_schedule_row,
                    train_run_0=train_run_row,
                    constraints_0=schedule_problem_description.route_dag_constraints_dict[row],
                    train_schedule_1=train_schedule_column,
                    train_run_1=train_run_column,
                    constraints_1=schedule_problem_description.route_dag_constraints_dict[column],
                    debug=debug
                )
                proximity_matrix[row, column] = undirected_distance.inverted_distance
                proximity_matrix[column, row] = undirected_distance.inverted_distance
                if debug:
                    print("test")
                    print(undirected_distance.inverted_distance)

    trains = train_schedule_dict.keys()
    # TODO fix transitive closure?
    if False:
        _transitive_closure_distance_matrix(proximity_matrix, trains)
    distance_matrix, additional_info = _convert_proximity_matrix_to_distance_matrix(proximity_matrix=proximity_matrix, trains=trains)
    return distance_matrix, additional_info


def _convert_proximity_matrix_to_distance_matrix(proximity_matrix: np.ndarray, trains: List[int]):
    additional_info = {}
    distance_matrix = proximity_matrix.copy()
    for row in trains:
        for column in trains:
            if column > row:
                proximity = proximity_matrix[row, column]
                if proximity == 0.0:
                    ALMOST_INFINTITY = 999999999999999
                    distance_matrix[row, column] = ALMOST_INFINTITY
                    distance_matrix[column, row] = ALMOST_INFINTITY
                else:
                    distance = 1.0 / proximity
                    distance_matrix[row, column] = distance
                    distance_matrix[column, row] = distance
                additional_info[(row, column)] = proximity
    return distance_matrix, additional_info


def _transitive_closure_distance_matrix(proximity_matrix: np.ndarray, trains: List[int]):
    # transitive closure, not optimized
    updated = True
    while updated:
        updated = False
        for row in trains:
            for column in trains:
                if row == column:
                    continue
                for intermediate in trains:
                    if intermediate == row or intermediate == column:
                        continue
                    # ist das sinnvoll?
                    indirect_proximity = proximity_matrix[row, intermediate] + proximity_matrix[intermediate, column]
                    proximity_sofar = proximity_matrix[row, column]
                    updated = updated or (indirect_proximity > proximity_sofar)
                    proximity_matrix[row, column] = max(indirect_proximity, proximity_sofar)
