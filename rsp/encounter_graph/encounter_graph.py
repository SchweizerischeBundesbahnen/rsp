from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.route_dag.route_dag import RouteDAGConstraints
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.utils.data_types import SymmetricEncounterGraphDistance
from rsp.utils.data_types import TrainSchedule
from rsp.utils.data_types import TrainScheduleDict

MetricFunction = Callable[[TrainSchedule,
                           Trainrun,
                           RouteDAGConstraints,
                           TrainSchedule,
                           Trainrun,
                           RouteDAGConstraints,
                           bool], SymmetricEncounterGraphDistance]


def symmetric_distance_between_trains_dummy_Euclidean(train_schedule_0: TrainSchedule,
                                                      train_run_0: Trainrun,
                                                      constraints_0: RouteDAGConstraints,
                                                      train_schedule_1: TrainSchedule,
                                                      train_run_1: Trainrun,
                                                      constraints_1: RouteDAGConstraints,
                                                      debug: bool = False
                                                      ) -> SymmetricEncounterGraphDistance:
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
    SymmetricEncounterGraphDistance
        contains the data related to the undirected encounter graph distance
    """
    train_0_start_time = train_run_0[0].scheduled_at
    train_0_end_time = train_run_0[-1].scheduled_at
    train_1_start_time = train_run_1[0].scheduled_at
    train_1_end_time = train_run_1[-1].scheduled_at

    # if the time window of the two trains do not overlap -> no relationship between trains
    if train_0_end_time < train_1_start_time or train_1_end_time < train_0_start_time:
        return SymmetricEncounterGraphDistance(proximity=0)

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

    distance = SymmetricEncounterGraphDistance(proximity=(1. / dist_between_trains))

    return distance


def symmetric_distance_between_trains_sum_of_time_window_overlaps(
        train_schedule_0: TrainSchedule,
        train_run_0: Trainrun,
        constraints_0: RouteDAGConstraints,
        train_schedule_1: TrainSchedule,
        train_run_1: Trainrun,
        constraints_1: RouteDAGConstraints,
        debug: bool = False
) -> SymmetricEncounterGraphDistance:
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

    # TODO wann kann waypoint None sein? Bug?
    train_0_positions: List[Waypoint] = [waypoint.position for i, waypoint in train_schedule_0.items()
                                         if waypoint is not None]
    train_1_positions: List[Waypoint] = [waypoint.position for i, waypoint in train_schedule_1.items()
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
                 for cell, earliest_0 in train_0_earliest.items()
                 if cell in train_1_positions and cell in train_0_positions

                 ]

    overlap_lengths = [abs(overlaps(*interval)) for interval in intervals]
    if debug:
        print(f"intervals={intervals}")
        print(f"overlap_lengths={overlap_lengths}")

    overlap_intervals = [
        overlap_interval((train_0_earliest[cell], train_0_latest[cell]), (train_1_earliest[cell], train_1_latest[cell]))
        for cell, earliest_0 in train_0_earliest.items()
        if cell in train_1_positions and cell in train_0_positions
    ]
    if debug:
        print(f"overlap_intervals={overlap_intervals}")
    overlap_starts = [interval[0] for interval in overlap_intervals if interval is not None]
    if debug:
        print(f"overlap_starts={overlap_starts}")
    if len(overlap_starts) == 0:
        return SymmetricEncounterGraphDistance(proximity=0)

    # heuristic: sum of time window overlaps of resources in schedule
    proximity_between_trains = np.sum(overlap_lengths)

    distance = SymmetricEncounterGraphDistance(proximity=proximity_between_trains)

    if debug:
        print(f"proximity_between_trains={proximity_between_trains}")

    return distance


def _extract_earliest_latest_dict(
        constraints: RouteDAGConstraints,
        train_positions: List[Waypoint],
        max_window_size_from_earliest: int = 30
) -> Tuple[Dict[Waypoint, int], Dict[Waypoint, int]]:
    """For all cells, extract earliest and latest from constraints.

    - If there are multiple vertices, take the min for earliest and the max for latest
    - Set latest to at most earliest + max_window_size_from_earliest
    # TODO why is this necessary? Is this because of aggregation over multiple vertices of the same cell or because of the agenda?

    Parameters
    ----------
    constraints
    train_positions
    max_window_size_from_earliest

    Returns
    -------
    """
    train_earliest: Dict[Waypoint, int] = {wp: np.inf for wp in train_positions}
    train_latest: Dict[Waypoint, int] = {wp: -np.inf for wp in train_positions}
    for wp, scheduled_at in constraints.freeze_earliest.items():
        cell = wp.position
        if cell in train_earliest:
            train_earliest[cell] = min(train_earliest[cell], scheduled_at)
    for wp, scheduled_at in constraints.freeze_latest.items():
        cell = wp.position
        if cell in train_latest:
            train_latest[cell] = max(train_latest[cell], scheduled_at)
    assert set(train_latest.keys()) == set(train_earliest.keys())
    # TODO this does not seem to work well:
    if False:
        for cell, earliest in train_earliest.items():
            train_latest[cell] = min(earliest + max_window_size_from_earliest, train_latest[cell])
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


def symmetric_temporal_distance_between_trains(train_schedule_0: TrainSchedule,
                                               train_run_0: Trainrun,
                                               constraints_0: RouteDAGConstraints,
                                               train_schedule_1: TrainSchedule,
                                               train_run_1: Trainrun,
                                               constraints_1: RouteDAGConstraints,
                                               debug: bool = False) -> SymmetricEncounterGraphDistance:
    """Compute the summed distance in time between two trains on shared
    ressources.

    Parameters
    ----------
    train_schedule_0: TrainSchedule,
    train_run_0: Trainrun,
    constraints_0: RouteDAGConstraints,
    train_schedule_1: TrainSchedule,
    train_run_1: Trainrun,
    constraints_1: RouteDAGConstraints,
    debug: bool = False

    Returns
    -------
    SymmetricEncounterGraphDistance
    contains the data related to the undirected encounter graph distance
    """
    time_distance = np.inf
    for time_0, waypoint_0 in train_schedule_0.items():
        for time_1, waypoint_1 in train_schedule_1.items():
            if waypoint_0 is not None and waypoint_1 is not None:
                if waypoint_0.position == waypoint_1.position:
                    tmp_dist = np.abs(time_1 - time_0)
                    if tmp_dist <= time_distance:
                        time_distance = tmp_dist

    distance = SymmetricEncounterGraphDistance(proximity=(1. / time_distance))

    return distance


def compute_symmetric_distance_matrix(trainrun_dict: TrainrunDict,
                                      schedule_problem_description: ScheduleProblemDescription,
                                      train_schedule_dict: TrainScheduleDict,
                                      metric_function: MetricFunction = None,
                                      debug_pair: Tuple[int, int] = None
                                      ) -> np.ndarray:
    """This method computes the distance matrix for a complete TrainrunDict ->
    each distance between each pair of trains is computed.

    Parameters
    ----------
    trainrun_dict
        Dictionary containing all the trainruns
    schedule_problem_description
        The schedule problem description for this case.
    train_schedule_dict
        Dictionary containing the schedules (Visited times and cells of all trains
    metric_function
        Metric function to be used to compute the distance matrix
    debug_pair
        Print debug information when dealing with the pair of agents.
    Returns
    -------
    distance_matrix
        the distance matrix as a symmetric matrix each entry corresponds to a pair of trains
    additional_info
        a dictionary with additional info like the time step at which the minimal distance happened and the location of
        the trains
    """
    if metric_function is None:
        metric_function = symmetric_distance_between_trains_dummy_Euclidean
    number_of_trains = len(trainrun_dict)
    proximity_matrix = np.zeros((number_of_trains, number_of_trains))

    for row, train_schedule_row in train_schedule_dict.items():
        for column, train_schedule_column in train_schedule_dict.items():
            if column > row:
                train_run_row = trainrun_dict.get(row)
                train_run_column = trainrun_dict.get(column)
                _debug = False
                if debug_pair is not None:
                    debug_row, debug_column = debug_pair
                    if row == debug_row and column == _debug:
                        print(f"{row} - {column}")
                        _debug = True
                undirected_distance: SymmetricEncounterGraphDistance = metric_function(
                    train_schedule_0=train_schedule_row,
                    train_run_0=train_run_row,
                    constraints_0=schedule_problem_description.route_dag_constraints_dict[row],
                    train_schedule_1=train_schedule_column,
                    train_run_1=train_run_column,
                    constraints_1=schedule_problem_description.route_dag_constraints_dict[column],
                    debug=_debug
                )
                proximity_matrix[row, column] = undirected_distance.proximity
                proximity_matrix[column, row] = undirected_distance.proximity

    trains = train_schedule_dict.keys()
    distance_matrix = _convert_proximity_matrix_to_normalized_distance_matrix(proximity_matrix=proximity_matrix, trains=trains)
    return distance_matrix


# TODO do we need this normalization? If yes, refactor, else remove.
def _convert_proximity_matrix_to_normalized_distance_matrix(proximity_matrix: np.ndarray, trains: List[int], separation_factor: float = 10.0):
    """

    Parameters
    ----------
    proximity_matrix
        non-negative symmetric proximities
    trains
    separation_factor:
        upon normalization, the maximum finite distance is set to 1/separation_factor, whereas infinite distance (no proximity) is set to 1.0

    Returns
    -------

    """
    distance_matrix = proximity_matrix.copy()
    max_distance = 0
    # 1. take inverse of finite proximities in range [0.0,np.inf) and to distances in range [0.0,np.inf] and keep track of maximum non-infinite distance
    # TODO do we need train as parameters? could we use array dimensions instead?
    for from_train in trains:
        for to_train in trains:
            if to_train > from_train:
                proximity = proximity_matrix[from_train, to_train]
                if proximity == 0.0:
                    distance_matrix[from_train, to_train] = np.inf
                    distance_matrix[to_train, from_train] = np.inf
                else:
                    distance = 1.0 / proximity
                    distance_matrix[from_train, to_train] = distance
                    distance_matrix[to_train, from_train] = distance
                    max_distance = max(max_distance, distance)
    # 2. normalize distance_matrix to range [0.0,1.0] by use of max_distance
    distance_matrix /= (max_distance * separation_factor)
    for from_train in trains:
        for to_train in trains:
            if to_train > from_train and distance_matrix[from_train, to_train] == np.inf:
                distance_matrix[from_train, to_train] = 1.0
                distance_matrix[to_train, from_train] = 1.0
    return distance_matrix


# TODO remove or use:
#     Do we need transitive closure: train 1 and train 2 are close, train 2 and train 3 are close, does this propagate to the proximity of 1 and 3?
#     Is this directed?
def _transitive_closure_distance_matrix(proximity_matrix: np.ndarray, trains: List[int]):
    # transitive closure, not optimized
    updated = True
    while updated:
        updated = False
        for from_train in trains:
            for to_train in trains:
                if from_train == to_train:
                    continue
                for intermediate in trains:
                    if intermediate == from_train or intermediate == to_train:
                        continue
                    # ist das sinnvoll?
                    indirect_proximity = proximity_matrix[from_train, intermediate] + proximity_matrix[intermediate, to_train]
                    proximity_sofar = proximity_matrix[from_train, to_train]
                    updated = updated or (indirect_proximity > proximity_sofar)
                    proximity_matrix[from_train, to_train] = max(indirect_proximity, proximity_sofar)
