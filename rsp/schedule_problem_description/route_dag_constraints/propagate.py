"""Generic route dag generation."""
from collections import deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.logger import rsp_logger
from rsp.schedule_problem_description.data_types_and_utils import get_paths_in_route_dag
from rsp.schedule_problem_description.data_types_and_utils import RouteDAGConstraints
from rsp.utils.data_types import ExperimentMalfunction


def _propagate_earliest(earliest_dict: Dict[Waypoint, int],
                        force_freeze_earliest: Set[Waypoint],
                        minimum_travel_time: int,
                        topo: nx.DiGraph) -> Dict[Waypoint, int]:
    """Extract earliest times at nodes by moving forward from source(s).
    Earliest time for the agent to reach this vertex given the freezed times.
    Pre-condition: all waypoints in `force_freeze_earliest` have a finite value (`< np.inf`) in `earliest_dict`.
    Caveat: `earliest_dict` is modified.

    Parameters
    ----------
    earliest_dict
    force_freeze_earliest: earliest must not be changed
    minimum_travel_time
    topo
    """
    assert force_freeze_earliest.issubset(set(earliest_dict.keys()))
    try:
        assert force_freeze_earliest.issubset(set(topo.nodes))
    except AssertionError as e:
        rsp_logger.error(f"force_freeze_earliest={force_freeze_earliest}, topo.nodes={list(topo.nodes)}")
        raise e

    # update as min(earliest_at_predecessor+minimum_travel_time,current_earliest) until no updates to process.
    open = deque()
    open.extend(force_freeze_earliest)

    while len(open) > 0:
        waypoint = open.pop()
        for successor in topo.successors(waypoint):
            if successor in force_freeze_earliest:
                continue
            earliest_dict[successor] = min(earliest_dict[waypoint] + minimum_travel_time, earliest_dict.get(successor, np.inf))
            open.append(successor)
    return earliest_dict


def _propagate_latest(
        force_freeze_latest: Set[Waypoint],
        latest_dict: Dict[Waypoint, int],
        minimum_travel_time: int,
        topo: nx.DiGraph):
    """Extract latest times at nodes by moving backward from sinks.
    Latest time for the agent to reach a target from this vertex given the freezed times.
    Pre-condition: all waypoints in `force_freeze_latest` have a finite value (`< np.inf`) in `latest_dict`.
    Caveat: `latest_dict` is modified.

    Parameters
    ----------
    latest_dict
    minimum_travel_time
    topo
    """
    assert force_freeze_latest.issubset(set(latest_dict.keys()))
    try:
        assert force_freeze_latest.issubset(set(topo.nodes))
    except AssertionError as e:
        rsp_logger.error(f"force_freeze_latest={force_freeze_latest}, topo.nodes={list(topo.nodes)}")
        raise e

    # update max(latest_at_next_node-minimum_travel_time,current_latest) until no updates to process.
    open = deque()
    open.extend(force_freeze_latest)
    while len(open) > 0:
        waypoint = open.pop()
        for predecessor in topo.predecessors(waypoint):
            if predecessor in force_freeze_latest:
                continue
            latest_dict[predecessor] = max(latest_dict[waypoint] - minimum_travel_time, latest_dict.get(predecessor, -np.inf))
            open.append(predecessor)


def _propagate_latest_forward_constant(earliest_dict: Dict[Waypoint, int],
                                       latest_arrival: int,
                                       max_window_size_from_earliest: int) -> Dict[Waypoint, int]:
    """Extract latest by adding a constant value to earliest.

    Parameters
    ----------
    earliest_dict
    latest_arrival
    max_window_size_from_earliest: int
        maximum window size as offset from earliest. => "Cuts off latest at earliest + earliest_time_windows when doing
        back propagation of latest"
    """
    latest_dict = {}
    for waypoint, earliest in earliest_dict.items():
        latest = min(earliest + max_window_size_from_earliest, latest_arrival)
        latest_dict[waypoint] = latest
    return latest_dict


def _get_reachable_given_frozen_set(topo: nx.DiGraph,
                                    must_be_visited: List[Waypoint]) -> Set[Waypoint]:
    """Determines which vertices can still be reached given the frozen set. We
    take all funnels forward and backward from these points and then the
    intersection of those. A source and sink node only have a forward and
    backward funnel, respectively. In FLATland, the source node is always
    unique, the sink node is made unique by a dummy node at the end (the agent
    may enter from more than one direction ino the target cell.)

    Parameters
    ----------
    topo
        directed graph
    must_be_visited
        the waypoints that must be visited

    Returns
    -------
    """
    forward_reachable = {waypoint: set() for waypoint in topo.nodes}
    backward_reachable = {waypoint: set() for waypoint in topo.nodes}

    # collect descendants and ancestors of freeze
    for waypoint in must_be_visited:
        forward_reachable[waypoint] = set(nx.descendants(topo, waypoint))
        backward_reachable[waypoint] = set(nx.ancestors(topo, waypoint))

    # reflexivity: add waypoint to its own closure (needed for building reachable_set below)
    for waypoint in topo.nodes:
        forward_reachable[waypoint].add(waypoint)
        backward_reachable[waypoint].add(waypoint)

    # reachable are only those that are either in the forward or backward "funnel" of all force freezes!
    reachable_set = set(topo.nodes)
    for waypoint in must_be_visited:
        forward_and_backward_reachable = forward_reachable[waypoint].union(backward_reachable[waypoint])
        reachable_set.intersection_update(forward_and_backward_reachable)
    return reachable_set


def propagate(  # noqa C901
        earliest_dict: Dict[Waypoint, int],
        latest_dict: Dict[Waypoint, int],
        topo: nx.DiGraph,
        force_freeze_earliest: Set[Waypoint],
        force_freeze_latest: Set[Waypoint],
        must_be_visited: Set[Waypoint],
        minimum_travel_time: int,
        latest_arrival: int,
        max_window_size_from_earliest: int = np.inf
):
    """If max_window_size_from_earliest is 0, then extract latest by moving
    backwards from sinks. Latest time for the agent to pass here in order to
    reach the target in time. Otherwise take the minimum of moving backwards or
    earliest + max_window_size_from_earliest.

    Parameters
    ----------
    force_freeze_latest
    earliest_dict
    latest_dict
    latest_arrival
    minimum_travel_time
    topo
    max_window_size_from_earliest: int
        maximum window size as offset from earliest. => "Cuts off latest at earliest + earliest_time_windows when doing
        back propagation of latest"
    """
    # remove nodes not reachable given the must_be_visited
    assert set(force_freeze_earliest).issubset(topo.nodes)
    assert set(force_freeze_latest).issubset(topo.nodes)
    reachable = _get_reachable_given_frozen_set(topo=topo, must_be_visited=must_be_visited)
    to_remove = {v for v in topo.nodes if v not in reachable}
    topo.remove_nodes_from(to_remove)
    try:
        assert set(must_be_visited).issubset(reachable)

    except AssertionError as e:
        rsp_logger.error(f"must_be_visited={must_be_visited}, reachable={reachable}")
        raise e
    assert set(force_freeze_earliest).issubset(reachable)
    assert set(force_freeze_latest).issubset(reachable)

    _propagate_earliest(
        earliest_dict=earliest_dict,
        force_freeze_earliest=force_freeze_earliest,
        minimum_travel_time=minimum_travel_time,
        topo=topo)
    _propagate_latest(
        force_freeze_latest=force_freeze_latest,
        latest_dict=latest_dict,
        minimum_travel_time=minimum_travel_time,
        topo=topo
    )
    if max_window_size_from_earliest < np.inf:
        latest_forward = _propagate_latest_forward_constant(
            earliest_dict=earliest_dict,
            latest_arrival=latest_arrival,
            max_window_size_from_earliest=max_window_size_from_earliest
        )
        for waypoint, latest_forward_time in latest_forward.items():
            latest_backward_time = latest_dict.get(waypoint)
            latest_dict[waypoint] = min(latest_forward_time, latest_backward_time)
        # apply latest again for consistency
        _propagate_latest(
            force_freeze_latest=force_freeze_latest,
            latest_dict=latest_dict,
            minimum_travel_time=minimum_travel_time,
            topo=topo
        )

    def _remove_waypoint_from_earliest_latest_topo(waypoint):
        # design choice: we give no earliest/latest for banned!
        if waypoint in earliest_dict:
            earliest_dict.pop(waypoint)
        if waypoint in latest_dict:
            latest_dict.pop(waypoint)
        topo.remove_node(waypoint)

    # remove nodes not reachable in time
    to_remove = set()
    for waypoint in topo.nodes:
        if (waypoint not in earliest_dict or waypoint not in latest_dict or  # noqa: W504
                earliest_dict[waypoint] > latest_dict[waypoint]):
            to_remove.add(waypoint)
    for wp in to_remove:
        _remove_waypoint_from_earliest_latest_topo(wp)


def _get_delayed_trainrun_waypoint_after_malfunction(
        agent_id: int,
        trainrun: Trainrun,
        malfunction: ExperimentMalfunction,
        minimum_travel_time: int) -> TrainrunWaypoint:
    """Returns the trainrun waypoint after the malfunction that needs to be re.

    Parameters
    ----------
    agent_id
    trainrun
    malfunction

    Returns
    -------
    """
    previous_scheduled = 0
    for trainrun_waypoint in trainrun:
        if trainrun_waypoint.scheduled_at > malfunction.time_step:
            if agent_id == malfunction.agent_id:
                end_of_malfunction_time_step = malfunction.time_step + malfunction.malfunction_duration
                elapsed_at_malfunction_start = malfunction.time_step - previous_scheduled
                remaining_minimum_travel_time = max(minimum_travel_time - elapsed_at_malfunction_start, 0)
                earliest = end_of_malfunction_time_step + remaining_minimum_travel_time
                return TrainrunWaypoint(
                    waypoint=trainrun_waypoint.waypoint,
                    scheduled_at=earliest)
            else:
                return trainrun_waypoint
        previous_scheduled = trainrun_waypoint.scheduled_at
    return trainrun[-1]


def verify_consistency_of_route_dag_constraints_for_agent(  # noqa: C901
        agent_id: int,
        route_dag_constraints: RouteDAGConstraints,
        topo: nx.DiGraph,
        malfunction: Optional[ExperimentMalfunction] = None,
        max_window_size_from_earliest: int = np.inf
):
    """Does the route_dag_constraints reflect the force freeze, route DAG and
    malfunctions correctly? Are the constraints consistent?

    0. assert all referenced waypoints are in topo
    1. all waypoints in topo must have earliest and latest s.t. earliest <= latest
    2. verify that all points up to malfunction are visited,
    3a. verify that all source-sink paths go through these points
    4. verify that latest-earliest <= max_window_size_from_earliest

    Parameters
    ----------
    agent_id
    route_dag_constraints
        the experiment freeze to be verified.
    topo
        the route DAG
    malfunction
        if it's the agent in malfunction, the experiment freeze should put a visit and earliest constraint
    max_window_size_from_earliest

    Returns
    -------
    """

    all_waypoints = topo.nodes

    # 00. TODO SIM-613->SIM-650 (temporary check, to be removed in SIM-650)
    assert not route_dag_constraints.freeze_visit
    assert not route_dag_constraints.freeze_banned

    # 0. assert all referenced waypoints are in topo
    for waypoint in route_dag_constraints.freeze_earliest:
        assert waypoint in all_waypoints, f"agent {agent_id}: {waypoint} has earliest, but not in topo"
    for waypoint in route_dag_constraints.freeze_latest:
        assert waypoint in all_waypoints, f"agent {agent_id}: {waypoint} has latest, but not in topo"

    # 1. all waypoints in topo must have earliest and latest s.t. earliest <= latest
    for waypoint in all_waypoints:
        assert waypoint in route_dag_constraints.freeze_earliest, \
            f"agent {agent_id} has no earliest for {waypoint}"
        assert waypoint in route_dag_constraints.freeze_latest, \
            f"agent {agent_id} has no latest for {waypoint}"
        assert route_dag_constraints.freeze_earliest[waypoint] <= route_dag_constraints.freeze_latest[waypoint], \
            f"agent {agent_id} at {waypoint}: earliest should be less or equal to latest, " + \
            f"found {route_dag_constraints.freeze_earliest[waypoint]} <= {route_dag_constraints.freeze_latest[waypoint]}"

    # 2. verify that all points up to malfunction are forced to be visited
    if malfunction:
        # 2a. verify that all source-sink paths go through these points
        all_paths = get_paths_in_route_dag(topo)
        all_path_vertices = [set(path) for path in all_paths]
        vertices_of_all_paths = all_path_vertices[0]
        for path_vertices in all_path_vertices[1:]:
            vertices_of_all_paths.intersection_update(path_vertices)

        for waypoint, earliest in route_dag_constraints.freeze_earliest.items():
            # everything before malfunction must be the same
            if earliest <= malfunction.time_step:
                assert route_dag_constraints.freeze_latest[waypoint] == earliest
                assert waypoint in vertices_of_all_paths
            # everything after malfunction must be respect malfunction duration (at least) for malfunction agent
            elif agent_id == malfunction.agent_id:
                assert earliest >= malfunction.time_step + malfunction.malfunction_duration, \
                    f"agent {agent_id} with malfunction {malfunction}. Found earliest={earliest} for {waypoint}"

    # 3. verify that latest-earliest <= max_window_size_from_earliest
    for waypoint in route_dag_constraints.freeze_earliest:
        assert (route_dag_constraints.freeze_latest[waypoint] - route_dag_constraints.freeze_earliest[waypoint] <= max_window_size_from_earliest), \
            f"{waypoint} of {agent_id}: [{route_dag_constraints.freeze_earliest[waypoint]},{route_dag_constraints.freeze_latest[waypoint]}], " \
            f"expected {max_window_size_from_earliest}"


def verify_trainrun_satisfies_route_dag_constraints(agent_id, route_dag_constraints, scheduled_trainrun):
    """Does the route_dag_constraints reflect the force freeze, route DAG and
    malfunctions correctly?

    Parameters
    ----------
    route_dag_constraints
        the experiment freeze to be verified.
    scheduled_trainrun
        verify that this whole train run is part of the solution space.
        With malfunctions, caller must ensure that only relevant part is passed to be verified!
    """

    scheduled_dict = {
        trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
        for trainrun_waypoint in scheduled_trainrun
    }
    for waypoint, scheduled_at in scheduled_dict.items():
        assert waypoint not in route_dag_constraints.freeze_banned, \
            f"agent {agent_id}: the known solution has " \
            f"schedule {waypoint} at {scheduled_at} - " \
            f"but banned constraint"
        assert waypoint in route_dag_constraints.freeze_earliest, \
            f"agent {agent_id}: the known solution has " \
            f"schedule {waypoint} at {scheduled_at} - " \
            f"but no earliest constraint"
        assert scheduled_at >= route_dag_constraints.freeze_earliest[waypoint], \
            f"agent {agent_id}: the known solution has " \
            f"schedule {waypoint} at {scheduled_at} - " \
            f"but found earliest {route_dag_constraints.freeze_latest[waypoint]}"
        assert waypoint in route_dag_constraints.freeze_latest, \
            f"agent {agent_id}: the known solution has " \
            f"schedule {waypoint} at {scheduled_at} - " \
            f"but no latest constraint"
        assert scheduled_at <= route_dag_constraints.freeze_latest[waypoint], \
            f"agent {agent_id}: the known solution has " \
            f"schedule {waypoint} at {scheduled_at} - " \
            f"but found latest {route_dag_constraints.freeze_latest[waypoint]}"
        assert waypoint not in route_dag_constraints.freeze_banned
