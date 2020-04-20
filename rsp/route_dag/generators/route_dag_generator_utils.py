"""Generic route dag generation."""
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.route_dag.route_dag import RouteDAGConstraints
from rsp.utils.data_types import ExperimentMalfunction


def propagate_earliest(banned_set: Set[Waypoint],
                       earliest_dict: Dict[Waypoint, int],
                       force_freeze_dict: Dict[Waypoint, int],
                       minimum_travel_time: int,
                       subdag_source: TrainrunWaypoint,
                       topo: nx.DiGraph) -> Dict[Waypoint, int]:
    """Extract earliest times at nodes by moving forward from source. Earliest
    time for the agent to reach this vertex given the freezed times.

    Parameters
    ----------
    banned_set
    earliest_dict
    force_freeze_dict
    minimum_travel_time
    subdag_source
    topo
    """
    # iterate as long as there are updates (not optimized!)
    # update as min(earliest_at_predecessor+minimum_travel_time,current_earliest) until fixed point reached
    done = False
    while not done:
        done = True
        for waypoint in topo.nodes:
            # https://networkx.github.io/documentation/stable/reference/classes/generated/networkx.DiGraph.predecessors.html
            for successor in topo.successors(waypoint):
                if successor in force_freeze_dict or successor in banned_set:
                    continue
                else:
                    minimum_travel_time_here = minimum_travel_time
                    # TODO SIM-322 put minimum_travel_time per edge and add dumy edges for source and sink as input
                    # minimum travel time is 1 (synchronization step) to one if we're coming from the source or goint to sink
                    if topo.in_degree[waypoint] == 0 or topo.out_degree[successor] == 0:
                        minimum_travel_time_here = 1
                    path_earliest = earliest_dict.get(waypoint, np.inf) + minimum_travel_time_here
                    earliest = min(path_earliest, earliest_dict.get(successor, np.inf))
                    if earliest > subdag_source.scheduled_at and earliest < np.inf:
                        earliest = int(earliest)
                        if earliest_dict.get(successor, None) != earliest:
                            done = False
                        earliest_dict[successor] = earliest
    return earliest_dict


def propagate_latest(banned_set: Set[Waypoint],
                     force_freeze_dict: Dict[Waypoint, int],
                     earliest_dict: Dict[Waypoint, int],
                     latest_dict: Dict[Waypoint, int],
                     latest_arrival: int,
                     minimum_travel_time: int,
                     topo: nx.DiGraph,
                     max_window_size_from_earliest: int = np.inf
                     ) -> Dict[Waypoint, int]:
    """If max_window_size_from_earliest is 0, then extract latest by moving
    backwards from sinks. Latest time for the agent to pass here in order to
    reach the target in time. Otherwise take the minimum of moving backwards or
    earliest + max_window_size_from_earliest.

    Parameters
    ----------
    banned_set
    force_freeze_dict
    earliest_dict
    latest_dict
    latest_arrival
    minimum_travel_time
    topo
    max_window_size_from_earliest: int
        maximum window size as offset from earliest. => "Cuts off latest at earliest + earliest_time_windows when doing
        back propagation of latest"
    """

    latest_backwards = _propagate_latest_backwards(banned_set=banned_set,
                                                   force_freeze_dict=force_freeze_dict,
                                                   latest_arrival=latest_arrival,
                                                   latest_dict=latest_dict,
                                                   minimum_travel_time=minimum_travel_time,
                                                   topo=topo)
    if max_window_size_from_earliest == np.inf:
        return latest_backwards
    else:
        latest_forward = _propagate_latest_forward_constant(earliest_dict=earliest_dict,
                                                            latest_dict={},
                                                            latest_arrival=latest_arrival,
                                                            max_window_size_from_earliest=max_window_size_from_earliest)

        latest = dict()
        for waypoint, latest_forward_time in latest_forward.items():
            latest_backward_time = latest_backwards.get(waypoint)
            latest[waypoint] = min(latest_forward_time, latest_backward_time)
        return latest


def _propagate_latest_forward_constant(earliest_dict: Dict[Waypoint, int],
                                       latest_dict: Dict[Waypoint, int],
                                       latest_arrival: int,
                                       max_window_size_from_earliest: int) -> Dict[Waypoint, int]:
    """Extract latest by adding a constant value to earliest.

    Parameters
    ----------
    earliest_dict
    latest_dict
    latest_arrival
    max_window_size_from_earliest: int
        maximum window size as offset from earliest. => "Cuts off latest at earliest + earliest_time_windows when doing
        back propagation of latest"
    """

    for waypoint, earliest in earliest_dict.items():
        latest = earliest + max_window_size_from_earliest
        latest = latest_arrival if latest > latest_arrival else latest
        latest_dict[waypoint] = latest
    return latest_dict


def _propagate_latest_backwards(banned_set: Set[Waypoint],
                                force_freeze_dict: Dict[Waypoint, int],
                                latest_arrival: int,
                                latest_dict: Dict[Waypoint, int],
                                minimum_travel_time: int,
                                topo: nx.DiGraph):
    """Extract latest by moving backwards from sinks. Latest time for the agent
    to pass here in order to reach the target in time.

    Parameters
    ----------
    banned_set
    force_freeze_dict
    latest_arrival
    latest_dict
    minimum_travel_time
    topo
    """
    # iterate as long as there are updates (not optimized!)
    # update max(latest_at_next_node-minimum_travel_time,current_latest) until fixed point reached
    done = False
    while not done:
        done = True
        for waypoint in topo.nodes:
            # https://networkx.github.io/documentation/stable/reference/classes/generated/networkx.DiGraph.predecessors.html
            for predecessor in topo.predecessors(waypoint):
                if predecessor in force_freeze_dict or predecessor in banned_set:
                    continue
                else:
                    minimum_travel_time_corrected = minimum_travel_time
                    # TODO SIM-322 put minimum_travel_time per edge and add dumy edges for source and sink as input
                    # minimum travel time is 1 (synchronization step) if we're goint to the sink or coming from source
                    if topo.out_degree[waypoint] == 0 or topo.in_degree[predecessor] == 0:
                        minimum_travel_time_corrected = 1

                    path_latest = latest_dict.get(waypoint, -np.inf) - minimum_travel_time_corrected
                    latest = max(path_latest, latest_dict.get(predecessor, -np.inf))
                    if latest < latest_arrival and latest > -np.inf:
                        latest = int(latest)
                        if latest_dict.get(predecessor, None) != latest:
                            done = False
                        latest_dict[predecessor] = latest
    return latest_dict


def get_delayed_trainrun_waypoint_after_malfunction(
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
                return TrainrunWaypoint(
                    waypoint=trainrun_waypoint.waypoint,
                    scheduled_at=previous_scheduled + malfunction.malfunction_duration + minimum_travel_time)
            else:
                return trainrun_waypoint
        previous_scheduled = trainrun_waypoint.scheduled_at
    return trainrun[-1]


def verify_route_dag_constraints_for_agent(
        agent_id: int,
        route_dag_constraints: RouteDAGConstraints,
        topo: nx.DiGraph,
        force_freeze: Optional[List[TrainrunWaypoint]] = None,
        malfunction: Optional[ExperimentMalfunction] = None,
        scheduled_trainrun: Optional[Trainrun] = None
):
    """Does the route_dag_constraints reflect the force freeze, route DAG and
    malfunctions correctly?

    Parameters
    ----------
    scheduled_trainrun
    route_dag_constraints
        the experiment freeze to be verified.
    topo
        the route DAG
    force_freeze
        the trainrun waypoints that must as given (consistency is not checked!)
    malfunction
        if it's the agent in malfunction, the experiment freeze should put a visit and earliest constraint
    scheduled_trainrun
        verify that this whole train run is part of the solution space.
        With malfunctions, caller must ensure that only relevant part is passed to be verified!

    Returns
    -------
    """

    all_waypoints = topo.nodes
    for waypoint in all_waypoints:
        # if waypoint is banned -> must not earliest/latest/visit
        if waypoint in route_dag_constraints.freeze_banned:
            assert waypoint not in route_dag_constraints.freeze_earliest, \
                f"agent {agent_id}: {waypoint} banned, should have no earliest"
            assert waypoint not in route_dag_constraints.freeze_latest, \
                f"agent {agent_id}: {waypoint} banned, should have no latest"
            assert waypoint not in route_dag_constraints.freeze_visit, \
                f"agent {agent_id}: {waypoint} banned, should have no visit"
            assert waypoint not in route_dag_constraints.freeze_visit, \
                f"agent {agent_id}: {waypoint} banned, should have no visit"
        else:
            # waypoint must have earliest and latest s.t. earliest <= latest
            assert waypoint in route_dag_constraints.freeze_earliest, \
                f"agent {agent_id} has no earliest for {waypoint}"
            assert waypoint in route_dag_constraints.freeze_latest, \
                f"agent {agent_id} has no latest for {waypoint}"
            assert route_dag_constraints.freeze_earliest[waypoint] <= route_dag_constraints.freeze_latest[waypoint], \
                f"agent {agent_id} at {waypoint}: earliest should be less or equal to latest, " + \
                f"found {route_dag_constraints.freeze_earliest[waypoint]} <= {route_dag_constraints.freeze_latest[waypoint]}"

    # verify that force is implemented correctly
    if force_freeze:
        for trainrun_waypoint in force_freeze:
            assert route_dag_constraints.freeze_latest[trainrun_waypoint.waypoint] == trainrun_waypoint.scheduled_at, \
                f"agent {agent_id}: should have latest requirement " \
                f"for {trainrun_waypoint.waypoint} at {trainrun_waypoint.scheduled_at} - " \
                f"found {route_dag_constraints.freeze_latest[trainrun_waypoint.waypoint]}"
            assert route_dag_constraints.freeze_earliest[trainrun_waypoint.waypoint] == trainrun_waypoint.scheduled_at, \
                f"agent {agent_id}: should have earliest requirement " \
                f"for {trainrun_waypoint.waypoint} at {trainrun_waypoint.scheduled_at} - " \
                f"found {route_dag_constraints.freeze_earliest[trainrun_waypoint.waypoint]}"
            assert trainrun_waypoint.waypoint in route_dag_constraints.freeze_visit, \
                f"agent {agent_id}: should have visit requirement " \
                f"for {trainrun_waypoint.waypoint}"
            assert trainrun_waypoint.waypoint not in route_dag_constraints.freeze_banned, \
                f"agent {agent_id}: should have no banned requirement " \
                f"for {trainrun_waypoint.waypoint}"

    # verify that all points up to malfunction are forced to be visited
    if malfunction:
        for waypoint, earliest in route_dag_constraints.freeze_earliest.items():
            # everything before malfunction must be the same
            if earliest <= malfunction.time_step:
                assert route_dag_constraints.freeze_latest[waypoint] == earliest
                assert waypoint in route_dag_constraints.freeze_visit
            else:
                assert earliest >= malfunction.time_step + malfunction.malfunction_duration, \
                    f"{agent_id} {malfunction}. found earliest={earliest} for {waypoint}"

    # verify that scheduled train run is in the solution space
    if scheduled_trainrun:
        scheduled_dict = {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
                          for trainrun_waypoint in scheduled_trainrun}
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
