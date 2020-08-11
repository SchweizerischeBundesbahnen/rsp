from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Set

import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.logger import rsp_logger
from rsp.schedule_problem_description.data_types_and_utils import get_sinks_for_topo
from rsp.schedule_problem_description.data_types_and_utils import RouteSectionPenaltiesDict
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.schedule_problem_description.data_types_and_utils import TopoDict
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_utils import get_delayed_trainrun_waypoint_after_malfunction
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_utils import propagate
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_utils import verify_consistency_of_route_dag_constraints_for_agent
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_utils import verify_trainrun_satisfies_route_dag_constraints
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.data_types import RouteDAGConstraints


def generic_schedule_problem_description_for_rescheduling(
        schedule_trainruns: TrainrunDict,
        minimum_travel_time_dict: Dict[int, int],
        topo_dict: TopoDict,
        force_freeze: Dict[int, List[TrainrunWaypoint]],
        malfunction: ExperimentMalfunction,
        latest_arrival: int,
        max_window_size_from_earliest: int = np.inf
) -> ScheduleProblemDescription:
    """Derives the experiment freeze given the malfunction and optionally a
    force freeze from an Oracle. The node after the malfunction time has to be
    visited with an earliest constraint.

    Parameters
    ----------
    schedule_trainruns
        the schedule before the malfunction happened
    minimum_travel_time_dict
        the agent's speed (constant for every agent, different among agents)
    topo_dict
        the topos for the agents
    force_freeze
        waypoints the oracle told to pass by
    malfunction
        malfunction
    latest_arrival
        end of the global time window
    max_window_size_from_earliest
        maximum window size as offset from earliest. => "Cuts off latest at earliest + earliest_time_windows when doing
        back propagation of latest"

    Returns
    -------
    """
    spd = ScheduleProblemDescription(
        route_dag_constraints_dict={
            agent_id: _generic_route_dag_contraints_for_rescheduling(
                schedule_trainrun=schedule_trainruns[agent_id],
                minimum_travel_time=minimum_travel_time_dict[agent_id],
                topo=topo_dict[agent_id],
                force_freeze=force_freeze[agent_id],
                malfunction=malfunction,
                agent_id=agent_id,
                latest_arrival=latest_arrival,
                max_window_size_from_earliest=max_window_size_from_earliest
            )
            for agent_id in schedule_trainruns},
        topo_dict=topo_dict,
        minimum_travel_time_dict=minimum_travel_time_dict,
        max_episode_steps=latest_arrival,
        route_section_penalties=_extract_route_section_penalties(schedule_trainruns, topo_dict),
        weight_lateness_seconds=1
    )
    # TODO SIM-324 pull out verification
    for agent_id in spd.route_dag_constraints_dict:
        verify_consistency_of_route_dag_constraints_for_agent(
            agent_id=agent_id,
            topo=topo_dict[agent_id],
            route_dag_constraints=spd.route_dag_constraints_dict[agent_id],
            force_freeze=force_freeze[agent_id],
            malfunction=malfunction if malfunction.agent_id == agent_id else None,
            max_window_size_from_earliest=max_window_size_from_earliest
        )
        verify_trainrun_satisfies_route_dag_constraints(
            agent_id=agent_id,
            route_dag_constraints=spd.route_dag_constraints_dict[agent_id],
            scheduled_trainrun=list(
                filter(lambda trainrun_waypoint: trainrun_waypoint.scheduled_at <= malfunction.time_step,
                       schedule_trainruns[agent_id]))
        )
    return spd


def _extract_route_section_penalties(schedule_trainruns: TrainrunDict, topo_dict: TopoDict):
    """Penalize edges by 1 in the topology departing from scheduled
    trainrun."""
    route_section_penalties: RouteSectionPenaltiesDict = {}
    for agent_id, schedule_trainrun in schedule_trainruns.items():
        route_section_penalties[agent_id] = {}
        waypoints_in_schedule = {trainrun_waypoint.waypoint for trainrun_waypoint in schedule_trainrun}
        topo = topo_dict[agent_id]
        for edge in topo.edges:
            (from_waypoint, to_waypoint) = edge
            if from_waypoint in waypoints_in_schedule and to_waypoint not in waypoints_in_schedule:
                route_section_penalties[agent_id][edge] = 1
    return route_section_penalties


# TODO SIM-613 this should become Delta_0_running as in paper
def _generic_route_dag_constraints_for_rescheduling_agent_while_running(  # noqa: C901
        minimum_travel_time: int,
        topo: nx.DiGraph,
        force_freeze: List[TrainrunWaypoint],
        subdag_source: TrainrunWaypoint,
        latest_arrival: int,
        max_window_size_from_earliest: int = np.inf
) -> RouteDAGConstraints:
    """Construct route DAG constraints for this agent. Consider only case where
    malfunction happens during schedule or if there is a (force freeze from the
    oracle).

    Parameters
    ----------

    minimum_travel_time
        the constant cell running time of trains
    topo: nx.DiGraph
        the agent's route DAG without constraints
    force_freeze: List[TrainrunWaypoint]
        vertices that need be visited and be visited at the given time
    subdag_source: TrainrunWaypoint
        the entry point into the dag that needs to be visited (the vertex after malfunction that is delayed);
        scheduled_at is interpreted as earliest
    latest_arrival: int

    Returns
    -------
    RouteDAGConstraints
        constraints for the situation
    """

    # force freeze
    force_freeze_dict = {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
                         for trainrun_waypoint in force_freeze}
    force_freeze_waypoints_set = {trainrun_waypoint.waypoint for trainrun_waypoint in force_freeze}

    # remove duplicates but deterministic (hashes of dict)
    all_waypoints: List[Waypoint] = topo.nodes

    # initialize visit, earliest, latest
    freeze_visit = []
    freeze_visit_waypoint_set: Set[Waypoint] = set()
    reachable_earliest_dict: [Waypoint, int] = OrderedDict()
    reachable_latest_dict: [Waypoint, int] = OrderedDict()

    def _remove_from_reachable(waypoint):
        # design choice: we give no earliest/latest for banned!
        if waypoint in reachable_earliest_dict:
            reachable_earliest_dict.pop(waypoint)
        if waypoint in reachable_latest_dict:
            reachable_latest_dict.pop(waypoint)
        topo.remove_node(waypoint)

    # 1. sub dag source must be visited (point after malfunction)
    freeze_visit.append(subdag_source.waypoint)
    freeze_visit_waypoint_set.add(subdag_source.waypoint)
    reachable_earliest_dict[subdag_source.waypoint] = subdag_source.scheduled_at

    # 2. latest for sinks
    # there may be multiple vertices by which the last cell may be entered!
    sinks = list(get_sinks_for_topo(topo))
    for sink in sinks:
        reachable_latest_dict[sink] = latest_arrival

    # 3. visit, earliest and latest for force_freeze
    for trainrun_waypoint in force_freeze:
        reachable_earliest_dict[trainrun_waypoint.waypoint] = trainrun_waypoint.scheduled_at
        reachable_latest_dict[trainrun_waypoint.waypoint] = trainrun_waypoint.scheduled_at
        freeze_visit.append(trainrun_waypoint.waypoint)
        freeze_visit_waypoint_set.add(trainrun_waypoint.waypoint)

    # 4. ban all that are not reachable in topology given the force_freeze
    # TODO this seems not correct: subdag source must be part of
    must_be_visited = {trainrun_waypoint.waypoint for trainrun_waypoint in force_freeze}
    must_be_visited.add(subdag_source.waypoint)
    reachable_set = _get_reachable_given_frozen_set(must_be_visited=must_be_visited, topo=topo)
    for trainrun_waypoint in force_freeze:
        reachable_set.add(trainrun_waypoint.waypoint)

    # ban all that are not reachable in topology
    banned, banned_set = _collect_banned_as_not_reached(all_waypoints, force_freeze_waypoints_set, reachable_set)
    # design choice: we give no earliest/latest for banned!
    for waypoint in banned_set:
        _remove_from_reachable(waypoint)
    # TODO SIM-622 refactor!
    banned = []
    banned_set = set()

    # 5. propagate earliest and latest in the sub-DAG
    # N.B. we cannot move along paths since this we the order would play a role (SIM-260)
    force_freeze_earliest = force_freeze_dict.copy()
    force_freeze_earliest[subdag_source.waypoint] = subdag_source.scheduled_at

    propagate(
        force_freeze_earliest=set(force_freeze_earliest.keys()),
        force_freeze_latest=set(force_freeze_dict.keys()).union(get_sinks_for_topo(topo)),
        latest_arrival=latest_arrival,
        latest_dict=reachable_latest_dict,
        earliest_dict=reachable_earliest_dict,
        minimum_travel_time=minimum_travel_time,
        max_window_size_from_earliest=max_window_size_from_earliest,
        topo=topo
    )

    # 6. ban all waypoints that are reachable in the toplogy but not in time (i.e. where earliest > latest)
    to_remove = set()
    for waypoint in all_waypoints:
        if (waypoint not in reachable_earliest_dict or waypoint not in reachable_latest_dict or  # noqa: W504
                reachable_earliest_dict[waypoint] > reachable_latest_dict[waypoint]):
            to_remove.add(waypoint)
    for waypoint in to_remove:
        _remove_from_reachable(waypoint)

    return RouteDAGConstraints(
        freeze_visit=freeze_visit,
        freeze_earliest=reachable_earliest_dict,
        freeze_banned=banned,
        freeze_latest=reachable_latest_dict
    )


def _collect_banned_as_not_reached(all_waypoints: List[Waypoint],
                                   force_freeze_waypoints_set: Set[Waypoint],
                                   reachable_set: Set[Waypoint]):
    """Bans all that are not either in the forward or backward funnel of the
    freezed ones.

    Returns them as list for iteration and as set for containment test.
    """
    banned: List[Waypoint] = []
    banned_set: Set[Waypoint] = set()
    for waypoint in all_waypoints:
        if waypoint not in reachable_set:
            banned.append(waypoint)
            banned_set.add(waypoint)
            assert waypoint not in force_freeze_waypoints_set, f"{waypoint}"

    return banned, banned_set


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


def _generic_route_dag_contraints_for_rescheduling(
        schedule_trainrun: Trainrun,
        minimum_travel_time: int,
        topo: nx.DiGraph,
        force_freeze: List[TrainrunWaypoint],
        malfunction: ExperimentMalfunction,
        agent_id: int,
        latest_arrival: int,
        max_window_size_from_earliest: int
) -> RouteDAGConstraints:
    """Derives the experiment freeze given the malfunction and optionally a
    force freeze from an Oracle. The node after the malfunction time has to be
    visited with an earliest constraint.

    Parameters
    ----------
    schedule_trainrun
        the schedule before the malfunction happened
    minimum_travel_time
        the agent's speed (constant for every agent, different among agents)
    topo
        the topos for the agents
    force_freeze
        waypoints the oracle told to pass by
    malfunction
        malfunction
    latest_arrival
        end of the global time window
    max_window_size_from_earliest
        maximum window size as offset from earliest. => "Cuts off latest at earliest + earliest_time_windows when doing
        back propagation of latest"

    Returns
    -------
    """
    if (malfunction.time_step >= schedule_trainrun[0].scheduled_at and  # noqa: W504
        malfunction.time_step < schedule_trainrun[-1].scheduled_at) or force_freeze:
        rsp_logger.debug(f"_generic_route_dag_contraints_for_rescheduling (1) for {agent_id}")
        return _generic_route_dag_constraints_for_rescheduling_agent_while_running(
            minimum_travel_time=minimum_travel_time,
            topo=topo,
            force_freeze=force_freeze,
            subdag_source=get_delayed_trainrun_waypoint_after_malfunction(
                agent_id=agent_id,
                trainrun=schedule_trainrun,
                malfunction=malfunction,
                minimum_travel_time=minimum_travel_time
            ),
            latest_arrival=latest_arrival,
            max_window_size_from_earliest=max_window_size_from_earliest
        )

    # handle the special case of malfunction before scheduled start or after scheduled arrival of agent
    elif malfunction.time_step < schedule_trainrun[0].scheduled_at:
        rsp_logger.debug(f"_generic_route_dag_contraints_for_rescheduling (2) for {agent_id}")
        # TODO should this be release time instead of -1?
        freeze_latest = {sink: latest_arrival - 1 for sink in get_sinks_for_topo(topo)}
        freeze_earliest = {schedule_trainrun[0].waypoint: schedule_trainrun[0].scheduled_at}
        propagate(
            earliest_dict=freeze_earliest,
            latest_dict=freeze_latest,
            latest_arrival=latest_arrival,
            max_window_size_from_earliest=max_window_size_from_earliest,
            minimum_travel_time=minimum_travel_time,
            force_freeze_earliest={schedule_trainrun[0].waypoint},
            force_freeze_latest=set(get_sinks_for_topo(topo)),
            topo=topo,
        )
        route_dag_constraints = RouteDAGConstraints(
            freeze_visit=[],
            freeze_earliest=freeze_earliest,
            freeze_latest=freeze_latest,
            freeze_banned=[],
        )
        freeze: RouteDAGConstraints = route_dag_constraints
        # N.B. copy keys into new list (cannot delete keys while looping concurrently looping over them)
        waypoints: List[Waypoint] = list(freeze.freeze_earliest.keys())
        for waypoint in waypoints:
            if freeze.freeze_earliest[waypoint] > freeze.freeze_latest[waypoint]:
                del freeze.freeze_latest[waypoint]
                del freeze.freeze_earliest[waypoint]
                freeze.freeze_banned.append(waypoint)
        return freeze
    elif malfunction.time_step >= schedule_trainrun[-1].scheduled_at:
        rsp_logger.debug(f"_generic_route_dag_contraints_for_rescheduling (3) for {agent_id}")
        visited = {trainrun_waypoint.waypoint for trainrun_waypoint in schedule_trainrun}
        all_waypoints = topo.nodes
        return RouteDAGConstraints(
            freeze_visit=[trainrun_waypoint.waypoint for trainrun_waypoint in schedule_trainrun],
            freeze_earliest={trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
                             for trainrun_waypoint in schedule_trainrun},
            freeze_latest={trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
                           for trainrun_waypoint in schedule_trainrun},
            freeze_banned=[waypoint
                           for waypoint in all_waypoints
                           if waypoint not in visited],
        )
    else:
        raise Exception(f"Unexepcted state for agent {agent_id} malfunction {malfunction}")
