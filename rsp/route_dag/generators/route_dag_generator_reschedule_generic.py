from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Set

import networkx as nx
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.route_dag.generators.route_dag_generator_utils import get_delayed_trainrun_waypoint_after_malfunction
from rsp.route_dag.generators.route_dag_generator_utils import propagate_earliest
from rsp.route_dag.generators.route_dag_generator_utils import propagate_latest
from rsp.route_dag.generators.route_dag_generator_utils import verify_route_dag_constraints_for_agent
from rsp.route_dag.route_dag import get_sinks_for_topo
from rsp.route_dag.route_dag import RouteSectionPenaltiesDict
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.route_dag.route_dag import TopoDict
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.data_types import RouteDAGConstraints


def generic_route_dag_constraints_for_rescheduling(
        schedule_trainruns: TrainrunDict,
        minimum_travel_time_dict: Dict[int, int],
        topo_dict: TopoDict,
        force_freeze: Dict[int, List[TrainrunWaypoint]],
        malfunction: ExperimentMalfunction,
        latest_arrival: int
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

    Returns
    -------
    """
    route_dag_constraints_dict = {
        agent_id: _generic_route_dag_constraints_for_rescheduling_agent_while_running(
            minimum_travel_time=minimum_travel_time_dict[agent_id],
            topo=topo_dict[agent_id],
            force_freeze=force_freeze[agent_id],
            subdag_source=get_delayed_trainrun_waypoint_after_malfunction(
                agent_id=agent_id,
                trainrun=schedule_trainruns[agent_id],
                malfunction=malfunction
            ),
            latest_arrival=latest_arrival

        )
        for agent_id, schedule_trainrun in schedule_trainruns.items()
        # ---> handle them special
        #  - if not started -> everything open
        #  - if already done -> everything remains the same
        if (malfunction.time_step >= schedule_trainrun[0].scheduled_at and  # noqa: W504
            malfunction.time_step < schedule_trainrun[-1].scheduled_at) or force_freeze[agent_id]

    }

    # inconsistent data if malfunction agent is not impacted by the malfunction!
    if malfunction.agent_id not in route_dag_constraints_dict:
        raise Exception(f"agent {malfunction.agent_id} has malfunction {malfunction} "
                        f"before scheduled start {schedule_trainruns[malfunction.agent_id] if malfunction.agent_id in schedule_trainruns else None}. ")

    # handle the special case of malfunction before scheduled start or after scheduled arrival of agent
    for agent_id, schedule_trainrun in schedule_trainruns.items():
        if agent_id not in route_dag_constraints_dict:
            if malfunction.time_step < schedule_trainrun[0].scheduled_at:

                route_dag_constraints_dict[agent_id] = RouteDAGConstraints(
                    freeze_visit=[],
                    freeze_earliest=propagate_earliest(
                        banned_set=[],
                        earliest_dict={schedule_trainrun[0].waypoint: schedule_trainrun[0].scheduled_at},
                        minimum_travel_time=minimum_travel_time_dict[agent_id],
                        force_freeze_dict={},
                        subdag_source=schedule_trainrun[0],
                        topo=topo_dict[agent_id],
                    ),
                    freeze_latest=propagate_latest(
                        banned_set=[],
                        latest_dict={sink: latest_arrival - 1 for sink in get_sinks_for_topo(topo_dict[agent_id])},
                        latest_arrival=latest_arrival,
                        minimum_travel_time=minimum_travel_time_dict[agent_id],
                        force_freeze_dict={},
                        topo=topo_dict[agent_id],
                    ),
                    freeze_banned=[],
                )
                freeze: RouteDAGConstraints = route_dag_constraints_dict[agent_id]
                # N.B. copy keys into new list (cannot delete keys while looping concurrently looping over them)
                waypoints: List[Waypoint] = list(freeze.freeze_earliest.keys())
                for waypoint in waypoints:
                    if freeze.freeze_earliest[waypoint] > freeze.freeze_latest[waypoint]:
                        del freeze.freeze_latest[waypoint]
                        del freeze.freeze_earliest[waypoint]
                        freeze.freeze_banned.append(waypoint)
            elif malfunction.time_step >= schedule_trainrun[-1].scheduled_at:
                visited = {trainrun_waypoint.waypoint for trainrun_waypoint in schedule_trainrun}
                all_waypoints = topo_dict[agent_id].nodes
                route_dag_constraints_dict[agent_id] = RouteDAGConstraints(
                    freeze_visit=[trainrun_waypoint.waypoint for trainrun_waypoint in schedule_trainrun],
                    freeze_earliest={trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
                                     for trainrun_waypoint in schedule_trainrun},
                    freeze_latest={trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
                                   for trainrun_waypoint in schedule_trainrun},
                    freeze_banned=[waypoint
                                   for waypoint in all_waypoints
                                   if waypoint not in visited],
                )
    # TODO SIM-324 pull out verification
    for agent_id in route_dag_constraints_dict:
        verify_route_dag_constraints_for_agent(
            agent_id=agent_id,
            topo=topo_dict[agent_id],
            route_dag_constraints=route_dag_constraints_dict[agent_id],
            force_freeze=force_freeze[agent_id],
            malfunction=malfunction if malfunction.agent_id == agent_id else None,
            scheduled_trainrun=list(
                filter(lambda trainrun_waypoint: trainrun_waypoint.scheduled_at <= malfunction.time_step,
                       schedule_trainruns[agent_id]))
        )

    route_section_penalties = extract_route_section_penalties(schedule_trainruns, topo_dict)

    return ScheduleProblemDescription(
        route_dag_constraints_dict=route_dag_constraints_dict,
        topo_dict=topo_dict,
        minimum_travel_time_dict=minimum_travel_time_dict,
        max_episode_steps=latest_arrival,
        route_section_penalties=route_section_penalties,
        weight_lateness_seconds=1
    )


def extract_route_section_penalties(schedule_trainruns: TrainrunDict, topo_dict: TopoDict):
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


def _generic_route_dag_constraints_for_rescheduling_agent_while_running(
        minimum_travel_time: int,
        topo: nx.DiGraph,
        force_freeze: List[TrainrunWaypoint],
        subdag_source: TrainrunWaypoint,
        latest_arrival: int

) -> RouteDAGConstraints:
    """Construct route DAG constraints for this agent. Consider only case where
    malfunction happens during schedule or if there is a (force freeze from the
    oracle).

    Parameters
    ----------

    minimum_travel_time
        the constant cell running time of trains
    agent_paths
        the paths spanning the agent's route DAG.
    force_freeze
        vertices that need be visited and be visited at the given time
    subdag_source
        the entry point into the dag that needs to be visited (the vertex after malfunction that is delayed)
    subdag_targets

    Returns
    -------
    """

    # force freeze in Delta re-scheduling
    force_freeze_dict = {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
                         for trainrun_waypoint in force_freeze}
    force_freeze_waypoints_set = {trainrun_waypoint.waypoint for trainrun_waypoint in force_freeze}

    # remove duplicates but deterministc (hashes of dict)
    all_waypoints: List[Waypoint] = topo.nodes

    # span a sub-dag for the problem
    # - for full scheduling, this is source vertex and time 0
    # - for full re-scheduling, this is the next waypoint after the malfunction (delayed for the agent in malfunction)
    # - for delta re-scheduling, if the Oracle tells that more can be freezed than up to malfunction, we use this!
    #   If the force freeze is not contiguous, we need to consider what can be reached given the freezes.

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

    # sub dag source must be visited (point after malfunction)
    freeze_visit.append(subdag_source.waypoint)
    freeze_visit_waypoint_set.add(subdag_source.waypoint)
    reachable_earliest_dict[subdag_source.waypoint] = subdag_source.scheduled_at

    # there may be multiple vertices by which the last cell may be entered!
    sinks = get_sinks_for_topo(topo)
    for sink in sinks:
        # TODO SIM-322 hard-coded assumption
        # -1 for occupying the cell for one time step!
        reachable_latest_dict[sink] = latest_arrival - 1

    for trainrun_waypoint in force_freeze:
        reachable_earliest_dict[trainrun_waypoint.waypoint] = trainrun_waypoint.scheduled_at
        reachable_latest_dict[trainrun_waypoint.waypoint] = trainrun_waypoint.scheduled_at
        freeze_visit.append(trainrun_waypoint.waypoint)
        freeze_visit_waypoint_set.add(trainrun_waypoint.waypoint)

    reachable_set = _get_reachable_given_frozen_set(force_freeze=force_freeze, topo=topo)
    reachable_set.add(subdag_source.waypoint)
    for trainrun_waypoint in force_freeze:
        reachable_set.add(trainrun_waypoint.waypoint)

    # ban all that are not reachable in topology
    banned, banned_set = _collect_banned_as_not_reached(all_waypoints, force_freeze_waypoints_set, reachable_set)
    # design choice: we give no earliest/latest for banned!
    for waypoint in banned_set:
        _remove_from_reachable(waypoint)

    # collect earliest and latest in the sub-DAG
    # N.B. we cannot move along paths since this we the order would play a role (SIM-260)
    propagate_earliest(banned_set, reachable_earliest_dict, force_freeze_dict, minimum_travel_time, subdag_source,
                       topo)
    propagate_latest(banned_set, force_freeze_dict, latest_arrival, reachable_latest_dict, minimum_travel_time, topo)

    # ban all waypoints that are reachable in the toplogy but not in time (i.e. where earliest > latest)
    for waypoint in all_waypoints:
        if (waypoint not in reachable_earliest_dict or waypoint not in reachable_latest_dict or  # noqa: W504
            reachable_earliest_dict[waypoint] > reachable_latest_dict[waypoint]) \
                and waypoint not in banned_set:
            banned.append(waypoint)
            banned_set.add(waypoint)
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
                                    force_freeze: List[TrainrunWaypoint]) -> Set[Waypoint]:
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
    force_freeze
        the waypoints that must be visited

    Returns
    -------
    """
    forward_reachable = {waypoint: set() for waypoint in topo.nodes}
    backward_reachable = {waypoint: set() for waypoint in topo.nodes}

    # collect descendants and ancestors of freeze
    for trainrun_waypoint in force_freeze:
        forward_reachable[trainrun_waypoint.waypoint] = set(nx.descendants(topo, trainrun_waypoint.waypoint))
        backward_reachable[trainrun_waypoint.waypoint] = set(nx.ancestors(topo, trainrun_waypoint.waypoint))

    # reflexivity: add waypoint to its own closure (needed for building reachable_set below)
    for waypoint in topo.nodes:
        forward_reachable[waypoint].add(waypoint)
        backward_reachable[waypoint].add(waypoint)

    # reachable are only those that are either in the forward or backward "funnel" of all force freezes!
    reachable_set = set(topo.nodes)
    for trainrun_waypoint in force_freeze:
        waypoint = trainrun_waypoint.waypoint
        forward_and_backward_reachable = forward_reachable[waypoint].union(backward_reachable[waypoint])
        reachable_set.intersection_update(forward_and_backward_reachable)
    return reachable_set
