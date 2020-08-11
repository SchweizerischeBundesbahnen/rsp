from typing import List
from typing import Set

import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.logger import rsp_logger
from rsp.schedule_problem_description.data_types_and_utils import get_sinks_for_topo
from rsp.schedule_problem_description.data_types_and_utils import RouteSectionPenaltiesDict
from rsp.schedule_problem_description.data_types_and_utils import TopoDict
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_utils import _get_delayed_trainrun_waypoint_after_malfunction
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_utils import propagate
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.data_types import RouteDAGConstraints


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


# TODO SIM-613 remove force_freeze param
def delta_zero_running(  # noqa: C901
        agent_id: int,
        schedule_trainrun: Trainrun,
        malfunction: ExperimentMalfunction,
        minimum_travel_time: int,
        topo: nx.DiGraph,
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
    latest_arrival: int

    Returns
    -------
    RouteDAGConstraints
        constraints for the situation
    """
    must_be_visited = {
        trainrun_waypoint.waypoint
        for trainrun_waypoint in schedule_trainrun
        if trainrun_waypoint.scheduled_at <= malfunction.time_step
    }
    earliest_dict = {
        trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
        for trainrun_waypoint in schedule_trainrun
        if trainrun_waypoint.scheduled_at <= malfunction.time_step

    }
    latest_dict = earliest_dict.copy()

    subdag_source = _get_delayed_trainrun_waypoint_after_malfunction(
        agent_id=agent_id,
        trainrun=schedule_trainrun,
        malfunction=malfunction,
        minimum_travel_time=minimum_travel_time
    )
    must_be_visited.add(subdag_source.waypoint)

    earliest_dict[subdag_source.waypoint] = subdag_source.scheduled_at
    if subdag_source.waypoint in latest_dict and subdag_source.scheduled_at > malfunction.time_step:
        latest_dict.pop(subdag_source.waypoint)

    for sink in get_sinks_for_topo(topo):
        if sink in must_be_visited:
            continue
        latest_dict[sink] = latest_arrival

    propagate(
        earliest_dict=earliest_dict,
        latest_dict=latest_dict,
        topo=topo,
        force_freeze_earliest=set(earliest_dict.keys()),
        force_freeze_latest=set(latest_dict.keys()),
        must_be_visited=must_be_visited,
        minimum_travel_time=minimum_travel_time,
        latest_arrival=latest_arrival,
        max_window_size_from_earliest=max_window_size_from_earliest
    )

    return RouteDAGConstraints(
        # TODO SIM-613 remove freeze_visit from RouteDAGConstraints?
        freeze_visit=must_be_visited,
        freeze_earliest=earliest_dict,
        # TODO SIM-613 remove freeze_visit from RouteDAGConstraints?
        freeze_banned=set(),
        freeze_latest=latest_dict
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


def delta_zero(
        schedule_trainrun: Trainrun,
        minimum_travel_time: int,
        topo: nx.DiGraph,
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
            malfunction.time_step < schedule_trainrun[-1].scheduled_at):
        rsp_logger.debug(f"_generic_route_dag_contraints_for_rescheduling (1) for {agent_id}: while running")
        return delta_zero_running(
            agent_id=agent_id,
            schedule_trainrun=schedule_trainrun,
            malfunction=malfunction,
            minimum_travel_time=minimum_travel_time,
            topo=topo,
            latest_arrival=latest_arrival,
            max_window_size_from_earliest=max_window_size_from_earliest
        )

    # handle the special case of malfunction before scheduled start or after scheduled arrival of agent
    elif malfunction.time_step < schedule_trainrun[0].scheduled_at:
        rsp_logger.debug(f"_generic_route_dag_contraints_for_rescheduling (2) for {agent_id}: malfunction before schedule start")
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
            must_be_visited=set(),
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
        rsp_logger.debug(f"_generic_route_dag_contraints_for_rescheduling (3) for {agent_id}: malfunction after scheduled arrival")
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
