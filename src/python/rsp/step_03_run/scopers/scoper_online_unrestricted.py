from typing import Dict
from typing import List

import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.scheduling.propagate import _get_delayed_trainrun_waypoint_after_malfunction
from rsp.scheduling.propagate import propagate
from rsp.scheduling.propagate import verify_consistency_of_route_dag_constraints_for_agent
from rsp.scheduling.propagate import verify_trainrun_satisfies_route_dag_constraints
from rsp.scheduling.scheduling_problem import get_sinks_for_topo
from rsp.scheduling.scheduling_problem import RouteDAGConstraints
from rsp.scheduling.scheduling_problem import RouteSectionPenaltiesDict
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.scheduling.scheduling_problem import TopoDict
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.utils.rsp_logger import rsp_logger
from rsp.utils.rsp_logger import VERBOSE


def _extract_route_section_penalties(schedule_trainruns: TrainrunDict, topo_dict: TopoDict, weight_route_change: int):
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
                route_section_penalties[agent_id][edge] = weight_route_change
    return route_section_penalties


def scoper_online_unrestricted_running(
    agent_id: int,
    schedule_trainrun: Trainrun,
    malfunction: ExperimentMalfunction,
    minimum_travel_time: int,
    topo: nx.DiGraph,
    latest_arrival: int,
    max_window_size_from_earliest: int = np.inf,
) -> RouteDAGConstraints:
    """Construct route DAG constraints for this agent. Consider only case where
    malfunction happens during schedule or if there is a (force freeze from the
    scoper).

    Parameters
    ----------
    malfunction
    schedule_trainrun
    agent_id
    minimum_travel_time
        the constant cell running time of trains
    topo: nx.DiGraph
        the agent's route DAG without constraints
    latest_arrival: int
    max_window_size_from_earliest

    Returns
    -------
    RouteDAGConstraints
        constraints for the situation
    """
    must_be_visited = {trainrun_waypoint.waypoint for trainrun_waypoint in schedule_trainrun if trainrun_waypoint.scheduled_at <= malfunction.time_step}
    earliest_dict = {
        trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
        for trainrun_waypoint in schedule_trainrun
        if trainrun_waypoint.scheduled_at <= malfunction.time_step
    }
    latest_dict = earliest_dict.copy()

    subdag_source = _get_delayed_trainrun_waypoint_after_malfunction(
        agent_id=agent_id, trainrun=schedule_trainrun, malfunction=malfunction, minimum_travel_time=minimum_travel_time
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
        force_earliest=set(earliest_dict.keys()),
        force_latest=set(latest_dict.keys()),
        must_be_visited=must_be_visited,
        minimum_travel_time=minimum_travel_time,
        latest_arrival=latest_arrival,
        max_window_size_from_earliest=max_window_size_from_earliest,
    )

    return RouteDAGConstraints(earliest=earliest_dict, latest=latest_dict)


def scoper_online_unrestricted(
    schedule_trainrun: Trainrun,
    minimum_travel_time: int,
    topo_: nx.DiGraph,
    malfunction: ExperimentMalfunction,
    agent_id: int,
    latest_arrival: int,
    max_window_size_from_earliest: int,
) -> RouteDAGConstraints:
    """Derives the `RouteDAGConstraints` given the malfunction. The node after
    the malfunction time has to be visited with an earliest constraint. Given
    the past up to the malfunction, nodes not reachable in space or time, are
    removed.

    Parameters
    ----------

    schedule_trainrun
        the schedule before the malfunction happened
    minimum_travel_time
        the agent's speed (constant for every agent, different among agents)
    topo_
        the topos for the agents
    malfunction
        malfunction
    agent_id
        which agent for
    latest_arrival
        end of the global time window
    max_window_size_from_earliest
        maximum window size as offset from earliest. => "Cuts off latest at earliest + earliest_time_windows when doing
        back propagation of latest"

    Returns
    -------
    RouteDAGConstraints
    """
    if malfunction.time_step >= schedule_trainrun[0].scheduled_at and malfunction.time_step < schedule_trainrun[-2].scheduled_at:  # noqa: W504
        rsp_logger.log(level=VERBOSE, msg=f"_generic_route_dag_contraints_for_rescheduling (1) for {agent_id}: while running")
        return scoper_online_unrestricted_running(
            agent_id=agent_id,
            schedule_trainrun=schedule_trainrun,
            malfunction=malfunction,
            minimum_travel_time=minimum_travel_time,
            topo=topo_,
            latest_arrival=latest_arrival,
            max_window_size_from_earliest=max_window_size_from_earliest,
        )

    # handle the special case of malfunction before scheduled start or after scheduled arrival of agent
    elif malfunction.time_step < schedule_trainrun[0].scheduled_at:
        rsp_logger.log(level=VERBOSE, msg=f"_generic_route_dag_contraints_for_rescheduling (2) for {agent_id}: malfunction before schedule start")
        latest = {sink: latest_arrival for sink in get_sinks_for_topo(topo_)}
        earliest = {schedule_trainrun[0].waypoint: schedule_trainrun[0].scheduled_at}
        propagate(
            earliest_dict=earliest,
            latest_dict=latest,
            latest_arrival=latest_arrival,
            max_window_size_from_earliest=max_window_size_from_earliest,
            minimum_travel_time=minimum_travel_time,
            force_earliest={schedule_trainrun[0].waypoint},
            force_latest=set(get_sinks_for_topo(topo_)),
            must_be_visited=set(),
            topo=topo_,
        )
        route_dag_constraints = RouteDAGConstraints(earliest=earliest, latest=latest,)
        freeze: RouteDAGConstraints = route_dag_constraints
        # N.B. copy keys into new list (cannot delete keys while looping concurrently looping over them)
        waypoints: List[Waypoint] = list(freeze.earliest.keys())
        for waypoint in waypoints:
            if freeze.earliest[waypoint] > freeze.latest[waypoint]:
                del freeze.latest[waypoint]
                del freeze.earliest[waypoint]
        return freeze
    elif malfunction.time_step >= schedule_trainrun[-2].scheduled_at:
        rsp_logger.log(level=VERBOSE, msg=f"_generic_route_dag_contraints_for_rescheduling (3) for {agent_id}: malfunction after scheduled arrival")
        visited = {trainrun_waypoint.waypoint for trainrun_waypoint in schedule_trainrun}
        topo_.remove_nodes_from(set(topo_.nodes).difference(visited))
        return RouteDAGConstraints(
            earliest={trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at for trainrun_waypoint in schedule_trainrun},
            latest={trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at for trainrun_waypoint in schedule_trainrun},
        )
    else:
        raise Exception(f"Unexepcted state for agent {agent_id} malfunction {malfunction}")


def scoper_online_unrestricted_for_all_agents(
    malfunction: ExperimentMalfunction,
    schedule_trainruns: TrainrunDict,
    minimum_travel_time_dict: Dict[int, int],
    topo_dict_: Dict[int, nx.DiGraph],
    latest_arrival: int,
    weight_route_change: int,
    weight_lateness_seconds: int,
    max_window_size_from_earliest: int = np.inf,
) -> ScheduleProblemDescription:
    """Returns the experiment freeze for the full re-scheduling problem. Wraps
    the generic freeze by freezing everything up to and including the
    malfunction.

    See param description there.
    """
    spd = ScheduleProblemDescription(
        route_dag_constraints_dict={
            agent_id: scoper_online_unrestricted(
                schedule_trainrun=schedule_trainruns[agent_id],
                minimum_travel_time=minimum_travel_time_dict[agent_id],
                topo_=topo_dict_[agent_id],
                malfunction=malfunction,
                agent_id=agent_id,
                latest_arrival=latest_arrival,
                max_window_size_from_earliest=max_window_size_from_earliest,
            )
            for agent_id in schedule_trainruns
        },
        topo_dict=topo_dict_,
        minimum_travel_time_dict=minimum_travel_time_dict,
        max_episode_steps=latest_arrival,
        route_section_penalties=_extract_route_section_penalties(schedule_trainruns, topo_dict_, weight_route_change),
        weight_lateness_seconds=weight_lateness_seconds,
    )
    # TODO SIM-324 pull out verification
    for agent_id in spd.route_dag_constraints_dict:
        verify_consistency_of_route_dag_constraints_for_agent(
            agent_id=agent_id,
            topo=spd.topo_dict[agent_id],
            route_dag_constraints=spd.route_dag_constraints_dict[agent_id],
            malfunction=malfunction if malfunction.agent_id == agent_id else None,
            max_window_size_from_earliest=max_window_size_from_earliest,
        )
        verify_trainrun_satisfies_route_dag_constraints(
            agent_id=agent_id,
            route_dag_constraints=spd.route_dag_constraints_dict[agent_id],
            scheduled_trainrun=list(filter(lambda trainrun_waypoint: trainrun_waypoint.scheduled_at <= malfunction.time_step, schedule_trainruns[agent_id])),
        )
    return spd
