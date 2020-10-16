import logging
import pprint
from typing import Dict

import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from rsp.scheduling.propagate import _get_delayed_trainrun_waypoint_after_malfunction
from rsp.scheduling.propagate import propagate
from rsp.scheduling.propagate import verify_consistency_of_route_dag_constraints_for_agent
from rsp.scheduling.propagate import verify_trainrun_satisfies_route_dag_constraints
from rsp.scheduling.scheduling_problem import get_sinks_for_topo
from rsp.scheduling.scheduling_problem import RouteDAGConstraints
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.scheduling.scheduling_problem import TopoDict
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.step_03_run.route_dag_constraints.scoper_zero import _extract_route_section_penalties
from rsp.utils.data_types import RouteDAGConstraintsDict
from rsp.utils.rsp_logger import rsp_logger

_pp = pprint.PrettyPrinter(indent=4)


def scoper_perfect(
    agent_id: int,
    # pytorch convention for in-place operations: postfixed with underscore.
    topo_: nx.DiGraph,
    schedule_trainrun: Trainrun,
    full_reschedule_trainrun: Trainrun,
    malfunction: ExperimentMalfunction,
    minimum_travel_time: int,
    latest_arrival: int,
    max_window_size_from_earliest: int = np.inf,
):
    """"Scoper perfect":

    - allow only paths either in schedule or re-schedule
    - if the same in location and time in schedule and re-schedule -> stay (implicitly includes everything up to malfunction)

    Caveat: In contrast to other methods, the topo is not modified.

    Parameters
    ----------
    agent_id
    topo_
    schedule_trainrun
    full_reschedule_trainrun
    malfunction
    minimum_travel_time
    latest_arrival
    max_window_size_from_earliest

    Returns
    -------
    """
    waypoints_same_location_and_time = {trainrun_waypoint.waypoint for trainrun_waypoint in set(full_reschedule_trainrun).intersection(set(schedule_trainrun))}
    if rsp_logger.isEnabledFor(logging.DEBUG):
        rsp_logger.debug(f"waypoints_same_location_and_time={waypoints_same_location_and_time}")

    schedule_waypoints = {trainrun_waypoint.waypoint for trainrun_waypoint in schedule_trainrun}
    reschedule_waypoints = {trainrun_waypoint.waypoint for trainrun_waypoint in full_reschedule_trainrun}
    assert schedule_waypoints.issubset(topo_.nodes), f"{schedule_waypoints} {topo_.nodes} {schedule_waypoints.difference(topo_.nodes)}"
    assert reschedule_waypoints.issubset(topo_.nodes), f"{reschedule_waypoints} {topo_.nodes} {reschedule_waypoints.difference(topo_.nodes)}"

    waypoints_same_location = list(schedule_waypoints.intersection(reschedule_waypoints))
    if rsp_logger.isEnabledFor(logging.DEBUG):
        rsp_logger.debug(f"waypoints_same_location={waypoints_same_location}")

    topo_out = topo_.copy()
    to_remove = set(topo_out.nodes).difference(schedule_waypoints.union(reschedule_waypoints))
    topo_out.remove_nodes_from(to_remove)

    earliest_dict = {}
    latest_dict = {}
    schedule = {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at for trainrun_waypoint in schedule_trainrun}
    for v in waypoints_same_location_and_time:
        earliest_dict[v] = schedule[v]
        latest_dict[v] = schedule[v]

    sinks = list(get_sinks_for_topo(topo_out))
    for sink in sinks:
        if sink in waypoints_same_location_and_time:
            continue
        latest_dict[sink] = latest_arrival

    # this is v_2 in paper
    delayed_trainrun_waypoint_after_malfunction = _get_delayed_trainrun_waypoint_after_malfunction(
        agent_id=agent_id, trainrun=schedule_trainrun, malfunction=malfunction, minimum_travel_time=minimum_travel_time
    )
    earliest_dict[delayed_trainrun_waypoint_after_malfunction.waypoint] = delayed_trainrun_waypoint_after_malfunction.scheduled_at

    force_earliest = waypoints_same_location_and_time.union({delayed_trainrun_waypoint_after_malfunction.waypoint})
    assert set(force_earliest).issubset(topo_out.nodes), (
        f"{force_earliest.difference(topo_out.nodes)} - {set(topo_out.nodes).difference(force_earliest)} // "
        f"{set(topo_out.nodes).intersection(force_earliest)} // {delayed_trainrun_waypoint_after_malfunction}"
    )
    propagate(
        earliest_dict=earliest_dict,
        latest_dict=latest_dict,
        topo=topo_out,
        force_earliest=force_earliest,
        force_latest=waypoints_same_location_and_time.union(sinks),
        must_be_visited=waypoints_same_location,
        minimum_travel_time=minimum_travel_time,
        latest_arrival=latest_arrival,
        max_window_size_from_earliest=max_window_size_from_earliest,
    )
    return earliest_dict, latest_dict, topo_out


def scoper_perfect_for_all_agents(
    full_reschedule_trainrun_dict: TrainrunDict,
    malfunction: ExperimentMalfunction,
    minimum_travel_time_dict: Dict[int, int],
    max_episode_steps: int,
    delta_perfect_reschedule_topo_dict_: TopoDict,
    schedule_trainrun_dict: TrainrunDict,
    weight_route_change: int,
    weight_lateness_seconds: int,
    max_window_size_from_earliest: int = np.inf,
) -> ScheduleProblemDescription:
    """The scoper perfect only opens up the differences between the schedule
    and the imaginary re-schedule. It gives no additional routing flexibility!

    Parameters
    ----------

    full_reschedule_trainrun_dict: TrainrunDict
        the magic information of the full re-schedule
    malfunction: ExperimentMalfunction
        the malfunction; used to determine the waypoint after the malfunction
    minimum_travel_time_dict: Dict[int,int]
        the minimumum travel times for the agents
    max_episode_steps:
        latest arrival
    delta_perfect_reschedule_topo_dict_:
        the topologies used for scheduling
    schedule_trainrun_dict: TrainrunDict
        the schedule S0
    max_window_size_from_earliest: int
        maximum window size as offset from earliest. => "Cuts off latest at earliest + earliest_time_windows when doing
        back propagation of latest"
    weight_lateness_seconds
        how much
    weight_route_change
    Returns
    -------
    ScheduleProblemDesccription
    """
    freeze_dict: RouteDAGConstraintsDict = {}
    topo_dict: TopoDict = {}
    for agent_id in schedule_trainrun_dict.keys():
        earliest_dict, latest_dict, topo = scoper_perfect(
            agent_id=agent_id,
            topo_=delta_perfect_reschedule_topo_dict_[agent_id],
            schedule_trainrun=schedule_trainrun_dict[agent_id],
            full_reschedule_trainrun=full_reschedule_trainrun_dict[agent_id],
            malfunction=malfunction,
            minimum_travel_time=minimum_travel_time_dict[agent_id],
            latest_arrival=max_episode_steps,
            max_window_size_from_earliest=max_window_size_from_earliest,
        )
        freeze_dict[agent_id] = RouteDAGConstraints(earliest=earliest_dict, latest=latest_dict)
        topo_dict[agent_id] = topo

    # TODO SIM-324 pull out verification
    for agent_id, _ in freeze_dict.items():
        verify_consistency_of_route_dag_constraints_for_agent(
            agent_id=agent_id,
            route_dag_constraints=freeze_dict[agent_id],
            topo=topo_dict[agent_id],
            malfunction=malfunction,
            max_window_size_from_earliest=max_window_size_from_earliest,
        )
        # re-schedule train run must be open in route dag constraints
        verify_trainrun_satisfies_route_dag_constraints(
            agent_id=agent_id, route_dag_constraints=freeze_dict[agent_id], scheduled_trainrun=full_reschedule_trainrun_dict[agent_id]
        )

    return ScheduleProblemDescription(
        route_dag_constraints_dict=freeze_dict,
        minimum_travel_time_dict=minimum_travel_time_dict,
        topo_dict=topo_dict,
        max_episode_steps=max_episode_steps,
        route_section_penalties=_extract_route_section_penalties(
            schedule_trainruns=schedule_trainrun_dict, topo_dict=topo_dict, weight_route_change=weight_route_change
        ),
        weight_lateness_seconds=weight_lateness_seconds,
    )
