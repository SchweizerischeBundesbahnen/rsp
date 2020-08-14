import pprint
from typing import Dict

import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.logger import rsp_logger
from rsp.schedule_problem_description.data_types_and_utils import get_sinks_for_topo
from rsp.schedule_problem_description.data_types_and_utils import RouteDAGConstraints
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.schedule_problem_description.data_types_and_utils import TopoDict
from rsp.schedule_problem_description.route_dag_constraints.delta_zero import _extract_route_section_penalties
from rsp.schedule_problem_description.route_dag_constraints.propagate import _get_delayed_trainrun_waypoint_after_malfunction
from rsp.schedule_problem_description.route_dag_constraints.propagate import propagate
from rsp.schedule_problem_description.route_dag_constraints.propagate import verify_consistency_of_route_dag_constraints_for_agent
from rsp.schedule_problem_description.route_dag_constraints.propagate import verify_trainrun_satisfies_route_dag_constraints
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.data_types import RouteDAGConstraintsDict

_pp = pprint.PrettyPrinter(indent=4)


def perfect_oracle(
        agent_id: int,
        topo: nx.DiGraph,
        schedule_trainrun: Trainrun,
        full_reschedule_trainrun: Trainrun,
        malfunction: ExperimentMalfunction,
        minimum_travel_time: int,
        latest_arrival: int,
        max_window_size_from_earliest: int = np.inf
):
    """"Perfect oracle":

    - allow only paths either in schedule or re-schedule
    - if the same in location and time in schedule and re-schedule -> stay (implicitly includes everything up to malfunction)

    Caveat: In contrast to other methods, the topo is not modified.

    Parameters
    ----------
    agent_id
    topo
    schedule_trainrun
    full_reschedule_trainrun
    malfunction
    minimum_travel_time
    latest_arrival
    max_window_size_from_earliest

    Returns
    -------
    """
    delta_same_location_and_time = {trainrun_waypoint.waypoint for trainrun_waypoint in set(full_reschedule_trainrun).intersection(set(schedule_trainrun))}
    rsp_logger.info(f"delta_same_location_and_time={delta_same_location_and_time}")

    schedule_waypoints = {trainrun_waypoint.waypoint for trainrun_waypoint in schedule_trainrun}
    reschedule_waypoints = {trainrun_waypoint.waypoint for trainrun_waypoint in full_reschedule_trainrun}
    assert schedule_waypoints.issubset(topo.nodes), f"{schedule_waypoints} {topo.nodes} {schedule_waypoints.difference(topo.nodes)}"
    assert reschedule_waypoints.issubset(topo.nodes), f"{reschedule_waypoints} {topo.nodes} {reschedule_waypoints.difference(topo.nodes)}"

    delta_same_location = list(schedule_waypoints.intersection(reschedule_waypoints))
    rsp_logger.info(f"delta_same_location={delta_same_location}")

    topo_out = topo.copy()
    to_remove = set(topo_out.nodes).difference(schedule_waypoints.union(reschedule_waypoints))
    topo_out.remove_nodes_from(to_remove)

    earliest_dict = {}
    latest_dict = {}
    schedule = {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at for trainrun_waypoint in schedule_trainrun}
    for v in delta_same_location_and_time:
        earliest_dict[v] = schedule[v]
        latest_dict[v] = schedule[v]

    sinks = list(get_sinks_for_topo(topo_out))
    for sink in sinks:
        if sink in delta_same_location_and_time:
            continue
        latest_dict[sink] = latest_arrival

    # this is v_2 in paper
    delayed_trainrun_waypoint_after_malfunction = _get_delayed_trainrun_waypoint_after_malfunction(
        agent_id=agent_id,
        trainrun=schedule_trainrun,
        malfunction=malfunction,
        minimum_travel_time=minimum_travel_time
    )
    earliest_dict[delayed_trainrun_waypoint_after_malfunction.waypoint] = delayed_trainrun_waypoint_after_malfunction.scheduled_at

    force_freeze_earliest = delta_same_location_and_time.union({delayed_trainrun_waypoint_after_malfunction.waypoint})
    assert set(force_freeze_earliest).issubset(topo_out.nodes), \
        f"{force_freeze_earliest.difference(topo_out.nodes)} - {set(topo_out.nodes).difference(force_freeze_earliest)} // " \
        f"{set(topo_out.nodes).intersection(force_freeze_earliest)} // {delayed_trainrun_waypoint_after_malfunction}"
    propagate(
        earliest_dict=earliest_dict,
        latest_dict=latest_dict,
        topo=topo_out,
        force_freeze_earliest=force_freeze_earliest,
        force_freeze_latest=delta_same_location_and_time.union(sinks),
        must_be_visited=delta_same_location,
        minimum_travel_time=minimum_travel_time,
        latest_arrival=latest_arrival,
        max_window_size_from_earliest=max_window_size_from_earliest
    )
    return earliest_dict, latest_dict, topo_out


def perfect_oracle_for_all_agents(
        full_reschedule_trainrun_dict: TrainrunDict,
        malfunction: ExperimentMalfunction,
        minimum_travel_time_dict: Dict[int, int],
        max_episode_steps: int,
        schedule_topo_dict: TopoDict,
        schedule_trainrun_dict: TrainrunDict,
        max_window_size_from_earliest: int = np.inf) -> ScheduleProblemDescription:
    """The perfect oracle only opens up the differences between the schedule
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
    schedule_topo_dict:
        the topologies used for scheduling
    schedule_trainrun_dict: TrainrunDict
        the schedule S0
    max_window_size_from_earliest: int
        maximum window size as offset from earliest. => "Cuts off latest at earliest + earliest_time_windows when doing
        back propagation of latest"

    Returns
    -------
    ScheduleProblemDesccription
    """
    freeze_dict: RouteDAGConstraintsDict = {}
    topo_dict: TopoDict = {}
    for agent_id in schedule_trainrun_dict.keys():
        earliest_dict, latest_dict, topo = perfect_oracle(
            agent_id=agent_id,
            topo=schedule_topo_dict[agent_id],
            schedule_trainrun=schedule_trainrun_dict[agent_id],
            full_reschedule_trainrun=full_reschedule_trainrun_dict[agent_id],
            malfunction=malfunction,
            minimum_travel_time=minimum_travel_time_dict[agent_id],
            latest_arrival=max_episode_steps,
            max_window_size_from_earliest=max_window_size_from_earliest

        )
        freeze_dict[agent_id] = RouteDAGConstraints(
            freeze_visit=[],
            freeze_banned=[],
            freeze_earliest=earliest_dict,
            freeze_latest=latest_dict
        )
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
            agent_id=agent_id,
            route_dag_constraints=freeze_dict[agent_id],
            scheduled_trainrun=full_reschedule_trainrun_dict[agent_id]
        )

    return ScheduleProblemDescription(
        route_dag_constraints_dict=freeze_dict,
        minimum_travel_time_dict=minimum_travel_time_dict,
        topo_dict=topo_dict,
        max_episode_steps=max_episode_steps,
        route_section_penalties=_extract_route_section_penalties(schedule_trainrun_dict, topo_dict),
        weight_lateness_seconds=1
    )
