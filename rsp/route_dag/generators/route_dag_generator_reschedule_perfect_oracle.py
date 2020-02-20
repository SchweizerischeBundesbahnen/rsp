import pprint
from typing import Dict
from typing import Set
from typing import Tuple

import networkx as nx
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.route_dag.generators.route_dag_generator_reschedule_generic import \
    generic_route_dag_constraints_for_rescheduling
from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.route_dag.route_dag import TopoDict
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.data_types import RouteDAGConstraints
from rsp.utils.data_types import RouteDAGConstraintsDict

_pp = pprint.PrettyPrinter(indent=4)


def perfect_oracle(
        full_reschedule_trainrun_waypoints_dict: TrainrunDict,
        malfunction: ExperimentMalfunction,
        minimum_travel_time_dict: Dict[int, int],
        max_episode_steps: int,
        schedule_topo_dict: TopoDict,
        schedule_trainrun_dict: TrainrunDict) -> ScheduleProblemDescription:
    """The perfect oracle only opens up the differences between the schedule
    and the imaginary re-schedule. It gives no additional routing flexibility!

    Parameters
    ----------
    full_reschedule_trainrun_waypoints_dict
    malfunction
    minimum_travel_time_dict
    max_episode_steps
    agents_paths_dict
    schedule_trainrun_dict

    Returns
    -------
    """

    # Delta is all train run way points in the re-schedule that are not also in the schedule
    delta: TrainrunDict = {
        agent_id: sorted(list(
            set(full_reschedule_trainrun_waypoints_dict[agent_id]).difference(
                set(schedule_trainrun_dict[agent_id]))),
            key=lambda p: p.scheduled_at)
        for agent_id in schedule_trainrun_dict.keys()
    }

    # freeze contains everything that stays the same (waypoint and time)
    force_freeze: TrainrunDict = \
        {agent_id: sorted(list(
            set(full_reschedule_trainrun_waypoints_dict[agent_id]).intersection(
                set(schedule_trainrun_dict[agent_id]))),
            key=lambda p: p.scheduled_at) for agent_id in delta.keys()}

    # TODO SIM-105 make option to switch verification on and off? is this the right place for this checks?
    # sanity checks
    for agent_id, delta_waypoints in delta.items():
        for delta_waypoint in delta_waypoints:
            assert delta_waypoint.scheduled_at >= malfunction.time_step, f"found \n\n"
            f"  **** delta_waypoint {delta_waypoint} of agent {agent_id},\n\n"
            f"  **** malfunction is {malfunction}.\n\n"
            f"  **** schedule={schedule_trainrun_dict[agent_id]}.\n\n"
            f"  **** full re-schedule={full_reschedule_trainrun_waypoints_dict[agent_id]}"

    # ban all that are neither in schedule nor re-schedule
    full_reschedule_waypoints: Dict[int, Set[Waypoint]] = \
        {agent_id: {trainrun_waypoint.waypoint
                    for trainrun_waypoint in full_reschedule_trainrun_waypoints_dict[agent_id]}
         for agent_id in delta.keys()}
    schedule_waypoints: Dict[int, Set[Waypoint]] = \
        {agent_id: {trainrun_waypoint.waypoint
                    for trainrun_waypoint in schedule_trainrun_dict[agent_id]}
         for agent_id in delta.keys()}
    all_waypoints: Dict[int, Set[Waypoint]] = \
        {agent_id: {waypoint
                    for waypoint in schedule_topo_dict[agent_id].nodes}
         for agent_id in delta.keys()}
    force_banned: Dict[int, Set[Waypoint]] = {
        agent_id: {
            waypoint
            for waypoint in all_waypoints[agent_id]
            # the trainruns returned by the solver do not include the dummy target node, therefore never ban it.
            if (waypoint not in schedule_waypoints[agent_id]
                and waypoint not in full_reschedule_waypoints[agent_id]
                and waypoint.direction != MAGIC_DIRECTION_FOR_SOURCE_TARGET)
        }
        for agent_id in delta.keys()}

    # build topos without banned
    topo_dict: Dict[int, nx.DiGraph] = {}
    for agent_id, schedule_topo in schedule_topo_dict.items():
        new_topo = nx.DiGraph()
        nodes_to_removes = [
            waypoint
            for waypoint in schedule_topo.nodes
            # the trainruns returned by the solver do not include the dummy target node, therefore never ban it.
            if (waypoint not in schedule_waypoints[agent_id] and
                waypoint not in full_reschedule_waypoints[agent_id] and
                waypoint.direction != MAGIC_DIRECTION_FOR_SOURCE_TARGET)
        ]
        for from_node, to_node in schedule_topo.edges:
            # the trainruns returned by the solver do not include the dummy target node, therefore do not remove corresponding edges
            if from_node not in nodes_to_removes and to_node not in nodes_to_removes:
                new_topo.add_edge(from_node, to_node)
        topo_dict[agent_id] = new_topo

    # build constraints given the topos, the force_freezes
    tc: ScheduleProblemDescription = generic_route_dag_constraints_for_rescheduling(
        schedule_trainruns=schedule_trainrun_dict,
        minimum_travel_time_dict=minimum_travel_time_dict,
        topo_dict=topo_dict,
        force_freeze=force_freeze,
        malfunction=malfunction,
        latest_arrival=max_episode_steps
    )
    freeze_dict: RouteDAGConstraintsDict = tc.route_dag_constraints_dict
    freeze_dict_all: RouteDAGConstraintsDict = {
        agent_id: RouteDAGConstraints(
            freeze_visit=freeze_dict[agent_id].freeze_visit,
            freeze_earliest=freeze_dict[agent_id].freeze_earliest,
            freeze_latest=freeze_dict[agent_id].freeze_latest,
            freeze_banned=freeze_dict[agent_id].freeze_banned + list(force_banned[agent_id]),
        )
        for agent_id in delta.keys()
    }
    return ScheduleProblemDescription(
        route_dag_constraints_dict=freeze_dict_all,
        minimum_travel_time_dict=tc.minimum_travel_time_dict,
        topo_dict=tc.topo_dict,
        max_episode_steps=tc.max_episode_steps,
        route_section_penalties={agent_id: {} for agent_id in delta.keys()}
    )


def _determine_delta(full_reschedule_trainrunwaypoints_dict: TrainrunDict,
                     malfunction: ExperimentMalfunction,
                     schedule_trainrunwaypoints: TrainrunDict,
                     verbose: bool = False) -> Tuple[TrainrunDict, TrainrunDict]:
    """Delta contains the information about what is changed by the malfunction
    with respect to the malfunction.

    - all train run way points in the re-schedule that are different from the initial schedule.
    - this includes the run way point after the malfunction which is delayed!

    Freeze contains all waypoints/times we can freeze/constrain:
    - all train run way points that are the same in the re-schedule

    Parameters
    ----------
    full_reschedule_trainrunwaypoints_dict
    malfunction
    schedule_trainrunwaypoints
    verbose
    """
    if verbose:
        print(f"  **** full re-schedule")
        print(_pp.pformat(full_reschedule_trainrunwaypoints_dict))
    # Delta is all train run way points in the re-schedule that are not also in the schedule
    delta: TrainrunDict = {
        agent_id: sorted(list(
            set(full_reschedule_trainrunwaypoints_dict[agent_id]).difference(
                set(schedule_trainrunwaypoints[agent_id]))),
            key=lambda p: p.scheduled_at)
        for agent_id in schedule_trainrunwaypoints.keys()
    }

    # freeze contains everything that stays the same
    freeze: TrainrunDict = \
        {agent_id: sorted(list(
            set(full_reschedule_trainrunwaypoints_dict[agent_id]).intersection(
                set(schedule_trainrunwaypoints[agent_id]))),
            key=lambda p: p.scheduled_at) for agent_id in delta.keys()}

    if verbose:
        print(f"  **** delta={_pp.pformat(delta)}")

    # TODO SIM-105 make option to switch verification on and off? is this the right place for this checks?
    # sanity checks
    for agent_id, delta_waypoints in delta.items():
        for delta_waypoint in delta_waypoints:
            assert delta_waypoint.scheduled_at >= malfunction.time_step, f"found \n\n"
            f"  **** delta_waypoint {delta_waypoint} of agent {agent_id},\n\n"
            f"  **** malfunction is {malfunction}.\n\n"
            f"  **** schedule={schedule_trainrunwaypoints[agent_id]}.\n\n"
            f"  **** full re-schedule={full_reschedule_trainrunwaypoints_dict[agent_id]}"
    # Freeze are all train run way points in the re-schedule that not in the delta
    if verbose:
        print(f"  **** freeze ={_pp.pformat(freeze)}")

    return delta, freeze
