import pprint
from typing import Dict
from typing import Set
from typing import Tuple

import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.schedule_problem_description.data_types_and_utils import TopoDict
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_reschedule_generic import \
    generic_schedule_problem_description_for_rescheduling
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_utils import verify_consistency_of_route_dag_constraints_for_agent
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_utils import verify_trainrun_satisfies_route_dag_constraints
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
        schedule_trainrun_dict: TrainrunDict,
        max_window_size_from_earliest: int = np.inf) -> ScheduleProblemDescription:
    """The perfect oracle only opens up the differences between the schedule
    and the imaginary re-schedule. It gives no additional routing flexibility!

    Parameters
    ----------
    full_reschedule_trainrun_waypoints_dict: TrainrunDict
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

    # (PO-1) FREEZE WHERE LOCATION AND TIME ARE THE SAME
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

    # (PO-2) BAN ALL THAT ARE NEITHER IN SCHEDULE NOR RE-SCHEDULE
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
            if (waypoint not in schedule_waypoints[agent_id]
                and waypoint not in full_reschedule_waypoints[agent_id])
        }
        for agent_id in delta.keys()}
    # (PO-3.1) build topos without banned
    topo_dict: Dict[int, nx.DiGraph] = {}
    for agent_id, schedule_topo in schedule_topo_dict.items():
        new_topo = nx.DiGraph()
        nodes_to_removes = [
            waypoint
            for waypoint in schedule_topo.nodes
            if (waypoint not in schedule_waypoints[agent_id] and
                waypoint not in full_reschedule_waypoints[agent_id])
        ]
        for from_node, to_node in schedule_topo.edges:
            if from_node not in nodes_to_removes and to_node not in nodes_to_removes:
                new_topo.add_edge(from_node, to_node)
        topo_dict[agent_id] = new_topo

    # (PO-3) BUILD CONSTRAINTS GIVEN THE TOPOS, THE FORCE_FREEZES
    # (PO-3.2) propagate earliest and latest
    schedule_problem_description: ScheduleProblemDescription = generic_schedule_problem_description_for_rescheduling(
        schedule_trainruns=schedule_trainrun_dict,
        minimum_travel_time_dict=minimum_travel_time_dict,
        topo_dict=topo_dict,
        force_freeze=force_freeze,
        malfunction=malfunction,
        latest_arrival=max_episode_steps,
        max_window_size_from_earliest=max_window_size_from_earliest
    )

    # TODO necessary?
    # (PO-3.3) add force_banned again to propagation constraints
    freeze_dict: RouteDAGConstraintsDict = schedule_problem_description.route_dag_constraints_dict
    freeze_dict_all: RouteDAGConstraintsDict = {
        agent_id: RouteDAGConstraints(
            freeze_visit=freeze_dict[agent_id].freeze_visit,
            freeze_earliest=freeze_dict[agent_id].freeze_earliest,
            freeze_latest=freeze_dict[agent_id].freeze_latest,
            freeze_banned=freeze_dict[agent_id].freeze_banned + list(force_banned[agent_id]),
        )
        for agent_id in delta.keys()
    }

    # TODO SIM-324 pull out verification
    for agent_id, _ in freeze_dict.items():
        verify_consistency_of_route_dag_constraints_for_agent(
            agent_id=agent_id,
            route_dag_constraints=freeze_dict[agent_id],
            topo=topo_dict[agent_id],
            # TODO perfect oracle seems not to respect malfunction!!
            # malfunction=malfunction, # noqa
            max_window_size_from_earliest=max_window_size_from_earliest,
        )
        # re-schedule train run must be open in route dag constraints
        verify_trainrun_satisfies_route_dag_constraints(
            agent_id=agent_id,
            route_dag_constraints=freeze_dict[agent_id],
            scheduled_trainrun=full_reschedule_trainrun_waypoints_dict[agent_id]
        )

    return ScheduleProblemDescription(
        route_dag_constraints_dict=freeze_dict_all,
        minimum_travel_time_dict=schedule_problem_description.minimum_travel_time_dict,
        topo_dict=schedule_problem_description.topo_dict,
        max_episode_steps=schedule_problem_description.max_episode_steps,
        route_section_penalties=schedule_problem_description.route_section_penalties,
        weight_lateness_seconds=1
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
