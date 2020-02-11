import pprint
from typing import Dict
from typing import Set
from typing import Tuple

import networkx as nx
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.rescheduling.rescheduling_utils import generic_experiment_freeze_for_rescheduling
from rsp.route_dag.route_dag import topo_from_agent_paths
from rsp.utils.data_types import AgentsPathsDict
from rsp.utils.data_types import ExperimentFreeze
from rsp.utils.data_types import ExperimentFreezeDict
from rsp.utils.data_types import ExperimentMalfunction

_pp = pprint.PrettyPrinter(indent=4)


# TODO SIM-239 the oracle should take correspond to https://confluence.sbb.ch/display/SIM/Ablauf+RSP+Pipeline+Hypothese+1
def perfect_oracle(
        full_reschedule_trainruns_dict: TrainrunDict,
        malfunction: ExperimentMalfunction,
        minimum_travel_time_dict: Dict[int, int],
        max_episode_steps: int,
        agents_path_dict: AgentsPathsDict,
        schedule_trainruns_dict: TrainrunDict):
    """The perfect oracle only opens up the differences between the schedule
    and the imaginary re-schedule. It gives no additional routing flexibility!

    Parameters
    ----------
    full_reschedule_trainruns_dict
    malfunction
    minimum_travel_time_dict
    max_episode_steps
    agents_path_dict
    schedule_trainruns_dict

    Returns
    -------
    """
    # force_freeze: all that are the same (waypoint and scheduled_at)
    # delta: all trainrun_waypoints in the re-schedule not the same
    delta, force_freeze = _determine_delta(full_reschedule_trainruns_dict,
                                           malfunction,
                                           schedule_trainruns_dict,
                                           verbose=False)

    # ban all that are neither in schedule nor re-schedule
    full_reschedule_waypoints: Dict[int, Set[Waypoint]] = \
        {agent_id: {trainrun_waypoint.waypoint
                    for trainrun_waypoint in full_reschedule_trainruns_dict[agent_id]}
         for agent_id in delta.keys()}
    schedule_waypoints: Dict[int, Set[Waypoint]] = \
        {agent_id: {trainrun_waypoint.waypoint
                    for trainrun_waypoint in schedule_trainruns_dict[agent_id]}
         for agent_id in delta.keys()}
    all_waypoints: Dict[int, Set[Waypoint]] = \
        {agent_id: {waypoint
                    for agent_path in agents_path_dict[agent_id]
                    for waypoint in agent_path}
         for agent_id in delta.keys()}
    # TODO OrderedSet
    force_banned: Dict[int, Set[Waypoint]] = {
        agent_id: {
            waypoint
            for waypoint in all_waypoints[agent_id]
            if (waypoint not in schedule_waypoints[agent_id] and waypoint not in full_reschedule_waypoints[agent_id])
        }
        for agent_id in delta.keys()}

    # build topos without banned
    topo_dict: Dict[int, nx.DiGraph] = {agent_id: topo_from_agent_paths(agents_path_dict[agent_id])
                                        for agent_id in agents_path_dict}
    for agent_id, topo in topo_dict.items():
        nodes_to_removes = [
            waypoint
            for waypoint in topo.nodes
            if (waypoint not in schedule_waypoints[agent_id] and waypoint not in full_reschedule_waypoints[agent_id])
        ]
        topo.remove_nodes_from(nodes_to_removes)

    # build constraints given the topos, the force_freezes
    freeze_dict: ExperimentFreezeDict = generic_experiment_freeze_for_rescheduling(
        schedule_trainruns=schedule_trainruns_dict,
        minimum_travel_time_dict=minimum_travel_time_dict,
        topo_dict=topo_dict,
        force_freeze=force_freeze,
        malfunction=malfunction,
        latest_arrival=max_episode_steps
    )

    freeze_dict_all: ExperimentFreezeDict = {
        agent_id: ExperimentFreeze(
            freeze_visit=freeze_dict[agent_id].freeze_visit,
            freeze_earliest=freeze_dict[agent_id].freeze_earliest,
            freeze_latest=freeze_dict[agent_id].freeze_latest,
            freeze_banned=freeze_dict[agent_id].freeze_banned + list(force_banned[agent_id]),
        )
        for agent_id in delta.keys()
    }
    return freeze_dict_all


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
