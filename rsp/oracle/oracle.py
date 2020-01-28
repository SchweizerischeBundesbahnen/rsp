import pprint
from typing import Tuple

from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.utils.data_types import ExperimentMalfunction

_pp = pprint.PrettyPrinter(indent=4)


# TODO SIM-239 the oracle should take correspond to https://confluence.sbb.ch/display/SIM/Ablauf+RSP+Pipeline+Hypothese+1
def determine_delta(full_reschedule_trainrunwaypoints_dict: TrainrunDict,
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
