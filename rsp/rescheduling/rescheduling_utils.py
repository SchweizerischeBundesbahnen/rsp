from collections import OrderedDict
from typing import Set, List, Dict, Optional

import numpy
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint, Waypoint, Trainrun, TrainrunDict

from rsp.utils.data_types import ExperimentMalfunction, ExperimentFreezeDict, ExperimentFreeze


def get_freeze_for_delta(schedule_trainruns: TrainrunDict,
                         full_reschedule_trainruns: TrainrunDict,
                         speed_dict: Dict[int, float],
                         agents_path_dict: Dict[int, List[List[Waypoint]]],
                         force_freeze: Dict[int, List[List[Waypoint]]],
                         malfunction: ExperimentMalfunction
                         ) -> ExperimentFreezeDict:
    """

    Parameters
    ----------
    malfunction
    schedule_trainruns
    env
    agents_path_dict
    force_freeze

    Returns
    -------

    """
    experiment_freeze_dict = {
        agent_id: _get_freeze_for_delta(
            minimum_travel_time=(1 / speed_dict[agent_id]),
            schedule_trainrun=schedule_trainruns[agent_id],
            full_reschedule_trainrun=full_reschedule_trainruns[agent_id],
            agent_paths=agents_path_dict[agent_id],
            force_freeze=force_freeze[agent_id],
            malfunction=malfunction if malfunction.agent_id == agent_id else None
        )
        for agent_id, schedule_trainrun in schedule_trainruns.items()}
    for agent_id in experiment_freeze_dict:
        verify_experiment_freeze_for_agent(
            agent_paths=agents_path_dict[agent_id],
            freeze=experiment_freeze_dict[agent_id])

    return experiment_freeze_dict


# this code would not work if we hat multiple possible start nodes!
def _get_freeze_for_delta(
        minimum_travel_time: int,
        schedule_trainrun: Trainrun,
        full_reschedule_trainrun: Trainrun,
        agent_paths: List[List[Waypoint]],
        force_freeze: List[TrainrunWaypoint],
        malfunction: Optional[ExperimentMalfunction]
) -> ExperimentFreeze:
    force_freeze_dict = {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at for trainrun_waypoint in
                         force_freeze}
    force_freeze_waypoints_set = {trainrun_waypoint.waypoint for trainrun_waypoint in force_freeze}

    # find first after malfunction and add to earliest and visit if this is the malfunctioning agent
    freeze_earliest_and_visit = []
    freeze_earliest_and_visit_waypoint_set: Set[Waypoint] = set()
    trainrun_waypoint_after_malfunction = None
    if malfunction:
        for trainrun_waypoint in full_reschedule_trainrun:
            if trainrun_waypoint.scheduled_at > malfunction.time_step:
                if trainrun_waypoint.waypoint not in force_freeze_waypoints_set:
                    freeze_earliest_and_visit.append(trainrun_waypoint)
                    trainrun_waypoint_after_malfunction = trainrun_waypoint
                    freeze_earliest_and_visit_waypoint_set.add(trainrun_waypoint.waypoint)
                break

    earliest_dict: Dict[Waypoint, int] = OrderedDict()

    banned_paths = []
    scheduled_start = schedule_trainrun[0].scheduled_at
    for agent_path in agent_paths:

        # if not all freezed are on this path, this path cannot be chosen!
        agent_path_set = set(agent_path)
        if not force_freeze_waypoints_set.issubset(agent_path_set):
            banned_paths.append(agent_path)
            continue

        previous_earliest = None
        for waypoint_index, waypoint in enumerate(agent_path):
            if waypoint_index == 0:
                if waypoint in force_freeze_dict:
                    previous_earliest = max(force_freeze_dict[waypoint], scheduled_start)
                else:
                    # do not start earlier than scheduled
                    earliest_dict[waypoint] = scheduled_start
                    previous_earliest = scheduled_start
            else:

                if trainrun_waypoint_after_malfunction and waypoint == trainrun_waypoint_after_malfunction.waypoint:
                    # if this is the agent in malfunction and we're at the vertex after malfunction, delay!
                    tentative_earliest = trainrun_waypoint_after_malfunction.scheduled_at
                else:
                    # propagate earliest via path
                    tentative_earliest = max(scheduled_start + waypoint_index * minimum_travel_time + 1,
                                             previous_earliest + minimum_travel_time)
                if waypoint in force_freeze_dict:
                    previous_earliest = max(tentative_earliest, force_freeze_dict[waypoint])
                else:
                    earliest_dict[waypoint] = min(earliest_dict.get(waypoint, numpy.inf), tentative_earliest)
                    previous_earliest = earliest_dict[waypoint]
    freeze_earliest_only = [TrainrunWaypoint(waypoint=waypoint, scheduled_at=int(scheduled_at))
                            for waypoint, scheduled_at in earliest_dict.items()
                            if waypoint not in freeze_earliest_and_visit_waypoint_set]

    banned = []
    banned_set = set()
    for agent_path in banned_paths:
        for waypoint_index, waypoint in enumerate(agent_path):
            if waypoint not in force_freeze_waypoints_set and waypoint not in earliest_dict and waypoint not in banned_set:
                banned.append(waypoint)
                banned_set.add(waypoint)

    return ExperimentFreeze(freeze_time_and_visit=force_freeze.copy(),
                            freeze_earliest_and_visit=freeze_earliest_and_visit,
                            freeze_earliest_only=freeze_earliest_only,
                            freeze_visit_only=[],
                            freeze_banned=banned,
                            )


# TODO SIM-146 docstring
def get_freeze_for_malfunction(malfunction, schedule_trainruns,
                               env: RailEnv,
                               agents_path_dict: Dict[int, List[List[Waypoint]]]
                               ) -> ExperimentFreezeDict:
    """

    Parameters
    ----------
    malfunction
    schedule_trainruns
    env
    agents_path_dict
    force_freeze

    Returns
    -------

    """
    experiment_freeze_dict = {
        agent_id: _get_freeze_for_malfunction_per_train(
            minimum_travel_time=(1 / env.agents[agent_id].speed_data['speed']),
            agent_id=agent_id,
            schedule_trainruns=schedule_trainrun,
            malfunction=malfunction,
            agent_paths=agents_path_dict[agent_id],
        )
        for agent_id, schedule_trainrun in schedule_trainruns.items()}
    for agent_id in experiment_freeze_dict:
        verify_experiment_freeze_for_agent(
            agent_paths=agents_path_dict[agent_id],
            freeze=experiment_freeze_dict[agent_id])

    return experiment_freeze_dict


# TODO same for scheduling: earliest idea

def _get_earliest(
        minimum_travel_time: int,
        agent_paths: List[List[Waypoint]],
        earliest: int = 0
) -> List[TrainrunWaypoint]:
    earliest_dict: Dict[Waypoint, int] = OrderedDict()
    for agent_path in agent_paths:
        for waypoint_index, waypoint in enumerate(agent_path):
            if waypoint_index == 0:
                earliest_dict[waypoint] = earliest
            else:
                earliest_dict[waypoint] = min(earliest_dict.get(waypoint, numpy.inf),
                                              earliest + waypoint_index * minimum_travel_time + 1)
    return [TrainrunWaypoint(waypoint=waypoint, scheduled_at=int(scheduled_at))
            for waypoint, scheduled_at in earliest_dict.items()]


# TODO SIM-146 update docstring
def _get_freeze_for_malfunction_per_train(
        minimum_travel_time: int,
        agent_id: int,
        schedule_trainruns: Trainrun,
        malfunction: ExperimentMalfunction,
        agent_paths: List[List[Waypoint]],
        force_freeze: Optional[List[TrainrunWaypoint]] = None
) -> ExperimentFreeze:
    """
    Returns the logical view of the freeze:
    - all trainrun waypoints up to malfunction time step
    - plus the next waypoint (the trains have entered the edge or are enteringt the time of malfunction).
    - add delay to train in malfunction for the next waypoint after the malfunction.


    Parameters
    ----------
    minimum_travel_time
    agent_paths
    force_freeze
    agent_id
    schedule_trainruns
    malfunction

    Returns
    -------
    ExperimentFreeze

    """

    # handle case if agent has not started yet
    if schedule_trainruns[0].scheduled_at > malfunction.time_step:
        earliest = schedule_trainruns[0].scheduled_at
        if agent_id == malfunction.agent_id:
            # malfunctioning agent must not start before scheduled departure nor end of malfunction -> max
            earliest = max(earliest, malfunction.time_step + malfunction.malfunction_duration)

        freeze_earliest_only = _get_earliest(minimum_travel_time=minimum_travel_time,
                                             agent_paths=agent_paths,
                                             earliest=earliest)
        return ExperimentFreeze(freeze_time_and_visit=[],
                                freeze_earliest_and_visit=[],
                                freeze_earliest_only=freeze_earliest_only,
                                freeze_visit_only=[],
                                freeze_banned=[],
                                )

    # collect
    # - freeze_time_and_visit -> up to malfunction
    # - freeze_earliest_and_visit -> next point after malfunction, we're already on the edge
    # times along scheduled path
    freeze_earliest_and_visit = []
    freeze_earliest_and_visit_waypoints_set: Set[Waypoint] = set()
    freeze_time_and_visit: List[TrainrunWaypoint] = force_freeze.copy() if force_freeze is not None else []
    freeze_time_and_visit_waypoints_set: Set[Waypoint] = {trainrun_waypoint.waypoint for trainrun_waypoint in
                                                          freeze_time_and_visit}
    previous_waypoint_before_or_at_malfunction = True
    for waypoint_index, trainrun_waypoint in enumerate(schedule_trainruns):
        if trainrun_waypoint.scheduled_at <= malfunction.time_step:
            if trainrun_waypoint.waypoint not in freeze_time_and_visit_waypoints_set:
                freeze_time_and_visit.append(trainrun_waypoint)
                freeze_time_and_visit_waypoints_set.add(trainrun_waypoint.waypoint)
            else:
                assert trainrun_waypoint in freeze_time_and_visit
        else:
            if malfunction.agent_id == agent_id:
                # if it's malfunctioning agent, add malfunction duration to everything
                earliest_time = trainrun_waypoint.scheduled_at + malfunction.malfunction_duration
                trainrun_waypoint = TrainrunWaypoint(scheduled_at=int(earliest_time),
                                                     waypoint=trainrun_waypoint.waypoint)

            if previous_waypoint_before_or_at_malfunction:
                # the train has already entered the edge leading to this vertex (speed fraction >= 0);
                # therefore, freeze this vertex as well since the train cannot "beam" to another edge
                freeze_earliest_and_visit.append(trainrun_waypoint)
                freeze_earliest_and_visit_waypoints_set.add(trainrun_waypoint.waypoint)
            previous_waypoint_before_or_at_malfunction = False

    # we should have collected the first vertex after the malfunction here
    assert len(freeze_earliest_and_visit) == 1

    # collect freeze_earliest_only for routing alternatives
    freeze_earliest_only = _derive_earliest_for_all_routing_alternatives(
        agent_paths=agent_paths,
        # take last element of what's fixed and search routing alternatives from there
        entry_point=freeze_earliest_and_visit[-1] if len(freeze_earliest_and_visit) > 0 else None,
        minimum_travel_time=minimum_travel_time)

    freeze_earliest_only_set = {trainrun_waypoint.waypoint for trainrun_waypoint in
                                freeze_earliest_only}
    # collect banned
    banned = list(OrderedDict.fromkeys(
        [waypoint
         for agent_path in agent_paths
         for waypoint in agent_path
         if waypoint not in freeze_time_and_visit_waypoints_set
         and waypoint not in freeze_earliest_and_visit_waypoints_set
         and waypoint not in freeze_earliest_only_set
         ]))

    return ExperimentFreeze(freeze_time_and_visit=freeze_time_and_visit,
                            freeze_earliest_and_visit=freeze_earliest_and_visit,
                            freeze_earliest_only=freeze_earliest_only,
                            freeze_visit_only=[],
                            freeze_banned=banned,
                            )


def _derive_earliest_for_all_routing_alternatives(
        agent_paths: List[List[Waypoint]],
        entry_point: Optional[TrainrunWaypoint],
        minimum_travel_time: int):
    # derive earliest for all routing alternatives
    earliest: Dict[Waypoint, int] = {}
    for agent_path in agent_paths:
        # TODO if malfunction before entering
        agent_path_reached = False
        previous_earliest = numpy.inf
        for waypoint_index, waypoint in enumerate(agent_path):
            # if the first waypoint after the malfunction is in the path, then we can hop on this path in route tree
            if waypoint == entry_point.waypoint:
                agent_path_reached = True
                previous_earliest = entry_point.scheduled_at
            # entry_point is earliest and visit, do not add again!
            if agent_path_reached and waypoint != entry_point.waypoint:
                earliest[waypoint] = min(earliest.get(waypoint, numpy.inf), previous_earliest + minimum_travel_time)
                previous_earliest += minimum_travel_time

    # now, we're able to collect earliest only among those which are not full time-frozen and not semi-frozen...
    freeze_earliest_only: List[TrainrunWaypoint] = [
        TrainrunWaypoint(waypoint=waypoint, scheduled_at=int(earliest_time))
        for waypoint, earliest_time in earliest.items()
    ]
    return freeze_earliest_only


def verify_experiment_freeze_for_agent(freeze: ExperimentFreeze, agent_paths: List[List[Waypoint]]):
    all_waypoints = {waypoint for agent_path in agent_paths for waypoint in agent_path}
    verified = set()
    for trainrun_waypoint in freeze.freeze_time_and_visit:
        waypoint = trainrun_waypoint.waypoint
        assert waypoint not in verified, f"duplicate waypoint {trainrun_waypoint}"
        verified.add(waypoint)
    for trainrun_waypoint in freeze.freeze_earliest_and_visit:
        waypoint = trainrun_waypoint.waypoint
        assert waypoint not in verified, f"duplicate waypoint {trainrun_waypoint}"
        verified.add(waypoint)
    for trainrun_waypoint in freeze.freeze_earliest_only:
        waypoint = trainrun_waypoint.waypoint
        assert waypoint not in verified, f"duplicate waypoint {trainrun_waypoint}"
        verified.add(waypoint)
    for waypoint in freeze.freeze_banned:
        assert waypoint not in verified, f"duplicate waypoint {waypoint}"
        verified.add(waypoint)
    assert all_waypoints == verified, \
        f"not verified: {all_waypoints.difference(verified)}, " + \
        f"not in route alternatives: {verified.difference(all_waypoints)}"
