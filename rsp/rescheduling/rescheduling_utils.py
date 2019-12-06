import warnings
from collections import OrderedDict
from typing import Set, List, Dict, Optional

import numpy
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint, Waypoint, Trainrun, TrainrunDict

from rsp.utils.data_types import ExperimentMalfunction, ExperimentFreezeDict, ExperimentFreeze, \
    experimentFreezePrettyPrint


def generic_experiment_freeze_for_rescheduling(
        schedule_trainruns: TrainrunDict,
        speed_dict: Dict[int, float],
        agents_path_dict: Dict[int, List[List[Waypoint]]],
        force_freeze: Dict[int, List[List[Waypoint]]],
        malfunction: ExperimentMalfunction
) -> ExperimentFreezeDict:
    """
    Derives the experiment freeze given the malfunction and optionally a force freeze from an Oracle.

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
        agent_id: _generic_experiment_freeze_for_rescheduling_agent(
            minimum_travel_time=int(1 / speed_dict[agent_id]),
            agent_paths=agents_path_dict[agent_id],
            force_freeze=force_freeze[agent_id],
            trainrun_waypoint_after_malfunction=_get_delayed_trainrun_waypoint_after_malfunction(
                agent_id=agent_id,
                trainrun=schedule_trainruns[agent_id],
                malfunction=malfunction
            )

        )
        for agent_id, schedule_trainrun in schedule_trainruns.items()}
    for agent_id in experiment_freeze_dict:
        verify_experiment_freeze_for_agent(
            agent_paths=agents_path_dict[agent_id],
            experiment_freeze=experiment_freeze_dict[agent_id],
            force_freeze=force_freeze[agent_id],
            malfunction=malfunction if malfunction.agent_id == agent_id else None
        )

    return experiment_freeze_dict


def _generic_experiment_freeze_for_rescheduling_agent(
        minimum_travel_time: int,
        agent_paths: List[List[Waypoint]],
        force_freeze: List[TrainrunWaypoint],
        trainrun_waypoint_after_malfunction: TrainrunWaypoint

) -> ExperimentFreeze:
    """

    Parameters
    ----------
    minimum_travel_time
    agent_paths
    force_freeze
    trainrun_waypoint_after_malfunction

    Returns
    -------

    """

    # force freeze in Delta re-scheduling
    force_freeze_dict = {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
                         for trainrun_waypoint in force_freeze}
    force_freeze_waypoints_set = {trainrun_waypoint.waypoint for trainrun_waypoint in force_freeze}

    # span a sub-dag for the problem
    # - for full scheduling (TODO SIM-173), this is source vertex and time 0
    # - for full re-scheduling, this is the next waypoint after the malfunction (delayed for the agent in malfunction)
    # - for delta re-scheduling, if the Oracle tells that more can be freezed than up to malfunction, we use this!
    #   it may be possible that the force freeze is not contiguous: we use the end of the first contiguous freezed chunk
    #   this might be dangerous since the sub-DAG may in this case contain vertices that are actually excluded by the freeze from the Oracle
    #   TODO SIM-146 discuss with Erik the next steps.
    sorted(force_freeze, key=lambda trainrun_waypoint: trainrun_waypoint.scheduled_at)

    last_forced_from_start: Optional[TrainrunWaypoint] = _search_last_contiguously_freezed_from_start(
        agent_paths,
        force_freeze,
        force_freeze_waypoints_set
    )

    freeze_earliest_and_visit = []
    freeze_earliest_and_visit_waypoint_set: Set[Waypoint] = set()

    if trainrun_waypoint_after_malfunction.waypoint not in force_freeze_waypoints_set:
        freeze_earliest_and_visit.append(trainrun_waypoint_after_malfunction)
        freeze_earliest_and_visit_waypoint_set.add(trainrun_waypoint_after_malfunction.waypoint)

    earliest_dict: [Waypoint, int] = OrderedDict()

    # decide where the funnel into the route dag should start:
    # use last_forced_from_start or trainrun_waypoint_after_malfunction, whichever is stronger
    # TODO SIM-173: generalize for scheduling, this would be the start node with earliest = 0.
    subdag_trainrun_waypoint_entry_candidates = []
    subdag_trainrun_waypoint_entry_candidates.append(trainrun_waypoint_after_malfunction)
    if last_forced_from_start:
        subdag_trainrun_waypoint_entry_candidates.append(last_forced_from_start)
    subdag_trainrun_waypoint_entry = subdag_trainrun_waypoint_entry_candidates[
        numpy.argmax([candidate.scheduled_at for candidate in subdag_trainrun_waypoint_entry_candidates])]

    # collect earliest in the sub-DAG
    earliest_dict[subdag_trainrun_waypoint_entry.waypoint] = subdag_trainrun_waypoint_entry.scheduled_at
    for agent_path in agent_paths:
        # if not all freezed are on this path, this path cannot be chosen!
        agent_path_set = set(agent_path)

        if not force_freeze_waypoints_set.issubset(
                agent_path_set) and subdag_trainrun_waypoint_entry.waypoint not in agent_path_set:
            continue

        _add_agent_path_for_get_freeze_for_delta(agent_path,
                                                 earliest_dict,
                                                 force_freeze_dict,
                                                 minimum_travel_time,
                                                 subdag_trainrun_waypoint_entry.scheduled_at
                                                 )

    freeze_earliest_only = [TrainrunWaypoint(waypoint=waypoint, scheduled_at=int(scheduled_at))
                            for waypoint, scheduled_at in earliest_dict.items()
                            if (waypoint not in force_freeze_waypoints_set and  # NOQA
                                waypoint not in freeze_earliest_and_visit_waypoint_set)]

    # banned are all waypoint not in the sub-DAG
    banned = []
    banned_set = set()
    for agent_path in agent_paths:
        for waypoint_index, waypoint in enumerate(agent_path):
            if (waypoint not in force_freeze_waypoints_set and  # NOQA
                    waypoint not in freeze_earliest_and_visit_waypoint_set and  # NOQA
                    waypoint not in earliest_dict and  # NOQA
                    waypoint not in banned_set):
                banned.append(waypoint)
                banned_set.add(waypoint)

    return ExperimentFreeze(freeze_time_and_visit=force_freeze.copy(),
                            freeze_earliest_and_visit=freeze_earliest_and_visit,
                            freeze_earliest_only=freeze_earliest_only,
                            freeze_banned=banned,
                            )


def _search_last_contiguously_freezed_from_start(
        agent_paths: List[List[Waypoint]],
        force_freeze: List[TrainrunWaypoint],
        force_freeze_waypoints_set: Set[TrainrunWaypoint],
) -> Optional[TrainrunWaypoint]:
    """
    Searches the last freezed waypoint such that all waypoints leading there are all freezed.

    Parameters
    ----------
    agent_paths
    force_freeze
    force_freeze_waypoints_set

    Returns
    -------

    """
    last_forced_from_start: Optional[TrainrunWaypoint] = None
    for agent_path in agent_paths:
        # if not all freezed are on this path, this path cannot be chosen!
        agent_path_set = set(agent_path)

        if not force_freeze_waypoints_set.issubset(agent_path_set):
            continue
        for index, force_freeze_waypoint in enumerate(force_freeze):
            if force_freeze_waypoint.waypoint != agent_path[index]:
                break  # go to next agent_path
            last_forced_from_start = force_freeze_waypoint
    return last_forced_from_start


def _add_agent_path_for_get_freeze_for_delta(agent_path,
                                             earliest_dict,
                                             force_freeze_dict,
                                             minimum_travel_time,
                                             fully_freezed_until):
    """
    Travese the sub-DAG and find earliest.

    Parameters
    ----------
    agent_path
    earliest_dict
    force_freeze_dict
    minimum_travel_time
    fully_freezed_until

    Returns
    -------

    """

    for waypoint_index, waypoint in enumerate(agent_path[1:]):
        if waypoint in force_freeze_dict:
            earliest_dict[waypoint] = force_freeze_dict[waypoint]
        else:
            # waypoint_index points to the previous!!!!
            candidate = min(earliest_dict.get(agent_path[waypoint_index], numpy.inf) + minimum_travel_time,
                            earliest_dict.get(waypoint, numpy.inf))
            if candidate < numpy.inf and candidate >= fully_freezed_until:
                earliest_dict[waypoint] = candidate


def _get_delayed_trainrun_waypoint_after_malfunction(
        agent_id: int,
        trainrun: Trainrun,
        malfunction: ExperimentMalfunction):
    for trainrun_waypoint in trainrun:
        if trainrun_waypoint.scheduled_at > malfunction.time_step:
            if agent_id == malfunction.agent_id:
                return TrainrunWaypoint(
                    waypoint=trainrun_waypoint.waypoint,
                    scheduled_at=trainrun_waypoint.scheduled_at + malfunction.malfunction_duration)
            else:
                return trainrun_waypoint


def get_freeze_for_full_rescheduling(malfunction: ExperimentMalfunction,
                                     schedule_trainruns: TrainrunDict,
                                     speed_dict: Dict[int, float],
                                     agents_path_dict: Dict[int, List[List[Waypoint]]]
                                     ) -> ExperimentFreezeDict:
    """
    Returns the experiment freeze for the full re-scheduling problem. Wraps the generic freeze

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
        agent_id: _generic_experiment_freeze_for_rescheduling_agent(
            minimum_travel_time=(1 / speed_dict[agent_id]),
            agent_paths=agents_path_dict[agent_id],
            force_freeze=[trainrun_waypoint
                          for trainrun_waypoint in schedule_trainrun
                          if trainrun_waypoint.scheduled_at <= malfunction.time_step],
            # N.B. if the agent has not yet entered the grid, we fix the scheduled
            # this would not work if we had multiple start points for an agent!
            trainrun_waypoint_after_malfunction=_get_delayed_trainrun_waypoint_after_malfunction(
                agent_id=agent_id,
                trainrun=schedule_trainrun,
                malfunction=malfunction
            )

        )
        for agent_id, schedule_trainrun in schedule_trainruns.items()
        if malfunction.time_step >= schedule_trainrun[0].scheduled_at
    }

    # handle the special case of malfunction before scheduled start of agent
    for agent_id, schedule_trainrun in schedule_trainruns.items():
        if agent_id not in experiment_freeze_dict:
            if malfunction.agent_id != agent_id:
                experiment_freeze_dict[agent_id] = ExperimentFreeze(
                    freeze_time_and_visit=[],
                    freeze_earliest_and_visit=[],
                    freeze_earliest_only=_get_earliest_entries_for_full_route_dag(
                        minimum_travel_time=int(1 / speed_dict[agent_id]),
                        agent_paths=agents_path_dict[agent_id],
                        earliest=schedule_trainrun[0].scheduled_at
                    ),
                    freeze_banned=[],
                )
            else:
                warnings.warn(f"agent {agent_id} has malfunction {malfunction} "
                              f"before scheduled start {schedule_trainrun}. "
                              f"Imposing to start at {malfunction.time_step + malfunction.malfunction_duration}")

                experiment_freeze_dict[agent_id] = ExperimentFreeze(
                    freeze_time_and_visit=[],
                    freeze_earliest_and_visit=[],
                    freeze_earliest_only=_get_earliest_entries_for_full_route_dag(
                        minimum_travel_time=int(1 / speed_dict[agent_id]),
                        agent_paths=agents_path_dict[agent_id],
                        earliest=malfunction.time_step + malfunction.malfunction_duration
                    ),
                    freeze_banned=[],
                )

    # TODO analyse SIM-175
    if False:
        print("experimentFreezePrettyPrint(experiment_freeze_dict[17])")
        experimentFreezePrettyPrint(experiment_freeze_dict[17])
    for agent_id in experiment_freeze_dict:
        verify_experiment_freeze_for_agent(
            agent_paths=agents_path_dict[agent_id],
            experiment_freeze=experiment_freeze_dict[agent_id],
            force_freeze=[],
            malfunction=malfunction if malfunction.agent_id == agent_id else None
        )

    return experiment_freeze_dict


# TODO SIM-173 use this for scheduling together with generic entry point!
def _get_earliest_entries_for_full_route_dag(
        minimum_travel_time: int,
        agent_paths: List[List[Waypoint]],
        earliest
) -> List[TrainrunWaypoint]:
    """
    Given the minimum travel time per cell (constant per agent), derive
    the earliest times in the route DAG spanned by the k shortest paths.

    If a vertex can be reached by multiple paths, take the earliest time it can be reached according to
      earliest(train,vertex) = minimum-number-of-hops-from-source(train,vertex) * minimum_running_time + 1

    Parameters
    ----------
    minimum_travel_time
    agent_paths
    earliest

    Returns
    -------

    """
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


def verify_experiment_freeze_for_agent(
        experiment_freeze: ExperimentFreeze,
        agent_paths: List[List[Waypoint]],
        force_freeze: Optional[List[TrainrunWaypoint]] = None,
        malfunction: Optional[ExperimentMalfunction] = None
):
    all_waypoints = {waypoint for agent_path in agent_paths for waypoint in agent_path}
    verified = set()
    for trainrun_waypoint in experiment_freeze.freeze_time_and_visit:
        waypoint = trainrun_waypoint.waypoint
        assert waypoint not in verified, f"duplicate waypoint {trainrun_waypoint}"
        verified.add(waypoint)
    for trainrun_waypoint in experiment_freeze.freeze_earliest_and_visit:
        waypoint = trainrun_waypoint.waypoint
        assert waypoint not in verified, f"duplicate waypoint {trainrun_waypoint}"
        verified.add(waypoint)
    for trainrun_waypoint in experiment_freeze.freeze_earliest_only:
        waypoint = trainrun_waypoint.waypoint
        assert waypoint not in verified, f"duplicate waypoint {trainrun_waypoint}"
        verified.add(waypoint)
    for waypoint in experiment_freeze.freeze_banned:
        assert waypoint not in verified, f"duplicate waypoint {waypoint}"
        verified.add(waypoint)
    assert all_waypoints == verified, \
        f"not verified: {all_waypoints.difference(verified)}, " + \
        f"not in route alternatives: {verified.difference(all_waypoints)}"

    # verify that force freeze is contained in the ExperimentFreeze
    if force_freeze:
        assert set(force_freeze).issubset(set(experiment_freeze.freeze_time_and_visit))

    # verify that all points up to malfunction are forced to be visited
    if malfunction:
        for trainrun_waypoint in experiment_freeze.freeze_earliest_only:
            assert trainrun_waypoint.scheduled_at >= malfunction.time_step + malfunction.malfunction_duration
