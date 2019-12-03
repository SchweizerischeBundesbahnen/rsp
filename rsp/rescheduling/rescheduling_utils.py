from typing import Set, List, Dict, Optional

import numpy
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint, Waypoint, Trainrun

from rsp.utils.data_types import ExperimentMalfunction, ExperimentFreezeDict, ExperimentFreeze


# TODO SIM-146 docstring
def get_freeze_for_malfunction(malfunction, schedule_trainruns,
                               static_rail_env: RailEnv,
                               agents_path_dict: Dict[int, List[List[Waypoint]]],
                               force_freeze: Optional[List[Waypoint]] = None
                               ) -> ExperimentFreezeDict:
    """

    Parameters
    ----------
    malfunction
    schedule_trainruns
    static_rail_env
    agents_path_dict
    force_freeze

    Returns
    -------

    """
    return {agent_id: _get_freeze_for_trainrun(env=static_rail_env,
                                               agent_id=agent_id,
                                               agent_solution_trainrun=schedule_trainrun,
                                               malfunction=malfunction,
                                               agent_paths=agents_path_dict[agent_id],
                                               force_freeze=force_freeze[agent_id]
                                               if force_freeze is not None else None
                                               )
            for agent_id, schedule_trainrun in schedule_trainruns.items()}


# TODO SIM-146 update docstring
def _get_freeze_for_trainrun(
        env: RailEnv,
        agent_id: int,
        agent_solution_trainrun: Trainrun,
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
    env
    agent_id
    agent_solution_trainrun
    malfunction

    Returns
    -------
    ExperimentFreeze

    """
    freeze_time_and_visit: List[TrainrunWaypoint] = force_freeze.copy() if force_freeze is not None else []
    freeze_time_and_visit_waypoints_set: Set[Waypoint] = {trainrun_waypoint.waypoint for trainrun_waypoint in
                                                          freeze_time_and_visit}

    freeze_earliest_and_visit = []
    freeze_earliest_and_visit_waypoints_set: Set[Waypoint] = set()

    earliest: Dict[Waypoint, int] = {}
    minimum_travel_time = int(1 / env.agents[agent_id].speed_data['speed'])

    # collect earliest times along schedule path
    previous_waypoint_before_or_at_malfunction = True

    if agent_solution_trainrun[0].scheduled_at > malfunction.time_step:
        for waypoint_index, trainrun_waypoint in enumerate(agent_solution_trainrun):
            if waypoint_index == 0:
                earliest[trainrun_waypoint.waypoint] = malfunction.time_step
            else:
                earliest[trainrun_waypoint.waypoint] = malfunction.time_step + waypoint_index * minimum_travel_time + 1

    else:
        for waypoint_index, trainrun_waypoint in enumerate(agent_solution_trainrun):

            if trainrun_waypoint.scheduled_at <= malfunction.time_step:
                if trainrun_waypoint.waypoint not in freeze_time_and_visit_waypoints_set:
                    freeze_time_and_visit.append(trainrun_waypoint)
                    freeze_time_and_visit_waypoints_set.add(trainrun_waypoint.waypoint)
                else:
                    assert trainrun_waypoint in freeze_time_and_visit
                previous_waypoint_before_or_at_malfunction = True
            else:
                # we're at the first vertex after the freeeze;
                # the train has already entered the edge leading to this vertex (speed fraction >= 0);
                # therefore, freeze this vertex as well since the train cannot "beam" to another edge
                if malfunction.agent_id == agent_id:
                    earliest_time = trainrun_waypoint.scheduled_at + malfunction.malfunction_duration
                    trainrun_waypoint = TrainrunWaypoint(scheduled_at=earliest_time,
                                                         waypoint=trainrun_waypoint.waypoint)

                if previous_waypoint_before_or_at_malfunction and \
                        trainrun_waypoint.waypoint not in freeze_time_and_visit_waypoints_set and \
                        trainrun_waypoint.waypoint not in freeze_earliest_and_visit_waypoints_set:
                    # we're the first after the malfunction (but not when we have forced freeze) -> fix visit and earliest, thus allowing for delay
                    freeze_earliest_and_visit.append(trainrun_waypoint)
                    freeze_earliest_and_visit_waypoints_set.add(trainrun_waypoint.waypoint)
                previous_waypoint_before_or_at_malfunction = False
            earliest[trainrun_waypoint.waypoint] = trainrun_waypoint.scheduled_at

    # only on the scheduled path, first after malfunction if already in grid
    if force_freeze is None:
        assert len(freeze_earliest_and_visit) <= 1

    banned, freeze_earliest_only = _derive_earliest_for_all_routing_alternatives(
        agent_paths, earliest,
        freeze_earliest_and_visit_waypoints_set,
        freeze_time_and_visit_waypoints_set,
        minimum_travel_time)

    return ExperimentFreeze(freeze_time_and_visit=freeze_time_and_visit,
                            freeze_earliest_and_visit=freeze_earliest_and_visit,
                            freeze_earliest_only=freeze_earliest_only,
                            freeze_visit_only=[],
                            freeze_banned=banned,
                            )


def _derive_earliest_for_all_routing_alternatives(
        agent_paths: List[List[Waypoint]],
        earliest: Dict[Waypoint, int],
        freeze_earliest_and_visit_waypoints_set: Set[Waypoint],
        freeze_time_and_visit_waypoints_set: Set[Waypoint],
        minimum_travel_time: int):
    # derive earliest for all routing alternatives
    banned: List[Waypoint] = []
    for agent_path in agent_paths:
        agent_path_reachable = False
        previous_earliest = numpy.inf
        for waypoint_index, waypoint in enumerate(agent_path):
            if waypoint in earliest:
                agent_path_reachable = True
            if not agent_path_reachable:
                banned.append(waypoint)
            else:
                if previous_earliest is not None:
                    if waypoint_index > 1:
                        earliest_here = previous_earliest + minimum_travel_time
                    else:
                        earliest_here = previous_earliest + minimum_travel_time + 1

                earliest[waypoint] = min(earliest.get(waypoint, numpy.inf), earliest_here)
                assert earliest[waypoint] < numpy.inf
                previous_earliest = earliest[waypoint]
    assert len(freeze_earliest_and_visit_waypoints_set.intersection(freeze_time_and_visit_waypoints_set)) == 0
    # now, we're able to collect earliest only among those which are not full time-frozen and not semi-frozen...
    freeze_earliest_only: List[TrainrunWaypoint] = [
        TrainrunWaypoint(waypoint=waypoint, scheduled_at=earliest_time)
        for waypoint, earliest_time in earliest.items()
        if
        waypoint not in freeze_time_and_visit_waypoints_set and waypoint not in freeze_earliest_and_visit_waypoints_set]
    return banned, freeze_earliest_only


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
