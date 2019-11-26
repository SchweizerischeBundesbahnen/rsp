from typing import Set, List

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint

from rsp.utils.data_types import Malfunction


def get_freeze_for_malfunction(malfunction, schedule_trainruns, static_rail_env):
    return {agent_id: _get_freeze_for_trainrun(static_rail_env, agent_id, schedule_trainrun, malfunction) for
            agent_id, schedule_trainrun in schedule_trainruns.items()}


def _get_freeze_for_trainrun(
        env: RailEnv,
        agent_id: int,
        agent_solution_trainrun: Set[TrainrunWaypoint],
        malfunction: Malfunction,
        verbose: bool = False) -> List[TrainrunWaypoint]:
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

    """
    frozen = []
    scheduled_at_previous = 0
    for waypoint_index, trainrun_waypoint in enumerate(agent_solution_trainrun):
        if trainrun_waypoint.scheduled_at <= malfunction.time_step:
            frozen.append(trainrun_waypoint)
        else:
            # we're at the first vertex after the freeeze;
            # the train has already entered the edge leading to this vertex (speed fraction >= 0);
            # therefore, freeze this vertex as well since the train cannot "beam" to another edge
            if waypoint_index > 0:
                if malfunction.agent_id == agent_id:
                    # TODO is this safe because of rounding errors?
                    minimum_travel_time = int(1 / env.agents[agent_id].speed_data['speed'])
                    earliest_time = scheduled_at_previous + minimum_travel_time + malfunction.malfunction_duration
                    frozen.append(TrainrunWaypoint(scheduled_at=earliest_time, waypoint=trainrun_waypoint.waypoint))
                else:
                    frozen.append(trainrun_waypoint)
            # do not consider remainder of this train!
            break
        scheduled_at_previous = trainrun_waypoint.scheduled_at
    return frozen
