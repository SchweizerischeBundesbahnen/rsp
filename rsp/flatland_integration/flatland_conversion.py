# TODO are these data structures duplicates? see encounter graph
from typing import Dict
from typing import Tuple

from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.utils.data_types import FLATlandPositionsPerTimeStep
from rsp.utils.data_types import TrainSchedule
from rsp.utils.data_types import TrainScheduleDict


def convert_trainrundict_to_entering_positions_for_all_timesteps(trainrun_dict: TrainrunDict) -> TrainScheduleDict:
    """Converts a `TrainrunDict` (only entry times into a new position) into a
    dict with the waypoint for each agent and agent time step.

    Parameters
    ----------
    trainrun_dict: TrainrunDict
        for each agent, a list of time steps with new position

    Returns
    -------
    TrainScheduleDict
        for each agent and time step, the current position (not considering release times)
    """
    train_schedule_dict: TrainScheduleDict = {}
    for agent_id, trainrun in trainrun_dict.items():
        train_schedule: TrainSchedule = {}
        train_schedule_dict[agent_id] = train_schedule
        time_step = 0
        current_position = None
        for trainrun_waypoint in trainrun:
            while time_step < trainrun_waypoint.scheduled_at:
                train_schedule[time_step] = current_position
                time_step += 1
            current_position = trainrun_waypoint.waypoint
            train_schedule[time_step] = current_position
    return train_schedule_dict


def convert_trainrundict_to_positions_after_flatland_timestep(trainrun_dict: TrainrunDict) -> TrainScheduleDict:
    """
    Converts a `TrainrunDict` (only entry times into a new position) into a dict with the waypoint for each agent and time step.
    Parameters
    ----------
    trainrun_dict: TrainrunDict
        for each agent, a list of time steps with new position

    Returns
    -------
    TrainScheduleDict
        for each agent and time step, the current position (not considering release times)

    """
    train_schedule_dict: TrainScheduleDict = {}
    for agent_id, trainrun in trainrun_dict.items():
        train_schedule: TrainSchedule = {}
        train_schedule_dict[agent_id] = train_schedule
        time_step = 0
        current_position = None
        end_time_step = trainrun[-1].scheduled_at
        for next_trainrun_waypoint in trainrun[1:]:
            while time_step + 1 < next_trainrun_waypoint.scheduled_at:
                train_schedule[time_step] = current_position
                time_step += 1
            assert time_step + 1 == next_trainrun_waypoint.scheduled_at
            if time_step + 1 == end_time_step:
                train_schedule[time_step] = None
                break
            current_position = next_trainrun_waypoint.waypoint
            train_schedule[time_step] = current_position
            time_step += 1
    return train_schedule_dict


def extract_trainrun_dict_from_flatland_positions(
        initial_directions: Dict[int, int],
        initial_positions: Dict[int, Tuple[int, int]],
        schedule: FLATlandPositionsPerTimeStep,
        targets: Dict[int, Tuple[int, int]]) -> TrainrunDict:
    """Convert FLATland positions to a TrainrunDict: for each agent, the cell
    entry events.

    Parameters
    ----------
    initial_directions
    initial_positions
    schedule
    targets

    Returns
    -------
    """
    trainrun_dict = {agent_id: [] for agent_id in initial_directions.keys()}
    for agent_id in trainrun_dict:
        curr_pos = None
        curr_dir = None
        for time_step in schedule:
            next_waypoint = schedule[time_step][agent_id]

            # are we running?
            if next_waypoint.position is not None:
                if next_waypoint.position != curr_pos:
                    # are we starting?
                    if curr_pos is None:
                        # sanity checks
                        assert time_step >= 1
                        assert next_waypoint.position == initial_positions[agent_id]
                        assert next_waypoint.direction == initial_directions[agent_id]

                        # when entering the grid in time_step t, the agent has a position only before t+1 -> entry event at t!
                        trainrun_dict[agent_id].append(
                            TrainrunWaypoint(
                                waypoint=Waypoint(
                                    position=next_waypoint.position,
                                    direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET),
                                scheduled_at=time_step - 1))

                    # when the agent has a new position before time_step t, this corresponds to an entry at t!
                    trainrun_dict[agent_id].append(
                        TrainrunWaypoint(
                            waypoint=next_waypoint,
                            scheduled_at=time_step))

            # are we done?
            if next_waypoint.position is None and curr_pos is not None:
                # when the agent enters the target cell, it vanishes immediately in FLATland.
                # TODO in the rsp model, we will add a transition to the dummy time of time 1 + release time 1 -> is there a problem?
                #  (We might lose capacity in the rsp formulation)
                trainrun_dict[agent_id].append(
                    TrainrunWaypoint(
                        waypoint=Waypoint(
                            position=targets[agent_id],
                            direction=curr_dir),
                        scheduled_at=time_step))

                # sanity check: no jumping in the grid, no full check that the we respect the infrastructure layout!
                assert abs(curr_pos[0] - targets[agent_id][0]) + abs(curr_pos[1] - targets[agent_id][1]) == 1, \
                    f"agent {agent_id}: curr_pos={curr_pos} - target={targets[agent_id]}"
            curr_pos = next_waypoint.position
            curr_dir = next_waypoint.direction
    return trainrun_dict
