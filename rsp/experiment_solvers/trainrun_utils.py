from typing import Dict
from typing import Optional

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.route_dag.route_dag import RouteDAGConstraintsDict
from rsp.utils.data_types import ExperimentMalfunction


def get_sum_running_times_trainruns_dict(trainruns_dict: TrainrunDict):
    return sum([
        agent_path[-1].scheduled_at - agent_path[0].scheduled_at
        for agent_id, agent_path in trainruns_dict.items()])


def get_delay_trainruns_dict(trainruns_dict_schedule: TrainrunDict, trainruns_dict_reschedule: TrainrunDict):
    return sum([
        max(trainruns_dict_reschedule[agent_id][-1].scheduled_at - trainruns_dict_schedule[agent_id][-1].scheduled_at,
            0)
        for agent_id in trainruns_dict_reschedule])


def verify_trainruns_dict(env: RailEnv,
                          trainrun_dict: TrainrunDict,
                          expected_malfunction: Optional[ExperimentMalfunction] = None,
                          expected_route_dag_constraints: Optional[RouteDAGConstraintsDict] = None
                          ):
    """Verify the consistency of a train run.

    1. ensure train runs are scheduled ascending, the train run is non-circular and respects the train's constant speed.
    2. verify mutual exclusion
    3. check that the paths lead from the desired start and goal
    4. check that the transitions are valid FLATland transitions according to the grid
    5. verify expected malfunction (if given)
    6. verfy freezes are respected (if given)


    Parameters
    ----------
    env
    trainrun_dict
    expected_malfunction
    expected_route_dag_constraints

    Returns
    -------
    """
    # 1. ensure train runs are scheduled ascending, the train run is non-circular and respects the train's constant speed.
    _verify_trainruns_1_path_consistency(env, trainrun_dict)

    # 2. verify mutual exclusion
    _verify_trainruns_2_mutual_exclusion(trainrun_dict)

    # 3. check that the paths lead from the desired start and goal
    _verify_trainruns_3_source_target(env, trainrun_dict)

    # 4. check that the transitions are valid FLATland transitions according to the grid
    _verify_trainruns_4_consistency_with_flatland_moves(env, trainrun_dict)

    # 5. verify expected malfunction
    if expected_malfunction:
        _verify_trainruns_5_malfunction(env, expected_malfunction, trainrun_dict)

    # 6. verify freezes are respected
    if expected_route_dag_constraints:
        _verify_trainruns_6_freeze(expected_route_dag_constraints, trainrun_dict)


def _verify_trainruns_5_malfunction(env, expected_malfunction, trainruns_dict):
    malfunction_agent_path = trainruns_dict[expected_malfunction.agent_id]
    # malfunction must not start before the agent is in the grid
    assert malfunction_agent_path[0].scheduled_at + 1 <= expected_malfunction.time_step
    previous_time = malfunction_agent_path[0].scheduled_at + 1
    agent_minimum_running_time = int(1 / env.agents[expected_malfunction.agent_id].speed_data['speed'])
    for waypoint_index, trainrun_waypoint in enumerate(malfunction_agent_path):
        if trainrun_waypoint.scheduled_at > expected_malfunction.time_step:
            lower_bound_for_scheduled_at = previous_time + agent_minimum_running_time + expected_malfunction.malfunction_duration
            assert trainrun_waypoint.scheduled_at >= lower_bound_for_scheduled_at, \
                f"malfunction={expected_malfunction}, " + \
                f"but found {malfunction_agent_path[max(0, waypoint_index - 1)]}{trainrun_waypoint}"
            break


def _verify_trainruns_6_freeze(expected_route_dag_constraints, trainruns_dict):
    for agent_id, route_dag_constraints in expected_route_dag_constraints.items():
        waypoint_dict: Dict[Waypoint, int] = {
            trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
            for trainrun_waypoint in trainruns_dict[agent_id]
        }

        # is freeze_visit respected?
        for waypoint in route_dag_constraints.freeze_visit:
            assert waypoint in waypoint_dict

        # is freeze_earliest respected?
        for waypoint, scheduled_at in route_dag_constraints.freeze_earliest.items():
            if waypoint in waypoint_dict:
                actual_scheduled_at = waypoint_dict[waypoint]
                assert actual_scheduled_at >= scheduled_at, \
                    f"expected {actual_scheduled_at} <= {scheduled_at} " + \
                    f"for {waypoint} of agent {agent_id}"

        # is freeze_latest respected?
        for waypoint, scheduled_at in route_dag_constraints.freeze_latest.items():
            if waypoint in waypoint_dict:
                actual_scheduled_at = waypoint_dict[waypoint]
                assert actual_scheduled_at <= scheduled_at, \
                    f"expected {actual_scheduled_at} <= {scheduled_at} " + \
                    f"for {waypoint} of agent {agent_id}"

        # is freeze_banned respected?
        for waypoint in route_dag_constraints.freeze_banned:
            assert waypoint not in waypoint_dict


def _verify_trainruns_4_consistency_with_flatland_moves(env, trainruns_dict):
    for agent_id, trainrun_sparse in trainruns_dict.items():
        previous_trainrun_waypoint: Optional[TrainrunWaypoint] = None
        for trainrun_waypoint in trainrun_sparse:
            if previous_trainrun_waypoint is not None:
                valid_next_waypoints = {
                    Waypoint(position=rail_env_next_action.next_position, direction=rail_env_next_action.next_direction)
                    for rail_env_next_action in
                    get_valid_move_actions_(agent_direction=previous_trainrun_waypoint.waypoint.direction,
                                            agent_position=previous_trainrun_waypoint.waypoint.position,
                                            rail=env.rail)}
                assert trainrun_waypoint.waypoint in valid_next_waypoints, \
                    f"invalid move for agent {agent_id}: {previous_trainrun_waypoint} -> {trainrun_waypoint}, expected one of {valid_next_waypoints}"


def _verify_trainruns_3_source_target(env, trainruns_dict):
    for agent_id, agent in enumerate(env.agents):
        assert agent_id == agent.handle
        assert agent.handle in trainruns_dict
        # initial trainrun_waypoint is first after dummy
        initial_trainrun_waypoint = trainruns_dict[agent_id][1]
        assert initial_trainrun_waypoint.waypoint.position == agent.initial_position, \
            f"agent {agent} does not start in expected initial position, found {initial_trainrun_waypoint}"
        assert initial_trainrun_waypoint.waypoint.direction == agent.initial_direction, \
            f"agent {agent} does not start in expected initial direction, found {initial_trainrun_waypoint}"
        # target trainrun waypoint is last before dummy
        final_trainrun_waypoint = trainruns_dict[agent_id][-1]
        assert final_trainrun_waypoint.waypoint.position == agent.target, \
            f"agent {agent} does not end in expected target position, found {final_trainrun_waypoint}"


def _verify_trainruns_2_mutual_exclusion(trainruns_dict):
    agent_positions_per_time_step = {}
    for agent_id, trainrun_sparse in trainruns_dict.items():

        previous_trainrun_waypoint: Optional[TrainrunWaypoint] = None
        for trainrun_waypoint in trainrun_sparse:
            if previous_trainrun_waypoint is not None:
                while trainrun_waypoint.scheduled_at > previous_trainrun_waypoint.scheduled_at + 1:
                    time_step = previous_trainrun_waypoint.scheduled_at + 1
                    agent_positions_per_time_step.setdefault(time_step, {})
                    previous_trainrun_waypoint = TrainrunWaypoint(waypoint=previous_trainrun_waypoint.waypoint,
                                                                  scheduled_at=time_step)

                    agent_positions_per_time_step[time_step][agent_id] = previous_trainrun_waypoint
            agent_positions_per_time_step.setdefault(trainrun_waypoint.scheduled_at, {})
            agent_positions_per_time_step[trainrun_waypoint.scheduled_at][agent_id] = trainrun_waypoint
            previous_trainrun_waypoint = trainrun_waypoint

        time_step = previous_trainrun_waypoint.scheduled_at + 1
        agent_positions_per_time_step.setdefault(time_step, {})
        trainrun_waypoint = TrainrunWaypoint(
            waypoint=Waypoint(position=previous_trainrun_waypoint.waypoint.position,
                              direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET),
            scheduled_at=time_step)
        agent_positions_per_time_step[time_step][agent_id] = trainrun_waypoint
    for time_step in agent_positions_per_time_step:
        positions = {trainrun_waypoint.waypoint.position for trainrun_waypoint in
                     agent_positions_per_time_step[time_step].values()}
        assert len(positions) == len(agent_positions_per_time_step[time_step]), \
            f"at {time_step}, conflicting positions: {agent_positions_per_time_step[time_step]}"


def _verify_trainruns_1_path_consistency(env, trainruns_dict):
    for agent_id, trainrun_sparse in trainruns_dict.items():
        minimum_running_time_per_cell = int(1 // env.agents[agent_id].speed_data['speed'])
        assert minimum_running_time_per_cell >= 1

        previous_trainrun_waypoint: Optional[TrainrunWaypoint] = None
        previous_waypoints = set()
        for trainrun_waypoint in trainrun_sparse:
            # 1.a) ensure schedule is ascending and respects the train's constant speed
            if previous_trainrun_waypoint is not None:
                # TODO SIM-322 hard-coded assumption
                if previous_trainrun_waypoint.waypoint.direction == MAGIC_DIRECTION_FOR_SOURCE_TARGET or \
                        trainrun_waypoint.waypoint.direction == MAGIC_DIRECTION_FOR_SOURCE_TARGET:
                    assert trainrun_waypoint.scheduled_at - previous_trainrun_waypoint.scheduled_at == 1
                else:
                    assert trainrun_waypoint.scheduled_at >= previous_trainrun_waypoint.scheduled_at + minimum_running_time_per_cell, \
                        f"agent {agent_id} inconsistency: to {trainrun_waypoint} " + \
                        f"from {previous_trainrun_waypoint} " + \
                        f"minimum running time={minimum_running_time_per_cell}"
            # 1.b) ensure train run is non-circular
            assert trainrun_waypoint not in previous_waypoints
            previous_trainrun_waypoint = trainrun_waypoint
            previous_waypoints.add(trainrun_waypoint.waypoint)
