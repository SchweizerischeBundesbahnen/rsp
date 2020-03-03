import warnings
from typing import Dict
from typing import Optional

import numpy as np
from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.route_dag.route_dag import RouteDAGConstraintsDict
from rsp.utils.data_types import ExperimentMalfunction


def create_action_plan(train_runs_dict: TrainrunDict, env: RailEnv) -> ControllerFromTrainruns:
    """Creates an action plan (fix schedule, when an action has which positions
    and when it has to take which actions).

    Returns
    -------
    ActionPlanDict
    """

    # tweak solutions
    solution_trainruns_tweaked = {}
    for agent_id, trainrun in train_runs_dict.items():
        solution_trainruns_tweaked[agent_id] = \
            [TrainrunWaypoint(waypoint=trainrun[1].waypoint, scheduled_at=trainrun[0].scheduled_at)] + trainrun[2:]

    return ControllerFromTrainruns(env, solution_trainruns_tweaked)


def get_summ_running_times_trainruns_dict(trainruns_dict: TrainrunDict):
    return sum([
        agent_path[-1].scheduled_at - agent_path[0].scheduled_at
        for agent_id, agent_path in trainruns_dict.items()])


def get_delay_trainruns_dict(trainruns_dict_schedule: TrainrunDict, trainruns_dict_reschedule: TrainrunDict):
    return sum([
        max(trainruns_dict_reschedule[agent_id][-1].scheduled_at - trainruns_dict_schedule[agent_id][-1].scheduled_at,
            0)
        for agent_id in trainruns_dict_reschedule])


def verify_trainruns_dict(env: RailEnv,
                          trainruns_dict: TrainrunDict,
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
    trainruns_dict
    expected_malfunction
    expected_route_dag_constraints

    Returns
    -------
    """
    # 1. ensure train runs are scheduled ascending, the train run is non-circular and respects the train's constant speed.
    _verify_trainruns_1_path_consistency(env, trainruns_dict)

    # 2. verify mutual exclusion
    _verify_trainruns_2_mutual_exclusion(trainruns_dict)

    # 3. check that the paths lead from the desired start and goal
    _verify_trainruns_3_source_target(env, trainruns_dict)

    # 4. check that the transitions are valid FLATland transitions according to the grid
    _verify_trainruns_4_consistency_with_flatland_moves(env, trainruns_dict)

    # 5. verify expected malfunction
    if expected_malfunction:
        _verify_trainruns_5_malfunction(env, expected_malfunction, trainruns_dict)

    # 6. verify freezes are respected
    if expected_route_dag_constraints:
        _verify_trainruns_6_freeze(expected_route_dag_constraints, trainruns_dict)


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


def replay(env: RailEnv,  # noqa: C901
           solver_name: str,
           controller_from_train_runs: ControllerFromTrainruns,
           expected_malfunction: Optional[ExperimentMalfunction] = None,
           rendering: bool = False,
           debug: bool = False,
           loop_index: int = 0,
           stop_on_malfunction: bool = False,
           disable_verification_in_replay: bool = False,
           image_output_directory: Optional[str] = None) -> Optional[ExperimentMalfunction]:
    """Replay the solution an check whether the actions againts FLATland env
    can be performed as against. Verifies that the solution is indeed a
    solution in the FLATland sense.

    Parameters
    ----------
    solver_name: bool
        The name of the solver for debugging purposes.
    disable_verification_in_replay
        Whether it is tested the replay corresponds to the problem's solution
        TODO SIM-105 Should there be option to disable replay completely? Profile experiments to test how much time replay takes in the experiments.
    env
        The env to run the verification with
    rendering_call_back
        Called every step in replay
    debug
        Display debugging information
    loop_index
        Used for display, should identify the problem instance
    expected_malfunction
        If provided and disable_verification_in_replay == False, it is checked that the malfunction happens as expected.
    stop_on_malfunction
        If true, stops and returns upon entering into malfunction; in this case returns the malfunction
    controller_from_train_runs: ActionPlanDict
        The action plan to replay

    Returns
    -------
    Optional[Malfunction]
        The malfunction in `stop_on_malfunction` mode, `None` else.
    """
    total_reward = 0
    time_step = 0
    if rendering:
        from rsp.utils.experiment_render_utils import init_renderer_for_env
        renderer = init_renderer_for_env(env, rendering)
    if expected_malfunction:
        malfunction_agent_minimum_travel_time = int(
            np.ceil(1 / env.agents[expected_malfunction.agent_id].speed_data['speed']))

        trainrun_waypoint_after_malfunction: TrainrunWaypoint = next(
            trainrun_waypoint for trainrun_waypoint in
            controller_from_train_runs.trainrun_dict[expected_malfunction.agent_id]
            if (trainrun_waypoint.scheduled_at >=
                (expected_malfunction.time_step + expected_malfunction.malfunction_duration)))
        malfunction_agent_time_step_to_take_action_before_malfunction = \
            trainrun_waypoint_after_malfunction.scheduled_at - expected_malfunction.malfunction_duration - \
            malfunction_agent_minimum_travel_time
    while not env.dones['__all__'] and time_step <= env._max_episode_steps:
        fail = False
        if disable_verification_in_replay:
            fail = _check_expected_position_and_malfunction(
                controller_from_train_runs, debug, env, expected_malfunction, time_step)
        if fail:
            raise Exception("Unexpected state. See above for !!=unexpected position, MM=unexpected malfuntion")

        actions = controller_from_train_runs.act(time_step)

        # TODO SIM-301 should we do we do this tweaking in FLATland?
        # controller_from_train_runs does not know about malfunctions!
        # with malfunction, instead of travel_time before the first point after the malfunction ends,
        # take that action at that time - malfunction_duration - travel_time!
        if expected_malfunction and time_step == malfunction_agent_time_step_to_take_action_before_malfunction:
            time_step_to_tweak = trainrun_waypoint_after_malfunction.scheduled_at - malfunction_agent_minimum_travel_time
            alt_actions = controller_from_train_runs.act(time_step_to_tweak)
            if debug:
                print(
                    f"agent {expected_malfunction.agent_id} at: "
                    f"action {time_step} {actions[expected_malfunction.agent_id]} -> "
                    f"{alt_actions[expected_malfunction.agent_id]}")
            actions[expected_malfunction.agent_id] = alt_actions[expected_malfunction.agent_id]

        if debug:
            print(f"env._elapsed_steps={env._elapsed_steps}")
            print("actions [{}]->[{}] actions={}".format(time_step, time_step + 1, actions))

        obs, all_rewards, done, _ = env.step(actions)
        total_reward += sum(np.array(list(all_rewards.values())))

        if stop_on_malfunction:
            for agent in env.agents:
                if agent.malfunction_data['malfunction'] > 0:
                    # malfunction duration is already decreased by one in this step(), therefore add +1!
                    return ExperimentMalfunction(time_step, agent.handle, agent.malfunction_data['malfunction'] + 1)

        if rendering:
            from rsp.utils.experiment_render_utils import render_env
            render_env(renderer, test_id=loop_index, solver_name=solver_name, i_step=time_step,
                       image_output_directory=image_output_directory)

        # if all agents have reached their goals, break
        if done['__all__']:
            break
        time_step += 1
    if rendering:
        from rsp.utils.experiment_render_utils import cleanup_renderer_for_env
        try:
            cleanup_renderer_for_env(renderer)
        except AttributeError as e:
            # TODO why does this happen?
            warnings.warn(e)
    if stop_on_malfunction:
        return None
    else:
        return total_reward


def _check_expected_position_and_malfunction(ap: ControllerFromTrainruns,
                                             debug: bool,
                                             env: RailEnv,
                                             malfunction: ExperimentMalfunction,
                                             time_step: int):
    """Checks whether the current position and malfunction are as expected from
    the action plan controller.

    Prints '!!=unexpected position' and 'MM=unexpected malfuntion'

    Parameters
    ----------
    ap: ControllerFromTrainruns
    debug: bool
    env: RailEnv
    malfunction: ExperimentMalfunction
    time_step: int

    Returns
    -------
    bool: is the replay still in harmony with action plan?
    """
    fail = False
    for agent in env.agents:
        prefix = ""
        we: Waypoint = ap.get_waypoint_before_or_at_step(agent.handle, time_step)
        if agent.position != we.position:
            prefix = "!!"
            fail = True
        if agent.malfunction_data['malfunction'] > 0 and (
                malfunction is None or agent.handle != malfunction.agent_id):
            prefix += "MM"
            fail = True
        if debug:
            print(
                f"{prefix}[{time_step}] agent={agent.handle} at position={agent.position} "
                f"in direction={agent.direction} "
                f"with agent.moving={agent.moving} "
                f"(initial_position={agent.initial_position}, initial_direction={agent.initial_direction}, target={agent.target} "
                f"with speed={agent.speed_data} and malfunction={agent.malfunction_data}, expected waypoint={we} "
            )
    return fail
