"""Solve a problem a."""
import pprint
from typing import Optional, NamedTuple, Set, Callable, Dict

import numpy as np
from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_
from flatland.envs.rail_trainrun_data_structures import Waypoint, TrainrunDict, TrainrunWaypoint

from rsp.abstract_problem_description import AbstractProblemDescription
from rsp.abstract_solution_description import AbstractSolutionDescription
from rsp.asp.asp_problem_description import ASPProblemDescription
from rsp.asp.asp_solution_description import ASPSolutionDescription
from rsp.rescheduling.rescheduling_utils import ExperimentFreezeDict
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.general_utils import current_milli_time, verification_by_file

SchedulingExperimentResult = NamedTuple('SchedulingExperimentResult',
                                        [('total_reward', int),
                                         ('solve_time', float),
                                         ('optimization_costs', float),
                                         ('build_problem_time', float),
                                         ('solution', AbstractSolutionDescription),
                                         ('experiment_freeze', Optional[ExperimentFreezeDict])
                                         ])

# test_id: int, solver_name: str, i_step: int
SolveProblemRenderCallback = Callable[[int, str, int], None]

_pp = pprint.PrettyPrinter(indent=4)


# --------------------------------------------------------------------------------------
# Solve an `AbstractProblemDescription`
# --------------------------------------------------------------------------------------
# TODO SIM-220 discuss with Adrian: get rid of "old world" (solve_utils/solve_tests/solve_envs)?
#  Then, we could get rid of this intermediate layer and move solve_problem to AbstractProblemDescription
def solve_problem(env: RailEnv,
                  problem: AbstractProblemDescription,
                  rendering_call_back: SolveProblemRenderCallback = lambda *a, **k: None,
                  debug: bool = False,
                  loop_index: int = 0,
                  disable_verification_in_replay: bool = False,
                  expected_malfunction: Optional[ExperimentMalfunction] = None
                  ) -> SchedulingExperimentResult:
    """
    Solves an :class:`AbstractProblemDescription` and optionally verifies it againts the provided :class:`RailEnv`.

    Parameters
    ----------
    problem
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
        Used in verification if provided

    Returns
    -------
    SchedulingExperimentResult

    """
    # --------------------------------------------------------------------------------------
    # Preparations
    # --------------------------------------------------------------------------------------
    minimum_number_of_shortest_paths_over_all_agents = np.min(
        [len(agents_paths) for agents_paths in problem.agents_path_dict.values()])

    if minimum_number_of_shortest_paths_over_all_agents == 0:
        raise Exception("At least one Agent has no path to its target!")

    # --------------------------------------------------------------------------------------
    # Solve the problem
    # --------------------------------------------------------------------------------------
    start_build_problem = current_milli_time()
    build_problem_time = (current_milli_time() - start_build_problem) / 1000.0

    start_solver = current_milli_time()
    solution: AbstractSolutionDescription = problem.solve()
    solve_time = (current_milli_time() - start_solver) / 1000.0
    assert solution.is_solved()

    trainruns_dict: TrainrunDict = solution.get_trainruns_dict()

    if isinstance(problem, ASPProblemDescription):
        aspsolution: ASPSolutionDescription = solution
        actual_answer_set: Set[str] = aspsolution.asp_solution.answer_sets

        verification_by_file("answer_set", actual_answer_set, loop_index, problem.get_solver_name())

    verification_by_file("solution_trainrun_dict", solution.get_trainruns_dict(), loop_index, problem.get_solver_name())

    if debug:
        print("####train runs dict")
        print(_pp.pformat(trainruns_dict))

    # --------------------------------------------------------------------------------------
    # Replay and verifiy the solution
    # --------------------------------------------------------------------------------------
    verify_trainruns_dict(env=env,
                          trainruns_dict=trainruns_dict,
                          expected_malfunction=expected_malfunction,
                          expected_experiment_freeze=problem.experiment_freeze_dict
                          )
    total_reward = replay(env=env, loop_index=loop_index,
                          expected_malfunction=expected_malfunction,
                          problem=problem,
                          rendering_call_back=rendering_call_back, solution=solution,
                          debug=debug,
                          disable_verification_in_replay=disable_verification_in_replay)

    return SchedulingExperimentResult(total_reward=total_reward,
                                      solve_time=solve_time,
                                      optimization_costs=solution.get_objective_value(),
                                      build_problem_time=build_problem_time,
                                      solution=solution,
                                      experiment_freeze=problem.experiment_freeze_dict)


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
                          expected_experiment_freeze: Optional[ExperimentFreezeDict] = None
                          ):
    """
    Verify the consistency of a train run.

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
    expected_experiment_freeze

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
    if expected_experiment_freeze:
        _verify_trainruns_6_freeze(expected_experiment_freeze, trainruns_dict)


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


def _verify_trainruns_6_freeze(expected_experiment_freeze, trainruns_dict):
    for agent_id, experiment_freeze in expected_experiment_freeze.items():
        waypoint_dict: Dict[Waypoint, int] = {
            trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
            for trainrun_waypoint in trainruns_dict[agent_id]
        }

        # is freeze_visit respected?
        for waypoint in experiment_freeze.freeze_visit:
            assert waypoint in waypoint_dict

        # is freeze_earliest respected?
        for waypoint, scheduled_at in experiment_freeze.freeze_earliest.items():
            if waypoint in waypoint_dict:
                actual_scheduled_at = waypoint_dict[waypoint]
                assert actual_scheduled_at >= scheduled_at, \
                    f"expected {actual_scheduled_at} <= {scheduled_at} " + \
                    f"for {waypoint} of agent {agent_id}"

        # is freeze_latest respected?
        for waypoint, scheduled_at in experiment_freeze.freeze_latest.items():
            if waypoint in waypoint_dict:
                actual_scheduled_at = waypoint_dict[waypoint]
                assert actual_scheduled_at <= scheduled_at, \
                    f"expected {actual_scheduled_at} <= {scheduled_at} " + \
                    f"for {waypoint} of agent {agent_id}"

        # is freeze_banned respected?
        for waypoint in experiment_freeze.freeze_banned:
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
        initial_trainrun_waypoint = trainruns_dict[agent_id][0]
        assert initial_trainrun_waypoint.waypoint.position == agent.initial_position, \
            f"agent {agent} does not start in expected initial position, found {initial_trainrun_waypoint}"
        assert initial_trainrun_waypoint.waypoint.direction == agent.initial_direction, \
            f"agent {agent} does not start in expected initial direction, found {initial_trainrun_waypoint}"
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
                              direction=ASPProblemDescription.MAGIC_DIRECTION_FOR_TARGET),
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
                assert trainrun_waypoint.scheduled_at >= previous_trainrun_waypoint.scheduled_at + minimum_running_time_per_cell, \
                    f"agent {agent_id} inconsistency: to {trainrun_waypoint} " + \
                    f"from {previous_trainrun_waypoint} " + \
                    f"minimum running time={minimum_running_time_per_cell}"
            # 1.b) ensure train run is non-circular
            assert trainrun_waypoint not in previous_waypoints
            previous_trainrun_waypoint = trainrun_waypoint
            previous_waypoints.add(trainrun_waypoint.waypoint)


def replay(env: RailEnv,
           problem: AbstractProblemDescription,
           solution: AbstractSolutionDescription,
           expected_malfunction: Optional[ExperimentMalfunction] = None,
           rendering_call_back: SolveProblemRenderCallback = lambda *a, **k: None,
           debug: bool = False,
           loop_index: int = 0,
           stop_on_malfunction: bool = False,
           disable_verification_in_replay: bool = False) -> Optional[ExperimentMalfunction]:
    """
    Replay the solution an check whether the actions againts FLATland env can be performed as against.
    Verifies that the solution is indeed a solution in the FLATland sense.

    Parameters
    ----------

    problem
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


    Returns
    -------
    Optional[Malfunction]
        The malfunction in `stop_on_malfunction` mode, `None` else.

    """
    total_reward = 0
    time_step = 0
    solver_name = problem.get_solver_name()
    ap: ControllerFromTrainruns = solution.create_action_plan()
    if debug:
        print("  **** solution to replay:")
        print(_pp.pformat(solution.get_trainruns_dict()))
        print("  **** action plan to replay:")
        ap.print_action_plan()
        print("  **** expected_malfunction to replay:")
        print(_pp.pformat(expected_malfunction))
    actual_action_plan = [ap.act(time_step) for time_step in range(env._max_episode_steps)]
    verification_by_file("action_plan", actual_action_plan, loop_index, solver_name)
    while not env.dones['__all__'] and time_step <= env._max_episode_steps:
        fail = _check_fail(ap, debug, disable_verification_in_replay, env, expected_malfunction, problem, time_step)
        if fail:
            raise Exception("Unexpected state. See above for !!=unexpected position, MM=unexpected malfuntion")

        actions = ap.act(time_step)

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

        rendering_call_back(test_id=loop_index, solver_name=solver_name, i_step=time_step)

        # if all agents have reached their goals, break
        if done['__all__']:
            break

        time_step += 1
    if stop_on_malfunction:
        return None
    else:
        return total_reward


def _check_fail(ap, debug, disable_verification_in_replay, env, malfunction, problem, time_step):
    fail = False
    for agent in env.agents:
        prefix = ""
        # TODO ortools does not support multispeed yet, hence we cannot test whether the entry times are correct
        if isinstance(problem, ASPProblemDescription) and not disable_verification_in_replay:
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
                    f"(initial_position={agent.initial_position}, initial_direction={agent.initial_direction}, target={agent.target} "
                    f"with speed={agent.speed_data} and malfunction={agent.malfunction_data}, expected waypoint={we} "

                )
    return fail
