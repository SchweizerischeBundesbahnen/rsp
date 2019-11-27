"""Solve a problem a."""
import pprint
from typing import Optional, NamedTuple, Set, Callable

import numpy as np
from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.abstract_problem_description import AbstractProblemDescription
from rsp.abstract_solution_description import AbstractSolutionDescription
from rsp.asp.asp_problem_description import ASPProblemDescription
from rsp.asp.asp_solution_description import ASPSolutionDescription
from rsp.utils.data_types import Malfunction
from rsp.utils.general_utils import current_milli_time, verification

SchedulingExperimentResult = NamedTuple('SchedulingExperimentResult',
                                        [('total_reward', int),
                                         ('solve_time', float),
                                         ('optimization_costs', float),
                                         ('build_problem_time', float),
                                         ('solution', AbstractSolutionDescription)])

# test_id: int, solver_name: str, i_step: int
SolveProblemRenderCallback = Callable[[int, str, int], None]

_pp = pprint.PrettyPrinter(indent=4)


# --------------------------------------------------------------------------------------
# Solve an `AbstractProblemDescription`
# --------------------------------------------------------------------------------------
def solve_problem(env: RailEnv,
                  problem: AbstractProblemDescription,
                  rendering_call_back: SolveProblemRenderCallback = lambda *a, **k: None,
                  debug: bool = False,
                  loop_index: int = 0,
                  disable_verification_in_replay: bool = False,
                  expected_malfunction: Optional[Malfunction] = None
                  ) -> SchedulingExperimentResult:
    """
    Solves an :class:`AbstractProblemDescription` and optionally verifies it againts the provided :class:`RailEnv`.

    Parameters
    ----------
    problem
    disable_verification_in_replay
        Whether it is tested the replay corresponds to the problem's solution
        TODO SIM-105 Should this be disable replay completely to gain time, i.e. whenever replay is perfor
        TODO SIM-105 profile experiments to test how much time replay takes in the experiments
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

    if isinstance(problem, ASPProblemDescription):
        aspsolution: ASPSolutionDescription = solution
        actual_answer_set: Set[str] = aspsolution.asp_solution.answer_sets

        verification("answer_set", actual_answer_set, loop_index, problem.get_solver_name())

    verification("solution_trainrun_dict", solution.get_trainruns_dict(), loop_index, problem.get_solver_name())

    # --------------------------------------------------------------------------------------
    # Replay and verifiy the solution
    # --------------------------------------------------------------------------------------
    total_reward = replay(env=env, loop_index=loop_index, expected_malfunction=expected_malfunction, problem=problem,
                          rendering_call_back=rendering_call_back, solution=solution,
                          debug=debug,
                          disable_verification_in_replay=disable_verification_in_replay)

    return SchedulingExperimentResult(total_reward=total_reward,
                                      solve_time=solve_time,
                                      optimization_costs=solution.get_objective_value(),
                                      build_problem_time=build_problem_time,
                                      solution=solution)


def replay(env: RailEnv,
           problem: AbstractProblemDescription,
           solution: AbstractSolutionDescription,
           expected_malfunction: Optional[Malfunction] = None,
           rendering_call_back: SolveProblemRenderCallback = lambda *a, **k: None,
           debug: bool = False,
           loop_index: int = 0,
           stop_on_malfunction: bool = False,
           disable_verification_in_replay: bool = False) -> Optional[Malfunction]:
    """
    Replay the solution an check whether the actions againts FLATland env can be performed as against.
    Verifies that the solution is indeed a solution in the FLATland sense.

    Parameters
    ----------

    problem
    disable_verification_in_replay
        Whether it is tested the replay corresponds to the problem's solution
        TODO SIM-105 Should this be disable replay completely to gain time, i.e. whenever replay is perfor
        TODO SIM-105 profile experiments to test how much time replay takes in the experiments
    env
        The env to run the verification with
    rendering_call_back
        Called every step in replay
    debug
        Display debugging information
    loop_index
        Used for display, should identify the problem instance
    expected_malfunction
        If provided and verification is enabled, it is checked that the malfunction happens as expected.
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
    verification("action_plan", actual_action_plan, loop_index, solver_name)
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
                    return Malfunction(time_step, agent.handle, agent.malfunction_data['malfunction'] + 1)

        rendering_call_back(test_id=loop_index, solver_name=solver_name, i_step=time_step)

        # if all agents have reached their goals, break
        if done['__all__']:
            break

        time_step += 1
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
                    f"with speed={agent.speed_data} and malfunction={agent.malfunction_data}, expected waypoint={we}")
    return fail
