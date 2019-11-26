"""Solve a problem a."""
from typing import Dict, List, Optional, NamedTuple, Set, Callable

import numpy as np
from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import Trainrun, Waypoint

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


# --------------------------------------------------------------------------------------
# Solve an `AbstractProblemDescription`
# --------------------------------------------------------------------------------------
def solve_problem(env: RailEnv,
                  problem: AbstractProblemDescription,
                  agents_paths_dict: Dict[int, List[Trainrun]],
                  rendering_call_back: SolveProblemRenderCallback = lambda *a, **k: None,
                  debug: bool = False,
                  loop_index: int = 0,
                  disable_verification_in_replay: bool = False,
                  malfunction: Optional[Malfunction] = None
                  ) -> SchedulingExperimentResult:
    solver_name = problem.get_solver_name()

    # --------------------------------------------------------------------------------------
    # Preparations
    # --------------------------------------------------------------------------------------
    minimum_number_of_shortest_paths_over_all_agents = np.min(
        [len(agents_paths) for agents_paths in agents_paths_dict.values()])

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
    total_reward = _replay(debug, disable_verification_in_replay, env, loop_index, malfunction, problem,
                           rendering_call_back, solution, solver_name)

    return SchedulingExperimentResult(total_reward=total_reward,
                                      solve_time=solve_time,
                                      optimization_costs=solution.get_objective_value(),
                                      build_problem_time=build_problem_time,
                                      solution=solution)


def _replay(debug, disable_verification_in_replay, env, loop_index, malfunction, problem, rendering_call_back, solution,
            solver_name):
    total_reward = 0
    time_step = 0
    ap: ControllerFromTrainruns = solution.create_action_plan()
    if debug:
        print(solution.get_trainruns_dict())
        ap.print_action_plan()
    actual_action_plan = [ap.act(time_step) for time_step in range(env._max_episode_steps)]
    verification("action_plan", actual_action_plan, loop_index, solver_name)
    while not env.dones['__all__'] and time_step <= env._max_episode_steps:
        fail = _check_fail(ap, debug, disable_verification_in_replay, env, malfunction, problem, time_step)
        if fail:
            raise Exception("Unexpected state. See above for !!=unexpected position, MM=unexpected malfuntion")

        actions = ap.act(time_step)

        if debug:
            print(f"env._elapsed_steps={env._elapsed_steps}")
            print("actions [{}]->[{}] actions={}".format(time_step, time_step + 1, actions))

        obs, all_rewards, done, _ = env.step(actions)
        total_reward += sum(np.array(list(all_rewards.values())))

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


# --------------------------------------------------------------------------------------
# Malfunction
# --------------------------------------------------------------------------------------
# TODO SIM-105 refactor: we copy alot of code with the replay function. should we condense this all into one function with optionals. Such as
#  Replay from time t
#  replay to time t
#  Replay from t1 to t2
#  Replay until malfunct

def replay_until_malfunction(solution: AbstractSolutionDescription,
                             env: RailEnv,
                             debug: bool = False) -> Optional[Malfunction]:
    """
    Replays the action plan from the solution until an agent gets a malfunction.

    Parameters
    ----------
    solution
    problem
    env
    debug

    Returns
    -------
    Tuple[int,int]
        time step of the malfunction

    """
    # --------------------------------------------------------------------------------------
    # Replay and verifiy the solution
    # --------------------------------------------------------------------------------------
    time_step = 0
    ap: ControllerFromTrainruns = solution.create_action_plan()

    while not env.dones['__all__'] and time_step <= env._max_episode_steps:
        if debug:
            for agent in env.agents:
                print("[{}] agent {} at {} {} ".format(time_step, agent.handle, agent.position, agent.direction, agent))

        for a, agent in enumerate(env.agents):
            we: Waypoint = ap.get_waypoint_before_or_at_step(a, time_step)
            assert agent.position == we.position, \
                "before [{}] agent {} replay expected position {}, actual position {}".format(
                    time_step, a, we.position, agent.position)
        actions = ap.act(time_step)

        obs, all_rewards, done, _ = env.step(actions)

        for agent in env.agents:
            if agent.malfunction_data['malfunction'] > 0:
                # malfunction duration is already decreased by one in this step(), therefore add +1!
                return Malfunction(time_step, agent.handle, agent.malfunction_data['malfunction'] + 1)

        # if all agents have reached their goals, break
        if done['__all__']:
            break

        time_step += 1
    return None
