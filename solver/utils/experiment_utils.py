"""Solve a problem a."""
import os
from typing import Dict, List, Optional, NamedTuple, Set

import numpy as np
from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import Trainrun, Waypoint
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

from asp.asp_solution_description import ASPSolutionDescription
from solver.abstract_problem_description import AbstractProblemDescription
from solver.abstract_solution_description import AbstractSolutionDescription
from solver.asp.asp_problem_description import ASPProblemDescription
from utils.data_types import Malfunction
from utils.general_utils import current_milli_time, verification

SchedulingExperimentResult = NamedTuple('SchedulingExperimentResult',
                                        [('total_reward', int),
                                         ('solve_time', float),
                                         ('build_problem_time', float),
                                         ('solution', AbstractSolutionDescription)])


# --------------------------------------------------------------------------------------
# Solve an `AbstractProblemDescription`
# --------------------------------------------------------------------------------------
def solve_problem(env: RailEnv,
                  problem: AbstractProblemDescription,
                  agents_paths_dict: Dict[int, List[Trainrun]],
                  rendering: bool = False,
                  debug: bool = False,
                  loop_index: int = 0,
                  disable_verification_in_replay: bool = False,
                  malfunction: Optional[Malfunction] = None
                  ) -> SchedulingExperimentResult:
    solver_name = problem.get_solver_name()

    # --------------------------------------------------------------------------------------
    # Rendering
    # --------------------------------------------------------------------------------------
    # if rendering is on
    renderer = None
    if rendering:
        renderer = RenderTool(env, gl="PILSVG",
                              agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                              show_debug=True,
                              clear_debug_text=True,
                              screen_height=1000,
                              screen_width=1000)
        _render(renderer, loop_index, solver_name, 0)

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
    total_reward = 0
    time_step = 0
    ap: ControllerFromTrainruns = solution.create_action_plan()
    if debug:
        print(solution.get_trainruns_dict())
        ap.print_action_plan()

    actual_action_plan = [ap.act(time_step) for time_step in range(env._max_episode_steps)]

    verification("action_plan", actual_action_plan, loop_index, solver_name)

    while not env.dones['__all__'] and time_step <= env._max_episode_steps:
        if debug:
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

                print(
                    f"{prefix}[{time_step}] agent={agent.handle} at position={agent.position} in direction={agent.direction} with speed={agent.speed_data} and malfunction={agent.malfunction_data}, expected waypoint={we}")
            if fail:
                raise Exception("Unexpected state. See above for !!=unexpected position, MM=unexpected malfuntion")

        actions = ap.act(time_step)

        if debug:
            print(f"env._elapsed_steps={env._elapsed_steps}")
            print("actions [{}]->[{}] actions={}".format(time_step, time_step + 1, actions))

        obs, all_rewards, done, _ = env.step(actions)
        total_reward += sum(np.array(list(all_rewards.values())))

        if rendering:
            _render(renderer=renderer, test_id=loop_index, solver_name=solver_name, i_step=time_step)

        # if all agents have reached their goals, break
        if done['__all__']:
            break

        time_step += 1

    # --------------------------------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------------------------------
    if renderer is not None:
        _render(renderer=renderer, test_id=loop_index, solver_name=solver_name, i_step=loop_index)
        # close renderer window
        renderer.close_window()

    return SchedulingExperimentResult(total_reward, solve_time, build_problem_time, solution)


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


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _render(renderer: Optional[RenderTool], test_id: int, solver_name, i_step: int,
            image_output_directory: Optional[str] = './rendering_output'):
    if renderer is not None:
        renderer.render_env(show=True, show_observations=False, show_predictions=False)
        if image_output_directory is not None:
            if not os.path.exists(image_output_directory):
                os.makedirs(image_output_directory)
            renderer.gl.save_image(os.path.join(image_output_directory,
                                                "flatland_frame_{:04d}_{:04d}_{}.png".format(test_id,
                                                                                             i_step,
                                                                                             solver_name)))
