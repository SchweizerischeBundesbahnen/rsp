from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths

from rsp.asp.asp_problem_description import ASPProblemDescription
from rsp.asp.asp_solution_description import ASPSolutionDescription
from rsp.rescheduling.rescheduling_utils import get_freeze_for_malfunction
from rsp.utils.experiment_solver import RendererForEnvInit, RendererForEnvCleanup, RendererForEnvRender
from rsp.utils.experiment_utils import solve_problem


# TODO SIM-105 docstring
def schedule_static(k: int,
                    static_rail_env: RailEnv,
                    rendering: bool = False,
                    init_renderer_for_env: RendererForEnvInit = lambda *args, **kwargs: None,
                    render_renderer_for_env: RendererForEnvRender = lambda *args, **kwargs: None,
                    cleanup_renderer_for_env: RendererForEnvCleanup = lambda *args, **kwargs: None, ):
    # --------------------------------------------------------------------------------------
    # Generate k shortest paths
    # --------------------------------------------------------------------------------------
    # TODO add method to FLATland to create of k shortest paths for all agents
    agents_paths_dict = {
        i: get_k_shortest_paths(static_rail_env,
                                agent.initial_position,
                                agent.initial_direction,
                                agent.target,
                                k) for i, agent in enumerate(static_rail_env.agents)
    }
    # --------------------------------------------------------------------------------------
    # Produce a full schedule_static
    # --------------------------------------------------------------------------------------
    schedule_problem = ASPProblemDescription(env=static_rail_env,
                                             agents_path_dict=agents_paths_dict)

    # rendering hooks
    renderer = init_renderer_for_env(static_rail_env, rendering)

    def render(test_id: int, solver_name, i_step: int):
        render_renderer_for_env(renderer, test_id, solver_name, i_step)

    schedule_result = solve_problem(
        env=static_rail_env,
        problem=schedule_problem,
        agents_paths_dict=agents_paths_dict,
        rendering_call_back=render,
        debug=False)
    schedule_solution: ASPSolutionDescription = schedule_result.solution

    # rendering hooks
    cleanup_renderer_for_env(renderer)

    # TODO SIM-105 data structure and return type hints
    return agents_paths_dict, schedule_problem, schedule_result, schedule_solution


# TODO get rid of rendering because of tests.
def reschedule(agents_paths_dict,
               malfunction,
               malfunction_env_reset,
               malfunction_rail_env,
               schedule_problem,
               schedule_trainruns,
               static_rail_env,
               debug: bool = False,
               disable_verification_in_replay: bool = False,
               rendering: bool = False,
               init_renderer_for_env: RendererForEnvInit = lambda *args, **kwargs: None,
               render_renderer_for_env: RendererForEnvRender = lambda *args, **kwargs: None,
               cleanup_renderer_for_env: RendererForEnvCleanup = lambda *args, **kwargs: None, ):
    freeze = get_freeze_for_malfunction(malfunction, schedule_trainruns, static_rail_env)
    full_reschedule_problem: ASPProblemDescription = schedule_problem.get_freezed_copy_for_rescheduling(
        malfunction=malfunction,
        freeze=freeze,
        schedule_trainruns=schedule_trainruns
    )
    renderer = init_renderer_for_env(malfunction_rail_env, rendering)

    def render(test_id: int, solver_name, i_step: int):
        render_renderer_for_env(renderer, test_id, solver_name, i_step)

    full_reschedule_result = solve_problem(
        env=malfunction_rail_env,
        problem=full_reschedule_problem,
        agents_paths_dict=agents_paths_dict,
        rendering_call_back=render,
        debug=debug,
        malfunction=malfunction,
        disable_verification_in_replay=disable_verification_in_replay
    )
    cleanup_renderer_for_env(renderer)
    malfunction_env_reset()
    full_reschedule_solution: ASPSolutionDescription = full_reschedule_result.solution

    # TODO SIM-105 data structure and retun type hints
    return full_reschedule_result, full_reschedule_solution
