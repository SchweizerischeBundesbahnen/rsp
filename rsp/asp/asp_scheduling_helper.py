import pprint
from typing import Dict

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_trainrun_data_structures import Trainrun

from rsp.asp.asp_problem_description import ASPProblemDescription
from rsp.asp.asp_solution_description import ASPSolutionDescription
from rsp.rescheduling.rescheduling_utils import get_freeze_for_malfunction
from rsp.utils.data_types import Malfunction
from rsp.utils.experiment_solver import RendererForEnvInit, RendererForEnvCleanup, RendererForEnvRender
from rsp.utils.experiment_utils import solve_problem

_pp = pprint.PrettyPrinter(indent=4)


# TODO SIM-105 docstring and type hints in this module
def schedule_full(k: int,
                  static_rail_env: RailEnv,
                  rendering: bool = False,
                  debug: bool = False,
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
    # Produce a full schedule
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
        debug=debug)
    schedule_solution: ASPSolutionDescription = schedule_result.solution

    # rendering hooks
    cleanup_renderer_for_env(renderer)

    # TODO SIM-105 data structure and return type hints
    return agents_paths_dict, schedule_problem, schedule_result, schedule_solution


def reschedule_full_after_malfunction(agents_paths_dict,
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
    full_reschedule_problem: ASPProblemDescription = schedule_problem.get_freezed_copy_for_rescheduling_full_after_malfunction(
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


def reschedule_delta_after_malfunction(agents_paths_dict,
                                       full_reschedule_trainruns,
                                       inverse_delta, malfunction, malfunction_rail_env,
                                       schedule_problem,
                                       rendering: bool = False,
                                       debug: bool = False,
                                       init_renderer_for_env: RendererForEnvInit = lambda *args, **kwargs: None,
                                       render_renderer_for_env: RendererForEnvRender = lambda *args, **kwargs: None,
                                       cleanup_renderer_for_env: RendererForEnvCleanup = lambda *args, **kwargs: None,
                                       ):
    delta_reschedule_problem: ASPProblemDescription = schedule_problem.get_freezed_copy_for_rescheduling_delta_after_malfunction(
        malfunction=malfunction,
        freeze=inverse_delta,
        schedule_trainruns=full_reschedule_trainruns
    )
    renderer = init_renderer_for_env(malfunction_rail_env, rendering)

    def render(test_id: int, solver_name, i_step: int):
        render_renderer_for_env(renderer, test_id, solver_name, i_step)

    delta_reschedule_result = solve_problem(
        env=malfunction_rail_env,
        problem=delta_reschedule_problem,
        agents_paths_dict=agents_paths_dict,
        rendering_call_back=render,
        debug=debug,
        malfunction=malfunction)
    cleanup_renderer_for_env(renderer)
    return delta_reschedule_result


# TODO ASP performance enhancement: we consider the worst case: we leave everything open;
#      we do not give the ASP solver the information about the full re-schedule!!
def determine_delta(full_reschedule_trainrunwaypoints_dict: Dict[int, Trainrun],
                    malfunction: Malfunction, schedule_trainrunwaypoints: Dict[int, Trainrun],
                    verbose: bool = False):
    """
    Delta contains the information about what is changed by the malfunction with respect to the malfunction
    - all train run way points in the re-schedule that are different from the initial schedule.
    - this includes the run way point after the malfunction which is delayed!

    Freeze contains all waypoints/times we can freeze/constrain:
    - all train run way points that are the same in the re-schedule
    - the train run way point after the malfunction (including the delay)

    We can then (see :meth:`rsp.asp.asp_problem_description.ASPProblemDescription._translate_freeze_full_after_malfunction_to_ASP`.)
    - constrain all times in the Inverse Delta to the value in the inverse delta
    - constrain all all other times to be greater or equal the start of the malfunction.

    """
    if verbose:
        print(f"  **** full re-schedule")
        print(_pp.pformat(full_reschedule_trainrunwaypoints_dict))
    # Delta is all train run way points in the re-schedule that are not also in the schedule
    # TODO SIM-105 add type definition TrainrunDict to FLATland
    delta: Dict[int, Trainrun] = {
        agent_id: sorted(list(
            set(full_reschedule_trainrunwaypoints_dict[agent_id]).difference(
                set(schedule_trainrunwaypoints[agent_id]))),
            key=lambda p: p.scheduled_at)
        for agent_id in schedule_trainrunwaypoints.keys()
    }

    freeze: Dict[int, Trainrun] = \
        {agent_id: sorted(list(
            set(full_reschedule_trainrunwaypoints_dict[agent_id]).intersection(
                set(schedule_trainrunwaypoints[agent_id]))),
            key=lambda p: p.scheduled_at) for agent_id in delta.keys()}
    # add first element after malfunction to inverse delta -> wee ned to freeze it!
    trainrun_waypoint_after_malfunction = next(
        trainrun_waypoint for trainrun_waypoint in full_reschedule_trainrunwaypoints_dict[malfunction.agent_id] if
        trainrun_waypoint.scheduled_at > malfunction.time_step)
    freeze[malfunction.agent_id].append(trainrun_waypoint_after_malfunction)
    if verbose:
        print(f"  **** delta={_pp.pformat(delta)}")

    # TODO SIM-105 make option to switch verification on and off? is this the right place for this checks?
    # sanity checks
    for agent_id, delta_waypoints in delta.items():
        for delta_waypoint in delta_waypoints:
            assert delta_waypoint.scheduled_at >= malfunction.time_step, f"found \n\n"
            "  **** delta_waypoint {delta_waypoint} of agent {agent_id},\n\n"
            "  **** malfunction is {malfunction}.\n\n"
            "  **** schedule={schedule_trainruns[agent_id]}.\n\n"
            "  **** full re-schedule={full_reschedule_trainruns[agent_id]}"
    # Freeze are all train run way points in the re-schedule that not in the delta
    if verbose:
        print(f"  **** freeze ={_pp.pformat(freeze)}")
    return delta, freeze
