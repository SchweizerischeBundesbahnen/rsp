import pprint
from typing import Callable, Tuple

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

# TODO SIM-146 refactor: nothing special to ASP -> add freeze stuff to abc
from rsp.asp.asp_problem_description import ASPProblemDescription
from rsp.rescheduling.rescheduling_utils import get_freeze_for_malfunction, ExperimentFreezeDict
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.experiment_solver import RendererForEnvInit, RendererForEnvCleanup, RendererForEnvRender
from rsp.utils.experiment_utils import solve_problem, SchedulingExperimentResult

_pp = pprint.PrettyPrinter(indent=4)


def schedule_full(k: int,
                  static_rail_env: RailEnv,
                  rendering: bool = False,
                  debug: bool = False,
                  ) -> Tuple[ASPProblemDescription, SchedulingExperimentResult]:
    """
    Solves the Full Scheduling Problem for static rail env (i.e. without malfunctions).

    Parameters
    ----------
    k
        number of routing alterantives to consider
    static_rail_env
    rendering
    debug

    Returns
    -------
    Tuple[ASPProblemDescription, SchedulingExperimentResult]
        the problem description and the results

    """
    # --------------------------------------------------------------------------------------
    # Generate k shortest paths
    # --------------------------------------------------------------------------------------
    # TODO https://gitlab.aicrowd.com/flatland/flatland/issues/302: add method to FLATland to create of k shortest paths for all agents
    agents_paths_dict = {
        i: get_k_shortest_paths(static_rail_env,
                                agent.initial_position,
                                agent.initial_direction,
                                agent.target,
                                k) for i, agent in enumerate(static_rail_env.agents)
    }

    # --------------------------------------------------------------------------------------
    # Rendering
    # --------------------------------------------------------------------------------------
    if rendering:
        from rsp.utils.experiment_render_utils import cleanup_renderer_for_env
        from rsp.utils.experiment_render_utils import render_env
        from rsp.utils.experiment_render_utils import init_renderer_for_env
        init_renderer_for_env = init_renderer_for_env
        render_renderer_for_env = render_env
        cleanup_renderer_for_env = cleanup_renderer_for_env
    else:
        init_renderer_for_env: RendererForEnvInit = lambda *args, **kwargs: None
        render_renderer_for_env: RendererForEnvRender = lambda *args, **kwargs: None
        cleanup_renderer_for_env: RendererForEnvCleanup = lambda *args, **kwargs: None

    renderer = init_renderer_for_env(static_rail_env, rendering)

    def rendering_call_back(test_id: int, solver_name, i_step: int):
        render_renderer_for_env(renderer, test_id, solver_name, i_step)

    # --------------------------------------------------------------------------------------
    # Produce a full schedule
    # --------------------------------------------------------------------------------------
    schedule_problem = ASPProblemDescription(env=static_rail_env,
                                             agents_path_dict=agents_paths_dict)

    schedule_result = solve_problem(
        env=static_rail_env,
        problem=schedule_problem,
        rendering_call_back=rendering_call_back,
        debug=debug)

    # rendering hooks
    cleanup_renderer_for_env(renderer)

    return schedule_problem, schedule_result


def reschedule_full_after_malfunction(
        schedule_problem: ASPProblemDescription,
        schedule_trainruns: TrainrunDict,
        static_rail_env: RailEnv,
        malfunction: ExperimentMalfunction,
        malfunction_rail_env: RailEnv,
        malfunction_env_reset: Callable[[], None],
        debug: bool = False,
        disable_verification_in_replay: bool = False,
        rendering: bool = False,
) -> SchedulingExperimentResult:
    """
    Solve the Full Re-Scheduling Problem for static rail env (i.e. without malfunctions).

    Parameters
    ----------
    schedule_problem
    schedule_trainruns
    static_rail_env
    malfunction
    malfunction_rail_env
    malfunction_env_reset
    debug
    disable_verification_in_replay
    rendering

    Returns
    -------
    SchedulingExperimentResult
    """

    # --------------------------------------------------------------------------------------
    # Rendering
    # --------------------------------------------------------------------------------------
    if rendering:
        from rsp.utils.experiment_render_utils import cleanup_renderer_for_env
        from rsp.utils.experiment_render_utils import render_env
        from rsp.utils.experiment_render_utils import init_renderer_for_env
        init_renderer_for_env = init_renderer_for_env
        render_renderer_for_env = render_env
        cleanup_renderer_for_env = cleanup_renderer_for_env
    else:
        init_renderer_for_env: RendererForEnvInit = lambda *args, **kwargs: None
        render_renderer_for_env: RendererForEnvRender = lambda *args, **kwargs: None
        cleanup_renderer_for_env: RendererForEnvCleanup = lambda *args, **kwargs: None

    renderer = init_renderer_for_env(static_rail_env, rendering)

    def rendering_call_back(test_id: int, solver_name, i_step: int):
        render_renderer_for_env(renderer, test_id, solver_name, i_step)

    # --------------------------------------------------------------------------------------
    # Full Re-Scheduling
    # --------------------------------------------------------------------------------------

    freeze_dict: ExperimentFreezeDict = get_freeze_for_malfunction(malfunction, schedule_trainruns, static_rail_env,
                                                                   schedule_problem.agents_path_dict)

    full_reschedule_problem: ASPProblemDescription = schedule_problem.get_copy_for_experiment_freeze(
        experiment_freeze_dict=freeze_dict,
        schedule_trainruns=schedule_trainruns
    )

    full_reschedule_result = solve_problem(
        env=malfunction_rail_env,
        problem=full_reschedule_problem,
        rendering_call_back=rendering_call_back,
        debug=debug,
        expected_malfunction=malfunction,
        disable_verification_in_replay=disable_verification_in_replay
    )
    cleanup_renderer_for_env(renderer)
    malfunction_env_reset()

    return full_reschedule_result


def reschedule_delta_after_malfunction(
        schedule_problem: ASPProblemDescription,
        full_reschedule_trainruns: TrainrunDict,
        schedule_trainruns: TrainrunDict,
        malfunction: ExperimentMalfunction,
        malfunction_rail_env: RailEnv,
        rendering: bool = False,
        debug: bool = False,
) -> SchedulingExperimentResult:
    """

    Parameters
    ----------
    schedule_problem
    full_reschedule_trainruns
    force_freeze
    malfunction
    malfunction_rail_env
    rendering
    debug

    Returns
    -------
    SchedulingExperimentResult

    """

    # --------------------------------------------------------------------------------------
    # Rendering
    # --------------------------------------------------------------------------------------
    if rendering:
        from rsp.utils.experiment_render_utils import cleanup_renderer_for_env
        from rsp.utils.experiment_render_utils import render_env
        from rsp.utils.experiment_render_utils import init_renderer_for_env
        init_renderer_for_env = init_renderer_for_env
        render_renderer_for_env = render_env
        cleanup_renderer_for_env = cleanup_renderer_for_env
    else:
        init_renderer_for_env: RendererForEnvInit = lambda *args, **kwargs: None
        render_renderer_for_env: RendererForEnvRender = lambda *args, **kwargs: None
        cleanup_renderer_for_env: RendererForEnvCleanup = lambda *args, **kwargs: None

    renderer = init_renderer_for_env(malfunction_rail_env, rendering)

    def rendering_call_back(test_id: int, solver_name, i_step: int):
        render_renderer_for_env(renderer, test_id, solver_name, i_step)

    # --------------------------------------------------------------------------------------
    # Delta Re-Scheduling
    # --------------------------------------------------------------------------------------

    delta, force_freeze = determine_delta(full_reschedule_trainruns,
                                          malfunction,
                                          schedule_trainruns,
                                          verbose=False)

    freeze_dict: ExperimentFreezeDict = get_freeze_for_malfunction(
        malfunction,
        full_reschedule_trainruns,
        malfunction_rail_env,
        schedule_problem.agents_path_dict,
        force_freeze=force_freeze
    )

    delta_reschedule_problem: ASPProblemDescription = schedule_problem.get_copy_for_experiment_freeze(
        experiment_freeze_dict=freeze_dict,
        schedule_trainruns=full_reschedule_trainruns
    )

    delta_reschedule_result = solve_problem(
        env=malfunction_rail_env,
        problem=delta_reschedule_problem,
        rendering_call_back=rendering_call_back,
        debug=debug,
        expected_malfunction=malfunction)
    cleanup_renderer_for_env(renderer)
    return delta_reschedule_result


# TODO SIM-146 ASP performance enhancement: we consider the worst case: we leave everything open;
#      we do not give the ASP solver the information about the full re-schedule!!
def determine_delta(full_reschedule_trainrunwaypoints_dict: TrainrunDict,
                    malfunction: ExperimentMalfunction,
                    schedule_trainrunwaypoints: TrainrunDict,
                    verbose: bool = False) -> Tuple[TrainrunDict, TrainrunDict]:
    """
    Delta contains the information about what is changed by the malfunction with respect to the malfunction
    - all train run way points in the re-schedule that are different from the initial schedule.
    - this includes the run way point after the malfunction which is delayed!

    Freeze contains all waypoints/times we can freeze/constrain:
    - all train run way points that are the same in the re-schedule
    - the train run way point after the malfunction (including the delay)

    We can then (see :meth:`rsp.asp.asp_problem_description.ASPProblemDescription._translate_experiment_freeze_to_ASP`.)
    - constrain all times in the Inverse Delta to the value in the inverse delta
    - constrain all all other times to be greater or equal the start of the malfunction.

    Parameters
    ----------
    full_reschedule_trainrunwaypoints_dict
    malfunction
    schedule_trainrunwaypoints
    verbose

    """
    if verbose:
        print(f"  **** full re-schedule")
        print(_pp.pformat(full_reschedule_trainrunwaypoints_dict))
    # Delta is all train run way points in the re-schedule that are not also in the schedule
    delta: TrainrunDict = {
        agent_id: sorted(list(
            set(full_reschedule_trainrunwaypoints_dict[agent_id]).difference(
                set(schedule_trainrunwaypoints[agent_id]))),
            key=lambda p: p.scheduled_at)
        for agent_id in schedule_trainrunwaypoints.keys()
    }

    freeze: TrainrunDict = \
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
