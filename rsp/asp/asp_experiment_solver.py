import pprint
from typing import Set, Dict, List

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint

from rsp.asp.asp_problem_description import ASPProblemDescription
from rsp.asp.asp_scheduling_helper import reschedule, schedule_static, determine_delta
from rsp.asp.asp_solution_description import ASPSolutionDescription
from rsp.utils.data_types import ExperimentResults
from rsp.utils.experiment_render_utils import render_env
from rsp.utils.experiment_solver import AbstractSolver, RendererForEnvInit, RendererForEnvRender, RendererForEnvCleanup
from rsp.utils.experiment_utils import solve_problem, replay_until_malfunction


class ASPExperimentSolver(AbstractSolver):
    """
    Implements `AbstractSolver` for ASP.

    Methods
    -------
    run_experiment_trial:
        Returns the correct data format to run tests on full research pipeline
    """
    _pp = pprint.PrettyPrinter(indent=4)

    def run_experiment_trial(
            self,
            static_rail_env: RailEnv,
            malfunction_rail_env: RailEnv,
            malfunction_env_reset,
            k: int = 10,
            disable_verification_by_replay: bool = False,
            verbose: bool = False,
            rendering: bool = False,
            init_renderer_for_env: RendererForEnvInit = lambda *args, **kwargs: None,
            render_renderer_for_env: RendererForEnvRender = lambda *args, **kwargs: None,
            cleanup_renderer_for_env: RendererForEnvCleanup = lambda *args, **kwargs: None,
    ) -> ExperimentResults:
        """
        Runs the experiment.

        Parameters
        ----------
        static_rail_env: RailEnv
            Rail environment without any malfunction
        malfunction_rail_env: RailEnv
            Rail environment with one single malfunction

        Returns
        -------
        ExperimentResults
        """
        agents_paths_dict, schedule_problem, schedule_result, schedule_solution = schedule_static(k, static_rail_env,
                                                                                                  rendering)

        schedule_trainruns: Dict[int, List[TrainrunWaypoint]] = schedule_solution.get_trainruns_dict()
        schedule_trainrunwaypoints = {agent_id: set(train_run) for agent_id, train_run in schedule_trainruns.items()}

        if verbose:
            print(f"  **** schedule_solution={schedule_trainruns}")

        # --------------------------------------------------------------------------------------
        # Generate malfuntion
        # --------------------------------------------------------------------------------------

        malfunction_env_reset()
        malfunction = replay_until_malfunction(solution=schedule_solution, env=malfunction_rail_env)
        malfunction_env_reset()
        if not malfunction:
            raise Exception("Could not produce a malfunction")

        if verbose:
            print(f"  **** malfunction={malfunction}")

        # --------------------------------------------------------------------------------------
        # Re-schedule_static Full
        # --------------------------------------------------------------------------------------

        full_reschedule_result, full_reschedule_solution = reschedule(agents_paths_dict, malfunction,
                                                                      malfunction_env_reset, malfunction_rail_env,
                                                                      schedule_problem,
                                                                      schedule_trainruns, static_rail_env, rendering)

        if verbose:
            print(f"  **** full re-schedule_solution=\n{full_reschedule_solution.get_trainruns_dict()}")
        full_reschedule_trainruns: Dict[int, List[TrainrunWaypoint]] = full_reschedule_solution.get_trainruns_dict()
        full_reschedule_trainrunwaypoints_dict: Dict[int, Set[TrainrunWaypoint]] = \
            {agent_id: set(train_run) for agent_id, train_run in full_reschedule_trainruns.items()}

        # --------------------------------------------------------------------------------------
        # Determine Delta
        # --------------------------------------------------------------------------------------

        delta, inverse_delta = determine_delta(full_reschedule_trainrunwaypoints_dict, malfunction,
                                               schedule_trainrunwaypoints, verbose)

        # --------------------------------------------------------------------------------------
        # Re-schedule_static Delta
        # --------------------------------------------------------------------------------------
        # does not work yet
        if False:
            delta_reschedule_problem: ASPProblemDescription = schedule_problem.get_freezed_copy_for_rescheduling(
                malfunction=malfunction,
                freeze=inverse_delta,
                schedule_trainruns=full_reschedule_trainruns
            )

            renderer = init_renderer_for_env(malfunction_rail_env, rendering)

            def render(test_id: int, solver_name, i_step: int):
                render_env(renderer, test_id, solver_name, i_step)

            delta_reschedule_result = solve_problem(
                env=malfunction_rail_env,
                problem=delta_reschedule_problem,
                agents_paths_dict=agents_paths_dict,
                rendering_call_back=render,
                debug=False,
                malfunction=malfunction)
            cleanup_renderer_for_env(renderer)
            malfunction_env_reset()
            delta_reschedule_solution: ASPSolutionDescription = delta_reschedule_result.solution

            if verbose:
                print(f"  **** delta re-schedule solution")
                print(delta_reschedule_solution.get_trainruns_dict())

        # TODO SIM-105 analyse running times (grounding vs solving - etc.)
        # TODO SIM-105 display sum of delays in both approaches

        # --------------------------------------------------------------------------------------
        # Result
        # --------------------------------------------------------------------------------------
        current_results = ExperimentResults(time_full=schedule_result.solve_time,
                                            time_full_after_malfunction=full_reschedule_result.solve_time,
                                            time_delta_after_malfunction=-1,
                                            solution_full=schedule_solution.get_trainruns_dict(),
                                            solution_full_after_malfunction=[],
                                            solution_delta_after_malfunction=[],
                                            costs_full=schedule_result.optimization_costs,
                                            costs_full_after_malfunction=full_reschedule_result.optimization_costs,
                                            costs_delta_after_malfunction=-1,
                                            delta=delta)
        return current_results
