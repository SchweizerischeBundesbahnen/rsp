import pprint
from typing import Dict, List

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint

from rsp.asp.asp_scheduling_helper import reschedule_full_after_malfunction, schedule_full, determine_delta, \
    reschedule_delta_after_malfunction
from rsp.asp.asp_solution_description import ASPSolutionDescription
from rsp.utils.data_types import ExperimentResults
from rsp.utils.experiment_solver import AbstractSolver, RendererForEnvInit, RendererForEnvRender, RendererForEnvCleanup
from rsp.utils.experiment_utils import replay_until_malfunction


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
        agents_paths_dict, schedule_problem, schedule_result, schedule_solution = schedule_full(k, static_rail_env,
                                                                                                rendering)

        schedule_trainruns: Dict[int, List[TrainrunWaypoint]] = schedule_solution.get_trainruns_dict()

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
        # Re-schedule Full
        # --------------------------------------------------------------------------------------

        full_reschedule_result, full_reschedule_solution = reschedule_full_after_malfunction(agents_paths_dict,
                                                                                             malfunction,
                                                                                             malfunction_env_reset,
                                                                                             malfunction_rail_env,
                                                                                             schedule_problem,
                                                                                             schedule_trainruns,
                                                                                             static_rail_env, rendering)
        malfunction_env_reset()

        if verbose:
            print(f"  **** full re-schedule_solution=\n{full_reschedule_solution.get_trainruns_dict()}")
        full_reschedule_trainruns: Dict[int, List[TrainrunWaypoint]] = full_reschedule_solution.get_trainruns_dict()

        # --------------------------------------------------------------------------------------
        # Re-Schedule Delta
        # --------------------------------------------------------------------------------------

        delta, inverse_delta = determine_delta(full_reschedule_trainruns, malfunction,
                                               schedule_trainruns, verbose=False)

        delta_reschedule_result = reschedule_delta_after_malfunction(agents_paths_dict,
                                                                     full_reschedule_trainruns,
                                                                     inverse_delta,
                                                                     malfunction,
                                                                     malfunction_rail_env,
                                                                     schedule_problem=schedule_problem,
                                                                     rendering=rendering,
                                                                     init_renderer_for_env=init_renderer_for_env,
                                                                     render_renderer_for_env=render_renderer_for_env,
                                                                     cleanup_renderer_for_env=cleanup_renderer_for_env)
        malfunction_env_reset()
        delta_reschedule_solution: ASPSolutionDescription = delta_reschedule_result.solution

        if verbose:
            print(f"  **** delta re-schedule solution")
            print(delta_reschedule_solution.get_trainruns_dict())

        # TODO ASP performance analyse running times (grounding vs solving - etc.)

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
