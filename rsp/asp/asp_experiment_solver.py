import pprint
from typing import Dict, List

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint

from rsp.asp.asp_scheduling_helper import reschedule_full_after_malfunction, schedule_full, \
    reschedule_delta_after_malfunction
from rsp.asp.asp_solution_description import ASPSolutionDescription
from rsp.utils.data_types import ExperimentResults
from rsp.utils.experiment_solver import AbstractSolver
from rsp.utils.experiment_utils import replay


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
            rendering: bool = False
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
        schedule_problem, schedule_result = schedule_full(k, static_rail_env, rendering=rendering, debug=False)
        schedule_solution = schedule_result.solution

        schedule_trainruns: Dict[int, List[TrainrunWaypoint]] = schedule_solution.get_trainruns_dict()

        if verbose:
            print(f"  **** schedule_solution={schedule_trainruns}")

        # --------------------------------------------------------------------------------------
        # Generate malfuntion
        # --------------------------------------------------------------------------------------

        malfunction_env_reset()
        malfunction = replay(solution=schedule_solution, env=malfunction_rail_env, stop_on_malfunction=True,
                             problem=schedule_problem,
                             disable_verification_in_replay=True)
        malfunction_env_reset()
        if not malfunction:
            raise Exception("Could not produce a malfunction")

        if verbose:
            print(f"  **** malfunction={malfunction}")

        # --------------------------------------------------------------------------------------
        # Re-schedule Full
        # --------------------------------------------------------------------------------------

        # TODO SIM-146 unify reschedule_full_after_malfunction and reschedule_delta_after_malfunction
        # TODO SIM-146 add verification that ExperimentFreeze is respected!
        full_reschedule_result = reschedule_full_after_malfunction(
            malfunction=malfunction,
            malfunction_env_reset=malfunction_env_reset,
            malfunction_rail_env=malfunction_rail_env,
            schedule_problem=schedule_problem,
            schedule_trainruns=schedule_trainruns,
            static_rail_env=static_rail_env,
            rendering=rendering,
            debug=verbose
        )
        malfunction_env_reset()
        full_reschedule_solution = full_reschedule_result.solution

        if verbose:
            print(f"  **** full re-schedule_solution=\n{full_reschedule_solution.get_trainruns_dict()}")
        full_reschedule_trainruns: Dict[int, List[TrainrunWaypoint]] = full_reschedule_solution.get_trainruns_dict()

        # --------------------------------------------------------------------------------------
        # Re-Schedule Delta
        # --------------------------------------------------------------------------------------

        delta_reschedule_result = reschedule_delta_after_malfunction(
            full_reschedule_trainruns=full_reschedule_trainruns,
            schedule_trainruns=schedule_trainruns,
            malfunction=malfunction,
            malfunction_rail_env=malfunction_rail_env,
            schedule_problem=schedule_problem,
            rendering=rendering)
        malfunction_env_reset()
        delta_reschedule_solution: ASPSolutionDescription = delta_reschedule_result.solution

        if verbose:
            print(f"  **** delta re-schedule solution")
            print(delta_reschedule_solution.get_trainruns_dict())

        # TODO SIM-146 ASP performance analyse running times (grounding vs solving - etc.)

        # --------------------------------------------------------------------------------------
        # Result
        # --------------------------------------------------------------------------------------
        current_results = ExperimentResults(time_full=schedule_result.solve_time,
                                            time_full_after_malfunction=full_reschedule_result.solve_time,
                                            time_delta_after_malfunction=delta_reschedule_result.solve_time,
                                            solution_full=schedule_solution.get_trainruns_dict(),
                                            solution_full_after_malfunction=full_reschedule_solution.get_trainruns_dict(),
                                            solution_delta_after_malfunction=delta_reschedule_solution.get_trainruns_dict(),
                                            costs_full=schedule_result.optimization_costs,
                                            costs_full_after_malfunction=full_reschedule_result.optimization_costs,
                                            costs_delta_after_malfunction=delta_reschedule_result.optimization_costs,
                                            experiment_freeze=delta_reschedule_result.experiment_freeze,
                                            malfunction=malfunction,
                                            agent_paths_dict=schedule_problem.agents_path_dict
                                            )
        return current_results
