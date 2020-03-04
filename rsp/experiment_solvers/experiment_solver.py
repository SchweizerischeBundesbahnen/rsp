import pprint
from typing import Callable

from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.experiment_solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.experiment_solvers.asp.asp_solve_problem import replay
from rsp.experiment_solvers.asp.asp_solve_problem import replay_and_verify_asp_solution
from rsp.experiment_solvers.asp.asp_solve_problem import solve_problem
from rsp.experiment_solvers.data_types import ScheduleAndMalfunction
from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.experiment_solvers.experiment_solver_utils import create_action_plan
from rsp.route_dag.generators.route_dag_generator_reschedule_full import get_schedule_problem_for_full_rescheduling
from rsp.route_dag.generators.route_dag_generator_reschedule_perfect_oracle import perfect_oracle
from rsp.route_dag.generators.route_dag_generator_schedule import schedule_problem_description_from_rail_env
from rsp.route_dag.route_dag import apply_weight_route_change
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.utils.data_types import experimentFreezeDictPrettyPrint
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ExperimentResults


class ASPExperimentSolver():
    """Implements `ASPExperimentSolver` for ASP.

    Methods
    -------
    run_experiment_trial:
        Returns the correct data format to run tests on full research pipeline
    """
    _pp = pprint.PrettyPrinter(indent=4)

    def gen_schedule_and_malfunction(self,
                                     static_rail_env: RailEnv,
                                     malfunction_rail_env: RailEnv,
                                     malfunction_env_reset,
                                     experiment_parameters: ExperimentParameters,
                                     disable_verification_by_replay: bool = False,
                                     verbose: bool = False,
                                     debug: bool = False,
                                     rendering: bool = False
                                     ) -> ScheduleAndMalfunction:
        """B1.

        Experiment Setup
        """
        tc_schedule_problem = schedule_problem_description_from_rail_env(
            env=static_rail_env,
            k=experiment_parameters.number_of_shortest_paths_per_agent
        )
        schedule_result = asp_schedule_wrapper(tc_schedule_problem,
                                               rendering=rendering,
                                               static_rail_env=static_rail_env,
                                               debug=debug)

        schedule_trainruns: TrainrunDict = schedule_result.trainruns_dict

        if verbose:
            print(f"  **** schedule_solution={schedule_trainruns}")

        # --------------------------------------------------------------------------------------
        # 1. Generate malfuntion
        # --------------------------------------------------------------------------------------
        malfunction_env_reset()
        controller_from_train_runs: ControllerFromTrainruns = create_action_plan(
            train_runs_dict=schedule_trainruns,
            env=malfunction_rail_env)
        malfunction_env_reset()
        malfunction = replay(
            controller_from_train_runs=controller_from_train_runs,
            env=malfunction_rail_env,
            stop_on_malfunction=True,
            solver_name="ASP",
            disable_verification_in_replay=True)
        malfunction_env_reset()
        # replay may return None (if the given malfunction does not happen during the agents time in the grid
        if malfunction is None:
            raise Exception("Could not produce a malfunction")

        if verbose:
            print(f"  **** malfunction={malfunction}")
        return ScheduleAndMalfunction(tc_schedule_problem, schedule_result, malfunction)

    def run_experiment_trial(
            self,
            schedule_and_malfunction: ScheduleAndMalfunction,
            malfunction_rail_env: RailEnv,
            malfunction_env_reset,
            experiment_parameters: ExperimentParameters,
            verbose: bool = False,
            debug: bool = False,
            rendering: bool = False
    ) -> ExperimentResults:
        """B2. Runs the experiment.

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
        tc_schedule_problem, schedule_result, malfunction = schedule_and_malfunction

        schedule_trainruns: TrainrunDict = schedule_result.trainruns_dict

        # --------------------------------------------------------------------------------------
        # 2. Re-schedule Full
        # --------------------------------------------------------------------------------------
        full_reschedule_problem: ScheduleProblemDescription = get_schedule_problem_for_full_rescheduling(
            malfunction=malfunction,
            schedule_trainruns=schedule_trainruns,
            minimum_travel_time_dict=tc_schedule_problem.minimum_travel_time_dict,
            latest_arrival=malfunction_rail_env._max_episode_steps,
            max_window_size_from_earliest=experiment_parameters.max_window_size_from_earliest,
            topo_dict=tc_schedule_problem.topo_dict
        )
        full_reschedule_problem = apply_weight_route_change(
            schedule_problem=full_reschedule_problem,
            weight_route_change=experiment_parameters.weight_route_change,
            weight_lateness_seconds=experiment_parameters.weight_lateness_seconds
        )
        full_reschedule_result = asp_reschedule_wrapper(
            malfunction_for_verification=malfunction,
            malfunction_env_reset=malfunction_env_reset,
            malfunction_rail_env_for_verification=malfunction_rail_env,
            reschedule_problem_description=full_reschedule_problem,
            rendering=rendering,
            debug=debug
        )
        malfunction_env_reset()

        full_reschedule_trainruns = full_reschedule_result.trainruns_dict

        if verbose:
            print(f"  **** full re-schedule_solution=\n{full_reschedule_trainruns}")

        # --------------------------------------------------------------------------------------
        # 3. Re-Schedule Delta
        # --------------------------------------------------------------------------------------
        delta_reschedule_problem = perfect_oracle(
            full_reschedule_trainrun_waypoints_dict=full_reschedule_trainruns,
            malfunction=malfunction,
            max_episode_steps=tc_schedule_problem.max_episode_steps,
            schedule_topo_dict=tc_schedule_problem.topo_dict,
            schedule_trainrun_dict=schedule_trainruns,
            minimum_travel_time_dict=tc_schedule_problem.minimum_travel_time_dict,
            max_window_size_from_earliest=experiment_parameters.max_window_size_from_earliest
        )
        delta_reschedule_problem = apply_weight_route_change(
            schedule_problem=delta_reschedule_problem,
            weight_route_change=experiment_parameters.weight_route_change,
            weight_lateness_seconds=experiment_parameters.weight_lateness_seconds
        )
        delta_reschedule_result = asp_reschedule_wrapper(
            malfunction_for_verification=malfunction,
            malfunction_rail_env_for_verification=malfunction_rail_env,
            reschedule_problem_description=delta_reschedule_problem,
            rendering=rendering,
            debug=debug,
            malfunction_env_reset=lambda *args, **kwargs: None
        )
        malfunction_env_reset()

        if verbose:
            print(f"  **** delta re-schedule solution")
            print(delta_reschedule_result.trainruns_dict)

        # --------------------------------------------------------------------------------------
        # 4. Result
        # --------------------------------------------------------------------------------------
        current_results = ExperimentResults(
            experiment_parameters=experiment_parameters,
            malfunction=malfunction,
            problem_full=tc_schedule_problem,
            problem_full_after_malfunction=full_reschedule_problem,
            problem_delta_after_malfunction=delta_reschedule_problem,
            results_full=schedule_result,
            results_full_after_malfunction=full_reschedule_result,
            results_delta_after_malfunction=delta_reschedule_result
        )
        return current_results


_pp = pprint.PrettyPrinter(indent=4)


def asp_schedule_wrapper(schedule_problem_description: ScheduleProblemDescription,
                         static_rail_env: RailEnv,
                         rendering: bool = False,
                         debug: bool = False,
                         ) -> SchedulingExperimentResult:
    """Solves the Full Scheduling Problem for static rail env (i.e. without
    malfunctions).

    Parameters
    ----------
    k:int
        number of routing alterantives to consider
    static_rail_env: RailEnv
    rendering: bool
    debug: bool

    Returns
    -------
    SchedulingExperimentResult
        the problem description and the results
    """

    # --------------------------------------------------------------------------------------
    # Produce a full schedule
    # --------------------------------------------------------------------------------------
    schedule_problem = ASPProblemDescription.factory_scheduling(
        tc=schedule_problem_description)

    schedule_result, schedule_solution = solve_problem(
        problem=schedule_problem,
        debug=debug)
    replay_and_verify_asp_solution(env=static_rail_env,
                                   problem_description=schedule_problem_description,
                                   asp_solution=schedule_solution,
                                   rendering=rendering,
                                   debug=debug)

    return schedule_result


def asp_reschedule_wrapper(
        reschedule_problem_description: ScheduleProblemDescription,
        malfunction_for_verification: ExperimentMalfunction,
        malfunction_rail_env_for_verification: RailEnv,
        malfunction_env_reset: Callable[[], None],
        debug: bool = False,
        rendering: bool = False
) -> SchedulingExperimentResult:
    """Solve the Full Re-Scheduling Problem for static rail env (i.e. without
    malfunctions).

    Returns
    -------
    SchedulingExperimentResult
    """

    # --------------------------------------------------------------------------------------
    # Full Re-Scheduling
    # --------------------------------------------------------------------------------------
    full_reschedule_problem: ASPProblemDescription = ASPProblemDescription.factory_rescheduling(
        tc=reschedule_problem_description
    )

    if debug:
        print("###reschedule")
        experimentFreezeDictPrettyPrint(reschedule_problem_description.route_dag_constraints_dict)

    full_reschedule_result, asp_solution = solve_problem(
        problem=full_reschedule_problem,
        debug=debug
    )
    if debug:
        print("###lates")
        print(asp_solution.extract_list_of_lates())
        print("###route penalties")
        print(asp_solution.extract_list_of_active_penalty())
        print("###reschedule")
        print(_pp.pformat(full_reschedule_result.trainruns_dict))

    replay_and_verify_asp_solution(env=malfunction_rail_env_for_verification,
                                   problem_description=reschedule_problem_description,
                                   asp_solution=asp_solution,
                                   rendering=rendering,
                                   debug=debug,
                                   expected_malfunction=malfunction_for_verification,
                                   # SIM-155 decision: we do not replay against FLATland any more but check the solution on the Trainrun data structure
                                   disable_verification_in_replay=True)
    malfunction_env_reset()

    return full_reschedule_result
