import pprint
from typing import Callable
from typing import Tuple

import numpy as np
from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.route_dag.generators.route_dag_generator_reschedule_full import get_freeze_for_full_rescheduling
from rsp.route_dag.generators.route_dag_generator_reschedule_perfect_oracle import perfect_oracle
from rsp.route_dag.generators.route_dag_generator_schedule import schedule_problem_description_from_rail_env
from rsp.route_dag.generators.route_dag_generator_utils import verify_experiment_freeze_for_agent
from rsp.route_dag.route_dag import RouteDAG
from rsp.solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.solvers.asp.asp_solution_description import ASPSolutionDescription
from rsp.solvers.asp.asp_solve_problem import replay
from rsp.solvers.asp.asp_solve_problem import SchedulingExperimentResult
from rsp.solvers.asp.asp_solve_problem import solve_problem
from rsp.utils.data_types import experimentFreezeDictPrettyPrint
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.data_types import ExperimentResults
from rsp.utils.experiment_solver import AbstractSolver


# TODO SIM-239 sollte nichts mit ASP zu tun haben
class ASPExperimentSolver(AbstractSolver):
    """Implements `AbstractSolver` for ASP.

    Methods
    -------
    run_experiment_trial:
        Returns the correct data format to run tests on full research pipeline
    """
    _pp = pprint.PrettyPrinter(indent=4)

    # TODO SIM-239 we should implement the general flow of the pipeline independent of ASP and
    #  pass the ASP solver to the instance
    def run_experiment_trial(
            self,
            static_rail_env: RailEnv,
            malfunction_rail_env: RailEnv,
            malfunction_env_reset,
            k: int = 10,
            disable_verification_by_replay: bool = False,
            verbose: bool = False,
            debug: bool = False,
            rendering: bool = False
    ) -> ExperimentResults:
        """Runs the experiment.

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
        rendering = True
        # TODO SIM-239 pass experiment_freeze into this
        schedule_problem, schedule_result, schedule_solution = schedule_full(k, static_rail_env, rendering=rendering,
                                                                             debug=debug)

        schedule_trainruns: TrainrunDict = schedule_result.trainruns_dict

        if verbose:
            print(f"  **** schedule_solution={schedule_trainruns}")

        # --------------------------------------------------------------------------------------
        # Generate malfuntion
        # --------------------------------------------------------------------------------------

        malfunction_env_reset()
        controller_from_train_runs: ControllerFromTrainruns = schedule_solution.create_action_plan(
            env=malfunction_rail_env)
        malfunction_env_reset()
        malfunction = replay(
            controller_from_train_runs=controller_from_train_runs,
            env=malfunction_rail_env,
            stop_on_malfunction=True,
            solver_name=schedule_problem.get_solver_name(),
            disable_verification_in_replay=True)
        if malfunction is None:
            return None
        malfunction_env_reset()
        # replay may return None (if the given malfunction does not happen during the agents time in the grid
        if malfunction is None:
            raise Exception("Could not produce a malfunction")

        if verbose:
            print(f"  **** malfunction={malfunction}")

        # --------------------------------------------------------------------------------------
        # Re-schedule Full
        # --------------------------------------------------------------------------------------

        # TODO SIM-239 pass experiment_freeze into this
        _, full_reschedule_result, _ = reschedule_full_after_malfunction(
            malfunction=malfunction,
            malfunction_env_reset=malfunction_env_reset,
            malfunction_rail_env=malfunction_rail_env,
            schedule_problem=schedule_problem,
            schedule_trainruns=schedule_trainruns,
            rendering=rendering,
            debug=debug
        )
        malfunction_env_reset()

        full_reschedule_trainruns = full_reschedule_result.trainruns_dict

        if verbose:
            print(f"  **** full re-schedule_solution=\n{full_reschedule_trainruns}")

        # --------------------------------------------------------------------------------------
        # Re-Schedule Delta
        # --------------------------------------------------------------------------------------
        # TODO SIM-239 pass experiment_freeze into this
        _, delta_reschedule_result, _ = reschedule_delta_after_malfunction(
            full_reschedule_trainruns=full_reschedule_trainruns,
            schedule_trainruns=schedule_trainruns,
            malfunction=malfunction,
            malfunction_rail_env=malfunction_rail_env,
            tc=schedule_problem.tc,
            rendering=rendering,
            debug=debug
        )
        malfunction_env_reset()

        if verbose:
            print(f"  **** delta re-schedule solution")
            print(delta_reschedule_result.trainruns_dict)

        # --------------------------------------------------------------------------------------
        # Result
        # --------------------------------------------------------------------------------------
        current_results = ExperimentResults(time_full=schedule_result.solve_time,
                                            time_full_after_malfunction=full_reschedule_result.solve_time,
                                            time_delta_after_malfunction=delta_reschedule_result.solve_time,
                                            solution_full=schedule_result.trainruns_dict,
                                            solution_full_after_malfunction=full_reschedule_result.trainruns_dict,
                                            solution_delta_after_malfunction=delta_reschedule_result.trainruns_dict,
                                            costs_full=schedule_result.optimization_costs,
                                            costs_full_after_malfunction=full_reschedule_result.optimization_costs,
                                            costs_delta_after_malfunction=delta_reschedule_result.optimization_costs,
                                            # TODO SIM-239 currently None. We do not pass experiment_freeze into solver yet,
                                            #  we do not have it for scheduling!
                                            experiment_freeze_full=schedule_result.experiment_freeze,
                                            experiment_freeze_full_after_malfunction=full_reschedule_result.experiment_freeze,
                                            experiment_freeze_delta_after_malfunction=delta_reschedule_result.experiment_freeze,
                                            malfunction=malfunction,
                                            topo_dict=schedule_problem.tc.topo_dict,
                                            nb_resource_conflicts_full=schedule_result.nb_conflicts,
                                            nb_resource_conflicts_full_after_malfunction=full_reschedule_result.nb_conflicts,
                                            nb_resource_conflicts_delta_after_malfunction=delta_reschedule_result.nb_conflicts
                                            )
        return current_results


_pp = pprint.PrettyPrinter(indent=4)


# TODO SIM-239 we should pass ExperimentFreeze as input
def schedule_full(k: int,
                  static_rail_env: RailEnv,
                  rendering: bool = False,
                  debug: bool = False,
                  ) -> Tuple[ASPProblemDescription, SchedulingExperimentResult, ASPSolutionDescription]:
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
    Tuple[ASPProblemDescription, SchedulingExperimentResult]
        the problem description and the results
    """

    # --------------------------------------------------------------------------------------
    # Produce a full schedule
    # --------------------------------------------------------------------------------------
    schedule_problem = ASPProblemDescription.factory_scheduling(
        tc=schedule_problem_description_from_rail_env(static_rail_env, k))

    schedule_result, schedule_solution = solve_problem(
        env=static_rail_env,
        problem=schedule_problem,
        rendering=rendering,
        debug=debug)

    return schedule_problem, schedule_result, schedule_solution


# TODO SIM-239 we should pass ExperimentFreeze as input and no FLATland env?
def reschedule_full_after_malfunction(
        schedule_problem: ASPProblemDescription,
        schedule_trainruns: TrainrunDict,
        malfunction: ExperimentMalfunction,
        malfunction_rail_env: RailEnv,
        malfunction_env_reset: Callable[[], None],
        debug: bool = False,
        rendering: bool = False,
) -> Tuple[ASPProblemDescription, SchedulingExperimentResult, ASPSolutionDescription]:
    """Solve the Full Re-Scheduling Problem for static rail env (i.e. without
    malfunctions).

    Parameters
    ----------
    schedule_problem
    schedule_trainruns
    static_rail_env
    malfunction
    malfunction_rail_env
    malfunction_env_reset
    debug
    rendering

    Returns
    -------
    SchedulingExperimentResult
    """

    # --------------------------------------------------------------------------------------
    # Full Re-Scheduling
    # --------------------------------------------------------------------------------------
    minimum_travel_time_dict = {agent.handle: int(np.ceil(1 / agent.speed_data['speed'])) for agent in
                                malfunction_rail_env.agents}
    tc: RouteDAG = get_freeze_for_full_rescheduling(
        malfunction=malfunction,
        schedule_trainruns=schedule_trainruns,
        minimum_travel_time_dict=minimum_travel_time_dict,
        latest_arrival=malfunction_rail_env._max_episode_steps,
        topo_dict=schedule_problem.tc.topo_dict
    )

    full_reschedule_problem: ASPProblemDescription = ASPProblemDescription.factory_rescheduling(
        tc=tc
    )

    if debug:
        print("###reschedule_full freeze_dict")
        experimentFreezeDictPrettyPrint(tc.experiment_freeze_dict)

    full_reschedule_result, full_reschedule_solution = solve_problem(
        env=malfunction_rail_env,
        problem=full_reschedule_problem,
        rendering=rendering,
        debug=debug,
        expected_malfunction=malfunction,
        # SIM-155 decision: we do not replay against FLATland any more but check the solution on the Trainrun data structure
        disable_verification_in_replay=True
    )
    malfunction_env_reset()

    if debug:
        print("###reschedule_full_after_malfunction")
        print(_pp.pformat(full_reschedule_result.solution.get_trainruns_dict()))

    return full_reschedule_problem, full_reschedule_result, full_reschedule_solution


# TODO SIM-239 we should pass ExperimentFreeze as input, remove rendering? sollte nur ScheduleProblemDescription zurückgeben
def reschedule_delta_after_malfunction(
        tc: RouteDAG,
        full_reschedule_trainruns: TrainrunDict,
        schedule_trainruns: TrainrunDict,
        malfunction: ExperimentMalfunction,
        malfunction_rail_env: RailEnv,
        rendering: bool = False,
        debug: bool = False,
) -> Tuple[ASPProblemDescription, SchedulingExperimentResult, ASPSolutionDescription]:
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

    # --------------------- -----------------------------------------------------------------
    # Delta Re-Scheduling with "perfect" oracle
    # --------------------------------------------------------------------------------------
    tc: RouteDAG = perfect_oracle(
        full_reschedule_trainrun_waypoints_dict=full_reschedule_trainruns,
        malfunction=malfunction,
        minimum_travel_time_dict=tc.minimum_travel_time_dict,
        max_episode_steps=tc.max_episode_steps,
        schedule_topo_dict=tc.topo_dict,
        schedule_trainrun_dict=schedule_trainruns
    )
    freeze_dict = tc.experiment_freeze_dict

    # verify that full reschedule solution is still a solution in the constrained solution space
    for agent_id in freeze_dict:
        verify_experiment_freeze_for_agent(
            agent_id=agent_id,
            topo=tc.topo_dict[agent_id],
            experiment_freeze=freeze_dict[agent_id],
            force_freeze=[],
            scheduled_trainrun=full_reschedule_trainruns[agent_id]
        )

    delta_reschedule_problem: ASPProblemDescription = ASPProblemDescription.factory_rescheduling(
        tc=tc
    )

    delta_reschedule_result, delta_reschedule_solution = solve_problem(
        env=malfunction_rail_env,
        problem=delta_reschedule_problem,
        rendering=rendering,
        debug=debug,
        expected_malfunction=malfunction,
        # TODO SIM-239 remove/extract?
        # SIM-155 decision: we do not replay against FLATland any more but check the solution on the Trainrun data structure
        disable_verification_in_replay=True
    )

    if debug:
        print("####delta train runs dict")
        print(_pp.pformat(delta_reschedule_result.solution.get_trainruns_dict()))

    return delta_reschedule_problem, delta_reschedule_result, delta_reschedule_solution
