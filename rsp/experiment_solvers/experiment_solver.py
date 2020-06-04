import pprint
from typing import Optional
from typing import Tuple

from flatland.envs.rail_env import RailEnv

from rsp.experiment_solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.experiment_solvers.asp.asp_solve_problem import solve_problem
from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.logger import rsp_logger
from rsp.route_dag.generators.route_dag_generator_schedule import schedule_problem_description_from_rail_env
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.utils.data_types import experimentFreezeDictPrettyPrint
from rsp.utils.data_types import ExperimentParameters


class ASPExperimentSolver():
    """Implements `ASPExperimentSolver` for ASP."""
    _pp = pprint.PrettyPrinter(indent=4)

    def gen_schedule(self,
                     static_rail_env: RailEnv,
                     experiment_parameters: ExperimentParameters,
                     verbose: bool = False,
                     debug: bool = False,
                     ) -> Tuple[ScheduleProblemDescription, SchedulingExperimentResult]:
        """A.2.2.

        Create Schedule.
        """
        rsp_logger.info("gen_schedule_and_malfunction")
        tc_schedule_problem = schedule_problem_description_from_rail_env(
            env=static_rail_env,
            k=experiment_parameters.number_of_shortest_paths_per_agent
        )
        schedule_result = asp_schedule_wrapper(tc_schedule_problem,
                                               asp_seed_value=experiment_parameters.asp_seed_value,
                                               debug=debug)

        if verbose:
            print(f"  **** schedule_solution={schedule_result.trainruns_dict}")
        return tc_schedule_problem, schedule_result


_pp = pprint.PrettyPrinter(indent=4)


def asp_schedule_wrapper(schedule_problem_description: ScheduleProblemDescription,
                         asp_seed_value: Optional[int] = None,
                         debug: bool = False,
                         no_optimize: bool = False
                         ) -> SchedulingExperimentResult:
    """Solves the Full Scheduling Problem for static rail env (i.e. without
    malfunctions).

    Parameters
    ----------
    schedule_problem_description
    asp_seed_value
    no_optimize
    debug: bool

    Returns
    -------
    SchedulingExperimentResult
        the problem description and the results
    """
    rsp_logger.info("schedule_wrapper")
    # --------------------------------------------------------------------------------------
    # Produce a full schedule
    # --------------------------------------------------------------------------------------
    schedule_problem = ASPProblemDescription.factory_scheduling(
        tc=schedule_problem_description,
        asp_seed_value=asp_seed_value,
        no_optimize=no_optimize
    )

    schedule_result, schedule_solution = solve_problem(
        problem=schedule_problem,
        debug=debug)

    return schedule_result


def asp_reschedule_wrapper(
        reschedule_problem_description: ScheduleProblemDescription,
        asp_seed_value: Optional[int] = None,
        debug: bool = False,
) -> SchedulingExperimentResult:
    """Solve the Full Re-Scheduling Problem for static rail env (i.e. without
    malfunctions).

    Returns
    -------
    SchedulingExperimentResult
    """
    rsp_logger.info("reschedule_wrapper")
    # --------------------------------------------------------------------------------------
    # Full Re-Scheduling
    # --------------------------------------------------------------------------------------
    full_reschedule_problem: ASPProblemDescription = ASPProblemDescription.factory_rescheduling(
        tc=reschedule_problem_description,
        asp_seed_value=asp_seed_value
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

    return full_reschedule_result
