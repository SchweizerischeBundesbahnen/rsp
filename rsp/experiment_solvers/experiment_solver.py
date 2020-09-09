import pprint
from typing import Optional

from rsp.experiment_solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.experiment_solvers.asp.asp_solve_problem import solve_problem
from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.utils.data_types import experiment_freeze_dict_pretty_print
from rsp.utils.rsp_logger import rsp_logger

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
        schedule_problem_description=schedule_problem_description,
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
        schedule_problem_description=reschedule_problem_description,
        asp_seed_value=asp_seed_value
    )

    if debug:
        print("###reschedule")
        experiment_freeze_dict_pretty_print(reschedule_problem_description.route_dag_constraints_dict)

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
