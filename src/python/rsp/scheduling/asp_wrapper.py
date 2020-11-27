import pprint
from typing import Optional

from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.scheduling.asp.asp_problem_description import ASPProblemDescription
from rsp.scheduling.asp.asp_solve_problem import solve_problem
from rsp.scheduling.schedule import SchedulingExperimentResult
from rsp.scheduling.scheduling_problem import experiment_freeze_dict_pretty_print
from rsp.scheduling.scheduling_problem import get_sinks_for_topo
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.utils.rsp_logger import rsp_logger

_pp = pprint.PrettyPrinter(indent=4)


def asp_schedule_wrapper(
    schedule_problem_description: ScheduleProblemDescription, asp_seed_value: Optional[int] = None, debug: bool = False, no_optimize: bool = False
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
        schedule_problem_description=schedule_problem_description, asp_seed_value=asp_seed_value, no_optimize=no_optimize
    )

    schedule_result, _ = solve_problem(problem=schedule_problem, debug=debug)

    return schedule_result


def asp_reschedule_wrapper(
    reschedule_problem_description: ScheduleProblemDescription, schedule: TrainrunDict, asp_seed_value: Optional[int] = None, debug: bool = False,
) -> SchedulingExperimentResult:
    """Solve the Full Re-Scheduling Problem for static rail env (i.e. without
    malfunctions).

    Returns
    -------
    SchedulingExperimentResult
    """
    rsp_logger.info("reschedule_wrapper")

    additional_costs_at_targets = {
        agent_id: {
            sink: reschedule_problem_description.route_dag_constraints_dict[agent_id].earliest[sink] - schedule[agent_id][-1].scheduled_at
            for sink in get_sinks_for_topo(topo)
        }
        for agent_id, topo in reschedule_problem_description.topo_dict.items()
    }
    reschedule_problem: ASPProblemDescription = ASPProblemDescription.factory_rescheduling(
        schedule_problem_description=reschedule_problem_description, additional_costs_at_targets=additional_costs_at_targets, asp_seed_value=asp_seed_value
    )

    if debug:
        print("###reschedule")
        experiment_freeze_dict_pretty_print(reschedule_problem_description.route_dag_constraints_dict)

    reschedule_result, asp_solution = solve_problem(problem=reschedule_problem, debug=debug)
    if debug:
        print("###lates")
        print(asp_solution.extract_list_of_lates())
        print("###route penalties")
        print(asp_solution.extract_list_of_active_penalty())
        print("###reschedule")
        print(_pp.pformat(reschedule_result.trainruns_dict))

    return reschedule_result
