"""Solve an `asp_problem_description` problem a."""
import pprint
from typing import Tuple

import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.experiment_solvers.asp.asp_helper import configuration_as_dict_from_control
from rsp.experiment_solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.experiment_solvers.asp.asp_solution_description import ASPSolutionDescription
from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.route_dag.route_dag import get_paths_in_route_dag
from rsp.utils.general_utils import current_milli_time

_pp = pprint.PrettyPrinter(indent=4)


def solve_problem(
        problem: ASPProblemDescription,
        debug: bool = False,
) -> Tuple[SchedulingExperimentResult, ASPSolutionDescription]:
    """Solves an :class:`AbstractProblemDescription` and optionally verifies it
    againts the provided :class:`RailEnv`.

    Parameters
    ----------
    problem
    env
        The env to run the verification with
    rendering_call_back
        Called every step in replay
    debug
        Display debugging information
    loop_index
        Used for display, should identify the problem instance
    expected_malfunction
        Used in verification if provided

    Returns
    -------
    SchedulingExperimentResult
    """
    # --------------------------------------------------------------------------------------
    # Preparations
    # --------------------------------------------------------------------------------------
    minimum_number_of_shortest_paths_over_all_agents = np.min(
        [len(get_paths_in_route_dag(topo)) for agent_id, topo in problem.tc.topo_dict.items()])

    if minimum_number_of_shortest_paths_over_all_agents == 0:
        raise Exception("At least one Agent has no path to its target!")

    # --------------------------------------------------------------------------------------
    # Solve the problem
    # --------------------------------------------------------------------------------------
    start_build_problem = current_milli_time()
    build_problem_time = (current_milli_time() - start_build_problem) / 1000.0

    start_solver = current_milli_time()
    solution: ASPSolutionDescription = problem.solve()
    solve_time = (current_milli_time() - start_solver) / 1000.0
    assert solution.is_solved()

    trainruns_dict: TrainrunDict = solution.get_trainruns_dict()

    if debug:
        print("####train runs dict")
        print(_pp.pformat(trainruns_dict))
    return SchedulingExperimentResult(
        total_reward=-np.inf,
        solve_time=solve_time,
        optimization_costs=solution.get_objective_value(),
        build_problem_time=build_problem_time,
        nb_conflicts=solution.extract_nb_resource_conflicts(),
        trainruns_dict=solution.get_trainruns_dict(),
        route_dag_constraints=problem.tc.route_dag_constraints_dict,
        solver_statistics=solution.asp_solution.stats,
        solver_result=solution.answer_set,
        solver_configuration=configuration_as_dict_from_control(solution.asp_solution.ctl),
        solver_seed=solution.asp_solution.asp_seed_value
    ), solution
