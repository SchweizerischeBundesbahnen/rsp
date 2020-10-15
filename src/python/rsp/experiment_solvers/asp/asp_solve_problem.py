"""Solve an `asp_problem_description` problem a."""
import pprint
from typing import Tuple

import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from rsp.experiment_solvers.asp.asp_helper import configuration_as_dict_from_control
from rsp.experiment_solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.experiment_solvers.asp.asp_solution_description import ASPSolutionDescription
from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.schedule_problem_description.data_types_and_utils import get_paths_in_route_dag

_pp = pprint.PrettyPrinter(indent=4)


def solve_problem(problem: ASPProblemDescription, debug: bool = False, verbose: bool = False) -> Tuple[SchedulingExperimentResult, ASPSolutionDescription]:
    """Solves an :class:`AbstractProblemDescription` and optionally verifies it
    againts the provided :class:`RailEnv`.

    Parameters
    ----------
    problem
        Called every step in replay
    debug
        Display debugging information
    verbose

    Returns
    -------
    SchedulingExperimentResult
    """
    # --------------------------------------------------------------------------------------
    # Preparations
    # --------------------------------------------------------------------------------------
    minimum_number_of_shortest_paths_over_all_agents = np.min(
        [len(get_paths_in_route_dag(topo)) for agent_id, topo in problem.schedule_problem_description.topo_dict.items()]
    )

    if minimum_number_of_shortest_paths_over_all_agents == 0:
        raise Exception("At least one Agent has no path to its target!")

    # --------------------------------------------------------------------------------------
    # Solve the problem
    # --------------------------------------------------------------------------------------
    solution: ASPSolutionDescription = problem.solve(verbose=verbose)
    assert solution.is_solved()

    solution.verify_correctness()
    trainruns_dict: TrainrunDict = solution.get_trainruns_dict()

    if debug:
        print("####train runs dict")
        print(_pp.pformat(trainruns_dict))
    return (
        SchedulingExperimentResult(
            total_reward=-np.inf,
            solve_time=solution.get_solve_time(),
            optimization_costs=solution.get_objective_value(),
            build_problem_time=solution.get_preprocessing_time(),
            nb_conflicts=solution.extract_nb_resource_conflicts(),
            trainruns_dict=solution.get_trainruns_dict(),
            route_dag_constraints=problem.schedule_problem_description.route_dag_constraints_dict,
            solver_statistics=solution.asp_solution.stats,
            solver_result=solution.answer_set,
            solver_configuration=configuration_as_dict_from_control(solution.asp_solution.ctl),
            solver_seed=solution.asp_solution.asp_seed_value,
            solver_program=problem.asp_program,
        ),
        solution,
    )
