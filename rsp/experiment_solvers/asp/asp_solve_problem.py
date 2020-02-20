"""Solve an `asp_problem_description` problem a."""
import pprint
from typing import Optional
from typing import Tuple

import numpy as np
from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.experiment_solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.experiment_solvers.asp.asp_solution_description import ASPSolutionDescription
from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.experiment_solvers.experiment_solver_utils import create_action_plan
from rsp.experiment_solvers.experiment_solver_utils import replay
from rsp.experiment_solvers.experiment_solver_utils import verify_trainruns_dict
from rsp.route_dag.route_dag import get_paths_in_route_dag
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.general_utils import current_milli_time

_pp = pprint.PrettyPrinter(indent=4)


def solve_problem(env: RailEnv,
                  problem: ASPProblemDescription,
                  rendering: bool = False,
                  debug: bool = False,
                  loop_index: int = 0,
                  disable_verification_in_replay: bool = False,
                  expected_malfunction: Optional[ExperimentMalfunction] = None
                  ) -> Tuple[SchedulingExperimentResult, ASPSolutionDescription]:
    """Solves an :class:`AbstractProblemDescription` and optionally verifies it
    againts the provided :class:`RailEnv`.

    Parameters
    ----------
    problem
    disable_verification_in_replay
        Whether it is tested the replay corresponds to the problem's solution
        TODO SIM-105 Should there be option to disable replay completely? Profile experiments to test how much time replay takes in the experiments.
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

    # --------------------------------------------------------------------------------------
    # Replay and verifiy the solution
    # --------------------------------------------------------------------------------------
    verify_trainruns_dict(env=env,
                          trainruns_dict=trainruns_dict,
                          expected_malfunction=expected_malfunction,
                          expected_route_dag_constraints=problem.tc.route_dag_constraints_dict
                          )
    controller_from_train_runs: ControllerFromTrainruns = create_action_plan(train_runs_dict=trainruns_dict, env=env)
    if debug:
        print("  **** solution to replay:")
        print(_pp.pformat(solution.get_trainruns_dict()))
        print("  **** action plan to replay:")
        controller_from_train_runs.print_action_plan()
        print("  **** expected_malfunction to replay:")
        print(_pp.pformat(expected_malfunction))

    total_reward = replay(env=env,
                          loop_index=loop_index,
                          expected_malfunction=expected_malfunction,
                          solver_name=problem.get_solver_name(),
                          rendering=rendering,
                          controller_from_train_runs=controller_from_train_runs,
                          debug=debug,
                          disable_verification_in_replay=disable_verification_in_replay)

    return SchedulingExperimentResult(total_reward=total_reward,
                                      solve_time=solve_time,
                                      optimization_costs=solution.get_objective_value(),
                                      build_problem_time=build_problem_time,
                                      nb_conflicts=solution.extract_nb_resource_conflicts(),
                                      trainruns_dict=solution.get_trainruns_dict(),
                                      route_dag_constraints=problem.tc.route_dag_constraints_dict), solution
