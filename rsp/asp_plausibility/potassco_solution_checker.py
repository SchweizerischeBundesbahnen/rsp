import argparse

from rsp.experiment_solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.experiment_solvers.asp.asp_solve_problem import solve_problem
from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.experiments import _get_asp_solver_details_from_statistics
from rsp.utils.experiments import create_experiment_filename
from rsp.utils.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.utils.experiments import save_experiment_results_to_file
from rsp.utils.rsp_logger import rsp_logger


def main(experiment_data_folder_name: str,
         experiment_id: int,
         problem_suffix: str,
         debug: bool,
         verbose: bool = False,
         save_output: bool = False):
    """

    Parameters
    ----------

    experiment_data_folder_name
    experiment_id
    save_output
    verbose
    problem_suffix
    debug
    """
    # We filter on a single experiment_id, so there should be only one one element in the list.
    experiment_results: ExperimentResultsAnalysis = load_and_expand_experiment_results_from_data_folder(
        experiment_data_folder_name=experiment_data_folder_name,
        experiment_ids=[experiment_id]
    )[0]

    problem: ScheduleProblemDescription = experiment_results._asdict()[f"problem_{problem_suffix}"]
    results: SchedulingExperimentResult = experiment_results._asdict()[f"results_{problem_suffix}"]

    if problem_suffix == "full":
        schedule_problem: ASPProblemDescription = ASPProblemDescription.factory_scheduling(
            schedule_problem_description=problem,
            asp_seed_value=experiment_results.experiment_parameters.asp_seed_value
        )
        schedule_result, asp_solution = solve_problem(
            problem=schedule_problem,
            verbose=verbose,
            debug=debug
        )
    elif problem_suffix in ["full_after_malfunction", "delta_perfect_after_malfunction"]:
        statistics = results.solver_statistics
        rsp_logger.info(f"Problem {problem_suffix} for experiment {experiment_results.experiment_id} from {experiment_data_folder_name} baseline was: "
                        f'{_get_asp_solver_details_from_statistics(elapsed_time=statistics["summary"]["times"]["total"], statistics=statistics)}')
        rsp_logger.info(f"Generating {problem_suffix} for experiment {experiment_results.experiment_id} from {experiment_data_folder_name}")
        reschedule_problem: ASPProblemDescription = ASPProblemDescription.factory_rescheduling(
            schedule_problem_description=problem,
            asp_seed_value=experiment_results.experiment_parameters.asp_seed_value
        )
        rsp_logger.info(f"Solving {problem_suffix} for experiment {experiment_results.experiment_id} from {experiment_data_folder_name}")
        solve_problem(
            problem=reschedule_problem,
            verbose=verbose,
            debug=debug
        )
        if save_output:
            filename = create_experiment_filename(experiment_data_folder_name=experiment_data_folder_name, experiment_id=experiment_id)
            save_experiment_results_to_file(experiment_results, filename)
        rsp_logger.info(f"Done {problem_suffix} for experiment {experiment_results.experiment_id} from {experiment_data_folder_name}")
    else:
        raise ValueError(f"problem={args.which} unknown")


if __name__ == '__main__':
    # sample call:
    # python rsp/asp_plausibility/potassco_solution_checker \
    #   --experiment_data_folder_name=../rsp-data/agent_0_malfunction_2020_05_27T19_45_49/data --experiment_id=0 --problem=full_after_malfunction
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_data_folder_name', type=str, nargs=1, help='../rsp-data/agent_0_malfunction_2020_05_27T19_45_49/data')
    parser.add_argument('--experiment_id', type=int, nargs=1, help='0,1,2,3...')
    parser.add_argument('--problem', type=str,
                        choices=['full_after_malfunction', 'full', 'delta_perfect_after_malfunction'],
                        help='which problem to check',
                        nargs=1)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--save_output', action="store_true")
    args = parser.parse_args()
    main(experiment_data_folder_name=args.experiment_data_folder_name[0],
         experiment_id=args.experiment_id[0],
         problem_suffix=args.problem[0],
         debug=args.debug,
         save_output=args.save_output,
         verbose=args.verbose)
