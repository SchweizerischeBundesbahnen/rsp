import argparse

from rsp.experiment_solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.experiment_solvers.asp.asp_solve_problem import solve_problem
from rsp.utils.experiments import load_and_expand_experiment_results_from_data_folder


def main(experiment_data_folder_name: str, experiment_id: int, problem: str, debug: bool):
    """

    Parameters
    ----------
    experiment_data_folder_name
    experiment_id
    problem
    debug
    """
    # We filter on a single experiment_id, so there should be only one one element in the list.
    experiment_results = load_and_expand_experiment_results_from_data_folder(
        experiment_data_folder_name=experiment_data_folder_name,
        experiment_ids=[experiment_id]
    )[0]

    tc = experiment_results._asdict()[f"problem_{problem}"]
    print(problem)
    if problem == "full":
        schedule_problem: ASPProblemDescription = ASPProblemDescription.factory_scheduling(
            tc=tc,
            asp_seed_value=experiment_results.experiment_parameters.asp_seed_value
        )
        schedule_result, asp_solution = solve_problem(
            problem=schedule_problem,
            debug=debug
        )
    elif problem in ["full_after_malfunction", "delta_after_malfunction"]:
        reschedule_problem: ASPProblemDescription = ASPProblemDescription.factory_rescheduling(
            tc=tc,
            asp_seed_value=experiment_results.experiment_parameters.asp_seed_value
        )
        reschedule_result, asp_solution = solve_problem(
            problem=reschedule_problem,
            debug=debug
        )
    else:
        raise ValueError(f"problem={args.which} unknown")


if __name__ == '__main__':
    # sample call:
    # python rsp/asp_plausibility/potassco_solution_checker \
    #   --experiment_data_folder_name=res/mini_toy_example/data --experiment_id=0 --problem=full_after_malfunction
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_data_folder_name', type=str, nargs=1, help='./res/mini_toy_example/data')
    parser.add_argument('--experiment_id', type=int, nargs=1, help='0,1,2,3...')
    parser.add_argument('--problem', type=str,
                        choices=['full_after_malfunction', 'full', 'delta_after_malfunction'],
                        help='which problem to check',
                        nargs=1)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    main(experiment_data_folder_name=args.experiment_data_folder_name[0], experiment_id=args.experiment_id[0], problem=args.problem[0], debug=args.debug)
