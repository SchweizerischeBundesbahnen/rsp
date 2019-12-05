from rsp.asp.asp_experiment_solver import ASPExperimentSolver
from rsp.utils.data_types import ParameterRanges
from rsp.utils.experiments import create_experiment_agenda, run_specific_experiments_from_research_agenda, \
    run_experiment_agenda, save_experiment_results_to_file

if __name__ == '__main__':
    # Define the parameter ranges we would like to test
    parameter_ranges = ParameterRanges(agent_range=[2, 50, 30],
                                       size_range=[30, 50, 10],
                                       in_city_rail_range=[6, 6, 1],
                                       out_city_rail_range=[2, 2, 1],
                                       city_range=[20, 20, 1],
                                       earliest_malfunction=[20, 20, 1],
                                       malfunction_duration=[20, 20, 1])

    # Create an experiment agenda out of the parameter ranges
    experiment_agenda = create_experiment_agenda(parameter_ranges, trials_per_experiment=10)

    # Import the solver for the experiments
    solver = ASPExperimentSolver()

    # Run experiments
    experiment_results = run_experiment_agenda(solver, experiment_agenda, verbose=True)

    # Save experiment results in a file
    save_experiment_results_to_file(experiment_results, "./experiment_data/test_setup.json")

    # Re-run desired experiments
    few_experiment_results = run_specific_experiments_from_research_agenda(solver, experiment_agenda, range(17, 18),
                                                                           verbose=True)
