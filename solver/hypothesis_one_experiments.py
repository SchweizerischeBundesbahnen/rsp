from asp.asp_experiment_solver import ASPExperimentSolver
from solver.utils.data_types import ParameterRanges
from solver.utils.experiments import create_experiment_agenda, run_specific_experiments_from_research_agenda
from solver.utils.experiments import run_experiment_agenda, save_experiment_results_to_file

if __name__ == '__main__':
    # Define the parameter ranges we would like to test
    parameter_ranges = ParameterRanges(agent_range=[5, 150, 10],
                                       size_range=[30, 100, 10],
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
    experiment_results = run_experiment_agenda(solver, experiment_agenda)

    # Re-run desired experiments
    few_experiment_results = run_specific_experiments_from_research_agenda(solver, experiment_agenda, [1, 3])

    # Save experiment results in a file
    save_experiment_results_to_file(experiment_results, "./experiment_data/test_setup.json")
