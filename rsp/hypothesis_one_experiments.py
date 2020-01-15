from rsp.asp.asp_experiment_solver import ASPExperimentSolver
from rsp.utils.data_types import ParameterRanges
from rsp.utils.experiments import create_experiment_agenda, run_specific_experiments_from_research_agenda, \
    save_experiment_results_to_file, run_experiment_agenda
from rsp.utils.experiments import create_experiment_agenda
from rsp.utils.experiments import run_experiment_agenda

if __name__ == '__main__':
    # Define the parameter ranges we would like to test
    parameter_ranges = ParameterRanges(agent_range=[5, 20, 10],
                                       size_range=[30, 30, 1],
                                       in_city_rail_range=[3, 3, 1],
                                       out_city_rail_range=[2, 2, 1],
                                       city_range=[20, 20, 1],
                                       earliest_malfunction=[20, 20, 1],
                                       malfunction_duration=[20, 20, 1])

    # Define the desired speed profiles
    speed_data = {1.: 1.,  # Fast passenger train
                  1. / 2.: 0.,  # Fast freight train
                  1. / 3.: 0.,  # Slow commuter train
                  1. / 4.: 0.}  # Slow freight train

    # Create an experiment agenda out of the parameter ranges
    experiment_agenda = create_experiment_agenda(experiment_name="exp_hypothesis_one",
                                                 speed_data=speed_data,
                                                 parameter_ranges=parameter_ranges,
                                                 trials_per_experiment=10)

    # Import the solver for the experiments
    solver = ASPExperimentSolver()

    # Run experiments
    run_experiment_agenda(solver=solver,
                          experiment_agenda=experiment_agenda,
                          run_experiments_parallel=False,
                          show_results_without_details=False,
                          verbose=False)
