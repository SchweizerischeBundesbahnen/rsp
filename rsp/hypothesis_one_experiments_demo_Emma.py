from rsp.asp.asp_experiment_solver import ASPExperimentSolver
from rsp.utils.data_types import ParameterRanges, ExperimentAgenda, ExperimentParameters
from rsp.utils.experiments import run_experiment_agenda, save_experiment_results_to_file

if __name__ == '__main__':
    # Define the parameter ranges we would like to test
    parameter_ranges = ParameterRanges(agent_range=[2, 50, 30],
                                       size_range=[30, 50, 10],
                                       in_city_rail_range=[6, 6, 1],
                                       out_city_rail_range=[2, 2, 1],
                                       city_range=[20, 20, 1],
                                       earliest_malfunction=[20, 20, 1],
                                       malfunction_duration=[20, 20, 1])

    experiment_agenda = ExperimentAgenda(experiments=[
        ExperimentParameters(experiment_id=0,
                             trials_in_experiment=1,
                             number_of_agents=50,
                             width=35,
                             height=35,
                             seed_value=12,
                             max_num_cities=2,
                             grid_mode=True,
                             max_rail_between_cities=2,
                             max_rail_in_city=4,
                             earliest_malfunction=20,
                             malfunction_duration=20)
    ])
    # Import the solver for the experiments
    solver = ASPExperimentSolver()

    # Run experiments
    experiment_results = run_experiment_agenda(solver, experiment_agenda, verbose=True)
