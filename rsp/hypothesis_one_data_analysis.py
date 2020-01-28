"""Analysis of the experiment data for hypothesis one.

Hypothesis 1:
    We can compute good recourse actions, i.e., an adapted plan within the time budget,
    if all the variables are fixed, except those related to services that are affected by the
    disruptions implicitly or explicitly.

Hypothesis 2:
    Machine learning can predict services that are affected by disruptions implicitly or
    explicitly. Hypothesis 3: If hypothesis 2 is true, in addition, machine
    learning can predict the state of the system in the next time period
    after re-scheduling.
"""
from pandas import DataFrame

from rsp.rescheduling.rescheduling_analysis_utils import analyze_experiment
from rsp.solvers.solve_problem import render_experiment
from rsp.utils.analysis_tools import average_over_trials
from rsp.utils.analysis_tools import three_dimensional_scatter_plot
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.experiments import load_experiment_agenda_from_file
from rsp.utils.experiments import load_experiment_results_from_folder

if __name__ == '__main__':
    # Import the desired experiment results
    data_folder = './exp_hypothesis_one_2020_01_28T11_54_24'
    experiment_data: DataFrame = load_experiment_results_from_folder(data_folder)
    experiment_agenda: ExperimentAgenda = load_experiment_agenda_from_file(data_folder)

    for key in ['size', 'n_agents', 'max_num_cities', 'max_rail_between_cities', 'max_rail_in_city']:
        experiment_data[key] = experiment_data[key].astype(float)

    # Average over the trials of each experiment
    averaged_data, std_data = average_over_trials(experiment_data)

    # quantitative analysis
    # Initially plot the computation time vs the level size and the number of agent
    three_dimensional_scatter_plot(data=averaged_data, error=std_data, columns=['n_agents', 'size', 'time_full'])

    three_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                   columns=['n_agents', 'size', 'time_full_after_malfunction'])
    three_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                   columns=['n_agents', 'size', 'time_delta_after_malfunction'])

    # qualitative explorative analysis
    experiments_ids = [233]
    filtered_experiments = list(filter(lambda experiment: experiment.experiment_id in experiments_ids,
                                       experiment_agenda.experiments))
    for experiment in filtered_experiments:
        analyze_experiment(experiment=experiment, data_frame=experiment_data)
        render_experiment(experiment=experiment, data_frame=experiment_data)
