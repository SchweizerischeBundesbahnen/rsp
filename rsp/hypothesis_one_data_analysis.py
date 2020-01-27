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
from rsp.utils.analysis_tools import average_over_trials
from rsp.utils.analysis_tools import three_dimensional_scatter_plot
from rsp.utils.experiments import load_experiment_agenda_from_file
from rsp.utils.experiments import load_experiment_results_from_folder

if __name__ == '__main__':
    # Import the desired experiment results
    data_folder = './exp_hypothesis_one_2019_12_19T09_58_07'
    data_folder = './results/early_alpha'
    data_file = './results/early_alpha/experiment_1.json'
    experiment_data = load_experiment_results_from_folder(data_folder)
    experiment_agenda = load_experiment_agenda_from_file(data_folder)

    for key in ['size', 'n_agents', 'max_num_cities', 'max_rail_between_cities', 'max_rail_in_city']:
        experiment_data[key] = experiment_data[key].astype(float)

    # Average over the trials of each experiment
    averaged_data, std_data = average_over_trials(experiment_data)
    print(experiment_data.keys())

    # Initially plot the computation time vs the level size and the number of agent
    three_dimensional_scatter_plot(data=averaged_data, error=std_data, columns=['n_agents', 'size', 'time_full'])

    three_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                   columns=['n_agents', 'size', 'time_full_after_malfunction'])
    three_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                   columns=['n_agents', 'size', 'time_delta_after_malfunction'])
