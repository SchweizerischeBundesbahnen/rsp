"""
Analysis of the experiment data for hypothesis one.

Hypothesis 1: We can compute good recourse actions, i.e., an adapted plan within the time budget,
              if all the variables are fixed, except those related to services that are affected
              by the disruptions implicitly or explicitly.
Hypothesis 2: Machine learning can predict services that are affected by disruptions implicitly or explicitly.
Hypothesis 3: If hypothesis 2 is true, in addition, machine learning can predict the state of the system in
              the next time period after re-scheduling.
"""

from rsp.utils.analysis_tools import average_over_trials, three_dimensional_scatter_plot
from rsp.utils.experiments import load_experiment_results_from_folder, load_experiment_results_from_file

if __name__ == '__main__':
    # Import the desired experiment results
    # data_folder = './results/early_alpha'
    data_file = './results/early_alpha/experiment_53.json'
    # experiment_data = load_experiment_results_from_folder(data_folder)
    ''
    experiment_data = load_experiment_results_from_file(data_file)

    # Average over the trials of each experiment
    averaged_data, std_data = average_over_trials(experiment_data)

    # Initially plot the computation time vs the level size and the number of agent
    three_dimensional_scatter_plot(data=averaged_data, error=std_data, columns=['size', 'n_agents', 'time_full'])
