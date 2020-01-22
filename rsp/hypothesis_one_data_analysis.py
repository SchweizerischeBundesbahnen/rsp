"""
Analysis of the experiment data for hypothesis one.

Hypothesis 1: We can compute good recourse actions, i.e., an adapted plan within the time budget,
              if all the variables are fixed, except those related to services that are affected
              by the disruptions implicitly or explicitly.
Hypothesis 2: Machine learning can predict services that are affected by disruptions implicitly or explicitly.
Hypothesis 3: If hypothesis 2 is true, in addition, machine learning can predict the state of the system in
              the next time period after re-scheduling.
"""

from rsp.utils.analysis_tools import average_over_trials, three_dimensional_scatter_plot, swap_columns
from rsp.utils.experiments import load_experiment_results_from_folder

if __name__ == '__main__':
    # Import the desired experiment results
    data_folder = './exp_hypothesis_one_gathered'
    experiment_data = load_experiment_results_from_folder(data_folder)

    for key in ['size', 'n_agents', 'max_num_cities', 'max_rail_between_cities', 'max_rail_in_city']:
        experiment_data[key] = experiment_data[key].astype(float)

    # / TODO SIM-151 re-generate data with bugfix, for the time being swap the wrong values
    swap_columns(experiment_data, 'time_full_after_malfunction', 'time_delta_after_malfunction')
    # \ TODO SIM-151 re-generate data with bugfix, for the time being swap the wrong values

    experiment_data['speed_up'] = \
        experiment_data['time_full_after_malfunction'] / experiment_data['time_delta_after_malfunction']
    # TODO SIM-151 invert and check range, color code


    # Average over the trials of each experiment
    averaged_data, std_data = average_over_trials(experiment_data)
    print(experiment_data.keys())

    # / TODO SIM-151 remove explorative code
    print(experiment_data['speed_up'])
    print(experiment_data['time_delta_after_malfunction'])
    print(experiment_data['time_full_after_malfunction'])
    print(experiment_data.loc[experiment_data['experiment_id'] == 58].to_json())
    # \ TODO SIM-151 remove explorative code

    # TODO SIM-151 can we display all 4 at the same time and save
    three_dimensional_scatter_plot(data=averaged_data, error=std_data, columns=['n_agents', 'size', 'speed_up'])

    # Initially plot the computation time vs the level size and the number of agent
    three_dimensional_scatter_plot(data=averaged_data, error=std_data, columns=['n_agents', 'size', 'time_full'])

    three_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                   columns=['n_agents', 'size', 'time_full_after_malfunction'])
    three_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                   columns=['n_agents', 'size', 'time_delta_after_malfunction'])
