"""
Analysis of the experiment data for hypothesis one.

Hypothesis 1: We can compute good recourse actions, i.e., an adapted plan within the time budget,
              if all the variables are fixed, except those related to services that are affected
              by the disruptions implicitly or explicitly.
Hypothesis 2: Machine learning can predict services that are affected by disruptions implicitly or explicitly.
Hypothesis 3: If hypothesis 2 is true, in addition, machine learning can predict the state of the system in
              the next time period after re-scheduling.
"""
from matplotlib import gridspec
from networkx.drawing.tests.test_pylab import plt

from rsp.utils.analysis_tools import average_over_trials, three_dimensional_scatter_plot
from rsp.utils.analysis_tools import swap_columns, \
    two_dimensional_scatter_plot
from rsp.utils.experiments import load_experiment_results_from_folder, load_experiment_agenda_from_file


def _2d_analysis():
    fig = plt.figure(constrained_layout=True)
    ncols = 2
    nrows = 5
    spec2 = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
    two_dimensional_scatter_plot(data=averaged_data,
                                 error=std_data,
                                 columns=['n_agents', 'speed_up'],
                                 fig=fig,
                                 subplot_pos=spec2[0, 0],
                                 colors=['black' if z_value < 1 else 'red' for z_value in averaged_data['speed_up']],
                                 title='speed up delta-rescheduling against re-scheduling'
                                 )
    two_dimensional_scatter_plot(data=averaged_data,
                                 error=std_data,
                                 columns=['n_agents', 'time_full'],
                                 fig=fig,
                                 subplot_pos=spec2[0, 1],
                                 title='scheduling for comparison',
                                 )
    two_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                 columns=['n_agents', 'time_full_after_malfunction'],
                                 fig=fig,
                                 subplot_pos=spec2[1, 0],
                                 title='re-scheduling'
                                 )
    two_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                 columns=['n_agents', 'time_delta_after_malfunction'],
                                 baseline=averaged_data['time_full_after_malfunction'],
                                 fig=fig,
                                 subplot_pos=spec2[1, 1],
                                 title='delta re-scheduling with re-scheduling as baseline'
                                 )
    two_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                 columns=['n_agents', 'time_delta_after_malfunction'],
                                 fig=fig,
                                 subplot_pos=spec2[2, 0],
                                 title='delta re-scheduling'
                                 )
    two_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                 columns=['size', 'time_full_after_malfunction'],
                                 fig=fig,
                                 subplot_pos=spec2[3, 0],
                                 title='re-scheduling'
                                 )
    two_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                 columns=['size', 'time_delta_after_malfunction'],
                                 baseline=averaged_data['time_full_after_malfunction'],
                                 fig=fig,
                                 subplot_pos=spec2[3, 1],
                                 title='delta re-scheduling with re-scheduling as baseline'
                                 )
    two_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                 columns=['size', 'time_delta_after_malfunction'],
                                 fig=fig,
                                 subplot_pos=spec2[4, 0],
                                 title='delta re-scheduling'
                                 )
    fig.set_size_inches(w=ncols * 8, h=nrows * 8)
    plt.savefig('2d.png')
    plt.show()


def _3d_analysis():
    fig = plt.figure()
    three_dimensional_scatter_plot(data=averaged_data,
                                   error=std_data,
                                   columns=['n_agents', 'size', 'speed_up'],
                                   fig=fig,
                                   subplot_pos='111',
                                   colors=['black' if z_value < 1 else 'red' for z_value in averaged_data['speed_up']])
    three_dimensional_scatter_plot(data=averaged_data, error=std_data, columns=['n_agents', 'size', 'time_full'],
                                   fig=fig,
                                   subplot_pos='121')
    three_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                   columns=['n_agents', 'size', 'time_full_after_malfunction'],
                                   fig=fig,
                                   subplot_pos='211')
    three_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                   columns=['n_agents', 'size', 'time_delta_after_malfunction'],
                                   fig=fig,
                                   subplot_pos='221', )
    fig.set_size_inches(15, 15)
    plt.show()


if __name__ == '__main__':
    # Import the desired experiment results
    data_folder = './exp_hypothesis_one_2020_01_24T09_19_03'
    experiment_data = load_experiment_results_from_folder(data_folder)
    experiment_agenda = load_experiment_agenda_from_file(data_folder)
    print(experiment_agenda)

    for key in ['size', 'n_agents', 'max_num_cities', 'max_rail_between_cities', 'max_rail_in_city']:
        experiment_data[key] = experiment_data[key].astype(float)

    # / TODO SIM-151 re-generate data with bugfix, for the time being swap the wrong values
    swap_columns(experiment_data, 'time_full_after_malfunction', 'time_delta_after_malfunction')
    # \ TODO SIM-151 re-generate data with bugfix, for the time being swap the wrong values

    experiment_data['speed_up'] = \
        experiment_data['time_delta_after_malfunction'] / experiment_data['time_full_after_malfunction']

    # Average over the trials of each experiment
    averaged_data, std_data = average_over_trials(experiment_data)
    print(experiment_data.keys())

    # / TODO SIM-151 remove explorative code
    print(experiment_data['speed_up'])
    print(experiment_data['time_delta_after_malfunction'])
    print(experiment_data['time_full_after_malfunction'])
    print(experiment_data.loc[experiment_data['experiment_id'] == 58].to_json())
    # \ TODO SIM-151 remove explorative code

    _2d_analysis()
    _3d_analysis()
