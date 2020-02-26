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
from typing import List
from typing import Set

import numpy as np
from flatland.envs.rail_trainrun_data_structures import Waypoint
from networkx.drawing.tests.test_pylab import plt
from pandas import DataFrame

from rsp.route_dag.analysis.rescheduling_analysis_utils import _extract_path_search_space
from rsp.route_dag.analysis.rescheduling_analysis_utils import analyze_experiment
from rsp.route_dag.analysis.rescheduling_verification_utils import plausibility_check_experiment_results
from rsp.utils.analysis_tools import average_over_trials
from rsp.utils.analysis_tools import three_dimensional_scatter_plot
from rsp.utils.analysis_tools import two_dimensional_scatter_plot
from rsp.utils.data_types import convert_pandas_series_experiment_results
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentResults
from rsp.utils.experiment_render_utils import visualize_experiment
from rsp.utils.experiments import load_experiment_agenda_from_file
from rsp.utils.experiments import load_experiment_results_from_folder
from rsp.utils.file_utils import check_create_folder


def _2d_analysis(averaged_data: DataFrame, std_data: DataFrame, output_folder: str = None):
    for column in ['n_agents', 'size', 'size_used']:
        two_dimensional_scatter_plot(data=averaged_data,
                                     error=std_data,
                                     columns=[column, 'speed_up'],
                                     colors=['black' if inv_speed_up < 1 else 'red' for inv_speed_up in
                                             averaged_data['time_delta_after_malfunction'] / averaged_data[
                                                 'time_full_after_malfunction']],
                                     title='speed_up delta-rescheduling against re-scheduling',
                                     output_folder=output_folder
                                     )
        two_dimensional_scatter_plot(data=averaged_data,
                                     error=std_data,
                                     columns=[column, 'time_full'],
                                     title='scheduling for comparison',
                                     output_folder=output_folder
                                     )
        two_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                     columns=[column, 'time_full_after_malfunction'],
                                     title='re-scheduling',
                                     output_folder=output_folder
                                     )
        two_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                     columns=[column, 'time_delta_after_malfunction'],
                                     title='delta re-scheduling',
                                     output_folder=output_folder
                                     )

    # resource conflicts
    two_dimensional_scatter_plot(data=averaged_data,
                                 error=std_data,
                                 columns=['nb_resource_conflicts_full',
                                          'time_full'],
                                 title='effect of resource conflicts',
                                 output_folder=output_folder,
                                 )
    two_dimensional_scatter_plot(data=averaged_data,
                                 error=std_data,
                                 columns=['nb_resource_conflicts_full_after_malfunction',
                                          'time_full_after_malfunction'],
                                 title='effect of resource conflicts',
                                 output_folder=output_folder,
                                 )
    two_dimensional_scatter_plot(data=averaged_data,
                                 error=std_data,
                                 columns=['nb_resource_conflicts_delta_after_malfunction',
                                          'time_delta_after_malfunction'],
                                 title='effect of resource conflicts',
                                 output_folder=output_folder,
                                 )

    # nb paths
    two_dimensional_scatter_plot(data=averaged_data,
                                 error=std_data,
                                 columns=['path_search_space_rsp_full',
                                          'time_full_after_malfunction'],
                                 title='impact of number of considered paths over all agents',
                                 output_folder=output_folder,
                                 xscale='log'
                                 )
    two_dimensional_scatter_plot(data=averaged_data,
                                 error=std_data,
                                 columns=['path_search_space_rsp_delta',
                                          'time_delta_after_malfunction'],
                                 title='impact of number of considered paths over all agents',
                                 output_folder=output_folder,
                                 xscale='log'
                                 )
    two_dimensional_scatter_plot(data=averaged_data,
                                 error=std_data,
                                 columns=['path_search_space_rsp_delta',
                                          'n_agents'],
                                 title='impact of number of considered paths over all agents',
                                 output_folder=output_folder,
                                 xscale='log'
                                 )


def _3d_analysis(averaged_data: DataFrame, std_data: DataFrame):
    fig = plt.figure()
    three_dimensional_scatter_plot(data=averaged_data,
                                   error=std_data,
                                   columns=['n_agents', 'size', 'speed_up'],
                                   fig=fig,
                                   subplot_pos='111',
                                   colors=['black' if z_value < 1 else 'red' for z_value in averaged_data['speed_up']])
    three_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                   columns=['n_agents', 'size', 'time_full'],
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


# TODO SIM-250 we should work with malfunction ranges instead of repeating the same experiment under different ids
def _malfunction_analysis(experiment_data: DataFrame):
    # add column 'malfunction_time_step'
    experiment_data['malfunction_time_step'] = 0.0
    experiment_data['experiment_id_group'] = 0.0
    experiment_data['malfunction_time_step'] = experiment_data['malfunction_time_step'].astype(float)
    experiment_data['malfunction_time_step'] = experiment_data['experiment_id_group'].astype(float)
    for index, row in experiment_data.iterrows():
        experiment_results = convert_pandas_series_experiment_results(row)
        time_step = float(experiment_results.malfunction.time_step)
        experiment_data.at[index, 'malfunction_time_step'] = time_step
        experiment_data.at[index, 'experiment_id_group'] = str(row['experiment_id']).split("_")[0]
    print(experiment_data.dtypes)

    # filter 'malfunction_time_step' <150
    experiment_data = experiment_data[experiment_data['malfunction_time_step'] < 150]

    # preview
    print(experiment_data['malfunction_time_step'])
    print(experiment_data['experiment_id_group'])
    malfunction_ids = np.unique(experiment_data['experiment_id_group'].to_numpy())
    print(malfunction_ids)

    # malfunction analysis where malfunction is encoded in experiment id
    check_create_folder('malfunction')
    for i in malfunction_ids:
        fig = plt.figure(constrained_layout=True)
        experiment_data_i = experiment_data[experiment_data['experiment_id_group'] == i]
        two_dimensional_scatter_plot(data=experiment_data_i,
                                     columns=['malfunction_time_step', 'time_full_after_malfunction'],
                                     fig=fig,
                                     title='malfunction_time_step - time_full_after_malfunction ' + str(i)
                                     )
        plt.savefig(f'malfunction/malfunction_{int(i):03d}.png')
        plt.close()


# TODO SIM-151 documentation of derived columns
def hypothesis_one_data_analysis(data_folder: str,
                                 analysis_2d: bool = False,
                                 analysis_3d: bool = False,
                                 malfunction_analysis: bool = False,
                                 qualitative_analysis_experiment_ids: List[str] = None,
                                 flatland_rendering: bool = True
                                 ):
    """

    Parameters
    ----------
    data_folder
    analysis_2d
    analysis_3d
    malfunction_analysis
    qualitative_analysis_experiment_ids
    """
    # Import the desired experiment results
    experiment_data: DataFrame = load_experiment_results_from_folder(data_folder)
    experiment_agenda: ExperimentAgenda = load_experiment_agenda_from_file(data_folder)
    print(data_folder)
    print(experiment_agenda)

    for key in ['size', 'n_agents', 'max_num_cities', 'max_rail_between_cities', 'max_rail_in_city',
                'nb_resource_conflicts_full',
                'nb_resource_conflicts_full_after_malfunction',
                'nb_resource_conflicts_delta_after_malfunction']:
        experiment_data[key] = experiment_data[key].astype(float)

    # add column 'speed_up'
    experiment_data['speed_up'] = \
        experiment_data['time_full_after_malfunction'] / experiment_data['time_delta_after_malfunction']

    # add column 'factor_resource_conflicts'
    experiment_data['factor_resource_conflicts'] = \
        experiment_data['nb_resource_conflicts_delta_after_malfunction'] / experiment_data[
            'nb_resource_conflicts_full_after_malfunction']

    # add column 'path_search_space_* and 'size_used'
    for key in ['path_search_space_schedule', 'path_search_space_rsp_full', 'path_search_space_rsp_delta',
                'factor_path_search_space', 'size_used']:
        experiment_data[key] = 0.0
        experiment_data[key] = experiment_data[key].astype(float)
    for index, row in experiment_data.iterrows():
        experiment_results: ExperimentResults = convert_pandas_series_experiment_results(row)
        path_search_space_rsp_delta, path_search_space_rsp_full, path_search_space_schedule = _extract_path_search_space(
            experiment_results=experiment_results)
        factor_path_search_space = path_search_space_rsp_delta / path_search_space_rsp_full
        experiment_data.at[index, 'path_search_space_schedule'] = path_search_space_schedule
        experiment_data.at[index, 'path_search_space_rsp_full'] = path_search_space_rsp_full
        experiment_data.at[index, 'path_search_space_rsp_delta'] = path_search_space_rsp_delta
        experiment_data.at[index, 'factor_path_search_space'] = factor_path_search_space

        used_cells: Set[Waypoint] = {
            waypoint for agent_id, topo in
            experiment_results.topo_dict.items()
            for waypoint in topo.nodes
        }
        experiment_data.at[index, 'size_used'] = len(used_cells)

    # Plausibility tests on experiment data
    _run_plausibility_tests_on_experiment_data(experiment_data)

    # Average over the trials of each experiment
    print("Averaging...")
    averaged_data, std_data = average_over_trials(experiment_data)
    print("  -> Done averaging.")

    # previews
    preview_cols = ['speed_up', 'time_delta_after_malfunction', 'experiment_id',
                    'nb_resource_conflicts_delta_after_malfunction', 'path_search_space_rsp_full']
    for preview_col in preview_cols:
        print(preview_col)
        print(experiment_data[preview_col])
        print(averaged_data[preview_col])
    print(experiment_data.loc[experiment_data['experiment_id'] == 58].to_json())
    print(experiment_data.dtypes)

    # quantitative analysis
    if malfunction_analysis:
        _malfunction_analysis(experiment_data)
    if analysis_2d:
        _2d_analysis(averaged_data, std_data, output_folder=data_folder)
    if analysis_3d:
        _3d_analysis(averaged_data, std_data)

    # qualitative explorative analysis
    if qualitative_analysis_experiment_ids:
        filtered_experiments = list(filter(
            lambda experiment: experiment.experiment_id in qualitative_analysis_experiment_ids,
            experiment_agenda.experiments))
        for experiment in filtered_experiments:
            analyze_experiment(experiment=experiment, data_frame=experiment_data)
            visualize_experiment(experiment=experiment,
                                 data_frame=experiment_data,
                                 data_folder=data_folder,
                                 flatland_rendering=flatland_rendering)


def _run_plausibility_tests_on_experiment_data(experiment_data):
    print("Running plausibility tests on experiment data...")
    for _, row in experiment_data.iterrows():
        experiment_results: ExperimentResults = convert_pandas_series_experiment_results(row)
        experiment_id = row['experiment_id']
        plausibility_check_experiment_results(experiment_results=experiment_results, experiment_id=experiment_id)
    print("  -> Done plausibility tests on experiment data.")


if __name__ == '__main__':
    hypothesis_one_data_analysis(data_folder='./exp_hypothesis_one_2020_02_19T19_13_25',
                                 analysis_2d=True,
                                 analysis_3d=False,
                                 malfunction_analysis=False,
                                 qualitative_analysis_experiment_ids=[9, 97, 190, 122, 257, 265],
                                 )
