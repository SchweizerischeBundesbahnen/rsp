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
import os
from functools import partial
from typing import List
from typing import Set

import numpy as np
from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import Waypoint
from networkx.drawing.tests.test_pylab import plt
from pandas import DataFrame

from rsp.rescheduling.rescheduling_analysis_utils import _extract_path_search_space
from rsp.rescheduling.rescheduling_analysis_utils import analyze_experiment
from rsp.solvers.solve_problem import replay
from rsp.utils.analysis_tools import average_over_trials
from rsp.utils.analysis_tools import three_dimensional_scatter_plot
from rsp.utils.analysis_tools import two_dimensional_scatter_plot
from rsp.utils.data_types import convert_pandas_series_experiment_results
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentFreezeDict
from rsp.utils.data_types import ExperimentMalfunction
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ExperimentResults
from rsp.utils.experiments import create_env_pair_for_experiment
from rsp.utils.experiments import load_experiment_agenda_from_file
from rsp.utils.experiments import load_experiment_results_from_folder
from rsp.utils.file_utils import check_create_folder
from rsp.utils.route_graph_analysis import visualize_experiment_freeze


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


def render_experiment(
        experiment: ExperimentParameters,
        data_frame: DataFrame,
        data_folder: str = None,
        rendering: bool = False,
        convert_to_mpeg: bool = True):
    """Render the experiment in the analysis.

    Parameters
    ----------
    experiment: ExperimentParameters
        experiment parameters for all trials
    data_frame: DataFrame
        Pandas data frame with one ore more trials of this experiment.
    data_folder
        Folder to store FLATland pngs and mpeg to
    rendering
        Flatland rendering?
    convert_to_mpeg
        Converts the rendering to mpeg
    """
    from rsp.utils.experiment_solver import RendererForEnvInit, RendererForEnvRender, RendererForEnvCleanup

    # find first row for this experiment (iloc[0]
    rows = data_frame.loc[data_frame['experiment_id'] == experiment.experiment_id]

    static_rail_env, malfunction_rail_env = create_env_pair_for_experiment(experiment)
    train_runs_input: TrainrunDict = rows['solution_full'].iloc[0]
    train_runs_full_after_malfunction: TrainrunDict = rows['solution_full_after_malfunction'].iloc[0]
    train_runs_delta_after_malfunction: TrainrunDict = rows['solution_delta_after_malfunction'].iloc[0]

    experiment_freeze_rsp_full: ExperimentFreezeDict = rows['experiment_freeze_full_after_malfunction'].iloc[0]
    experiment_freeze_rsp_delta: ExperimentFreezeDict = rows['experiment_freeze_delta_after_malfunction'].iloc[0]
    malfunction: ExperimentMalfunction = rows['malfunction'].iloc[0]
    n_agents: int = rows['n_agents'].iloc[0]

    agents_paths_dict = rows['agents_paths_dict'].iloc[0]

    for agent_id in experiment_freeze_rsp_delta:
        experiment_output_folder = f"{data_folder}/experiment_{experiment.experiment_id:04d}_analysis"
        check_create_folder(experiment_output_folder)
        visualize_experiment_freeze(
            agent_paths=agents_paths_dict[agent_id],
            train_run_input=train_runs_input[agent_id],
            train_run_full_after_malfunction=train_runs_full_after_malfunction[agent_id],
            train_run_delta_after_malfunction=train_runs_delta_after_malfunction[agent_id],
            f=experiment_freeze_rsp_delta[agent_id],
            title=f"experiment {experiment.experiment_id}\nagent {agent_id}/{n_agents}\n{malfunction}",
            file_name=(os.path.join(experiment_output_folder,
                                    f"experiment_{experiment.experiment_id:04d}_agent_{agent_id}_route_graph_rsp_delta.png")
                       if data_folder is not None else None)
        )
        visualize_experiment_freeze(
            agent_paths=agents_paths_dict[agent_id],
            train_run_input=train_runs_input[agent_id],
            train_run_full_after_malfunction=train_runs_full_after_malfunction[agent_id],
            train_run_delta_after_malfunction=train_runs_delta_after_malfunction[agent_id],
            f=experiment_freeze_rsp_full[agent_id],
            title=f"experiment {experiment.experiment_id}\nagent {agent_id}/{n_agents}\n{malfunction}",
            file_name=(os.path.join(experiment_output_folder,
                                    f"experiment_{experiment.experiment_id:04d}_agent_{agent_id}_route_graph_rsp_full.png")
                       if data_folder is not None else None)
        )

    controller_from_train_runs_rsp_full = ControllerFromTrainruns(malfunction_rail_env,
                                                                  train_runs_full_after_malfunction)

    if rendering:
        from rsp.utils.experiment_render_utils import cleanup_renderer_for_env
        from rsp.utils.experiment_render_utils import render_env
        from rsp.utils.experiment_render_utils import init_renderer_for_env

        init_renderer_for_env = init_renderer_for_env
        image_output_directory = os.path.join(data_folder, f"experiment_{experiment.experiment_id:04d}_analysis",
                                              f"experiment_{experiment.experiment_id}_rendering_output")
        check_create_folder(image_output_directory)
        render_renderer_for_env = partial(render_env, image_output_directory=image_output_directory)
        cleanup_renderer_for_env = cleanup_renderer_for_env
    else:
        init_renderer_for_env: RendererForEnvInit = lambda *args, **kwargs: None
        render_renderer_for_env: RendererForEnvRender = lambda *args, **kwargs: None
        cleanup_renderer_for_env: RendererForEnvCleanup = lambda *args, **kwargs: None

    renderer = init_renderer_for_env(malfunction_rail_env, rendering)

    def rendering_call_back(test_id: int, solver_name, i_step: int):
        render_renderer_for_env(renderer, test_id, solver_name, i_step)

    replay(
        controller_from_train_runs=controller_from_train_runs_rsp_full,
        env=malfunction_rail_env,
        stop_on_malfunction=False,
        solver_name='data_analysis',
        disable_verification_in_replay=False,
        rendering_call_back=rendering_call_back
    )
    cleanup_renderer_for_env(renderer)
    if convert_to_mpeg:
        experiment_output_folder = f"{data_folder}/experiment_{experiment.experiment_id:04d}_analysis"
        check_create_folder(experiment_output_folder)
        import ffmpeg
        (ffmpeg
         .input(f'{image_output_directory}/flatland_frame_0000_%04d_data_analysis.png', r='5', s='1920x1080')
         .output(f'{experiment_output_folder}/experiment_{experiment.experiment_id}_flatland_data_analysis.mp4', crf=15,
                 pix_fmt='yuv420p', vcodec='libx264')
         .overwrite_output()
         .run()
         )


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
                                 qualitative_analysis_experiment_ids: List[str] = None):
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
            experiment_results=experiment_results, experiment_id=row['experiment_id'])
        factor_path_search_space = path_search_space_rsp_delta / path_search_space_rsp_full
        experiment_data.at[index, 'path_search_space_schedule'] = path_search_space_schedule
        experiment_data.at[index, 'path_search_space_rsp_full'] = path_search_space_rsp_full
        experiment_data.at[index, 'path_search_space_rsp_delta'] = path_search_space_rsp_delta
        experiment_data.at[index, 'factor_path_search_space'] = factor_path_search_space

        used_cells: Set[Waypoint] = {waypoint.position for agent_id, agent_paths in experiment_results.agents_paths_dict.items()
                                     for agent_path in agent_paths
                                     for waypoint in agent_path}
        experiment_data.at[index, 'size_used'] = len(used_cells)

    # Average over the trials of each experiment
    averaged_data, std_data = average_over_trials(experiment_data)

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
            render_experiment(experiment=experiment,
                              data_frame=experiment_data,
                              data_folder=data_folder,
                              rendering=True)


if __name__ == '__main__':
    hypothesis_one_data_analysis(data_folder='./exp_hypothesis_one_2020_01_29T16_24_52-with-conflicts',
                                 analysis_2d=True,
                                 analysis_3d=False,
                                 malfunction_analysis=False,
                                 qualitative_analysis_experiment_ids=[]  # list(range(0, 301))
                                 )
