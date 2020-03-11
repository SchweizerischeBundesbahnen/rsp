"""Tools and Methods to analyse the data generated by the experiments.

Methods
-------
average_over_grid_id
    Average over all the experiments of the same grid_id
"""
import os
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from matplotlib import axes
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame
from pandas import Series

from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.utils.data_types import COLUMNS_ANALYSIS
from rsp.utils.data_types import convert_experiment_results_analysis_to_data_frame
from rsp.utils.data_types import convert_pandas_series_experiment_results
from rsp.utils.data_types import expand_experiment_results_for_analysis
from rsp.utils.data_types import ExperimentResults
from rsp.utils.data_types import ExperimentResultsAnalysis

# workaround: WORKSPACE is defined in ci where we do not have Qt installed

if 'WORKSPACE' not in os.environ:
    matplotlib.use('Qt5Agg')
# Dummy import currently because otherwise the import is removed all the time but used by 3d scatter plot
axes3d = Axes3D


# https://stackoverflow.com/questions/25649429/how-to-swap-two-dataframe-columns
def swap_columns(df: DataFrame, c1: int, c2: int):
    """Swap columns in a data frame.

    Parameters
    ----------
    df: DataFrame
        The data frame.
    c1: int
        the column index
    c2: int
        the other column index.
    """
    df['temp'] = df[c1]
    df[c1] = df[c2]
    df[c2] = df['temp']
    df.drop(columns=['temp'], inplace=True)


def average_over_grid_id(experimental_data: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Average over all the experiment runs with the same grid_id (different seeds for FLATland).
    Parameters
    ----------
    experimental_data

    Returns
    -------
    DataFrame of mean data and DataFram of standard deviations
    """

    averaged_data = experimental_data.groupby(['grid_id']).mean().reset_index()
    standard_deviation_data = experimental_data.groupby(['grid_id']).std().reset_index()
    return averaged_data, standard_deviation_data


def three_dimensional_scatter_plot(data: DataFrame,
                                   columns: DataFrame.columns,
                                   error: DataFrame = None,
                                   file_name: str = "",
                                   fig: Optional[matplotlib.figure.Figure] = None,
                                   subplot_pos: str = '111',
                                   subplot_title: str = '',
                                   colors: Optional[List[str]] = None):
    """Adds a 3d-scatterplot as a subplot to a figure.

    Parameters
    ----------
    data: DataFrame
        DataFrame containing data to be plotted
    error: DataFrame
        DataFrame containing error of z values to plot
    columns: DataFrame.columns
        Three columns of that data frame to be plotted against each other, x_values, y_values,z_values
    file_name: string
        If provided the plot is stored to figure instead of shown
    fig: Optional[matplotlib.figure.Figure]
        If given, adds the subplot to this figure without showing it, else creates a new one and shows it.
    subplot_pos: str
        a 3-digit integer describing the position of the subplot.
    colors: List[str]
        List of colors for the data points.

    Returns
    -------
    """
    x_values = data[columns[0]].values
    y_values = data[columns[1]].values
    z_values = data[columns[2]].values
    experiment_ids = data['experiment_id'].values

    show = False
    if fig is None:
        show = True
        fig = plt.figure()

    ax: axes.Axes = fig.add_subplot(subplot_pos, projection='3d')
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_zlabel(columns[2])
    if not subplot_title:
        ax.set_title(str(columns))
    else:
        ax.set_title(subplot_title)

    ax.scatter(x_values, y_values, z_values, color=colors)
    for i in np.arange(0, len(z_values)):
        ax.text(x_values[i], y_values[i], z_values[i], "{}".format(experiment_ids[i]))

    if error is not None:
        # plot errorbars
        z_error = error[columns[2]].values
        for i in np.arange(0, len(z_values)):
            ax.plot([x_values[i], x_values[i]], [y_values[i], y_values[i]],
                    [z_values[i] + z_error[i], z_values[i] - z_error[i]], marker="_")

    if len(file_name) > 1:
        plt.savefig(file_name)
        plt.close()
    elif show:
        plt.show()


def two_dimensional_scatter_plot(data: DataFrame,
                                 columns: DataFrame.columns,
                                 error: DataFrame = None,
                                 baseline_data: DataFrame = None,
                                 link_column: str = 'size',
                                 file_name: Optional[str] = None,
                                 output_folder: Optional[str] = None,
                                 fig: Optional[matplotlib.figure.Figure] = None,
                                 subplot_pos: str = '111',
                                 title: str = None,
                                 colors: Optional[List[str]] = None,
                                 xscale: Optional[str] = None,
                                 yscale: Optional[str] = None
                                 ):
    """Adds a 2d-scatterplot as a subplot to a figure.

    Parameters
    ----------
    data: DataFrame
        DataFrame containing data to be plotted
    error: DataFrame
        DataFrame containing error of z values to plot
    columns: DataFrame.columns
        Three columns of that data frame to be plotted against each other, x_values, y_values,z_values
    file_name: string
        If provided the plot is stored to figure instead of shown
    fig: Optional[matplotlib.figure.Figure]
        If given, adds the subplot to this figure without showing it, else creates a new one and shows it.
    subplot_pos: str
        a 3-digit integer describing the position of the subplot.
    colors: List[str]
        List of colors for the data points.
    link_column: str
        Group data points by this column and draw a bar between consecutive data points.
    baseline_data
        data points that define a baseline. Visualized by a bar
    output_folder
        Save plot to this folder.
    title
        Plot title
    xscale
        Passed to matplotlib. See there for possible values such as 'log'.
    yscale
        Passed to matplotlib. See there for possible values such as 'log'.

    Returns
    -------
    """
    x_values: Series = data[columns[0]].values
    y_values: Series = data[columns[1]].values
    experiment_ids = data['experiment_id'].values

    if fig is None:
        fig = plt.figure()
        fig.set_size_inches(w=15, h=15)

    ax: axes.Axes = fig.add_subplot(subplot_pos)
    if title:
        ax.set_title(title)
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)

    ax.scatter(x_values, y_values, color=colors)
    _2d_plot_label_scatterpoints(ax, experiment_ids, x_values, y_values)
    if error is not None:
        _2d_plot_errorbars(ax, columns, error, x_values, y_values)
    if baseline_data is not None:
        y_values_baseline: Series = baseline_data[columns[1]].values
        _2d_plot_baseline(ax, y_values_baseline, x_values, y_values)
        _2d_plot_label_scatterpoints(ax, experiment_ids, x_values, y_values_baseline, suffix='b')

    if link_column is not None:
        _2d_plot_link_column(ax, columns, data, link_column)

    if file_name is not None:
        plt.savefig(file_name)
        plt.close()
    elif output_folder is not None:
        plt.savefig(os.path.join(output_folder, 'experiment_agenda_analysis_' + '_'.join(columns) + '.png'))
        plt.close()


def _2d_plot_label_scatterpoints(ax: axes.Axes, experiment_ids: Series, x_values: Series, y_values: Series,
                                 suffix: str = None):
    """Add experiment id to data points."""
    for i in np.arange(0, min(len(y_values), len(x_values))):
        ax.text(x_values[i], y_values[i], "{}{}".format(experiment_ids[i], '' if suffix is None else suffix))


def _2d_plot_link_column(ax: axes.Axes, columns: DataFrame.columns, data: DataFrame, link_column: str):
    """Group data by a column and draw a line between consecutive data points
    of that group.

    It is assumed that the group has consecutive experiment ids!
    """
    grouped_data = data.groupby([link_column])
    cmap = plt.get_cmap("tab10")
    group_index = 0
    for _, group in grouped_data:
        sorted_group = group.sort_values('experiment_id')
        # index is experiment_id! Therefore, count the number of iterations
        count = 0
        for index, _ in sorted_group.iterrows():
            count += 1
            if count >= len(sorted_group):
                break
            ax.plot([sorted_group.at[index, columns[0]], sorted_group.at[index + 1, columns[0]]],
                    [sorted_group.at[index, columns[1]], sorted_group.at[index + 1, columns[1]]],
                    marker="_",
                    color=cmap(group_index))
        group_index += 1


def _2d_plot_errorbars(ax: axes.Axes, columns: DataFrame.columns, error: DataFrame, x_values: Series, y_values: Series):
    """Plot error range."""
    y_error = error[columns[1]].values
    for i in np.arange(0, len(y_values)):
        ax.plot([x_values[i], x_values[i]],
                [y_values[i] + y_error[i], y_values[i] - y_error[i]], marker="_")


def _2d_plot_baseline(ax: axes.Axes, y_values_baseline: Series, x_values: Series, y_values: Series):
    """Plot baseline y values and draw a line to the data points."""
    for i in np.arange(0, min(len(y_values), len(y_values_baseline))):
        ax.plot([x_values[i], x_values[i]],
                [y_values_baseline[i], y_values[i]], marker="_")


def expand_experiment_data_for_analysis(
        experiment_data: DataFrame, debug: bool = False
) -> DataFrame:
    """Derive additional fields from the computed results without.

    Do it here for the following reasons:
    1. no need to re-run the experiments for new derived properties;
    2. re-run new analysis logic on existing experiment data
    3. keep experiment logic as simple as possible
    """
    data = []

    for _, row in experiment_data.iterrows():
        experiment_results: ExperimentResults = convert_pandas_series_experiment_results(row)

        expanded_experiment_results = expand_experiment_results_for_analysis(
            experiment_results=experiment_results,
            debug=debug
        )
        data.append(convert_experiment_results_analysis_to_data_frame(
            experiment_results=expanded_experiment_results
        ))

    data_frame = pd.DataFrame(columns=COLUMNS_ANALYSIS, data=data)
    for key in ['speed_up',
                'size',
                'n_agents',
                'max_num_cities',
                'max_rail_between_cities',
                'max_rail_in_city',
                'nb_resource_conflicts_full',
                'nb_resource_conflicts_full_after_malfunction',
                'nb_resource_conflicts_delta_after_malfunction',
                'path_search_space_schedule',
                'path_search_space_rsp_full', 'path_search_space_rsp_delta',
                'factor_path_search_space', 'size_used',
                'time_full',
                'time_full_after_malfunction',
                'time_delta_after_malfunction',
                ]:
        print(key)
        print(data_frame[key])
        data_frame[key] = data_frame[key].astype(float)
    return data_frame


def visualize_agent_density(experiment_data: ExperimentResultsAnalysis, output_folder: str):
    """Method to visualize the density of agents in the full schedule for the
    whole episode length. For each time step the number of active agents is
    plotted^.

    Parameters
    ----------
    experiment_data : ExperimentResultsAnalysis
        Data we want to visualize
    output_folder :
        Location to store data

    Returns
    -------
    """
    train_runs_input: TrainrunDict = experiment_data.solution_full
    problem_description: ScheduleProblemDescription = experiment_data.problem_full
    max_episode_steps = problem_description.max_episode_steps
    agent_density = np.zeros(max_episode_steps)

    for train_run in train_runs_input:
        start_time = train_runs_input[train_run][0][0]
        end_time = train_runs_input[train_run][-1][0]
        agent_density[start_time:end_time + 1] += 1

    fig = plt.figure()
    fig.set_size_inches(w=15, h=15)
    ax: plt.axes.Axes = fig.add_subplot(111)
    ax.set_title('Agent density during schedule')
    ax.set_xlabel('Time')
    ax.set_ylabel('Nr. active Agents')
    plt.plot(agent_density)
    plt.savefig(os.path.join(output_folder, 'experiment_agenda_analysis_agent_density.png'))
    plt.close()


def plot_weg_zeit_diagramm_3d(experiment_data: ExperimentResultsAnalysis, volumetric: bool = False):
    """Method to draw ressource-time diagrams in 2d or 3d.

    Parameters
    ----------
    experiment_data : ExperimentResultsAnalysis
        Data from experiment for plot
    output_folder : str
        Folder to store plots
    three_dimensional : bool
        Flag to choose 3D plot
    volumetric : bool
        Flat to choose volumetric plot in 3D

    Returns
    -------
    """
    schedule = experiment_data.solution_full
    reschedule = experiment_data.solution_full_after_malfunction
    max_episode_steps = experiment_data.problem_full.max_episode_steps
    width = experiment_data.experiment_parameters.width
    height = experiment_data.experiment_parameters.height

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('Time-Ressource-Diagram 3D')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Time')
    if volumetric:
        voxels, colors = weg_zeit_3d_voxels(schedule=schedule, width=width, height=height,
                                            max_episode_steps=max_episode_steps)
        ax.voxels(voxels, facecolors=colors)
        plt.show()
    else:
        train_time_paths = weg_zeit_3d_path(schedule=reschedule)
        for train_path in train_time_paths:
            x, y, z = zip(*train_path)
            ax.plot(x, y, z, linewidth=2)
        plt.show()


def save_weg_zeit_diagramm_2d(experiment_data: ExperimentResultsAnalysis, output_folder: str):
    """Method to draw ressource-time diagrams in 2d or 3d.

    Parameters
    ----------
    experiment_data : ExperimentResultsAnalysis
        Data from experiment for plot
    output_folder : str
        Folder to store plots
    three_dimensional : bool
        Flag to choose 3D plot
    volumetric : bool
        Flat to choose volumetric plot in 3D

    Returns
    -------
    """
    schedule = experiment_data.solution_full
    reschedule = experiment_data.solution_full_after_malfunction
    max_episode_steps = experiment_data.problem_full.max_episode_steps
    malfunction_agent = experiment_data.malfunction.agent_id
    width = experiment_data.experiment_parameters.width
    height = experiment_data.experiment_parameters.height

    weg_zeit_matrix_schedule, sorting = weg_zeit_matrix_from_schedule(schedule=schedule, width=width,
                                                                      height=height,
                                                                      malfunction_agent_id=malfunction_agent,
                                                                      max_episode_steps=max_episode_steps,
                                                                      sorting=None)
    weg_zeit_matrix_reschedule, _ = weg_zeit_matrix_from_schedule(schedule=reschedule, width=width, height=height,
                                                                  max_episode_steps=max_episode_steps,
                                                                  malfunction_agent_id=malfunction_agent,
                                                                  sorting=sorting)
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(w=45, h=15)
    ax[0].set_title('Time-Ressource-Diagram: Full Schedule')
    ax[0].set_xlabel('Ressource')
    ax[0].set_ylabel('Time')
    ax[0].matshow(np.transpose(weg_zeit_matrix_schedule), cmap='gist_ncar')
    ax[1].set_title('Time-Ressource-Diagram: Re-Schedule')
    ax[1].set_xlabel('Ressource')
    ax[1].set_ylabel('Time')
    ax[1].matshow(np.transpose(weg_zeit_matrix_reschedule), cmap='gist_ncar')
    ax[2].set_title('Time-Ressource-Diagram: Changes')
    ax[2].set_xlabel('Ressource')
    ax[2].set_ylabel('Time')
    ax[2].matshow(np.abs(np.transpose(weg_zeit_matrix_reschedule) - np.transpose(weg_zeit_matrix_schedule)),
                  cmap='gist_ncar')
    plt.savefig(os.path.join(output_folder, 'experiment_agenda_analysis_time_ressource_diagram.png'))
    plt.close()


def weg_zeit_matrix_from_schedule(schedule: TrainrunDict, width: int, height: int, max_episode_steps: int,
                                  malfunction_agent_id: int = -1, sorting: List[int] = None) -> (np.ndarray, List[int]):
    """Method to produce sorted matrix of all train runs. Each train run is
    given an individual value for better visualization. The matrix can besorted
    according to a predefined soring or accordin to first agent or
    malfunction_agent.

    Parameters
    ----------
    schedule :TrainrunDict
        Contains all the trainruns of the provided schedule
    width : Int
        Width of Flatland env used to span matrix
    height: Int
        Height of Flatland env used to span matrix
    max_episode_steps : int
        Number of time steps in epsidoed used to span matrix
    malfunction_agent_id : int
        ID of malfunctin agent used for sorting
    sorting: List[int]
        Predefined sorting used to maintain soring

    Returns
    -------
    Matrix of Int containing all the reserved ressoruces of all trains.
    """
    weg_zeit_matrix = np.zeros(shape=(width * height, max_episode_steps))
    if sorting is None:
        sorting = []
        if malfunction_agent_id >= 0:
            for waypoint in schedule[malfunction_agent_id]:
                position = coordinate_to_position(width, [waypoint.waypoint.position])  # or is it height?
                if position not in sorting:
                    sorting.append(position)
    for train_run in schedule:
        pre_waypoint = schedule[train_run][0]
        for waypoint in schedule[train_run][1:]:
            pre_time = pre_waypoint.scheduled_at
            time = waypoint.scheduled_at
            position = coordinate_to_position(width, [pre_waypoint.waypoint.position])  # or is it height?
            weg_zeit_matrix[position, pre_time:time] += train_run
            pre_waypoint = waypoint
            if position not in sorting:
                sorting.append(position)
    weg_zeit_matrix = weg_zeit_matrix[sorting][:, 0, :]
    return weg_zeit_matrix, sorting


def weg_zeit_3d_path(schedule: TrainrunDict) -> List[Tuple[int, int, int]]:
    """Method to define the time-space paths of each train in three dimensions.

    Parameters
    ----------
    schedule: TrainrunDict
        Contains all the trainruns

    Returns
    -------
    List of List of coordinate tuples (x,y,z)
    """
    all_train_time_paths = []
    for train_run in schedule:
        train_time_path = []
        pre_waypoint = schedule[train_run][0]
        for waypoint in schedule[train_run][1:]:
            time_pre = pre_waypoint.scheduled_at
            (x_pre, y_pre) = pre_waypoint.waypoint.position
            time = waypoint.scheduled_at
            train_time_path.append((x_pre, y_pre, time_pre))
            train_time_path.append((x_pre, y_pre, time))
            pre_waypoint = waypoint
        all_train_time_paths.append(train_time_path)
    return all_train_time_paths


def weg_zeit_3d_voxels(schedule: TrainrunDict, width: int, height: int, max_episode_steps: int) -> (
        np.ndarray, np.ndarray):
    """
    Parameters
    ----------
    schedule :TrainrunDict
        Contains all the trainruns of the provided schedule
    width : int
        Width of Flatland env used to span matrix
    height: int
        Height of Flatland env used to span matrix
    max_episode_steps : int
        Number of time steps in epsidoed used to span matrix

    Returns
    -------
    Binary matrix (widht,height,max_episode_steps) of occupied ressources, Color for each occupied ressoruce
    """
    voxels = np.zeros(shape=(width, height, max_episode_steps), dtype=int)
    cmap = matplotlib.cm.get_cmap('gist_ncar')
    colors = np.empty(voxels.shape, dtype=object)
    for train_run in schedule:
        pre_waypoint = schedule[train_run][0]
        color = cmap(train_run / len(schedule))
        for waypoint in schedule[train_run][1:]:
            pre_time = pre_waypoint.scheduled_at
            time = waypoint.scheduled_at
            (x, y) = pre_waypoint.waypoint.position
            voxels[x, y, pre_time:time] = 1
            colors[x, y, pre_time:time] = [color]
            pre_waypoint = waypoint
    return voxels, colors
