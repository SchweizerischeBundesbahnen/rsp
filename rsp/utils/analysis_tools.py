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
from matplotlib import axes
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame
from pandas import Series

from rsp.utils.data_types import COLUMNS_ANALYSIS
from rsp.utils.data_types import convert_experiment_results_analysis_to_data_frame
from rsp.utils.data_types import convert_pandas_series_experiment_results
from rsp.utils.data_types import expand_experiment_results_for_analysis
from rsp.utils.data_types import ExperimentResults

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
    elif show:
        plt.show()


def two_dimensional_scatter_plot(data: DataFrame,
                                 columns: DataFrame.columns,
                                 error: DataFrame = None,
                                 baseline: DataFrame = None,
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
    baseline
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
    if baseline is not None:
        _2d_plot_baseline(ax, baseline, x_values, y_values)

    if link_column is not None:
        _2d_plot_link_column(ax, columns, data, link_column)

    if file_name is not None:
        plt.savefig(file_name)
    elif output_folder is not None:
        plt.savefig(os.path.join(output_folder, 'experiment_agenda_analysis_' + '_'.join(columns) + '.png'))


def _2d_plot_label_scatterpoints(ax: axes.Axes, experiment_ids: Series, x_values: Series, y_values: Series):
    """Add experiment id to data points."""
    for i in np.arange(0, len(y_values)):
        ax.text(x_values[i], y_values[i], "{}".format(experiment_ids[i]))


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


def _2d_plot_baseline(ax: axes.Axes, baseline: DataFrame, x_values: Series, y_values: Series):
    """Plot baseline y values and draw a line to the data points."""
    for i in np.arange(0, len(y_values)):
        ax.plot([x_values[i], x_values[i]],
                [baseline[i], y_values[i]], marker="_")


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
