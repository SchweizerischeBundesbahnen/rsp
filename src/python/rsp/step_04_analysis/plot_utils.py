import os.path
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from _plotly_utils.colors.qualitative import Plotly
from pandas import DataFrame
from rsp.utils.file_utils import check_create_folder

PLOTLY_COLORLIST = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
                    'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
                    'blueviolet', 'brown', 'burlywood', 'cadetblue',
                    'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
                    'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
                    'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
                    'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
                    'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
                    'darkslateblue', 'darkslategray', 'darkslategrey',
                    'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
                    'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
                    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
                    'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green',
                    'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo',
                    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
                    'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
                    'lightgoldenrodyellow', 'lightgray', 'lightgrey',
                    'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen',
                    'lightskyblue', 'lightslategray', 'lightslategrey',
                    'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
                    'linen', 'magenta', 'maroon', 'mediumaquamarine',
                    'mediumblue', 'mediumorchid', 'mediumpurple',
                    'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
                    'mediumturquoise', 'mediumvioletred', 'midnightblue',
                    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy',
                    'oldlace', 'olive', 'olivedrab', 'orange', 'orangered',
                    'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
                    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink',
                    'plum', 'powderblue', 'purple', 'red', 'rosybrown',
                    'royalblue', 'saddlebrown', 'salmon', 'sandybrown',
                    'seagreen', 'seashell', 'sienna', 'silver', 'skyblue',
                    'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen',
                    'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise',
                    'violet', 'wheat', 'white', 'whitesmoke', 'yellow',
                    'yellowgreen']

GREY_BACKGROUND_COLOR = 'rgba(46,49,49,1)'


def plot_box_plot(
        experiment_data: DataFrame,
        axis_of_interest: str,
        columns_of_interest: List[str],
        output_folder: Optional[str] = None,
        y_axis_title: str = "Time[s]",
        title: str = "Computational Times per",
        file_name_prefix: str = "",
        color_offset: int = 0
):
    """Plot the computational times of experiments.

    Parameters
    ----------

    experiment_data: DataFrame
        DataFrame containing all the results from hypothesis one experiments
    axis_of_interest: str
        Defines along what axis the data will be plotted
    columns_of_interest: List[str]
        Defines which columns of a dataset will be plotted as traces
    output_folder
        if defined, do not show plot but write to file in this folder
    title
        title of the diagram
    file_name_prefix
        prefix for file name
    y_axis_title
    color_offset
        start with offset into defalt colors for columns

    Returns
    -------
    """
    traces = [(axis_of_interest, column) for column in columns_of_interest]
    # prevent too long file names
    pdf_file = f'{file_name_prefix}_{axis_of_interest}__' + ('_'.join(columns_of_interest))[0:15] + '.pdf'
    plot_box_plot_from_traces(experiment_data=experiment_data,
                              traces=traces,
                              output_folder=output_folder,
                              x_axis_title=axis_of_interest,
                              y_axis_title=y_axis_title,
                              pdf_file=pdf_file,
                              title=f"{title} {axis_of_interest}",
                              color_offset=color_offset)


def plot_box_plot_from_traces(
        experiment_data: DataFrame,
        traces: List[Tuple[str, str]],
        x_axis_title: str,
        y_axis_title: str = "Time[s]",
        output_folder: Optional[str] = None,
        pdf_file: Optional[str] = None,
        title: str = "Computational Times",
        color_offset: int = 0
):
    """Plot the computational times of experiments based on traces, i.e.
    (x_axis, y_axis) pairs.

    Parameters
    ----------

    experiment_data: DataFrame
        DataFrame containing all the results from hypothesis one experiments
        Defines which columns of a dataset will be plotted as traces
    traces: List[Tuple[str,str]]
         which pairings of axis_of_interest and column should be displayed?
    output_folder
    pdf_file
        if both defined, do not show plot but write to file in this folder
    title
        title of the diagram
    x_axis_title
        title for x axis (in the case of traces, cannot derived directly from column name)
    color_offset
        start with offset into defalt colors for columns
    y_axis_title

    Returns
    -------
    """
    fig = go.Figure()
    for index, (axis_of_interest, column) in enumerate(traces):
        fig.add_trace(go.Box(x=experiment_data[axis_of_interest],
                             y=experiment_data[column],
                             name=str(column),
                             pointpos=-1,
                             boxpoints='all',
                             customdata=np.dstack((experiment_data['n_agents'],
                                                   experiment_data['size'],
                                                   experiment_data['speed_up_delta_perfect_after_malfunction']))[0],
                             hovertext=experiment_data['experiment_id'],
                             hovertemplate='<b>Time</b>: %{y:.2f}s<br>' +
                                           '<b>Nr. Agents</b>: %{customdata[0]}<br>' +
                                           '<b>Grid Size:</b> %{customdata[1]}<br>' +
                                           '<b>Speed Up:</b> %{customdata[2]:.2f}<br>' +
                                           '<b>Experiment id:</b>%{hovertext}',
                             marker=dict(size=3),
                             marker_color=Plotly[(index + color_offset) % len(Plotly)]
                             ))
    fig.update_layout(boxmode='group')
    fig.update_layout(title_text=f"{title}")
    fig.update_xaxes(title=x_axis_title)
    fig.update_yaxes(title=y_axis_title)
    if output_folder is None or pdf_file is None:
        fig.show()
    else:
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, pdf_file)
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file)


def plot_binned_box_plot(
        experiment_data: DataFrame,
        axis_of_interest: str,
        cols: List[str],
        output_folder: Optional[str] = None,
        y_axis_title: str = "Speed Up Factor",
        axis_of_interest_suffix: str = "",
        nb_bins: Optional[int] = 10,

):
    """

    Parameters
    ----------

    experiment_data: DataFrame
        DataFrame containing all the results from hypothesis one experiments
    axis_of_interest
        Defines along what axis the data will be plotted
    output_folder
        if defined, do not show plot but write to file in this folder
    cols
        columns for y axis
    y_axis_title
        title for y axis instead of technical column name
    axis_of_interest_suffix
        label for x axis will be technical `axis_of_interest` column name plus this suffix
    Returns
    -------

    """
    fig = go.Figure()

    min_value = experiment_data[axis_of_interest].min()
    max_value = experiment_data[axis_of_interest].max()
    inc = (max_value - min_value) / nb_bins
    axis_of_interest_binned = axis_of_interest + "_binned"
    experiment_data.sort_values(by=axis_of_interest, inplace=True)

    binned = axis_of_interest != 'experiment_id'
    if binned:
        experiment_data[axis_of_interest_binned] = experiment_data[axis_of_interest].astype(float).map(
            lambda fl: f"[{((fl - min_value) // inc) * inc + min_value:.2f},{(((fl - min_value) // inc) + 1) * inc + min_value :.2f}]")

    for col_index, col in enumerate(cols):
        fig.add_trace(
            go.Box(
                x=experiment_data[axis_of_interest_binned if binned else axis_of_interest],
                y=experiment_data[col],
                pointpos=-1,
                boxpoints='all',
                name=col,
                customdata=np.dstack((experiment_data['n_agents'],
                                      experiment_data['size'],
                                      experiment_data['solver_statistics_times_total_full'],
                                      experiment_data['solver_statistics_times_total_full_after_malfunction'],
                                      experiment_data['solver_statistics_times_total_delta_perfect_after_malfunction'],
                                      experiment_data['solver_statistics_times_total_delta_no_rerouting_after_malfunction'],
                                      ))[0],
                hovertext=experiment_data['experiment_id'],
                hovertemplate='<b>Speed Up</b>: %{y:.2f}<br>' +
                              '<b>Nr. Agents</b>: %{customdata[0]}<br>' +
                              '<b>Grid Size:</b> %{customdata[1]}<br>' +
                              '<b>Schedule Time:</b> %{customdata[2]:.2f}s<br>' +
                              '<b>Re-Schedule Full Time:</b> %{customdata[3]:.2f}s<br>' +
                              '<b>Delta perfect:</b> %{customdata[4]:.2f}s<br>' +
                              '<b>Delta naive:</b> %{customdata[5]:.2f}s<br>' +
                              '<b>Experiment id:</b>%{hovertext}',
                # dirty workaround: use same color as if we had trace for schedule as well
                marker_color=Plotly[(col_index + 2) % len(Plotly)],
            )
        )

    fig.update_layout(title_text=f"{y_axis_title} per {axis_of_interest}")
    fig.update_layout(boxmode='group')
    fig.update_xaxes(title=f"{axis_of_interest} {axis_of_interest_suffix}")
    fig.update_yaxes(title=y_axis_title)
    if output_folder is None:
        fig.show()
    else:
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, f'{axis_of_interest}__speed_up.pdf')
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file)
