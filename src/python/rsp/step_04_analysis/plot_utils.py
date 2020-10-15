import os.path
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from _plotly_utils.colors.qualitative import Plotly
from flatland.core.grid.grid_utils import coordinate_to_position
from pandas import DataFrame
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import ResourceOccupation
from rsp.utils.data_types import ScheduleAsResourceOccupations
from rsp.utils.data_types_converters_and_validators import extract_resource_occupations
from rsp.utils.data_types_converters_and_validators import verify_schedule_as_resource_occupations
from rsp.utils.file_utils import check_create_folder
from rsp.utils.global_constants import RELEASE_TIME
from rsp.utils.plotting_data_types import PlottingInformation
from rsp.utils.plotting_data_types import SchedulePlotting

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


def plot_speed_up(
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


def extract_schedule_plotting(
        experiment_result: ExperimentResultsAnalysis,
        sorting_agent_id: Optional[int] = None) -> SchedulePlotting:
    """Extract the scheduling information from a experiment data for plotting.

    Parameters
    ----------
    experiment_result
        Experiment results for plotting
    sorting_agent_id
        Agent according to which trainrun the resources will be sorted
    Returns
    -------
    """
    schedule = experiment_result.solution_full
    reschedule_full = experiment_result.solution_full_after_malfunction
    reschedule_delta_perfect = experiment_result.solution_delta_perfect_after_malfunction
    schedule_as_resource_occupations: ScheduleAsResourceOccupations = extract_resource_occupations(
        schedule=schedule,
        release_time=RELEASE_TIME)
    verify_schedule_as_resource_occupations(schedule_as_resource_occupations=schedule_as_resource_occupations,
                                            release_time=RELEASE_TIME)
    reschedule_full_as_resource_occupations = extract_resource_occupations(
        schedule=reschedule_full,
        release_time=RELEASE_TIME)
    verify_schedule_as_resource_occupations(schedule_as_resource_occupations=reschedule_full_as_resource_occupations,
                                            release_time=RELEASE_TIME)
    reschedule_delta_perfect_as_resource_occupations = extract_resource_occupations(
        schedule=reschedule_delta_perfect,
        release_time=RELEASE_TIME)
    verify_schedule_as_resource_occupations(schedule_as_resource_occupations=reschedule_delta_perfect_as_resource_occupations,
                                            release_time=RELEASE_TIME)
    plotting_information: PlottingInformation = extract_plotting_information(
        schedule_as_resource_occupations=schedule_as_resource_occupations,
        grid_depth=experiment_result.experiment_parameters.infra_parameters.width,
        sorting_agent_id=sorting_agent_id)
    return SchedulePlotting(
        schedule_as_resource_occupations=schedule_as_resource_occupations,
        reschedule_full_as_resource_occupations=reschedule_full_as_resource_occupations,
        reschedule_delta_perfect_as_resource_occupations=reschedule_delta_perfect_as_resource_occupations,
        plotting_information=plotting_information,
        malfunction=experiment_result.malfunction
    )


def extract_plotting_information(
        schedule_as_resource_occupations: ScheduleAsResourceOccupations,
        grid_depth: int,
        sorting_agent_id: Optional[int] = None) -> PlottingInformation:
    """Extract plotting information.

    Parameters
    ----------
    schedule_as_resource_occupations:
    grid_depth
        Ranges of the window to be shown, used for consistent plotting
    sorting_agent_id
        agent id to be used for sorting the resources
    Returns
    -------
    PlottingInformation
        The extracted plotting information.
    """
    sorted_index = 0
    max_time = 0
    sorting = {}
    # If specified, sort according to path of agent with sorting_agent_id
    if sorting_agent_id is not None and sorting_agent_id in schedule_as_resource_occupations.sorted_resource_occupations_per_agent:
        for resource_occupation in sorted(schedule_as_resource_occupations.sorted_resource_occupations_per_agent[sorting_agent_id]):
            position = coordinate_to_position(grid_depth, [resource_occupation.resource])[0]
            time = resource_occupation.interval.to_excl
            if time > max_time:
                max_time = time
            if position not in sorting:
                sorting[position] = sorted_index
                sorted_index += 1

    # Sort the rest of the resources according to agent handle sorting
    for _, sorted_resource_occupations in sorted(schedule_as_resource_occupations.sorted_resource_occupations_per_agent.items()):
        for resource_occupation in sorted_resource_occupations:
            resource_occupation: ResourceOccupation = resource_occupation
            time = resource_occupation.interval.to_excl
            if time > max_time:
                max_time = time
            position = coordinate_to_position(grid_depth, [resource_occupation.resource])[0]
            if position not in sorting:
                sorting[position] = sorted_index
                sorted_index += 1
    max_ressource = max(list(sorting.values()))
    plotting_information = PlottingInformation(sorting=sorting, dimensions=(max_ressource, max_time), grid_width=grid_depth)
    return plotting_information
