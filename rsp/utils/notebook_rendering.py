"""Rendering methods to use with jupyter notebooks."""
import os.path
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from pandas import DataFrame

from rsp.utils.analysis_tools import resource_time_2d
from rsp.utils.data_types import ExperimentResultsAnalysis, TimeResourceTrajectories
from rsp.utils.data_types import convert_pandas_series_experiment_results_analysis
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.experiments import create_env_pair_for_experiment
from rsp.utils.flatland_replay_utils import replay_and_verify_trainruns

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


def plot_computational_times(experiment_data: DataFrame, axis_of_interest: str,
                             columns_of_interest: List[str]):
    """Plot the computational times of experiments.

    Parameters
    ----------
    experiment_data: DataFrame
        DataFrame containing all the results from hypothesis one experiments
    axis_of_interest: str
        Defines along what axis the data will be plotted
    columns_of_interest: List[str]
        Defines which columns of a dataset will be plotted as traces

    Returns
    -------
    """
    fig = go.Figure()
    for column in columns_of_interest:
        fig.add_trace(go.Box(x=experiment_data[axis_of_interest],
                             y=experiment_data[column],
                             name=column, pointpos=-1,
                             boxpoints='suspectedoutliers',
                             customdata=np.dstack((experiment_data['size'], experiment_data['speed_up']))[0],
                             hovertext=experiment_data['experiment_id'],
                             hovertemplate='<b>Time</b>: %{y:.2f}s<br>' +
                                           '<b>Nr. Agents</b>: %{x}<br>' +
                                           '<b>Grid Size:</b> %{customdata[0]}<br>' +
                                           '<b>Speed Up:</b> %{customdata[1]:.2f}<br>' +
                                           '<b>Experiment id:</b>%{hovertext}',
                             marker=dict(size=3)))
    fig.update_layout(boxmode='group')
    fig.update_layout(title_text="Computational Times")
    fig.update_xaxes(title=axis_of_interest)
    fig.update_yaxes(title="Time[s]")
    fig.show()


def plot_speed_up(experiment_data: DataFrame, axis_of_interest: str):
    """

    Parameters
    ----------
    experiment_data: DataFrame
        DataFrame containing all the results from hypothesis one experiments
    axis_of_interest
        Defines along what axis the data will be plotted
    Returns
    -------

    """
    fig = go.Figure()

    fig.add_trace(go.Box(x=experiment_data[axis_of_interest],
                         y=experiment_data['speed_up'],
                         pointpos=-1,
                         customdata=np.dstack((experiment_data['size'], experiment_data['time_full'],
                                               experiment_data['time_full_after_malfunction'],
                                               experiment_data['time_delta_after_malfunction']))[0],
                         hovertext=experiment_data['experiment_id'],
                         hovertemplate='<b>Speed Up</b>: %{y:.2f}<br>' +
                                       '<b>Nr. Agents</b>: %{x}<br>' +
                                       '<b>Grid Size:</b> %{customdata[0]}<br>' +
                                       '<b>Full Time:</b> %{customdata[1]:.2f}s<br>' +
                                       '<b>Full Time after:</b> %{customdata[2]:.2f}s<br>' +
                                       '<b>Full Delta after:</b> %{customdata[3]:.2f}s<br>' +
                                       '<b>Experiment id:</b>%{hovertext}',
                         marker=dict(size=3, color='blue')))

    fig.update_layout(boxmode='group')
    fig.update_layout(title_text="Speed Up Factors")
    fig.update_xaxes(title="Agents[#]")
    fig.update_yaxes(title="Speed Up Factor")
    fig.show()


def plot_many_time_resource_diagrams(experiment_data_frame: DataFrame, experiment_id: int, with_diff):
    """Method to draw resource-time diagrams in 2d.

    Parameters
    ----------
    with_diff
    experiment_data_frame : DataFrame
        Data from experiment for plot
    experiment_id: int
        Experiment id used to plot the specific Weg-Zeit-Diagram

    Returns
    -------
    """
    # Extract data
    experiment_data_series = experiment_data_frame.loc[experiment_data_frame['experiment_id'] == experiment_id].iloc[0]
    experiment_data: ExperimentResultsAnalysis = convert_pandas_series_experiment_results_analysis(
        experiment_data_series)
    schedule = experiment_data.solution_full
    reschedule_full = experiment_data.solution_full_after_malfunction
    reschedule_delta = experiment_data.solution_delta_after_malfunction
    malfunction_agent = experiment_data.malfunction.agent_id
    malfunction_start = experiment_data.malfunction.time_step
    malfunction_duration = experiment_data.malfunction.malfunction_duration
    width = experiment_data.experiment_parameters.width
    lateness_delta_after_malfunction = experiment_data.lateness_delta_after_malfunction
    total_delay = sum(lateness_delta_after_malfunction.values())

    # Get full schedule Time-resource-Data
    time_resource_schedule: TimeResourceTrajectories = resource_time_2d(schedule=schedule,
                                                                        width=width,
                                                                        malfunction_agent_id=malfunction_agent,
                                                                        sorting=None)
    # Get full reschedule Time-resource-Data
    time_resource_reschedule_full: TimeResourceTrajectories = resource_time_2d(schedule=reschedule_full,
                                                                               width=width,
                                                                               malfunction_agent_id=malfunction_agent,
                                                                               sorting=None)
    # Get delta reschedule Time-resource-Data
    time_resource_reschedule_delta: TimeResourceTrajectories = resource_time_2d(schedule=reschedule_delta,
                                                                                width=width,
                                                                                malfunction_agent_id=malfunction_agent,
                                                                                sorting=None)

    # Compute the difference between schedules and return traces for plotting
    traces_influenced_agents, plotting_information_traces, nr_influenced_agents = _get_difference_in_time_space(
        time_resource_matrix_a=time_resource_schedule.trajectories,
        time_resource_matrix_b=time_resource_reschedule_delta.trajectories)

    # Printing situation overview
    print(
        "Agent nr.{} has a malfunction at time {} for {} s and influenced {} other agents. Total delay = {}.".format(
            malfunction_agent, malfunction_start,
            malfunction_duration, nr_influenced_agents, total_delay))

    # Plotting the graphs
    ranges = (max(time_resource_schedule.max_resource_id,
                  time_resource_reschedule_full.max_resource_id,
                  time_resource_reschedule_delta.max_resource_id),
              max(time_resource_schedule.max_time,
                  time_resource_reschedule_full.max_time,
                  time_resource_reschedule_delta.max_time))

    # Plot Schedule
    plot_time_resource_data(time_resource_data=time_resource_schedule.trajectories, title='Schedule', ranges=ranges)

    # Plot Reschedule Full only plot this if there is an actual difference to the delta reschedule
    _, _, diff_full_delta = _get_difference_in_time_space(
        time_resource_matrix_a=time_resource_reschedule_full.trajectories,
        time_resource_matrix_b=time_resource_reschedule_delta.trajectories)
    if diff_full_delta > 0:
        plot_time_resource_data(time_resource_data=time_resource_reschedule_full.trajectories, title='Full Reschedule',
                                ranges=ranges)

    # Plot Reschedule Delta with additional data
    additional_data = dict()
    additional_data.update({'Changed': plotting_information_traces})
    delay_information = _map_variable_to_trainruns(variable=lateness_delta_after_malfunction,
                                                   trainruns=time_resource_reschedule_delta.trajectories)
    additional_data.update({'Delay': delay_information})
    plot_time_resource_data(time_resource_data=time_resource_reschedule_delta.trajectories, title='Delta Reschedule',
                            ranges=ranges, additional_data=additional_data)

    # Plot difference
    if with_diff:
        plot_time_resource_data(time_resource_data=traces_influenced_agents, title='Changed Agents',
                                ranges=ranges)

    return


def plot_time_resource_data(title: str, time_resource_data: List[List[Tuple[int, int]]], ranges: Tuple[int, int],
                            additional_data: Dict = None):
    """
    Plot the time-resource-diagram with additional data for each train
    Parameters
    ----------
    title: str
        Title of the plot
    time_resource_data:
        Data to be shown, contains tuples for all occupied ressources during train run
    additional_data
        Dict containing additional data. Each additional data must have the same dimensins as time_resource_data
    ranges
        Ranges of the window to be shown, used for consistent plotting

    Returns
    -------

    """
    layout = go.Layout(
        plot_bgcolor='rgba(46,49,49,1)'
    )
    fig = go.Figure(layout=layout)
    # Get keys and information to add to hover data
    hovertemplate = '<b>Ressource ID:<b> %{x}<br>' + '<b>Time:<b> %{y}<br>'
    if additional_data is not None:
        list_keys = [k for k in additional_data]
        list_values = [v for v in additional_data.values()]
        # Build hovertemplate
        for idx, data_point in enumerate(list_keys):
            hovertemplate += '<b>' + str(data_point) + '</b>: %{{customdata[{}]}}<br>'.format(idx)
        for idx, line in enumerate(time_resource_data):
            x, y = zip(*line)
            trace_color = PLOTLY_COLORLIST[int(idx % len(PLOTLY_COLORLIST))]

            fig.add_trace(go.Scatter(x=x,
                                     y=y,
                                     mode='lines+markers',
                                     marker=dict(size=2, color=trace_color),
                                     line=dict(color=trace_color),
                                     name="Agent {}".format(idx),
                                     customdata=np.dstack([list_values[:][k][idx] for k in range(len(list_values[:]))])[
                                         0],
                                     hovertemplate=hovertemplate
                                     ))
    else:
        for idx, line in enumerate(time_resource_data):
            x, y = zip(*line)
            trace_color = PLOTLY_COLORLIST[int(idx % len(PLOTLY_COLORLIST))]

            fig.add_trace(go.Scatter(x=x,
                                     y=y,
                                     mode='lines+markers',
                                     marker=dict(size=2, color=trace_color),
                                     line=dict(color=trace_color),
                                     name="Agent {}".format(idx),
                                     hovertemplate=hovertemplate
                                     ))

    fig.update_layout(title_text=title, xaxis_showgrid=True, yaxis_showgrid=False)
    fig.update_xaxes(title="Sorted resources", range=[0, ranges[0]])
    fig.update_yaxes(title="Time", range=[ranges[1], 0])
    fig.show()


def plot_histogram_from_delay_data(experiment_data_frame, experiment_id):
    """
    Plot a histogram of the delay of agents in the full and delta reschedule compared to the schedule
    Parameters
    ----------
    experiment_data_frame
    experiment_id

    Returns
    -------

    """
    experiment_data_series = experiment_data_frame.loc[experiment_data_frame['experiment_id'] == experiment_id].iloc[0]

    lateness_full_after_malfunction = experiment_data_series.lateness_full_after_malfunction
    lateness_delta_after_malfunction = experiment_data_series.lateness_delta_after_malfunction
    lateness_full_values = [v for v in lateness_full_after_malfunction.values()]
    lateness_delta_values = [v for v in lateness_delta_after_malfunction.values()]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=np.arange(len(lateness_full_values)),
                         y=lateness_full_values, name='Full Reschedule',
                         hovertemplate='<b>Agent ID</b>: %{x}<br>' +
                                       '<b>Delay</b>: %{y}<br>'
                         ))
    fig.add_trace(go.Bar(x=np.arange(len(lateness_delta_values)),
                         y=lateness_delta_values, name='Delta Reschedule',
                         hovertemplate='<b>Agent ID</b>: %{x}<br>' +
                                       '<b>Delay</b>: %{y}<br>'
                         ))

    fig.update_layout(barmode='group')
    fig.update_layout(title_text="Delay per Agent")
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(len(lateness_full_values)),
        ))
    fig.update_xaxes(title="Agents ID")
    fig.update_yaxes(title="Delay [s]")
    fig.show()


def render_flatland_env(data_folder: str, experiment_data_frame: DataFrame, experiment_id: int,
                        render_schedule: bool = True, render_reschedule: bool = True):
    """
    Method to render the environment for visual inspection
    Parameters
    ----------
    data_folder: str
        Folder name to store and load images from
    experiment_data_frame: DataFrame
        experiment data used for visualization
    experiment_id: int
        ID of experiment we like to visualize

    Returns
    -------
    File paths to generated videos to render in the notebook
    """

    # Extract data
    experiment_data_series = experiment_data_frame.loc[experiment_data_frame['experiment_id'] == experiment_id].iloc[0]
    experiment_data: ExperimentResultsAnalysis = convert_pandas_series_experiment_results_analysis(
        experiment_data_series)

    # Generate environment for rendering
    static_rail_env, malfunction_rail_env = create_env_pair_for_experiment(experiment_data.experiment_parameters)
    # Generate aggregated visualization
    output_folder = f'{data_folder}/{EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME}/'

    # Generate the Schedule video
    if render_schedule:
        # Import the generated video
        title = 'Schedule'
        video_src_schedule = os.path.join(output_folder, f"experiment_{experiment_data.experiment_id:04d}_analysis",
                                          f"experiment_{experiment_data.experiment_id}_rendering_output_{title}/",
                                          f"experiment_{experiment_id}_flatland_data_analysis.mp4")

        # Only render if file is not yet created
        if not os.path.exists(video_src_schedule):
            replay_and_verify_trainruns(data_folder=output_folder,
                                        experiment_id=experiment_data.experiment_id,
                                        title=title,
                                        rendering=True,
                                        rail_env=static_rail_env,
                                        trainruns=experiment_data.solution_full,
                                        convert_to_mpeg=True)
    else:
        video_src_reschedule = None

    # Generate the Reschedule video
    if render_reschedule:
        # Import the generated video
        title = 'Reschedule'
        video_src_reschedule = os.path.join(output_folder, f"experiment_{experiment_data.experiment_id:04d}_analysis",
                                            f"experiment_{experiment_data.experiment_id}_rendering_output_{title}/",
                                            f"experiment_{experiment_id}_flatland_data_analysis.mp4")
        # Only render if file is not yet created
        if not os.path.exists(video_src_reschedule):
            replay_and_verify_trainruns(data_folder=output_folder,
                                        experiment_id=experiment_data.experiment_id,
                                        expected_malfunction=experiment_data.malfunction,
                                        title=title,
                                        rendering=True,
                                        rail_env=malfunction_rail_env,
                                        trainruns=experiment_data.solution_full_after_malfunction,
                                        convert_to_mpeg=True)
    else:
        video_src_reschedule = None

    return Path(video_src_schedule), Path(video_src_reschedule)


def _map_variable_to_trainruns(variable: Dict, trainruns: List[List[Tuple[int, int]]]) -> List[List[object]]:
    """Map data to trainruns for plotting as additional information. There must
    be as many variables in the dict as there are trains in the trainrun list.

    Parameters
    ----------
    variable
        Dictionary containing the variable to be mapped to each individual trainrun
    trainruns
        List of all the trainruns to be plotted in time-ressource-diagram

    Returns
    -------
    List[List[variable]] that can be used as additional information in plotting time-resource-diagrams
    """
    mapped_data = []
    # Get keys and information to map to trainruns
    if variable is not None:
        list_values = [v for v in variable.values()]
    for idx, trainrun in enumerate(trainruns):
        trainrun_mapping = []
        for _ in trainrun:
            trainrun_mapping.append(list_values[idx])
        mapped_data.append(trainrun_mapping)
    return mapped_data


def _get_difference_in_time_space(time_resource_matrix_a, time_resource_matrix_b) -> \
        Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]], int]:
    """
    Compute the difference between schedules and return in plot ready format
    Parameters
    ----------
    time_resource_matrix_a
    time_resource_matrix_b

    Returns
    -------

    """
    # Detect changes to original schedule
    traces_influenced_agents = []
    plotting_information_traces = []
    nr_influenced_agents = 0
    for idx, trainrun in enumerate(time_resource_matrix_a):
        trainrun_difference = []
        for waypoint in trainrun:
            if waypoint not in time_resource_matrix_b[idx]:
                if len(trainrun_difference) > 0:
                    if waypoint[0] != trainrun_difference[-1][0]:
                        trainrun_difference.append((None, None))
                trainrun_difference.append(waypoint)

        if len(trainrun_difference) > 0:
            traces_influenced_agents.append(trainrun_difference)
            plotting_information_traces.append([True for i in range(len(time_resource_matrix_b[idx]))])
            nr_influenced_agents += 1
        else:
            traces_influenced_agents.append([(None, None)])
            plotting_information_traces.append([False for i in range(len(time_resource_matrix_b[idx]))])

    return traces_influenced_agents, plotting_information_traces, nr_influenced_agents
