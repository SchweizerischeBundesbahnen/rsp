"""Rendering methods to use with jupyter notebooks."""
import os.path
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from pandas import DataFrame

from rsp.route_dag.analysis.route_dag_analysis import visualize_route_dag_constraints
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.route_dag.route_dag import ScheduleProblemEnum
from rsp.utils.data_types import convert_pandas_series_experiment_results_analysis
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import TimeResourceTrajectories
from rsp.utils.experiments import create_env_pair_for_experiment
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.file_utils import check_create_folder
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


def plot_computational_times(
        experiment_data: DataFrame, axis_of_interest: str,
        columns_of_interest: List[str],
        output_folder: Optional[str] = None,
        y_axis_title: str = "Time[s]",
        title: str = "Computational Times",
        file_name_prefix: str = ""
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

    Returns
    -------
    """
    traces = [(axis_of_interest, column) for column in columns_of_interest]
    # prevent too long file names
    pdf_file = f'{file_name_prefix}_{axis_of_interest}__' + ('_'.join(columns_of_interest))[0:15] + '.pdf'
    plot_computional_times_from_traces(experiment_data=experiment_data,
                                       traces=traces,
                                       output_folder=output_folder,
                                       x_axis_title=axis_of_interest,
                                       y_axis_title=y_axis_title,
                                       pdf_file=pdf_file,
                                       title=f"{title} {axis_of_interest}")


def plot_computional_times_from_traces(experiment_data: DataFrame,
                                       traces: List[Tuple[str, str]],
                                       x_axis_title: str,
                                       y_axis_title: str = "Time[s]",
                                       output_folder: Optional[str] = None,
                                       pdf_file: Optional[str] = None,
                                       title: str = "Computational Times"):
    """Plot the computational times of experiments.

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
    Returns
    -------
    """
    fig = go.Figure()
    for axis_of_interest, column in traces:
        fig.add_trace(go.Box(x=experiment_data[axis_of_interest],
                             y=experiment_data[column],
                             name=column, pointpos=-1,
                             boxpoints='all',
                             customdata=np.dstack((experiment_data['size'], experiment_data['speed_up']))[0],
                             hovertext=experiment_data['experiment_id'],
                             hovertemplate='<b>Time</b>: %{y:.2f}s<br>' +
                                           '<b>Nr. Agents</b>: %{x}<br>' +
                                           '<b>Grid Size:</b> %{customdata[0]}<br>' +
                                           '<b>Speed Up:</b> %{customdata[1]:.2f}<br>' +
                                           '<b>Experiment id:</b>%{hovertext}',
                             marker=dict(size=3)))
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
        output_folder: Optional[str] = None
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
    Returns
    -------

    """
    fig = go.Figure()

    fig.add_trace(go.Box(x=experiment_data[axis_of_interest],
                         y=experiment_data['speed_up'],
                         pointpos=-1,
                         boxpoints='all',
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
    fig.update_layout(title_text=f"Speed Up Factors {axis_of_interest}")
    fig.update_xaxes(title="Agents[#]")
    fig.update_yaxes(title="Speed Up Factor")
    if output_folder is None:
        fig.show()
    else:
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, f'{axis_of_interest}__speed_up.pdf')
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file)


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
    time_resource_schedule, ressource_sorting = resource_time_2d(schedule=schedule,
                                                                 width=width,
                                                                 malfunction_agent_id=malfunction_agent,
                                                                 sorting=None)
    # Get full reschedule Time-resource-Data
    time_resource_reschedule_full, _ = resource_time_2d(schedule=reschedule_full,
                                                        width=width,
                                                        malfunction_agent_id=malfunction_agent,
                                                        sorting=ressource_sorting)
    # Get delta reschedule Time-resource-Data
    time_resource_reschedule_delta, _ = resource_time_2d(schedule=reschedule_delta,
                                                         width=width,
                                                         malfunction_agent_id=malfunction_agent,
                                                         sorting=ressource_sorting)

    # Compute the difference between schedules and return traces for plotting
    traces_influenced_agents, changed_agents_list, nr_influenced_agents = _get_difference_in_time_space(
        time_resource_matrix_a=time_resource_reschedule_delta.trajectories,
        time_resource_matrix_b=time_resource_schedule.trajectories)

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
    changed_agents_traces = _map_variable_to_trainruns(variable=changed_agents_list,
                                                       trainruns=time_resource_reschedule_delta.trajectories)
    additional_data.update({'Changed': changed_agents_traces})
    delay_information = _map_variable_to_trainruns(variable=lateness_delta_after_malfunction,
                                                   trainruns=time_resource_reschedule_delta.trajectories)
    additional_data.update({'Delay': delay_information})
    plot_time_resource_data(time_resource_data=time_resource_reschedule_delta.trajectories, title='Delta Reschedule',
                            ranges=ranges, additional_data=additional_data)

    # Plot difference
    if with_diff:
        additional_data = dict()
        delay_information = _map_variable_to_trainruns(variable=lateness_delta_after_malfunction,
                                                       trainruns=traces_influenced_agents)
        additional_data.update({'Delay': delay_information})
        plot_time_resource_data(time_resource_data=traces_influenced_agents, title='Changed Agents',
                                ranges=ranges, additional_data=additional_data)

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
    fig.add_trace(go.Histogram(x=lateness_full_values, name='Full Reschedule'
                               ))
    fig.add_trace(go.Histogram(x=lateness_delta_values, name='Delta Reschedule'
                               ))

    # fig.update_layout(barmode='group')
    fig.update_layout(title_text="Delay per Agent")
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(len(lateness_full_values)),
        ))
    fig.update_xaxes(title="Delay [s]")
    fig.show()


def plot_route_dag(experiment_results_analysis: ExperimentResultsAnalysis,
                   agent_id: int,
                   suffix_of_constraints_to_visualize: ScheduleProblemEnum,
                   save: bool = False
                   ):
    train_runs_input: TrainrunDict = experiment_results_analysis.solution_full
    train_runs_full_after_malfunction: TrainrunDict = experiment_results_analysis.solution_full_after_malfunction
    train_runs_delta_after_malfunction: TrainrunDict = experiment_results_analysis.solution_delta_after_malfunction
    train_run_input: Trainrun = train_runs_input[agent_id]
    train_run_full_after_malfunction: Trainrun = train_runs_full_after_malfunction[agent_id]
    train_run_delta_after_malfunction: Trainrun = train_runs_delta_after_malfunction[agent_id]
    problem_schedule: ScheduleProblemDescription = experiment_results_analysis.problem_full
    problem_rsp_full: ScheduleProblemDescription = experiment_results_analysis.problem_full_after_malfunction
    problem_rsp_delta: ScheduleProblemDescription = experiment_results_analysis.problem_delta_after_malfunction
    topo = problem_schedule.topo_dict[agent_id]

    config = {
        ScheduleProblemEnum.PROBLEM_SCHEDULE: [problem_schedule, f'Schedule RouteDAG for agent {agent_id}'],
        ScheduleProblemEnum.PROBLEM_RSP_FULL: [problem_rsp_full, f'Full Reschedule RouteDAG for agent {agent_id}'],
        ScheduleProblemEnum.PROBLEM_RSP_DELTA: [problem_rsp_delta, f'Delta Reschedule RouteDAG for agent {agent_id}'],
    }

    problem_to_visualize, title = config[suffix_of_constraints_to_visualize]

    visualize_route_dag_constraints(
        topo=topo,
        train_run_input=train_run_input,
        train_run_full_after_malfunction=train_run_full_after_malfunction,
        train_run_delta_after_malfunction=train_run_delta_after_malfunction,
        f=problem_to_visualize.route_dag_constraints_dict[agent_id],
        vertex_eff_lateness={},
        edge_eff_route_penalties={},
        route_section_penalties=problem_to_visualize.route_section_penalties[agent_id],
        title=title,
        file_name=(
            f"experiment_{experiment_results_analysis.experiment_id:04d}_agent_{agent_id}_route_graph_schedule.pdf"
            if save else None)
    )


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
    additional_information = dict()
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
            additional_information.update({idx: True})
            nr_influenced_agents += 1
        else:
            traces_influenced_agents.append([(None, None)])
            additional_information.update({idx: False})

    return traces_influenced_agents, additional_information, nr_influenced_agents


def resource_time_2d(schedule: TrainrunDict,
                     width: int,
                     malfunction_agent_id: Optional[int] = -1,
                     sorting: Optional[Dict] = None) -> TimeResourceTrajectories:
    """Method to define the time-space paths of each train in two dimensions.
    Initially we order them by the malfunctioning train.

    Parameters
    ----------
    schedule: TrainrunDict
        Contains all the trainruns

    width: int
        width of grid, used to number ressources

    malfunction_agent_id: int
        agent which had malfunctino (used for sorting)

    sorting: List[int]
        Predefined sorting of ressources, if nothing is defined soring is according to first appearing agent (id:0,...)
    Returns
    -------
    All the train trajectories and the max time and max ressource number for plotting
    """
    all_train_time_paths = []
    max_time = 0

    # Sort according to malfunctioning agent
    if sorting is None:
        sorting = dict()
        if malfunction_agent_id >= 0:
            index = 0
            for waypoint in schedule[malfunction_agent_id]:
                position = coordinate_to_position(width, [waypoint.waypoint.position])[0]
                if position not in sorting:
                    sorting.update({position: index})
                    index += 1
    else:
        index = int(len(sorting) + 1)

    for train_run in schedule:
        train_time_path = []
        pre_waypoint = schedule[train_run][0]
        for waypoint in schedule[train_run][1:]:
            pre_time = pre_waypoint.scheduled_at
            time = waypoint.scheduled_at
            if time > max_time:
                max_time = time
            position = coordinate_to_position(width, [pre_waypoint.waypoint.position])[0]
            if position not in sorting:
                sorting.update({position: index})
                index += 1
            train_time_path.append((sorting[position], pre_time))
            train_time_path.append((sorting[position], time))
            train_time_path.append((None, None))
            pre_waypoint = waypoint
        all_train_time_paths.append(train_time_path)
    time_ressource_data = TimeResourceTrajectories(trajectories=all_train_time_paths,
                                                   max_resource_id=index - 1,
                                                   max_time=max_time)
    return time_ressource_data, sorting
