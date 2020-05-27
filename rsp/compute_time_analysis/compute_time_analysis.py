"""Rendering methods to use with jupyter notebooks."""
import os.path
import re
from pathlib import Path
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import Waypoint
from matplotlib import pyplot as plt
from pandas import DataFrame

from rsp.experiment_solvers.data_types import ExperimentMalfunction
from rsp.route_dag.analysis.route_dag_analysis import visualize_route_dag_constraints
from rsp.route_dag.route_dag import ScheduleProblemDescription
from rsp.route_dag.route_dag import ScheduleProblemEnum
from rsp.utils.data_types import convert_pandas_series_experiment_results_analysis
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import PlottingInformation
from rsp.utils.data_types import ResourceSorting
from rsp.utils.data_types import RessourceAgentDict
from rsp.utils.data_types import RessourceScheduleDict
from rsp.utils.data_types import SortedResourceOccupationsPerAgentDict
from rsp.utils.data_types import TimeAgentDict
from rsp.utils.data_types import TimeScheduleDict
from rsp.utils.data_types import TrainScheduleDict
from rsp.utils.data_types_converters_and_validators import extract_resource_occupations
from rsp.utils.data_types_converters_and_validators import verify_extracted_resource_occupations
from rsp.utils.experiments import create_env_pair_for_experiment
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.file_utils import check_create_folder
from rsp.utils.flatland_replay_utils import convert_trainrundict_to_entering_positions_for_all_timesteps
from rsp.utils.flatland_replay_utils import replay_and_verify_trainruns
from rsp.utils.global_constants import RELEASE_TIME

Trajectories = List[List[Tuple[int, int]]]
SpaceTimeDifference = NamedTuple('Space_Time_Difference', [('changed_agents', Trajectories),
                                                           ('additional_information', Dict)])

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
                             customdata=np.dstack((experiment_data['n_agents'],
                                                   experiment_data['size'],
                                                   experiment_data['speed_up']))[0],
                             hovertext=experiment_data['experiment_id'],
                             hovertemplate='<b>Time</b>: %{y:.2f}s<br>' +
                                           '<b>Nr. Agents</b>: %{customdata[0]}<br>' +
                                           '<b>Grid Size:</b> %{customdata[1]}<br>' +
                                           '<b>Speed Up:</b> %{customdata[2]:.2f}<br>' +
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
                         customdata=np.dstack((experiment_data['n_agents'],
                                               experiment_data['size'],
                                               experiment_data['time_full'],
                                               experiment_data['time_full_after_malfunction'],
                                               experiment_data['time_delta_after_malfunction']))[0],
                         hovertext=experiment_data['experiment_id'],
                         hovertemplate='<b>Speed Up</b>: %{y:.2f}<br>' +
                                       '<b>Nr. Agents</b>: %{customdata[0]}<br>' +
                                       '<b>Grid Size:</b> %{customdata[1]}<br>' +
                                       '<b>Full Time:</b> %{customdata[2]:.2f}s<br>' +
                                       '<b>Full Time after:</b> %{customdata[3]:.2f}s<br>' +
                                       '<b>Full Delta after:</b> %{customdata[4]:.2f}s<br>' +
                                       '<b>Experiment id:</b>%{hovertext}',
                         marker=dict(size=3, color='blue')))

    fig.update_layout(boxmode='group')
    fig.update_layout(title_text=f"Speed Up Factors {axis_of_interest}")
    fig.update_xaxes(title=axis_of_interest)
    fig.update_yaxes(title="Speed Up Factor")
    if output_folder is None:
        fig.show()
    else:
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, f'{axis_of_interest}__speed_up.pdf')
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file)


def plot_many_time_resource_diagrams(experiment_data_frame: DataFrame, experiment_id: int, with_diff: bool = True) -> Dict[int, bool]:
    """Method to draw resource-time diagrams in 2d.

    Parameters
    ----------

    experiment_data_frame : DataFrame
        Data from experiment for plot
    experiment_id: int
        Experiment id used to plot the specific Weg-Zeit-Diagram
    with_diff
        plot difference as well?

    Returns
    -------
        List of agent ids that changed between schedule an reschedule full
    """
    # Extract data
    experiment_data_series = experiment_data_frame.loc[experiment_data_frame['experiment_id'] == experiment_id].iloc[0]
    experiment_result: ExperimentResultsAnalysis = convert_pandas_series_experiment_results_analysis(
        experiment_data_series)

    schedule = experiment_result.solution_full
    reschedule_full = experiment_result.solution_full_after_malfunction

    schedule_resource_occupations_per_resource, schedule_resource_occupations_per_agent = extract_resource_occupations(
        schedule=schedule,
        release_time=RELEASE_TIME)
    verify_extracted_resource_occupations(resource_occupations_per_agent=schedule_resource_occupations_per_agent,
                                          resource_occupations_per_resource=schedule_resource_occupations_per_resource,
                                          release_time=RELEASE_TIME)
    reschedule_resource_occupations_per_resource, reschedule_resource_occupations_per_agent = extract_resource_occupations(
        schedule=reschedule_full,
        release_time=RELEASE_TIME)
    verify_extracted_resource_occupations(resource_occupations_per_agent=reschedule_resource_occupations_per_agent,
                                          resource_occupations_per_resource=reschedule_resource_occupations_per_resource,
                                          release_time=RELEASE_TIME)

    reschedule_delta_resource_occupations_per_resource, reschedule_delta_resource_occupations_per_agent = extract_resource_occupations(
        schedule=reschedule_full,
        release_time=RELEASE_TIME)
    verify_extracted_resource_occupations(resource_occupations_per_agent=reschedule_delta_resource_occupations_per_agent,
                                          resource_occupations_per_resource=reschedule_delta_resource_occupations_per_resource,
                                          release_time=RELEASE_TIME)

    plotting_information: PlottingInformation = extract_plotting_information_from_train_schedule_dict(
        schedule_data=convert_trainrundict_to_entering_positions_for_all_timesteps(schedule, only_travelled_positions=True),
        width=experiment_result.experiment_parameters.width)

    return _plot_resource_time_diagram(
        malfunction=experiment_result.malfunction,
        plotting_information=plotting_information,
        resource_occupations_schedule=schedule_resource_occupations_per_agent,
        resource_occupations_reschedule_full=reschedule_resource_occupations_per_agent,
        resource_occupations_reschedule_delta=reschedule_delta_resource_occupations_per_agent
    )


def extract_plotting_information_from_train_schedule_dict(
        schedule_data: TrainScheduleDict,
        width: int) -> PlottingInformation:
    """Extract plotting information.

    Parameters
    ----------
    schedule_data:
        Data to be shown, contains tuples for all occupied ressources during train run
    width
        Ranges of the window to be shown, used for consistent plotting

    Returns
    -------
    """
    sorted_index = 0
    max_time = 0
    max_ressource = 0
    sorting = {}

    for agent_idx in sorted(schedule_data):
        trace = schedule_data[agent_idx]
        for time, waypoint in trace.items():
            if time > max_time:
                max_time = time
            ressource = coordinate_to_position(width, [waypoint.position])
            if ressource[0] not in sorting:
                sorting[ressource[0]] = sorted_index
                sorted_index += 1
            if sorted_index > max_ressource:
                max_ressource = sorted_index
    plotting_parameters = PlottingInformation(sorting=sorting, dimensions=(max_ressource, max_time), grid_width=width)
    return plotting_parameters


# TODO SIM-537 convert scheduleproblemdescription to resource occupations and derive Trajectories using trajectories_from_resource_occupations_per_agent
def _trajectories_from_time_windows(problem: ScheduleProblemDescription, resource_sorting: ResourceSorting, width) -> Trajectories:
    schedule_trajectories: Trajectories = []

    for _, route_dag_constraints in problem.route_dag_constraints_dict.items():
        train_time_path = []
        earliest_resource = {}
        latest_resource = {}
        for waypoint, earliest in route_dag_constraints.freeze_earliest.items():
            waypoint: Waypoint = waypoint
            resource = waypoint.position
            earliest_resource.setdefault(resource, earliest)
            earliest_resource[resource] = min(earliest_resource[resource], earliest)
        for waypoint, latest in route_dag_constraints.freeze_latest.items():
            waypoint: Waypoint = waypoint
            resource = waypoint.position
            latest_resource.setdefault(resource, latest)
            latest_resource[resource] = max(latest_resource[resource], latest)
        for resource, earliest in earliest_resource.items():
            position = coordinate_to_position(width, [resource])[0]
            # TODO dirty hack: add positions not in schedule, improve resource_sorting!
            if position not in resource_sorting:
                resource_sorting[position] = len(resource_sorting)
            train_time_path.append((resource_sorting[position], earliest))
            train_time_path.append((resource_sorting[position], latest_resource[resource]))
            train_time_path.append((None, None))
        schedule_trajectories.append(train_time_path)
    return schedule_trajectories


def plot_time_window_resource_trajectories(
        experiment_result: ExperimentResultsAnalysis,
        show: bool = True):
    """Plot time-window -- resource diagram for all three problems.

    Parameters
    ----------
    experiment_result
    show
    """
    width = experiment_result.experiment_parameters.width
    plotting_parameters: PlottingInformation = extract_plotting_information_from_train_schedule_dict(
        schedule_data=convert_trainrundict_to_entering_positions_for_all_timesteps(
            trainrun_dict=experiment_result.results_full.trainruns_dict,
            only_travelled_positions=True),
        width=width)
    ranges = (len(plotting_parameters.sorting),
              max(experiment_result.problem_full.max_episode_steps,
                  experiment_result.problem_full_after_malfunction.max_episode_steps,
                  experiment_result.problem_delta_after_malfunction.max_episode_steps))
    for title, (problem, malfunction) in {
        'Schedule': (experiment_result.problem_full, None),
        'Full Re-Schedule': (experiment_result.problem_full_after_malfunction, experiment_result.malfunction),
        'Delta Re-Schedule': (experiment_result.problem_delta_after_malfunction, experiment_result.malfunction)
    }.items():
        trajectories = _trajectories_from_time_windows(problem, plotting_parameters.sorting, width)
        plot_time_resource_trajectories(trajectories=trajectories, title=title, ranges=ranges, show=show, malfunction=malfunction)


def plot_shared_heatmap(
        experiment_result: ExperimentResultsAnalysis,
        show: bool = True):
    """Plot a heat map of how many shareds are on the resources.

    Parameters
    ----------
    experiment_result
    show
    """
    for title, result in {
        'Schedule': experiment_result.results_full,
        'Full Re-Schedule': experiment_result.results_full_after_malfunction,
        'Delta Re-Schedule': experiment_result.results_delta_after_malfunction
    }.items():
        shared = list(filter(lambda s: s.startswith('shared'), result.solver_result))
        distance_matrix = np.zeros((experiment_result.experiment_parameters.height, experiment_result.experiment_parameters.width))
        for sh in shared:
            sh = sh.replace('shared', '')
            sh = re.sub('t[0-9]+', '"XXX"', sh)
            (t0, (wp00, wp01), t1, (wp10, wp11)) = eval(sh)
            distance_matrix[wp00[0]] += 1
            distance_matrix[wp01[0]] += 1
        distance_matrix /= np.max(distance_matrix)
        fig = plt.figure(figsize=(18, 12), dpi=80)
        fig.suptitle(title, fontsize=16)
        plt.subplot(121)
        plt.imshow(distance_matrix, cmap='hot', interpolation='nearest')
        if show:
            fig.show()


def _plot_resource_time_diagram(malfunction: ExperimentMalfunction,
                                plotting_information: PlottingInformation,
                                resource_occupations_schedule: SortedResourceOccupationsPerAgentDict,
                                resource_occupations_reschedule_full: SortedResourceOccupationsPerAgentDict,
                                resource_occupations_reschedule_delta: SortedResourceOccupationsPerAgentDict,
                                with_diff: bool = True
                                ) -> Dict[int, bool]:
    resource_sorting = plotting_information.sorting
    width = plotting_information.grid_width
    trajectories_schedule: Trajectories = trajectories_from_resource_occupations_per_agent(resource_occupations_schedule, resource_sorting, width)
    trajectories_reschedule_full: Trajectories = trajectories_from_resource_occupations_per_agent(resource_occupations_reschedule_full, resource_sorting, width)
    trajectories_reschedule_delta: Trajectories = trajectories_from_resource_occupations_per_agent(resource_occupations_reschedule_delta, resource_sorting,
                                                                                                   width)

    total_delay = sum(
        max(resource_occupations_schedule[agent_id][-1].interval.to_excl - sorted_resource_occupations_reschedule_delta[-1].interval.to_excl, 0)
        for agent_id, sorted_resource_occupations_reschedule_delta in resource_occupations_reschedule_delta.items()
    )

    # Plot Schedule
    plot_time_resource_trajectories(
        title='Schedule',
        ranges=plotting_information.dimensions,
        trajectories=trajectories_schedule)

    # Plot Reschedule Full only plot this if there is an actual difference between schedule and reschedule
    trajectories_influenced_agents, changed_agents_list = _get_difference_in_time_space_trajectories(
        trajectories_a=trajectories_schedule,
        trajectories_b=trajectories_reschedule_full)

    # Printing situation overview
    nb_changed_agents = sum([1 for changed in changed_agents_list.values() if changed])
    print(
        "Agent nr.{} has a malfunction at time {} for {} s and influenced {} other agents. Total delay = {}.".format(
            malfunction.agent_id,
            malfunction.time_step,
            malfunction.malfunction_duration,
            nb_changed_agents,
            total_delay))
    # Plot Reschedule Full only if something has changed
    if nb_changed_agents > 0:
        plot_time_resource_trajectories(
            trajectories=trajectories_reschedule_full,
            title='Full Reschedule',
            malfunction=malfunction,
            ranges=plotting_information.dimensions
        )

    # Plot Reschedule Delta with additional data
    plot_time_resource_trajectories(
        title='Delta Reschedule', ranges=plotting_information.dimensions,
        trajectories=trajectories_reschedule_delta,
        malfunction=malfunction
    )

    # Plot difference if asked for
    if with_diff:
        plot_time_resource_trajectories(
            trajectories=trajectories_influenced_agents,
            title='Changed Agents',
            malfunction=malfunction,
            ranges=plotting_information.dimensions
        )

    return changed_agents_list


def plot_time_resource_trajectories(
        title: str,
        trajectories: Trajectories,
        ranges: Tuple[int, int],
        additional_data: Dict = None,
        malfunction: ExperimentMalfunction = None,
        show: bool = True
):
    """
    Plot the time-resource-diagram with additional data for each train
    Parameters
    ----------

    title: str
        Title of the plot
    trajectories:
        Data to be shown, contains tuples for all occupied resources during train run
    additional_data
        Dict containing additional data. Each additional data must have the same dimensins as time_resource_data
    ranges
        Ranges of the window to be shown, used for consistent plotting
    malfunction: ExperimentMalfunction

    show: bool

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
        for idx, line in enumerate(trajectories):
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
        for idx, line in enumerate(trajectories):
            # skip empty schedule (re-schedle for our ghost agent representing the wave front)
            if len(line) == 0:
                continue
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
    if malfunction is not None:
        x = [-10, ranges[1] + 10]
        y = [malfunction.time_step, malfunction.time_step]
        fig.add_trace(go.Scatter(x=x, y=y, name='malfunction start', line=dict(color='red')))
        y = [malfunction.time_step + malfunction.malfunction_duration, malfunction.time_step + malfunction.malfunction_duration]
        fig.add_trace(go.Scatter(x=x, y=y, name='malfunction end', line=dict(color='red', dash='dash')))
    fig.update_layout(title_text=title, xaxis_showgrid=True, yaxis_showgrid=False)
    fig.update_xaxes(title="Sorted resources", range=[0, ranges[0]])
    fig.update_yaxes(title="Time", range=[ranges[1], 0])
    if show:
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
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    fig.update_layout(title_text="Delay distributions")
    fig.update_xaxes(title="Delay [s]")
    fig.show()


def plot_route_dag(experiment_results_analysis: ExperimentResultsAnalysis,
                   agent_id: int,
                   suffix_of_constraints_to_visualize: ScheduleProblemEnum,
                   save: bool = False
                   ):
    train_runs_full: TrainrunDict = experiment_results_analysis.solution_full
    train_runs_full_after_malfunction: TrainrunDict = experiment_results_analysis.solution_full_after_malfunction
    train_runs_delta_after_malfunction: TrainrunDict = experiment_results_analysis.solution_delta_after_malfunction
    train_run_full: Trainrun = train_runs_full[agent_id]
    train_run_full_after_malfunction: Trainrun = train_runs_full_after_malfunction[agent_id]
    train_run_delta_after_malfunction: Trainrun = train_runs_delta_after_malfunction[agent_id]
    problem_schedule: ScheduleProblemDescription = experiment_results_analysis.problem_full
    problem_rsp_full: ScheduleProblemDescription = experiment_results_analysis.problem_full_after_malfunction
    problem_rsp_delta: ScheduleProblemDescription = experiment_results_analysis.problem_delta_after_malfunction
    topo = problem_schedule.topo_dict[agent_id]

    config = {
        ScheduleProblemEnum.PROBLEM_SCHEDULE: [
            problem_schedule,
            f'Schedule RouteDAG for agent {agent_id} in experiment {experiment_results_analysis.experiment_id}',
            train_run_full],
        ScheduleProblemEnum.PROBLEM_RSP_FULL: [
            problem_rsp_full,
            f'Full Reschedule RouteDAG for agent {agent_id} in experiment {experiment_results_analysis.experiment_id}',
            train_run_full_after_malfunction],
        ScheduleProblemEnum.PROBLEM_RSP_DELTA: [
            problem_rsp_delta,
            f'Delta Reschedule RouteDAG for agent {agent_id} in experiment {experiment_results_analysis.experiment_id}',
            train_run_delta_after_malfunction],
    }

    problem_to_visualize, title, trainrun_to_visualize = config[suffix_of_constraints_to_visualize]

    visualize_route_dag_constraints(
        topo=topo,
        train_run_full=train_run_full,
        train_run_full_after_malfunction=train_run_full_after_malfunction,
        train_run_delta_after_malfunction=train_run_delta_after_malfunction,
        constraints_to_visualize=problem_to_visualize.route_dag_constraints_dict[agent_id],
        trainrun_to_visualize=trainrun_to_visualize,
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


def _get_difference_in_time_space_trajectories(trajectories_a: Trajectories, trajectories_b: Trajectories) -> SpaceTimeDifference:
    """
    Compute the difference between schedules and return in plot ready format
    Parameters
    ----------
    trajectories_a
    trajectories_b

    Returns
    -------

    """
    # Detect changes to original schedule
    traces_influenced_agents: Trajectories = []
    additional_information = dict()
    for idx, trainrun in enumerate(trajectories_a):
        trainrun_difference = []
        for waypoint in trainrun:
            if waypoint not in trajectories_b[idx]:
                if len(trainrun_difference) > 0 and waypoint[0] != trainrun_difference[-1][0]:
                    trainrun_difference.append((None, None))
                trainrun_difference.append(waypoint)

        if len(trainrun_difference) > 0:
            traces_influenced_agents.append(trainrun_difference)
            additional_information.update({idx: True})
        else:
            traces_influenced_agents.append([(None, None)])
            additional_information.update({idx: False})
    space_time_difference = SpaceTimeDifference(changed_agents=traces_influenced_agents,
                                                additional_information=additional_information)
    return space_time_difference


def plot_schedule_metrics(experiment_data_frame: ExperimentResultsAnalysis, experiment_id: int):
    experiment_data_series = experiment_data_frame.loc[experiment_data_frame['experiment_id'] == experiment_id].iloc[0]
    experiment_data: ExperimentResultsAnalysis = convert_pandas_series_experiment_results_analysis(
        experiment_data_series)
    malfunction = experiment_data.malfunction
    delay = experiment_data.lateness_delta_after_malfunction
    schedule = experiment_data.solution_full
    reschedule_delta = experiment_data.solution_delta_after_malfunction
    width = experiment_data.experiment_parameters.width

    # Get full schedule Time-resource-Data
    train_schedule_dict_schedule: TrainScheduleDict = \
        convert_trainrundict_to_entering_positions_for_all_timesteps(schedule, only_travelled_positions=True)

    # Get delta reschedule Time-resource-Data
    train_schedule_dict_reschedule_delta: TrainScheduleDict = convert_trainrundict_to_entering_positions_for_all_timesteps(
        reschedule_delta, only_travelled_positions=True)

    # Compute the difference between schedules and return traces for plotting
    plotting_information = extract_plotting_information_from_train_schedule_dict(schedule_data=train_schedule_dict_schedule, width=width)
    trajectories_influenced_agents, changed_agents_list = _get_difference_in_time_space_trajectories(
        trajectories_a=trajectories_from_train_schedule_dict(train_schedule_dict_reschedule_delta, plotting_information=plotting_information),
        trajectories_b=trajectories_from_train_schedule_dict(train_schedule_dict_schedule, plotting_information=plotting_information)
    )

    schedule_times, schedule_resources = _schedule_to_time_ressource_dicts(train_schedule_dict_schedule)
    _, re_schedule_resources = _schedule_to_time_ressource_dicts(train_schedule_dict_reschedule_delta)

    # Plot Density over time
    _plot_time_density(schedule_times)

    # Plot Occupancy over space
    _plot_ressource_occupation(schedule_resources, width=width)
    _plot_ressource_occupation(re_schedule_resources, width=width)

    # Plot Delay propagation
    changed_agent_dict = {}
    for agent, changed in changed_agents_list.items():
        if changed:
            changed_agent_dict[agent] = train_schedule_dict_schedule[agent]
    delay_depth_dict = _delay_cause_level(train_schedule_dict_schedule, malfunction=malfunction)
    _plot_delay_propagation(changed_agent_dict, malfunction=malfunction, delay_information=delay, width=width,
                            depth_dict=delay_depth_dict)
    print(delay_depth_dict)
    return schedule_times, schedule_resources


def _schedule_to_time_ressource_dicts(schedule: TrainScheduleDict) -> Tuple[TimeScheduleDict, RessourceScheduleDict]:
    """Convert TrainScheuldeDict into dicts for all time steps and all
    ressources such that analysis on densities can be made.

    Parameters
    ----------
    schedule: TrainScheduleDict
        Schedule with all the trainruns that will be transfered

    Returns
    -------
    Tuple[TimeScheduleDict, RessourceScheduleDict]
        Containing the views in time and ressources on agent handles an occupancies
    """
    timescheduledict: TimeScheduleDict = {}
    ressourcescheduledict: RessourceScheduleDict = {}
    for train_id in schedule:
        for time in schedule[train_id]:
            ressource = schedule[train_id][time].position
            if time not in timescheduledict:
                timescheduledict[time]: RessourceAgentDict = {}
            if ressource not in ressourcescheduledict:
                ressourcescheduledict[ressource]: TimeAgentDict = {}
            timescheduledict[time][ressource] = train_id
            ressourcescheduledict[ressource][time] = train_id
    return timescheduledict, ressourcescheduledict


def _plot_ressource_occupation(schedule_ressources: RessourceScheduleDict, width: int):
    """
    Plot agent density over ressource
    Parameters
    ----------
    schedule_ressources
        Dict containing all the times and agent handles for all ressources

    Returns
    -------

    """
    x = []
    y = []
    size = []
    color = []
    layout = go.Layout(
        plot_bgcolor='rgba(46,49,49,1)'
    )
    fig = go.Figure(layout=layout)

    for waypoint in schedule_ressources:
        x.append(waypoint[1])
        y.append(waypoint[0])
        size.append((len(schedule_ressources[waypoint])))
        times = np.array(sorted(schedule_ressources[waypoint].keys()))
        if len(times) > 1:
            mean_temp_dist = np.mean(np.clip(times[1:] - times[:-1], 0, 50))
            color.append(mean_temp_dist)
        else:
            color.append(50)
    fig.add_trace(go.Scatter(x=x,
                             y=y,
                             mode='markers',
                             name="Schedule",
                             marker=dict(
                                 color=size,
                                 symbol='square',
                                 showscale=True,
                                 reversescale=False
                             )))
    fig.update_layout(title_text="Train Density at Ressources",
                      autosize=False,
                      width=1000,
                      height=1000)

    fig.update_yaxes(zeroline=False, showgrid=True, range=[width, 0], tick0=-0.5, dtick=1, gridcolor='Grey')
    fig.update_xaxes(zeroline=False, showgrid=True, range=[0, width], tick0=-0.5, dtick=1, gridcolor='Grey')

    fig.show()


def _plot_delay_propagation(schedule: TrainScheduleDict, malfunction: ExperimentMalfunction,
                            delay_information, width: int, depth_dict: dict):
    """
    Plot agent density over ressource
    Parameters
    ----------
    schedule_ressources
        Dict containing all the times and agent handles for all ressources

    Returns
    -------

    """

    MARKER_LIST = ['triangle-up', 'triangle-right', 'triangle-down', 'triangle-left']
    DEPTH_COLOR = ['red', 'orange', 'yellow', 'white', 'LightGreen', 'green']
    layout = go.Layout(
        plot_bgcolor='rgba(46,49,49,1)'
    )
    fig = go.Figure(layout=layout)

    # Sort agents according to influence depth for plotting
    agents = []
    for agent, _depth in sorted(depth_dict.items(), key=lambda item: item[1], reverse=True):
        if agent in schedule:
            agents.append(agent)
    for agent in schedule:
        if agent not in agents:
            agents.append(agent)

    # Plot traces of agents
    for agent_id in agents:
        x = []
        y = []
        size = []
        marker = []
        times = []
        delay = []
        conflict_depth = []
        for time in schedule[agent_id]:
            waypoint = schedule[agent_id][time]
            x.append(waypoint.position[1])
            y.append(waypoint.position[0])
            size.append(max(10, delay_information[agent_id]))
            marker.append(MARKER_LIST[int(np.clip(waypoint.direction, 0, 3))])
            times.append(time)
            delay.append(delay_information[agent_id])
            if agent_id in depth_dict:
                conflict_depth.append(depth_dict[agent_id])
            else:
                conflict_depth.append("None")
        if agent_id in depth_dict:
            color = DEPTH_COLOR[int(np.clip(depth_dict[agent_id], 0, 5))]
        else:
            color = DEPTH_COLOR[-1]
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode='markers',
                                 name="Train {}".format(agent_id),
                                 marker_symbol=marker,
                                 customdata=list(zip(times, delay, conflict_depth)),
                                 marker_size=size,
                                 marker_opacity=0.1,
                                 marker_color=color,
                                 marker_line_color=color,
                                 hovertemplate="Time:\t%{customdata[0]}<br>" +
                                               "Delay:\t%{customdata[1]}<br>" +
                                               "Influence depth:\t%{customdata[2]}"
                                 ))
    # Plot malfunction
    waypoint = list(schedule[malfunction.agent_id].values())[0].position
    fig.add_trace(go.Scatter(x=[waypoint[1]],
                             y=[waypoint[0]],
                             mode='markers',
                             name="Malfunction",
                             marker_symbol='x',
                             marker_size=25,
                             marker_line_color='black',
                             marker_color='black'))
    fig.update_layout(title_text="Malfunction position and effects",
                      autosize=False,
                      width=1000,
                      height=1000)

    fig.update_yaxes(zeroline=False, showgrid=True, range=[width, 0], tick0=-0.5, dtick=1, gridcolor='Grey')
    fig.update_xaxes(zeroline=False, showgrid=True, range=[0, width], tick0=-0.5, dtick=1, gridcolor='Grey')

    fig.show()


def _plot_time_density(schedule_times: TimeScheduleDict):
    """Plot agent density over time.

    Parameters
    ----------
    schedule_times
        Dict containing ressources and agent handles for all times

    Returns
    -------
    """
    x = []
    y = []
    layout = go.Layout(
        plot_bgcolor='rgba(46,49,49,1)'
    )
    fig = go.Figure(layout=layout)

    for time in sorted(schedule_times):
        x.append(time)
        y.append(len(schedule_times[time]))
    fig.add_trace(go.Scatter(x=x, y=y, name="Schedule"))
    fig.update_layout(title_text="Train Density over Time", xaxis_showgrid=True, yaxis_showgrid=False)
    fig.show()


# Currently running very slow...
def _delay_cause_level(schedule: TrainScheduleDict, malfunction: ExperimentMalfunction):
    malfunction_id = malfunction.agent_id
    schedule_times, schedule_ressources = _schedule_to_time_ressource_dicts(schedule)
    depth = 0
    depth_dict = {malfunction_id: 0}
    lower_depth_dict = _find_neighbours(schedule=schedule, schedule_ressources=schedule_ressources,
                                        agent_id=malfunction_id, depth=depth, known_neighbours=[malfunction_id],
                                        incident_time=malfunction.time_step)
    for lower_neighbour in lower_depth_dict:
        if lower_neighbour not in depth_dict:
            depth_dict[lower_neighbour] = lower_depth_dict[lower_neighbour]
        elif depth_dict[lower_neighbour] > lower_depth_dict[lower_neighbour]:
            depth_dict[lower_neighbour] = lower_depth_dict[lower_neighbour]

    return depth_dict


def _find_neighbours(schedule, schedule_ressources, agent_id, depth, known_neighbours, incident_time):
    depth_dict = {}
    lower_depth_dict = {}
    for time in sorted(schedule[agent_id]):
        if time < incident_time:
            continue
        waypoint = schedule[agent_id][time]
        ressource_times_full = sorted(schedule_ressources[waypoint.position])
        ressource_times = [i for i in ressource_times_full if i >= time]

        if len(ressource_times) < 1:
            break

        current_neighbour = schedule_ressources[waypoint.position][ressource_times[0]]
        while current_neighbour == agent_id:
            ressource_times.pop(0)
            if len(ressource_times) < 1:
                break
            current_neighbour = schedule_ressources[waypoint.position][ressource_times[0]]

        if current_neighbour not in known_neighbours:
            known_neighbours.append(current_neighbour)
            depth_dict[current_neighbour] = depth
            lower_depth_dict = _find_neighbours(schedule, schedule_ressources, current_neighbour, depth=depth + 1,
                                                known_neighbours=[current_neighbour], incident_time=ressource_times[0])

        for lower_neighbour in lower_depth_dict:
            if lower_neighbour not in depth_dict:
                depth_dict[lower_neighbour] = lower_depth_dict[lower_neighbour]
            elif depth_dict[lower_neighbour] > lower_depth_dict[lower_neighbour]:
                depth_dict[lower_neighbour] = lower_depth_dict[lower_neighbour]

    return depth_dict


def trajectories_from_resource_occupations_per_agent(
        resource_occupations_schedule: SortedResourceOccupationsPerAgentDict,
        resource_sorting: ResourceSorting,
        width: int) -> Trajectories:
    """

    Parameters
    ----------
    resource_occupations_schedule
    resource_sorting
    width

    Returns
    -------

    """
    schedule_trajectories: Trajectories = []
    for _, resource_ocupations in resource_occupations_schedule.items():
        train_time_path = []
        for resource_ocupation in resource_ocupations:
            position = coordinate_to_position(width, [resource_ocupation.resource])[0]
            # TODO dirty hack: add positions from re-scheduling to resource_sorting in the first place instead of workaround here!
            if position not in resource_sorting:
                resource_sorting[position] = len(resource_sorting)
            train_time_path.append((resource_sorting[position], resource_ocupation.interval.from_incl))
            train_time_path.append((resource_sorting[position], resource_ocupation.interval.to_excl))
            train_time_path.append((None, None))
        schedule_trajectories.append(train_time_path)
    return schedule_trajectories


# TODO SIM-537 remove trajectories_from_train_schedule_dict, use only trajectories_from_resource_occupations_per_agent
def trajectories_from_train_schedule_dict(
        train_schedule_dict: TrainScheduleDict,
        plotting_information: PlottingInformation,
) -> Trajectories:
    """

    Parameters
    ----------
    plotting_information
    train_schedule_dict

    Returns
    -------

    """
    resource_sorting = plotting_information.sorting
    schedule_trajectories: Trajectories = []
    width = plotting_information.grid_width
    for _, train_schedule in train_schedule_dict.items():
        train_time_path = []
        previous_waypoint = None
        entry_time = -1
        for time_step, waypoint in train_schedule.items():

            if previous_waypoint is None and waypoint is not None:
                entry_time = time_step
                previous_waypoint = waypoint
            elif previous_waypoint != waypoint:
                position = coordinate_to_position(width, [previous_waypoint.position])[0]
                # TODO dirty hack: add positions from re-scheduling to resource_sorting in the first place instead of workaround here!
                if position not in resource_sorting:
                    resource_sorting[position] = len(resource_sorting)

                train_time_path.append((resource_sorting[position], entry_time))
                train_time_path.append((resource_sorting[position], time_step))
                train_time_path.append((None, None))

                previous_waypoint = waypoint
                entry_time = time_step

        schedule_trajectories.append(train_time_path)
    return schedule_trajectories
