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
from pandas import DataFrame

from rsp.experiment_solvers.data_types import ExperimentMalfunction
from rsp.schedule_problem_description.analysis.route_dag_analysis import visualize_route_dag_constraints
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemEnum
from rsp.transmission_chains.transmission_chains import distance_matrix_from_tranmission_chains
from rsp.transmission_chains.transmission_chains import extract_transmission_chains_from_schedule
from rsp.utils.data_types import convert_pandas_series_experiment_results_analysis
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import LeftClosedInterval
from rsp.utils.data_types import ResourceOccupation
from rsp.utils.data_types import ScheduleAsResourceOccupations
from rsp.utils.data_types import SortedResourceOccupationsPerAgent
from rsp.utils.data_types_converters_and_validators import extract_resource_occupations
from rsp.utils.data_types_converters_and_validators import verify_schedule_as_resource_occupations
from rsp.utils.experiments import create_env_from_experiment_parameters
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.file_utils import check_create_folder
from rsp.utils.flatland_replay_utils import render_trainruns
from rsp.utils.global_constants import RELEASE_TIME
from rsp.utils.plotting_data_types import PlottingInformation
from rsp.utils.plotting_data_types import SchedulePlotting

Trajectories = Dict[int, List[Tuple[int, int]]]  # Int in the dict is the agent handle
SpaceTimeDifference = NamedTuple('Space_Time_Difference', [('changed_agents', Trajectories),
                                                           ('additional_information', Dict)])

# Information used for plotting time-resource-graphs: Sorting is dict mapping ressource to int value used to sort
# ressources for nice visualization

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


def extract_schedule_plotting(
        experiment_result: ExperimentResultsAnalysis) -> SchedulePlotting:
    """Extract the scheduling information from a experiment data for plotting.

    Parameters
    ----------
    experiment_result

    Returns
    -------
    """
    schedule = experiment_result.solution_full
    reschedule_full = experiment_result.solution_full_after_malfunction
    reschedule_delta = experiment_result.solution_delta_after_malfunction
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
    reschedule_delta_as_resource_occupations = extract_resource_occupations(
        schedule=reschedule_delta,
        release_time=RELEASE_TIME)
    verify_schedule_as_resource_occupations(schedule_as_resource_occupations=reschedule_delta_as_resource_occupations,
                                            release_time=RELEASE_TIME)
    plotting_information: PlottingInformation = extract_plotting_information(
        schedule_as_resource_occupations=schedule_as_resource_occupations,
        grid_depth=experiment_result.experiment_parameters.width)
    return SchedulePlotting(
        schedule_as_resource_occupations=schedule_as_resource_occupations,
        reschedule_full_as_resource_occupations=reschedule_full_as_resource_occupations,
        reschedule_delta_as_resource_occupations=reschedule_delta_as_resource_occupations,
        plotting_information=plotting_information,
        malfunction=experiment_result.malfunction
    )


def extract_plotting_information(
        schedule_as_resource_occupations: ScheduleAsResourceOccupations,
        grid_depth: int) -> PlottingInformation:
    """Extract plotting information.

    Parameters
    ----------
    schedule_as_resource_occupations:
    grid_depth
        Ranges of the window to be shown, used for consistent plotting

    Returns
    -------
    PlottingInformation
        The extracted plotting information.
    """
    sorted_index = 0
    max_time = 0
    max_ressource = 0
    sorting = {}

    for _, sorted_resource_occupations in sorted(schedule_as_resource_occupations.sorted_resource_occupations_per_agent.items()):
        for resource_occupation in sorted_resource_occupations:
            resource_occupation: ResourceOccupation = resource_occupation
            time = resource_occupation.interval.from_incl
            if time > max_time:
                max_time = time
            position = coordinate_to_position(grid_depth, [resource_occupation.resource])
            if position[0] not in sorting:
                sorting[position[0]] = sorted_index
                sorted_index += 1
            if sorted_index > max_ressource:
                max_ressource = sorted_index
    plotting_information = PlottingInformation(sorting=sorting, dimensions=(max_ressource, max_time), grid_width=grid_depth)
    return plotting_information


def time_windows_as_resource_occupations_per_agent(problem: ScheduleProblemDescription) -> SortedResourceOccupationsPerAgent:
    time_windows_per_agent = {}

    for agent_id, route_dag_constraints in problem.route_dag_constraints_dict.items():
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
        time_windows_per_agent[agent_id] = []
        for resource, earliest in earliest_resource.items():
            time_windows_per_agent[agent_id].append(ResourceOccupation(
                interval=LeftClosedInterval(earliest, latest_resource[resource] + RELEASE_TIME),
                resource=resource,
                agent_id=agent_id,
                # we aggregate over all directions
                direction=-1
            ))
    return time_windows_per_agent


def plot_time_window_resource_trajectories(
        experiment_result: ExperimentResultsAnalysis,
        plotting_information: PlottingInformation,
        show: bool = True):
    """Plot time-window -- resource diagram for all three problems.

    Parameters
    ----------
    experiment_result
    plotting_information
    show
    """
    ranges = (len(plotting_information.sorting),
              max(experiment_result.problem_full.max_episode_steps,
                  experiment_result.problem_full_after_malfunction.max_episode_steps,
                  experiment_result.problem_delta_after_malfunction.max_episode_steps))
    for title, (problem, malfunction) in {
        'Schedule': (experiment_result.problem_full, None),
        'Full Re-Schedule': (experiment_result.problem_full_after_malfunction, experiment_result.malfunction),
        'Delta Re-Schedule': (experiment_result.problem_delta_after_malfunction, experiment_result.malfunction)
    }.items():
        resource_occupations_schedule = time_windows_as_resource_occupations_per_agent(problem=problem)
        trajectories = trajectories_from_resource_occupations_per_agent(
            resource_occupations_schedule=resource_occupations_schedule,
            plotting_information=plotting_information)
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
            #  the position of each entry waypoint is the cell that will be in conflict
            (t0, (wp00, _), t1, (wp10, _)) = eval(sh)
            distance_matrix[wp00[0]] += 1
            distance_matrix[wp10[0]] += 1
        distance_matrix /= np.max(distance_matrix)
        fig = go.Figure(
            data=go.Heatmap(
                z=distance_matrix,
                colorscale="Hot"))
        fig.update_layout(
            title='Heatmap shared {}'.format(title),
            width=700,
            height=700
        )
        fig.update_yaxes(autorange="reversed")
        fig.show()


def plot_resource_time_diagrams(schedule_plotting: SchedulePlotting, with_diff: bool = True) -> Dict[int, bool]:
    """Method to draw resource-time diagrams in 2d.

    Parameters
    ----------
    schedule_plotting
    with_diff
        plot difference as well?

    Returns
    -------
        List of agent ids that changed between schedule an reschedule full
    """
    malfunction = schedule_plotting.malfunction
    plotting_information = schedule_plotting.plotting_information
    resource_occupations_schedule: SortedResourceOccupationsPerAgent = schedule_plotting.schedule_as_resource_occupations.sorted_resource_occupations_per_agent
    resource_occupations_reschedule_full: SortedResourceOccupationsPerAgent = \
        schedule_plotting.reschedule_full_as_resource_occupations.sorted_resource_occupations_per_agent
    resource_occupations_reschedule_delta: SortedResourceOccupationsPerAgent = \
        schedule_plotting.reschedule_delta_as_resource_occupations.sorted_resource_occupations_per_agent
    trajectories_schedule: Trajectories = trajectories_from_resource_occupations_per_agent(
        resource_occupations_schedule=resource_occupations_schedule,
        plotting_information=plotting_information)
    trajectories_reschedule_full: Trajectories = trajectories_from_resource_occupations_per_agent(
        resource_occupations_schedule=resource_occupations_reschedule_full,
        plotting_information=plotting_information)
    trajectories_reschedule_delta: Trajectories = trajectories_from_resource_occupations_per_agent(
        resource_occupations_schedule=resource_occupations_reschedule_delta,
        plotting_information=plotting_information)

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
    trajectories_influenced_agents, changed_agents_dict = get_difference_in_time_space_trajectories(
        trajectories_b=trajectories_schedule,
        trajectories_a=trajectories_reschedule_full)

    # Printing situation overview
    nb_changed_agents = sum([1 for changed in changed_agents_dict.values() if changed])
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

    return changed_agents_dict


def plot_time_resource_trajectories(
        title: str,
        trajectories: Trajectories,
        ranges: Tuple[int, int],
        additional_data: Dict = None,
        malfunction: ExperimentMalfunction = None,
        malfunction_wave: Trajectories = None,
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
        for idx, line in trajectories.items():
            if len(line) < 2:
                continue
            x, y = zip(*line)
            trace_color = PLOTLY_COLORLIST[int(idx % len(PLOTLY_COLORLIST))]

            fig.add_trace(go.Scattergl(
                x=x,
                y=y,
                mode='lines+markers',
                marker=dict(size=2, color=trace_color),
                line=dict(color=trace_color),
                name="Agent {}".format(idx),
                customdata=np.dstack([list_values[:][k][idx] for k in range(len(list_values[:]))])[0],
                hovertemplate=hovertemplate
            ))
    else:
        for idx, line in trajectories.items():
            if len(line) < 2:
                continue
            x, y = zip(*line)
            trace_color = PLOTLY_COLORLIST[int(idx % len(PLOTLY_COLORLIST))]

            fig.add_trace(
                go.Scattergl(x=x,
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
        fig.add_trace(go.Scattergl(x=x, y=y, name='malfunction start', line=dict(color='red')))
        y = [malfunction.time_step + malfunction.malfunction_duration, malfunction.time_step + malfunction.malfunction_duration]
        fig.add_trace(go.Scattergl(x=x, y=y, name='malfunction end', line=dict(color='red', dash='dash')))
    if malfunction_wave is not None:
        x, y = zip(*list(malfunction_wave.values())[0])
        fig.add_trace(
            go.Scattergl(x=x,
                         y=y,
                         mode='lines+markers',
                         marker=dict(size=2, color="red"),
                         line=dict(color="red"),
                         name="Malfunction Wave",
                         hovertemplate=hovertemplate
                         ))
    fig.update_layout(title_text=title, xaxis_showgrid=True, yaxis_showgrid=False)
    fig.update_xaxes(title="Sorted resources", range=[0, ranges[0]])
    fig.update_yaxes(title="Time", range=[ranges[1], 0])
    if show:
        fig.show()


def plot_histogram_from_delay_data(experiment_results: ExperimentResultsAnalysis):
    """
    Plot a histogram of the delay of agents in the full and delta reschedule compared to the schedule
    Parameters
    ----------
    experiment_data_frame
    experiment_id

    Returns
    -------

    """

    lateness_full_after_malfunction = experiment_results.lateness_full_after_malfunction
    lateness_delta_after_malfunction = experiment_results.lateness_delta_after_malfunction
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


def render_flatland_env(data_folder: str, experiment_data: ExperimentResultsAnalysis, experiment_id: int,
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

    # Generate environment for rendering
    rail_env = create_env_from_experiment_parameters(experiment_data.experiment_parameters)
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
            render_trainruns(data_folder=output_folder,
                             experiment_id=experiment_data.experiment_id,
                             title=title,
                             rail_env=rail_env,
                             trainruns=experiment_data.solution_full,
                             malfunction=experiment_data.malfunction,
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
            render_trainruns(data_folder=output_folder,
                             experiment_id=experiment_data.experiment_id,
                             malfunction=experiment_data.malfunction,
                             title=title,
                             rail_env=rail_env,
                             trainruns=experiment_data.solution_full_after_malfunction,
                             convert_to_mpeg=True)
    else:
        video_src_reschedule = None

    return Path(video_src_schedule), Path(video_src_reschedule)


def get_difference_in_time_space_trajectories(trajectories_a: Trajectories, trajectories_b: Trajectories) -> SpaceTimeDifference:
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
    traces_influenced_agents: Trajectories = {}
    additional_information = dict()
    for idx, trainrun in trajectories_a.items():
        trainrun_difference = []
        for waypoint in trainrun:
            if waypoint not in trajectories_b[idx]:
                if len(trainrun_difference) > 0 and waypoint[0] != trainrun_difference[-1][0]:
                    trainrun_difference.append((None, None))
                trainrun_difference.append(waypoint)

        if len(trainrun_difference) > 0:
            traces_influenced_agents[idx] = trainrun_difference
            additional_information.update({idx: True})
        else:
            traces_influenced_agents[idx] = [(None, None)]
            additional_information.update({idx: False})
    space_time_difference = SpaceTimeDifference(changed_agents=traces_influenced_agents,
                                                additional_information=additional_information)
    return space_time_difference


def plot_schedule_metrics(schedule_plotting: SchedulePlotting, lateness_delta_after_malfunction: Dict[int, int]):
    """

    Parameters
    ----------
    schedule_plotting
    lateness_delta_after_malfunction
    """
    # Plot Density over time
    plot_time_density(schedule_plotting.schedule_as_resource_occupations)

    # Plot Occupancy over space
    plot_resource_occupation_heat_map(
        schedule_as_resource_occupations=schedule_plotting.schedule_as_resource_occupations,
        plotting_information=schedule_plotting.plotting_information,
        title_suffix='Schedule'
    )
    plot_resource_occupation_heat_map(
        schedule_as_resource_occupations=schedule_plotting.reschedule_delta_as_resource_occupations,
        plotting_information=schedule_plotting.plotting_information,
        title_suffix='Re-Schedule Delta'
    )

    # Plot Delay propagation
    transmission_chains = extract_transmission_chains_from_schedule(schedule_plotting)
    _, _, minimal_depth, _ = distance_matrix_from_tranmission_chains(
        number_of_trains=len(schedule_plotting.schedule_as_resource_occupations.sorted_resource_occupations_per_agent),
        transmission_chains=transmission_chains
    )
    plot_delay_propagation_2d(plotting_data=schedule_plotting,
                              delay_information=lateness_delta_after_malfunction,
                              depth_dict=minimal_depth)


def plot_resource_occupation_heat_map(
        schedule_as_resource_occupations: ScheduleAsResourceOccupations,
        plotting_information: PlottingInformation,
        title_suffix: str = ''
):
    """Plot agent density over resource.

    Parameters
    ----------
    schedule_as_resource_occupations
    plotting_information : PlottingInformation

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

    for resource, resource_occupations in schedule_as_resource_occupations.sorted_resource_occupations_per_resource.items():
        x.append(resource.column)
        y.append(resource.row)
        size.append((len(resource_occupations)))
        times = np.array(sorted([ro.interval.from_incl for ro in resource_occupations]))
        if len(times) > 1:
            mean_temp_dist = np.mean(np.clip(times[1:] - times[:-1], 0, 50))
            color.append(mean_temp_dist)
        else:
            color.append(50)
    fig.add_trace(go.Scattergl(x=x,
                               y=y,
                               mode='markers',
                               name="Schedule",
                               marker=dict(
                                   color=size,
                                   symbol='square',
                                   showscale=True,
                                   reversescale=False,
                                   colorbar=dict(
                                       title="Colorbar"
                                   ), colorscale="Hot"
                               )))
    fig.update_layout(title_text=f"Train Density at Resources {title_suffix}",
                      autosize=False,
                      width=1000,
                      height=1000)

    fig.update_yaxes(zeroline=False, showgrid=True, range=[plotting_information.grid_width, 0], tick0=-0.5, dtick=1, gridcolor='Grey')
    fig.update_xaxes(zeroline=False, showgrid=True, range=[0, plotting_information.grid_width], tick0=-0.5, dtick=1, gridcolor='Grey')

    fig.show()


def plot_delay_propagation_2d(
        plotting_data: SchedulePlotting,
        delay_information: Dict[int, int],
        depth_dict: Dict[int, int],
        changed_agents: Optional[Dict[int, bool]] = None):
    """
    Plot agent delay over ressource.
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

    # Sort agents according to influence depth for plotting only plot disturbed agents
    agents = []
    if changed_agents is None:
        for agent, _depth in sorted(depth_dict.items(), key=lambda item: item[1], reverse=True):
            if agent in plotting_data.schedule_as_resource_occupations.sorted_resource_occupations_per_agent:
                agents.append(agent)
    else:
        for agent, _depth in sorted(depth_dict.items(), key=lambda item: item[1], reverse=True):
            if agent in plotting_data.schedule_as_resource_occupations.sorted_resource_occupations_per_agent and changed_agents[agent]:
                agents.append(agent)
    # Add the malfunction source agent
    agents.append(plotting_data.malfunction.agent_id)

    # Plot only after the malfunciton happend
    malfunction_time = plotting_data.malfunction.time_step
    # Plot traces of agents
    for agent_id in agents:
        x = []
        y = []
        size = []
        marker = []
        times = []
        delay = []
        conflict_depth = []
        for resource_occupation in plotting_data.schedule_as_resource_occupations.sorted_resource_occupations_per_agent[agent_id]:
            time = resource_occupation.interval.from_incl
            if time < malfunction_time:
                continue
            malfunction_resource = resource_occupation.resource
            x.append(malfunction_resource[1])
            y.append(malfunction_resource[0])
            size.append(max(10, delay_information[agent_id]))
            marker.append(MARKER_LIST[int(np.clip(resource_occupation.direction, 0, 3))])
            times.append(time)
            delay.append(delay_information[agent_id])
            if agent_id in depth_dict:
                conflict_depth.append(depth_dict[agent_id])
            else:
                conflict_depth.append("None")
        if agent_id in depth_dict:
            color = DEPTH_COLOR[int(np.clip(depth_dict[agent_id], 0, 5))]
        else:
            color = "red"
        fig.add_trace(go.Scattergl(x=x,
                                   y=y,
                                   mode='markers',
                                   name="Train {}".format(agent_id),
                                   marker_symbol=marker,
                                   customdata=list(zip(times, delay, conflict_depth)),
                                   marker_size=size,
                                   marker_opacity=0.2,
                                   marker_color=color,
                                   marker_line_color=color,
                                   hovertemplate="Time:\t%{customdata[0]}<br>" +
                                                 "Delay:\t%{customdata[1]}<br>" +
                                                 "Influence depth:\t%{customdata[2]}"
                                   ))
    # Plot malfunction
    malfunction_resource = plotting_data.schedule_as_resource_occupations.resource_occupations_per_agent_and_time_step[
        (plotting_data.malfunction.agent_id, plotting_data.malfunction.time_step)][
        0].resource
    fig.add_trace(go.Scattergl(x=[malfunction_resource[1]],
                               y=[malfunction_resource[0]],
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

    fig.update_yaxes(zeroline=False, showgrid=True, range=[plotting_data.plotting_information.grid_width, 0], tick0=-0.5, dtick=1, gridcolor='Grey')
    fig.update_xaxes(zeroline=False, showgrid=True, range=[0, plotting_data.plotting_information.grid_width], tick0=-0.5, dtick=1, gridcolor='Grey')

    fig.show()


def plot_time_density(schedule_as_resource_occupations: ScheduleAsResourceOccupations):
    """Plot agent density over time.

    Parameters
    ----------
    schedule_as_resource_occupations

    Returns
    -------
    """
    x = []
    y = []
    layout = go.Layout(
        plot_bgcolor='rgba(46,49,49,1)'
    )
    fig = go.Figure(layout=layout)

    schedule_times = {}
    for _, time_step in schedule_as_resource_occupations.resource_occupations_per_agent_and_time_step.keys():
        schedule_times[time_step] = schedule_times.setdefault(time_step, 0) + 1

    for time_step, nb_agents in sorted(schedule_times.items()):
        x.append(time_step)
        y.append(nb_agents)
    fig.add_trace(go.Scattergl(x=x, y=y, name="Schedule"))
    fig.update_layout(title_text="Train Density over Time", xaxis_showgrid=True, yaxis_showgrid=False)
    fig.update_xaxes(title="Time [steps]")
    fig.update_yaxes(title="Active Agents")
    fig.show()


def trajectories_from_resource_occupations_per_agent(
        resource_occupations_schedule: SortedResourceOccupationsPerAgent,
        plotting_information: PlottingInformation
) -> Trajectories:
    """

    Parameters
    ----------
    resource_occupations_schedule

    Returns
    -------

    """
    resource_sorting = plotting_information.sorting
    width = plotting_information.grid_width
    schedule_trajectories: Trajectories = {}
    for idx, resource_ocupations in resource_occupations_schedule.items():
        train_time_path = []
        for resource_ocupation in resource_ocupations:
            position = coordinate_to_position(width, [resource_ocupation.resource])[0]
            # TODO dirty hack: add positions from re-scheduling to resource_sorting in the first place instead of workaround here!
            if position not in resource_sorting:
                resource_sorting[position] = len(resource_sorting)
            train_time_path.append((resource_sorting[position], resource_ocupation.interval.from_incl))
            train_time_path.append((resource_sorting[position], resource_ocupation.interval.to_excl))
            train_time_path.append((None, None))
        schedule_trajectories[idx] = train_time_path
    return schedule_trajectories
