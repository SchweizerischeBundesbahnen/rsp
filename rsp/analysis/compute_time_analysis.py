"""Rendering methods to use with jupyter notebooks."""
import os.path
import re
from copy import copy
from pathlib import Path
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from _plotly_utils.colors.qualitative import Plotly
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.core.grid.grid_utils import position_to_coordinate
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import Waypoint
from pandas import DataFrame

from rsp.schedule_problem_description.analysis.route_dag_analysis import visualize_route_dag_constraints
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemEnum
from rsp.utils.data_types import after_malfunction_scopes
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import LeftClosedInterval
from rsp.utils.data_types import Resource
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

Trajectory = List[Tuple[Optional[int], Optional[int]]]  # Time and sorted ressource, optional
Trajectories = Dict[int, Trajectory]  # Int in the dict is the agent handle
SpaceTimeDifference = NamedTuple('Space_Time_Difference', [('changed_agents', Trajectories),
                                                           ('additional_information', Dict)])

# Information used for plotting time-resource-graphs: Sorting is dict mapping ressource to int value used to sort
# resources for nice visualization

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


def plot_computational_times(
        experiment_data: DataFrame,
        axis_of_interest: str,
        columns_of_interest: List[str],
        experiment_data_baseline: Optional[DataFrame] = None,
        experiment_data_baseline_suffix: Optional[str] = '_baseline',
        experiment_data_suffix: Optional[str] = '',
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

    Returns
    -------
    """
    traces = [(axis_of_interest, column) for column in columns_of_interest]
    # prevent too long file names
    pdf_file = f'{file_name_prefix}_{axis_of_interest}__' + ('_'.join(columns_of_interest))[0:15] + '.pdf'
    plot_computional_times_from_traces(experiment_data=experiment_data,
                                       experiment_data_baseline=experiment_data_baseline,
                                       experiment_data_baseline_suffix=experiment_data_baseline_suffix,
                                       experiment_data_suffix=experiment_data_suffix,
                                       traces=traces,
                                       output_folder=output_folder,
                                       x_axis_title=axis_of_interest,
                                       y_axis_title=y_axis_title,
                                       pdf_file=pdf_file,
                                       title=f"{title} {axis_of_interest}",
                                       color_offset=color_offset)


# TODO SIM-672 duplicate?
def plot_computional_times_from_traces(
        experiment_data: DataFrame,
        traces: List[Tuple[str, str]],
        x_axis_title: str,
        y_axis_title: str = "Time[s]",
        experiment_data_baseline: Optional[DataFrame] = None,
        experiment_data_baseline_suffix: Optional[str] = '_baseline',
        experiment_data_suffix: Optional[str] = '',
        output_folder: Optional[str] = None,
        pdf_file: Optional[str] = None,
        title: str = "Computational Times",
        color_offset: int = 0
):
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
    for index, (axis_of_interest, column) in enumerate(traces):
        fig.add_trace(go.Box(x=experiment_data[axis_of_interest],
                             y=experiment_data[column],
                             name=str(column) + experiment_data_suffix,
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
        if experiment_data_baseline is not None:
            fig.add_trace(go.Box(x=experiment_data_baseline[axis_of_interest],
                                 y=experiment_data_baseline[column],
                                 name=column + experiment_data_baseline_suffix,
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
                                 marker_color=Plotly[(index + color_offset) % len(Plotly)]))
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
        axis_of_interest_suffix: str = ""
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
    nb_x_values = len(experiment_data[axis_of_interest].value_counts())

    min_value = experiment_data[axis_of_interest].min()
    max_value = experiment_data[axis_of_interest].max()
    # TODO SIM-672 configurable?
    nb_bins = 10
    inc = (max_value - min_value) / nb_bins
    axis_of_interest_binned = axis_of_interest + "_binned"
    experiment_data.sort_values(by=axis_of_interest, inplace=True)

    # TODO SIM-672 configurable?
    binned = nb_x_values >= 10 and axis_of_interest != 'experiment_id'
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
                # TODO SIM-672 mean?
                customdata=np.dstack((experiment_data['n_agents'],
                                      experiment_data['size'],
                                      experiment_data['solver_statistics_times_total_full'],
                                      experiment_data['solver_statistics_times_total_full_after_malfunction'],
                                      experiment_data['solver_statistics_times_total_delta_perfect_after_malfunction'],
                                      experiment_data['solver_statistics_times_total_delta_naive_after_malfunction'],
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


def time_windows_as_resource_occupations_per_agent(problem: ScheduleProblemDescription) -> SortedResourceOccupationsPerAgent:
    time_windows_per_agent = {}

    for agent_id, route_dag_constraints in problem.route_dag_constraints_dict.items():
        time_windows_per_agent[agent_id] = []
        for waypoint, earliest in route_dag_constraints.earliest.items():
            waypoint: Waypoint = waypoint
            resource = waypoint.position
            latest = route_dag_constraints.latest[waypoint]
            time_windows_per_agent[agent_id].append(ResourceOccupation(
                interval=LeftClosedInterval(earliest, latest + RELEASE_TIME),
                resource=resource,
                agent_id=agent_id,
                direction=waypoint.direction
            ))
    return time_windows_per_agent


def plot_time_window_resource_trajectories(
        experiment_result: ExperimentResultsAnalysis,
        schedule_plotting: SchedulePlotting,
        show: bool = True):
    """Plot time-window -- resource diagram for all three problems.

    Parameters
    ----------
    experiment_result
    schedule_plotting
    show
    """
    for title, problem in {
        'Schedule': experiment_result.problem_full,
        'Full Re-Schedule': experiment_result.problem_full_after_malfunction,
        'scope perfect re-schedule': experiment_result.problem_delta_perfect_after_malfunction
    }.items():
        resource_occupations_schedule = time_windows_as_resource_occupations_per_agent(problem=problem)
        trajectories = trajectories_from_resource_occupations_per_agent(
            resource_occupations_schedule=resource_occupations_schedule,
            plotting_information=schedule_plotting.plotting_information)
        plot_time_resource_trajectories(trajectories=trajectories, title=title, schedule_plotting=schedule_plotting)


# TODO SIM-674 should be covered by testing, called from notebooks only
def plot_shared_heatmap(schedule_plotting: SchedulePlotting, experiment_result: ExperimentResultsAnalysis):
    """Plot a heat map of how many shareds are on the resources.

    Parameters
    ----------
    experiment_result
    show
    """
    layout = go.Layout(
        plot_bgcolor=GREY_BACKGROUND_COLOR
    )
    fig = go.Figure(layout=layout)
    plotting_information = schedule_plotting.plotting_information
    for title, result in {
        'Schedule': experiment_result.results_full,
        'Full Re-Schedule': experiment_result.results_full_after_malfunction,
        'scope perfect re-schedule': experiment_result.results_delta_perfect_after_malfunction
    }.items():
        shared = list(filter(lambda s: s.startswith('shared'), result.solver_result))
        shared_per_resource = {}
        for sh in shared:
            sh = sh.replace('shared', '')
            sh = re.sub('t[0-9]+', '"XXX"', sh)
            #  the position of each entry waypoint is the cell that will be in conflict
            (t0, (wp00, _), t1, (wp10, _)) = eval(sh)
            if wp00[0] not in shared_per_resource:
                shared_per_resource[wp00[0]] = 1
            else:
                shared_per_resource[wp00[0]] += 1
            if wp10[0] not in shared_per_resource:
                shared_per_resource[wp10[0]] = 1
            else:
                shared_per_resource[wp10[0]] += 1
        x = []
        y = []
        z = []
        for resource, occupancy in shared_per_resource.items():
            x.append(resource[1])
            y.append(resource[0])
            z.append(occupancy)

        fig.add_trace(go.Scattergl(
            x=x,
            y=y,
            mode='markers',
            name=title,
            marker=dict(
                color=z,
                size=15,
                symbol='square',
                showscale=True,
                reversescale=False,
                colorbar=dict(
                    title="Number of shared", len=0.75
                ), colorscale="Hot"
            )))

    fig.update_layout(title_text="Shared Resources",
                      autosize=False,
                      width=1000,
                      height=1000)
    fig.update_xaxes(zeroline=False, showgrid=True, range=[0, plotting_information.grid_width], tick0=-0.5, dtick=1, gridcolor='Grey')
    fig.update_yaxes(zeroline=False, showgrid=True, range=[plotting_information.grid_width, 0], tick0=-0.5, dtick=1, gridcolor='Grey')
    fig.show()


# TODO SIM-674 should be covered by testing, called from notebooks only
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
    plotting_information = schedule_plotting.plotting_information
    resource_occupations_schedule: SortedResourceOccupationsPerAgent = schedule_plotting.schedule_as_resource_occupations.sorted_resource_occupations_per_agent
    resource_occupations_reschedule_full: SortedResourceOccupationsPerAgent = \
        schedule_plotting.reschedule_full_as_resource_occupations.sorted_resource_occupations_per_agent
    resource_occupations_reschedule_delta_perfect: SortedResourceOccupationsPerAgent = \
        schedule_plotting.reschedule_delta_perfect_as_resource_occupations.sorted_resource_occupations_per_agent
    trajectories_schedule: Trajectories = trajectories_from_resource_occupations_per_agent(
        resource_occupations_schedule=resource_occupations_schedule,
        plotting_information=plotting_information)
    trajectories_reschedule_full: Trajectories = trajectories_from_resource_occupations_per_agent(
        resource_occupations_schedule=resource_occupations_reschedule_full,
        plotting_information=plotting_information)
    trajectories_reschedule_delta_perfect: Trajectories = trajectories_from_resource_occupations_per_agent(
        resource_occupations_schedule=resource_occupations_reschedule_delta_perfect,
        plotting_information=plotting_information)

    # Plot Schedule
    plot_time_resource_trajectories(
        title='Schedule',
        schedule_plotting=schedule_plotting,
        trajectories=trajectories_schedule)

    # Plot Reschedule Full only plot this if there is an actual difference between schedule and reschedule
    trajectories_influenced_agents, changed_agents_dict = get_difference_in_time_space_trajectories(
        target_trajectories=trajectories_schedule,
        base_trajectories=trajectories_reschedule_full)

    # Printing situation overview

    print_situation_overview(schedule_plotting=schedule_plotting, changed_agents_dict=changed_agents_dict)

    # Plot Reschedule Full only if svomething has changed
    nb_changed_agents = sum([1 for changed in changed_agents_dict.values() if changed])
    if nb_changed_agents > 0:
        plot_time_resource_trajectories(
            trajectories=trajectories_reschedule_full,
            title='Full Reschedule',
            schedule_plotting=schedule_plotting
        )

    # Plot Reschedule Delta Perfect with additional data
    plot_time_resource_trajectories(
        title='Delta Perfect Reschedule', schedule_plotting=schedule_plotting,
        trajectories=trajectories_reschedule_delta_perfect,
    )

    # Plot difference if asked for
    if with_diff:
        plot_time_resource_trajectories(
            trajectories=trajectories_influenced_agents,
            title='Changed Agents',
            schedule_plotting=schedule_plotting
        )

    return changed_agents_dict


def print_situation_overview(schedule_plotting: SchedulePlotting, changed_agents_dict: Dict):
    # Printing situation overview
    malfunction = schedule_plotting.malfunction
    resource_occupations_schedule: SortedResourceOccupationsPerAgent = schedule_plotting.schedule_as_resource_occupations.sorted_resource_occupations_per_agent
    resource_occupations_reschedule_delta_perfect: SortedResourceOccupationsPerAgent = \
        schedule_plotting.reschedule_delta_perfect_as_resource_occupations.sorted_resource_occupations_per_agent

    nb_changed_agents = sum([1 for changed in changed_agents_dict.values() if changed])
    total_lateness = sum(
        max(sorted_resource_occupations_reschedule_delta_perfect[-1].interval.to_excl - resource_occupations_schedule[agent_id][-1].interval.to_excl, 0)
        for agent_id, sorted_resource_occupations_reschedule_delta_perfect in resource_occupations_reschedule_delta_perfect.items()
    )
    print(
        "Agent nr.{} has a malfunction at time {} for {} s and influenced {} other agents. Total delay = {}.".format(
            malfunction.agent_id,
            malfunction.time_step,
            malfunction.malfunction_duration,
            nb_changed_agents,
            total_lateness))


def plot_time_resource_trajectories(
        title: str,
        trajectories: Trajectories,
        schedule_plotting: SchedulePlotting,
        additional_data: Dict = None,
        malfunction_wave: Trajectories = None,
        show: bool = True
):
    """
    Plot the time-resource-diagram with additional data for each train
    Parameters
    ----------

    schedule_plotting
    malfunction_wave
    title: str
        Title of the plot
    trajectories:
        Data to be shown, contains tuples for all occupied resources during train run
    additional_data
        Dict containing additional data. Each additional data must have the same dimensins as time_resource_data

    show: bool

    Returns
    -------

    """
    layout = go.Layout(
        plot_bgcolor=GREY_BACKGROUND_COLOR
    )
    fig = go.Figure(layout=layout)
    ranges = schedule_plotting.plotting_information.dimensions
    malfunction = schedule_plotting.malfunction
    ticks = [position_to_coordinate(schedule_plotting.plotting_information.grid_width, [key])[0]
             for key in schedule_plotting.plotting_information.sorting.keys()]

    # Get keys and information to add to hover data
    hovertemplate = '<b>Resource ID:<b> %{x}<br>' + '<b>Time:<b> %{y}<br>'
    if additional_data is not None:
        list_keys = [k for k in additional_data]
        list_values = [v for v in additional_data.values()]
        # Build hovertemplate
        for idx, data_point in enumerate(list_keys):
            hovertemplate += '<b>' + str(data_point) + '</b>: %{{customdata[{}]}}<br>'.format(idx)
        for idx, line in trajectories.items():
            # Don't plot trains with no paths --> this is just to make plots more readable
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
            # Don't plot trains with no paths --> this is just to make plots more readable
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
        x, y = zip(*list(malfunction_wave[0].values())[0])
        fig.add_trace(
            go.Scattergl(x=x,
                         y=y,
                         mode='lines+markers',
                         marker=dict(size=2, color="red"),
                         line=dict(color="red"),
                         name="True Positives",
                         hovertemplate=hovertemplate
                         ))
        x, y = zip(*list(malfunction_wave[1].values())[0])
        fig.add_trace(
            go.Scattergl(x=x,
                         y=y,
                         mode='lines+markers',
                         marker=dict(size=2, color="yellow"),
                         line=dict(color="yellow"),
                         name="False Positives",
                         hovertemplate=hovertemplate
                         ))
    fig.update_layout(title_text=title, xaxis_showgrid=False, yaxis_showgrid=False,
                      xaxis=dict(
                          tickmode='array',
                          tickvals=np.arange(len(ticks)),
                          ticktext=ticks,
                          tickangle=270))
    fig.update_xaxes(title="Resource Coordinates", range=[0, ranges[0]])

    fig.update_yaxes(title="Time", range=[ranges[1], 0])
    if show:
        fig.show()


# TODO SIM-674 should be covered by testing, called from notebooks only
def plot_histogram_from_delay_data(experiment_results: ExperimentResultsAnalysis):
    """
    Plot a histogram of the delay of agents in the full and delta perfect reschedule compared to the schedule
    Parameters
    ----------
    experiment_data_frame
    experiment_id

    Returns
    -------

    """

    fig = go.Figure()
    for scope in after_malfunction_scopes:
        fig.add_trace(go.Histogram(x=[v for v in experiment_results._asdict()[f'lateness_{scope}'].values()], name=f'results_{scope}'))
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    fig.update_layout(title_text="Delay distributions")
    fig.update_xaxes(title="Delay [s]")

    fig.show()


def plot_total_lateness(experiment_results: ExperimentResultsAnalysis):
    """
    Plot a histogram of the delay of agents in the full and delta perfect reschedule compared to the schedule
    Parameters
    ----------
    experiment_data_frame
    experiment_id

    Returns
    -------

    """
    fig = go.Figure()
    for scope in after_malfunction_scopes:
        fig.add_trace(go.Bar(x=[f'effective_costs_{scope}'], y=[experiment_results._asdict()[f'effective_costs_{scope}']], name=f'effective_costs_{scope}'))
        fig.add_trace(go.Bar(x=[f'total_lateness_{scope}'], y=[experiment_results._asdict()[f'total_lateness_{scope}']], name=f'total_lateness_{scope}'))
        fig.add_trace(go.Bar(x=[f'effective_total_costs_from_route_section_penalties_{scope}'],
                             y=[experiment_results._asdict()[f'effective_total_costs_from_route_section_penalties_{scope}']],
                             name=f'effective_total_costs_from_route_section_penalties_{scope}'))
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    fig.update_layout(title_text="Total delay and Solver objective")
    fig.update_yaxes(title="discrete time steps [-] / weighted sum [-]")

    fig.show()


def plot_agent_specific_delay(experiment_results: ExperimentResultsAnalysis):
    """
    Plot a histogram of the delay of agents in the full and reschedule delta perfect compared to the schedule
    Parameters
    ----------
    experiment_data_frame
    experiment_id

    Returns
    -------

    """
    fig = go.Figure()
    for scope in after_malfunction_scopes:
        # TODO SIM-672 distinguish lateness and weighted lateness?
        d = {}
        for dim in ['lateness', 'effective_costs_from_route_section_penalties']:
            values = list(experiment_results._asdict()[f'{dim}_{scope}'].values())
            d[dim] = sum(values)
            fig.add_trace(go.Bar(x=np.arange(len(values)), y=values, name=f'{dim}_{scope}'))
    fig.update_traces(opacity=0.75)
    fig.update_layout(title_text="Delay per Train")
    fig.update_xaxes(title="Train ID")
    fig.update_yaxes(title="Delay in Seconds")

    fig.show()


def plot_changed_agents(experiment_results: ExperimentResultsAnalysis):
    """
    Plot a histogram of the delay of agents in the full and reschedule delta perfect compared to the schedule
    Parameters
    ----------
    experiment_data_frame
    experiment_id

    Returns
    -------

    """
    fig = go.Figure()
    schedule_trainruns_dict = experiment_results.results_full.trainruns_dict
    for scope in after_malfunction_scopes:
        reschedule_trainruns_dict = experiment_results._asdict()[f'results_{scope}'].trainruns_dict
        values = [
            1.0 if set(schedule_trainruns_dict[agent_id]) != set(trainrun_reschedule) else 0.0
            for agent_id, trainrun_reschedule in reschedule_trainruns_dict.items()

        ]
        fig.add_trace(go.Bar(x=np.arange(len(values)), y=values, name=f'results_{scope}'))
    fig.update_traces(opacity=0.75)
    fig.update_layout(title_text="Changed per Train")
    fig.update_xaxes(title="Train ID")
    fig.update_yaxes(title="Changed 1.0=yes, 0.0=no [-]")

    fig.show()

    fig = go.Figure()
    schedule_trainruns_dict = experiment_results.results_full.trainruns_dict
    for scope in after_malfunction_scopes:
        reschedule_trainruns_dict = experiment_results._asdict()[f'results_{scope}'].trainruns_dict
        values = [
            1.0
            if ({trainrun_waypoint.waypoint for trainrun_waypoint in schedule_trainruns_dict[agent_id]} !=
                {trainrun_waypoint.waypoint for trainrun_waypoint in trainrun_reschedule})
            else 0.0
            for agent_id, trainrun_reschedule in reschedule_trainruns_dict.items()
        ]
        fig.add_trace(go.Bar(x=np.arange(len(values)), y=values, name=f'results_{scope}'))
    fig.update_traces(opacity=0.75)
    fig.update_layout(title_text="Changed routes per Train")
    fig.update_xaxes(title="Train ID")
    fig.update_yaxes(title="Changed 1.0=yes, 0.0=no [-]")

    fig.show()


def plot_route_dag(experiment_results_analysis: ExperimentResultsAnalysis,
                   agent_id: int,
                   suffix_of_constraints_to_visualize: ScheduleProblemEnum,
                   save: bool = False
                   ):
    train_runs_full: TrainrunDict = experiment_results_analysis.solution_full
    train_runs_full_after_malfunction: TrainrunDict = experiment_results_analysis.solution_full_after_malfunction
    train_runs_delta_perfect_after_malfunction: TrainrunDict = experiment_results_analysis.solution_delta_perfect_after_malfunction
    train_run_full: Trainrun = train_runs_full[agent_id]
    train_run_full_after_malfunction: Trainrun = train_runs_full_after_malfunction[agent_id]
    train_run_delta_perfect_after_malfunction: Trainrun = train_runs_delta_perfect_after_malfunction[agent_id]
    problem_schedule: ScheduleProblemDescription = experiment_results_analysis.problem_full
    problem_rsp_full: ScheduleProblemDescription = experiment_results_analysis.problem_full_after_malfunction
    problem_rsp_reduced_scope_perfect: ScheduleProblemDescription = experiment_results_analysis.problem_delta_perfect_after_malfunction
    # TODO hacky, we should take the topo_dict from infrastructure maybe?
    topo = experiment_results_analysis.problem_full_after_malfunction.topo_dict[agent_id]

    config = {
        ScheduleProblemEnum.PROBLEM_SCHEDULE: [
            problem_schedule,
            f'Schedule RouteDAG for agent {agent_id} in experiment {experiment_results_analysis.experiment_id}',
            train_run_full],
        ScheduleProblemEnum.PROBLEM_RSP_FULL: [
            problem_rsp_full,
            f'Full Reschedule RouteDAG for agent {agent_id} in experiment {experiment_results_analysis.experiment_id}',
            train_run_full_after_malfunction],
        ScheduleProblemEnum.PROBLEM_RSP_REDUCED_SCOPE: [
            problem_rsp_reduced_scope_perfect,
            f'Delta Perfect Reschedule RouteDAG for agent {agent_id} in experiment {experiment_results_analysis.experiment_id}',
            train_run_delta_perfect_after_malfunction],
    }

    problem_to_visualize, title, trainrun_to_visualize = config[suffix_of_constraints_to_visualize]

    visualize_route_dag_constraints(
        topo=topo,
        train_run_full=train_run_full,
        train_run_full_after_malfunction=train_run_full_after_malfunction,
        train_run_delta_perfect_after_malfunction=train_run_delta_perfect_after_malfunction,
        constraints_to_visualize=problem_to_visualize.route_dag_constraints_dict[agent_id],
        trainrun_to_visualize=trainrun_to_visualize,
        vertex_lateness={},
        effective_costs_from_route_section_penalties_per_edge={},
        route_section_penalties=problem_to_visualize.route_section_penalties[agent_id],
        title=title,
        file_name=(
            f"experiment_{experiment_results_analysis.experiment_id:04d}_agent_{agent_id}_route_graph_schedule.pdf"
            if save else None)
    )


# TODO SIM-674 should be covered by testing, called from notebooks only
def render_flatland_env(data_folder: str,
                        experiment_data: ExperimentResultsAnalysis,
                        experiment_id: int,
                        render_schedule: bool = True,
                        render_reschedule: bool = True):
    """
    Method to render the environment for visual inspection
    Parameters
    ----------
render_flatland_env
    data_folder: str
        Folder name to store and load images from
    experiment_data: ExperimentResultsAnalysis
        experiment data used for visualization
    experiment_id: int
        ID of experiment we like to visualize
    render_reschedule
    render_schedule

    Returns
    -------
    File paths to generated videos to render in the notebook
    """

    # Generate environment for rendering
    rail_env = create_env_from_experiment_parameters(experiment_data.experiment_parameters.infra_parameters)

    # Generate aggregated visualization
    output_folder = f'{data_folder}/{EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME}/'
    video_src_schedule = None
    video_src_reschedule = None

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
                             convert_to_mpeg=True)

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

    return Path(video_src_schedule) if render_schedule else None, Path(video_src_reschedule) if render_reschedule else None


def explode_trajectories(trajectories: Trajectories) -> Dict[int, Set[Tuple[int, int]]]:
    """Return for each agent the pairs of `(resource,time)` corresponding to
    the trajectories.

    Parameters
    ----------
    trajectories

    Returns
    -------
    Dict indexed by `agent_id`, containing `(resource,time_step)` pairs.
    """
    exploded = {agent_id: set() for agent_id in trajectories.keys()}
    for agent_id, trajectory in trajectories.items():
        # ensure we have triplets (resource,from_time), (resource,to_time), (None,None)
        assert len(trajectory) % 3 == 0
        while len(trajectory) > 0:
            (resource, from_time), (resource, to_time), (_, _) = trajectory[:3]
            for time in range(from_time, to_time + 1):
                exploded[agent_id].add((resource, time))
            trajectory = trajectory[3:]
    return exploded


def get_difference_in_time_space_trajectories(base_trajectories: Trajectories, target_trajectories: Trajectories) -> SpaceTimeDifference:
    """
    Compute the difference between schedules and return in plot ready format (in base but not in target)
    Parameters
    ----------
    base_trajectories
    target_trajectories

    Returns
    -------

    """
    # Detect changes to original schedule
    traces_influenced_agents: Trajectories = {}
    additional_information = dict()
    # explode trajectories in order to be able to do point-wise diff!
    base_trajectories_exploded = explode_trajectories(base_trajectories)
    target_trajectories_exploded = explode_trajectories(target_trajectories)
    for agent_id in base_trajectories.keys():
        difference_exploded = base_trajectories_exploded[agent_id] - target_trajectories_exploded[agent_id]

        if len(difference_exploded) > 0:
            trace = []
            for (resource, time_step) in difference_exploded:
                # TODO we draw one-dot strokes, should we collapse to longer strokes?
                #  We want to keep the triplet structure in the trajectories in order not to have to distinguish between cases!
                trace.append((resource, time_step))
                trace.append((resource, time_step))
                trace.append((None, None))
            traces_influenced_agents[agent_id] = trace
            additional_information.update({agent_id: True})
        else:
            traces_influenced_agents[agent_id] = [(None, None)]
            additional_information.update({agent_id: False})
    space_time_difference = SpaceTimeDifference(changed_agents=traces_influenced_agents,
                                                additional_information=additional_information)
    return space_time_difference


def plot_resource_occupation_heat_map(
        schedule_plotting: SchedulePlotting,
        plotting_information: PlottingInformation,
        title_suffix: str = ''):
    """Plot agent density over resource.

    Parameters
    ----------
    schedule_plotting
    plotting_information : PlottingInformation

    Returns
    -------
    """
    x = []
    y = []
    size = []
    layout = go.Layout(
        plot_bgcolor=GREY_BACKGROUND_COLOR
    )
    fig = go.Figure(layout=layout)

    schedule_as_resource_occupations = schedule_plotting.schedule_as_resource_occupations
    reschedule_as_resource_occupations = schedule_plotting.reschedule_delta_perfect_as_resource_occupations

    # Count agents per resource for full episode
    for resource, resource_occupations in schedule_as_resource_occupations.sorted_resource_occupations_per_resource.items():
        x.append(resource.column)
        y.append(resource.row)
        size.append(len(resource_occupations))

    # Generate diff between schedule and re-schedule
    # TODO Update this to account for ressources not occupied in both schedule and re-schedule
    x_r = []
    y_r = []
    size_r = []
    for resource, resource_occupations in reschedule_as_resource_occupations.sorted_resource_occupations_per_resource.items():
        x_r.append(resource.column)
        y_r.append(resource.row)
        if resource in schedule_as_resource_occupations.sorted_resource_occupations_per_resource:
            schedule_number_of_occupations = len(schedule_as_resource_occupations.sorted_resource_occupations_per_resource[resource])
        else:
            schedule_number_of_occupations = 0
        size_r.append((len(resource_occupations)) - schedule_number_of_occupations)

    # Count start-target occupations
    starts_target = {}
    for agent in schedule_as_resource_occupations.sorted_resource_occupations_per_agent:
        curr_start = schedule_as_resource_occupations.sorted_resource_occupations_per_agent[agent][0].resource
        curr_target = schedule_as_resource_occupations.sorted_resource_occupations_per_agent[agent][-1].resource

        if curr_start not in starts_target:
            starts_target[curr_start] = 1
        else:
            starts_target[curr_start] += 1

        if curr_target not in starts_target:
            starts_target[curr_target] = 1
        else:
            starts_target[curr_target] += 1

    # Condense to fewer city points for better overview
    # all variables here are abbreviated with _st meaning start-target
    cities = _condense_to_cities(starts_target)
    x_st = []
    y_st = []
    size_st = []
    for start, city_size in cities.items():
        x_st.append(start.column)
        y_st.append(start.row)
        size_st.append(city_size)

    # Plot resource occupations diff
    fig.add_trace(go.Scattergl(x=x_r,
                               y=y_r,
                               mode='markers',
                               name="Resources Occupation Diff",
                               marker=dict(
                                   color=size_r,
                                   size=15,
                                   symbol='square',
                                   showscale=True,
                                   reversescale=False,
                                   colorbar=dict(
                                       title="Resource Occupations", len=0.75
                                   ), colorscale="Hot"
                               )))

    # Plot resource occupations
    fig.add_trace(go.Scattergl(x=x,
                               y=y,
                               mode='markers',
                               name="Schedule Resources",
                               marker=dict(
                                   color=size,
                                   size=15,
                                   symbol='square',
                                   showscale=True,
                                   reversescale=False,
                                   colorbar=dict(
                                       title="Resource Occupations", len=0.75
                                   ), colorscale="Hot"
                               )))

    # Plot targets and starts
    fig.add_trace(go.Scattergl(x=x_st,
                               y=y_st,
                               mode='markers',
                               name="Schedule Start-Targets",
                               hovertext=size_st,
                               hovertemplate="Nr. Agents %{hovertext}",
                               marker=dict(
                                   color=size_st,
                                   size=100 * size_st,
                                   sizemode='area',
                                   sizeref=2. * max(size) / (40. ** 2),
                                   sizemin=4,
                                   symbol='circle',
                                   opacity=1.,
                                   showscale=True,
                                   reversescale=False,
                                   colorbar=dict(
                                       title="Targets", len=0.75
                                   ), colorscale="Hot"
                               )))

    fig.update_layout(title_text=f"Train Density at Resources {title_suffix}",
                      autosize=False,
                      width=1000,
                      height=1000)

    fig.update_xaxes(zeroline=False, showgrid=True, range=[0, plotting_information.grid_width], tick0=-0.5, dtick=1, gridcolor='Grey')
    fig.update_yaxes(zeroline=False, showgrid=True, range=[plotting_information.grid_width, 0], tick0=-0.5, dtick=1, gridcolor='Grey')

    fig.show()


def _condense_to_cities(positions: Dict[Resource, int]) -> Dict[Resource, int]:
    """Condenses start or targets points to a city point.

    Parameters
    ----------
    positions
        original positions of starts or targets with occupation counts

    Returns
    -------
    dict containing the new coordinates and occupations
    """
    cluster = copy(positions)
    # inefficient and dirty way to do this
    old_len_cluster = 0
    while old_len_cluster != len(cluster):
        cluster_copy = copy(cluster)
        old_len_cluster = len(cluster)
        for resource, occupation in cluster_copy.items():
            for neighb_resource, neighb_occupation in cluster_copy.items():
                if neighb_resource != resource:
                    if np.linalg.norm(np.array(resource) - np.array(neighb_resource)) < 5:
                        new_column = (resource.column + neighb_resource.column) // 2
                        new_row = (resource.row + neighb_resource.row) // 2
                        city = Resource(column=new_column, row=new_row)
                        cluster.pop(resource, None)
                        cluster.pop(neighb_resource, None)
                        cluster[city] = occupation + neighb_occupation

    return cluster


def plot_delay_propagation_2d(
        plotting_data: SchedulePlotting,
        delay_information: Dict[int, int],
        depth_dict: Dict[int, int],
        changed_agents: Optional[Dict[int, bool]] = None,
        file_name: Optional[str] = None):
    """
    Plot agent delay over ressource, only plot agents that are affected by the malfunction.
    Parameters
    ----------
    schedule_resources
        Dict containing all the times and agent handles for all resources

    Returns
    -------

    """

    marker_list = ['triangle-up', 'triangle-right', 'triangle-down', 'triangle-left']
    depth_color = ['red', 'orange', 'yellow', 'white', 'LightGreen', 'green']
    layout = go.Layout(
        plot_bgcolor=GREY_BACKGROUND_COLOR
    )
    fig = go.Figure(layout=layout)

    # Sort agents according to influence depth for plotting only plot disturbed agents
    sorted_agents = []
    for agent, _depth in sorted(depth_dict.items(), key=lambda item: item[1], reverse=True):
        if agent in plotting_data.schedule_as_resource_occupations.sorted_resource_occupations_per_agent:
            sorted_agents.append(agent)
    if changed_agents is not None:
        agents = [agent for agent in sorted_agents if changed_agents[agent]]
    else:
        agents = sorted_agents

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
            marker.append(marker_list[int(np.clip(resource_occupation.direction, 0, 3))])
            times.append(time)
            delay.append(delay_information[agent_id])
            if agent_id in depth_dict:
                conflict_depth.append(depth_dict[agent_id])
            else:
                conflict_depth.append("None")
        if agent_id in depth_dict:
            color = depth_color[int(np.clip(depth_dict[agent_id], 0, 5))]
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
    if file_name is None:
        fig.show()
    else:
        fig.write_image(file_name)


def plot_train_paths(
        plotting_data: SchedulePlotting,
        agent_ids: List[int],
        file_name: Optional[str] = None):
    """
    Plot agent delay over ressource, only plot agents that are affected by the malfunction.
    Parameters
    ----------
    schedule_resources
        Dict containing all the times and agent handles for all resources

    Returns
    -------

    """

    marker_list = ['triangle-up', 'triangle-right', 'triangle-down', 'triangle-left']
    layout = go.Layout(
        plot_bgcolor=GREY_BACKGROUND_COLOR
    )
    fig = go.Figure(layout=layout)

    # Plot traces of agents
    for agent_id in agent_ids:
        x = []
        y = []
        marker = []
        times = []
        delay = []
        conflict_depth = []
        for resource_occupation in plotting_data.schedule_as_resource_occupations.sorted_resource_occupations_per_agent[agent_id]:
            time = resource_occupation.interval.from_incl

            malfunction_resource = resource_occupation.resource
            x.append(malfunction_resource[1])
            y.append(malfunction_resource[0])
            marker.append(marker_list[int(np.clip(resource_occupation.direction, 0, 3))])
            times.append(time)
            color = PLOTLY_COLORLIST[agent_id]

        fig.add_trace(go.Scattergl(x=x,
                                   y=y,
                                   mode='markers',
                                   name="Train {}".format(agent_id),
                                   marker_symbol=marker,
                                   customdata=list(zip(times, delay, conflict_depth)),
                                   marker_size=10,
                                   marker_opacity=1,
                                   marker_color=color,
                                   marker_line_color=color,
                                   hovertemplate="Time:\t%{customdata[0]}<br>" +
                                                 "Delay:\t%{customdata[1]}<br>" +
                                                 "Influence depth:\t%{customdata[2]}"
                                   ))
    fig.update_layout(title_text="Malfunction position and effects",
                      autosize=False,
                      width=1000,
                      height=1000)

    fig.update_yaxes(zeroline=False, showgrid=True, range=[plotting_data.plotting_information.grid_width, 0], tick0=-0.5, dtick=1, gridcolor='Grey')
    fig.update_xaxes(zeroline=False, showgrid=True, range=[0, plotting_data.plotting_information.grid_width], tick0=-0.5, dtick=1, gridcolor='Grey')
    if file_name is None:
        fig.show()
    else:
        fig.write_image(file_name)


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
        plot_bgcolor=GREY_BACKGROUND_COLOR
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
    Build trajectories for time-resource graph
    Parameters
    ----------
    resource_occupations_schedule

    Returns
    -------

    """
    resource_sorting = plotting_information.sorting
    width = plotting_information.grid_width
    schedule_trajectories: Trajectories = {}
    for agent_handle, resource_ocupations in resource_occupations_schedule.items():
        train_time_path = []
        for resource_ocupation in resource_ocupations:
            position = coordinate_to_position(width, [resource_ocupation.resource])[0]
            # TODO dirty hack: add positions from re-scheduling to resource_sorting in the first place instead of workaround here!
            if position not in resource_sorting:
                resource_sorting[position] = len(resource_sorting)
            train_time_path.append((resource_sorting[position], resource_ocupation.interval.from_incl))
            train_time_path.append((resource_sorting[position], resource_ocupation.interval.to_excl))
            train_time_path.append((None, None))
        schedule_trajectories[agent_handle] = train_time_path
    return schedule_trajectories
