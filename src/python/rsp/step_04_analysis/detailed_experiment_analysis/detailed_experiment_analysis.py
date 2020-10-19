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
import re
from copy import copy
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from flatland.core.grid.grid_utils import position_to_coordinate
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from rsp.scheduling.scheduling_problem import get_paths_in_route_dag
from rsp.scheduling.scheduling_problem import RouteDAGConstraintsDict
from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.scheduling.scheduling_problem import ScheduleProblemEnum
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.step_03_run.experiment_results_analysis import all_scopes
from rsp.step_03_run.experiment_results_analysis import ExperimentResultsAnalysis
from rsp.step_03_run.experiment_results_analysis import rescheduling_scopes
from rsp.step_04_analysis.detailed_experiment_analysis.route_dag_analysis import visualize_route_dag_constraints
from rsp.step_04_analysis.detailed_experiment_analysis.schedule_plotting import PlottingInformation
from rsp.step_04_analysis.detailed_experiment_analysis.schedule_plotting import Resource
from rsp.step_04_analysis.detailed_experiment_analysis.schedule_plotting import SchedulePlotting
from rsp.step_04_analysis.detailed_experiment_analysis.trajectories import get_difference_in_time_space_trajectories
from rsp.step_04_analysis.detailed_experiment_analysis.trajectories import time_windows_as_resource_occupations_per_agent
from rsp.step_04_analysis.detailed_experiment_analysis.trajectories import Trajectories
from rsp.step_04_analysis.detailed_experiment_analysis.trajectories import trajectories_from_resource_occupations_per_agent
from rsp.step_04_analysis.plot_utils import GREY_BACKGROUND_COLOR
from rsp.step_04_analysis.plot_utils import PLOTLY_COLORLIST
from rsp.utils.file_utils import check_create_folder
from rsp.utils.resource_occupation import ScheduleAsResourceOccupations
from rsp.utils.resource_occupation import SortedResourceOccupationsPerAgent


def plot_time_window_resource_trajectories(
    experiment_result: ExperimentResultsAnalysis, schedule_plotting: SchedulePlotting, output_folder: Optional[str] = None
):
    """Plot time-window -- resource diagram for all three problems.

    Parameters
    ----------
    experiment_result
    schedule_plotting
    """
    for title, problem in {
        "Schedule": experiment_result.problem_schedule,
        "Full Re-Schedule": experiment_result.problem_online_unrestricted,
        "scope perfect re-schedule": experiment_result.problem_offline_delta,
    }.items():
        resource_occupations_schedule = time_windows_as_resource_occupations_per_agent(problem=problem)
        trajectories = trajectories_from_resource_occupations_per_agent(
            resource_occupations_schedule=resource_occupations_schedule, plotting_information=schedule_plotting.plotting_information
        )
        plot_time_resource_trajectories(
            trajectories=trajectories,
            title=title,
            plotting_information=schedule_plotting.plotting_information,
            malfunction=schedule_plotting.malfunction,
            output_folder=output_folder,
        )


def plot_shared_heatmap(schedule_plotting: SchedulePlotting, experiment_result: ExperimentResultsAnalysis, output_folder: Optional[str] = None):
    """Plot a heat map of how many shareds are on the resources.

    Parameters
    ----------
    experiment_result
    """
    layout = go.Layout(plot_bgcolor=GREY_BACKGROUND_COLOR)
    fig = go.Figure(layout=layout)
    plotting_information = schedule_plotting.plotting_information
    for title, result in {
        "Schedule": experiment_result.results_schedule,
        "Full Re-Schedule": experiment_result.results_online_unrestricted,
        "scope perfect re-schedule": experiment_result.results_offline_delta,
    }.items():
        shared = list(filter(lambda s: s.startswith("shared"), result.solver_result))
        shared_per_resource = {}
        for sh in shared:
            sh = sh.replace("shared", "")
            sh = re.sub("t[0-9]+", '"XXX"', sh)
            #  the position of each entry waypoint is the cell that will be in conflict
            (_, (wp00, _), _, (wp10, _)) = eval(sh)
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

        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                name=title,
                marker=dict(
                    color=z, size=15, symbol="square", showscale=True, reversescale=False, colorbar=dict(title="Number of shared", len=0.75), colorscale="Hot"
                ),
            )
        )

    fig.update_layout(title_text="Shared Resources", autosize=False, width=1000, height=1000)
    fig.update_xaxes(zeroline=False, showgrid=True, range=[0, plotting_information.grid_width], tick0=-0.5, dtick=1, gridcolor="Grey")
    fig.update_yaxes(zeroline=False, showgrid=True, range=[plotting_information.grid_width, 0], tick0=-0.5, dtick=1, gridcolor="Grey")
    if output_folder is None:
        fig.show()
    else:
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, f"shared_heatmap.pdf")
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file)


def plot_resource_time_diagrams(schedule_plotting: SchedulePlotting, with_diff: bool = True, output_folder: Optional[str] = None) -> Dict[int, bool]:
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
    resource_occupations_schedule = schedule_plotting.schedule_as_resource_occupations.sorted_resource_occupations_per_agent
    resource_occupations_online_unrestricted = schedule_plotting.online_unrestricted_as_resource_occupations.sorted_resource_occupations_per_agent
    resource_occupations_offline_delta = schedule_plotting.offline_delta_as_resource_occupations.sorted_resource_occupations_per_agent
    trajectories_schedule: Trajectories = trajectories_from_resource_occupations_per_agent(
        resource_occupations_schedule=resource_occupations_schedule, plotting_information=plotting_information
    )
    trajectories_online_unrestricted: Trajectories = trajectories_from_resource_occupations_per_agent(
        resource_occupations_schedule=resource_occupations_online_unrestricted, plotting_information=plotting_information
    )
    trajectories_offline_delta: Trajectories = trajectories_from_resource_occupations_per_agent(
        resource_occupations_schedule=resource_occupations_offline_delta, plotting_information=plotting_information
    )

    # Plot Schedule
    plot_time_resource_trajectories(
        title="Schedule",
        plotting_information=schedule_plotting.plotting_information,
        malfunction=schedule_plotting.malfunction,
        trajectories=trajectories_schedule,
        output_folder=output_folder,
    )

    # Plot Reschedule Full only plot this if there is an actual difference between schedule and reschedule
    trajectories_influenced_agents, changed_agents_dict = get_difference_in_time_space_trajectories(
        target_trajectories=trajectories_schedule, base_trajectories=trajectories_online_unrestricted
    )

    # Printing situation overview

    print_situation_overview(schedule_plotting=schedule_plotting, changed_agents_dict=changed_agents_dict)

    # Plot Reschedule Full only if svomething has changed
    nb_changed_agents = sum([1 for changed in changed_agents_dict.values() if changed])
    if nb_changed_agents > 0:
        plot_time_resource_trajectories(
            trajectories=trajectories_online_unrestricted,
            title="Full Reschedule",
            plotting_information=schedule_plotting.plotting_information,
            malfunction=schedule_plotting.malfunction,
            output_folder=output_folder,
        )

    # Plot Reschedule Delta Perfect with additional data
    plot_time_resource_trajectories(
        title="Delta Perfect Reschedule",
        plotting_information=schedule_plotting.plotting_information,
        malfunction=schedule_plotting.malfunction,
        trajectories=trajectories_offline_delta,
        output_folder=output_folder,
    )

    # Plot difference if asked for
    if with_diff:
        plot_time_resource_trajectories(
            trajectories=trajectories_influenced_agents,
            title="Changed Agents",
            plotting_information=schedule_plotting.plotting_information,
            malfunction=schedule_plotting.malfunction,
            output_folder=output_folder,
        )

    return changed_agents_dict


def print_situation_overview(schedule_plotting: SchedulePlotting, changed_agents_dict: Dict):
    # Printing situation overview
    malfunction = schedule_plotting.malfunction
    resource_occupations_schedule: SortedResourceOccupationsPerAgent = schedule_plotting.schedule_as_resource_occupations.sorted_resource_occupations_per_agent
    resource_occupations_offline_delta = schedule_plotting.offline_delta_as_resource_occupations.sorted_resource_occupations_per_agent

    nb_changed_agents = sum([1 for changed in changed_agents_dict.values() if changed])
    total_lateness = sum(
        max(sorted_resource_occupations_offline_delta[-1].interval.to_excl - resource_occupations_schedule[agent_id][-1].interval.to_excl, 0)
        for agent_id, sorted_resource_occupations_offline_delta in resource_occupations_offline_delta.items()
    )
    print(
        "Agent nr.{} has a malfunction at time {} for {} s and influenced {} other agents. Total delay = {}.".format(
            malfunction.agent_id, malfunction.time_step, malfunction.malfunction_duration, nb_changed_agents, total_lateness
        )
    )


def plot_time_resource_trajectories(
    title: str,
    trajectories: Trajectories,
    plotting_information: PlottingInformation,
    malfunction: ExperimentMalfunction,
    additional_data: Dict = None,
    malfunction_wave: List[Trajectories] = None,
    output_folder: Optional[str] = None,
):
    """
    Plot the time-resource-diagram with additional data for each train
    Parameters
    ----------
    malfunction_wave:
        two-element list containing true and false positives
        TODO SIM-672 make dict or rename parameter
    title: str
        Title of the plot
    trajectories:
        Data to be shown, contains tuples for all occupied resources during train run
    additional_data
        Dict containing additional data. Each additional data must have the same dimensins as time_resource_data
    output_folder
    malfunction
    plotting_information

    Returns
    -------

    """
    layout = go.Layout(plot_bgcolor=GREY_BACKGROUND_COLOR)
    fig = go.Figure(layout=layout)
    ranges = plotting_information.dimensions
    ticks = [position_to_coordinate(plotting_information.grid_width, [key])[0] for key in plotting_information.sorting.keys()]

    # Get keys and information to add to hover data
    hovertemplate = "<b>Resource ID:<b> %{x}<br>" + "<b>Time:<b> %{y}<br>"
    if additional_data is not None:
        list_keys = [k for k in additional_data]
        list_values = [v for v in additional_data.values()]
        # Build hovertemplate
        for idx, data_point in enumerate(list_keys):
            hovertemplate += "<b>" + str(data_point) + "</b>: %{{customdata[{}]}}<br>".format(idx)
        for idx, line in trajectories.items():
            # Don't plot trains with no paths --> this is just to make plots more readable
            if len(line) < 2:
                continue
            x, y = zip(*line)
            trace_color = PLOTLY_COLORLIST[int(idx % len(PLOTLY_COLORLIST))]

            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    marker=dict(size=2, color=trace_color),
                    line=dict(color=trace_color),
                    name="Agent {}".format(idx),
                    customdata=np.dstack([list_values[:][k][idx] for k in range(len(list_values[:]))])[0],
                    hovertemplate=hovertemplate,
                )
            )
    else:
        for idx, line in trajectories.items():
            # Don't plot trains with no paths --> this is just to make plots more readable
            if len(line) < 2:
                continue
            x, y = zip(*line)
            trace_color = PLOTLY_COLORLIST[int(idx % len(PLOTLY_COLORLIST))]

            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    marker=dict(size=2, color=trace_color),
                    line=dict(color=trace_color),
                    name="Agent {}".format(idx),
                    hovertemplate=hovertemplate,
                )
            )
    if malfunction is not None:
        x = [-10, ranges[1] + 10]
        y = [malfunction.time_step, malfunction.time_step]
        fig.add_trace(go.Scattergl(x=x, y=y, name="malfunction start", line=dict(color="red")))
        y = [malfunction.time_step + malfunction.malfunction_duration, malfunction.time_step + malfunction.malfunction_duration]
        fig.add_trace(go.Scattergl(x=x, y=y, name="malfunction end", line=dict(color="red", dash="dash")))

    if malfunction_wave is not None:
        x, y = zip(*list(malfunction_wave[0].values())[0])
        fig.add_trace(
            go.Scattergl(
                x=x, y=y, mode="lines+markers", marker=dict(size=2, color="red"), line=dict(color="red"), name="True Positives", hovertemplate=hovertemplate
            )
        )
        x, y = zip(*list(malfunction_wave[1].values())[0])
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="lines+markers",
                marker=dict(size=2, color="yellow"),
                line=dict(color="yellow"),
                name="False Positives",
                hovertemplate=hovertemplate,
            )
        )
    fig.update_layout(
        title_text=title,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis=dict(tickmode="array", tickvals=np.arange(len(ticks)), ticktext=ticks, tickangle=270),
    )
    fig.update_xaxes(title="Resource Coordinates", range=[0, ranges[0]])

    fig.update_yaxes(title="Time", range=[ranges[1], 0])
    if output_folder is None:
        fig.show()
    else:
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, f'time_resource_trajectories_{title.replace(" ", "_")}.pdf')
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file)


def plot_histogram_from_delay_data(experiment_results: ExperimentResultsAnalysis, output_folder: Optional[str] = None):
    """Plot a histogram of the delay of agents in the full and delta perfect
    reschedule compared to the schedule."""

    fig = go.Figure()
    for scope in rescheduling_scopes:
        fig.add_trace(go.Histogram(x=[v for v in experiment_results._asdict()[f"lateness_per_agent_{scope}"].values()], name=f"results_{scope}"))
    fig.update_layout(barmode="group", legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1))

    fig.update_traces(opacity=0.75)
    fig.update_layout(title_text="Delay distributions")
    fig.update_xaxes(title="Delay [s]")

    if output_folder is None:
        fig.show()
    else:
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, f"delay_histogram.pdf")
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file)


def plot_lateness(experiment_results: ExperimentResultsAnalysis, output_folder: Optional[str] = None):
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
    for scope in rescheduling_scopes:
        fig.add_trace(go.Bar(x=[f"costs_{scope}"], y=[experiment_results._asdict()[f"costs_{scope}"]], name=f"costs_{scope}"))
        fig.add_trace(go.Bar(x=[f"lateness_{scope}"], y=[experiment_results._asdict()[f"lateness_{scope}"]], name=f"lateness_{scope}"))
        fig.add_trace(
            go.Bar(
                x=[f"costs_from_route_section_penalties_{scope}"],
                y=[experiment_results._asdict()[f"costs_from_route_section_penalties_{scope}"]],
                name=f"costs_from_route_section_penalties_{scope}",
            )
        )
    fig.update_layout(barmode="overlay", legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_traces(opacity=0.75)
    fig.update_layout(title_text="Total delay and Solver objective")
    fig.update_yaxes(title="discrete time steps [-] / weighted sum [-]")

    if output_folder is None:
        fig.show()
    else:
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, f"lateness.pdf")
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file)


def plot_agent_specific_delay(experiment_results: ExperimentResultsAnalysis, output_folder: Optional[str] = None):
    """plot_histogram_from_delay_data
    Plot a histogram of the delay of agents in the full and reschedule delta perfect compared to the schedule
    Parameters
    ----------
    experiment_data_frame
    experiment_id

    Returns
    -------

    """
    fig = go.Figure()
    for scope in rescheduling_scopes:
        d = {}
        for dim in ["lateness_per_agent", "costs_from_route_section_penalties_per_agent"]:
            values = list(experiment_results._asdict()[f"{dim}_{scope}"].values())
            d[dim] = sum(values)
            fig.add_trace(go.Bar(x=np.arange(len(values)), y=values, name=f"{dim}_{scope}"))
    fig.update_traces(opacity=0.75)
    fig.update_layout(title_text="Delay per Train")
    fig.update_xaxes(title="Train ID")
    fig.update_yaxes(title="Delay in Seconds")

    if output_folder is None:
        fig.show()
    else:
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, f"agen_specific_delay.pdf")
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file)


def plot_changed_agents(experiment_results: ExperimentResultsAnalysis, output_folder: Optional[str] = None):
    """Plot a histogram of the delay of agents in the full and reschedule delta
    perfect compared to the schedule.

    Returns
    -------
    """
    fig = go.Figure()
    schedule_trainruns_dict = experiment_results.results_schedule.trainruns_dict
    for scope in rescheduling_scopes:
        reschedule_trainruns_dict = experiment_results._asdict()[f"results_{scope}"].trainruns_dict
        values = [
            1.0 if set(schedule_trainruns_dict[agent_id]) != set(trainrun_reschedule) else 0.0
            for agent_id, trainrun_reschedule in reschedule_trainruns_dict.items()
        ]
        fig.add_trace(go.Bar(x=np.arange(len(values)), y=values, name=f"results_{scope}"))
    fig.update_traces(opacity=0.75)
    fig.update_layout(title_text="Changed per Train")
    fig.update_xaxes(title="Train ID")
    fig.update_yaxes(title="Changed 1.0=yes, 0.0=no [-]")

    if output_folder is None:
        fig.show()
    else:
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, f"changed_per_train.pdf")
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file)

    fig = go.Figure()
    schedule_trainruns_dict = experiment_results.results_schedule.trainruns_dict
    for scope in rescheduling_scopes:
        reschedule_trainruns_dict = experiment_results._asdict()[f"results_{scope}"].trainruns_dict
        values = [
            1.0
            if (
                {trainrun_waypoint.waypoint for trainrun_waypoint in schedule_trainruns_dict[agent_id]}
                != {trainrun_waypoint.waypoint for trainrun_waypoint in trainrun_reschedule}
            )
            else 0.0
            for agent_id, trainrun_reschedule in reschedule_trainruns_dict.items()
        ]
        fig.add_trace(go.Bar(x=np.arange(len(values)), y=values, name=f"results_{scope}"))
    fig.update_traces(opacity=0.75)
    fig.update_layout(title_text="Changed routes per Train")
    fig.update_xaxes(title="Train ID")
    fig.update_yaxes(title="Changed 1.0=yes, 0.0=no [-]")

    if output_folder is None:
        fig.show()
    else:
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, f"changed_routes_per_train.pdf")
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file)


def plot_route_dag(
    experiment_results_analysis: ExperimentResultsAnalysis,
    agent_id: int,
    suffix_of_constraints_to_visualize: ScheduleProblemEnum,
    save: bool = False,
    output_folder: Optional[str] = None,
):
    train_runs_schedule: TrainrunDict = experiment_results_analysis.solution_schedule
    train_runs_online_unrestricted: TrainrunDict = experiment_results_analysis.solution_online_unrestricted
    train_runs_offline_delta: TrainrunDict = experiment_results_analysis.solution_offline_delta
    train_run_schedule: Trainrun = train_runs_schedule[agent_id]
    train_run_online_unrestricted: Trainrun = train_runs_online_unrestricted[agent_id]
    train_run_offline_delta: Trainrun = train_runs_offline_delta[agent_id]
    problem_schedule: ScheduleProblemDescription = experiment_results_analysis.problem_schedule
    problem_rsp_schedule: ScheduleProblemDescription = experiment_results_analysis.problem_online_unrestricted
    problem_rsp_reduced_scope_perfect: ScheduleProblemDescription = experiment_results_analysis.problem_offline_delta
    # TODO hacky, we should take the topo_dict from infrastructure maybe?
    topo = experiment_results_analysis.problem_online_unrestricted.topo_dict[agent_id]

    config = {
        ScheduleProblemEnum.PROBLEM_SCHEDULE: [
            problem_schedule,
            f"Schedule RouteDAG for agent {agent_id} in experiment {experiment_results_analysis.experiment_id}",
            train_run_schedule,
        ],
        ScheduleProblemEnum.PROBLEM_RSP_FULL_AFTER_MALFUNCTION: [
            problem_rsp_schedule,
            f"Full Reschedule RouteDAG for agent {agent_id} in experiment {experiment_results_analysis.experiment_id}",
            train_run_online_unrestricted,
        ],
        ScheduleProblemEnum.PROBLEM_RSP_DELTA_PERFECT_AFTER_MALFUNCTION: [
            problem_rsp_reduced_scope_perfect,
            f"Delta Perfect Reschedule RouteDAG for agent {agent_id} in experiment {experiment_results_analysis.experiment_id}",
            train_run_offline_delta,
        ],
        ScheduleProblemEnum.PROBLEM_RSP_DELTA_ONLINE_AFTER_MALFUNCTION: [
            problem_rsp_reduced_scope_perfect,
            f"Online Reschedule RouteDAG for agent {agent_id} in experiment {experiment_results_analysis.experiment_id}",
            train_run_offline_delta,
        ],
        ScheduleProblemEnum.PROBLEM_RSP_DELTA_RANDOM_AFTER_MALFUNCTION: [
            problem_rsp_reduced_scope_perfect,
            f"Delta Random Reschedule RouteDAG for agent {agent_id} in experiment {experiment_results_analysis.experiment_id}",
            train_run_offline_delta,
        ],
    }

    problem_to_visualize, title, trainrun_to_visualize = config[suffix_of_constraints_to_visualize]

    visualize_route_dag_constraints(
        topo=topo,
        train_run_schedule=train_run_schedule,
        train_run_online_unrestricted=train_run_online_unrestricted,
        train_run_offline_delta=train_run_offline_delta,
        constraints_to_visualize=problem_to_visualize.route_dag_constraints_dict[agent_id],
        trainrun_to_visualize=trainrun_to_visualize,
        vertex_lateness={},
        costs_from_route_section_penalties_per_agent_and_edge={},
        route_section_penalties=problem_to_visualize.route_section_penalties[agent_id],
        title=title,
        file_name=(f"{output_folder}/experiment_{experiment_results_analysis.experiment_id:04d}_agent_{agent_id}_route_graph_schedule.pdf" if save else None),
    )


def plot_resource_occupation_heat_map(
    schedule_plotting: SchedulePlotting, plotting_information: PlottingInformation, title_suffix: str = "", output_folder: Optional[str] = None
):
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
    layout = go.Layout(plot_bgcolor=GREY_BACKGROUND_COLOR)
    fig = go.Figure(layout=layout)

    schedule_as_resource_occupations = schedule_plotting.schedule_as_resource_occupations
    reschedule_as_resource_occupations = schedule_plotting.offline_delta_as_resource_occupations

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
    fig.add_trace(
        go.Scattergl(
            x=x_r,
            y=y_r,
            mode="markers",
            name="Resources Occupation Diff",
            marker=dict(
                color=size_r,
                size=15,
                symbol="square",
                showscale=True,
                reversescale=False,
                colorbar=dict(title="Resource Occupations", len=0.75),
                colorscale="Hot",
            ),
        )
    )

    # Plot resource occupations
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            name="Schedule Resources",
            marker=dict(
                color=size,
                size=15,
                symbol="square",
                showscale=True,
                reversescale=False,
                colorbar=dict(title="Resource Occupations", len=0.75),
                colorscale="Hot",
            ),
        )
    )

    # Plot targets and starts
    fig.add_trace(
        go.Scattergl(
            x=x_st,
            y=y_st,
            mode="markers",
            name="Schedule Start-Targets",
            hovertext=size_st,
            hovertemplate="Nr. Agents %{hovertext}",
            marker=dict(
                color=size_st,
                size=100 * size_st,
                sizemode="area",
                sizeref=2.0 * max(size) / (40.0 ** 2),
                sizemin=4,
                symbol="circle",
                opacity=1.0,
                showscale=True,
                reversescale=False,
                colorbar=dict(title="Targets", len=0.75),
                colorscale="Hot",
            ),
        )
    )

    fig.update_layout(title_text=f"Train Density at Resources {title_suffix}", autosize=False, width=1000, height=1000)

    fig.update_xaxes(zeroline=False, showgrid=True, range=[0, plotting_information.grid_width], tick0=-0.5, dtick=1, gridcolor="Grey")
    fig.update_yaxes(zeroline=False, showgrid=True, range=[plotting_information.grid_width, 0], tick0=-0.5, dtick=1, gridcolor="Grey")

    if output_folder is None:
        fig.show()
    else:
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, f"resource_occupation_heat_map.pdf")
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file)


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
    file_name: Optional[str] = None,
):
    """
    Plot agent delay over ressource, only plot agents that are affected by the malfunction.
    Parameters
    ----------
    schedule_resources
        Dict containing all the times and agent handles for all resources

    Returns
    -------

    """

    marker_list = ["triangle-up", "triangle-right", "triangle-down", "triangle-left"]
    depth_color = ["red", "orange", "yellow", "white", "LightGreen", "green"]
    layout = go.Layout(plot_bgcolor=GREY_BACKGROUND_COLOR)
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
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                name="Train {}".format(agent_id),
                marker_symbol=marker,
                customdata=list(zip(times, delay, conflict_depth)),
                marker_size=size,
                marker_opacity=0.2,
                marker_color=color,
                marker_line_color=color,
                hovertemplate="Time:\t%{customdata[0]}<br>" + "Delay:\t%{customdata[1]}<br>" + "Influence depth:\t%{customdata[2]}",
            )
        )
    # Plot malfunction
    malfunction_resource = plotting_data.schedule_as_resource_occupations.resource_occupations_per_agent_and_time_step[
        (plotting_data.malfunction.agent_id, plotting_data.malfunction.time_step)
    ][0].resource
    fig.add_trace(
        go.Scattergl(
            x=[malfunction_resource[1]],
            y=[malfunction_resource[0]],
            mode="markers",
            name="Malfunction",
            marker_symbol="x",
            marker_size=25,
            marker_line_color="black",
            marker_color="black",
        )
    )
    fig.update_layout(title_text="Malfunction position and effects", autosize=False, width=1000, height=1000)

    fig.update_yaxes(zeroline=False, showgrid=True, range=[plotting_data.plotting_information.grid_width, 0], tick0=-0.5, dtick=1, gridcolor="Grey")
    fig.update_xaxes(zeroline=False, showgrid=True, range=[0, plotting_data.plotting_information.grid_width], tick0=-0.5, dtick=1, gridcolor="Grey")
    if file_name is None:
        fig.show()
    else:
        fig.write_image(file_name)


def plot_train_paths(plotting_data: SchedulePlotting, agent_ids: List[int], file_name: Optional[str] = None):
    """
    Plot agent delay over ressource, only plot agents that are affected by the malfunction.
    Parameters
    ----------
    schedule_resources
        Dict containing all the times and agent handles for all resources

    Returns
    -------

    """

    marker_list = ["triangle-up", "triangle-right", "triangle-down", "triangle-left"]
    layout = go.Layout(plot_bgcolor=GREY_BACKGROUND_COLOR)
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

        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                name="Train {}".format(agent_id),
                marker_symbol=marker,
                customdata=list(zip(times, delay, conflict_depth)),
                marker_size=10,
                marker_opacity=1,
                marker_color=color,
                marker_line_color=color,
                hovertemplate="Time:\t%{customdata[0]}<br>" + "Delay:\t%{customdata[1]}<br>" + "Influence depth:\t%{customdata[2]}",
            )
        )
    fig.update_layout(title_text="Malfunction position and effects", autosize=False, width=1000, height=1000)

    fig.update_yaxes(zeroline=False, showgrid=True, range=[plotting_data.plotting_information.grid_width, 0], tick0=-0.5, dtick=1, gridcolor="Grey")
    fig.update_xaxes(zeroline=False, showgrid=True, range=[0, plotting_data.plotting_information.grid_width], tick0=-0.5, dtick=1, gridcolor="Grey")
    if file_name is None:
        fig.show()
    else:
        fig.write_image(file_name)


def plot_time_density(schedule_as_resource_occupations: ScheduleAsResourceOccupations, output_folder: Optional[str] = None):
    """Plot agent density over time.

    Parameters
    ----------
    schedule_as_resource_occupations

    Returns
    -------
    """
    x = []
    y = []
    layout = go.Layout(plot_bgcolor=GREY_BACKGROUND_COLOR)
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
    if output_folder is None:
        fig.show()
    else:
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, f"time_density.pdf")
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file)


def plot_nb_route_alternatives(experiment_results: ExperimentResultsAnalysis, output_folder: Optional[str] = None):
    """Plot a histogram of the delay of agents in the full and reschedule delta
    perfect compared to the schedule.

    Returns
    -------
    """
    fig = go.Figure()
    for scope in all_scopes:
        topo_dict = experiment_results._asdict()[f"problem_{scope}"].topo_dict
        values = [len(get_paths_in_route_dag(topo)) for _, topo in topo_dict.items()]
        fig.add_trace(go.Bar(x=np.arange(len(values)), y=values, name=f"{scope}"))
    fig.update_traces(opacity=0.75)
    fig.update_layout(title_text="Routing alternatives")
    fig.update_xaxes(title="Train ID")
    fig.update_yaxes(title="Nb routing alternatives [-]")

    if output_folder is None:
        fig.show()
    else:
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, f"nb_route_alternatives.pdf")
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file)


def plot_agent_speeds(experiment_results: ExperimentResultsAnalysis, output_folder: Optional[str] = None):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(experiment_results.problem_schedule.minimum_travel_time_dict.keys()),
            y=list(experiment_results.problem_schedule.minimum_travel_time_dict.values()),
        )
    )
    fig.update_traces(opacity=0.75)
    fig.update_layout(title_text="Speed")
    fig.update_xaxes(title="Train ID")
    fig.update_yaxes(title="Minimum running time [time steps per cell]")
    if output_folder is None:
        fig.show()
    else:
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, f"speeds.pdf")
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file)


def plot_time_window_sizes(experiment_results: ExperimentResultsAnalysis, output_folder: Optional[str] = None):
    fig = go.Figure()
    for scope in all_scopes:
        route_dag_constraints_dict: RouteDAGConstraintsDict = experiment_results._asdict()[f"problem_{scope}"].route_dag_constraints_dict
        vals = [
            (constraints.latest[v] - constraints.earliest[v], agent_id, v)
            for agent_id, constraints in route_dag_constraints_dict.items()
            for v in constraints.latest
        ]
        fig.add_trace(go.Histogram(x=[val[0] for val in vals], name=f"{scope}",))
    fig.update_layout(barmode="group", legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_traces(hovertemplate="<i>Time Window Size</i>:" + "%{x}" + "<extra></extra>", selector=dict(type="histogram"))

    fig.update_layout(title_text="Time Window Size Distribution")
    fig.update_xaxes(title="Time Window Size [time steps]")
    fig.update_yaxes(title="Counts over all agents and vertices [time steps]")
    if output_folder is None:
        fig.show()
    else:
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, f"time_window_sizes.pdf")
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file)
