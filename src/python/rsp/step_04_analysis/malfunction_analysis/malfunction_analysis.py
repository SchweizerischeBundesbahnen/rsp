import os
from typing import Dict
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.step_03_run.experiment_results_analysis import ExperimentResultsAnalysis
from rsp.step_03_run.experiment_results_analysis import rescheduling_scopes
from rsp.step_04_analysis.detailed_experiment_analysis.resources_plotting_information import PlottingInformation
from rsp.step_04_analysis.plot_utils import GREY_BACKGROUND_COLOR
from rsp.step_04_analysis.plot_utils import PDF_HEIGHT
from rsp.step_04_analysis.plot_utils import PDF_WIDTH
from rsp.utils.file_utils import check_create_folder
from rsp.utils.resource_occupation import ScheduleAsResourceOccupations
from rsp.utils.resource_occupation import ScheduleAsResourceOccupationsAllScopes
from rsp.utils.resource_occupation import SortedResourceOccupationsPerAgent


def plot_delay_propagation_2d(
    plotting_information: PlottingInformation,
    malfunction: ExperimentMalfunction,
    schedule_as_resource_occupations: ScheduleAsResourceOccupations,
    delay_information: Dict[int, int],
    depth_dict: Dict[int, int],
    changed_agents: Optional[Dict[int, bool]] = None,
    pdf_file: Optional[str] = None,
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
        if agent in schedule_as_resource_occupations.sorted_resource_occupations_per_agent:
            sorted_agents.append(agent)
    if changed_agents is not None:
        agents = [agent for agent in sorted_agents if changed_agents[agent]]
    else:
        agents = sorted_agents

    # Add the malfunction source agent
    agents.append(malfunction.agent_id)

    # Plot only after the malfunciton happend
    malfunction_time = malfunction.time_step
    # Plot traces of agents
    for agent_id in agents:
        x = []
        y = []
        size = []
        marker = []
        times = []
        delay = []
        conflict_depth = []
        for resource_occupation in schedule_as_resource_occupations.sorted_resource_occupations_per_agent[agent_id]:
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
    malfunction_resource = schedule_as_resource_occupations.resource_occupations_per_agent_and_time_step[(malfunction.agent_id, malfunction.time_step)][
        0
    ].resource
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

    fig.update_yaxes(zeroline=False, showgrid=True, range=[plotting_information.grid_width, 0], tick0=-0.5, dtick=1, gridcolor="Grey")
    fig.update_xaxes(zeroline=False, showgrid=True, range=[0, plotting_information.grid_width], tick0=-0.5, dtick=1, gridcolor="Grey")
    if pdf_file is None:
        fig.show()
    else:
        fig.write_image(pdf_file, width=PDF_WIDTH, height=PDF_HEIGHT)


def print_situation_overview(malfunction: ExperimentMalfunction, resource_occupations_for_all_scopes: ScheduleAsResourceOccupationsAllScopes):
    # Printing situation overview
    resource_occupations_schedule: SortedResourceOccupationsPerAgent = resource_occupations_for_all_scopes.schedule.sorted_resource_occupations_per_agent
    resource_occupations_offline_delta = resource_occupations_for_all_scopes.offline_delta.sorted_resource_occupations_per_agent

    nb_changed_agents = sum(
        [1 for agent_id in resource_occupations_schedule if resource_occupations_schedule[agent_id] != resource_occupations_offline_delta[agent_id]]
    )
    total_lateness = sum(
        max(resource_occupations_offline_delta[agent_id][-1].interval.to_excl - resource_occupations_schedule[agent_id][-1].interval.to_excl, 0)
        for agent_id in resource_occupations_offline_delta
    )
    print(
        "Agent nr.{} has a malfunction at time {} for {} s and influenced {} other agents. Total delay = {}.".format(
            malfunction.agent_id, malfunction.time_step, malfunction.malfunction_duration, nb_changed_agents, total_lateness
        )
    )


def plot_histogram_from_delay_data(experiment_results_analysis: ExperimentResultsAnalysis, output_folder: Optional[str] = None):
    """Plot a histogram of the delay of agents in the full and delta perfect
    reschedule compared to the schedule."""

    fig = go.Figure()
    for scope in rescheduling_scopes:
        fig.add_trace(go.Histogram(x=[v for v in experiment_results_analysis._asdict()[f"lateness_per_agent_{scope}"].values()], name=f"results_{scope}"))
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
        fig.write_image(pdf_file, width=PDF_WIDTH, height=PDF_HEIGHT)


def plot_lateness(experiment_results_analysis: ExperimentResultsAnalysis, output_folder: Optional[str] = None):
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
        fig.add_trace(go.Bar(x=[f"costs_{scope}"], y=[experiment_results_analysis._asdict()[f"costs_{scope}"]], name=f"costs_{scope}"))
        fig.add_trace(go.Bar(x=[f"lateness_{scope}"], y=[experiment_results_analysis._asdict()[f"lateness_{scope}"]], name=f"lateness_{scope}"))
        fig.add_trace(
            go.Bar(
                x=[f"costs_from_route_section_penalties_{scope}"],
                y=[experiment_results_analysis._asdict()[f"costs_from_route_section_penalties_{scope}"]],
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
        fig.write_image(pdf_file, width=PDF_WIDTH, height=PDF_HEIGHT)
