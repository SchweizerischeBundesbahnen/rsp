from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas import DataFrame

from rsp.scheduling.scheduling_problem import ScheduleProblemDescription
from rsp.step_02_setup.data_types import ExperimentMalfunction
from rsp.step_03_run.experiment_results import ExperimentResults
from rsp.step_03_run.experiment_results_analysis import all_scopes
from rsp.step_04_analysis.detailed_experiment_analysis.figure_utils import figure_show_or_save
from rsp.step_04_analysis.plot_utils import PLOTLY_COLORLIST
from rsp.utils.global_constants import GlobalConstants
from rsp.utils.resource_occupation import extract_resource_occupations
from rsp.utils.resource_occupation import extract_time_windows


# TODO bad code smell: global_constant should be part of results!
def extract_full_df_from_experiment_results(exp_results_of_experiment_of_interest: ExperimentResults, global_constants: GlobalConstants):
    """Extract a data frame for visualization of solution from experiment
    results for all scopes."""
    sorted_resource_occupations_per_scope_and_agent = {
        scope: extract_resource_occupations(
            exp_results_of_experiment_of_interest._asdict()[f"results_{scope}"].trainruns_dict, global_constants.RELEASE_TIME
        ).sorted_resource_occupations_per_agent
        for scope in all_scopes
    }
    occupations = []
    resources = set()
    for scope in all_scopes:
        sorted_resource_occupations_per_agent = sorted_resource_occupations_per_scope_and_agent[scope]
        for agent_id, sorted_resource_occupations_per_agent in sorted_resource_occupations_per_agent.items():
            for ro in sorted_resource_occupations_per_agent:
                for time_step in range(ro.interval.from_incl, ro.interval.to_excl):
                    occupations.append((scope, ro.resource.row, ro.resource.column, time_step, agent_id, ro.direction, str(tuple(ro.resource))))
                    resources.add(tuple(ro.resource))

    full_df = pd.DataFrame(occupations, columns=["scope", "row", "column", "time_step", "agent_id", "direction", "position"])
    full_df.sort_values(["agent_id", "time_step", "scope"], ascending=[True, True, True])

    df_background = pd.DataFrame(resources, columns=["row", "column"])
    return full_df, df_background, sorted_resource_occupations_per_scope_and_agent


def extract_full_df_time_windows_from_experiment_results(exp_results_of_experiment_of_interest: ExperimentResults, global_constants: GlobalConstants):  # noqa
    """Extract a data frame for visualization of problem from experiment
    results for all scopes."""

    occupations = []
    resources = set()
    sorted_resource_occupations_per_scope_and_agent = {}
    for scope in all_scopes:
        problem: ScheduleProblemDescription = exp_results_of_experiment_of_interest._asdict()[f"problem_{scope}"]
        sorted_time_windows_per_agent = extract_time_windows(
            route_dag_constraints_dict=problem.route_dag_constraints_dict,
            minimum_travel_time_dict=problem.minimum_travel_time_dict,
            release_time=global_constants.RELEASE_TIME,
        ).time_windows_per_agent_sorted_by_lower_bound
        sorted_resource_occupations_per_scope_and_agent[scope] = sorted_time_windows_per_agent
        for agent_id, sorted_time_windows_per_agent in sorted_time_windows_per_agent.items():
            for ro in sorted_time_windows_per_agent:
                for time_step in range(ro.interval.from_incl, ro.interval.to_excl):
                    occupations.append((scope, ro.resource.row, ro.resource.column, time_step, agent_id, ro.direction, str(tuple(ro.resource))))
                    resources.add(tuple(ro.resource))

    full_df = pd.DataFrame(occupations, columns=["scope", "row", "column", "time_step", "agent_id", "direction", "position"])
    full_df.sort_values(["agent_id", "time_step", "scope"], ascending=[True, True, True])

    df_background = pd.DataFrame(resources, columns=["row", "column"])
    return full_df, df_background, sorted_resource_occupations_per_scope_and_agent


def _prepare_df(agents_of_interest, full_df, scopes, time_step_interval):
    df = full_df[full_df["scope"].isin(scopes)]
    if agents_of_interest is not None:
        df = df[df["agent_id"].isin(agents_of_interest)]
    df = df[df["time_step"].isin(reversed(list(range(*time_step_interval))))]
    df["agent_id"] = pd.Categorical(df["agent_id"], categories=agents_of_interest, ordered=True)
    df = df.sort_values(["agent_id", "time_step", "scope"], ascending=[True, True, True])
    return df


def time_resource_graph_from_df(
    full_df: DataFrame,
    scopes: List[str],
    time_step_interval: List[int],
    num_agents: int,
    agents_of_interest: List[int],
    malfunction: ExperimentMalfunction,
    symbol: str = "triangle-down",
    output_folder: str = None,
    file_name: str = None,
):
    """Plot time resource diagram from data frame."""
    df = _prepare_df(agents_of_interest, full_df, scopes, time_step_interval)

    color_continuous_scale = []
    for agent_id in range(num_agents):
        color_continuous_scale.append((agent_id / num_agents, PLOTLY_COLORLIST[int(agent_id % len(PLOTLY_COLORLIST))]))
        color_continuous_scale.append(((agent_id + 1) / num_agents, PLOTLY_COLORLIST[int(agent_id % len(PLOTLY_COLORLIST))]))
    fig = px.scatter(
        df,
        x="position",
        y="time_step",
        range_y=list(reversed(time_step_interval)),
        # https://stackoverflow.com/questions/60962274/plotly-how-to-change-the-colorscheme-of-a-plotly-express-scatterplot
        color="agent_id",
        color_continuous_scale=color_continuous_scale,
        range_color=[-0.5, num_agents - 0.5],
        symbol="agent_id",
        symbol_map={agent_id: symbol for agent_id in agents_of_interest},
        title="+".join(scopes),
        height=1000,
        width=1500,
    )
    fig.add_trace(
        go.Scatter(x=df["position"], y=[malfunction.time_step] * len(df["position"]), mode="lines", line=go.scatter.Line(color="red"), showlegend=False)
    )
    fig.add_trace(
        go.Scatter(
            x=df["position"],
            y=[malfunction.time_step + malfunction.malfunction_duration] * len(df["position"]),
            mode="lines",
            line=go.scatter.Line(color="red", dash="dot"),
            showlegend=False,
        )
    )
    figure_show_or_save(fig=fig, output_folder=output_folder, file_name=file_name)


def time_resource_graph_3d_from_df(  # noqa
    full_df: DataFrame,
    scopes: List[str],
    time_step_interval: List[int],
    num_agents: int,
    grid_width: int,
    grid_height: int,
    agents_of_interest: List[int] = None,
    output_folder: str = None,
    file_name: str = None,
):
    """Plot 3d time resource diagram from data frame."""
    df = _prepare_df(agents_of_interest, full_df, scopes, time_step_interval)

    color_continuous_scale = []
    for agent_id in range(num_agents):
        color_continuous_scale.append((agent_id / num_agents, PLOTLY_COLORLIST[int(agent_id % len(PLOTLY_COLORLIST))]))
        color_continuous_scale.append(((agent_id + 1) / num_agents, PLOTLY_COLORLIST[int(agent_id % len(PLOTLY_COLORLIST))]))
    fig = px.scatter_3d(
        df,
        x="column",
        y="row",
        z="time_step",
        # https://stackoverflow.com/questions/60962274/plotly-how-to-change-the-colorscheme-of-a-plotly-express-scatterplot
        color="agent_id",
        color_continuous_scale=color_continuous_scale,
        range_color=[-0.5, num_agents - 0.5],
        range_x=[0, grid_width],
        range_y=[grid_height, 0],
        title="+".join(scopes),
    )
    figure_show_or_save(fig=fig, output_folder=output_folder, file_name=file_name)
