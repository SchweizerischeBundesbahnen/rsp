from typing import List

import numpy as np
import plotly.graph_objects as go
from pandas import DataFrame

from rsp.step_04_analysis.plot_utils import PLOTLY_COLORLIST


# TODO remove hard-coded in scatter_with_slider


def scatter_with_slider(  # noqa
    full_df: DataFrame,
    scopes: List[str],
    time_step_interval: List[int],
    num_agents: int,
    agents_of_interest: List[int],
    x_dim: str,
    y_dim: str,
    slider_dim: str,
    range_x: List[int],
    range_y: List[int],
    range_slider: List[int],
    df_background: DataFrame = None,
):
    df = full_df[full_df["scope"].isin(scopes)]
    if agents_of_interest is not None:
        df = df[df["agent_id"].isin(agents_of_interest)]
    df = df[df["time_step"].isin(reversed(list(range(*time_step_interval))))]
    df = df.sort_values(["agent_id", "time_step", "scope"], ascending=[True, True, True])

    # Create figure
    fig = go.Figure()

    # Add traces, one for each slider step
    slider_lb, slider_ub = range_slider
    direction_symbol_map = {0: "triangle-up", 1: "triangle-right", 2: "triangle-down", 3: "triangle-left"}
    color_continuous_scale = []

    for agent_id in range(num_agents):
        color_continuous_scale.append((agent_id / num_agents, PLOTLY_COLORLIST[int(agent_id % len(PLOTLY_COLORLIST))]))
        color_continuous_scale.append(((agent_id + 1) / num_agents, PLOTLY_COLORLIST[int(agent_id % len(PLOTLY_COLORLIST))]))
    for step in range(slider_lb, slider_ub):
        df_step = df[df[slider_dim] == step]
        fig.add_trace(
            go.Scattergl(
                x=df_step[x_dim],
                y=df_step[y_dim],
                mode="markers",
                hovertext=df_step["agent_id"].apply(lambda x: f"Agent {x}"),
                marker=dict(
                    color=df_step["agent_id"],
                    # https://plotly.com/python/colorscales/#custom-discretized-heatmap-color-scale-with-graph-objects
                    colorscale=color_continuous_scale,
                    cmax=num_agents - 0.5,
                    cmin=-0.5,
                    line_width=1,
                    symbol=[direction_symbol_map[d] for d in df_step["direction"]],
                    colorbar=dict(tickmode="array", tickvals=np.arange(num_agents), ticktext=np.arange(num_agents)),
                ),
                name=f"Time step {step}",
            )
        )

    fig.add_trace(
        go.Scattergl(x=df_background[x_dim], y=df_background[y_dim], mode="markers", marker=dict(color="grey", symbol="square", opacity=0.1), name="Grid")
    )
    fig.update_xaxes(title_text=x_dim, range=range_x, tick0=-0.5, dtick=1, showticklabels=False)
    fig.update_yaxes(title_text=y_dim, range=range_y, tick0=-0.5, dtick=1, showticklabels=False)
    # fig.update_xaxes(zeroline=False, showgrid=True, range=[0, plotting_information.grid_width], , gridcolor="Grey")
    fig.update_layout(title_text="Time steper " + "+".join(scopes), autosize=False, width=1000, height=1000)
    fig.update(layout_showlegend=False)

    for step in range(0, slider_ub - slider_lb):
        fig.data[step].visible = False  # noqa
    fig.data[0].visible = True  # noqa

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}, {"title": "Slider switched to step: " + str(i + slider_lb)}],  # layout attribute
            label=str(i + slider_lb),
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][slider_ub - slider_lb] = True  # Toggle background trace to "visible"
        steps.append(step)

    sliders = [dict(active=0, steps=steps)]

    fig.update_layout(sliders=sliders)
    fig.show()
