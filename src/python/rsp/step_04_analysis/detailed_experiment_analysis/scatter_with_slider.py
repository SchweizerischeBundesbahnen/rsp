from typing import List

import numpy as np
import plotly.graph_objects as go
from pandas import DataFrame
from rsp.step_04_analysis.plot_utils import PLOTLY_COLORLIST


def scatter_with_slider(  # noqa
    df: DataFrame,
    x_dim: str,
    y_dim: str,
    slider_dim: str,
    range_x: List[int],
    range_y: List[int],
    range_slider: List[int],
    agents_list: List[int],
    df_background: DataFrame = None,
):
    # Create figure
    fig = go.Figure()

    # Add traces, one for each slider step
    slider_lb, slider_ub = range_slider
    direction_symbol_map = {
        0: "triangle-up",
        1: "triangle-right",
        2: "triangle-down",
        3: "triangle-left",
    }
    for step in range(slider_lb, slider_ub):
        df_step = df[df[slider_dim] == step]
        fig.add_trace(
            go.Scattergl(
                x=df_step[x_dim],
                y=df_step[y_dim],
                mode="markers",
                hovertext="blup",
                marker=dict(
                    color=df_step["agent_id"],
                    colorscale=PLOTLY_COLORLIST,
                    line_width=1,
                    symbol=[direction_symbol_map[d] for d in df_step["direction"]],
                    colorbar=dict(
                        title="agent_id",
                        titleside="top",
                        tickmode="array",
                        tickvals=np.arange(0, len(agents_list), 1),
                        ticktext=np.arange(0, len(agents_list), 1),
                    ),
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
    fig.update_layout(title_text="Malfunction position and effects", autosize=False, width=1000, height=1000)

    for step in range(0, slider_ub - slider_lb):
        fig.data[step].visible = False  # noqa
    fig.data[0].visible = True  # noqa

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update", args=[{"visible": [False] * len(fig.data)}, {"title": "Slider switched to step: " + str(i + slider_lb)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][slider_ub - slider_lb] = True  # Toggle background trace to "visible"
        steps.append(step)

    sliders = [dict(active=0, steps=steps)]

    fig.update_layout(sliders=sliders)

    fig.show()
