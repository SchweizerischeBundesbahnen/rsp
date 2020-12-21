import os.path
from typing import Callable
from typing import List
from typing import NamedTuple
from typing import Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from _plotly_utils.colors.qualitative import Plotly
from pandas import DataFrame

from rsp.utils.file_utils import check_create_folder
from rsp.utils.rsp_logger import rsp_logger

PDF_WIDTH = 1200
PDF_HEIGHT = 800

PLOTLY_COLORLIST = [
    "aliceblue",
    "antiquewhite",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "black",
    "blanchedalmond",
    "blue",
    "blueviolet",
    "brown",
    "burlywood",
    "cadetblue",
    "chartreuse",
    "chocolate",
    "coral",
    "cornflowerblue",
    "cornsilk",
    "crimson",
    "cyan",
    "darkblue",
    "darkcyan",
    "darkgoldenrod",
    "darkgray",
    "darkgrey",
    "darkgreen",
    "darkkhaki",
    "darkmagenta",
    "darkolivegreen",
    "darkorange",
    "darkorchid",
    "darkred",
    "darksalmon",
    "darkseagreen",
    "darkslateblue",
    "darkslategray",
    "darkslategrey",
    "darkturquoise",
    "darkviolet",
    "deeppink",
    "deepskyblue",
    "dimgray",
    "dimgrey",
    "dodgerblue",
    "firebrick",
    "floralwhite",
    "forestgreen",
    "fuchsia",
    "gainsboro",
    "ghostwhite",
    "gold",
    "goldenrod",
    "gray",
    "grey",
    "green",
    "greenyellow",
    "honeydew",
    "hotpink",
    "indianred",
    "indigo",
    "ivory",
    "khaki",
    "lavender",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgray",
    "lightgrey",
    "lightgreen",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "lightskyblue",
    "lightslategray",
    "lightslategrey",
    "lightsteelblue",
    "lightyellow",
    "lime",
    "limegreen",
    "linen",
    "magenta",
    "maroon",
    "mediumaquamarine",
    "mediumblue",
    "mediumorchid",
    "mediumpurple",
    "mediumseagreen",
    "mediumslateblue",
    "mediumspringgreen",
    "mediumturquoise",
    "mediumvioletred",
    "midnightblue",
    "mintcream",
    "mistyrose",
    "moccasin",
    "navajowhite",
    "navy",
    "oldlace",
    "olive",
    "olivedrab",
    "orange",
    "orangered",
    "orchid",
    "palegoldenrod",
    "palegreen",
    "paleturquoise",
    "palevioletred",
    "papayawhip",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "purple",
    "red",
    "rosybrown",
    "royalblue",
    "saddlebrown",
    "salmon",
    "sandybrown",
    "seagreen",
    "seashell",
    "sienna",
    "silver",
    "skyblue",
    "slateblue",
    "slategray",
    "slategrey",
    "snow",
    "springgreen",
    "steelblue",
    "tan",
    "teal",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "white",
    "whitesmoke",
    "yellow",
    "yellowgreen",
]

GREY_BACKGROUND_COLOR = "rgba(46,49,49,1)"
COLOR_MAP_PER_SCOPE = {
    "online_unrestricted": Plotly[0],
    "offline_fully_restricted": Plotly[1],
    "offline_delta": Plotly[2],
    "offline_delta_weak": Plotly[3],
    "online_route_restricted": Plotly[4],
    "online_transmission_chains_fully_restricted": Plotly[5],
    "online_transmission_chains_route_restricted": Plotly[6],
    "online_random_average": Plotly[7],
    "schedule": Plotly[8],
}


def marker_color_scope(index: int, column: str):
    for scope, color in COLOR_MAP_PER_SCOPE.items():
        if column.endswith(scope):
            return color
    raise Exception(f"Column {column}")


class ColumnSpec(NamedTuple):
    prefix: str
    scope: Optional[str] = None
    dimension: Optional[str] = None


def plot_binned_box_plot(  # noqa: C901
    experiment_data: DataFrame,
    axis_of_interest: str,
    cols: List[ColumnSpec],
    title_text: str,
    experiment_data_comparison: DataFrame = None,
    axis_of_interest_dimension: Optional[str] = None,
    output_folder: Optional[str] = None,
    file_name: Optional[str] = None,
    nb_bins: Optional[int] = 10,
    show_bin_counts: Optional[bool] = False,
    marker_color: Callable[[int, str], str] = None,
    marker_symbol: Callable[[int, str], str] = None,
    one_field_many_scopes: bool = False,
    width: int = PDF_WIDTH,
    height: int = PDF_HEIGHT,
    binned: bool = True,
    experiment_data_suffix: str = None,
    experiment_data_comparison_suffix: str = None,
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
    axis_of_interest_dimension
        label for x axis will be technical `axis_of_interest` column name plus this suffix
    Returns
    -------

    """
    fig = go.Figure()
    # epsilon ensures that first/last bin not only contains min_value/max_value
    epsilon = 0.00001
    min_value = experiment_data[axis_of_interest].min() - epsilon
    max_value = experiment_data[axis_of_interest].max() + epsilon
    inc = (max_value - min_value) / nb_bins
    axis_of_interest_binned = axis_of_interest + "_binned"
    experiment_data = experiment_data.sort_values(by=axis_of_interest)

    if binned:
        experiment_data[axis_of_interest_binned] = (
            experiment_data[axis_of_interest]
            .astype(float)
            .map(lambda fl: f"[{((fl - min_value) // inc) * inc + min_value:.2f},{(max_value - ((max_value - fl) // inc) * inc)  :.2f}]")
        )

    for col_index, col_spec in enumerate(cols):
        col = f"{col_spec.prefix}" + (f"_{col_spec.scope}" if col_spec.scope is not None else "")
        if one_field_many_scopes:
            col_name = col_spec.scope
        else:
            col_name = col
        if not one_field_many_scopes:
            col_name += " [-]" if col_spec.dimension is None else f" [{col_spec.dimension}]"
        data = {col_name + (experiment_data_suffix if experiment_data_suffix is not None else ""): experiment_data}
        if experiment_data_comparison is not None:
            data[
                col_name + (experiment_data_comparison_suffix if experiment_data_comparison_suffix is not None else "_comparison")
            ] = experiment_data_comparison
        for col_name, d in data.items():
            fig.add_trace(
                go.Box(
                    x=d[axis_of_interest_binned if binned else axis_of_interest],
                    y=d[col],
                    pointpos=-1,
                    boxpoints="all",
                    name=col_name,
                    # TODO is this correct if we bin?
                    customdata=np.dstack(
                        (
                            d["n_agents"],
                            d["size"],
                            d["solver_statistics_times_total_schedule"] if "solver_statistics_times_total_schedule" in d.columns else None,
                            d["solver_statistics_times_total_online_unrestricted"]
                            if "solver_statistics_times_total_online_unrestricted" in d.columns
                            else None,
                            d["solver_statistics_times_total_offline_delta"] if "solver_statistics_times_total_offline_delta" in d.columns else None,
                            d["solver_statistics_times_total_online_route_restricted"]
                            if "solver_statistics_times_total_online_route_restricted" in d.columns
                            else None,
                        )
                    )[0],
                    hovertext=d["experiment_id"],
                    hovertemplate="<b>Speed Up</b>: %{y:.2f}<br>"
                    + "<b>Nr. Agents</b>: %{customdata[0]}<br>"
                    + "<b>Grid Size:</b> %{customdata[1]}<br>"
                    + "<b>Schedule Time:</b> %{customdata[2]:.2f}s<br>"
                    + "<b>Re-Schedule Full Time:</b> %{customdata[3]:.2f}s<br>"
                    + "<b>Delta perfect:</b> %{customdata[4]:.2f}s<br>"
                    + "<b>Delta naive:</b> %{customdata[5]:.2f}s<br>"
                    + "<b>Experiment id:</b>%{hovertext}",
                    marker=dict(
                        color=marker_color(col_index, col) if marker_color is not None else Plotly[col_index % len(Plotly)],
                        symbol=marker_symbol(col_index, col_spec.prefix) if marker_symbol is not None else "circle",
                    ),
                ),
            )
    if binned and show_bin_counts:
        fig.add_trace(
            go.Histogram(
                x=experiment_data[axis_of_interest_binned if binned else axis_of_interest].values,
                name=f"counts({axis_of_interest_binned if binned else axis_of_interest})",
            )
        )

    if one_field_many_scopes:
        first_col = cols[0]
        y_axis_title = f"{first_col.prefix} [{first_col.dimension if first_col.dimension is not None else '-'}]"
    else:
        y_axis_title = ""

    fig.update_layout(title_text=title_text)
    fig.update_layout(boxmode="group")
    fig.update_xaxes(title=f"{axis_of_interest} [{axis_of_interest_dimension if axis_of_interest_dimension is not None else '-'}]")
    fig.update_yaxes(title=y_axis_title)
    fig.update_layout(width=width, height=height)
    fig.update_layout(legend=dict(yanchor="top", y=-0.2, x=0.01))

    if binned:
        del experiment_data[axis_of_interest_binned]
    if output_folder is None and file_name is None:
        fig.show()
    else:
        if output_folder is None:
            output_folder = "."
        if file_name is None:
            file_name = f"{axis_of_interest}.pdf"
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, file_name)
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file, width=width, height=height)
        rsp_logger.info(msg=f"wrote {pdf_file}")


def density_hist_plot_2d(
    title: str, data_frame, width: int = PDF_WIDTH, height: int = PDF_HEIGHT, output_folder: Optional[str] = None, file_name: Optional[str] = None
):
    """

    Returns
    -------

    """

    fig = px.density_heatmap(
        data_frame=data_frame,
        x="speed_up_" + title,
        y="additional_costs_" + title,
        nbinsx=20,
        nbinsy=20,
        width=1000,
        height=1000,
        marginal_x="histogram",
        marginal_y="histogram",
    )
    fig.update_layout(title=title, xaxis_title="Speed Up", yaxis_title="Additional Cost")

    fig.update_xaxes(range=[0, 10])
    fig.update_yaxes(range=[-10, 500])
    if output_folder is None and file_name is None:
        fig.show()
    else:
        if output_folder is None:
            output_folder = "."
        if file_name is None:
            file_name = f"{title}.pdf"
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, file_name)
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file, width=width, height=height)
        rsp_logger.info(msg=f"wrote {pdf_file}")
