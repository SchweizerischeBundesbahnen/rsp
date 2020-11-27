from typing import Optional

import pandas as pd

from rsp.step_03_run.experiment_results_analysis import all_scopes_visualization
from rsp.step_04_analysis.plot_utils import ColumnSpec
from rsp.step_04_analysis.plot_utils import marker_color_scope
from rsp.step_04_analysis.plot_utils import plot_binned_box_plot


def visualize_asp_problem_reduction(experiment_data: pd.DataFrame, output_folder: Optional[str] = None):
    suffixes = all_scopes_visualization

    plot_binned_box_plot(
        experiment_data=experiment_data,
        axis_of_interest="experiment_id",
        cols=[(ColumnSpec(prefix="nb_resource_conflicts", scope=scope)) for scope in suffixes],
        title_text="problem reduction in terms of resource conflicts (shared)",
        output_folder=output_folder,
    )
    plot_binned_box_plot(
        experiment_data=experiment_data,
        axis_of_interest="experiment_id",
        cols=[(ColumnSpec(prefix="nb_resource_conflicts_ratio", scope=scope)) for scope in suffixes],
        title_text="problem reduction in terms of resource conflicts (shared)",
        output_folder=output_folder,
    )
    plot_binned_box_plot(
        experiment_data=experiment_data,
        axis_of_interest="experiment_id",
        cols=[(ColumnSpec(prefix="solver_statistics_conflicts", scope=scope)) for scope in suffixes],
        title_text="problem reduction ratio in terms of conflicts in the model",
        output_folder=output_folder,
    )
    plot_binned_box_plot(
        experiment_data=experiment_data,
        axis_of_interest="experiment_id",
        cols=[(ColumnSpec(prefix="solver_statistics_conflicts_ratio", scope=scope)) for scope in suffixes],
        title_text="problem reduction in terms of conflicts in the model",
        output_folder=output_folder,
    )
    plot_binned_box_plot(
        experiment_data=experiment_data,
        axis_of_interest="experiment_id",
        cols=[(ColumnSpec(prefix="solver_statistics_choices", scope=scope)) for scope in suffixes],
        title_text="problem reduction in terms of choices during solving",
        output_folder=output_folder,
    )
    plot_binned_box_plot(
        experiment_data=experiment_data,
        axis_of_interest="experiment_id",
        cols=[(ColumnSpec(prefix="solver_statistics_choices_ratio", scope=scope)) for scope in suffixes],
        title_text="problem reduction in terms of choices during solving",
        output_folder=output_folder,
    )


def visualize_asp_solver_stats(experiment_data: pd.DataFrame, output_folder: Optional[str] = None):
    suffixes = all_scopes_visualization
    # 003_ratio_asp_grounding_solving: solver should spend most of the time solving: compare solve and total times
    plot_binned_box_plot(
        experiment_data=experiment_data,
        axis_of_interest="experiment_id",
        cols=[ColumnSpec(prefix="solve_total_ratio", scope=scope) for scope in suffixes],
        title_text="ratio asp grounding solving:\n" "solver should spend most of the time solving: compare solve and total solver time",
        output_folder=output_folder,
        one_field_many_scopes=True,
        marker_color=marker_color_scope,
        binned=False,
    )

    # 002_asp_absolute_total_solver_times
    # https://plotly.com/python/marker-style/
    def marker_symbol_solver_times(index, column):
        if "solver_statistics_times_total_without_solve" in column:
            return "triangle-up-open"
        elif "solver_statistics_times_total" in column:
            return "triangle-down"
        elif "solver_statistics_times_solve" in column:
            return "line-ew"
        elif "solver_statistics_times_unsat" in column:
            return "pentagon"
        elif "solver_statistics_times_sat" in column:
            return "star"
        else:

            return "circle"

    for axis_of_interst in ["experiment_id", "solver_statistics_times_total_online_unrestricted"]:
        plot_binned_box_plot(
            experiment_data=experiment_data,
            axis_of_interest=axis_of_interst,
            cols=[ColumnSpec(prefix="solver_statistics_times_total", scope=scope, dimension="s") for scope in suffixes]
            + [ColumnSpec(prefix="solver_statistics_times_total_without_solve", scope=scope, dimension="s") for scope in suffixes]
            + [ColumnSpec(prefix="solver_statistics_times_solve", scope=scope, dimension="s") for scope in suffixes]
            + [ColumnSpec(prefix="solver_statistics_times_unsat", scope=scope, dimension="s") for scope in suffixes]
            + [ColumnSpec(prefix="solver_statistics_times_sat", scope=scope, dimension="s") for scope in suffixes],
            title_text=f"asp absolute total solver times:\n" f" solver should spend most of the time solving: comparison total_time time and solve_time ",
            output_folder=output_folder,
            marker_color=marker_color_scope,
            marker_symbol=marker_symbol_solver_times,
            one_field_many_scopes=False,
            binned=False,
            height=1200,
        )

    # 004_ratio_asp_solve_propagation: propagation times should be low in comparison to solve times
    plot_binned_box_plot(
        experiment_data=experiment_data,
        axis_of_interest="experiment_id",
        cols=[ColumnSpec(prefix="summed_user_accu_propagations", scope=scope, dimension="s") for scope in suffixes]
        + [ColumnSpec(prefix="solver_statistics_times_solve", scope=scope, dimension="s") for scope in suffixes],
        title_text=f"ratio asp solve propagation:\n"
        "propagation times should be low in comparison to solve times: "
        f"compare solve_time (b) against summed propagation times of user accu",
        output_folder=output_folder,
        marker_color=marker_color_scope,
        one_field_many_scopes=False,
        binned=False,
    )
    plot_binned_box_plot(
        experiment_data=experiment_data,
        axis_of_interest="experiment_id",
        cols=[ColumnSpec(prefix="summed_user_step_propagations", scope=scope, dimension="s") for scope in suffixes]
        + [ColumnSpec(prefix="solver_statistics_times_solve", scope=scope, dimension="s") for scope in suffixes],
        title_text="ratio asp solve propagation:\n"
        "propagation times should be low in comparison to solve times: "
        f"comparison of solve_time (b) against summed propagation times of user step",
        output_folder=output_folder,
        binned=False,
    )

    # 005_ratio_asp_conflict_choice: choice conflict ratio should be close to 1;
    # if the ratio is high, the problem might be large, but not difficult
    plot_binned_box_plot(
        experiment_data=experiment_data,
        axis_of_interest="experiment_id",
        cols=[ColumnSpec(prefix="choice_conflict_ratio", scope=scope) for scope in suffixes],
        title_text=f"ratio asp conflict choice: choice conflict ratio should be small; "
        f"if the ratio is high, the problem might be large, but not difficult; "
        f"choice conflict ratio",
        output_folder=output_folder,
        binned=False,
    )
