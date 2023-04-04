"""Rendering methods to use with jupyter notebooks."""
from typing import List
from typing import Optional

from pandas import DataFrame

from rsp.step_05_experiment_run.experiment_results_analysis import all_scopes_visualization
from rsp.step_05_experiment_run.experiment_results_analysis import prediction_scopes_visualization
from rsp.step_05_experiment_run.experiment_results_analysis import speed_up_scopes_visualization
from rsp.step_06_analysis.plot_utils import ColumnSpec
from rsp.step_06_analysis.plot_utils import density_hist_plot_2d
from rsp.step_06_analysis.plot_utils import marker_color_scope
from rsp.step_06_analysis.plot_utils import plot_binned_box_plot


def hypothesis_one_analysis_visualize_agenda(experiment_data: DataFrame, output_folder: Optional[str] = None, file_name: Optional[str] = None):
    plot_binned_box_plot(
        experiment_data=experiment_data,
        axis_of_interest="experiment_id",
        cols=[
            ColumnSpec(prefix="n_agents"),
            ColumnSpec(prefix="infra_id"),
            ColumnSpec(prefix="schedule_id"),
            ColumnSpec(prefix="size"),
            ColumnSpec(prefix="earliest_malfunction"),
            ColumnSpec(prefix="malfunction_duration"),
            ColumnSpec(prefix="malfunction_agent_id"),
        ],
        title_text="Agenda overview",
        output_folder=output_folder,
        file_name=file_name,
        one_field_many_scopes=False,
        binned=False,
        height=800,
        data_instead_of_box=True,
    )


def hypothesis_one_analysis_visualize_computational_time_comparison(
    experiment_data: DataFrame,
    output_folder: str = None,
    columns_of_interest: List[ColumnSpec] = None,
    experiment_data_comparison: DataFrame = None,
    experiment_data_suffix: str = None,
    experiment_data_comparison_suffix: str = None,
):
    if columns_of_interest is None:
        columns_of_interest = [ColumnSpec(prefix="solver_statistics_times_total", scope=scope) for scope in all_scopes_visualization]
    for axis_of_interest in [
        "experiment_id",
        "n_agents_running",
        "n_agents",
        "size",
        "rescheduling_horizon",
        "solver_statistics_times_total_online_unrestricted",
    ]:
        plot_binned_box_plot(
            experiment_data=experiment_data,
            experiment_data_comparison=experiment_data_comparison,
            axis_of_interest=axis_of_interest,
            cols=columns_of_interest,
            output_folder=output_folder,
            title_text=f"Computational times per {axis_of_interest}",
            one_field_many_scopes=True,
            height=1000,
            width=1200,
            binned=False,
            experiment_data_suffix=experiment_data_suffix,
            experiment_data_comparison_suffix=experiment_data_comparison_suffix,
        )


def hypothesis_one_analysis_visualize_lateness(experiment_data: DataFrame, output_folder: str = None):
    for axis_of_interest, axis_of_interest_suffix in {"infra_id_schedule_id": "", "experiment_id": ""}.items():
        for speed_up_col_pattern, title_text in [
            ("costs", "Costs"),
            ("additional_costs", "Additional costs"),
            ("costs_ratio", "Costs ratio"),
            ("lateness", "Lateness (unweighted)"),
            ("additional_lateness", "Additional lateness (unweighted)"),
            ("costs_from_route_section_penalties", "Weighted costs from route section penalties"),
            ("additional_costs_from_route_section_penalties", "Additional weighted costs from route section penalties"),
        ]:
            plot_binned_box_plot(
                experiment_data=experiment_data,
                axis_of_interest=axis_of_interest,
                axis_of_interest_dimension=axis_of_interest_suffix,
                output_folder=output_folder,
                cols=[ColumnSpec(prefix=speed_up_col_pattern, scope=speed_up_series) for speed_up_series in speed_up_scopes_visualization],
                title_text=title_text,
                one_field_many_scopes=True,
                marker_color=marker_color_scope,
            )


def hypothesis_one_analysis_prediction_quality(experiment_data: DataFrame, output_folder: str = None):
    for axis_of_interest, axis_of_interest_suffix in {"experiment_id": "", "infra_id_schedule_id": ""}.items():
        plot_binned_box_plot(
            experiment_data=experiment_data,
            axis_of_interest=axis_of_interest,
            axis_of_interest_dimension=axis_of_interest_suffix,
            output_folder=output_folder,
            cols=[ColumnSpec(prefix="n_agents"), ColumnSpec(prefix="changed_agents", scope="online_unrestricted")]
            + [
                ColumnSpec(prefix=prediction_col, scope=scope)
                for scope in prediction_scopes_visualization
                for prediction_col in [
                    "changed_agents",
                    "predicted_changed_agents_number",
                    "predicted_changed_agents_false_positives",
                    "predicted_changed_agents_false_negatives",
                ]
            ],
            title_text="Prediction Quality Counts",
        )
        plot_binned_box_plot(
            experiment_data=experiment_data,
            axis_of_interest=axis_of_interest,
            axis_of_interest_dimension=axis_of_interest_suffix,
            output_folder=output_folder,
            cols=[ColumnSpec(prefix="changed_agents_percentage", scope="online_unrestricted")]
            + [
                ColumnSpec(prefix=prediction_col, scope=scope)
                for scope in prediction_scopes_visualization
                for prediction_col in [
                    "changed_agents_percentage",
                    "predicted_changed_agents_percentage",
                    "predicted_changed_agents_false_positives_percentage",
                    "predicted_changed_agents_false_negatives_percentage",
                ]
            ],
            title_text="Prediction Quality Percentage",
        )


def hypothesis_one_analysis_scope_quality(experiment_data: DataFrame, output_folder: str = None):
    for axis_of_interest, axis_of_interest_suffix in {"experiment_id": "", "infra_id_schedule_id": ""}.items():
        plot_binned_box_plot(
            experiment_data=experiment_data,
            axis_of_interest=axis_of_interest,
            axis_of_interest_dimension=axis_of_interest_suffix,
            output_folder=output_folder,
            cols=[ColumnSpec(prefix="n_agents"), ColumnSpec(prefix="changed_agents", scope="online_unrestricted")]
            + [ColumnSpec(prefix=prediction_col, scope=scope) for scope in prediction_scopes_visualization for prediction_col in ["prediction_quality"]],
            title_text="Scope Quality",
        )
        plot_binned_box_plot(
            experiment_data=experiment_data,
            axis_of_interest=axis_of_interest,
            axis_of_interest_dimension=axis_of_interest_suffix,
            output_folder=output_folder,
            cols=[ColumnSpec(prefix="changed_agents_percentage", scope="online_unrestricted")]
            + [
                ColumnSpec(prefix=prediction_col, scope=scope)
                for scope in prediction_scopes_visualization
                for prediction_col in [
                    "changed_agents_percentage",
                    "predicted_changed_agents_percentage",
                    "predicted_changed_agents_false_positives_percentage",
                    "predicted_changed_agents_false_negatives_percentage",
                ]
            ],
            title_text="Scope Quality",
        )


def hypothesis_one_analysis_visualize_speed_up(
    experiment_data: DataFrame, output_folder: str = None, nb_bins: Optional[int] = 10, show_bin_counts: bool = False
):
    for axis_of_interest, axis_of_interest_suffix in {"experiment_id": None, "solver_statistics_times_total_online_unrestricted": "s"}.items():
        for speed_up_col_pattern, title_text in [
            ("speed_up", "Speed-up full solver time"),
            ("speed_up_solve_time", "Speed-up solver time solving only"),
            ("speed_up_non_solve_time", "Speed-up solver time non-processing (grounding etc.)"),
        ]:
            plot_binned_box_plot(
                experiment_data=experiment_data,
                axis_of_interest=axis_of_interest,
                axis_of_interest_dimension=axis_of_interest_suffix,
                output_folder=output_folder,
                cols=[ColumnSpec(prefix=speed_up_col_pattern, scope=speed_up_series) for speed_up_series in speed_up_scopes_visualization],
                title_text=title_text,
                nb_bins=nb_bins,
                show_bin_counts=show_bin_counts,
                one_field_many_scopes=True,
                marker_color=marker_color_scope,
            )


def hypothesis_one_analysis_visualize_changed_agents(experiment_data: DataFrame, output_folder: str = None):
    for axis_of_interest, axis_of_interest_suffix in {"infra_id_schedule_id": "", "solver_statistics_times_total_online_unrestricted": "[s]"}.items():
        for speed_up_col_pattern, title_text in [
            ("changed_agents", "Number of changed agents"),
            ("additional_changed_agents", "Additional number of changed agents"),
            ("changed_agents_percentage", "Percentage of changed agents"),
        ]:
            plot_binned_box_plot(
                experiment_data=experiment_data,
                axis_of_interest=axis_of_interest,
                axis_of_interest_dimension=axis_of_interest_suffix,
                output_folder=output_folder,
                cols=[ColumnSpec(prefix=speed_up_col_pattern, scope=speed_up_series) for speed_up_series in speed_up_scopes_visualization],
                title_text=title_text,
                one_field_many_scopes=True,
                marker_color=marker_color_scope,
            )


def speed_up_vs_performance(experiment_data: DataFrame, output_folder: str = None):
    for scoper in prediction_scopes_visualization:
        density_hist_plot_2d(title=scoper, data_frame=experiment_data, output_folder=output_folder)
