from functools import partial

from pandas import DataFrame

from rsp.step_03_run.experiment_results_analysis import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.step_03_run.experiment_results_analysis import filter_experiment_results_analysis_data_frame
from rsp.step_03_run.experiment_results_analysis import rescheduling_scopes_visualization
from rsp.step_03_run.experiment_results_analysis import speed_up_scopes_visualization
from rsp.step_03_run.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.step_03_run.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.step_03_run.experiments import load_data_from_individual_csv_in_data_folder
from rsp.step_04_analysis.compute_time_analysis.compute_time_analysis import hypothesis_one_analysis_visualize_agenda
from rsp.step_04_analysis.plot_utils import ColumnSpec
from rsp.step_04_analysis.plot_utils import marker_color_scope
from rsp.step_04_analysis.plot_utils import plot_binned_box_plot
from rsp.utils.global_data_configuration import BASELINE_DATA_FOLDER
from rsp.utils.rsp_logger import rsp_logger, VERBOSE


def main(experiment_base_directory: str = BASELINE_DATA_FOLDER, from_individual_csv: bool = True, experiments_of_interest=None):
    if from_individual_csv:
        experiment_data: DataFrame = load_data_from_individual_csv_in_data_folder(
            experiment_data_folder_name=f"{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}", experiment_ids=experiments_of_interest
        )

    else:
        experiment_results_list = load_and_expand_experiment_results_from_data_folder(
            experiment_data_folder_name=f"{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}",
            experiment_ids=experiments_of_interest,
            nonify_all_structured_fields=True,
        )
        experiment_data: DataFrame = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_list)
    local_filter_experiment_results_analysis_data_frame = partial(
        filter_experiment_results_analysis_data_frame, min_time_online_unrestricted=20, max_time_online_unrestricted=200, max_time_online_unrestricted_q=1.0
    )
    experiment_data_filtered = local_filter_experiment_results_analysis_data_frame(experiment_data)

    output_folder = "doc/overleaf/Figures/04_computational_results"

    hypothesis_one_analysis_visualize_agenda(experiment_data=experiment_data_filtered, output_folder=output_folder, file_name="agenda.pdf")

    def marker_symbol_positive_negative(index, column):
        if "positive" in column:
            return "cross"
        elif "negative" in column:
            return "line-ns"
        else:
            return "circle"

    plot_binned_box_plot(
        experiment_data=experiment_data_filtered,
        axis_of_interest="infra_id_schedule_id",
        title_text="Additional changed agents per schedule",
        cols=[ColumnSpec(prefix="additional_changed_agents", scope=speed_up_series) for speed_up_series in speed_up_scopes_visualization],
        output_folder=output_folder,
        file_name="additional_changed_agents.pdf",
        show_bin_counts=False,
        marker_color=marker_color_scope,
        one_field_many_scopes=True,
    )

    plot_binned_box_plot(
        experiment_data=experiment_data_filtered,
        axis_of_interest="infra_id_schedule_id",
        cols=[ColumnSpec(prefix="additional_costs", scope=speed_up_series) for speed_up_series in speed_up_scopes_visualization],
        output_folder=output_folder,
        title_text="Additional costs per schedule",
        file_name="additional_costs.pdf",
        show_bin_counts=False,
        marker_color=marker_color_scope,
        one_field_many_scopes=True,
    )

    plot_binned_box_plot(
        experiment_data=experiment_data_filtered,
        axis_of_interest="infra_id_schedule_id",
        cols=[ColumnSpec(prefix="additional_lateness", scope=speed_up_series) for speed_up_series in speed_up_scopes_visualization],
        title_text="Additional (unweighted) lateness per schedule",
        output_folder=output_folder,
        file_name="additional_lateness.pdf",
        show_bin_counts=False,
        marker_color=marker_color_scope,
        one_field_many_scopes=True,
    )

    plot_binned_box_plot(
        experiment_data=experiment_data_filtered,
        axis_of_interest="infra_id_schedule_id",
        cols=[ColumnSpec(prefix="additional_costs_from_route_section_penalties", scope=speed_up_series) for speed_up_series in speed_up_scopes_visualization],
        title_text="Additional weighted costs from route section penalties per schedule",
        output_folder=output_folder,
        file_name="additional_route_section_penalties.pdf",
        show_bin_counts=False,
        marker_color=marker_color_scope,
        one_field_many_scopes=True,
    )

    plot_binned_box_plot(
        experiment_data=experiment_data_filtered,
        axis_of_interest="experiment_id",
        axis_of_interest_dimension="",
        output_folder=output_folder,
        file_name="prediction_quality.pdf",
        cols=[ColumnSpec(prefix="changed_agents_percentage", scope="online_unrestricted")]
             + [
                 ColumnSpec(prefix=prediction_col, scope=scope)
                 for scope in ["online_transmission_chains_route_restricted", "online_random_average"]
                 for prediction_col in ["predicted_changed_agents_false_positives_percentage", "predicted_changed_agents_false_negatives_percentage"]
             ],
        title_text="Prediction Quality Percentage",
        marker_color=marker_color_scope,
        marker_symbol=marker_symbol_positive_negative,
        one_field_many_scopes=False,
    )

    for speed_up_scope in rescheduling_scopes_visualization:
        plot_binned_box_plot(
            experiment_data=experiment_data_filtered,
            axis_of_interest="experiment_id",
            cols=[
                ColumnSpec(prefix=prefix, scope=speed_up_scope, dimension="s")
                for prefix in ["solver_statistics_times_total", "solver_statistics_times_total_without_solve", "solver_statistics_times_solve"]
            ],
            title_text=f"Comparison of total solver time spent for solving and non-solving (grounding etc.)",
            output_folder=output_folder,
            file_name=f"solve_non_solve_{speed_up_scope}.pdf",
            one_field_many_scopes=False,
        )

        for speed_up_col, title_text in [
            ("speed_up", "Speed-up full solver time"),
            ("speed_up_solve_time", "Speed-up solver time solving only"),
            ("speed_up_non_solve_time", "Speed-up solver time non-processing (grounding etc.)"),
        ]:
            plot_binned_box_plot(
                experiment_data=experiment_data_filtered,
                axis_of_interest="solver_statistics_times_total_online_unrestricted",
                axis_of_interest_dimension="s",
                output_folder=output_folder,
                cols=[ColumnSpec(prefix=speed_up_col, scope=speed_up_series) for speed_up_series in speed_up_scopes_visualization],
                nb_bins=10,
                file_name=f"{speed_up_col}_per_times_total_online_unrestricted.pdf",
                show_bin_counts=False,
                marker_color=marker_color_scope,
                title_text=title_text,
                one_field_many_scopes=True,
            )

        plot_binned_box_plot(
            experiment_data=experiment_data_filtered,
            axis_of_interest="experiment_id",
            cols=[ColumnSpec(prefix="solver_statistics_times_total", scope="online_unrestricted")],
            output_folder=output_folder,
            title_text="Computational times per experiment_id",
            file_name="times_total_per_experiment_id.pdf",
            marker_color=marker_color_scope,
            one_field_many_scopes=True,
            binned=False
        )

        plot_binned_box_plot(
            experiment_data=experiment_data_filtered,
            axis_of_interest="solver_statistics_times_total_online_unrestricted",
            cols=[ColumnSpec(prefix="experiment_id")],
            output_folder=output_folder,
            title_text="Computational times bin histogram",
            file_name="times_total_histogram.pdf",
            show_bin_counts=True
        )

        plot_binned_box_plot(
            experiment_data=experiment_data_filtered,
            axis_of_interest="solver_statistics_times_total_online_unrestricted",
            cols=[
                ColumnSpec(prefix="solver_statistics_times_total", scope=scope, dimension="s")
                for scope in ["online_unrestricted", "offline_fully_restricted", "offline_delta", "online_route_restricted"]
            ],
            output_folder=output_folder,
            file_name="times_total_per_times_total_online_unrestricted_1.pdf",
            title_text="Total solver time per solver_statistics_times_total_online_unrestricted (1)",
            marker_color=marker_color_scope,
        )
        plot_binned_box_plot(
            experiment_data=experiment_data_filtered,
            axis_of_interest="solver_statistics_times_total_online_unrestricted",
            cols=[
                ColumnSpec(prefix="solver_statistics_times_total", scope=scope, dimension="s")
                for scope in [
                    "online_unrestricted",
                    "online_transmission_chains_fully_restricted",
                    "online_transmission_chains_route_restricted",
                    "online_random_average",
                ]
            ],
            output_folder=output_folder,
            file_name="times_total_per_times_total_online_unrestricted_2.pdf",
            title_text="Total solver time per solver_statistics_times_total_online_unrestricted (2)",
            marker_color=marker_color_scope,
        )


if __name__ == "__main__":
    main()
