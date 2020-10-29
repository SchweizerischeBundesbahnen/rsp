from functools import partial

from pandas import DataFrame
from rsp.step_03_run.experiment_results_analysis import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.step_03_run.experiment_results_analysis import filter_experiment_results_analysis_data_frame
from rsp.step_03_run.experiment_results_analysis import rescheduling_scopes_visualization
from rsp.step_03_run.experiment_results_analysis import speed_up_scopes_visualization
from rsp.step_03_run.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.step_03_run.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.step_03_run.experiments import load_data_from_individual_csv_in_data_folder
from rsp.step_04_analysis.plot_utils import plot_binned_box_plot
from rsp.step_04_analysis.plot_utils import plot_box_plot
from rsp.utils.global_data_configuration import BASELINE_DATA_FOLDER


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
    plot_box_plot(
        experiment_data=experiment_data_filtered,
        axis_of_interest="experiment_id",
        columns_of_interest=["n_agents", "infra_id", "schedule_id", "size", "earliest_malfunction", "malfunction_duration", "malfunction_agent_id"],
        title="Number of agents",
        y_axis_title="[-]",
        color_offset=1,
        output_folder=output_folder,
        file_name="agenda.pdf",
    )

    plot_binned_box_plot(
        experiment_data=experiment_data,
        axis_of_interest="infra_id_schedule_id",
        cols=["additional_changed_agents_{}".format(speed_up_series) for speed_up_series in speed_up_scopes_visualization],
        y_axis_title="Additional number of changed agents [-]",
        output_folder=output_folder,
        file_name="additional_changed_agents.pdf",
        show_bin_counts=False,
    )

    plot_binned_box_plot(
        experiment_data=experiment_data,
        axis_of_interest="infra_id_schedule_id",
        cols=["additional_costs_{}".format(speed_up_series) for speed_up_series in speed_up_scopes_visualization],
        y_axis_title="Additional costs [-]",
        output_folder=output_folder,
        file_name="additional_costs.pdf",
        show_bin_counts=False,
    )

    plot_binned_box_plot(
        experiment_data=experiment_data,
        axis_of_interest="infra_id_schedule_id",
        cols=["additional_lateness_{}".format(speed_up_series) for speed_up_series in speed_up_scopes_visualization],
        y_axis_title="Additional (unweighted) lateness [-]",
        output_folder=output_folder,
        file_name="additional_lateness.pdf",
        show_bin_counts=False,
    )

    plot_binned_box_plot(
        experiment_data=experiment_data,
        axis_of_interest="infra_id_schedule_id",
        cols=["additional_costs_from_route_section_penalties_{}".format(speed_up_series) for speed_up_series in speed_up_scopes_visualization],
        y_axis_title="Additional weighted costs from route section penalties [-]",
        output_folder=output_folder,
        file_name="additional_route_section_penalties.pdf",
        show_bin_counts=False,
    )

    plot_binned_box_plot(
        experiment_data=experiment_data,
        axis_of_interest="experiment_id",
        axis_of_interest_suffix="",
        output_folder=output_folder,
        file_name="prediction_quality.pdf",
        cols=["changed_agents_percentage_online_unrestricted"]
        + [
            prediction_col + "_" + scope
            for scope in ["online_transmission_chains_route_restricted", "online_random_average"]
            for prediction_col in ["predicted_changed_agents_false_positives_percentage", "predicted_changed_agents_false_negatives_percentage"]
        ],
        y_axis_title="Prediction Quality Percentage",
    )

    for speed_up_scope in rescheduling_scopes_visualization:
        suffixes = [speed_up_scope]
        plot_box_plot(
            experiment_data=experiment_data,
            axis_of_interest="experiment_id",
            columns_of_interest=[f"solver_statistics_times_total_" + item for item in suffixes]
            + [f"solver_statistics_times_total_without_solve_" + item for item in suffixes]
            + [f"solver_statistics_times_solve_" + item for item in suffixes],
            title=f"Comparison of total solver time spent for solving and non-solving (grounding etc.)",
            output_folder=output_folder,
            file_name=f"solve_non_solve_{speed_up_scope}.pdf",
        )

        for speed_up_col_pattern, y_axis_title in [
            ("speed_up_{}", "Speed-up full solver time [-]"),
            ("speed_up_solve_time_{}", "Speed-up solver time solving only [-]"),
            ("speed_up_non_solve_time_{}", "Speed-up solver time non-processing (grounding etc.) [-]"),
        ]:
            plot_binned_box_plot(
                experiment_data=experiment_data,
                axis_of_interest="solver_statistics_times_total_online_unrestricted",
                axis_of_interest_suffix="[s]",
                output_folder=output_folder,
                cols=[speed_up_col_pattern.format(speed_up_series) for speed_up_series in speed_up_scopes_visualization],
                y_axis_title=y_axis_title,
                nb_bins=10,
                file_name=f"{speed_up_col_pattern.format('')}per_times_total_online_unrestricted.pdf",
                show_bin_counts=False,
            )

        plot_box_plot(
            experiment_data=experiment_data,
            axis_of_interest="experiment_id",
            columns_of_interest=["solver_statistics_times_total_online_unrestricted"],
            output_folder=output_folder,
            title="Computational times per experiment_id",
            y_axis_title="solver_statistics_times_total_online_unrestricted [s]",
            file_name="times_total_per_experiment_id.pdf",
        )
        plot_box_plot(
            experiment_data=experiment_data,
            axis_of_interest="solver_statistics_times_total_online_unrestricted",
            columns_of_interest=[
                f"solver_statistics_times_total_{scope}"
                for scope in ["online_unrestricted", "offline_fully_restricted", "offline_delta", "online_route_restricted"]
            ],
            output_folder=output_folder,
            file_name="times_total_per_times_total_online_unrestricted_1.pdf",
            y_axis_title="solver_statistics_times_total [s]",
        )
        plot_box_plot(
            experiment_data=experiment_data,
            axis_of_interest="solver_statistics_times_total_online_unrestricted",
            columns_of_interest=[
                f"solver_statistics_times_total_{scope}"
                for scope in [
                    "online_unrestricted",
                    "online_transmission_chains_fully_restricted",
                    "online_transmission_chains_route_restricted",
                    "online_random_average",
                ]
            ],
            output_folder=output_folder,
            file_name="times_total_per_times_total_online_unrestricted_2.pdf",
            y_axis_title="solver_statistics_times_total [s]",
        )


if __name__ == "__main__":
    main()
