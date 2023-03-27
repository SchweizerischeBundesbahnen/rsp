from functools import partial

from pandas import DataFrame

from rsp.global_data_configuration import BASELINE_DATA_FOLDER
from rsp.global_data_configuration import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.step_05_experiment_run.experiment_malfunction import gen_malfunction
from rsp.step_05_experiment_run.experiment_results_analysis import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.step_05_experiment_run.experiment_results_analysis import filter_experiment_results_analysis_data_frame
from rsp.step_05_experiment_run.experiment_results_analysis import rescheduling_scopes_visualization
from rsp.step_05_experiment_run.experiment_results_analysis import speed_up_scopes_visualization
from rsp.step_05_experiment_run.experiment_run import load_and_expand_experiment_results_from_data_folder
from rsp.step_05_experiment_run.experiment_run import load_data_from_individual_csv_in_data_folder
from rsp.step_05_experiment_run.experiment_run import load_experiment_agenda_from_file
from rsp.step_06_analysis.compute_time_analysis.compute_time_analysis import hypothesis_one_analysis_visualize_agenda
from rsp.step_06_analysis.detailed_experiment_analysis.detailed_experiment_analysis import plot_costs
from rsp.step_06_analysis.detailed_experiment_analysis.time_resource_plots_from_data_frames import extract_full_df_from_experiment_results
from rsp.step_06_analysis.detailed_experiment_analysis.time_resource_plots_from_data_frames import time_resource_graph_from_df
from rsp.step_06_analysis.plot_utils import ColumnSpec
from rsp.step_06_analysis.plot_utils import marker_color_scope
from rsp.step_06_analysis.plot_utils import plot_binned_box_plot


def main(experiment_base_directory: str = BASELINE_DATA_FOLDER, from_individual_csv: bool = True, experiments_of_interest=None):
    # ==============================================================================================================
    # chapter 4: computational results
    # ==============================================================================================================
    if from_individual_csv:
        experiment_data: DataFrame = load_data_from_individual_csv_in_data_folder(
            experiment_data_folder_name=f"{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}", experiment_ids=experiments_of_interest
        )

    else:
        _, experiment_results_analysis_list = load_and_expand_experiment_results_from_data_folder(
            experiment_data_folder_name=f"{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}", experiment_ids=experiments_of_interest,
        )
        experiment_data: DataFrame = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_analysis_list)

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
        file_name="prediction_quality_false_positive.pdf",
        cols=[
            ColumnSpec(prefix=prediction_col, scope=scope)
            for scope in ["online_transmission_chains_route_restricted", "online_random_average"]
            for prediction_col in ["predicted_changed_agents_false_positives_percentage"]
        ],
        title_text="Prediction Quality Percentage",
        marker_color=marker_color_scope,
        marker_symbol=marker_symbol_positive_negative,
        one_field_many_scopes=False,
    )

    plot_binned_box_plot(
        experiment_data=experiment_data_filtered,
        axis_of_interest="experiment_id",
        axis_of_interest_dimension="",
        output_folder=output_folder,
        file_name="prediction_quality_false_negative.pdf",
        cols=[
            ColumnSpec(prefix=prediction_col, scope=scope)
            for scope in ["online_transmission_chains_route_restricted", "online_random_average"]
            for prediction_col in [ "predicted_changed_agents_false_negatives_percentage"]
        ],
        title_text="Prediction Quality Percentage",
        marker_color=marker_color_scope,
        marker_symbol=marker_symbol_positive_negative,
        one_field_many_scopes=False,
    )
    "plot F1 Score"
    "Compute F1 Score"
    for scope in ["online_random_average", "online_transmission_chains_route_restricted"]:
        print(scope)

        f_n = experiment_data_filtered["predicted_changed_agents_false_negatives_" + scope]
        f_p = experiment_data_filtered["predicted_changed_agents_false_positives_" + scope]
        t_p = experiment_data_filtered["predicted_changed_agents_number_" + scope] - experiment_data_filtered[
            "predicted_changed_agents_false_positives_" + scope]
        experiment_data_filtered["f1_" + scope] = t_p / (t_p + 0.5 * (f_p + f_n))
    "plot Score"
    plot_binned_box_plot(
        experiment_data=experiment_data_filtered,
        axis_of_interest="experiment_id",
        axis_of_interest_dimension="",
        output_folder=output_folder,
        file_name="prediction_quality_f1_score.pdf",
        cols=[ColumnSpec("f1", "online_random_average"), ColumnSpec("f1", "online_transmission_chains_route_restricted")],
        title_text="F_1 Score Scoper",
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
            binned=False,
        )

        plot_binned_box_plot(
            experiment_data=experiment_data_filtered,
            axis_of_interest="solver_statistics_times_total_online_unrestricted",
            cols=[ColumnSpec(prefix="experiment_id")],
            output_folder=output_folder,
            title_text="Computational times bin histogram",
            file_name="times_total_histogram.pdf",
            show_bin_counts=True,
            data_instead_of_box=True,
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
    print("Plotted results")
    # ==============================================================================================================
    # chapter 5: case studies
    # ==============================================================================================================
    output_folder = "doc/overleaf/Figures/05_use_cases"

    # ==============================================================================================================
    # chapter 5.1: cost equivalence
    # ==============================================================================================================
    experiment_results_list, experiment_results_analysis_list = load_and_expand_experiment_results_from_data_folder(
        experiment_data_folder_name=f"{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}", experiment_ids=[342],
    )
    print(experiment_results_analysis_list)
    exp_results_analysis_of_experiment_of_interest = experiment_results_analysis_list[0]

    plot_costs(
        experiment_results_analysis=exp_results_analysis_of_experiment_of_interest,
        output_folder=output_folder,
        scopes=["online_unrestricted", "online_route_restricted"],
    )

    exp_results_of_experiment_of_interest = experiment_results_list[0]
    experiment_agenda = load_experiment_agenda_from_file(experiment_base_directory)

    full_df, df_background, sorted_resource_occupations_per_scope_and_agent = extract_full_df_from_experiment_results(
        exp_results_of_experiment_of_interest=exp_results_of_experiment_of_interest, global_constants=experiment_agenda.global_constants
    )

    agents_of_interest = [20, 40, 31, 18]
    time_steps_interval_of_interest = [900, 1720]
    scopes_of_interest = ["online_unrestricted"]
    num_agents = exp_results_of_experiment_of_interest.experiment_parameters.infra_parameters.number_of_agents

    malfunction = gen_malfunction(
        earliest_malfunction=exp_results_of_experiment_of_interest.experiment_parameters.re_schedule_parameters.earliest_malfunction,
        malfunction_duration=exp_results_of_experiment_of_interest.experiment_parameters.re_schedule_parameters.malfunction_duration,
        malfunction_agent_id=exp_results_of_experiment_of_interest.experiment_parameters.re_schedule_parameters.malfunction_agent_id,
        schedule_trainruns=exp_results_of_experiment_of_interest.results_schedule.trainruns_dict,
    )

    time_resource_graph_from_df(
        full_df=full_df,
        scopes=scopes_of_interest,
        time_step_interval=time_steps_interval_of_interest,
        num_agents=num_agents,
        agents_of_interest=agents_of_interest,
        malfunction=malfunction,
        output_folder=output_folder,
        file_name=f"time_resource_{scopes_of_interest[0]}.pdf",
    )
    scopes_of_interest = ["online_route_restricted"]
    time_resource_graph_from_df(
        full_df=full_df,
        scopes=scopes_of_interest,
        time_step_interval=time_steps_interval_of_interest,
        num_agents=num_agents,
        agents_of_interest=agents_of_interest,
        malfunction=malfunction,
        output_folder=output_folder,
        file_name=f"time_resource_{scopes_of_interest[0]}.pdf",
    )


if __name__ == "__main__":
    main()
