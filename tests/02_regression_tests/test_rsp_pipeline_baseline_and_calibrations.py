"""Run tests for different experiment methods."""
from rsp.pipeline.rsp_pipeline_baseline_and_calibrations import rsp_pipeline_baseline_and_calibrations
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import InfrastructureParametersRange
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ReScheduleParametersRange
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ScheduleParametersRange
from rsp.step_05_experiment_run.experiment_results_analysis_load_and_save import load_and_filter_experiment_results_analysis_online_unrestricted
from rsp.step_05_experiment_run.experiment_run import create_experiment_folder_name
from rsp.step_05_experiment_run.experiment_run import delete_experiment_folder
from rsp.step_05_experiment_run.experiment_run import load_and_filter_experiment_results_analysis
from rsp.step_06_analysis.compute_time_analysis.compute_time_analysis import hypothesis_one_analysis_visualize_computational_time_comparison
from rsp.step_06_analysis.plot_utils import ColumnSpec


# TODO reduce size
def test_rsp_pipeline_baseline_and_calibrations():
    experiment_name = "test_calibrations"
    base_directory = "target/" + create_experiment_folder_name(experiment_name)
    try:
        baseline_data_folder = rsp_pipeline_baseline_and_calibrations(
            infra_parameters_range=InfrastructureParametersRange(
                number_of_agents=[10, 10, 1],
                width=[60, 60, 1],
                height=[60, 60, 1],
                flatland_seed_value=[10, 10, 1],
                max_num_cities=[4, 16, 1],
                max_rail_in_city=[3, 3, 1],
                max_rail_between_cities=[1, 1, 1],
                number_of_shortest_paths_per_agent=[10, 10, 1],
            ),
            schedule_parameters_range=ScheduleParametersRange(asp_seed_value=[1, 104, 1], number_of_shortest_paths_per_agent_schedule=[1, 1, 1],),
            reschedule_parameters_range=ReScheduleParametersRange(
                earliest_malfunction=[30, 30, 1],
                malfunction_duration=[50, 50, 1],
                # take all agents (200 is larger than largest number of agents)
                malfunction_agent_id=[0, 200, 1],
                number_of_shortest_paths_per_agent=[10, 10, 1],
                max_window_size_from_earliest=[60, 60, 1],
                asp_seed_value=[99, 99, 1],
                # route change is penalized the same as 30 seconds delay
                weight_route_change=[30, 30, 1],
                weight_lateness_seconds=[1, 1, 1],
            ),
            base_directory=base_directory,
            experiment_output_base_directory=None,
        )
        print(baseline_data_folder)
        experiment_data_baseline = load_and_filter_experiment_results_analysis(experiment_base_directory=baseline_data_folder,)
        assert len(experiment_data_baseline) == 1
        suffixes = ["with_SEQ", "with_delay_model_resolution_2", "with_delay_model_resolution_5", "with_delay_model_resolution_10", "without_propagate_partial"]
        for suffix in suffixes:
            experiment_data_comparison = load_and_filter_experiment_results_analysis_online_unrestricted(
                experiment_base_directory=baseline_data_folder.replace("baseline", suffix), from_individual_csv=True
            )
            assert len(experiment_data_comparison) == 1

            hypothesis_one_analysis_visualize_computational_time_comparison(
                experiment_data=experiment_data_baseline,
                experiment_data_comparison=experiment_data_comparison,
                columns_of_interest=[ColumnSpec(prefix="solver_statistics_times_total", scope="online_unrestricted", dimension="s")],
                experiment_data_suffix="_baseline",
                experiment_data_comparison_suffix=f"_{suffix}",
                output_folder=base_directory + f"/baseline_{suffix}",
            )
    finally:
        delete_experiment_folder(base_directory)
