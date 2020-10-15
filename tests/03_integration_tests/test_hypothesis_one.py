import numpy as np
from rsp.hypothesis_one_pipeline_all_in_one import hypothesis_one_pipeline_all_in_one
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import ParameterRangesAndSpeedData
from rsp.utils.experiments import create_experiment_folder_name
from rsp.utils.experiments import delete_experiment_folder


def test_hypothesis_one_all_in_one():
    """Run hypothesis one with qualitative analysis and potassco export and
    check that expected files are present wihout inspecting them."""

    experiment_base_directory = create_experiment_folder_name("test_hypothesis_one_all_in_one")
    try:
        hypothesis_one_pipeline_all_in_one(
            parameter_ranges_and_speed_data=ParameterRangesAndSpeedData(
                parameter_ranges=ParameterRanges(agent_range=[5, 5, 1],
                                                 size_range=[30, 30, 1],
                                                 in_city_rail_range=[6, 6, 1],
                                                 out_city_rail_range=[2, 2, 1],
                                                 city_range=[20, 20, 1],
                                                 earliest_malfunction=[20, 20, 1],
                                                 malfunction_duration=[20, 20, 1],
                                                 number_of_shortest_paths_per_agent=[10, 10, 1],
                                                 max_window_size_from_earliest=[np.inf, np.inf, 1],
                                                 asp_seed_value=[94, 94, 1],
                                                 weight_route_change=[60, 60, 1],
                                                 weight_lateness_seconds=[1, 1, 1]),
                speed_data={1.: 0.25,  # Fast passenger train
                            1. / 2.: 0.25,  # Fast freight train
                            1. / 3.: 0.25,  # Slow commuter train
                            1. / 4.: 0.25}  # Slow freight train

            ),
            asp_export_experiment_ids=[0],
            qualitative_analysis_experiment_ids=[0],
            experiment_name="test_hypothesis_one",
            run_analysis=False,
            experiment_base_directory=experiment_base_directory
        )
    finally:
        delete_experiment_folder(experiment_base_directory)


def test_parallel_experiment_execution():
    """Run a parallel experiment agenda."""
    experiment_folder_name = create_experiment_folder_name("test_parallel_experiment_execution")
    try:
        hypothesis_one_pipeline_all_in_one(
            parameter_ranges_and_speed_data=ParameterRangesAndSpeedData(
                parameter_ranges=ParameterRanges(agent_range=[5, 15, 5],
                                                 size_range=[30, 30, 1],
                                                 in_city_rail_range=[6, 6, 1],
                                                 out_city_rail_range=[2, 2, 1],
                                                 city_range=[20, 20, 1],
                                                 earliest_malfunction=[20, 20, 1],
                                                 malfunction_duration=[20, 20, 1],
                                                 number_of_shortest_paths_per_agent=[10, 10, 1],
                                                 max_window_size_from_earliest=[np.inf, np.inf, 1],
                                                 asp_seed_value=[94, 94, 1],
                                                 weight_route_change=[60, 60, 1],
                                                 weight_lateness_seconds=[1, 1, 1]),
                speed_data={1.: 0.25,  # Fast passenger train
                            1. / 2.: 0.25,  # Fast freight train
                            1. / 3.: 0.25,  # Slow commuter train
                            1. / 4.: 0.25}  # Slow freight train

            ),
            run_analysis=False,
            experiment_name="test_parallel_experiment_execution",
            experiment_base_directory=experiment_folder_name
        )
    finally:
        delete_experiment_folder(experiment_folder_name)
