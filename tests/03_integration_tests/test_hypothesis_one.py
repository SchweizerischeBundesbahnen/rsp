import os

import numpy as np

from rsp.hypothesis_one_pipeline_all_in_one import hypothesis_one_pipeline_all_in_one
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import ParameterRangesAndSpeedData
from rsp.utils.experiments import delete_experiment_folder
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_POTASSCO_SUBDIRECTORY_NAME


def get_dummy_params() -> ParameterRangesAndSpeedData:
    parameter_ranges = ParameterRanges(agent_range=[5, 5, 1],
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
                                       weight_lateness_seconds=[1, 1, 1])
    # Define the desired speed profiles
    speed_data = {1.: 0.25,  # Fast passenger train
                  1. / 2.: 0.25,  # Fast freight train
                  1. / 3.: 0.25,  # Slow commuter train
                  1. / 4.: 0.25}  # Slow freight train
    return ParameterRangesAndSpeedData(parameter_ranges=parameter_ranges, speed_data=speed_data)


def test_hypothesis_one():
    """Run hypothesis one with qualitative analysis and potassco export and
    check that expected files are present wihout inspecting them."""
    # TODO skip run_analysis under Linux because ffmpeg not available in ci -> make it available or add option to skip ffmpeg conversion
    run_analysis = True
    from sys import platform
    if platform == "linux" or platform == "linux2":
        run_analysis = False

    hypothesis_base_folder, _ = hypothesis_one_pipeline_all_in_one(
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
        experiment_ids=[0],
        asp_export_experiment_ids=[0],
        qualitative_analysis_experiment_ids=[0],
        experiment_name="test_hypothesis_one",
        run_analysis=run_analysis
    )
    try:
        data_file_list = os.listdir(os.path.join(hypothesis_base_folder, EXPERIMENT_DATA_SUBDIRECTORY_NAME))
        assert len(data_file_list) > 0
        if run_analysis:
            analysis_file_list = os.listdir(os.path.join(hypothesis_base_folder, EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME))
            assert len(analysis_file_list) > 0
            potassco_file_list = os.listdir(os.path.join(hypothesis_base_folder, EXPERIMENT_POTASSCO_SUBDIRECTORY_NAME))
            assert len(potassco_file_list) > 0
            print(data_file_list)
            print(analysis_file_list)
            print(potassco_file_list)
    finally:
        delete_experiment_folder(hypothesis_base_folder)


def test_parallel_experiment_execution():
    """Run a parallel experiment agenda."""
    hypothesis_base_folder, _ = hypothesis_one_pipeline_all_in_one(
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
    )
    delete_experiment_folder(hypothesis_base_folder)
