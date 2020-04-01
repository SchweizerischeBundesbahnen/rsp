import os
from functools import partial

import numpy as np

from rsp.hypothesis_testing.run_null_alt_agenda import compare_agendas
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import ParameterRangesAndSpeedData
from rsp.utils.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import delete_experiment_folder


def get_dummy_params_null() -> ParameterRangesAndSpeedData:
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


def get_params_alt(window_size: int) -> ParameterRangesAndSpeedData:
    """Take params null and change `max_window_size_from_earliest` to that
    given."""
    params = get_dummy_params_null()
    parameter_ranges_max_window_size_from_earliest = ParameterRanges(
        **dict(params.parameter_ranges._asdict(), **{'max_window_size_from_earliest': [window_size, window_size, 1]}))
    return ParameterRangesAndSpeedData(
        parameter_ranges=parameter_ranges_max_window_size_from_earliest,
        speed_data=params.speed_data)


def test_compare_agendas():
    """Run null and alt_0 and alt_1 hypotheses and check that expected files
    are present without inspecting them."""
    null_hypothesis_base_folder, alternative_hypothesis_base_folders, comparison_folders = compare_agendas(
        get_params_null=get_dummy_params_null,
        get_params_alternatives=[
            partial(get_params_alt, window_size=30),
            partial(get_params_alt, window_size=60)
        ],
        experiment_name="test_compare_agendas"
    )
    try:
        assert len(os.listdir(os.path.join(null_hypothesis_base_folder, EXPERIMENT_DATA_SUBDIRECTORY_NAME))) > 0
        assert len(os.listdir(os.path.join(null_hypothesis_base_folder, EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME))) > 0
        assert len(alternative_hypothesis_base_folders) == len(set(alternative_hypothesis_base_folders))
        assert len(alternative_hypothesis_base_folders) == 2
        for alternative_hypothesis_base_folder in alternative_hypothesis_base_folders:
            assert len(
                os.listdir(os.path.join(alternative_hypothesis_base_folder, EXPERIMENT_DATA_SUBDIRECTORY_NAME))) > 0
            assert len(
                os.listdir(os.path.join(alternative_hypothesis_base_folder, EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME))) > 0
        assert len(comparison_folders) == 2
        assert len(set(comparison_folders)) == len(comparison_folders)
        for comparison_folder in comparison_folders:
            print(comparison_folder)
            assert len(os.listdir(comparison_folder)) == 8
    finally:
        for f in [comparison_folder, null_hypothesis_base_folder, alternative_hypothesis_base_folder]:
            delete_experiment_folder(f)