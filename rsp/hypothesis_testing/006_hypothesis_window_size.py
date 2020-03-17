from functools import partial

import numpy as np

from rsp.hypothesis_testing.run_null_alt_agenda import compare_agendas
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import ParameterRangesAndSpeedData


def get_params_null() -> ParameterRangesAndSpeedData:
    parameter_ranges = ParameterRanges(agent_range=[2, 50, 30],
                                       size_range=[30, 50, 10],
                                       in_city_rail_range=[6, 6, 1],
                                       out_city_rail_range=[2, 2, 1],
                                       city_range=[20, 20, 1],
                                       earliest_malfunction=[20, 20, 1],
                                       malfunction_duration=[20, 20, 1],
                                       number_of_shortest_paths_per_agent=[10, 10, 1],
                                       max_window_size_from_earliest=[np.inf, np.inf, 1])
    # Define the desired speed profiles
    speed_data = {1.: 0.25,  # Fast passenger train
                  1. / 2.: 0.25,  # Fast freight train
                  1. / 3.: 0.25,  # Slow commuter train
                  1. / 4.: 0.25}  # Slow freight train
    return parameter_ranges, speed_data


def get_params_alt(window_size: int) -> ParameterRangesAndSpeedData:
    parameter_ranges, speed_data = get_params_null()
    parameter_ranges_max_window_size_from_earliest = ParameterRanges(
        **dict(parameter_ranges._asdict(), **{'max_window_size_from_earliest': [window_size, window_size, 1]}))
    return parameter_ranges_max_window_size_from_earliest, speed_data


if __name__ == '__main__':
    compare_agendas(
        get_params_null=get_params_null,
        get_params_alternatives=[
            partial(get_params_alt, window_size=30),
            partial(get_params_alt, window_size=60)
        ],
        experiment_name="exp_hypothesis_006"
    )
