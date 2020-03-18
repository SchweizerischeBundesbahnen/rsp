from functools import partial

from rsp.hypothesis_one_experiments import get_first_agenda_pipeline_params
from rsp.hypothesis_testing.run_null_alt_agenda import compare_agendas
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import ParameterRangesAndSpeedData

get_params_null = get_first_agenda_pipeline_params


def get_params_alt(window_size: int) -> ParameterRangesAndSpeedData:
    params = get_params_null()
    parameter_ranges_max_window_size_from_earliest = ParameterRanges(
        **dict(params.parameter_ranges._asdict(), **{'max_window_size_from_earliest': [window_size, window_size, 1]}))
    return ParameterRangesAndSpeedData(
        parameter_ranges=parameter_ranges_max_window_size_from_earliest,
        speed_data=params.speed_data)


def hypothesis_006_window_size_main():
    compare_agendas(
        get_params_null=get_params_null,
        get_params_alternatives=[
            partial(get_params_alt, window_size=30),
            partial(get_params_alt, window_size=60)
        ],
        experiment_name="exp_006_hypothesis_window_size"
    )


if __name__ == '__main__':
    hypothesis_006_window_size_main()
