from functools import partial

from rsp.hypothesis_one_experiments import get_first_agenda_pipeline_params
from rsp.hypothesis_testing.run_null_alt_agenda import compare_agendas
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import ParameterRangesAndSpeedData


def get_params_alt(seed: int) -> ParameterRangesAndSpeedData:
    params = get_params_null()
    parameter_ranges_max_window_size_from_earliest = ParameterRanges(
        **dict(params.parameter_ranges._asdict(), **{'asp_seed_value': [seed, seed, 1]}))
    return ParameterRangesAndSpeedData(
        parameter_ranges=parameter_ranges_max_window_size_from_earliest,
        speed_data=params.speed_data)


get_params_null = get_first_agenda_pipeline_params


def hypothesis_001_solver_seeding_irrelevant_main():
    compare_agendas(
        get_params_null=get_params_null,
        get_params_alternatives=[partial(get_params_alt, seed=(94 + inc)) for inc in range(5)],
        experiment_name="exp_001_hypothesis_solver_seeding_irrelevant"
    )


if __name__ == '__main__':
    hypothesis_001_solver_seeding_irrelevant_main()
