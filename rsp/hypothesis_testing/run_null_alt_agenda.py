import os
from typing import Callable
from typing import Tuple

from rsp.hypothesis_one_experiments import hypothesis_one_pipeline
from rsp.hypothesis_testing.compare_runtimes import compare_runtimes
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import ParameterRangesAndSpeedData
from rsp.utils.data_types import SpeedData
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME

GetParams = Callable[[], ParameterRangesAndSpeedData]


def run_agendas_with_same_rescheduling_problems(
        parameter_ranges_null: ParameterRanges,
        speed_data_null: SpeedData,
        parameter_ranges_alt: ParameterRanges,
        speed_data_alt: SpeedData,

) -> Tuple[str, str]:
    """Run two agendas. The second again takes the schedule from the first as
    input.

    Parameters
    ----------
    parameter_ranges_null
    speed_data_null
    parameter_ranges_alt
    speed_data_alt

    Returns
    -------
    base folders of the two agendas
    """
    print("run null hypothesis")
    null_hypothesis_agenda_folder = hypothesis_one_pipeline(
        parameter_ranges=parameter_ranges_null,
        speed_data=speed_data_null,
        experiment_ids=None,  # no filtering
        copy_agenda_from_base_directory=None  # regenerate schedules
    )
    print("run alt hypothesis")
    alternative_hypothesis_agenda_folder = hypothesis_one_pipeline(
        parameter_ranges=parameter_ranges_alt,
        speed_data=speed_data_alt,
        experiment_ids=None,  # no filtering
        copy_agenda_from_base_directory=null_hypothesis_agenda_folder
    )
    return null_hypothesis_agenda_folder, alternative_hypothesis_agenda_folder


def compare_agendas(get_params_null: GetParams, get_params_alt: GetParams) -> Tuple[str, str, str]:
    """Run and compare two agendas.

    Parameters
    ----------
    get_params_null
    get_params_alt
    """
    parameter_ranges_null, speed_data_null = get_params_null()
    parameter_ranges_alt, speed_data_alt = get_params_alt()
    null_hypothesis_base_folder, alternative_hypothesis_base_folder = run_agendas_with_same_rescheduling_problems(
        parameter_ranges_null, speed_data_null, parameter_ranges_alt, speed_data_alt)
    comparison_folder = compare_runtimes(
        data_folder1=os.path.join(null_hypothesis_base_folder, EXPERIMENT_DATA_SUBDIRECTORY_NAME),
        data_folder2=os.path.join(alternative_hypothesis_base_folder, EXPERIMENT_DATA_SUBDIRECTORY_NAME),
        output_enclosing_folder='.',
        experiment_ids=[]
    )
    return comparison_folder, null_hypothesis_base_folder, alternative_hypothesis_base_folder
