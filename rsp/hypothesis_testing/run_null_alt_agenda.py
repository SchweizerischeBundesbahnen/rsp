import os
from typing import Callable
from typing import List

from rsp.hypothesis_one_experiments import hypothesis_one_pipeline
from rsp.hypothesis_testing.compare_runtimes import compare_runtimes
from rsp.utils.data_types import ParameterRangesAndSpeedData
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import create_experiment_folder_name
from rsp.utils.file_utils import check_create_folder

GetParams = Callable[[], ParameterRangesAndSpeedData]


def compare_agendas(
        get_params_null: GetParams,
        get_params_alternatives: List[GetParams],
        experiment_name: str
) -> [str, List[str]]:
    """Run and compare two agendas. Scheduling is run only once (non-
    deterministic mode). Re-scheduling on same schedules for null and
    alternative hypotheses.
    Parameters
    ----------
    get_params_null: GetParams
    get_params_alt: List[GetParams]
    """

    # do everything in a subfoleder
    base_folder = create_experiment_folder_name(experiment_name=experiment_name)
    check_create_folder(base_folder)
    os.chdir(base_folder)

    parameter_ranges_and_speed_data = get_params_null()
    print("\n\n\n\n")
    print(f"=========================================================")
    print(f"NULL HYPOTHESIS {parameter_ranges_and_speed_data}")
    print(f"=========================================================")
    null_hypothesis_base_folder = hypothesis_one_pipeline(
        parameter_ranges_and_speed_data=parameter_ranges_and_speed_data,
        experiment_ids=None,
        copy_agenda_from_base_directory=None,  # generate schedules
        experiment_name=experiment_name + "_null"
    )
    alternative_hypothesis_base_folders = []
    comparison_folders = []
    for i, get_params_alt in enumerate(get_params_alternatives):
        parameter_ranges_and_speed_data = get_params_alt()
        print("\n\n\n\n")
        print(f"=========================================================")
        print(f"ALTERNATIVE HYPOTHESIS {i}:  {parameter_ranges_and_speed_data}")
        print(f"=========================================================")
        alternative_hypothesis_base_folder = hypothesis_one_pipeline(
            parameter_ranges_and_speed_data=parameter_ranges_and_speed_data,
            experiment_ids=None,
            copy_agenda_from_base_directory=null_hypothesis_base_folder,
            experiment_name=experiment_name + f"_alt{i:03d}"
        )
        alternative_hypothesis_base_folders.append(alternative_hypothesis_base_folder)
        comparison_folder = compare_runtimes(
            data_folder1=os.path.join(null_hypothesis_base_folder, EXPERIMENT_DATA_SUBDIRECTORY_NAME),
            data_folder2=os.path.join(alternative_hypothesis_base_folder, EXPERIMENT_DATA_SUBDIRECTORY_NAME),
            experiment_ids=[]
        )
        comparison_folders.append(comparison_folder)
    return null_hypothesis_base_folder, alternative_hypothesis_base_folders, comparison_folders