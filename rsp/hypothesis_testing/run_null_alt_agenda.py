import os
from typing import Callable
from typing import List
from typing import Optional

from rsp.hypothesis_one_experiments import hypothesis_one_pipeline_without_setup
from rsp.hypothesis_testing.compare_runtimes import compare_runtimes
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ParameterRangesAndSpeedData
from rsp.utils.experiments import create_experiment_folder_name
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.file_utils import check_create_folder

GetParams = Callable[[], ParameterRangesAndSpeedData]


def compare_agendas(
        experiment_agenda_null: ExperimentAgenda,
        experiment_agenda_alternatives: List[ExperimentAgenda],
        experiment_name: str,
        copy_agenda_from_base_directory: Optional[str] = None
) -> [str, List[str]]:
    """Run and compare two agendas. Scheduling is run only once (non-
    deterministic mode). Re-scheduling on same schedules for null and
    alternative hypotheses.
    Parameters
    ----------
    experiment_agenda_null: ExperimentAgenda
    experiment_agenda_alternatives: List[ExperimentAgenda]
    experiment_name: str
    copy_agenda_from_base_directory: Optional[str] = None
    """

    # do everything in a subfoleder
    base_folder = create_experiment_folder_name(experiment_name=experiment_name)
    check_create_folder(base_folder)
    os.chdir(base_folder)

    print("\n\n\n\n")
    print(f"=========================================================")
    print(f"NULL HYPOTHESIS {experiment_agenda_null}")
    print(f"=========================================================")
    null_hypothesis_base_folder = hypothesis_one_pipeline_without_setup(
        experiment_agenda=experiment_agenda_null,
        experiment_ids=None,
        copy_agenda_from_base_directory=copy_agenda_from_base_directory
    )
    alternative_hypothesis_base_folders = []
    comparison_folders = []
    for i, experiment_agenda_alternative in enumerate(experiment_agenda_alternatives):
        print("\n\n\n\n")
        print(f"=========================================================")
        print(f"ALTERNATIVE HYPOTHESIS {i}:  {experiment_agenda_alternative}")
        print(f"=========================================================")
        alternative_hypothesis_base_folder = hypothesis_one_pipeline_without_setup(
            experiment_agenda=experiment_agenda_alternative,
            experiment_ids=None,
            copy_agenda_from_base_directory=null_hypothesis_base_folder
        )
        alternative_hypothesis_base_folders.append(alternative_hypothesis_base_folder)
        comparison_folder = compare_runtimes(
            data_folder1=os.path.join(null_hypothesis_base_folder, EXPERIMENT_DATA_SUBDIRECTORY_NAME),
            data_folder2=os.path.join(alternative_hypothesis_base_folder, EXPERIMENT_DATA_SUBDIRECTORY_NAME),
            experiment_ids=[]
        )
        comparison_folders.append(comparison_folder)
    return null_hypothesis_base_folder, alternative_hypothesis_base_folders, comparison_folders
