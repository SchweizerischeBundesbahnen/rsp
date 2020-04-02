import os
from typing import Callable
from typing import List
from typing import Optional

from rsp.hypothesis_one_experiments import hypothesis_one_pipeline_without_setup
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ParameterRangesAndSpeedData
from rsp.utils.experiments import create_experiment_folder_name
from rsp.utils.file_utils import check_create_folder

GetParams = Callable[[], ParameterRangesAndSpeedData]


def compare_agendas(
        experiment_agenda: ExperimentAgenda,
        experiment_name: str,
        experiment_ids: Optional[List[int]] = None,
        copy_agenda_from_base_directory: Optional[str] = None,
        run_analysis: bool = True,
        parallel_compute: bool = True
) -> str:
    """Run and compare two agendas. Scheduling is run only once (non-
    deterministic mode). Re-scheduling on same schedules for null and
    alternative hypotheses.
    Parameters
    ----------
    experiment_agenda: ExperimentAgenda
    experiment_name: str
    copy_agenda_from_base_directory: Optional[str] = None,
    experiment_ids: Optional[List[int]] = None
    parallel_compute
    run_analysis
    """

    # do everything in a subfolder
    base_folder = create_experiment_folder_name(experiment_name=experiment_name)
    check_create_folder(base_folder)
    os.chdir(base_folder)

    print("\n\n\n\n")
    print(f"=========================================================")
    print(f"EXPERIMENTS FOR {experiment_name}")
    print(f"=========================================================")
    experiment_base_folder = hypothesis_one_pipeline_without_setup(
        experiment_agenda=experiment_agenda,
        experiment_ids=experiment_ids,
        copy_agenda_from_base_directory=copy_agenda_from_base_directory,
        run_analysis=run_analysis,
        parallel_compute=parallel_compute
    )

    # TODO compare

    return experiment_base_folder
