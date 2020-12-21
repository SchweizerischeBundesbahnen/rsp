import glob

import pytest

from rsp.global_data_configuration import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.global_data_configuration import INFRAS_AND_SCHEDULES_FOLDER
from rsp.rsp_malfunction_variation import malfunction_variation_for_one_schedule
from rsp.step_05_experiment_run.experiment_run import create_experiment_folder_name
from rsp.step_05_experiment_run.experiment_run import delete_experiment_folder


# TODO skip this test since rsp-data not available here - bad design smell
@pytest.mark.skip
def test_malfunction_variation():
    experiment_output_base_directory = "target/" + create_experiment_folder_name("test_malfunction_variation")
    # TODO skip this test since rsp-data not available here - bad design smell
    from sys import platform

    if platform == "linux" or platform == "linux2" or platform == "win32":
        return
    try:
        output_dir = malfunction_variation_for_one_schedule(
            infra_id=0,
            schedule_id=0,
            experiments_per_grid_element=1,
            experiment_base_directory=INFRAS_AND_SCHEDULES_FOLDER,
            experiment_output_base_directory=experiment_output_base_directory,
            # run only small fraction resulting in choosing one agent
            fraction_of_malfunction_agents=0.013,
        )
        files = glob.glob(f"{output_dir}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}/experiment*.pkl")
        assert len(files) == 1, f"found {len(files)} files: {files}"

    finally:
        delete_experiment_folder(experiment_output_base_directory)
