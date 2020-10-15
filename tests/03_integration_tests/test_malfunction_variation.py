import glob

from rsp.hypothesis_one_malfunction_experiments import malfunction_variation_for_one_schedule
from rsp.utils.experiments import create_experiment_folder_name
from rsp.utils.experiments import delete_experiment_folder
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.global_data_configuration import INFRAS_AND_SCHEDULES_FOLDER


def test_malfunction_variation():
    experiment_output_base_directory = create_experiment_folder_name("test_malfunction_variation")
    # TODO skip this test since src.python.rsp-data not available here - bad design smell
    from sys import platform
    if platform == "linux" or platform == "linux2":
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
        files = glob.glob(f'{output_dir}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}/experiment*.pkl')
        assert len(files) == 1, f"found {len(files)} files"

    finally:
        delete_experiment_folder(experiment_output_base_directory)
