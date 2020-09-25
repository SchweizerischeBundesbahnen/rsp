import glob

from rsp.hypothesis_one_malfunction_experiments import malfunction_variation_for_one_schedule
from rsp.utils.experiments import create_experiment_folder_name
from rsp.utils.experiments import delete_experiment_folder
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.global_data_configuration import INFRAS_AND_SCHEDULES_FOLDER


def test_malfunction_variation():
    experiment_output_base_directory = create_experiment_folder_name("test_malfunction_variation")

    try:
        output_dir = malfunction_variation_for_one_schedule(
            infra_id=0,
            schedule_id=0,
            experiments_per_grid_element=1,
            experiment_base_directory=INFRAS_AND_SCHEDULES_FOLDER,
            experiment_output_base_directory=experiment_output_base_directory,
            # run only small fraction
            fraction_of_malfunction_agents=0.1,
        )
        files = glob.glob(f'{output_dir}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}/experiment*.pkl')
        assert len(files) == 5
    finally:
        delete_experiment_folder(experiment_output_base_directory)
