import glob

from rsp.hypothesis_one_malfunction_experiments import malfunction_variation_for_one_schedule
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME


# TODO SIM-673 we should take a smaller test case, takes too much time.
def test_malfunction_variation():
    output_dir = malfunction_variation_for_one_schedule(
        infra_id=0,
        schedule_id=0,
        experiments_per_grid_element=1,
        experiment_base_directory="../rsp-data/h1_2020_08_24T21_04_42",
        # run only small fraction
        fraction_of_malfunction_agents=0.1
    )
    files = glob.glob(f'{output_dir}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}/experiment*.pkl')
    assert len(files) == 5
