from rsp.hypothesis_two_encounter_graph import hypothesis_two_disturbance_propagation_graph
from rsp.utils.experiments import EXPERIMENT_AGENDA_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import load_experiment_result_without_expanding
from rsp.utils.experiments import load_schedule_and_malfunction
from rsp.utils.temporary_dummy_stuff_removal_helpers import remove_dummy_stuff_from_experiment_results_file
from rsp.utils.temporary_dummy_stuff_removal_helpers import remove_dummy_stuff_from_schedule_and_malfunction_pickle


def test_hypothesis_two(remove_dummy: bool = False, re_save: bool = False):
    """Run hypothesis two."""
    experiment_base_directory = './tests/03_integration_tests/mini_toy_example'
    experiment_agenda_directory = f'{experiment_base_directory}/{EXPERIMENT_AGENDA_SUBDIRECTORY_NAME}'
    experiment_data_directory = f'{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}'
    experiment_id = 0

    # used if module path used in pickle has changed
    # use with wrapper file https://stackoverflow.com/questions/13398462/unpickling-python-objects-with-a-changed-module-path
    if re_save:
        load_experiment_result_without_expanding(experiment_data_folder_name=experiment_data_directory, experiment_id=experiment_id, re_save=True)
        load_schedule_and_malfunction(experiment_agenda_directory=experiment_agenda_directory, experiment_id=experiment_id, re_save=True)
    if remove_dummy:
        remove_dummy_stuff_from_schedule_and_malfunction_pickle(experiment_agenda_directory=experiment_agenda_directory, experiment_id=experiment_id)
        remove_dummy_stuff_from_experiment_results_file(experiment_data_folder_name=experiment_data_directory, experiment_id=experiment_id)

    hypothesis_two_disturbance_propagation_graph(
        experiment_base_directory=experiment_base_directory,
        experiment_ids=[experiment_id],
        show=False
    )
