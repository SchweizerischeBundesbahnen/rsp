import os

from rsp.hypothesis_one_pipeline_all_in_one import list_from_base_directory_and_run_experiment_agenda
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ReScheduleParametersRange


def regen_dummydata(base_directory: str):
    reschedule_parameters_range = ReScheduleParametersRange(
        earliest_malfunction=[1, 1, 1],
        malfunction_duration=[50, 50, 1],
        malfunction_agent_id=[0, 0, 1],

        number_of_shortest_paths_per_agent=[10, 10, 1],

        max_window_size_from_earliest=[60, 60, 1],
        asp_seed_value=[99, 99, 1],

        # route change is penalized the same as 30 seconds delay
        weight_route_change=[30, 30, 1],
        weight_lateness_seconds=[1, 1, 1]
    )

    def filter_experiment_agenda(params: ExperimentParameters):
        return params.infra_parameters.infra_id == 0 and params.schedule_parameters.schedule_id in [0, 1]

    list_from_base_directory_and_run_experiment_agenda(
        experiment_base_directory=base_directory,
        reschedule_parameters_range=reschedule_parameters_range,
        experiment_name=(os.path.basename(base_directory) + "_dummydata"),
        filter_experiment_agenda=filter_experiment_agenda,
        experiment_output_base_directory=os.path.dirname(base_directory)
    )


if __name__ == '__main__':
    regen_dummydata(base_directory="../rsp-data/h1_2020_08_24T21_04_42")
