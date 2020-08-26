from typing import Callable

from rsp.hypothesis_one_pipeline_all_in_one import list_from_base_directory_and_run_experiment_agenda
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ReScheduleParametersRange


def get_malfunction_variation_re_schedule_parameters_range() -> ReScheduleParametersRange:
    return ReScheduleParametersRange(
        earliest_malfunction=[1, 300, 50],
        malfunction_duration=[50, 50, 1],
        malfunction_agent_id=[34, 34, 1],
        number_of_shortest_paths_per_agent=[10, 10, 1],
        max_window_size_from_earliest=[100, 100, 1],
        asp_seed_value=[1, 1, 1],
        # route change is penalized the same as 1 second delay
        weight_route_change=[20, 20, 1],
        weight_lateness_seconds=[1, 1, 1]
    )


def get_filter(infra_id: int, schedule_id: int) -> Callable[[ExperimentParameters], bool]:
    def experiment_agenda_filter(experiment_parameters: ExperimentParameters):
        return experiment_parameters.schedule_parameters.infra_id == infra_id and experiment_parameters.schedule_parameters.schedule_id == schedule_id

    return experiment_agenda_filter


if __name__ == '__main__':
    # do not commit your own calls !
    # Define an experiment name, if the experiment already exists we load the schedule from existing experiment
    # Beware of time-stamps when re-runing experiments

    # Generate schedule with n_agents
    infra_id = 0,
    schedule_id = 0
    experiment_name = f"malfunction_variation_{infra_id}_{schedule_id}"
    experiment_base_directory = "../rsp-data/h1_2020_08_24T21_04_42"

    list_from_base_directory_and_run_experiment_agenda(
        experiment_base_directory=experiment_base_directory,
        reschedule_parameters_range=get_malfunction_variation_re_schedule_parameters_range(),
        filter_experiment_agenda=get_filter(infra_id=infra_id, schedule_id=schedule_id),
        experiment_name=experiment_name
    )
