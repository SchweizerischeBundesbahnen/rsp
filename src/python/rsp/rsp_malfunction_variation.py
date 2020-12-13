from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from rsp.global_data_configuration import INFRAS_AND_SCHEDULES_FOLDER
from rsp.scheduling.schedule import Schedule
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ExperimentAgenda
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ExperimentParameters
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import InfrastructureParameters
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ReScheduleParameters
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ScheduleParameters
from rsp.step_01_agenda_expansion.global_constants import get_defaults
from rsp.step_05_experiment_run.experiment_run import list_infrastructure_and_schedule_params_from_base_directory
from rsp.step_05_experiment_run.experiment_run import run_experiment_agenda


def create_malfunction_agenda_from_infrastructure_and_schedule_ranges(
    experiment_name: str,
    infra_parameters_list: List[InfrastructureParameters],
    infra_schedule_dict: Dict[InfrastructureParameters, List[Tuple[ScheduleParameters, Schedule]]],
    latest_malfunction_as_fraction_of_max_episode_steps: float,
    malfunction_interval_as_fraction_of_max_episode_steps: float,
    fraction_of_malfunction_agents: float,
    malfunction_duration: int,
    weight_route_change: int,
    weight_lateness_seconds: int,
    max_window_size_from_earliest: int,
    experiments_per_grid_element: int = 1,
):
    """Create malfunction variation for these schedules over a range of agents
    and a range of malfunction time steps.

    Parameters
    ----------
    experiment_name
        the name for the run folder
    infra_parameters_list
        as from `list_infrastructure_and_schedule_params_from_base_directory`
    infra_schedule_dict
        as from `list_infrastructure_and_schedule_params_from_base_directory`
    malfunction_interval_as_fraction_of_max_episode_steps
        malfunctions will be at `0, int(malfunction_interval_as_fraction_of_max_episode_steps*max_episode_steps), ...`
    latest_malfunction_as_fraction_of_max_episode_steps
        latest malfunction will be at `int(latest_malfunction_as_fraction_of_max_episode_steps * max_episode_steps)`
    fraction_of_malfunction_agents
        agents `0,...,int(fraction_of_malfunction_agents*number_of_agents)` will have a malfunction
    malfunction_duration
        same for all malfunctions
    weight_route_change
        same for all
    weight_lateness_seconds
        same for all
    max_window_size_from_earliest
        same for all
    experiments_per_grid_element
        repeat the same malfunction multiple times?

    Returns
    -------
    """
    infra_parameters_dict = {infra_parameters.infra_id: infra_parameters for infra_parameters in infra_parameters_list}
    experiments = []

    # we want to be able to have different number of schedules for two infrastructures, therefore, we increment counters
    experiment_id = 0
    grid_id = 0
    infra_id_schedule_id = 0
    for infra_id, list_of_schedule_parameters in infra_schedule_dict.items():
        infra_parameters: InfrastructureParameters = infra_parameters_dict[infra_id]
        for schedule_parameters, schedule in list_of_schedule_parameters:
            max_episode_steps = schedule.schedule_problem_description.max_episode_steps
            malfunction_interval_absolute = int(malfunction_interval_as_fraction_of_max_episode_steps * max_episode_steps)
            for malfunction_agent_id in range(int(fraction_of_malfunction_agents * infra_parameters.number_of_agents)):
                train_run_start = schedule.schedule_experiment_result.trainruns_dict[malfunction_agent_id][0].scheduled_at
                train_run_end = schedule.schedule_experiment_result.trainruns_dict[malfunction_agent_id][-1].scheduled_at
                for i in range(int(latest_malfunction_as_fraction_of_max_episode_steps / malfunction_interval_as_fraction_of_max_episode_steps)):
                    earliest_malfunction = i * malfunction_interval_absolute
                    if earliest_malfunction >= train_run_end - train_run_start:
                        continue
                    for _ in range(experiments_per_grid_element):
                        experiments.append(
                            ExperimentParameters(
                                experiment_id=experiment_id,
                                schedule_parameters=schedule_parameters,
                                infra_parameters=infra_parameters,
                                grid_id=grid_id,
                                infra_id_schedule_id=infra_id_schedule_id,
                                re_schedule_parameters=ReScheduleParameters(
                                    earliest_malfunction=earliest_malfunction,
                                    malfunction_duration=malfunction_duration,
                                    malfunction_agent_id=malfunction_agent_id,
                                    weight_route_change=weight_route_change,
                                    weight_lateness_seconds=weight_lateness_seconds,
                                    max_window_size_from_earliest=max_window_size_from_earliest,
                                ),
                            )
                        )
                        experiment_id += 1
                    grid_id += 1
            infra_id_schedule_id += 1
    return ExperimentAgenda(experiment_name=experiment_name, experiments=experiments, global_constants=get_defaults(),)


def get_filter(infra_id: int, schedule_id: int) -> Callable[[int, int], bool]:
    def experiment_agenda_filter(infra_id_arg: int, schedule_id_arg: int):
        return infra_id_arg == infra_id and schedule_id_arg == schedule_id

    return experiment_agenda_filter


def malfunction_variation_for_one_schedule(
    infra_id: int,
    schedule_id: int,
    experiments_per_grid_element: int,
    experiment_base_directory: str,
    experiment_output_base_directory: Optional[str] = None,
    latest_malfunction_as_fraction_of_max_episode_steps: float = 0.5,
    malfunction_interval_as_fraction_of_max_episode_steps: float = 0.1,
    fraction_of_malfunction_agents: float = 1.0,
    malfunction_duration: int = 50,
    weight_route_change: int = 30,
    weight_lateness_seconds: int = 1,
    max_window_size_from_earliest: int = 60,
):
    experiment_name = f"malfunction_variation_{infra_id}_{schedule_id}"

    infra_parameters_list, infra_schedule_dict = list_infrastructure_and_schedule_params_from_base_directory(
        base_directory=experiment_base_directory, filter_experiment_agenda=get_filter(infra_id=infra_id, schedule_id=schedule_id)
    )

    experiment_agenda = create_malfunction_agenda_from_infrastructure_and_schedule_ranges(
        experiment_name=experiment_name,
        infra_parameters_list=infra_parameters_list,
        infra_schedule_dict=infra_schedule_dict,
        experiments_per_grid_element=experiments_per_grid_element,
        latest_malfunction_as_fraction_of_max_episode_steps=latest_malfunction_as_fraction_of_max_episode_steps,
        malfunction_interval_as_fraction_of_max_episode_steps=malfunction_interval_as_fraction_of_max_episode_steps,
        fraction_of_malfunction_agents=fraction_of_malfunction_agents,
        malfunction_duration=malfunction_duration,
        weight_route_change=weight_route_change,
        weight_lateness_seconds=weight_lateness_seconds,
        max_window_size_from_earliest=max_window_size_from_earliest,
    )

    experiment_output_directory = run_experiment_agenda(
        experiment_agenda=experiment_agenda, experiment_base_directory=experiment_base_directory, experiment_output_directory=experiment_output_base_directory,
    )

    return experiment_output_directory


if __name__ == "__main__":
    # sample call
    malfunction_variation_for_one_schedule(infra_id=0, schedule_id=0, experiments_per_grid_element=1, experiment_base_directory=INFRAS_AND_SCHEDULES_FOLDER)
