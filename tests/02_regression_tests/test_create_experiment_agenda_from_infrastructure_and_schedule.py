from rsp.utils.data_types import InfrastructureParametersRange
from rsp.utils.data_types import ReScheduleParametersRange
from rsp.utils.data_types import ScheduleParametersRange
from rsp.utils.experiments import create_experiment_agenda_from_infrastructure_and_schedule_ranges
from rsp.utils.experiments import create_experiment_folder_name
from rsp.utils.experiments import create_infrastructure_and_schedule_from_ranges
from rsp.utils.experiments import list_infrastructure_and_schedule_params_from_base_directory
from rsp.utils.file_utils import check_create_folder


def test_create_experiment_agenda_from_infrastructure_and_schedule():
    experiment_name = "test_create_experiment_agenda_from_infrastructure_and_schedule"
    base_directory = create_experiment_folder_name(experiment_name)
    check_create_folder(base_directory)
    infrastructure_parameter_range = InfrastructureParametersRange(
        width=[30, 30, 1],
        height=[31, 31, 1],
        flatland_seed_value=[3, 3, 1],
        max_num_cities=[4, 4, 1],
        max_rail_between_cities=[2, 2, 1],
        max_rail_in_city=[7, 7, 1],
        number_of_agents=[7, 9, 2],
        number_of_shortest_paths_per_agent=[10, 10, 1],
    )
    schedule_parameters_range = ScheduleParametersRange(
        asp_seed_value=[33, 37, 3],
        number_of_shortest_paths_per_agent_schedule=[34, 36, 2]
    )

    create_infrastructure_and_schedule_from_ranges(
        base_directory=base_directory,
        infrastructure_parameters_range=infrastructure_parameter_range,
        schedule_parameters_range=schedule_parameters_range,
        speed_data={1.0: 1.0}
    )
    infra_parameters_list, infra_schedule_dict = list_infrastructure_and_schedule_params_from_base_directory(
        base_directory=base_directory
    )
    experiment_agenda = create_experiment_agenda_from_infrastructure_and_schedule_ranges(
        experiment_name=experiment_name,
        reschedule_parameters_range=ReScheduleParametersRange(
            earliest_malfunction=[1, 3, 2],
            malfunction_duration=[50, 50, 1],
            malfunction_agent_id=[0, 0, 1],
            number_of_shortest_paths_per_agent=[10, 10, 1],
            max_window_size_from_earliest=[100, 100, 1],
            asp_seed_value=[1, 1, 1],
            weight_route_change=[20, 22, 2],
            weight_lateness_seconds=[1, 1, 1]
        ),
        infra_parameters_list=infra_parameters_list,
        infra_schedule_dict=infra_schedule_dict,
        experiments_per_grid_element=2
    )
    assert (2 * 3 * 2) * (2 * 2 * 2) == len(experiment_agenda.experiments)
