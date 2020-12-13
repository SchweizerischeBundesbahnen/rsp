import glob

from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import InfrastructureParametersRange
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ScheduleParametersRange
from rsp.step_05_experiment_run.experiment_run import create_experiment_folder_name
from rsp.step_05_experiment_run.experiment_run import create_infrastructure_and_schedule_from_ranges
from rsp.step_05_experiment_run.experiment_run import delete_experiment_folder
from rsp.step_05_experiment_run.experiment_run import expand_schedule_parameter_range
from rsp.step_05_experiment_run.experiment_run import load_schedule
from rsp.utils.file_utils import check_create_folder


def test_expand_schedule_parameter_range():
    expanded = expand_schedule_parameter_range(
        schedule_parameter_range=ScheduleParametersRange(asp_seed_value=[33, 38, 5], number_of_shortest_paths_per_agent_schedule=[1, 11, 10]), infra_id=33
    )
    assert len(expanded) == 5 * 10
    assert expanded[0].asp_seed_value == 33
    assert expanded[0].number_of_shortest_paths_per_agent_schedule == 1
    assert expanded[-1].asp_seed_value == 37
    assert expanded[-1].number_of_shortest_paths_per_agent_schedule == 10


def test_expand_schedule_parameter_range_and_save():
    folder_name = "target/" + create_experiment_folder_name("test_expand_schedule_parameter_range_and_save")
    check_create_folder(folder_name)

    try:
        infrastructure_parameters_range = InfrastructureParametersRange(
            width=[30, 30, 1],
            height=[31, 31, 1],
            flatland_seed_value=[3, 3, 1],
            max_num_cities=[4, 4, 1],
            max_rail_between_cities=[2, 2, 1],
            max_rail_in_city=[7, 7, 1],
            number_of_agents=[7, 9, 2],
            number_of_shortest_paths_per_agent=[10, 10, 1],
        )
        schedule_parameters_range = ScheduleParametersRange(asp_seed_value=[33, 37, 3], number_of_shortest_paths_per_agent_schedule=[34, 36, 2])

        list_of_schedule_parameters = create_infrastructure_and_schedule_from_ranges(
            infrastructure_parameters_range=infrastructure_parameters_range,
            schedule_parameters_range=schedule_parameters_range,
            base_directory=folder_name,
            speed_data={1.0: 1.0},
        )

        assert len(list_of_schedule_parameters) == 2 * 3 * 2, f"found {len(list_of_schedule_parameters)}"
        assert len(glob.glob(f"{folder_name}/infra/**/schedule/**/schedule.pkl")) == 2 * 3 * 2
        assert len(glob.glob(f"{folder_name}/infra/**/schedule/**/schedule_parameters.pkl")) == 2 * 3 * 2
        for schedule_parameters in list_of_schedule_parameters:
            _, schedule_parameters_loaded = load_schedule(
                base_directory=folder_name, infra_id=schedule_parameters.infra_id, schedule_id=schedule_parameters.schedule_id
            )
            assert schedule_parameters_loaded == schedule_parameters
    finally:
        delete_experiment_folder(folder_name)
