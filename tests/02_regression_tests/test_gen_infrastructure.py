import glob

from rsp.step_01_planning.experiment_parameters_and_ranges import InfrastructureParametersRange
from rsp.step_03_run.experiments import create_experiment_folder_name
from rsp.step_03_run.experiments import delete_experiment_folder
from rsp.step_03_run.experiments import expand_infrastructure_parameter_range
from rsp.step_03_run.experiments import expand_infrastructure_parameter_range_and_generate_infrastructure
from rsp.step_03_run.experiments import load_infrastructure
from rsp.utils.file_utils import check_create_folder

infrastructure_parameter_range = InfrastructureParametersRange(
    width=[30, 30, 1],
    height=[31, 31, 1],
    flatland_seed_value=[3, 13, 10],
    max_num_cities=[4, 4, 1],
    max_rail_between_cities=[2, 2, 1],
    max_rail_in_city=[7, 7, 1],
    number_of_agents=[8, 8, 1],
    number_of_shortest_paths_per_agent=[10, 10, 1],
)


def test_expand_infrastructure_ranges():
    infrastructure_parameters = expand_infrastructure_parameter_range(
        infrastructure_parameter_range=infrastructure_parameter_range, grid_mode=False, speed_data={1.0: 0.25}
    )
    assert len(infrastructure_parameters) == 10
    for i, first_infrastructure_parameters in enumerate(infrastructure_parameters):
        assert first_infrastructure_parameters.width == 30
        assert first_infrastructure_parameters.height == 31
        assert first_infrastructure_parameters.flatland_seed_value == 3 + i
        assert first_infrastructure_parameters.max_num_cities == 4
        assert first_infrastructure_parameters.grid_mode is False
        assert first_infrastructure_parameters.max_rail_between_cities == 2
        assert first_infrastructure_parameters.max_rail_in_city == 7
        assert first_infrastructure_parameters.number_of_agents == 8
        assert first_infrastructure_parameters.speed_data == {1.0: 0.25}
        assert first_infrastructure_parameters.number_of_shortest_paths_per_agent == 10


def test_expand_infrastructure_parameter_range_and_save():
    folder_name = "target/" + create_experiment_folder_name("test_expand_infrastructure_parameter_range_and_save")
    check_create_folder(folder_name)

    try:
        list_of_infra_parameters = expand_infrastructure_parameter_range_and_generate_infrastructure(
            infrastructure_parameter_range=infrastructure_parameter_range, base_directory=folder_name, speed_data={1.0: 1.0}
        )
        assert len(list_of_infra_parameters) == 10

        assert len(glob.glob(f"{folder_name}/infra/[0-9]*/")) == 10
        assert len(glob.glob(f"{folder_name}/infra/**/infrastructure.pkl")) == 10
        assert len(glob.glob(f"{folder_name}/infra/**/infrastructure_parameters.pkl")) == 10

        for i, infra_parameters in enumerate(list_of_infra_parameters):
            assert i == infra_parameters.infra_id
            _, infra_parameters_loaded = load_infrastructure(base_directory=folder_name, infra_id=infra_parameters.infra_id)
            assert infra_parameters_loaded == infra_parameters
    finally:
        delete_experiment_folder(folder_name)
