"""Run tests for different experiment methods."""
import glob

import numpy as np

from rsp.global_data_configuration import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.pipeline.rsp_pipeline import generate_infras_and_schedules
from rsp.step_01_agenda_expansion.agenda_expansion import create_experiment_agenda_from_infrastructure_and_schedule_ranges
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ExperimentAgenda
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ExperimentParameters
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import InfrastructureParameters
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import InfrastructureParametersRange
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ReScheduleParameters
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ReScheduleParametersRange
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ScheduleParameters
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ScheduleParametersRange
from rsp.step_01_agenda_expansion.global_constants import get_defaults
from rsp.step_02_infrastructure_generation.infrastructure import create_env_from_experiment_parameters
from rsp.step_05_experiment_run.experiment_results_analysis import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.step_05_experiment_run.experiment_results_analysis_load_and_save import load_and_expand_experiment_results_from_data_folder
from rsp.step_05_experiment_run.experiment_run import create_experiment_folder_name
from rsp.step_05_experiment_run.experiment_run import create_infrastructure_and_schedule_from_ranges
from rsp.step_05_experiment_run.experiment_run import delete_experiment_folder
from rsp.step_05_experiment_run.experiment_run import list_infrastructure_and_schedule_params_from_base_directory
from rsp.step_05_experiment_run.experiment_run import run_experiment_agenda
from rsp.utils.rsp_logger import rsp_logger


def test_created_env_tuple():
    """Test that the tuple of created envs are identical."""
    expected_grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [
            0,
            16386,
            1025,
            5633,
            17411,
            1025,
            1025,
            1025,
            5633,
            17411,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            5633,
            17411,
            1025,
            1025,
            1025,
            5633,
            17411,
            1025,
            4608,
        ],
        [
            0,
            49186,
            1025,
            1097,
            3089,
            5633,
            1025,
            17411,
            1097,
            3089,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1097,
            3089,
            1025,
            1025,
            1025,
            1097,
            3089,
            1025,
            37408,
        ],
        [0, 32800, 0, 0, 0, 72, 1025, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [0, 32872, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386, 34864],
        [16386, 34864, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
        [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
        [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
        [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
        [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
        [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
        [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800, 32800],
        [32800, 49186, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 37408],
        [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
        [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386, 1025, 4608, 0, 0, 0, 32800],
        [
            72,
            33897,
            1025,
            5633,
            17411,
            1025,
            1025,
            1025,
            5633,
            17411,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            17411,
            1025,
            5633,
            17411,
            3089,
            1025,
            1097,
            5633,
            17411,
            1025,
            2064,
        ],
        [
            0,
            72,
            1025,
            1097,
            3089,
            5633,
            1025,
            17411,
            1097,
            3089,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            1025,
            2064,
            0,
            72,
            3089,
            5633,
            1025,
            17411,
            1097,
            2064,
            0,
            0,
        ],
        [0, 0, 0, 0, 0, 72, 1025, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 1025, 2064, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    test_parameters = ExperimentParameters(
        experiment_id=0,
        grid_id=0,
        infra_id_schedule_id=0,
        infra_parameters=InfrastructureParameters(
            infra_id=0,
            width=30,
            height=30,
            number_of_agents=7,
            flatland_seed_value=12,
            max_num_cities=5,
            grid_mode=True,
            max_rail_between_cities=2,
            max_rail_in_city=4,
            speed_data={1: 1.0},
            number_of_shortest_paths_per_agent=10,
        ),
        schedule_parameters=ScheduleParameters(infra_id=0, schedule_id=0, asp_seed_value=94, number_of_shortest_paths_per_agent_schedule=1),
        re_schedule_parameters=ReScheduleParameters(
            earliest_malfunction=10,
            malfunction_duration=20,
            malfunction_agent_id=0,
            weight_route_change=1,
            weight_lateness_seconds=1,
            max_window_size_from_earliest=np.inf,
            number_of_shortest_paths_per_agent=10,
            asp_seed_value=94,
        ),
    )

    # Generate the tuple of environments
    static_env = create_env_from_experiment_parameters(params=test_parameters.infra_parameters)
    print(static_env.rail.grid.tolist())

    # Check that the same grid was created
    assert static_env.rail.grid.tolist() == expected_grid


def test_run_experiment_agenda():
    """Run a simple agenda as regression test.

    It verifies that we can start from a set of schedules and
    deterministically and produces an equivalent results with the same
    costs. Results may differ on different platforms event with the same
    seed because we use 2 threads.
    """
    agenda = ExperimentAgenda(
        experiment_name="test_regression_experiment_agenda",
        global_constants=get_defaults(),
        experiments=[
            ExperimentParameters(
                experiment_id=0,
                grid_id=0,
                infra_id_schedule_id=0,
                infra_parameters=InfrastructureParameters(
                    infra_id=0,
                    width=30,
                    height=30,
                    number_of_agents=2,
                    flatland_seed_value=12,
                    max_num_cities=20,
                    grid_mode=True,
                    max_rail_between_cities=2,
                    max_rail_in_city=6,
                    speed_data={1: 1.0},
                    number_of_shortest_paths_per_agent=10,
                ),
                schedule_parameters=ScheduleParameters(infra_id=0, schedule_id=0, asp_seed_value=94, number_of_shortest_paths_per_agent_schedule=1),
                re_schedule_parameters=ReScheduleParameters(
                    earliest_malfunction=20,
                    malfunction_duration=20,
                    malfunction_agent_id=0,
                    weight_route_change=1,
                    weight_lateness_seconds=1,
                    max_window_size_from_earliest=np.inf,
                    number_of_shortest_paths_per_agent=10,
                    asp_seed_value=94,
                ),
            )
        ],
    )

    # Import the solver for the experiments
    experiment_output_directory = "target/" + create_experiment_folder_name(agenda.experiment_name)
    try:
        experiment_folder_name = run_experiment_agenda(
            experiment_agenda=agenda,
            # do not clutter folder
            experiment_output_directory=experiment_output_directory,
            run_experiments_parallel=1,
            experiment_base_directory="tests/02_regression_tests/data/regression_experiment_agenda",
        )

        # load results
        _, experiment_results_for_analysis = load_and_expand_experiment_results_from_data_folder(
            f"{experiment_folder_name}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}"
        )
        result_dict = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_for_analysis).to_dict()

        expected_result_dict = {
            "solver_statistics_costs_offline_delta": {0: 20.0},
            "solver_statistics_costs_schedule": {0: 0.0},
            "solver_statistics_costs_online_unrestricted": {0: 20.0},
            "experiment_id": {0: 0},
            "max_num_cities": {0: 20},
            "max_rail_between_cities": {0: 2},
            "max_rail_in_city": {0: 6},
            "n_agents": {0: 2},
            "size": {0: 30},
        }
        print("solution_online_unrestricted")
        print(experiment_results_for_analysis[0].solution_online_unrestricted)
        print("solution_offline_delta")
        print(experiment_results_for_analysis[0].solution_offline_delta)

        for key in expected_result_dict:
            if expected_result_dict[key] != result_dict[key]:
                rsp_logger.warn(f"{key} should be equal; expected{expected_result_dict[key]}, but got {result_dict[key]}")
            assert expected_result_dict[key] == result_dict[key], f"{key} should be equal; expected{expected_result_dict[key]}, but got {result_dict[key]}"
    finally:
        delete_experiment_folder(experiment_output_directory)


def assert_expected_experiment_pkl_and_csv(experiment_output_directory, nb_csvs, nb_pkls):
    file_names = glob.glob(f"{experiment_output_directory}/data/experiment_*.csv")
    assert len(file_names) == nb_csvs, f"found {file_names} in {experiment_output_directory}, expected {nb_csvs}"
    file_names = glob.glob(f"{experiment_output_directory}/data/experiment_*.pkl")
    assert len(file_names) == nb_pkls, f"found {file_names} in {experiment_output_directory}, expected {nb_pkls}"


def test_rerun_single_experiment_after_csv_only():
    """Run a simple agenda as regression test.

    It verifies that we can start from a set of schedules and
    deterministically and produces an equivalent results with the same
    costs. Results may differ on different platforms event with the same
    seed because we use 2 threads.
    """
    agenda = ExperimentAgenda(
        experiment_name="test_rerun_single_experiment_after_csv_only",
        global_constants=get_defaults(),
        experiments=[
            ExperimentParameters(
                experiment_id=0,
                grid_id=0,
                infra_id_schedule_id=0,
                infra_parameters=InfrastructureParameters(
                    infra_id=0,
                    width=30,
                    height=30,
                    number_of_agents=2,
                    flatland_seed_value=12,
                    max_num_cities=20,
                    grid_mode=True,
                    max_rail_between_cities=2,
                    max_rail_in_city=6,
                    speed_data={1: 1.0},
                    number_of_shortest_paths_per_agent=10,
                ),
                schedule_parameters=ScheduleParameters(infra_id=0, schedule_id=0, asp_seed_value=94, number_of_shortest_paths_per_agent_schedule=1),
                re_schedule_parameters=ReScheduleParameters(
                    earliest_malfunction=20,
                    malfunction_duration=20,
                    malfunction_agent_id=0,
                    weight_route_change=1,
                    weight_lateness_seconds=1,
                    max_window_size_from_earliest=np.inf,
                    number_of_shortest_paths_per_agent=10,
                    asp_seed_value=94,
                ),
            )
        ],
    )

    # Import the solver for the experiments
    experiment_output_directory = "target/" + create_experiment_folder_name(agenda.experiment_name)
    try:
        run_experiment_agenda(
            experiment_agenda=agenda,
            # do not clutter folder
            experiment_output_directory=experiment_output_directory,
            run_experiments_parallel=1,
            experiment_base_directory="tests/02_regression_tests/data/regression_experiment_agenda",
            csv_only=True,
        )

        assert_expected_experiment_pkl_and_csv(experiment_output_directory, nb_csvs=1, nb_pkls=0)

        def filter_experiment_agenda(params: ExperimentParameters):
            return params.experiment_id == 0

        run_experiment_agenda(
            experiment_output_directory=experiment_output_directory,
            experiment_base_directory="tests/02_regression_tests/data/regression_experiment_agenda",
            csv_only=False,
            filter_experiment_agenda=filter_experiment_agenda,
        )
        # the first csv is not removed, but the pkl is generated alongside
        assert_expected_experiment_pkl_and_csv(experiment_output_directory=experiment_output_directory, nb_csvs=2, nb_pkls=1)
        run_experiment_agenda(
            experiment_output_directory=experiment_output_directory,
            experiment_base_directory="tests/02_regression_tests/data/regression_experiment_agenda",
            csv_only=False,
            filter_experiment_agenda=filter_experiment_agenda,
        )
        # since there is a pkl, the experiment is not re-run
        assert_expected_experiment_pkl_and_csv(experiment_output_directory=experiment_output_directory, nb_csvs=2, nb_pkls=1)

    finally:
        delete_experiment_folder(experiment_output_directory)


def assert_expected_infrastructures_and_schedules(experiment_output_directory, nb_infras, nb_schedules):
    file_names = glob.glob(f"{experiment_output_directory}/**/infrastructure.pkl", recursive=True)
    assert len(file_names) == nb_infras, f"found {file_names} in {experiment_output_directory}, expected {nb_infras}"
    file_names = glob.glob(f"{experiment_output_directory}/**/schedule.pkl", recursive=True)
    assert len(file_names) == nb_schedules, f"found {file_names} in {experiment_output_directory}, expected {nb_schedules}"


def test_incremental_generate_infras_and_schedules():
    experiment_output_directory = "target/" + create_experiment_folder_name("test_incremental_generate_infras_and_schedules")
    try:
        infrastructure_parameter_range = InfrastructureParametersRange(
            width=[30, 30, 1],
            height=[31, 31, 1],
            flatland_seed_value=[3, 3, 1],
            max_num_cities=[4, 4, 1],
            max_rail_between_cities=[2, 2, 1],
            max_rail_in_city=[7, 7, 1],
            number_of_agents=[8, 8, 1],
            number_of_shortest_paths_per_agent=[10, 10, 1],
        )
        schedule_parameters_range = ScheduleParametersRange(asp_seed_value=[814, 814, 1], number_of_shortest_paths_per_agent_schedule=[1, 1, 1],)

        assert_expected_infrastructures_and_schedules(experiment_output_directory=experiment_output_directory, nb_infras=0, nb_schedules=0)

        generate_infras_and_schedules(
            base_directory=experiment_output_directory,
            infra_parameters_range=infrastructure_parameter_range,
            schedule_parameters_range=schedule_parameters_range,
        )
        assert_expected_infrastructures_and_schedules(experiment_output_directory=experiment_output_directory, nb_infras=1, nb_schedules=1)

        generate_infras_and_schedules(
            base_directory=experiment_output_directory,
            infra_parameters_range=InfrastructureParametersRange(**dict(infrastructure_parameter_range._asdict(), flatland_seed_value=[3, 4, 2])),
            schedule_parameters_range=ScheduleParametersRange(**dict(schedule_parameters_range._asdict(), asp_seed_value=[814, 815, 2])),
        )
        assert_expected_infrastructures_and_schedules(experiment_output_directory=experiment_output_directory, nb_infras=2, nb_schedules=4)
    finally:
        delete_experiment_folder(experiment_output_directory)


def test_run_agenda_incremental():
    experiment_name = "test_rsp_pipeline_incremental"
    experiment_base_directory = "target/" + create_experiment_folder_name(experiment_name)
    try:
        create_infrastructure_and_schedule_from_ranges(
            base_directory=experiment_base_directory,
            infrastructure_parameters_range=InfrastructureParametersRange(
                number_of_agents=[2, 2, 1],
                width=[30, 30, 1],
                height=[30, 30, 1],
                flatland_seed_value=[190, 195, 4],
                max_rail_in_city=[6, 6, 1],
                max_rail_between_cities=[2, 2, 1],
                max_num_cities=[20, 20, 1],
                number_of_shortest_paths_per_agent=[10, 10, 1],
            ),
            schedule_parameters_range=ScheduleParametersRange(asp_seed_value=[814, 814, 1], number_of_shortest_paths_per_agent_schedule=[1, 1, 1],),
            speed_data={1.0: 1.0},
        )
        infra_parameters_list, infra_schedule_dict = list_infrastructure_and_schedule_params_from_base_directory(base_directory=experiment_base_directory)
        experiment_agenda = create_experiment_agenda_from_infrastructure_and_schedule_ranges(
            experiment_name=experiment_name,
            reschedule_parameters_range=ReScheduleParametersRange(
                earliest_malfunction=[20, 20, 1],
                malfunction_duration=[20, 20, 1],
                malfunction_agent_id=[0, 2, 2],
                number_of_shortest_paths_per_agent=[10, 10, 1],
                max_window_size_from_earliest=[60, 60, 1],
                asp_seed_value=[99, 101, 1],
                # route change is penalized the same as 30 seconds delay
                weight_route_change=[60, 60, 1],
                weight_lateness_seconds=[1, 1, 1],
                # Define the desired speed profiles
            ),
            infra_parameters_list=infra_parameters_list,
            infra_schedule_dict=infra_schedule_dict,
            experiments_per_grid_element=1,
        )
        # run experiment 2
        experiment_output_directory = run_experiment_agenda(
            experiment_base_directory=experiment_base_directory,
            experiment_agenda=experiment_agenda,
            filter_experiment_agenda=lambda experiment_parameters: experiment_parameters.experiment_id == 2,
        )
        assert_expected_experiment_pkl_and_csv(experiment_output_directory=experiment_output_directory, nb_csvs=1, nb_pkls=1)

        # run experiments 0...3 (2 is not re-run)
        experiment_output_directory = run_experiment_agenda(
            experiment_base_directory=experiment_base_directory,
            experiment_output_directory=experiment_output_directory,
            filter_experiment_agenda=lambda experiment_parameters: experiment_parameters.experiment_id < 4,
        )
        assert_expected_experiment_pkl_and_csv(experiment_output_directory=experiment_output_directory, nb_csvs=4, nb_pkls=4)

    finally:
        delete_experiment_folder(experiment_base_directory)
