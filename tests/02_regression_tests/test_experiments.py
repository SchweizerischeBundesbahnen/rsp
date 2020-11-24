"""Run tests for different experiment methods."""
import numpy as np
from rsp.step_01_planning.experiment_parameters_and_ranges import ExperimentAgenda
from rsp.step_01_planning.experiment_parameters_and_ranges import ExperimentParameters
from rsp.step_01_planning.experiment_parameters_and_ranges import InfrastructureParameters
from rsp.step_01_planning.experiment_parameters_and_ranges import ReScheduleParameters
from rsp.step_01_planning.experiment_parameters_and_ranges import ScheduleParameters
from rsp.step_03_run.experiment_results_analysis import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.step_03_run.experiments import create_env_from_experiment_parameters
from rsp.step_03_run.experiments import create_experiment_folder_name
from rsp.step_03_run.experiments import delete_experiment_folder
from rsp.step_03_run.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.step_03_run.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.step_03_run.experiments import run_experiment_agenda
from rsp.step_04_analysis.data_analysis_all_in_one import hypothesis_one_data_analysis
from rsp.utils.global_constants import get_defaults
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

        hypothesis_one_data_analysis(
            experiment_output_directory=experiment_folder_name, analysis_2d=True, qualitative_analysis_experiment_ids=[0],
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
