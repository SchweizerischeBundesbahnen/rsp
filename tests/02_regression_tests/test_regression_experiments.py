"""Run tests for different experiment methods."""
import os
from typing import List

import numpy as np

from rsp.experiment_solvers.data_types import schedule_experiment_results_equals_modulo_solve_time
from rsp.hypothesis_one_data_analysis import hypothesis_one_data_analysis
from rsp.hypothesis_one_pipeline_all_in_one import hypothesis_one_pipeline_all_in_one
from rsp.logger import rsp_logger
from rsp.schedule_problem_description.data_types_and_utils import schedule_problem_description_equals
from rsp.utils.data_types import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.utils.data_types import convert_list_of_experiment_results_to_data_frame
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ExperimentResults
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import ParameterRangesAndSpeedData
from rsp.utils.experiments import create_env_from_experiment_parameters
from rsp.utils.experiments import create_experiment_folder_name
from rsp.utils.experiments import delete_experiment_folder
from rsp.utils.experiments import EXPERIMENT_AGENDA_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import gen_malfunction
from rsp.utils.experiments import gen_schedule
from rsp.utils.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.utils.experiments import load_malfunction
from rsp.utils.experiments import load_schedule
from rsp.utils.experiments import run_experiment_agenda
from rsp.utils.experiments import run_experiment_from_schedule_and_malfunction
from rsp.utils.experiments import save_experiment_agenda_and_hash_to_file
from rsp.utils.experiments import save_malfunction
from rsp.utils.experiments import save_schedule


def test_created_env_tuple():
    """Test that the tuple of created envs are identical."""
    expected_grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 16386, 1025, 5633, 17411, 1025, 1025, 1025, 5633, 17411, 1025, 1025, 1025, 1025, 1025, 1025,
                      1025, 1025, 1025, 1025, 1025, 5633, 17411, 1025, 1025, 1025, 5633, 17411, 1025, 4608],
                     [0, 49186, 1025, 1097, 3089, 5633, 1025, 17411, 1097, 3089, 1025, 1025, 1025, 1025, 1025, 1025,
                      1025, 1025, 1025, 1025, 1025, 1097, 3089, 1025, 1025, 1025, 1097, 3089, 1025, 37408],
                     [0, 32800, 0, 0, 0, 72, 1025, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
                     [0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32800],
                     [0, 32872, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386,
                      34864],
                     [16386, 34864, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800, 32800],
                     [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800, 32800],
                     [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800, 32800],
                     [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800, 32800],
                     [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800, 32800],
                     [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800, 32800],
                     [32800, 32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800, 32800],
                     [32800, 49186, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72,
                      37408],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386, 1025, 4608, 0,
                      0, 0, 32800],
                     [72, 33897, 1025, 5633, 17411, 1025, 1025, 1025, 5633, 17411, 1025, 1025, 1025, 1025, 1025, 1025,
                      1025, 1025, 1025, 17411, 1025, 5633, 17411, 3089, 1025, 1097, 5633, 17411, 1025, 2064],
                     [0, 72, 1025, 1097, 3089, 5633, 1025, 17411, 1097, 3089, 1025, 1025, 1025, 1025, 1025, 1025, 1025,
                      1025, 1025, 2064, 0, 72, 3089, 5633, 1025, 17411, 1097, 2064, 0, 0],
                     [0, 0, 0, 0, 0, 72, 1025, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 1025, 2064, 0, 0,
                      0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    test_parameters = ExperimentParameters(experiment_id=0,
                                           grid_id=0,
                                           number_of_agents=7,
                                           width=30,
                                           height=30,
                                           flatland_seed_value=12,
                                           asp_seed_value=94,
                                           max_num_cities=5,
                                           grid_mode=True,
                                           max_rail_between_cities=2,
                                           max_rail_in_city=4,
                                           earliest_malfunction=10,
                                           malfunction_duration=20,
                                           speed_data={1: 1.0},
                                           number_of_shortest_paths_per_agent=10,
                                           weight_route_change=1,
                                           weight_lateness_seconds=1,
                                           max_window_size_from_earliest=np.inf)

    # Generate the tuple of environments
    static_env = create_env_from_experiment_parameters(params=test_parameters)
    print(static_env.rail.grid.tolist())

    # Check that the same grid was created
    assert static_env.rail.grid.tolist() == expected_grid


def test_regression_experiment_agenda(regen: bool = False, re_save: bool = False):
    """Run a simple agenda as regression test.

    It verifies that we can start from a set of schedules and
    deterministically and produces an equivalent results with the same
    costs. Results may differ on different platforms event with the same
    seed because we use 2 threads.
    """
    agenda = ExperimentAgenda(experiment_name="test_regression_experiment_agenda", experiments=[
        ExperimentParameters(experiment_id=0, grid_id=0, number_of_agents=2,
                             width=30, height=30,
                             flatland_seed_value=12, asp_seed_value=94,
                             max_num_cities=20, grid_mode=True, max_rail_between_cities=2,
                             max_rail_in_city=6, earliest_malfunction=20, malfunction_duration=20,
                             speed_data={1: 1.0}, number_of_shortest_paths_per_agent=10,
                             weight_route_change=1, weight_lateness_seconds=1, max_window_size_from_earliest=np.inf
                             )])
    experiment_agenda_directory = os.path.join("tests", "02_regression_tests", "data", "test_regression_experiment_agenda", EXPERIMENT_AGENDA_SUBDIRECTORY_NAME)

    # used if module path used in pickle has changed
    # use with wrapper file https://stackoverflow.com/questions/13398462/unpickling-python-objects-with-a-changed-module-path
    if re_save:
        load_schedule(experiment_agenda_directory=experiment_agenda_directory, experiment_id=0)
        load_malfunction(experiment_agenda_directory=experiment_agenda_directory, experiment_id=0)
    if regen:
        save_experiment_agenda_and_hash_to_file(
            experiment_agenda_folder_name=experiment_agenda_directory,
            experiment_agenda=agenda)

    # Import the solver for the experiments
    experiment_folder_name = run_experiment_agenda(
        experiment_agenda=agenda,
        # do not clutter folder
        experiment_output_base_directory='.',
        run_experiments_parallel=1,
        verbose=True,
        experiment_base_directory="tests/02_regression_tests/data/test_regression_experiment_agenda"
    )

    hypothesis_one_data_analysis(
        experiment_output_directory=experiment_folder_name,
        analysis_2d=True,
        qualitative_analysis_experiment_ids=[0],
    )

    # load results
    experiment_results_for_analysis = load_and_expand_experiment_results_from_data_folder(f"{experiment_folder_name}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}")
    if not regen:
        delete_experiment_folder(experiment_folder_name)
    result_dict = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_for_analysis).to_dict()

    expected_result_dict = {
        # costs in full and delta are delay with respect to constraints induced by malfunction,
        # i.e. malfunction has to be added to get delay with respect to initial schedule!
        'costs_delta_after_malfunction': {0: 0.0}, 'costs_full': {0: 0.0}, 'costs_full_after_malfunction': {0: 0.0},
        'experiment_id': {0: 0}, 'max_num_cities': {0: 20}, 'max_rail_between_cities': {0: 2},
        'max_rail_in_city': {0: 6}, 'n_agents': {0: 2}, 'size': {0: 30}}
    print("solution_full_after_malfunction")
    print(experiment_results_for_analysis[0].solution_full_after_malfunction)
    print("solution_delta_after_malfunction")
    print(experiment_results_for_analysis[0].solution_delta_after_malfunction)

    for key in expected_result_dict:
        if expected_result_dict[key] != result_dict[key]:
            rsp_logger.warn(f"{key} should be equal; expected{expected_result_dict[key]}, but got {result_dict[key]}")
        assert expected_result_dict[key] == result_dict[key], \
            f"{key} should be equal; expected{expected_result_dict[key]}, but got {result_dict[key]}"


def test_hypothesis_one_pipeline_all_in_one():
    """Run a simple agenda and save and load the results.

    Check that loading gives the same result.
    """
    experiment_base_directory = create_experiment_folder_name("test_hypothesis_one_pipeline_all_in_one")
    experiment_folder_name, experiment_agenda = \
        hypothesis_one_pipeline_all_in_one(
            experiment_base_directory=experiment_base_directory,
            parameter_ranges_and_speed_data=ParameterRangesAndSpeedData(
                parameter_ranges=ParameterRanges(agent_range=[2, 2, 1],
                                                 size_range=[30, 30, 1],
                                                 in_city_rail_range=[6, 6, 1],
                                                 out_city_rail_range=[2, 2, 1],
                                                 city_range=[20, 20, 1],
                                                 earliest_malfunction=[20, 20, 1],
                                                 malfunction_duration=[20, 20, 1],
                                                 number_of_shortest_paths_per_agent=[10, 10, 1],
                                                 max_window_size_from_earliest=[np.inf, np.inf, 1],
                                                 asp_seed_value=[94, 94, 1],
                                                 weight_route_change=[60, 60, 1],
                                                 weight_lateness_seconds=[1, 1, 1]),
                # Define the desired speed profiles
                speed_data={1.: 1}
            )
        )

    # load results
    experiment_data_folder = f"{experiment_folder_name}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}"
    loaded_results: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_data_folder(
        experiment_data_folder_name=experiment_data_folder)
    delete_experiment_folder(experiment_base_directory)

    # since we do not return the results in memory from run_experiment_agenda (SIM-393), do some sanity checks:
    assert len(loaded_results) == 1, len(loaded_results)
    loaded_result: ExperimentResultsAnalysis = loaded_results[0]
    assert loaded_result.results_full_after_malfunction.solver_statistics is not None

    experiment_results = loaded_result
    experiment_parameters = experiment_agenda.experiments[0]

    # check that asp seed value is received in solver
    assert experiment_results.results_full.solver_seed == experiment_parameters.asp_seed_value, \
        f"actual={experiment_results.results_full.solver_seed}, " \
        f"expected={experiment_parameters.asp_seed_value}"
    assert experiment_results.results_full_after_malfunction.solver_seed == experiment_parameters.asp_seed_value, \
        f"actual={experiment_results.results_full_after_malfunction.solver_seed}, " \
        f"expected={experiment_parameters.asp_seed_value}"
    assert experiment_results.results_delta_after_malfunction.solver_seed == experiment_parameters.asp_seed_value, \
        f"actual={experiment_results.results_delta_after_malfunction.solver_seed}, " \
        f"expected={experiment_parameters.asp_seed_value}"


def _assert_results_dict_equals(experiment_results: List[ExperimentResults],
                                loaded_results: List[ExperimentResultsAnalysis]):
    loaded_result_dict = convert_list_of_experiment_results_analysis_to_data_frame(loaded_results).to_dict()
    experiment_results_dict = convert_list_of_experiment_results_to_data_frame(experiment_results).to_dict()
    for key in experiment_results_dict:
        if key.startswith("problem_"):
            assert len(loaded_result_dict[key]) == len(experiment_results_dict[key])
            for index in loaded_result_dict[key]:
                assert schedule_problem_description_equals(loaded_result_dict[key][index],
                                                           experiment_results_dict[key][index]), \
                    f"not equal {key}{index}: \n" \
                    f"  loaded: {loaded_result_dict[key][index]}\n" \
                    f"  in memory: {experiment_results_dict[key][index]}"
        elif key.startswith('results_'):
            assert len(loaded_result_dict[key]) == len(experiment_results_dict[key])
            for index in loaded_result_dict[key]:
                equals_modulo_solve_time = schedule_experiment_results_equals_modulo_solve_time(
                    experiment_results_dict[key][index],
                    loaded_result_dict[key][index])

                assert equals_modulo_solve_time, \
                    f"{key} for index {index} should be equal modulo solve_time; \n" \
                    f"  expected{experiment_results_dict[key][index]}, \n" \
                    f"  actual {loaded_result_dict[key][index]}"
        else:
            assert experiment_results_dict[key] == loaded_result_dict[key], \
                f"{key} should be equal; \n" \
                f"  expected{experiment_results_dict[key]}, \n" \
                f"  actual {loaded_result_dict[key]}"


def test_run_alpha_beta(regen_schedule: bool = False, re_save: bool = False):
    """Ensure that we get the exact same solution if we multiply the weights
    for route change and lateness by the same factor."""

    experiment_parameters = ExperimentParameters(
        experiment_id=9, grid_id=0, number_of_agents=11,
        speed_data={1.0: 1.0, 0.5: 0.0, 0.3333333333333333: 0.0, 0.25: 0.0}, width=30, height=30,
        flatland_seed_value=12, asp_seed_value=94,
        max_num_cities=20, grid_mode=False, max_rail_between_cities=2, max_rail_in_city=6, earliest_malfunction=20,
        malfunction_duration=20, number_of_shortest_paths_per_agent=10,
        weight_route_change=20, weight_lateness_seconds=1, max_window_size_from_earliest=np.inf
    )
    scale = 5
    experiment_parameters_scaled = ExperimentParameters(**dict(
        experiment_parameters._asdict(),
        **{
            'weight_route_change': experiment_parameters.weight_route_change * scale,
            'weight_lateness_seconds': experiment_parameters.weight_lateness_seconds * scale
        }))

    static_rail_env = create_env_from_experiment_parameters(experiment_parameters)
    static_rail_env.load_resource('tests.02_regression_tests.data.alpha_beta', "static_env_alpha_beta.pkl")

    # used if module path used in pickle has changed
    # use with wrapper file https://stackoverflow.com/questions/13398462/unpickling-python-objects-with-a-changed-module-path
    if re_save:
        for experiment_id in range(2):
            load_schedule(
                experiment_agenda_directory="tests/02_regression_tests/data/alpha_beta",
                experiment_id=experiment_id,
                re_save=True
            )
            load_malfunction(
                experiment_agenda_directory="tests/02_regression_tests/data/alpha_beta",
                experiment_id=experiment_id,
                re_save=True
            )

    # since schedule generation is not deterministic, we need to pickle the output of A.2 experiment setup
    # regen_schedule to fix the regression test in case of breaking API change in the pickled content
    if regen_schedule:
        schedule_scaled = gen_schedule(
            experiment_parameters=experiment_parameters_scaled,
        )
        malfunction_scaled = gen_malfunction(
            earliest_malfunction=experiment_parameters.earliest_malfunction,
            malfunction_duration=experiment_parameters.malfunction_duration,
            schedule_trainruns=schedule_scaled.schedule_experiment_result.trainruns_dict
        )
        schedule = gen_schedule(
            experiment_parameters=experiment_parameters
        )
        malfunction = gen_malfunction(
            earliest_malfunction=experiment_parameters.earliest_malfunction,
            malfunction_duration=experiment_parameters.malfunction_duration,
            schedule_trainruns=schedule.schedule_experiment_result.trainruns_dict
        )
        save_schedule(
            schedule=schedule_scaled,
            experiment_agenda_directory="tests/02_regression_tests/data/alpha_beta",
            experiment_id=0)
        save_malfunction(
            experiment_malfunction=malfunction_scaled,
            experiment_agenda_directory="tests/02_regression_tests/data/alpha_beta",
            experiment_id=0)
        save_schedule(
            schedule=schedule,
            experiment_agenda_directory="tests/02_regression_tests/data/alpha_beta",
            experiment_id=0)
        save_malfunction(
            experiment_malfunction=malfunction,
            experiment_agenda_directory="tests/02_regression_tests/data/alpha_beta",
            experiment_id=0)

    schedule_scaled = load_schedule(
        experiment_agenda_directory="tests/02_regression_tests/data/alpha_beta",
        experiment_id=0
    )
    malfunction_scaled = load_malfunction(
        experiment_agenda_directory="tests/02_regression_tests/data/alpha_beta",
        experiment_id=0
    )
    schedule = load_schedule(
        experiment_agenda_directory="tests/02_regression_tests/data/alpha_beta",
        experiment_id=1
    )
    malfunction = load_malfunction(
        experiment_agenda_directory="tests/02_regression_tests/data/alpha_beta",
        experiment_id=1
    )

    experiment_result_scaled: ExperimentResults = run_experiment_from_schedule_and_malfunction(
        schedule=schedule_scaled,
        experiment_malfunction=malfunction_scaled,
        experiment_parameters=experiment_parameters_scaled,
    )

    experiment_result: ExperimentResults = run_experiment_from_schedule_and_malfunction(
        schedule=schedule,
        experiment_malfunction=malfunction,
        experiment_parameters=experiment_parameters,
    )

    # although re-scheduling is not deterministic, it should produce solutions with the same costs
    costs_full_after_malfunction = experiment_result.results_full_after_malfunction.optimization_costs
    assert costs_full_after_malfunction > 0
    costs_full_after_malfunction_scaled = experiment_result_scaled.results_full_after_malfunction.optimization_costs
    assert (costs_full_after_malfunction * scale ==
            costs_full_after_malfunction_scaled)
    assert (experiment_result.results_full_after_malfunction.trainruns_dict ==
            experiment_result_scaled.results_full_after_malfunction.trainruns_dict)
