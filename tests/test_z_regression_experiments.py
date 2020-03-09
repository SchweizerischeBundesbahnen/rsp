"""Run tests for different experiment methods."""
import time
from typing import List

import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.experiment_solvers.data_types import schedule_experiment_results_equals_modulo_solve_time
from rsp.experiment_solvers.data_types import ScheduleAndMalfunction
from rsp.experiment_solvers.experiment_solver import ASPExperimentSolver
from rsp.hypothesis_one_data_analysis import hypothesis_one_data_analysis
from rsp.route_dag.route_dag import schedule_problem_description_equals
from rsp.utils.data_types import convert_list_of_experiment_results_analysis_to_data_frame
from rsp.utils.data_types import convert_list_of_experiment_results_to_data_frame
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ExperimentResults
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.data_types import ParameterRanges
from rsp.utils.experiments import create_env_pair_for_experiment
from rsp.utils.experiments import create_experiment_agenda
from rsp.utils.experiments import delete_experiment_folder
from rsp.utils.experiments import load_and_expand_experiment_results_from_folder
from rsp.utils.experiments import run_experiment
from rsp.utils.experiments import run_experiment_agenda


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
    static_env, dynamic_env = create_env_pair_for_experiment(params=test_parameters)
    print(static_env.rail.grid.tolist())

    # Check that the same number of agents were created
    assert static_env.get_num_agents() == dynamic_env.get_num_agents()

    # Check that the same grid was created
    assert static_env.rail.grid.tolist() == expected_grid
    assert dynamic_env.rail.grid.tolist() == expected_grid

    # Check agent information
    for agent_index in range(static_env.get_num_agents()):
        assert static_env.agents[agent_index] == dynamic_env.agents[agent_index]


def test_regression_experiment_agenda():
    """Run a simple agenda as regression test."""
    agenda = ExperimentAgenda(experiment_name="test_regression_experiment_agenda", experiments=[
        ExperimentParameters(experiment_id=0, grid_id=0, number_of_agents=2,
                             width=30, height=30,
                             flatland_seed_value=12, asp_seed_value=94,
                             max_num_cities=20, grid_mode=True, max_rail_between_cities=2,
                             max_rail_in_city=6, earliest_malfunction=20, malfunction_duration=20,
                             speed_data={1: 1.0}, number_of_shortest_paths_per_agent=10,
                             weight_route_change=1, weight_lateness_seconds=1, max_window_size_from_earliest=np.inf
                             )])

    # Import the solver for the experiments
    experiment_folder_name = run_experiment_agenda(agenda, run_experiments_parallel=False, verbose=True)

    hypothesis_one_data_analysis(
        data_folder=experiment_folder_name,
        analysis_2d=True,
        analysis_3d=False,
        malfunction_analysis=False,
        qualitative_analysis_experiment_ids=[0],
        flatland_rendering=False
    )

    # load results
    experiment_results_for_analysis = load_and_expand_experiment_results_from_folder(experiment_folder_name)
    delete_experiment_folder(experiment_folder_name)
    result_dict = convert_list_of_experiment_results_analysis_to_data_frame(experiment_results_for_analysis).to_dict()

    expected_result_dict = {
        # costs in full and delta are delay with respect to constraints induced by malfunction,
        # i.e. malfunction has to be added to get delay with respect to initial schedule!
        'costs_delta_after_malfunction': {0: 0.0}, 'costs_full': {0: 0.0}, 'costs_full_after_malfunction': {0: 0.0},
        'delta': {0: {0: [TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(21, 29), direction=2)),
                          TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(22, 29), direction=2)),
                          TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(23, 29), direction=2)),
                          TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(24, 29), direction=2)),
                          TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(24, 28), direction=3)),
                          TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(24, 27), direction=3)),
                          TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(24, 26), direction=3)),
                          TrainrunWaypoint(scheduled_at=47, waypoint=Waypoint(position=(24, 25), direction=3)),
                          TrainrunWaypoint(scheduled_at=48, waypoint=Waypoint(position=(24, 24), direction=3)),
                          TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(24, 23), direction=3))],
                      1: [TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(23, 29), direction=1)),
                          TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(22, 29), direction=0)),
                          TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(21, 29), direction=0)),
                          TrainrunWaypoint(scheduled_at=47, waypoint=Waypoint(position=(20, 29), direction=0)),
                          TrainrunWaypoint(scheduled_at=48, waypoint=Waypoint(position=(19, 29), direction=0)),
                          TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(18, 29), direction=0)),
                          TrainrunWaypoint(scheduled_at=50, waypoint=Waypoint(position=(17, 29), direction=0)),
                          TrainrunWaypoint(scheduled_at=51, waypoint=Waypoint(position=(16, 29), direction=0)),
                          TrainrunWaypoint(scheduled_at=52, waypoint=Waypoint(position=(15, 29), direction=0)),
                          TrainrunWaypoint(scheduled_at=53, waypoint=Waypoint(position=(14, 29), direction=0)),
                          TrainrunWaypoint(scheduled_at=54, waypoint=Waypoint(position=(13, 29), direction=0)),
                          TrainrunWaypoint(scheduled_at=55, waypoint=Waypoint(position=(12, 29), direction=0)),
                          TrainrunWaypoint(scheduled_at=56, waypoint=Waypoint(position=(11, 29), direction=0)),
                          TrainrunWaypoint(scheduled_at=57, waypoint=Waypoint(position=(10, 29), direction=0)),
                          TrainrunWaypoint(scheduled_at=58, waypoint=Waypoint(position=(9, 29), direction=0)),
                          TrainrunWaypoint(scheduled_at=59, waypoint=Waypoint(position=(8, 29), direction=0)),
                          TrainrunWaypoint(scheduled_at=60, waypoint=Waypoint(position=(8, 28), direction=3)),
                          TrainrunWaypoint(scheduled_at=61, waypoint=Waypoint(position=(8, 27), direction=3)),
                          TrainrunWaypoint(scheduled_at=62, waypoint=Waypoint(position=(8, 26), direction=3)),
                          TrainrunWaypoint(scheduled_at=63, waypoint=Waypoint(position=(7, 26), direction=0)),
                          TrainrunWaypoint(scheduled_at=64, waypoint=Waypoint(position=(7, 25), direction=3)),
                          TrainrunWaypoint(scheduled_at=65, waypoint=Waypoint(position=(7, 24), direction=3)),
                          TrainrunWaypoint(scheduled_at=66, waypoint=Waypoint(position=(7, 23), direction=3))]}},
        'experiment_id': {0: 0}, 'max_num_cities': {0: 20}, 'max_rail_between_cities': {0: 2},
        'max_rail_in_city': {0: 6}, 'n_agents': {0: 2}, 'size': {0: 30},
        'solution_delta_after_malfunction': {0: {
            0: [TrainrunWaypoint(scheduled_at=0, waypoint=Waypoint(position=(8, 23), direction=1)),
                TrainrunWaypoint(scheduled_at=2, waypoint=Waypoint(position=(8, 24), direction=1)),
                TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(8, 25), direction=1)),
                TrainrunWaypoint(scheduled_at=4, waypoint=Waypoint(position=(8, 26), direction=1)),
                TrainrunWaypoint(scheduled_at=5, waypoint=Waypoint(position=(8, 27), direction=1)),
                TrainrunWaypoint(scheduled_at=6, waypoint=Waypoint(position=(8, 28), direction=1)),
                TrainrunWaypoint(scheduled_at=7, waypoint=Waypoint(position=(8, 29), direction=1)),
                TrainrunWaypoint(scheduled_at=8, waypoint=Waypoint(position=(9, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=9, waypoint=Waypoint(position=(10, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=10, waypoint=Waypoint(position=(11, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=11, waypoint=Waypoint(position=(12, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=12, waypoint=Waypoint(position=(13, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=13, waypoint=Waypoint(position=(14, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(15, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=15, waypoint=Waypoint(position=(16, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=16, waypoint=Waypoint(position=(17, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(18, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(19, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(20, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(21, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(22, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(23, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(24, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(24, 28), direction=3)),
                TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(24, 27), direction=3)),
                TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(24, 26), direction=3)),
                TrainrunWaypoint(scheduled_at=47, waypoint=Waypoint(position=(24, 25), direction=3)),
                TrainrunWaypoint(scheduled_at=48, waypoint=Waypoint(position=(24, 24), direction=3)),
                TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(24, 23), direction=3))],
            1: [TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(23, 23), direction=1)),
                TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(23, 24), direction=1)),
                TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(23, 25), direction=1)),
                TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(23, 26), direction=1)),
                TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(23, 27), direction=1)),
                TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(23, 28), direction=1)),
                TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(23, 29), direction=1)),
                TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(22, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(21, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=47, waypoint=Waypoint(position=(20, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=48, waypoint=Waypoint(position=(19, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(18, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=50, waypoint=Waypoint(position=(17, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=51, waypoint=Waypoint(position=(16, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=52, waypoint=Waypoint(position=(15, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=53, waypoint=Waypoint(position=(14, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=54, waypoint=Waypoint(position=(13, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=55, waypoint=Waypoint(position=(12, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=56, waypoint=Waypoint(position=(11, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=57, waypoint=Waypoint(position=(10, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=58, waypoint=Waypoint(position=(9, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=59, waypoint=Waypoint(position=(8, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=60, waypoint=Waypoint(position=(7, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=61, waypoint=Waypoint(position=(7, 28), direction=3)),
                TrainrunWaypoint(scheduled_at=62, waypoint=Waypoint(position=(7, 27), direction=3)),
                TrainrunWaypoint(scheduled_at=63, waypoint=Waypoint(position=(7, 26), direction=3)),
                TrainrunWaypoint(scheduled_at=64, waypoint=Waypoint(position=(7, 25), direction=3)),
                TrainrunWaypoint(scheduled_at=65, waypoint=Waypoint(position=(7, 24), direction=3)),
                TrainrunWaypoint(scheduled_at=66, waypoint=Waypoint(position=(7, 23), direction=3))]}},
        'solution_full': {0: {0: [TrainrunWaypoint(scheduled_at=0, waypoint=Waypoint(position=(8, 23), direction=1)),
                                  TrainrunWaypoint(scheduled_at=2, waypoint=Waypoint(position=(8, 24), direction=1)),
                                  TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(8, 25), direction=1)),
                                  TrainrunWaypoint(scheduled_at=4, waypoint=Waypoint(position=(8, 26), direction=1)),
                                  TrainrunWaypoint(scheduled_at=5, waypoint=Waypoint(position=(8, 27), direction=1)),
                                  TrainrunWaypoint(scheduled_at=6, waypoint=Waypoint(position=(8, 28), direction=1)),
                                  TrainrunWaypoint(scheduled_at=7, waypoint=Waypoint(position=(8, 29), direction=1)),
                                  TrainrunWaypoint(scheduled_at=8, waypoint=Waypoint(position=(9, 29), direction=2)),
                                  TrainrunWaypoint(scheduled_at=9, waypoint=Waypoint(position=(10, 29), direction=2)),
                                  TrainrunWaypoint(scheduled_at=10, waypoint=Waypoint(position=(11, 29), direction=2)),
                                  TrainrunWaypoint(scheduled_at=11, waypoint=Waypoint(position=(12, 29), direction=2)),
                                  TrainrunWaypoint(scheduled_at=12, waypoint=Waypoint(position=(13, 29), direction=2)),
                                  TrainrunWaypoint(scheduled_at=13, waypoint=Waypoint(position=(14, 29), direction=2)),
                                  TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(15, 29), direction=2)),
                                  TrainrunWaypoint(scheduled_at=15, waypoint=Waypoint(position=(16, 29), direction=2)),
                                  TrainrunWaypoint(scheduled_at=16, waypoint=Waypoint(position=(17, 29), direction=2)),
                                  TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(18, 29), direction=2)),
                                  TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(19, 29), direction=2)),
                                  TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(20, 29), direction=2)),
                                  TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(21, 29), direction=2)),
                                  TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(22, 29), direction=2)),
                                  TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(23, 29), direction=2)),
                                  TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(24, 29), direction=2)),
                                  TrainrunWaypoint(scheduled_at=24, waypoint=Waypoint(position=(24, 28), direction=3)),
                                  TrainrunWaypoint(scheduled_at=25, waypoint=Waypoint(position=(24, 27), direction=3)),
                                  TrainrunWaypoint(scheduled_at=26, waypoint=Waypoint(position=(24, 26), direction=3)),
                                  TrainrunWaypoint(scheduled_at=27, waypoint=Waypoint(position=(24, 25), direction=3)),
                                  TrainrunWaypoint(scheduled_at=28, waypoint=Waypoint(position=(24, 24), direction=3)),
                                  TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(24, 23), direction=3))],
                              1: [TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(23, 23), direction=1)),
                                  TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(23, 24), direction=1)),
                                  TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(23, 25), direction=1)),
                                  TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(23, 26), direction=1)),
                                  TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(23, 27), direction=1)),
                                  TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(23, 28), direction=1)),
                                  TrainrunWaypoint(scheduled_at=24, waypoint=Waypoint(position=(23, 29), direction=1)),
                                  TrainrunWaypoint(scheduled_at=25, waypoint=Waypoint(position=(22, 29), direction=0)),
                                  TrainrunWaypoint(scheduled_at=26, waypoint=Waypoint(position=(21, 29), direction=0)),
                                  TrainrunWaypoint(scheduled_at=27, waypoint=Waypoint(position=(20, 29), direction=0)),
                                  TrainrunWaypoint(scheduled_at=28, waypoint=Waypoint(position=(19, 29), direction=0)),
                                  TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(18, 29), direction=0)),
                                  TrainrunWaypoint(scheduled_at=30, waypoint=Waypoint(position=(17, 29), direction=0)),
                                  TrainrunWaypoint(scheduled_at=31, waypoint=Waypoint(position=(16, 29), direction=0)),
                                  TrainrunWaypoint(scheduled_at=32, waypoint=Waypoint(position=(15, 29), direction=0)),
                                  TrainrunWaypoint(scheduled_at=33, waypoint=Waypoint(position=(14, 29), direction=0)),
                                  TrainrunWaypoint(scheduled_at=34, waypoint=Waypoint(position=(13, 29), direction=0)),
                                  TrainrunWaypoint(scheduled_at=35, waypoint=Waypoint(position=(12, 29), direction=0)),
                                  TrainrunWaypoint(scheduled_at=36, waypoint=Waypoint(position=(11, 29), direction=0)),
                                  TrainrunWaypoint(scheduled_at=37, waypoint=Waypoint(position=(10, 29), direction=0)),
                                  TrainrunWaypoint(scheduled_at=38, waypoint=Waypoint(position=(9, 29), direction=0)),
                                  TrainrunWaypoint(scheduled_at=39, waypoint=Waypoint(position=(8, 29), direction=0)),
                                  TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(8, 28), direction=3)),
                                  TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(8, 27), direction=3)),
                                  TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(8, 26), direction=3)),
                                  TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(7, 26), direction=0)),
                                  TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(7, 25), direction=3)),
                                  TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(7, 24), direction=3)),
                                  TrainrunWaypoint(scheduled_at=46,
                                                   waypoint=Waypoint(position=(7, 23), direction=3))]}},
        'solution_full_after_malfunction': {0: {
            0: [TrainrunWaypoint(scheduled_at=0, waypoint=Waypoint(position=(8, 23), direction=1)),
                TrainrunWaypoint(scheduled_at=2, waypoint=Waypoint(position=(8, 24), direction=1)),
                TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(8, 25), direction=1)),
                TrainrunWaypoint(scheduled_at=4, waypoint=Waypoint(position=(8, 26), direction=1)),
                TrainrunWaypoint(scheduled_at=5, waypoint=Waypoint(position=(8, 27), direction=1)),
                TrainrunWaypoint(scheduled_at=6, waypoint=Waypoint(position=(8, 28), direction=1)),
                TrainrunWaypoint(scheduled_at=7, waypoint=Waypoint(position=(8, 29), direction=1)),
                TrainrunWaypoint(scheduled_at=8, waypoint=Waypoint(position=(9, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=9, waypoint=Waypoint(position=(10, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=10, waypoint=Waypoint(position=(11, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=11, waypoint=Waypoint(position=(12, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=12, waypoint=Waypoint(position=(13, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=13, waypoint=Waypoint(position=(14, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(15, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=15, waypoint=Waypoint(position=(16, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=16, waypoint=Waypoint(position=(17, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(18, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(19, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(20, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(21, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(22, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(23, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(24, 29), direction=2)),
                TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(24, 28), direction=3)),
                TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(24, 27), direction=3)),
                TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(24, 26), direction=3)),
                TrainrunWaypoint(scheduled_at=47, waypoint=Waypoint(position=(24, 25), direction=3)),
                TrainrunWaypoint(scheduled_at=48, waypoint=Waypoint(position=(24, 24), direction=3)),
                TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(24, 23), direction=3))],
            1: [TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(23, 23), direction=1)),
                TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(23, 24), direction=1)),
                TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(23, 25), direction=1)),
                TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(23, 26), direction=1)),
                TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(23, 27), direction=1)),
                TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(23, 28), direction=1)),
                TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(23, 29), direction=1)),
                TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(22, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(21, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=47, waypoint=Waypoint(position=(20, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=48, waypoint=Waypoint(position=(19, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(18, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=50, waypoint=Waypoint(position=(17, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=51, waypoint=Waypoint(position=(16, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=52, waypoint=Waypoint(position=(15, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=53, waypoint=Waypoint(position=(14, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=54, waypoint=Waypoint(position=(13, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=55, waypoint=Waypoint(position=(12, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=56, waypoint=Waypoint(position=(11, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=57, waypoint=Waypoint(position=(10, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=58, waypoint=Waypoint(position=(9, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=59, waypoint=Waypoint(position=(8, 29), direction=0)),
                TrainrunWaypoint(scheduled_at=60, waypoint=Waypoint(position=(8, 28), direction=3)),
                TrainrunWaypoint(scheduled_at=61, waypoint=Waypoint(position=(8, 27), direction=3)),
                TrainrunWaypoint(scheduled_at=62, waypoint=Waypoint(position=(8, 26), direction=3)),
                TrainrunWaypoint(scheduled_at=63, waypoint=Waypoint(position=(7, 26), direction=0)),
                TrainrunWaypoint(scheduled_at=64, waypoint=Waypoint(position=(7, 25), direction=3)),
                TrainrunWaypoint(scheduled_at=65, waypoint=Waypoint(position=(7, 24), direction=3)),
                TrainrunWaypoint(scheduled_at=66, waypoint=Waypoint(position=(7, 23), direction=3))]}},
        'time_delta_after_malfunction': {0: 0.208}, 'time_full': {0: 0.205}, 'time_full_after_malfunction': {0: 0.257}}

    for key in expected_result_dict:
        # TODO remove keys in expected_result_dict instead
        skip = key.startswith("time")
        skip = skip or key.startswith("solution")
        skip = skip or key.startswith("delta")
        skip = skip or key.startswith('problem')
        if not skip:
            assert expected_result_dict[key] == result_dict[key], \
                f"{key} should be equal; expected{expected_result_dict[key]}, but got {result_dict[key]}"


def test_save_and_load_experiment_results():
    """Run a simple agenda and save and load the results.

    Check that loading gives the same result.
    """
    agenda = ExperimentAgenda(experiment_name="test_save_and_load_experiment_results", experiments=[
        ExperimentParameters(experiment_id=0, grid_id=0, number_of_agents=2,
                             width=30, height=30,
                             flatland_seed_value=12, asp_seed_value=94,
                             max_num_cities=20, grid_mode=True, max_rail_between_cities=2,
                             max_rail_in_city=6, earliest_malfunction=20, malfunction_duration=20,
                             speed_data={1: 1.0}, number_of_shortest_paths_per_agent=10,
                             weight_route_change=1, weight_lateness_seconds=1, max_window_size_from_earliest=np.inf)])

    solver = ASPExperimentSolver()
    experiment_folder_name = run_experiment_agenda(agenda, run_experiments_parallel=False)

    # load results
    loaded_results: List[ExperimentResultsAnalysis] = load_and_expand_experiment_results_from_folder(
        experiment_folder_name)
    delete_experiment_folder(experiment_folder_name)

    experiment_results_list = []
    for current_experiment_parameters in agenda.experiments:
        single_experiment_result: ExperimentResults = run_experiment(solver=solver,
                                                                     experiment_parameters=current_experiment_parameters,
                                                                     verbose=False)
        experiment_results_list.append(single_experiment_result)

    _assert_results_dict_equals(experiment_results_list, loaded_results)


def _assert_results_dict_equals(experiment_results: List[ExperimentResults],
                                loaded_results: List[ExperimentResultsAnalysis]):
    loaded_result_dict = convert_list_of_experiment_results_analysis_to_data_frame(loaded_results).to_dict()
    experiment_results_dict = convert_list_of_experiment_results_to_data_frame(experiment_results).to_dict()
    for key in experiment_results_dict:
        if key.startswith("problem_"):
            assert len(loaded_result_dict[key]) == len(experiment_results_dict[key])
            for index in loaded_result_dict[key]:
                assert schedule_problem_description_equals(loaded_result_dict[key][index],
                                                           experiment_results_dict[key][index])
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


def test_run_full_pipeline():
    """Ensure that the full pipeline runs without error on a simple agenda."""
    agenda = create_experiment_agenda(
        experiment_name="test_run_full_pipeline",
        experiments_per_grid_element=3,
        parameter_ranges=ParameterRanges(agent_range=[2, 2, 1],
                                         size_range=[30, 30, 1],
                                         in_city_rail_range=[6, 6, 1],
                                         out_city_rail_range=[2, 2, 1],
                                         city_range=[20, 20, 1],
                                         earliest_malfunction=[20, 20, 1],
                                         malfunction_duration=[20, 20, 1],
                                         number_of_shortest_paths_per_agent=[10, 10, 1],
                                         max_window_size_from_earliest=[np.inf, np.inf, 1]),
        speed_data={1: 1.0},
    )
    experiment_folder_name = run_experiment_agenda(agenda, run_experiments_parallel=False)

    hypothesis_one_data_analysis(
        data_folder=experiment_folder_name,
        analysis_2d=True,
        analysis_3d=False,
        malfunction_analysis=False,
        qualitative_analysis_experiment_ids=[0],
        flatland_rendering=False
    )

    # cleanup
    delete_experiment_folder(experiment_folder_name)


def test_run_alpha_beta():
    """Ensure that we get the exact same solution if we multiply the weights
    for route change and lateness by the same factor."""

    # TODO SIM-339: skip this test in ci under Linux
    from sys import platform
    if platform == "linux" or platform == "linux2":
        return

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

    solver = ASPExperimentSolver()

    # since Linux and Windows do not produce do not produces the same results with the same seed,
    #  we want to at least control the environments (the schedule will not be the same!)
    # environments not correctly initialized if not created the same way, therefore use create_env_pair_for_experiment
    static_rail_env, malfunction_rail_env = create_env_pair_for_experiment(experiment_parameters)
    # override grid from loaded file
    static_rail_env.load_resource('tests.data.alpha_beta', "static_env_alpha_beta.pkl")
    malfunction_rail_env.load_resource('tests.data.alpha_beta', "malfunction_env_alpha_beta.pkl")

    def malfunction_env_reset():
        malfunction_rail_env.reset(False, False, False, experiment_parameters.flatland_seed_value)

    print("schedule_and malfunction_scaled...")
    start_time = time.time()
    schedule_and_malfunction_scaled: ScheduleAndMalfunction = solver.gen_schedule_and_malfunction(
        static_rail_env=static_rail_env,
        malfunction_rail_env=malfunction_rail_env,
        malfunction_env_reset=malfunction_env_reset,
        experiment_parameters=experiment_parameters_scaled
    )
    print(" -> {:5.3f}s".format(time.time() - start_time))

    print("experiment scaled...")
    start_time = time.time()
    experiment_result_scaled: ExperimentResults = solver._run_experiment_from_environment(
        schedule_and_malfunction=schedule_and_malfunction_scaled,
        malfunction_rail_env=malfunction_rail_env,
        malfunction_env_reset=malfunction_env_reset,
        experiment_parameters=experiment_parameters_scaled,
    )
    print(" -> {:5.3f}s".format(time.time() - start_time))

    print("schedule_and malfunction...")
    start_time = time.time()
    schedule_and_malfunction: ScheduleAndMalfunction = solver.gen_schedule_and_malfunction(
        static_rail_env=static_rail_env,
        malfunction_rail_env=malfunction_rail_env,
        malfunction_env_reset=malfunction_env_reset,
        experiment_parameters=experiment_parameters
    )
    print(" -> {:5.3f}s".format(time.time() - start_time))

    print("experiment...")
    start_time = time.time()
    experiment_result: ExperimentResults = solver._run_experiment_from_environment(
        schedule_and_malfunction=schedule_and_malfunction,
        malfunction_rail_env=malfunction_rail_env,
        malfunction_env_reset=malfunction_env_reset,
        experiment_parameters=experiment_parameters,
    )
    print(" -> {:5.3f}s".format(time.time() - start_time))

    costs_full_after_malfunction = experiment_result.results_full_after_malfunction.optimization_costs
    assert costs_full_after_malfunction > 0
    costs_full_after_malfunction_scaled = experiment_result_scaled.results_full_after_malfunction.optimization_costs
    assert (costs_full_after_malfunction * scale ==
            costs_full_after_malfunction_scaled)
    assert (experiment_result.results_full_after_malfunction.trainruns_dict ==
            experiment_result_scaled.results_full_after_malfunction.trainruns_dict)


def test_seed():
    experiment_parameters = ExperimentParameters(
        experiment_id=0, grid_id=0, number_of_agents=2,
        width=30, height=30,
        flatland_seed_value=12, asp_seed_value=77,
        max_num_cities=20, grid_mode=True, max_rail_between_cities=2,
        max_rail_in_city=6, earliest_malfunction=20, malfunction_duration=20, speed_data={1: 1.0},
        number_of_shortest_paths_per_agent=10, weight_route_change=1, weight_lateness_seconds=1,
        max_window_size_from_earliest=np.inf)

    experiment_results: ExperimentResults = run_experiment(ASPExperimentSolver(),
                                                           experiment_parameters=experiment_parameters)

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


def test_parallel_experiment_execution():
    """Run a parallel experiment agenda."""
    agenda = ExperimentAgenda(experiment_name="test_parallel_experiment_execution", experiments=[
        ExperimentParameters(experiment_id=0, grid_id=0, number_of_agents=2,
                             width=30, height=30,
                             flatland_seed_value=12, asp_seed_value=94,
                             max_num_cities=20, grid_mode=True, max_rail_between_cities=2,
                             max_rail_in_city=6, earliest_malfunction=20, malfunction_duration=20, speed_data={1: 1.0},
                             number_of_shortest_paths_per_agent=10, weight_route_change=1, weight_lateness_seconds=1,
                             max_window_size_from_earliest=np.inf),
        ExperimentParameters(experiment_id=1, grid_id=0, number_of_agents=2,
                             width=30, height=30,
                             flatland_seed_value=13, asp_seed_value=94,
                             max_num_cities=20, grid_mode=True, max_rail_between_cities=2,
                             max_rail_in_city=6, earliest_malfunction=20, malfunction_duration=20, speed_data={1: 1.0},
                             number_of_shortest_paths_per_agent=10, weight_route_change=1, weight_lateness_seconds=1,
                             max_window_size_from_earliest=np.inf),
        ExperimentParameters(experiment_id=2, grid_id=0, number_of_agents=2,
                             width=30, height=30,
                             flatland_seed_value=14, asp_seed_value=94,
                             max_num_cities=20, grid_mode=True, max_rail_between_cities=2,
                             max_rail_in_city=6, earliest_malfunction=20, malfunction_duration=20, speed_data={1: 1.0},
                             number_of_shortest_paths_per_agent=10, weight_route_change=1, weight_lateness_seconds=1,
                             max_window_size_from_earliest=np.inf),
        ExperimentParameters(experiment_id=3, grid_id=0, number_of_agents=3,
                             width=30, height=30,
                             flatland_seed_value=11, asp_seed_value=94,
                             max_num_cities=20, grid_mode=True, max_rail_between_cities=2,
                             max_rail_in_city=7, earliest_malfunction=15, malfunction_duration=15, speed_data={1: 1.0},
                             number_of_shortest_paths_per_agent=10, weight_route_change=1, weight_lateness_seconds=1,
                             max_window_size_from_earliest=np.inf),
        ExperimentParameters(experiment_id=4, grid_id=0, number_of_agents=4,
                             width=30, height=30,
                             flatland_seed_value=10, asp_seed_value=94,
                             max_num_cities=20, grid_mode=True, max_rail_between_cities=2,
                             max_rail_in_city=8, earliest_malfunction=1, malfunction_duration=10, speed_data={1: 1.0},
                             number_of_shortest_paths_per_agent=10, weight_route_change=1, weight_lateness_seconds=1,
                             max_window_size_from_earliest=np.inf)])

    experiment_folder_name = run_experiment_agenda(agenda, run_experiments_parallel=True)
    delete_experiment_folder(experiment_folder_name)


def test_deterministic_1():
    _test_deterministic(
        ExperimentParameters(experiment_id=0, grid_id=0, number_of_agents=2, width=30,
                             height=30,
                             flatland_seed_value=12, asp_seed_value=94,
                             max_num_cities=20, grid_mode=True, max_rail_between_cities=2,
                             max_rail_in_city=6, earliest_malfunction=20, malfunction_duration=20, speed_data={1: 1.0},
                             number_of_shortest_paths_per_agent=10, weight_route_change=1, weight_lateness_seconds=1,
                             max_window_size_from_earliest=np.inf))


def test_deterministic_2():
    _test_deterministic(
        ExperimentParameters(experiment_id=1, grid_id=0, number_of_agents=3, width=30,
                             height=30,
                             flatland_seed_value=11, asp_seed_value=94,
                             max_num_cities=20, grid_mode=True, max_rail_between_cities=2,
                             max_rail_in_city=7, earliest_malfunction=15, malfunction_duration=15, speed_data={1: 1.0},
                             number_of_shortest_paths_per_agent=10, weight_route_change=1, weight_lateness_seconds=1,
                             max_window_size_from_earliest=np.inf))


def test_deterministic_3():
    _test_deterministic(
        ExperimentParameters(experiment_id=2, grid_id=0, number_of_agents=4, width=30,
                             height=30,
                             flatland_seed_value=10, asp_seed_value=94,
                             max_num_cities=20, grid_mode=True, max_rail_between_cities=2,
                             max_rail_in_city=8, earliest_malfunction=1, malfunction_duration=10, speed_data={1: 1.0},
                             number_of_shortest_paths_per_agent=10, weight_route_change=1, weight_lateness_seconds=1,
                             max_window_size_from_earliest=np.inf))


def _test_deterministic(params: ExperimentParameters):
    """Ensure that two runs of the same experiment yields the same result."""

    solver = ASPExperimentSolver()
    single_experiment_result1: ExperimentResults = run_experiment(solver=solver, experiment_parameters=params,
                                                                  verbose=False)
    single_experiment_result2 = run_experiment(solver=solver, experiment_parameters=params, verbose=False)

    _assert_results_dict_equals([single_experiment_result1], [single_experiment_result2])
