import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunDict

from rsp.experiment_solvers.experiment_solver import asp_schedule_wrapper
from rsp.experiment_solvers.trainrun_utils import get_sum_running_times_trainruns_dict
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.experiments import create_env_from_experiment_parameters
from rsp.utils.experiments import create_infrastructure_from_rail_env
from rsp.utils.experiments import create_schedule_problem_description_from_instructure


def test_scheduling():
    test_parameters = ExperimentParameters(experiment_id=0, grid_id=0,
                                           number_of_agents=2, width=30,
                                           height=30, flatland_seed_value=12,
                                           asp_seed_value=94, max_num_cities=20,
                                           grid_mode=True,
                                           max_rail_between_cities=2, max_rail_in_city=6, earliest_malfunction=20,
                                           malfunction_duration=20, speed_data={1: 1.0},
                                           number_of_shortest_paths_per_agent=10,
                                           weight_route_change=1,
                                           weight_lateness_seconds=1,
                                           max_window_size_from_earliest=np.inf)
    static_env = create_env_from_experiment_parameters(params=test_parameters)

    expected_grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 16386, 1025, 1025, 1025, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16386, 1025, 1025,
                      1025, 4608, 0, 0, 0, 0],
                     [0, 16386, 1025, 5633, 17411, 3089, 1025, 1025, 1025, 1097, 5633, 17411, 1025, 1025, 1025, 1025,
                      1025, 1025, 1025, 5633, 17411, 3089, 1025, 1025, 1025, 1097, 5633, 17411, 1025, 4608],
                     [0, 49186, 1025, 1097, 3089, 5633, 1025, 1025, 1025, 17411, 1097, 3089, 1025, 1025, 1025, 1025,
                      1025, 1025, 1025, 1097, 3089, 5633, 1025, 1025, 1025, 17411, 1097, 3089, 1025, 37408],
                     [0, 32800, 0, 0, 0, 72, 5633, 1025, 17411, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 5633, 1025,
                      17411, 2064, 0, 0, 0, 32800],
                     [0, 32800, 0, 0, 0, 0, 72, 1025, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 1025, 2064, 0, 0,
                      0, 0, 32800],
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
                     [32800, 49186, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72,
                      37408],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [32800, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      32800],
                     [72, 33897, 1025, 5633, 17411, 1025, 1025, 1025, 1025, 1025, 5633, 17411, 1025, 1025, 1025, 1025,
                      1025, 1025, 1025, 5633, 17411, 1025, 1025, 1025, 1025, 1025, 5633, 17411, 1025, 34864],
                     [0, 72, 1025, 1097, 3089, 5633, 1025, 1025, 1025, 17411, 1097, 3089, 1025, 1025, 1025, 1025, 1025,
                      1025, 1025, 1097, 3089, 5633, 1025, 1025, 1025, 17411, 1097, 3089, 1025, 2064],
                     [0, 0, 0, 0, 0, 72, 1025, 1025, 1025, 2064, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 1025, 1025, 1025,
                      2064, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    assert static_env.rail.grid.tolist() == expected_grid

    schedule_problem = schedule_problem = create_schedule_problem_description_from_instructure(create_infrastructure_from_rail_env(static_env, 10))
    schedule_result = asp_schedule_wrapper(
        schedule_problem_description=schedule_problem,
        asp_seed_value=94,
        no_optimize=False
    )
    schedule_trainruns: TrainrunDict = schedule_result.trainruns_dict

    # sanity check for our expected data
    for agent in static_env.agents:
        assert schedule_trainruns[agent.handle][0].waypoint.position == agent.initial_position
        assert schedule_trainruns[agent.handle][0].waypoint.direction == agent.initial_direction
        assert schedule_trainruns[agent.handle][-1].waypoint.position == agent.target

    # train 0: 23 -> 51
    # train 1: 0 -> 28
    expected_total_running_times = 56

    # optimization costs must be zero since we have no delay with respect to earliest
    expected_objective = 0
    actual_objective = schedule_result.optimization_costs
    assert actual_objective == expected_objective, f"actual_objective={actual_objective}, expected_objective={expected_objective}"
    actual_sum_running_times = get_sum_running_times_trainruns_dict(schedule_result.trainruns_dict)
    assert actual_sum_running_times == expected_total_running_times, \
        f"actual_sum_running_times={actual_sum_running_times}, expected_total_running_times={expected_total_running_times}"
