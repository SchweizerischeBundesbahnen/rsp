from typing import Dict, List

from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint, Waypoint

from rsp.asp.asp_scheduling_helper import schedule_static, reschedule
from rsp.rescheduling.rescheduling_utils import get_freeze_for_malfunction
from rsp.utils.data_types import Malfunction, ExperimentParameters
from rsp.utils.experiments import create_env_pair_for_experiment


def test_rescheduling():
    test_parameters = ExperimentParameters(experiment_id=0, trials_in_experiment=10, number_of_agents=2, width=30,
                                           height=30, seed_value=12, max_num_cities=20, grid_mode=True,
                                           max_rail_between_cities=2, max_rail_in_city=6, earliest_malfunction=20,
                                           malfunction_duration=20)
    static_env, dynamic_env = create_env_pair_for_experiment(params=test_parameters)

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
    assert dynamic_env.rail.grid.tolist() == expected_grid

    expected_schedule_train_runs = {
        0: [TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(8, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(8, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(8, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(8, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(8, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(8, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=24, waypoint=Waypoint(position=(8, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=25, waypoint=Waypoint(position=(9, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=26, waypoint=Waypoint(position=(10, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=27, waypoint=Waypoint(position=(11, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=28, waypoint=Waypoint(position=(12, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(13, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=30, waypoint=Waypoint(position=(14, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=31, waypoint=Waypoint(position=(15, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=32, waypoint=Waypoint(position=(16, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=33, waypoint=Waypoint(position=(17, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=34, waypoint=Waypoint(position=(18, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=35, waypoint=Waypoint(position=(19, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=36, waypoint=Waypoint(position=(20, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=37, waypoint=Waypoint(position=(21, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=38, waypoint=Waypoint(position=(22, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=39, waypoint=Waypoint(position=(23, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(23, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(23, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(24, 27), direction=2)),
            TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(24, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(24, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(24, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(24, 23), direction=3))],
        1: [TrainrunWaypoint(scheduled_at=0, waypoint=Waypoint(position=(23, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=2, waypoint=Waypoint(position=(23, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(23, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=4, waypoint=Waypoint(position=(23, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=5, waypoint=Waypoint(position=(23, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=6, waypoint=Waypoint(position=(23, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=7, waypoint=Waypoint(position=(23, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=8, waypoint=Waypoint(position=(22, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=9, waypoint=Waypoint(position=(21, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=10, waypoint=Waypoint(position=(20, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=11, waypoint=Waypoint(position=(19, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=12, waypoint=Waypoint(position=(18, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=13, waypoint=Waypoint(position=(17, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(16, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=15, waypoint=Waypoint(position=(15, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=16, waypoint=Waypoint(position=(14, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(13, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(12, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(11, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(10, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(9, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(8, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(7, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=24, waypoint=Waypoint(position=(7, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=25, waypoint=Waypoint(position=(7, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=26, waypoint=Waypoint(position=(7, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=27, waypoint=Waypoint(position=(7, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=28, waypoint=Waypoint(position=(7, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(7, 23), direction=3))]}
    expected_freeze = {0: [TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(8, 23), direction=1)),
                           TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(8, 24), direction=1)),
                           TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(8, 25), direction=1))],
                       1: [TrainrunWaypoint(scheduled_at=0, waypoint=Waypoint(position=(23, 23), direction=1)),
                           TrainrunWaypoint(scheduled_at=2, waypoint=Waypoint(position=(23, 24), direction=1)),
                           TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(23, 25), direction=1)),
                           TrainrunWaypoint(scheduled_at=4, waypoint=Waypoint(position=(23, 26), direction=1)),
                           TrainrunWaypoint(scheduled_at=5, waypoint=Waypoint(position=(23, 27), direction=1)),
                           TrainrunWaypoint(scheduled_at=6, waypoint=Waypoint(position=(23, 28), direction=1)),
                           TrainrunWaypoint(scheduled_at=7, waypoint=Waypoint(position=(23, 29), direction=1)),
                           TrainrunWaypoint(scheduled_at=8, waypoint=Waypoint(position=(22, 29), direction=0)),
                           TrainrunWaypoint(scheduled_at=9, waypoint=Waypoint(position=(21, 29), direction=0)),
                           TrainrunWaypoint(scheduled_at=10, waypoint=Waypoint(position=(20, 29), direction=0)),
                           TrainrunWaypoint(scheduled_at=11, waypoint=Waypoint(position=(19, 29), direction=0)),
                           TrainrunWaypoint(scheduled_at=12, waypoint=Waypoint(position=(18, 29), direction=0)),
                           TrainrunWaypoint(scheduled_at=13, waypoint=Waypoint(position=(17, 29), direction=0)),
                           TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(16, 29), direction=0)),
                           TrainrunWaypoint(scheduled_at=15, waypoint=Waypoint(position=(15, 29), direction=0)),
                           TrainrunWaypoint(scheduled_at=16, waypoint=Waypoint(position=(14, 29), direction=0)),
                           TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(13, 29), direction=0)),
                           TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(12, 29), direction=0)),
                           TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(11, 29), direction=0)),
                           TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(10, 29), direction=0)),
                           ]}
    expected_reschedule = {0: [TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(8, 23), direction=1)),
                               TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(8, 24), direction=1)),
                               TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(8, 25), direction=1)),
                               TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(8, 26), direction=1)),
                               TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(8, 27), direction=1)),
                               TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(8, 28), direction=1)),
                               TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(8, 29), direction=1)),
                               TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(9, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(10, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=47, waypoint=Waypoint(position=(11, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=48, waypoint=Waypoint(position=(12, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(13, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=50, waypoint=Waypoint(position=(14, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=51, waypoint=Waypoint(position=(15, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=52, waypoint=Waypoint(position=(16, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=53, waypoint=Waypoint(position=(17, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=54, waypoint=Waypoint(position=(18, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=55, waypoint=Waypoint(position=(19, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=56, waypoint=Waypoint(position=(20, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=57, waypoint=Waypoint(position=(21, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=58, waypoint=Waypoint(position=(22, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=59, waypoint=Waypoint(position=(23, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=60, waypoint=Waypoint(position=(23, 28), direction=3)),
                               TrainrunWaypoint(scheduled_at=61, waypoint=Waypoint(position=(23, 27), direction=3)),
                               TrainrunWaypoint(scheduled_at=62, waypoint=Waypoint(position=(24, 27), direction=2)),
                               TrainrunWaypoint(scheduled_at=63, waypoint=Waypoint(position=(24, 26), direction=3)),
                               TrainrunWaypoint(scheduled_at=64, waypoint=Waypoint(position=(24, 25), direction=3)),
                               TrainrunWaypoint(scheduled_at=65, waypoint=Waypoint(position=(24, 24), direction=3)),
                               TrainrunWaypoint(scheduled_at=66, waypoint=Waypoint(position=(24, 23), direction=3))],
                           1: [TrainrunWaypoint(scheduled_at=0, waypoint=Waypoint(position=(23, 23), direction=1)),
                               TrainrunWaypoint(scheduled_at=2, waypoint=Waypoint(position=(23, 24), direction=1)),
                               TrainrunWaypoint(scheduled_at=3, waypoint=Waypoint(position=(23, 25), direction=1)),
                               TrainrunWaypoint(scheduled_at=4, waypoint=Waypoint(position=(23, 26), direction=1)),
                               TrainrunWaypoint(scheduled_at=5, waypoint=Waypoint(position=(23, 27), direction=1)),
                               TrainrunWaypoint(scheduled_at=6, waypoint=Waypoint(position=(23, 28), direction=1)),
                               TrainrunWaypoint(scheduled_at=7, waypoint=Waypoint(position=(23, 29), direction=1)),
                               TrainrunWaypoint(scheduled_at=8, waypoint=Waypoint(position=(22, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=9, waypoint=Waypoint(position=(21, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=10, waypoint=Waypoint(position=(20, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=11, waypoint=Waypoint(position=(19, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=12, waypoint=Waypoint(position=(18, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=13, waypoint=Waypoint(position=(17, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(16, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=15, waypoint=Waypoint(position=(15, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=16, waypoint=Waypoint(position=(14, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(13, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(12, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(11, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(10, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(9, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(8, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(7, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=24, waypoint=Waypoint(position=(7, 28), direction=3)),
                               TrainrunWaypoint(scheduled_at=25, waypoint=Waypoint(position=(7, 27), direction=3)),
                               TrainrunWaypoint(scheduled_at=26, waypoint=Waypoint(position=(7, 26), direction=3)),
                               TrainrunWaypoint(scheduled_at=27, waypoint=Waypoint(position=(7, 25), direction=3)),
                               TrainrunWaypoint(scheduled_at=28, waypoint=Waypoint(position=(7, 24), direction=3)),
                               TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(7, 23), direction=3))]}

    malfunction = Malfunction(time_step=19, agent_id=0, malfunction_duration=20)

    agents_paths_dict, schedule_problem, schedule_result, schedule_solution = schedule_static(10, static_env)

    schedule_trainruns: Dict[int, List[TrainrunWaypoint]] = schedule_solution.get_trainruns_dict()
    assert schedule_trainruns == expected_schedule_train_runs
    freeze = get_freeze_for_malfunction(malfunction, schedule_trainruns, static_env)
    assert freeze == expected_freeze
    schedule_problem.get_freezed_copy_for_rescheduling(malfunction=malfunction, freeze=freeze,
                                                       schedule_trainruns=schedule_trainruns)

    full_reschedule_result, full_reschedule_solution = reschedule(agents_paths_dict, malfunction,
                                                                  malfunction_env_reset=lambda *args, **kwargs: None,
                                                                  malfunction_rail_env=dynamic_env,
                                                                  schedule_problem=schedule_problem,
                                                                  schedule_trainruns=schedule_trainruns,
                                                                  static_rail_env=static_env,
                                                                  rendering=False,
                                                                  debug=False
                                                                  )
    full_reschedule_trainruns: Dict[int, List[TrainrunWaypoint]] = full_reschedule_solution.get_trainruns_dict()

    print(full_reschedule_trainruns)
    assert full_reschedule_trainruns == expected_reschedule

    for agent_id in expected_freeze.keys():
        assert set(full_reschedule_trainruns[agent_id]).issuperset(set(freeze[agent_id])), f"{agent_id}"

    # agent 0: scheduled arrival was 46, new arrival is 66 something not working properly! -> penalty = 20
    # agent 1: scheduled arrival was 29, new arrival is 29 -> penalty = 0
    actual_costs = full_reschedule_solution.asp_solution.stats['summary']['costs'][0]
    assert actual_costs == 20, f"actual costs {actual_costs}"

# TODO add test for scheduling!
# TODO same test but t1 breaks down at about 18
