from typing import Dict, List, Optional

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.malfunction_generators import Malfunction as FLMalfunction
from flatland.envs.malfunction_generators import MalfunctionProcessData
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint, Waypoint
from numpy.random.mtrand import RandomState

from rsp.asp.asp_problem_description import ASPProblemDescription
from rsp.asp.asp_scheduling_helper import reschedule_full_after_malfunction, determine_delta
from rsp.rescheduling.rescheduling_utils import get_freeze_for_malfunction, verify_experiment_freeze_for_agent
from rsp.utils.data_types import ExperimentParameters, ExperimentMalfunction
from rsp.utils.experiments import create_env_pair_for_experiment


# TODO SIM-146 verify action plan: should end with STOP_MOVING, should start with, ...,  no two agents in the same cell at the same time
# TODO SIM-146 verify Trainrun for ASP no two agents in the same cell at the same time, should be ascending, distances at least minimum running time, verify costs

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

    fake_schedule = {
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
                           ],
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

                           ]}
    expected_semi_freeze = {
        0: [TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(8, 25), direction=1))],
        1: [TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(10, 29), direction=0))]
    }
    expected_freeze_visit_only_only = {
        0: [],
        1: []
    }
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

    fake_malfunction = ExperimentMalfunction(time_step=19, agent_id=0, malfunction_duration=20)
    k = 10
    agents_paths_dict = {
        i: get_k_shortest_paths(static_env,
                                agent.initial_position,
                                agent.initial_direction,
                                agent.target,
                                k) for i, agent in enumerate(static_env.agents)
    }
    # we derive the re-schedule problem from the schedule problem
    schedule_problem = ASPProblemDescription(env=static_env,
                                             agents_path_dict=agents_paths_dict)

    freeze_dict = get_freeze_for_malfunction(malfunction=fake_malfunction,
                                             schedule_trainruns=fake_schedule,
                                             static_rail_env=static_env,
                                             agents_path_dict=agents_paths_dict)
    # TODO SIM-146 refactor: write assertions in the new way
    freeze = {agent_id: experiment_freeze.freeze_time_and_visit for agent_id, experiment_freeze in freeze_dict.items()}
    print(freeze_dict)
    semi_freeze = {agent_id: experiment_freeze.freeze_time_and_visit for agent_id, experiment_freeze in
                   freeze_dict.items()}
    for agent_id, experiment_freeze in freeze_dict.items():
        verify_experiment_freeze_for_agent(freeze_dict[agent_id], agents_paths_dict[agent_id])
    # assert freeze == expected_freeze
    # assert semi_freeze == expected_semi_freeze

    schedule_problem.get_copy_for_experiment_freeze(freeze=freeze_dict,
                                                    schedule_trainruns=fake_schedule)

    full_reschedule_result = reschedule_full_after_malfunction(
        malfunction=fake_malfunction,
        malfunction_rail_env=dynamic_env,
        schedule_problem=schedule_problem,
        schedule_trainruns=fake_schedule,
        static_rail_env=static_env,
        rendering=False,
        debug=False,
        malfunction_env_reset=lambda *args, **kwargs: None
    )
    full_reschedule_solution = full_reschedule_result.solution
    full_reschedule_trainruns: Dict[int, List[TrainrunWaypoint]] = full_reschedule_solution.get_trainruns_dict()

    print(full_reschedule_trainruns)

    # agent 0: scheduled arrival was 46, new arrival is 66 -> penalty = 20 (=delay)
    # agent 1: scheduled arrival was 29, new arrival is 29 -> penalty = 0
    actual_costs = full_reschedule_solution.asp_solution.stats['summary']['costs'][0]
    assert actual_costs == 20, f"actual costs {actual_costs}"

    assert full_reschedule_trainruns == expected_reschedule

    for agent_id in expected_freeze.keys():
        assert set(full_reschedule_trainruns[agent_id]).issuperset(set(freeze[agent_id])), f"{agent_id}"

    delta, delta_freeze = determine_delta(full_reschedule_trainruns, fake_malfunction,
                                          fake_schedule, verbose=True)

    assert delta == {
        0: [TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(8, 25), direction=1)),
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
        1: []
    }
    assert delta_freeze == {
        0: [
            TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(8, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(8, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(8, 25), direction=1)),
        ],
        1: [
            TrainrunWaypoint(scheduled_at=0, waypoint=Waypoint(position=(23, 23), direction=1)),
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
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(7, 23), direction=3))]
    }


def test_rescheduling_first_train_goes_earlier():
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

    fake_schedule = {
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
    expected_freeze = {0: [],
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
                           # delay by 20 to enter next: 35 = 15 + 20
                           TrainrunWaypoint(scheduled_at=35, waypoint=Waypoint(position=(15, 29), direction=0)),
                           ]}
    expected_reschedule = {0: [TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(8, 23), direction=1)),
                               TrainrunWaypoint(scheduled_at=16, waypoint=Waypoint(position=(8, 24), direction=1)),
                               TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(8, 25), direction=1)),
                               TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(8, 26), direction=1)),
                               TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(8, 27), direction=1)),
                               TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(8, 28), direction=1)),
                               TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(8, 29), direction=1)),
                               TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(9, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(10, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=24, waypoint=Waypoint(position=(11, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=25, waypoint=Waypoint(position=(12, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=26, waypoint=Waypoint(position=(13, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=27, waypoint=Waypoint(position=(13, 28), direction=3)),
                               TrainrunWaypoint(scheduled_at=28, waypoint=Waypoint(position=(14, 28), direction=2)),
                               TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(15, 28), direction=2)),
                               TrainrunWaypoint(scheduled_at=30, waypoint=Waypoint(position=(16, 28), direction=2)),
                               TrainrunWaypoint(scheduled_at=31, waypoint=Waypoint(position=(17, 28), direction=2)),
                               TrainrunWaypoint(scheduled_at=32, waypoint=Waypoint(position=(17, 29), direction=1)),
                               TrainrunWaypoint(scheduled_at=33, waypoint=Waypoint(position=(18, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=34, waypoint=Waypoint(position=(19, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=35, waypoint=Waypoint(position=(20, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=36, waypoint=Waypoint(position=(21, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=37, waypoint=Waypoint(position=(22, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=38, waypoint=Waypoint(position=(23, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=39, waypoint=Waypoint(position=(24, 29), direction=2)),
                               TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(24, 28), direction=3)),
                               TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(24, 27), direction=3)),
                               TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(24, 26), direction=3)),
                               TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(24, 25), direction=3)),
                               TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(24, 24), direction=3)),
                               TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(24, 23), direction=3))],
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
                               TrainrunWaypoint(scheduled_at=35, waypoint=Waypoint(position=(15, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=36, waypoint=Waypoint(position=(14, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=37, waypoint=Waypoint(position=(13, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=38, waypoint=Waypoint(position=(12, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=39, waypoint=Waypoint(position=(11, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(10, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(9, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(8, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(7, 29), direction=0)),
                               TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(7, 28), direction=3)),
                               TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(7, 27), direction=3)),
                               TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(7, 26), direction=3)),
                               TrainrunWaypoint(scheduled_at=47, waypoint=Waypoint(position=(7, 25), direction=3)),
                               TrainrunWaypoint(scheduled_at=48, waypoint=Waypoint(position=(7, 24), direction=3)),
                               TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(7, 23), direction=3))]}

    fake_malfunction = ExperimentMalfunction(time_step=14, agent_id=1, malfunction_duration=20)
    k = 10
    agents_paths_dict = {
        i: get_k_shortest_paths(static_env,
                                agent.initial_position,
                                agent.initial_direction,
                                agent.target,
                                k) for i, agent in enumerate(static_env.agents)
    }
    # we derive the re-schedule problem from the schedule problem
    schedule_problem = ASPProblemDescription(env=static_env,
                                             agents_path_dict=agents_paths_dict)

    freeze_dict = get_freeze_for_malfunction(malfunction=fake_malfunction,
                                             schedule_trainruns=fake_schedule,
                                             static_rail_env=static_env,
                                             agents_path_dict=agents_paths_dict)

    for agent_id, experiment_freeze in freeze_dict.items():
        verify_experiment_freeze_for_agent(freeze_dict[agent_id], agents_paths_dict[agent_id])

    schedule_problem.get_copy_for_experiment_freeze(
        freeze=freeze_dict,
        schedule_trainruns=fake_schedule
    )
    global_nr_malfunctions = 0
    malfunction_calls = dict()

    def generator(agent: EnvAgent = None, np_random: RandomState = None, reset=False) \
            -> Optional[ExperimentMalfunction]:
        # We use the global variable to assure only a single malfunction in the env
        nonlocal global_nr_malfunctions
        nonlocal malfunction_calls

        # Reset malfunciton generator
        if reset:
            nonlocal global_nr_malfunctions
            nonlocal malfunction_calls
            global_nr_malfunctions = 0
            malfunction_calls = dict()
            return FLMalfunction(0)

        # Update number of calls per agent
        if agent.handle in malfunction_calls:
            malfunction_calls[agent.handle] += 1
        else:
            malfunction_calls[agent.handle] = 1

        # Break an agent that is active at the time of the malfunction
        # N.B. we have just incremented the number of malfunction calls!
        if agent.handle == fake_malfunction.agent_id and \
                malfunction_calls[agent.handle] - 1 == fake_malfunction.time_step:
            global_nr_malfunctions += 1
            return FLMalfunction(20)
        else:
            return FLMalfunction(0)

    dynamic_env.malfunction_generator, dynamic_env.malfunction_process_data = generator, MalfunctionProcessData(0, 0, 0)

    full_reschedule_result = reschedule_full_after_malfunction(
        malfunction=fake_malfunction,
        malfunction_env_reset=lambda *args, **kwargs: None,
        malfunction_rail_env=dynamic_env,
        schedule_problem=schedule_problem,
        schedule_trainruns=fake_schedule,
        static_rail_env=static_env,
        # rendering=False,
        # debug=False,

    )
    full_reschedule_solution = full_reschedule_result.solution
    full_reschedule_trainruns: Dict[int, List[TrainrunWaypoint]] = full_reschedule_solution.get_trainruns_dict()

    # agent 0: scheduled arrival was 46, new arrival is 45 -> penalty = 0 (no negative delay!)
    # agent 1: scheduled arrival was 29, new arrival is 49 -> penalty = 20 = delay
    actual_costs = full_reschedule_solution.asp_solution.stats['summary']['costs'][0]
    assert actual_costs == 20, f"actual costs {actual_costs}"

    assert full_reschedule_trainruns == expected_reschedule

    for agent_id in expected_freeze.keys():
        assert set(full_reschedule_trainruns[agent_id]).issuperset(set(freeze[agent_id])), f"{agent_id}"

    delta, freeze = determine_delta(full_reschedule_trainruns, fake_malfunction,
                                    fake_schedule, verbose=True)
    assert delta == {
        0: [
            TrainrunWaypoint(scheduled_at=14, waypoint=Waypoint(position=(8, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=16, waypoint=Waypoint(position=(8, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(8, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=18, waypoint=Waypoint(position=(8, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(8, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(8, 28), direction=1)),
            TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(8, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(9, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(10, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=24, waypoint=Waypoint(position=(11, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=25, waypoint=Waypoint(position=(12, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=26, waypoint=Waypoint(position=(13, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=27, waypoint=Waypoint(position=(13, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=28, waypoint=Waypoint(position=(14, 28), direction=2)),
            TrainrunWaypoint(scheduled_at=29, waypoint=Waypoint(position=(15, 28), direction=2)),
            TrainrunWaypoint(scheduled_at=30, waypoint=Waypoint(position=(16, 28), direction=2)),
            TrainrunWaypoint(scheduled_at=31, waypoint=Waypoint(position=(17, 28), direction=2)),
            TrainrunWaypoint(scheduled_at=32, waypoint=Waypoint(position=(17, 29), direction=1)),
            TrainrunWaypoint(scheduled_at=33, waypoint=Waypoint(position=(18, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=34, waypoint=Waypoint(position=(19, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=35, waypoint=Waypoint(position=(20, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=36, waypoint=Waypoint(position=(21, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=37, waypoint=Waypoint(position=(22, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=38, waypoint=Waypoint(position=(23, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=39, waypoint=Waypoint(position=(24, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(24, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(24, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(24, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(24, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(24, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(24, 23), direction=3))],
        1: [
            TrainrunWaypoint(scheduled_at=35, waypoint=Waypoint(position=(15, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=36, waypoint=Waypoint(position=(14, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=37, waypoint=Waypoint(position=(13, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=38, waypoint=Waypoint(position=(12, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=39, waypoint=Waypoint(position=(11, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(10, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(9, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(8, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(7, 29), direction=0)),
            TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(7, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(7, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(7, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=47, waypoint=Waypoint(position=(7, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=48, waypoint=Waypoint(position=(7, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(7, 23), direction=3))]}
    assert freeze == {
        0: [],
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
            TrainrunWaypoint(scheduled_at=35, waypoint=Waypoint(position=(15, 29), direction=0))
            ]}


def test_delta_freeze():
    schedule_solution = {0: [TrainrunWaypoint(scheduled_at=0, waypoint=Waypoint(position=(8, 23), direction=1)),
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
                             TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(7, 23), direction=3))]}
    reschedule_solution = {0: [TrainrunWaypoint(scheduled_at=0, waypoint=Waypoint(position=(8, 23), direction=1)),
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
                               TrainrunWaypoint(scheduled_at=66, waypoint=Waypoint(position=(7, 23), direction=3))]}
    malfunction = ExperimentMalfunction(time_step=19, agent_id=0, malfunction_duration=20)
    delta, freeze = determine_delta(reschedule_solution, malfunction,
                                    schedule_solution, verbose=True)

    assert delta == {
        0: [
            TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(21, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=41, waypoint=Waypoint(position=(22, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=42, waypoint=Waypoint(position=(23, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=43, waypoint=Waypoint(position=(24, 29), direction=2)),
            TrainrunWaypoint(scheduled_at=44, waypoint=Waypoint(position=(24, 28), direction=3)),
            TrainrunWaypoint(scheduled_at=45, waypoint=Waypoint(position=(24, 27), direction=3)),
            TrainrunWaypoint(scheduled_at=46, waypoint=Waypoint(position=(24, 26), direction=3)),
            TrainrunWaypoint(scheduled_at=47, waypoint=Waypoint(position=(24, 25), direction=3)),
            TrainrunWaypoint(scheduled_at=48, waypoint=Waypoint(position=(24, 24), direction=3)),
            TrainrunWaypoint(scheduled_at=49, waypoint=Waypoint(position=(24, 23), direction=3))
        ],
        1: [
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
            TrainrunWaypoint(scheduled_at=66, waypoint=Waypoint(position=(7, 23), direction=3))
        ]
    }

    assert freeze == {
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
            TrainrunWaypoint(scheduled_at=40, waypoint=Waypoint(position=(21, 29), direction=2))
            ],
        1: [TrainrunWaypoint(scheduled_at=17, waypoint=Waypoint(position=(23, 23), direction=1)),
            TrainrunWaypoint(scheduled_at=19, waypoint=Waypoint(position=(23, 24), direction=1)),
            TrainrunWaypoint(scheduled_at=20, waypoint=Waypoint(position=(23, 25), direction=1)),
            TrainrunWaypoint(scheduled_at=21, waypoint=Waypoint(position=(23, 26), direction=1)),
            TrainrunWaypoint(scheduled_at=22, waypoint=Waypoint(position=(23, 27), direction=1)),
            TrainrunWaypoint(scheduled_at=23, waypoint=Waypoint(position=(23, 28), direction=1))]
    }
