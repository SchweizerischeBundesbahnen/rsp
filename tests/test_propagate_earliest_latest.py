import numpy as np
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.route_dag.generators.route_dag_generator_utils import propagate_earliest
from rsp.route_dag.generators.route_dag_generator_utils import propagate_latest
from rsp.route_dag.generators.route_dag_generator_utils import propagate_latest_constant
from rsp.route_dag.route_dag import _get_topology_with_dummy_nodes_from_agent_paths_dict
from rsp.route_dag.route_dag import get_sinks_for_topo
from rsp.utils.experiment_env_generators import create_flatland_environment


def _get_test_env():
    env = create_flatland_environment(number_of_agents=1,
                                      width=30,
                                      height=30,
                                      seed_value=12,
                                      max_num_cities=20,
                                      grid_mode=True,
                                      max_rails_between_cities=2,
                                      max_rails_in_city=6,
                                      speed_data={1: 1.0})
    env.reset(random_seed=12)

    k_shortest_paths = get_k_shortest_paths(env,
                                            env.agents[0].initial_position,
                                            env.agents[0].initial_direction,
                                            env.agents[0].target,
                                            10)
    dummy_source_dict, topo_dict = _get_topology_with_dummy_nodes_from_agent_paths_dict({0: k_shortest_paths})
    minimum_travel_time = int(np.ceil(1 / env.agents[0].speed_data['speed']))
    latest_arrival = env._max_episode_steps

    return dummy_source_dict, topo_dict, minimum_travel_time, latest_arrival


def test_scheduling_propagate_earliest():
    earliest_truth = {Waypoint(position=(8, 23), direction=5): 0,
                      Waypoint(position=(8, 23), direction=1): 1,
                      Waypoint(position=(8, 24), direction=1): 2,
                      Waypoint(position=(8, 25), direction=1): 3,
                      Waypoint(position=(8, 26), direction=1): 4,
                      Waypoint(position=(8, 27), direction=1): 5,
                      Waypoint(position=(8, 28), direction=1): 6,
                      Waypoint(position=(7, 27), direction=0): 6,
                      Waypoint(position=(8, 29), direction=1): 7,
                      Waypoint(position=(9, 29), direction=2): 8,
                      Waypoint(position=(10, 29), direction=2): 9,
                      Waypoint(position=(11, 29), direction=2): 10,
                      Waypoint(position=(12, 29), direction=2): 11,
                      Waypoint(position=(13, 29), direction=2): 12,
                      Waypoint(position=(14, 29), direction=2): 13,
                      Waypoint(position=(13, 28), direction=3): 13,
                      Waypoint(position=(15, 29), direction=2): 14,
                      Waypoint(position=(16, 29), direction=2): 15,
                      Waypoint(position=(17, 29), direction=2): 16,
                      Waypoint(position=(18, 29), direction=2): 17,
                      Waypoint(position=(19, 29), direction=2): 18,
                      Waypoint(position=(20, 29), direction=2): 19,
                      Waypoint(position=(21, 29), direction=2): 20,
                      Waypoint(position=(22, 29), direction=2): 21,
                      Waypoint(position=(23, 29), direction=2): 22,
                      Waypoint(position=(24, 29), direction=2): 23,
                      Waypoint(position=(23, 28), direction=3): 23,
                      Waypoint(position=(24, 28), direction=3): 24,
                      Waypoint(position=(24, 27), direction=3): 25,
                      Waypoint(position=(24, 26), direction=3): 26,
                      Waypoint(position=(24, 25), direction=3): 27,
                      Waypoint(position=(24, 24), direction=3): 28,
                      Waypoint(position=(24, 23), direction=3): 29,
                      Waypoint(position=(24, 23), direction=5): 30,
                      Waypoint(position=(23, 27), direction=3): 24,
                      Waypoint(position=(24, 27), direction=2): 25,
                      Waypoint(position=(7, 28), direction=1): 7,
                      Waypoint(position=(7, 29), direction=1): 8,
                      Waypoint(position=(8, 29), direction=2): 9,
                      Waypoint(position=(14, 28), direction=2): 14,
                      Waypoint(position=(15, 28), direction=2): 15,
                      Waypoint(position=(16, 28), direction=2): 16,
                      Waypoint(position=(17, 28), direction=2): 17,
                      Waypoint(position=(17, 29), direction=1): 18}

    dummy_source_dict, topo_dict, minimum_travel_time, latest_arrival = _get_test_env()

    earliest = propagate_earliest(
        banned_set=set(),
        earliest_dict={dummy_source_dict[0]: 0},
        minimum_travel_time=minimum_travel_time,
        force_freeze_dict={},
        subdag_source=TrainrunWaypoint(waypoint=dummy_source_dict[0], scheduled_at=0),
        topo=topo_dict[0],
    )

    for waypoint, earliest_time in earliest.items():
        assert earliest_truth.get(waypoint) == earliest_time


def test_scheduling_propagate_latest():
    latest_truth = {Waypoint(position=(24, 23), direction=5): 481,
                    Waypoint(position=(24, 23), direction=3): 480,
                    Waypoint(position=(24, 24), direction=3): 479,
                    Waypoint(position=(24, 25), direction=3): 478,
                    Waypoint(position=(24, 26), direction=3): 477,
                    Waypoint(position=(24, 27), direction=3): 476,
                    Waypoint(position=(24, 27), direction=2): 476,
                    Waypoint(position=(23, 27), direction=3): 475,
                    Waypoint(position=(24, 28), direction=3): 475,
                    Waypoint(position=(23, 28), direction=3): 474,
                    Waypoint(position=(24, 29), direction=2): 474,
                    Waypoint(position=(23, 29), direction=2): 473,
                    Waypoint(position=(22, 29), direction=2): 472,
                    Waypoint(position=(21, 29), direction=2): 471,
                    Waypoint(position=(20, 29), direction=2): 470,
                    Waypoint(position=(19, 29), direction=2): 469,
                    Waypoint(position=(18, 29), direction=2): 468,
                    Waypoint(position=(17, 29), direction=2): 467,
                    Waypoint(position=(17, 29), direction=1): 467,
                    Waypoint(position=(17, 28), direction=2): 466,
                    Waypoint(position=(16, 29), direction=2): 466,
                    Waypoint(position=(16, 28), direction=2): 465,
                    Waypoint(position=(15, 29), direction=2): 465,
                    Waypoint(position=(15, 28), direction=2): 464,
                    Waypoint(position=(14, 29), direction=2): 464,
                    Waypoint(position=(14, 28), direction=2): 463,
                    Waypoint(position=(13, 29), direction=2): 463,
                    Waypoint(position=(13, 28), direction=3): 462,
                    Waypoint(position=(12, 29), direction=2): 462,
                    Waypoint(position=(11, 29), direction=2): 461,
                    Waypoint(position=(10, 29), direction=2): 460,
                    Waypoint(position=(9, 29), direction=2): 459,
                    Waypoint(position=(8, 29), direction=1): 458,
                    Waypoint(position=(8, 29), direction=2): 458,
                    Waypoint(position=(7, 29), direction=1): 457,
                    Waypoint(position=(8, 28), direction=1): 457,
                    Waypoint(position=(7, 28), direction=1): 456,
                    Waypoint(position=(8, 27), direction=1): 456,
                    Waypoint(position=(7, 27), direction=0): 455,
                    Waypoint(position=(8, 26), direction=1): 455,
                    Waypoint(position=(8, 25), direction=1): 454,
                    Waypoint(position=(8, 24), direction=1): 453,
                    Waypoint(position=(8, 23), direction=1): 452,
                    Waypoint(position=(8, 23), direction=5): 451}

    dummy_source_dict, topo_dict, minimum_travel_time, latest_arrival = _get_test_env()

    latest = propagate_latest(
        banned_set=set(),
        latest_dict={sink: latest_arrival - 1 for sink in get_sinks_for_topo(topo_dict[0])},
        latest_arrival=latest_arrival,
        minimum_travel_time=minimum_travel_time,
        force_freeze_dict={},
        topo=topo_dict[0],
    )

    for waypoint, earliest_time in latest.items():
        assert latest_truth.get(waypoint) == earliest_time


def test_scheduling_propagate_latest_constant():
    dummy_source_dict, topo_dict, minimum_travel_time, latest_arrival = _get_test_env()
    dummy_source = dummy_source_dict[0]

    freeze_earliest = propagate_earliest(
        banned_set=set(),
        earliest_dict={dummy_source: 0},
        minimum_travel_time=minimum_travel_time,
        force_freeze_dict={},
        subdag_source=TrainrunWaypoint(waypoint=dummy_source, scheduled_at=0),
        topo=topo_dict[0],
    )

    latest_constant = 180
    freeze_latest = propagate_latest_constant(
        latest_dict={},
        earliest_dict=freeze_earliest,
        latest_constant=latest_constant,
        latest_arrival=latest_arrival
    )

    for waypoint, earliest_time in freeze_earliest.items():
        latest_time = freeze_latest.get(waypoint)
        assert latest_time == earliest_time + latest_constant
