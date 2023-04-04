import numpy as np
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.rspflatland.experiment_env_generators import create_flatland_environment
from rsp.scheduling.propagate import _propagate_earliest
from rsp.scheduling.propagate import propagate
from rsp.scheduling.scheduling_problem import _get_topology_from_agents_path_dict
from rsp.scheduling.scheduling_problem import get_sinks_for_topo
from rsp.scheduling.scheduling_problem import get_sources_for_topo


def _get_test_env():
    env = create_flatland_environment(
        number_of_agents=1,
        width=30,
        height=30,
        flatland_seed_value=12,
        max_num_cities=20,
        grid_mode=True,
        max_rails_between_cities=2,
        max_rails_in_city=6,
        speed_data={1: 1.0},
    )
    env.reset(random_seed=12)

    k_shortest_paths = get_k_shortest_paths(env, env.agents[0].initial_position, env.agents[0].initial_direction, env.agents[0].target, 10)
    topo_dict = _get_topology_from_agents_path_dict({0: k_shortest_paths})
    minimum_travel_time = int(np.ceil(1 / env.agents[0].speed_data["speed"]))
    latest_arrival = env._max_episode_steps

    return topo_dict, minimum_travel_time, latest_arrival


def test_scheduling_propagate_earliest():
    earliest_truth = {
        Waypoint(position=(8, 23), direction=1): 1 - 1,
        Waypoint(position=(8, 24), direction=1): 2 - 1,
        Waypoint(position=(8, 25), direction=1): 3 - 1,
        Waypoint(position=(8, 26), direction=1): 4 - 1,
        Waypoint(position=(8, 27), direction=1): 5 - 1,
        Waypoint(position=(8, 28), direction=1): 6 - 1,
        Waypoint(position=(7, 27), direction=0): 6 - 1,
        Waypoint(position=(8, 29), direction=1): 7 - 1,
        Waypoint(position=(9, 29), direction=2): 8 - 1,
        Waypoint(position=(10, 29), direction=2): 9 - 1,
        Waypoint(position=(11, 29), direction=2): 10 - 1,
        Waypoint(position=(12, 29), direction=2): 11 - 1,
        Waypoint(position=(13, 29), direction=2): 12 - 1,
        Waypoint(position=(14, 29), direction=2): 13 - 1,
        Waypoint(position=(13, 28), direction=3): 13 - 1,
        Waypoint(position=(15, 29), direction=2): 14 - 1,
        Waypoint(position=(16, 29), direction=2): 15 - 1,
        Waypoint(position=(17, 29), direction=2): 16 - 1,
        Waypoint(position=(18, 29), direction=2): 17 - 1,
        Waypoint(position=(19, 29), direction=2): 18 - 1,
        Waypoint(position=(20, 29), direction=2): 19 - 1,
        Waypoint(position=(21, 29), direction=2): 20 - 1,
        Waypoint(position=(22, 29), direction=2): 21 - 1,
        Waypoint(position=(23, 29), direction=2): 22 - 1,
        Waypoint(position=(24, 29), direction=2): 23 - 1,
        Waypoint(position=(23, 28), direction=3): 23 - 1,
        Waypoint(position=(24, 28), direction=3): 24 - 1,
        Waypoint(position=(24, 27), direction=3): 25 - 1,
        Waypoint(position=(24, 26), direction=3): 26 - 1,
        Waypoint(position=(24, 25), direction=3): 27 - 1,
        Waypoint(position=(24, 24), direction=3): 28 - 1,
        Waypoint(position=(24, 23), direction=3): 29 - 1,
        Waypoint(position=(24, 23), direction=5): 30 - 1,
        Waypoint(position=(23, 27), direction=3): 24 - 1,
        Waypoint(position=(24, 27), direction=2): 25 - 1,
        Waypoint(position=(7, 28), direction=1): 7 - 1,
        Waypoint(position=(7, 29), direction=1): 8 - 1,
        Waypoint(position=(8, 29), direction=2): 9 - 1,
        Waypoint(position=(14, 28), direction=2): 14 - 1,
        Waypoint(position=(15, 28), direction=2): 15 - 1,
        Waypoint(position=(16, 28), direction=2): 16 - 1,
        Waypoint(position=(17, 28), direction=2): 17 - 1,
        Waypoint(position=(17, 29), direction=1): 18 - 1,
    }

    topo_dict, minimum_travel_time, latest_arrival = _get_test_env()
    source_waypoint = next(get_sources_for_topo(topo_dict[0]))

    earliest = _propagate_earliest(
        earliest_dict={source_waypoint: 0}, minimum_travel_time=minimum_travel_time, force_earliest={source_waypoint}, topo=topo_dict[0],
    )

    for waypoint, earliest_time in earliest.items():
        assert earliest_truth.get(waypoint) == earliest_time


def test_scheduling_propagate_latest_backwards():
    latest_truth = {
        Waypoint(position=(24, 23), direction=3): 480 + 1,
        Waypoint(position=(24, 24), direction=3): 479 + 1,
        Waypoint(position=(24, 25), direction=3): 478 + 1,
        Waypoint(position=(24, 26), direction=3): 477 + 1,
        Waypoint(position=(24, 27), direction=3): 476 + 1,
        Waypoint(position=(24, 27), direction=2): 476 + 1,
        Waypoint(position=(23, 27), direction=3): 475 + 1,
        Waypoint(position=(24, 28), direction=3): 475 + 1,
        Waypoint(position=(23, 28), direction=3): 474 + 1,
        Waypoint(position=(24, 29), direction=2): 474 + 1,
        Waypoint(position=(23, 29), direction=2): 473 + 1,
        Waypoint(position=(22, 29), direction=2): 472 + 1,
        Waypoint(position=(21, 29), direction=2): 471 + 1,
        Waypoint(position=(20, 29), direction=2): 470 + 1,
        Waypoint(position=(19, 29), direction=2): 469 + 1,
        Waypoint(position=(18, 29), direction=2): 468 + 1,
        Waypoint(position=(17, 29), direction=2): 467 + 1,
        Waypoint(position=(17, 29), direction=1): 467 + 1,
        Waypoint(position=(17, 28), direction=2): 466 + 1,
        Waypoint(position=(16, 29), direction=2): 466 + 1,
        Waypoint(position=(16, 28), direction=2): 465 + 1,
        Waypoint(position=(15, 29), direction=2): 465 + 1,
        Waypoint(position=(15, 28), direction=2): 464 + 1,
        Waypoint(position=(14, 29), direction=2): 464 + 1,
        Waypoint(position=(14, 28), direction=2): 463 + 1,
        Waypoint(position=(13, 29), direction=2): 463 + 1,
        Waypoint(position=(13, 28), direction=3): 462 + 1,
        Waypoint(position=(12, 29), direction=2): 462 + 1,
        Waypoint(position=(11, 29), direction=2): 461 + 1,
        Waypoint(position=(10, 29), direction=2): 460 + 1,
        Waypoint(position=(9, 29), direction=2): 459 + 1,
        Waypoint(position=(8, 29), direction=1): 458 + 1,
        Waypoint(position=(8, 29), direction=2): 458 + 1,
        Waypoint(position=(7, 29), direction=1): 457 + 1,
        Waypoint(position=(8, 28), direction=1): 457 + 1,
        Waypoint(position=(7, 28), direction=1): 456 + 1,
        Waypoint(position=(8, 27), direction=1): 456 + 1,
        Waypoint(position=(7, 27), direction=0): 455 + 1,
        Waypoint(position=(8, 26), direction=1): 455 + 1,
        Waypoint(position=(8, 25), direction=1): 454 + 1,
        Waypoint(position=(8, 24), direction=1): 453 + 1,
        Waypoint(position=(8, 23), direction=1): 452 + 1,
    }

    topo_dict, minimum_travel_time, latest_arrival = _get_test_env()

    latest = {sink: latest_arrival - 1 for sink in get_sinks_for_topo(topo_dict[0])}
    propagate(
        earliest_dict={},
        latest_dict=latest,
        force_earliest=set(),
        force_latest=set(get_sinks_for_topo(topo_dict[0])),
        latest_arrival=latest_arrival,
        max_window_size_from_earliest=np.inf,
        minimum_travel_time=minimum_travel_time,
        topo=topo_dict[0],
        must_be_visited=set(),
    )

    for waypoint, earliest_time in latest.items():
        assert latest_truth.get(waypoint) == earliest_time


def test_scheduling_propagate_latest_forward():
    topo_dict, minimum_travel_time, latest_arrival = _get_test_env()
    source_waypoint = next(get_sources_for_topo(topo_dict[0]))

    max_window_size_from_earliest = 180
    earliest = {source_waypoint: 0}
    latest = {sink: latest_arrival - 1 for sink in get_sinks_for_topo(topo_dict[0])}
    propagate(
        earliest_dict=earliest,
        latest_dict=latest,
        force_earliest={source_waypoint},
        force_latest=set(get_sinks_for_topo(topo_dict[0])),
        latest_arrival=latest_arrival,
        max_window_size_from_earliest=max_window_size_from_earliest,
        minimum_travel_time=minimum_travel_time,
        topo=topo_dict[0],
        must_be_visited=set(),
    )

    for waypoint, earliest_time in earliest.items():
        latest_time = latest.get(waypoint)
        assert latest_time == earliest_time + max_window_size_from_earliest


def test_scheduling_propagate_latest_forward_backward_min():
    latest_truth = {
        Waypoint(position=(24, 23), direction=3): 480 + 1,
        Waypoint(position=(24, 24), direction=3): 479 + 1,
        Waypoint(position=(24, 25), direction=3): 478 + 1,
        Waypoint(position=(24, 26), direction=3): 477 + 1,
        Waypoint(position=(24, 27), direction=3): 476 + 1,
        Waypoint(position=(24, 27), direction=2): 476 + 1,
        Waypoint(position=(23, 27), direction=3): 475 + 1,
        Waypoint(position=(24, 28), direction=3): 475 + 1,
        Waypoint(position=(23, 28), direction=3): 474 + 1,
        Waypoint(position=(24, 29), direction=2): 474 + 1,
        Waypoint(position=(23, 29), direction=2): 473 + 1,
        Waypoint(position=(22, 29), direction=2): 472 + 1,
        Waypoint(position=(21, 29), direction=2): 471 + 1,
        Waypoint(position=(20, 29), direction=2): 470 + 1,
        Waypoint(position=(19, 29), direction=2): 469 + 1,
        Waypoint(position=(18, 29), direction=2): 468 + 1,
        Waypoint(position=(17, 29), direction=2): 467 + 1,
        Waypoint(position=(17, 29), direction=1): 467 + 1,
        Waypoint(position=(17, 28), direction=2): 466 + 1,
        Waypoint(position=(16, 29), direction=2): 466 + 1,
        Waypoint(position=(16, 28), direction=2): 465 + 1,
        Waypoint(position=(15, 29), direction=2): 465 + 1,
        Waypoint(position=(15, 28), direction=2): 464 + 1,
        Waypoint(position=(14, 29), direction=2): 464 + 1,
        Waypoint(position=(14, 28), direction=2): 463 + 1,
        Waypoint(position=(13, 29), direction=2): 463 + 1,
        Waypoint(position=(13, 28), direction=3): 462 + 1,
        Waypoint(position=(12, 29), direction=2): 462 + 1,
        Waypoint(position=(11, 29), direction=2): 461 + 1,
        Waypoint(position=(10, 29), direction=2): 460 + 1,
        Waypoint(position=(9, 29), direction=2): 459 + 1,
        Waypoint(position=(8, 29), direction=1): 458 + 1,
        Waypoint(position=(8, 29), direction=2): 458 + 1,
        Waypoint(position=(7, 29), direction=1): 457 + 1,
        Waypoint(position=(8, 28), direction=1): 457 + 1,
        Waypoint(position=(7, 28), direction=1): 456 + 1,
        Waypoint(position=(8, 27), direction=1): 456 + 1,
        Waypoint(position=(7, 27), direction=0): 455 + 1,
        Waypoint(position=(8, 26), direction=1): 455 + 1,
        Waypoint(position=(8, 25), direction=1): 454 + 1,
        Waypoint(position=(8, 24), direction=1): 453 + 1,
        Waypoint(position=(8, 23), direction=1): 452 + 1,
        Waypoint(position=(8, 23), direction=5): 451 + 1,
    }

    topo_dict, minimum_travel_time, latest_arrival = _get_test_env()
    source_waypoint = next(get_sources_for_topo(topo_dict[0]))

    earliest = {source_waypoint: 0}

    max_window_size_from_earliest = 600
    latest = {sink: latest_arrival - 1 for sink in get_sinks_for_topo(topo_dict[0])}
    propagate(
        earliest_dict=earliest,
        latest_dict=latest,
        force_earliest={source_waypoint},
        force_latest=set(get_sinks_for_topo(topo_dict[0])),
        latest_arrival=latest_arrival,
        max_window_size_from_earliest=max_window_size_from_earliest,
        minimum_travel_time=minimum_travel_time,
        topo=topo_dict[0],
        must_be_visited=set(),
    )

    for waypoint, latest_time in latest.items():
        assert latest_truth.get(waypoint) == latest_time
