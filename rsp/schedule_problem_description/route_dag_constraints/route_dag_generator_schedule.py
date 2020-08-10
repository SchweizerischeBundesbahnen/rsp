import math

import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.schedule_problem_description.data_types_and_utils import get_paths_in_route_dag
from rsp.schedule_problem_description.data_types_and_utils import get_sinks_for_topo
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_utils import propagate_earliest
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_utils import propagate_latest
from rsp.utils.data_types import RouteDAGConstraints


def _get_route_dag_constraints_for_scheduling(
        topo: nx.DiGraph,
        source_waypoint: Waypoint,
        minimum_travel_time: int,
        latest_arrival: int
) -> RouteDAGConstraints:
    baender = False

    earliest_departure = 0
    max_episode_steps = latest_arrival
    steps = max([len(p) for p in get_paths_in_route_dag(topo)])
    minimum_journey_time = steps * minimum_travel_time

    if baender:
        earliest_departure = math.floor(np.random.random() * (latest_arrival - minimum_journey_time) / 2)
        latest_arrival = min(earliest_departure + minimum_journey_time + 400, max_episode_steps)
        assert earliest_departure + minimum_journey_time < latest_arrival

    print(
        f"earliest departure {earliest_departure}, "
        f"latest arrival {latest_arrival}, "
        f"max_episode_steps {max_episode_steps}, "
        f"steps={steps}, mrt {minimum_travel_time}, "
        f"mjt {minimum_journey_time}, "
        f"earliest arrival {earliest_departure + minimum_journey_time}")

    return RouteDAGConstraints(
        freeze_visit=[],
        freeze_earliest=propagate_earliest(
            banned_set=set(),
            earliest_dict={source_waypoint: earliest_departure},
            minimum_travel_time=minimum_travel_time,
            force_freeze_dict={},
            subdag_source=TrainrunWaypoint(waypoint=source_waypoint, scheduled_at=earliest_departure),
            topo=topo,
        ),
        freeze_latest=propagate_latest(
            banned_set=set(),
            latest_dict={sink: latest_arrival - 1 for sink in get_sinks_for_topo(topo)},
            earliest_dict={},
            latest_arrival=latest_arrival,
            minimum_travel_time=minimum_travel_time,
            force_freeze_dict={},
            topo=topo,
        ),
        freeze_banned=[],
    )
