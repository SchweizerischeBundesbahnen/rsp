import networkx as nx
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.schedule_problem_description.data_types_and_utils import get_sinks_for_topo
from rsp.schedule_problem_description.route_dag_constraints.route_dag_generator_utils import propagate
from rsp.utils.data_types import RouteDAGConstraints


def _get_route_dag_constraints_for_scheduling(
        topo: nx.DiGraph,
        source_waypoint: Waypoint,
        minimum_travel_time: int,
        latest_arrival: int
) -> RouteDAGConstraints:
    freeze_earliest = {source_waypoint: 0}
    freeze_latest = {sink: latest_arrival - 1 for sink in get_sinks_for_topo(topo)}
    propagate(
        latest_dict=freeze_latest,
        earliest_dict=freeze_earliest,
        latest_arrival=latest_arrival,
        minimum_travel_time=minimum_travel_time,
        force_freeze_earliest={source_waypoint},
        force_freeze_latest=set(get_sinks_for_topo(topo)),
        must_be_visited=set(),
        topo=topo,
    )
    return RouteDAGConstraints(
        freeze_visit=[],
        freeze_earliest=freeze_earliest,
        freeze_latest=freeze_latest,
        freeze_banned=[],
    )
