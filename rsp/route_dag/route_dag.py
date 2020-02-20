"""Route DAG data structures and utils."""
from typing import Dict
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import networkx as nx
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

MAGIC_DIRECTION_FOR_SOURCE_TARGET = 5

TopoDict = Dict[int, nx.DiGraph]
AgentPaths = List[List[Waypoint]]
AgentsPathsDict = Dict[int, AgentPaths]

RouteDAGConstraints = NamedTuple('RouteDAGConstraints', [
    ('freeze_visit', List[TrainrunWaypoint]),
    ('freeze_earliest', Dict[Waypoint, int]),
    ('freeze_latest', Dict[Waypoint, int]),
    ('freeze_banned', List[Waypoint])
])

RouteDAGConstraintsDict = Dict[int, RouteDAGConstraints]
RouteDagEdge = Tuple[Waypoint, Waypoint]
RouteSectionPenaltyDict = Dict[RouteDagEdge, int]
ScheduleProblemDescription = NamedTuple('ScheduleProblemDescription', [
    ('route_dag_constraints_dict', RouteDAGConstraintsDict),
    ('minimum_travel_time_dict', Dict[int, int]),
    ('topo_dict', Dict[int, nx.DiGraph]),
    ('max_episode_steps', int),
    ('route_section_penalties', Dict[int, RouteSectionPenaltyDict])
])


def route_dag_constraints_dict_from_list_of_train_run_waypoint(
        l: List[TrainrunWaypoint]
) -> Dict[TrainrunWaypoint, int]:
    """Generate dictionary of scheduled time at waypoint.

    Parameters
    ----------
    l train run waypoints

    Returns
    -------
    """
    return {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at for trainrun_waypoint in l}


def _paths_in_route_dag(topo: nx.DiGraph) -> List[List[Waypoint]]:
    """Get the paths of all source nodes (no incoming edges) to all sink nodes
    (no outgoing edges).

    Parameters
    ----------
    topo: DiGraph

    Returns
    -------
    List[List[Waypoint]]
    """
    sources = get_sources_for_topo(topo)
    sinks = get_sinks_for_topo(topo)
    all_paths = []
    for source in sources:
        for sink in sinks:
            source_sink_paths = list(nx.all_simple_paths(topo, source, sink))
            all_paths += source_sink_paths
    return all_paths


def get_sinks_for_topo(topo: nx.DiGraph) -> Iterator[Waypoint]:
    sinks = (node for node, out_degree in topo.out_degree if out_degree == 0)
    return sinks


def get_sources_for_topo(topo: nx.DiGraph) -> Iterator[Waypoint]:
    sources = (node for node, in_degree in topo.in_degree if in_degree == 0)
    return sources


def topo_from_agent_paths(agent_paths: AgentPaths) -> nx.DiGraph:
    """Extract  the agent's topology.

    Parameters
    ----------
    agent_paths: AgentPaths

    Returns
    -------
    nx.DiGraph
        topology
    """
    topo = nx.DiGraph()
    for path in agent_paths:
        for wp1, wp2 in zip(path, path[1:]):
            topo.add_edge(wp1, wp2)
    return topo


def get_paths_for_route_dag_constraints(
        topo: nx.DiGraph,
        route_dag_constraints: Optional[RouteDAGConstraints] = None) -> List[List[Waypoint]]:
    """Determine the routes through the route graph given the constraints.

    Parameters
    ----------
    agent_paths
    route_dag_constraints

    Returns
    -------
    """
    if route_dag_constraints:
        for wp in route_dag_constraints.freeze_banned:
            if wp in topo.nodes:
                topo.remove_node(wp)
    paths = _paths_in_route_dag(topo)
    return paths


def _get_topology_with_dummy_nodes_from_agent_paths_dict(agents_paths_dict: AgentsPathsDict):
    # get topology from agent paths
    topo_dict = {agent_id: topo_from_agent_paths(agents_paths_dict[agent_id])
                 for agent_id in agents_paths_dict}
    # add dummy nodes
    dummy_source_dict: Dict[int, Waypoint] = {}
    dummy_sink_dict: Dict[int, Waypoint] = {}
    for agent_id, topo in topo_dict.items():
        sources = list(get_sources_for_topo(topo))
        sinks = list(get_sinks_for_topo(topo))

        dummy_sink_waypoint = Waypoint(position=agents_paths_dict[agent_id][0][-1].position,
                                       direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET)
        dummy_sink_dict[agent_id] = dummy_sink_waypoint
        dummy_source_waypoint = Waypoint(position=agents_paths_dict[agent_id][0][0].position,
                                         direction=MAGIC_DIRECTION_FOR_SOURCE_TARGET)
        dummy_source_dict[agent_id] = dummy_source_waypoint
        for source in sources:
            topo.add_edge(dummy_source_waypoint, source)
        for sink in sinks:
            topo.add_edge(sink, dummy_sink_waypoint)
    return dummy_source_dict, topo_dict
