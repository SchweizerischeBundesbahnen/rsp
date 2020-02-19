from typing import Dict
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import Optional

import networkx as nx
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

ExperimentFreeze = NamedTuple('ExperimentFreeze', [
    ('freeze_visit', List[TrainrunWaypoint]),
    ('freeze_earliest', Dict[Waypoint, int]),
    ('freeze_latest', Dict[Waypoint, int]),
    ('freeze_banned', List[Waypoint])
])
ExperimentFreezeDict = Dict[int, ExperimentFreeze]

AgentPaths = List[List[Waypoint]]
AgentsPathsDict = Dict[int, AgentPaths]
MAGIC_DIRECTION_FOR_SOURCE_TARGET = 5


def experiment_freeze_dict_from_list_of_train_run_waypoint(l: List[TrainrunWaypoint]) -> Dict[TrainrunWaypoint, int]:
    """Generate dictionary of scheduled time at waypoint.

    Parameters
    ----------
    l train run waypoints

    Returns
    -------
    """
    return {trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at for trainrun_waypoint in l}


# TODO SIM-239 necessary/ScheduleProblemDescription?
RouteDag = NamedTuple('RouteDag', [
    ('topo', nx.DiGraph),
    ('constraints', ExperimentFreeze),
])

RouteDagDict = Dict[int, RouteDag]
TopoDict = Dict[int, nx.DiGraph]


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


def route_dag_from_agent_paths_and_freeze(agent_paths: AgentPaths, f: ExperimentFreeze) -> RouteDag:
    """Extract  the agent's route DAG.

    Parameters
    ----------
    agent_paths: AgentPaths
    f: ExperimentFreeze

    Returns
    -------
    RouteDag
        the route dag (topology and constraints)
    """

    return RouteDag(topo_from_agent_paths(agent_paths=agent_paths), f)


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


def get_paths_for_experiment_freeze(
        topo: nx.DiGraph,
        experiment_freeze: Optional[ExperimentFreeze] = None) -> List[List[Waypoint]]:
    """Determine the routes through the route graph given the constraints.

    Parameters
    ----------
    agent_paths
    experiment_freeze

    Returns
    -------
    """
    if experiment_freeze:
        for wp in experiment_freeze.freeze_banned:
            if wp in topo.nodes:
                topo.remove_node(wp)
    paths = _paths_in_route_dag(topo)
    return paths
