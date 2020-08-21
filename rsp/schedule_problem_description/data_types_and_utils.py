"""Route DAG data structures and utils."""
from enum import Enum
from typing import Dict
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import networkx as nx
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.logger import rsp_logger

TopoDict = Dict[int, nx.DiGraph]
AgentPaths = List[List[Waypoint]]
AgentsPathsDict = Dict[int, AgentPaths]

RouteDAGConstraints = NamedTuple('RouteDAGConstraints', [
    ('earliest', Dict[Waypoint, int]),
    ('latest', Dict[Waypoint, int])
])


RouteDAGConstraintsDict = Dict[int, RouteDAGConstraints]
RouteDagEdge = Tuple[Waypoint, Waypoint]
RouteSectionPenalties = Dict[RouteDagEdge, int]
WaypointPenalties = Dict[Waypoint, int]
RouteSectionPenaltiesDict = Dict[int, RouteSectionPenalties]
ScheduleProblemDescription = NamedTuple('ScheduleProblemDescription', [
    ('route_dag_constraints_dict', RouteDAGConstraintsDict),
    ('minimum_travel_time_dict', Dict[int, int]),
    ('topo_dict', Dict[int, nx.DiGraph]),
    ('max_episode_steps', int),
    ('route_section_penalties', RouteSectionPenaltiesDict),
    ('weight_lateness_seconds', int),
])


class ScheduleProblemEnum(Enum):
    PROBLEM_SCHEDULE = "PROBLEM_SCHEDULE"
    PROBLEM_RSP_FULL = "PROBLEM_RSP_FULL"
    PROBLEM_RSP_DELTA = "PROBLEM_RSP_DELTA"


def schedule_problem_description_equals(s1: ScheduleProblemDescription, s2: ScheduleProblemDescription):
    """Tests whether two schedule_problem_descriptions are the equal.

    We cannot test by == since we have `nx.DiGraph` objects as values.
    """
    for index, slot in enumerate(s1._fields):
        if slot == 'topo_dict':
            if s1.topo_dict.keys() != s2.topo_dict.keys():
                return False
            for agent_id in s1.topo_dict:
                s1_topo = s1.topo_dict[agent_id]
                s2_topo = s2.topo_dict[agent_id]
                if set(s1_topo.nodes) != set(s2_topo.nodes):
                    return False
                if set(s1_topo.edges) != set(s2_topo.edges):
                    return False
        elif s1[index] != s2[index]:
            return False
    return True


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


def get_paths_in_route_dag(topo: nx.DiGraph) -> List[List[Waypoint]]:
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
    """Extract  the agent's topology. Skip agent paths that make the graph
    acyclic. Every single path is acyclic by construction.

    Parameters
    ----------
    agent_paths: AgentPaths

    Returns
    -------
    nx.DiGraph
        topology
    """

    topo = nx.DiGraph()
    skip_count = 0
    for index, path in enumerate(agent_paths):
        topo_path = nx.DiGraph()

        # add edges only to a copy
        topo_copy = topo.copy()
        for wp1, wp2 in zip(path, path[1:]):
            topo_copy.add_edge(wp1, wp2)
            topo_path.add_edge(wp1, wp2)

        # the path must have no cycles
        topo_path_cycles = list(nx.simple_cycles(topo_path))
        assert len(topo_path_cycles) == 0, f"cycle in shortest path {index}: {topo_path_cycles}"

        # if the copy has no cycles, take the copy.
        cycles = list(nx.simple_cycles(topo_copy))
        if len(cycles) == 0:
            topo = topo_copy
        else:
            skip_count += 1
    if skip_count > 0:
        rsp_logger.info(f"skipped {skip_count}  paths of {len(agent_paths)}")

    cycles = list(nx.simple_cycles(topo))

    assert len(cycles) == 0, f"cycle in re-combination of shortest paths, {cycles}"
    assert len(get_paths_in_route_dag(topo)) > 0, "no path after removing loopy paths"
    return topo


def get_paths_for_route_dag_constraints(
        topo: nx.DiGraph,
        route_dag_constraints: Optional[RouteDAGConstraints] = None) -> List[List[Waypoint]]:
    """Determine the routes through the route graph given the constraints.

    Parameters
    ----------
    route_dag_constraints

    Returns
    -------
    """
    paths = get_paths_in_route_dag(topo)
    return paths


def _get_topology_from_agents_path_dict(agents_paths_dict: AgentsPathsDict) -> TopoDict:
    # get topology from agent paths
    topo_dict = {agent_id: topo_from_agent_paths(agents_paths_dict[agent_id])
                 for agent_id in agents_paths_dict}

    return topo_dict


def apply_weight_route_change(
        schedule_problem: ScheduleProblemDescription,
        weight_route_change: int,
        weight_lateness_seconds: int
):
    """Returns a new `ScheduleProblemDescription` with all route section
    penalties scaled by the factor and with `weight_lateness_seconds`set as
    given.

    Parameters
    ----------
    schedule_problem: ScheduleProblemDescription
    weight_route_change: int
    weight_lateness_seconds: int

    Returns
    -------
    ScheduleProblemDescription
    """
    problem_weights = dict(
        schedule_problem._asdict(),
        **{
            'route_section_penalties': {agent_id: {
                edge: penalty * weight_route_change
                for edge, penalty in agent_route_section_penalties.items()
            } for agent_id, agent_route_section_penalties in
                schedule_problem.route_section_penalties.items()},
            'weight_lateness_seconds': weight_lateness_seconds
        })
    schedule_problem = ScheduleProblemDescription(**problem_weights)
    return schedule_problem
