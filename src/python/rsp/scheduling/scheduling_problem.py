"""Route DAG data structures and utils."""
import pprint
from enum import Enum
from typing import Dict
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import Tuple

import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.utils.rsp_logger import rsp_logger

TopoDict = Dict[int, nx.DiGraph]
AgentPaths = List[List[Waypoint]]
AgentsPathsDict = Dict[int, AgentPaths]

RouteDAGConstraints = NamedTuple("RouteDAGConstraints", [("earliest", Dict[Waypoint, int]), ("latest", Dict[Waypoint, int])])

RouteDAGConstraintsDict = Dict[int, RouteDAGConstraints]
RouteDagEdge = Tuple[Waypoint, Waypoint]
RouteSectionPenalties = Dict[RouteDagEdge, int]
WaypointPenalties = Dict[Waypoint, int]
RouteSectionPenaltiesDict = Dict[int, RouteSectionPenalties]
ScheduleProblemDescription = NamedTuple(
    "ScheduleProblemDescription",
    [
        ("route_dag_constraints_dict", RouteDAGConstraintsDict),
        ("minimum_travel_time_dict", Dict[int, int]),
        ("topo_dict", Dict[int, nx.DiGraph]),
        ("max_episode_steps", int),
        ("route_section_penalties", RouteSectionPenaltiesDict),
        ("weight_lateness_seconds", int),
    ],
)


class ScheduleProblemEnum(Enum):
    PROBLEM_SCHEDULE = "PROBLEM_SCHEDULE"
    PROBLEM_RSP_FULL_AFTER_MALFUNCTION = "PROBLEM_RSP_FULL_AFTER_MALFUNCTION"
    PROBLEM_RSP_DELTA_PERFECT_AFTER_MALFUNCTION = "PROBLEM_RSP_DELTA_PERFECT_AFTER_MALFUNCTION"
    PROBLEM_RSP_DELTA_RANDOM_AFTER_MALFUNCTION = "PROBLEM_RSP_DELTA_RANDOM_AFTER_MALFUNCTION"
    PROBLEM_RSP_DELTA_ONLINE_AFTER_MALFUNCTION = "PROBLEM_RSP_DELTA_ONLINE_AFTER_MALFUNCTION"


def schedule_problem_description_equals(s1: ScheduleProblemDescription, s2: ScheduleProblemDescription):
    """Tests whether two schedule_problem_descriptions are the equal.

    We cannot test by == since we have `nx.DiGraph` objects as values.
    """
    for index, slot in enumerate(s1._fields):
        if slot == "topo_dict":
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


def route_dag_constraints_dict_from_list_of_train_run_waypoint(l: List[TrainrunWaypoint]) -> Dict[TrainrunWaypoint, int]:
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


def path_stats(nb_paths: List[int]) -> str:
    return (
        f"min={min(nb_paths)}, "
        f"max={max(nb_paths)}, "
        f"avg={np.mean(nb_paths)}, "
        f"median={np.median(nb_paths)}, "
        f"q70={np.quantile(nb_paths, 0.7)}, "
        f"q90={np.quantile(nb_paths, 0.9)}"
    )


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


def _get_topology_from_agents_path_dict(agents_paths_dict: AgentsPathsDict) -> TopoDict:
    # get topology from agent paths
    topo_dict = {agent_id: topo_from_agent_paths(agents_paths_dict[agent_id]) for agent_id in agents_paths_dict}

    return topo_dict


_pp = pprint.PrettyPrinter(indent=4)


def experiment_freeze_dict_pretty_print(d: RouteDAGConstraintsDict):
    for agent_id, route_dag_constraints in d.items():
        prefix = f"agent {agent_id} "
        experiment_freeze_pretty_print(route_dag_constraints, prefix)


def experiment_freeze_pretty_print(route_dag_constraints: RouteDAGConstraints, prefix: str = ""):
    print(f"{prefix}earliest={_pp.pformat(route_dag_constraints.earliest)}")
    print(f"{prefix}latest={_pp.pformat(route_dag_constraints.latest)}")
