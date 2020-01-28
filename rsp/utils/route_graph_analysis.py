"""Analysis the ExperimentFreeze as Route Graph."""
from typing import Dict
from typing import Optional
from typing import Set
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.utils.ordered_set import OrderedSet

from rsp.utils.data_types import AgentPaths
from rsp.utils.data_types import ExperimentFreeze


def visualize_experiment_freeze(agent_paths: AgentPaths,
                                f: ExperimentFreeze,
                                train_run_input: Trainrun,
                                train_run_full_after_malfunction: Trainrun,
                                train_run_delta_after_malfunction: Trainrun,
                                file_name: Optional[str] = None,
                                title: Optional[str] = None,
                                scale: int = 2,
                                ) -> nx.DiGraph:
    """Draws an agent's route graph with constraints into a file.

    Parameters
    ----------
    agent_paths
        the agent's paths spanning its routes graph
    f
        constraints for this agent
    file_name
        save graph to this file
    title
        title in the picture
    scale
        scale in or out
    """
    # N.B. FLATland uses row-column indexing, plt uses x-y (horizontal,vertical with vertical axis going bottom-top)

    # nx directed graph
    all_waypoints, topo = _extract_all_waypoints_and_digraph_from_spanning_paths(agent_paths)

    # figsize
    flatland_positions = np.array([waypoint.position for waypoint in all_waypoints])
    flatland_figsize = np.max(flatland_positions, axis=0) - np.min(flatland_positions, axis=0)
    plt.figure(figsize=(flatland_figsize[1] * scale, flatland_figsize[0] * scale))

    # plt title
    if title:
        plt.title(title)

    # positions with offsets for the pins
    offset = 0.25
    flatland_offset_pattern = {
        # heading north = coming from south: +row
        0: np.array([offset, 0]),
        # heading east = coming from west: -col
        1: np.array([0, -offset]),
        # heading south = coming from north: -row
        2: np.array([-offset, 0]),
        # heading west = coming from east: +col
        3: np.array([0, offset]),
    }
    flatland_pos_with_offset = {wp: np.array(wp.position) + flatland_offset_pattern[wp.direction] for wp in
                                all_waypoints}

    plt_pos = {wp: np.array([p[1], p[0]]) for wp, p in flatland_pos_with_offset.items()}

    tr_input_d: Dict[Waypoint, int] = {
        trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
        for trainrun_waypoint in train_run_input
    }
    tr_fam_d: Dict[Waypoint, int] = {
        trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
        for trainrun_waypoint in train_run_full_after_malfunction
    }
    tr_dam_d: Dict[Waypoint, int] = {
        trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
        for trainrun_waypoint in train_run_delta_after_malfunction
    }

    plt_color_map = [_get_color_for_node(node, f) for node in topo.nodes()]

    plt_labels = {
        wp: f"{wp.position[0]},{wp.position[1]},{wp.direction}\n"
            f"{_get_label_for_constraint_for_waypoint(wp, f)}\n"
            f"{_get_label_for_schedule_for_waypoint(wp, tr_input_d, tr_fam_d, tr_dam_d)}"
        for wp in
        all_waypoints}
    nx.draw(topo, plt_pos,
            labels=plt_labels,
            edge_color='black',
            width=1,
            linewidths=1,
            node_size=1500,
            node_color=plt_color_map,
            alpha=0.9)

    plt.gca().invert_yaxis()
    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()

    return topo


def _number_of_paths_in_route_dag(topo: nx.DiGraph) -> int:
    """Get the number of all source nodes (no incoming edges) to all sink nodes
    (no outgoing edges).

    Parameters
    ----------
    topo: DiGraph

    Returns
    -------
    int
    """
    sources = (node for node, in_degree in topo.in_degree if in_degree == 0)
    sinks = (node for node, out_degree in topo.out_degree if out_degree == 0)
    all_paths = []
    for source in sources:
        for sink in sinks:
            source_sink_paths = list(nx.all_simple_paths(topo, source, sink))
            all_paths += source_sink_paths
    return len(all_paths)


def _get_label_for_constraint_for_waypoint(waypoint: Waypoint, f: ExperimentFreeze) -> str:
    if waypoint in f.freeze_banned:
        return "X"
    s: str = "["
    if waypoint in f.freeze_visit:
        s = "! ["
    s += str(f.freeze_earliest[waypoint])
    s += ","
    s += str(f.freeze_latest[waypoint])
    s += "]"
    return s


def _get_color_for_node(n: Waypoint, f: ExperimentFreeze):
    # https://matplotlib.org/examples/color/named_colors.html
    if n in f.freeze_banned:
        return 'salmon'
    elif n in f.freeze_visit:
        return 'orange'
    else:
        return 'lightgreen'


def _get_label_for_schedule_for_waypoint(
        waypoint: Waypoint,
        train_run_input_dict: Dict[Waypoint, int],
        train_run_full_after_malfunction_dict: Dict[Waypoint, int],
        train_run_delta_after_malfunction_dict: Dict[Waypoint, int]
) -> str:
    s = []
    if waypoint in train_run_input_dict:
        s.append(f"S0: {train_run_input_dict[waypoint]}")
    if waypoint in train_run_full_after_malfunction_dict:
        s.append(f"S: {train_run_full_after_malfunction_dict[waypoint]}")
    if waypoint in train_run_delta_after_malfunction_dict:
        s.append(f"S': {train_run_delta_after_malfunction_dict[waypoint]}")
    return "\n".join(s)


def _extract_all_waypoints_and_digraph_from_spanning_paths(
        agent_paths: AgentPaths) -> Tuple[Set[Waypoint], nx.DiGraph]:
    """Extract  the agent's route DAG and all waypoints in it.

    Parameters
    ----------
    agent_paths: AgentPaths

    Returns
    -------
    Set[Waypoint], nx.DiGraph()
        the waypoints and the directed graph
    """
    topo = nx.DiGraph()
    all_waypoints: Set[Waypoint] = OrderedSet()
    for path in agent_paths:
        for wp1, wp2 in zip(path, path[1:]):
            topo.add_edge(wp1, wp2)
            all_waypoints.add(wp1)
            all_waypoints.add(wp2)
    return all_waypoints, topo


def get_number_of_paths_for_experiment_freeze(
        agent_paths: AgentPaths,
        experiment_freeze: Optional[ExperimentFreeze] = None) -> int:
    """Determine the number of routes through the route graph given the
    constraints.

    Parameters
    ----------
    agent_paths
    experiment_freeze

    Returns
    -------
    """
    _, topo = _extract_all_waypoints_and_digraph_from_spanning_paths(agent_paths)
    if experiment_freeze:
        for wp in experiment_freeze.freeze_banned:
            topo.remove_node(wp)
    nb_paths_after = _number_of_paths_in_route_dag(topo)
    return nb_paths_after
