"""Analysis the ExperimentFreeze as Route Graph."""
from typing import Dict
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.logger import rsp_logger
from rsp.logger import VERBOSE
from rsp.route_dag.route_dag import RouteDagEdge
from rsp.route_dag.route_dag import RouteSectionPenalties
from rsp.route_dag.route_dag import WaypointPenalties
from rsp.utils.data_types import RouteDAGConstraints

OFFSET = 0.25
FLATLAND_OFFSET_PATTERN = {
    # heading north = coming from south: +row
    0: np.array([OFFSET, 0]),
    # heading east = coming from west: -col
    1: np.array([0, -OFFSET]),
    # heading south = coming from north: -row
    2: np.array([-OFFSET, 0]),
    # heading west = coming from east: +col
    3: np.array([0, OFFSET]),
    # dummy heading = no offset
    5: np.array([0.5 * -OFFSET, 0.5 * -OFFSET])
}


def visualize_route_dag_constraints_simple(
        topo: nx.DiGraph,
        f: RouteDAGConstraints,
        train_run: Trainrun,
        file_name: Optional[str] = None,
        title: Optional[str] = None,
        scale: int = 4,
) -> nx.DiGraph:
    """Draws an agent's route graph with constraints into a file.
    Parameters
    ----------
    topo
    f
        constraints for this agent
    train_run
    file_name
        save graph to this file
    title
        title in the picture
    scale
        scale in or out
    """
    # N.B. FLATland uses row-column indexing, plt uses x-y (horizontal,vertical with vertical axis going bottom-top)

    # nx directed graph
    all_waypoints: List[Waypoint] = list(topo.nodes)

    schedule = {
        trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
        for trainrun_waypoint in train_run
    }
    print(schedule)

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
        # dummy heading = top left, not in the center to better see arrows
        5: np.array([-offset, -offset])
    }
    flatland_pos_with_offset = {wp: np.array(wp.position) + flatland_offset_pattern[wp.direction] for wp in
                                all_waypoints}
    print(flatland_pos_with_offset)

    plt_pos = {wp: np.array([p[1], p[0]]) for wp, p in flatland_pos_with_offset.items()}

    plt_color_map = [_get_color_for_node(node, f) for node in topo.nodes()]

    plt_labels = {
        wp: f"{wp.position[0]},{wp.position[1]},{wp.direction}\n"
            f"{_get_label_for_constraint_for_waypoint(wp, f)}\n"
            f"{str(schedule[wp]) if wp in schedule else ''}"
        for wp in
        all_waypoints}

    nx.draw(topo,
            plt_pos,
            labels=plt_labels,
            edge_color='black',
            width=1,
            linewidths=1,
            node_size=1500,
            node_color=plt_color_map,
            alpha=0.9)

    plt.gca().invert_yaxis()
    print(file_name)
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()
    plt.close()

    return topo


def visualize_route_dag_constraints(
        topo: nx.DiGraph,
        f: RouteDAGConstraints,
        route_section_penalties: RouteSectionPenalties,
        edge_eff_route_penalties: RouteSectionPenalties,
        vertex_eff_lateness: WaypointPenalties,
        train_run_input: Trainrun,
        train_run_full_after_malfunction: Trainrun,
        train_run_delta_after_malfunction: Trainrun,
        file_name: Optional[str] = None,
        title: Optional[str] = None,
        scale: int = 4,
) -> nx.DiGraph:
    """Draws an agent's route graph with constraints into a file.
    Parameters
    ----------
    edge_lateness
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
    all_waypoints: List[Waypoint] = list(topo.nodes)

    # figsize
    flatland_positions = np.array([waypoint.position for waypoint in all_waypoints])
    flatland_figsize = np.max(flatland_positions, axis=0) - np.min(flatland_positions, axis=0)
    plt.figure(figsize=(flatland_figsize[1] * scale, flatland_figsize[0] * scale))

    # plt title
    if title:
        plt.title(title)

    # positions with offsets for the pins
    flatland_pos_with_offset = {wp: np.array(wp.position) + FLATLAND_OFFSET_PATTERN[wp.direction] for wp in
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
            f"{_get_label_for_schedule_for_waypoint(wp, tr_input_d, tr_fam_d, tr_dam_d)}" +
            (f"\neff late: {vertex_eff_lateness.get(wp, '')}" if vertex_eff_lateness.get(wp, 0) > 0 else "")
        for wp in
        all_waypoints}

    edge_labels = {
        edge: _get_edge_label(edge, route_section_penalties, edge_eff_route_penalties)
        for edge in topo.edges
    }

    nx.draw(topo,
            plt_pos,
            labels=plt_labels,
            edge_color='black',
            width=1,
            linewidths=1,
            node_size=1500,
            node_color=plt_color_map,
            alpha=0.9)
    nx.draw_networkx_edge_labels(topo, plt_pos, edge_labels=edge_labels)

    plt.gca().invert_yaxis()
    rsp_logger.log(VERBOSE, file_name)
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()
    #plt.close()

    return topo


def _get_edge_label(edge: RouteDagEdge,
                    route_section_penalties: RouteSectionPenalties,
                    eff_edge_route_penalties: RouteSectionPenalties) -> str:
    label = ""
    label += str(eff_edge_route_penalties.get(edge, 0))
    label += " / "
    label += str(route_section_penalties.get(edge, 0))

    if label == "0 / 0":
        return ""
    return label


def _get_label_for_constraint_for_waypoint(waypoint: Waypoint, f: RouteDAGConstraints) -> str:
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


def _get_color_for_node(n: Waypoint, f: RouteDAGConstraints):
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
