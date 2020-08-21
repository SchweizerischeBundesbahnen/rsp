"""Analysis the ExperimentFreeze as Route Graph."""
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.experiment_solvers.data_types import ExperimentMalfunction
from rsp.logger import rsp_logger
from rsp.logger import VERBOSE
from rsp.schedule_problem_description.data_types_and_utils import AgentPaths
from rsp.schedule_problem_description.data_types_and_utils import RouteDAGConstraints
from rsp.schedule_problem_description.data_types_and_utils import RouteDagEdge
from rsp.schedule_problem_description.data_types_and_utils import RouteSectionPenalties
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.schedule_problem_description.data_types_and_utils import WaypointPenalties

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
}


def visualize_route_dag_constraints_simple_wrapper(
        schedule_problem_description: ScheduleProblemDescription,
        experiment_malfunction: ExperimentMalfunction,
        agent_id: int,
        trainrun_dict: Optional[TrainrunDict] = None,
        file_name: Optional[str] = None,
):
    visualize_route_dag_constraints_simple(
        topo=schedule_problem_description.topo_dict[agent_id],
        f=schedule_problem_description.route_dag_constraints_dict[agent_id],
        train_run=trainrun_dict[agent_id] if trainrun_dict is not None else None,
        title=f"agent {agent_id}, malfunction={experiment_malfunction}",
        file_name=file_name
    )


def visualize_route_dag_constraints_simple(
        topo: nx.DiGraph,
        f: Optional[RouteDAGConstraints] = None,
        train_run: Optional[Trainrun] = None,
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

    schedule = None
    if train_run:
        schedule = {
            trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
            for trainrun_waypoint in train_run
        }

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

    plt_color_map = [_get_color_for_node(node, f) for node in topo.nodes()]

    plt_labels = {
        wp: f"{wp.position[0]},{wp.position[1]},{wp.direction}\n"
            f"{_get_label_for_constraint_for_waypoint(wp, f) if f is not None else ''}\n"
            f"{str(schedule[wp]) if schedule and wp in schedule else ''}"
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

    plt.legend(handles=_get_color_labels())

    plt.gca().invert_yaxis()
    if file_name is not None:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()
    return topo


def visualize_route_dag_constraints(
        topo: nx.DiGraph,
        constraints_to_visualize: RouteDAGConstraints,
        trainrun_to_visualize: Trainrun,
        route_section_penalties: RouteSectionPenalties,
        edge_eff_route_penalties: RouteSectionPenalties,
        vertex_eff_lateness: WaypointPenalties,
        train_run_full: Trainrun,
        train_run_full_after_malfunction: Trainrun,
        train_run_delta_after_malfunction: Trainrun,
        file_name: Optional[str] = None,
        title: Optional[str] = None,
        scale: int = 4,
) -> nx.DiGraph:
    """Draws an agent's route graph with constraints into a file.
    Parameters
    ----------
    topo
    constraints_to_visualize
        constraints for this agent to visualize, draws nodes in different corresponding to their constraints
    trainrun_to_visualize
        trainrun for this agents, used for visualizing a trainrun in red and for visualizing delay (different to earliest from constraints_to_visualize)
    file_name
        save graph to this file
    title
        title in the picture
    scale
        scale in or out
    route_section_penalties
        route penalty is displayed in edge label
    edge_eff_route_penalties
        route penalty is displayed in edge label
    vertex_eff_lateness
        lateness is displayed in node labels
    train_run_full
        used in labels for S0
    train_run_full_after_malfunction
        used in labels for S
    train_run_delta_after_malfunction
        used in labels for S'
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
        for trainrun_waypoint in train_run_full
    }
    tr_fam_d: Dict[Waypoint, int] = {
        trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
        for trainrun_waypoint in train_run_full_after_malfunction
    }
    tr_dam_d: Dict[Waypoint, int] = {
        trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
        for trainrun_waypoint in train_run_delta_after_malfunction
    }

    plt_color_map = [_get_color_for_node(node, constraints_to_visualize) for node in topo.nodes()]

    plt_labels = {
        wp: f"{wp.position[0]},{wp.position[1]},{wp.direction}\n"
            f"{_get_label_for_constraint_for_waypoint(wp, constraints_to_visualize)}\n"
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

    # legend
    plt.legend(handles=_get_color_labels())

    # draw edges of train run red
    trainrun_edges = [(twp1.waypoint, twp2.waypoint) for twp1, twp2 in zip(trainrun_to_visualize, trainrun_to_visualize[1:])]
    nx.draw_networkx_edges(topo, plt_pos, edgelist=trainrun_edges, edge_color='red')

    # delay labels: delay with respect to earliest and increase in delay wrt earliest (red if non-zero delay or non-zero increase in delay)
    delay_with_respect_to_earliest = {twp.waypoint: twp.scheduled_at - constraints_to_visualize.earliest[twp.waypoint] for twp in trainrun_to_visualize}
    delay_increase = {trainrun_to_visualize[0].waypoint: 0}
    for twp1, twp2 in zip(trainrun_to_visualize, trainrun_to_visualize[1:]):
        delay_increase[twp2.waypoint] = delay_with_respect_to_earliest[twp2.waypoint] - delay_with_respect_to_earliest[twp1.waypoint]
    plt_alt_labels = {
        twp.waypoint: f"scheduled_at={twp.scheduled_at}\ndelay={delay_with_respect_to_earliest[twp.waypoint]:+d}\ndelay inc={delay_increase[twp.waypoint]:+d})"
        for twp in trainrun_to_visualize
    }
    plt_alt_pos = {wp: (pos[0] - 0.25, pos[1] + 0.25) for wp, pos in plt_pos.items()}
    nx.draw_networkx_labels(topo, plt_alt_pos, plt_alt_labels, font_color='black')
    plt_alt_labels_red = {
        wp: label
        for wp, label in plt_alt_labels.items()
        if delay_with_respect_to_earliest[wp] != 0 or delay_increase[wp] != 0
    }
    nx.draw_networkx_labels(topo, plt_alt_pos, plt_alt_labels_red, font_color='red')

    plt.legend(handles=_get_color_labels())

    plt.gca().invert_yaxis()
    rsp_logger.log(VERBOSE, file_name)
    if file_name is not None:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()

    return topo


def _visualize_cycles_in_route_graph(agent_paths: AgentPaths, cycles: List[List[Tuple[Waypoint, Waypoint]]], topo: nx.DiGraph):
    """Visualize the individual paths of the agent and cycles into separate
    files.

    Parameters
    ----------
    agent_paths
    cycles
    topo
    """
    for k, cycle in enumerate(cycles):
        visualize_cycle_in_route_dag(
            topo=topo,
            file_name=f"cycle_{k}.pdf",
            cycle=cycle
        )
    for k, path in enumerate(agent_paths):
        topo_path = nx.DiGraph()
        print((path[0], path[-1]))
        for wp1, wp2 in zip(path, path[1:]):
            topo.add_edge(wp1, wp2)
            topo_path.add_edge(wp1, wp2)
        visualize_cycle_in_route_dag(
            topo=topo_path,
            file_name=f"path_{k}.pdf",
            cycle=[]
        )


def visualize_cycle_in_route_dag(
        topo: nx.DiGraph,
        cycle: List[Tuple[TrainrunWaypoint, TrainrunWaypoint]],
        train_run: Optional[Trainrun] = None,
        file_name: Optional[str] = None,
        title: Optional[str] = None,
        scale: int = 4,

) -> nx.DiGraph:
    """Visualize topology, cycle as red, dummy edges.

    Parameters
    ----------
    topo
    train_run
    file_name
        save graph to this file
    title
        title in the picture
    scale
        scale in or out
    cycle
        drawn red
    """
    # N.B. FLATland uses row-column indexing, plt uses x-y (horizontal,vertical with vertical axis going bottom-top)

    # nx directed graph
    all_waypoints: List[Waypoint] = list(topo.nodes)

    schedule = None
    if train_run:
        schedule = {
            trainrun_waypoint.waypoint: trainrun_waypoint.scheduled_at
            for trainrun_waypoint in train_run
        }

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
    plt_color_map = [_get_color_for_node_cycle(node, cycle) for node in topo.nodes()]
    plt.legend(handles=_get_color_labels_cycle())

    plt_labels = {
        wp: f"{wp.position[0]},{wp.position[1]},{wp.direction}\n"
            f"{str(schedule[wp]) if schedule and wp in schedule else ''}"
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
    if file_name is not None:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()
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
    # mark missing nodes
    if waypoint not in f.earliest or waypoint not in f.latest:
        return "X"
    s: str = "["
    s += str(f.earliest[waypoint])
    s += ","
    s += str(f.latest[waypoint])
    s += "]"
    return s


def _get_color_for_node(n: Waypoint, f: RouteDAGConstraints):
    # https://matplotlib.org/examples/color/named_colors.html
    # mark missing nodes
    if n not in f.earliest or n not in f.latest:
        return 'salmon'
    elif f.earliest[n] == f.latest[n]:
        return 'yellow'
    else:
        return 'lightgreen'


def _get_color_labels():
    salmon_patch = mpatches.Patch(color='salmon', label='removed')
    yellow_patch = mpatches.Patch(color='yellow', label='exact time if visited')
    lightgreen_patch = mpatches.Patch(color='lightgreen', label='non-zero length interval if visited')
    legend_patches = [salmon_patch, yellow_patch, lightgreen_patch]
    return legend_patches


def _get_color_for_node_cycle(node, cycle):
    if node in cycle:
        return 'red'
    else:
        return 'lightblue'


def _get_color_labels_cycle():
    yellow_patch = mpatches.Patch(color='yellow', label='dummy')
    red_patch = mpatches.Patch(color='red', label='cycle')
    lightblue_patch = mpatches.Patch(color='lightblue', label='other')
    legend_patches = [yellow_patch, red_patch, lightblue_patch]
    return legend_patches


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
