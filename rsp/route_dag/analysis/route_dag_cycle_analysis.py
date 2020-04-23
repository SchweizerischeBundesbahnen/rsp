"""Route DAG cycle visualization."""
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint


def _visualize_cycles_in_route_graph(agent_paths, cycles, topo):
    for k, cycle in enumerate(cycles):
        visualize_cycle_in_route_dag(
            topo=topo,
            file_name=f"cycle_{k}.pdf",
            cycle=cycle,
            dummies=[agent_path[1] for agent_path in agent_paths] + [agent_path[-2] for agent_path in agent_paths]
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
            cycle=[],
            dummies=[path[0], path[-1]]
        )


def _get_color_for_node(node, cycle, dummies):
    if node in dummies:
        return 'yellow'
    elif node in cycle:
        return 'red'
    else:
        return 'lightblue'


def visualize_cycle_in_route_dag(
        topo: nx.DiGraph,
        cycle: List[Tuple[TrainrunWaypoint, TrainrunWaypoint]],
        dummies: List[Tuple[TrainrunWaypoint, TrainrunWaypoint]],
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

    plt_pos = {wp: np.array([p[1], p[0]]) for wp, p in flatland_pos_with_offset.items()}
    plt_color_map = [_get_color_for_node(node, cycle, dummies) for node in topo.nodes()]

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
