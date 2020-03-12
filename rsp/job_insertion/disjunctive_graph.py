from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.route_dag.route_dag import ScheduleProblemDescription


class DisjunctiveGraphEdgeType(Enum):
    DISJUNCTIVE = 0
    CONJUNCTIVE = 1


DisjunctiveGraph = nx.DiGraph
DisjunctiveGraphNodeType = Tuple[int, Waypoint]

TrainrunWithId = Tuple[int, Trainrun]


def sort_vertices_by_train_start_and_earliest(problem: ScheduleProblemDescription,
                                              solution: SchedulingExperimentResult):
    all_waypoints = {
        node
        for train, topo in problem.topo_dict.items()
        for node in topo.nodes
    }

    # sort trains by their start time
    sorted_trains: List[int] = \
        list(map(lambda p: p[0],
                 sorted(
                     [
                         (train, trainrun)
                         for train, trainrun in solution.trainruns_dict.items()
                     ],
                     key=lambda p: p[1][0].scheduled_at)))

    # sort vertices
    # 1. train
    # 2. earliest (reflects depth)
    sorted_vertices: List[Waypoint] = []
    for train in sorted_trains:
        for earliest in problem.route_dag_constraints_dict[train].freeze_earliest.keys():
            vertices_at_earliest: List[Tuple[TrainrunWaypoint, int]] = list(
                filter(lambda kv: kv[0] == earliest, problem.route_dag_constraints_dict[train].freeze_earliest.items()))
            for waypoint, _ in vertices_at_earliest:
                if waypoint not in sorted_vertices:
                    sorted_vertices.append(waypoint)

    assert len(sorted_vertices) == len(all_waypoints)

    return sorted_trains, sorted_vertices


DisjunctiveNode = Tuple[int, Waypoint]


def make_disjunctive_graph(problem: ScheduleProblemDescription) -> DisjunctiveGraph:
    disjunctive_graph = nx.DiGraph()
    # TODO add dummy source and dummy sink

    for train, topo in problem.topo_dict.items():
        def wrap_waypoint(wp):
            return (train, wp)

        for vertex in topo.nodes:
            topo = problem.topo_dict[train]
            if vertex not in topo.nodes:
                continue
            for edge_from, edge_to in topo.edges:
                edge_from: Waypoint = edge_from
                edge_to: Waypoint = edge_to
                disjunctive_graph.add_edge(
                    wrap_waypoint(edge_from),
                    wrap_waypoint(edge_to),
                    type=DisjunctiveGraphEdgeType.DISJUNCTIVE
                    if topo.out_degree[edge_from] > 1
                    else DisjunctiveGraphEdgeType.CONJUNCTIVE,
                    weight=problem.minimum_travel_time_dict[train]
                    if (edge_from.direction == MAGIC_DIRECTION_FOR_SOURCE_TARGET or
                        edge_to.direction == MAGIC_DIRECTION_FOR_SOURCE_TARGET)
                    # TODO SIM-3222 hard-coded assumption that last segment is 1
                    else 1
                )

    return disjunctive_graph


def draw_disjunctive_graph(disjunctive_graph: DisjunctiveGraph,
                           file_name: str,
                           problem: ScheduleProblemDescription,
                           solution: SchedulingExperimentResult,
                           scale: int = 4,
                           padding: int = 5,
                           title: Optional[str] = None):
    # sort trains and vertices
    sorted_trains, sorted_vertices = sort_vertices_by_train_start_and_earliest(problem, solution)

    # returns the sort index of the Waypoint
    vertex_index: Dict[Waypoint, int] = {
        waypoint: index
        for index, waypoint in enumerate(sorted_vertices)
    }
    # returns the sort index of the train
    train_index: Dict[int, int] = {
        train: index
        for index, train in enumerate(sorted_trains)
    }

    # rows -> vertices
    # columns -> trains
    # somehow inverted in matplotlib
    pos = {(train, wp): np.array([train_index[train], vertex_index[wp]])
           for train, wp in disjunctive_graph.nodes}
    node_labels = {
        (train, waypoint): f"t{train}\n({waypoint.position}, {waypoint.direction})"
        for train, waypoint in disjunctive_graph.nodes
    }
    edge_labels = {
        (u, v): d['weight']
        for (u, v, d) in disjunctive_graph.edges(data=True)
    }

    nb_trains = len(problem.topo_dict)
    nb_vertices = len(sorted_vertices)
    nb_rows = nb_vertices
    nb_columns = nb_trains

    # add more since we use curved edges
    plt.figure(figsize=(scale * (nb_columns + padding), (nb_rows + padding) * scale))
    conjunctive_edges = [(u, v)
                         for (u, v, d) in disjunctive_graph.edges(data=True)
                         if d['type'] == DisjunctiveGraphEdgeType.CONJUNCTIVE]
    disjunctive_edges = [(u, v)
                         for (u, v, d) in disjunctive_graph.edges(data=True)
                         if d['type'] == DisjunctiveGraphEdgeType.DISJUNCTIVE]
    nx.draw_networkx_labels(disjunctive_graph,
                            pos,
                            labels=node_labels,
                            font_size=20,
                            font_family='sans-serif')
    nx.draw_networkx_nodes(disjunctive_graph,
                           pos,
                           label=node_labels,
                           node_size=1500,
                           alpha=0.9)
    collection = nx.draw_networkx_edges(G=disjunctive_graph,
                                        pos=pos,
                                        label=edge_labels,
                                        edge_color='black',
                                        edgelist=conjunctive_edges,
                                        width=3,
                                        node_size=1500,
                                        arrowsize=20,
                                        connectionstyle="arc3,rad=0.1")
    # workaround for style="dotted": https://stackoverflow.com/questions/51138059/no-dotted-line-with-networkx-drawn-on-basemap
    for patch in collection:
        patch.set_linestyle('dotted')
    collection = nx.draw_networkx_edges(G=disjunctive_graph,
                                        pos=pos,
                                        label=edge_labels,
                                        edgelist=disjunctive_edges,
                                        width=3,
                                        node_size=1500,
                                        arrowsize=20,
                                        edge_color='grey',
                                        # TODO does not work
                                        style='dotted',
                                        connectionstyle="arc3,rad=0.1")
    for patch in collection:
        patch.set_linestyle('dotted')

    nx.draw_networkx_edge_labels(G=disjunctive_graph,
                                 pos=pos,
                                 edge_labels=edge_labels,
                                 label_pos=0.5
                                 )
    # plt title
    if title is not None:
        plt.title(title)
    plt.gca().invert_yaxis()
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()
    plt.close()
