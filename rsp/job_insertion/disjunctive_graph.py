import pprint
from collections import deque
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from flatland.envs.rail_trainrun_data_structures import Trainrun
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from flatland.envs.rail_trainrun_data_structures import TrainrunWaypoint
from flatland.envs.rail_trainrun_data_structures import Waypoint

from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.route_dag.route_dag import MAGIC_DIRECTION_FOR_SOURCE_TARGET
from rsp.route_dag.route_dag import ScheduleProblemDescription

_pp = pprint.PrettyPrinter(indent=4)
Route = List[Waypoint]
Trainroute = List[Waypoint]
TrainrouteDict = Dict[int, Trainroute]
Schedule = TrainrunDict
Resource = Tuple[int, int]


class DisjunctiveGraphEdgeType(Enum):
    DISJUNCTIVE = 0
    CONJUNCTIVE = 1


MAGIC_DIRECTION_FOR_EXIT_EVENT = 6

DisjunctiveGraph = nx.DiGraph
DisjunctiveNode = Tuple[int, Waypoint]
DisjunctiveGraphNode = Union[Tuple[int, Waypoint], None]
DisjunctiveGraphEdge = Tuple[DisjunctiveGraphNode, DisjunctiveGraphNode]
Selection = Set[DisjunctiveGraphEdge]

TrainrunWithId = Tuple[int, Trainrun]

OrderedTrains = List[int]

OrderedDisjunctiveGraphNodes = List[DisjunctiveGraphNode]


def sort_vertices_by_train_start_and_earliest(
        problem: ScheduleProblemDescription,
        solution: SchedulingExperimentResult) -> Tuple[OrderedTrains, OrderedDisjunctiveGraphNodes]:
    """Order trains and section entry points for display in a disjunctive
    graph:

    - columns = trains
    - rows = nodes

    Trains are sorted by their departure time

    Vertices are sorted
    1. their first appearance in train in their order
    2. earliest time as received from the ASP solution (reflects depth)

    Parameters
    ----------
    problem
    solution

    Returns
    -------
    """
    all_waypoints = {
        node
        for train, topo in problem.topo_dict.items()
        for node in topo.nodes
    }

    # sort trains by their start time
    sorted_trains: OrderedTrains = \
        list(map(lambda p: p[0],
                 sorted(
                     [
                         (train, trainrun)
                         for train, trainrun in solution.trainruns_dict.items()
                     ],
                     key=lambda p: p[1][0].scheduled_at)))

    # sort vertices
    # 1. train they appear first in in the order of the sorted trains
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


def disjunctive_graph_is_dummy_source(node: DisjunctiveGraphNode) -> bool:
    # we use None as marker for 'sigma'
    return node is None


def wrap_waypoint(train: int, wp: Waypoint) -> DisjunctiveGraphNode:
    """Wraps train and waypoint into `DisjunctiveGraphNode`

    Parameters
    ----------
    train
    wp

    Returns
    -------
    """
    return (train, wp)


def get_trainroute_from_trainrun(trainrun: Trainrun) -> Route:
    """Extract the route from the train's schedule (aka Trainrun)

    Parameters
    ----------
    trainrun

    Returns
    -------
    """
    return [trainrun_waypoint.waypoint for trainrun_waypoint in trainrun]


def make_disjunctive_graph(
        minimum_travel_time_dict: Dict[int, int],
        trainroute_dict: TrainrouteDict,
        start_time_dict: Dict[int, int],
        release_time: int = 1,
        no_disjunctions: Set[int] = None,
        debug: bool = False) -> DisjunctiveGraph:
    """Make a disjunctive problem for the problem. Used for visualisation only.
    Only job insertion graphs are used in neighborhood search (with disjunctive
    arcs fixed for all but one train).

    Parameters
    ----------
    minimum_travel_time_dict
    trainroute_dict
    start_time_dict
    release_time
    no_disjunctions
    debug

    Returns
    -------
    """
    disjunctive_graph = nx.DiGraph()

    train_resource_to_edge_mapping: Dict[Tuple[int, Resource], DisjunctiveGraphEdge] = {}
    resource_conflicts: Dict[Resource, Set[int]] = {}
    for train, trainroute in trainroute_dict.items():
        disjunctive_graph.add_edge(
            None,
            (train, trainroute[0]),
            type=DisjunctiveGraphEdgeType.CONJUNCTIVE,
            weight=start_time_dict[train]
        )

        for wp1, wp2 in zip(trainroute, trainroute[1:]):
            node1 = wrap_waypoint(train, wp1)
            node2 = wrap_waypoint(train, wp2)
            disjunctive_graph.add_edge(
                node1,
                node2,
                type=DisjunctiveGraphEdgeType.CONJUNCTIVE,
                # TODO SIM-322 hard-coded assumption that last segment is 1
                weight=(minimum_travel_time_dict[train]
                        if (wp1.direction != MAGIC_DIRECTION_FOR_SOURCE_TARGET and
                            wp2.direction != MAGIC_DIRECTION_FOR_SOURCE_TARGET)
                        else 1)
            )
            edge = (node1, node2)
            resource = wp1.position
            train_resource_to_edge_mapping[(train, resource)] = edge
            resource_conflicts.setdefault(resource, set()).add(train)

    # in our setting, resources are the cell positions
    for resource, trains in resource_conflicts.items():
        if len(trains) < 2:
            continue
        for t1 in trains:
            for t2 in trains:
                # if both trains are kept fixed, no disjunctions between them
                if no_disjunctions is not None and t1 in no_disjunctions and t2 in no_disjunctions:
                    continue

                if t1 >= t2:
                    continue
                # exit + release time <= next entry
                # a) t1 before t2
                disjunctive_graph.add_edge(
                    train_resource_to_edge_mapping[(t1, resource)][1],
                    train_resource_to_edge_mapping[(t2, resource)][0],
                    type=DisjunctiveGraphEdgeType.DISJUNCTIVE,
                    weight=release_time
                )
                # b) t2 before t1
                disjunctive_graph.add_edge(
                    train_resource_to_edge_mapping[(t2, resource)][1],
                    train_resource_to_edge_mapping[(t1, resource)][0],
                    type=DisjunctiveGraphEdgeType.DISJUNCTIVE,
                    weight=release_time
                )
    # TODO unit test instead
    sanity_check_disjunctive_graph(disjunctive_graph=disjunctive_graph)
    return disjunctive_graph


def draw_disjunctive_graph(disjunctive_graph: DisjunctiveGraph,
                           file_name: str,
                           sorted_trains: OrderedTrains,
                           sorted_vertices: OrderedDisjunctiveGraphNodes,
                           highlight_edges: List[DisjunctiveGraphEdge] = None,
                           scale: int = 4,
                           padding: int = 2,
                           title: Optional[str] = None):
    """Create networkx plot of the disjunctive graph. Needs an orde  for trains
    (columns) and vertices (rows).

    Parameters
    ----------
    disjunctive_graph
    file_name
    sorted_trains
    sorted_vertices
    highlight_edges
    scale
    padding
    title
    """
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

    nb_trains = len(sorted_trains)
    nb_vertices = len(sorted_vertices)
    nb_rows = nb_vertices
    nb_columns = nb_trains

    # rows -> vertices
    # columns -> trains
    # somehow inverted in matplotlib
    pos = {
        (train, wp): np.array([train_index[train] + padding, vertex_index[wp] + padding])
        for node in disjunctive_graph.nodes
        if not disjunctive_graph_is_dummy_source(node)
        for train, wp in [node]

    }
    pos[None] = np.array([nb_trains / 2 + padding, 0])
    node_labels = {
        (train, waypoint): f"t{train}\n({waypoint.position}, {waypoint.direction})"
        for node in disjunctive_graph.nodes
        if not disjunctive_graph_is_dummy_source(node)
        for train, waypoint in [node]
    }
    node_labels[None] = 'sigma'
    edge_labels = {
        (u, v): d['weight']
        for (u, v, d) in disjunctive_graph.edges(data=True)
    }

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
    nx.draw_networkx_nodes(disjunctive_graph,
                           pos,
                           nodelist=[None],
                           label=node_labels,
                           node_size=1500,
                           alpha=0.9,
                           node_color='red')
    if len(conjunctive_edges) > 0:
        nx.draw_networkx_edges(G=disjunctive_graph,
                               pos=pos,
                               label=edge_labels,
                               edge_color='black',
                               edgelist=conjunctive_edges,
                               width=3,
                               node_size=1500,
                               arrowsize=20,
                               connectionstyle="arc3,rad=0.1")
    if len(disjunctive_edges) > 0:
        collection = nx.draw_networkx_edges(G=disjunctive_graph,
                                            pos=pos,
                                            label=edge_labels,
                                            edgelist=disjunctive_edges,
                                            width=3,
                                            node_size=1500,
                                            arrowsize=20,
                                            edge_color='grey',
                                            style='dotted',
                                            connectionstyle="arc3,rad=0.1")
        # workaround for style="dotted": https://stackoverflow.com/questions/51138059/no-dotted-line-with-networkx-drawn-on-basemap
        for patch in collection:
            patch.set_linestyle('dotted')

    if highlight_edges is not None:
        nx.draw_networkx_edges(G=disjunctive_graph,
                               pos=pos,
                               label=edge_labels,
                               edge_color='red',
                               edgelist=highlight_edges,
                               width=3,
                               node_size=1500,
                               arrowsize=20,
                               connectionstyle="arc3,rad=0.1")

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


def get_conjunctive_graph_by_inserting_at_end(
        job_insertion_graph: DisjunctiveGraph,
        train_to_insert: int) -> Tuple[DisjunctiveGraph, List[DisjunctiveGraphEdge]]:
    """Apply selection which inserts the train to insert after all other
    trains. Returns a conjunctive graph.

    Parameters
    ----------
    job_insertion_graph
    train_to_insert

    Returns
    -------
    """
    conjunctive_graph = job_insertion_graph.copy()

    # edges leaving the train to insert would make it go before -> remove
    edges_to_remove = [
        (u, v)
        for (u, v, d) in conjunctive_graph.edges(data=True)
        if d['type'] == DisjunctiveGraphEdgeType.DISJUNCTIVE and u[0] == train_to_insert
    ]
    conjunctive_graph.remove_edges_from(edges_to_remove)

    # select all remaining disjunctive edges, make them conjunctive
    selection = []
    for (u, v, d) in conjunctive_graph.edges(data=True):
        if d['type'] == DisjunctiveGraphEdgeType.DISJUNCTIVE:
            d['type'] = DisjunctiveGraphEdgeType.CONJUNCTIVE
            selection.append((u, v))
    return conjunctive_graph, selection


def left_closure(
        conjunctive_graph: DisjunctiveGraph,
        critical_arc: DisjunctiveGraphEdge,
        release_time: int = 1,
        debug: bool = False
):
    """Add critical arc to queue. Until queue is empty:

        - pop arc from queue and swap
        - determine incoming edges of positive cycles introduced by swapping, add them to the queue

    Parameters
    ----------
    conjunctive_graph
    critical_arc
    release_time
    debug

    Returns
    -------
    """
    conjunctive_graph = conjunctive_graph.copy()

    # add critical arc to our queue
    d = deque()
    d.append(critical_arc)
    closure = []
    while d:
        edge = d.pop()

        # if added multiple times to queue, it might not be in the graph any more
        if edge not in conjunctive_graph.edges():
            continue

        # invert by removing the edge and inserting the mate
        mate = get_mate(
            disjunctive_graph=conjunctive_graph,
            edge=edge
        )
        conjunctive_graph.remove_edge(*edge)
        conjunctive_graph.add_edge(*mate, weight=release_time, type=DisjunctiveGraphEdgeType.CONJUNCTIVE)
        closure.append(mate)
        mate_tail, mate_head = mate
        leaving_from_train = mate_tail[0]
        if debug:
            print(
                f" -> replacing by {mate}: train {mate_tail[0]} before train {mate_head[0]} at {mate_head[1]}=={edge[1][1]}")
        # TODO optimize, no need to enumerate all cycles, exploit structure
        for cycle in nx.simple_cycles(G=conjunctive_graph):
            if mate_head in cycle and mate_tail in cycle:
                # sum over all edges in cycle
                sum_weights = sum([
                    conjunctive_graph.get_edge_data(n1, n2)['weight']
                    for n1, n2 in zip(cycle, cycle[1:] + [cycle[0]])])
                # positive cycle found!
                if sum_weights > 0:
                    for n1, n2 in zip(cycle, cycle[1:] + [cycle[0]]):
                        train1, _ = n1
                        train2, _ = n2
                        # find incoming edge

                        if train1 != leaving_from_train and train2 == leaving_from_train:
                            # add the incoming edge to our queue: must be swapped later!
                            d.append((n1, n2))
                            # TODO is it safe to assume the cycle contains only one incoming edge?
                            break

    return conjunctive_graph, closure


def force_disjunctive_edges_from_schedule(disjunctive_graph: DisjunctiveGraph,
                                          schedule: Schedule,
                                          ) -> DisjunctiveGraph:
    """Apply a full selection to a disjunctive graph. Educational only: we
    would never do that in a neighborhood search scheme, we would leave one
    train out to get a job insertion graph.

    Parameters
    ----------
    disjunctive_graph
    schedule

    Returns
    -------
    """
    schedule_at: Dict[DisjunctiveGraphNode, int] = {
        (train, trainrun_waypoint.waypoint): trainrun_waypoint.scheduled_at
        for train, trainrun in schedule.items()
        for trainrun_waypoint in trainrun
    }

    disjunctive_graph_from_schedule = nx.DiGraph()
    for u, v, d in disjunctive_graph.edges(data=True):
        if d['type'] == DisjunctiveGraphEdgeType.CONJUNCTIVE:
            disjunctive_graph_from_schedule.add_edge(u, v, **d)
        else:
            if disjunctive_graph_is_dummy_source(u):
                disjunctive_graph_from_schedule.add_edge(u, v, **d)
            else:
                exit_node_train_1 = u
                entry_node_train_2 = v
                release_time = d['weight']

                # if train is removed, add disjunctive edge
                if exit_node_train_1 not in schedule_at or entry_node_train_2 not in schedule_at:
                    disjunctive_graph_from_schedule.add_edge(u, v, **d)
                    continue

                if schedule_at[exit_node_train_1] + release_time <= schedule_at[entry_node_train_2]:
                    disjunctive_graph_from_schedule.add_edge(u, v, **dict(d, **{
                        'type': DisjunctiveGraphEdgeType.CONJUNCTIVE}))
                    mate = get_mate(disjunctive_graph, (u, v))
                    exit_node_train_1_mate, entry_node_train_2_mate = mate
                    assert schedule_at[exit_node_train_1_mate] + release_time > schedule_at[entry_node_train_2_mate]
    return disjunctive_graph_from_schedule


def path_successor(disjunctive_graph: DisjunctiveGraph, u: DisjunctiveGraphNode):
    """

    Parameters
    ----------
    disjunctive_graph
    u

    Returns
    -------

    """
    train, _ = u
    successors_by_conjunctive_edge = [
        v
        for v in disjunctive_graph.successors(u)
        if disjunctive_graph.get_edge_data(u, v)[
               'type'] == DisjunctiveGraphEdgeType.CONJUNCTIVE
    ]

    next_node_in_path = successors_by_conjunctive_edge[0]

    # sanity checks
    # TODO no assertions hidden in code, move into unit test
    train2, _ = next_node_in_path
    assert train == train2
    assert u != next_node_in_path

    return next_node_in_path


def path_predecessor(disjunctive_graph: DisjunctiveGraph, v: DisjunctiveGraphNode):
    train, _ = v
    predecessors_by_conjunctive_edge = [
        u
        for u in disjunctive_graph.predecessors(v)
        if disjunctive_graph.get_edge_data(u, v)['type'] == DisjunctiveGraphEdgeType.CONJUNCTIVE
    ]

    previous_node_in_path = predecessors_by_conjunctive_edge[0]
    train2, _ = previous_node_in_path
    # sanity checks
    # TODO no assertions hidden in code, move into unit test
    assert train == train2
    assert v != previous_node_in_path
    return previous_node_in_path


def get_mate(disjunctive_graph: DisjunctiveGraph, edge: DisjunctiveGraphEdge) -> DisjunctiveGraphEdge:
    """Since we have no multiple resources per route section, we can determine
    the mate by inspecting the disjunctive graph. Probably, in a general
    setting, we would have to keep track of mates in some dict.

      train_1: u_1 -----> v_1
                   X
      train_2: u_2 -----> v_2

                u_1, u_2 entry into same cell
                a) train_1 before train_2: v_1 + release_time <= u_2
                b) train_2 before train_1: v_2 + release_time <= u_1

    Parameters
    ----------
    disjunctive_graph
    edge

    Returns
    -------
    """
    (v_1, u_2) = edge

    u_1 = path_predecessor(disjunctive_graph, v_1)
    v_2 = path_successor(disjunctive_graph, u_2)

    train_u_1, wp_u_1 = u_1
    train_v_1, _ = v_1
    train_u_2, wp_u_2 = u_2
    train_v_2, _ = v_2

    # sanity checks
    # TODO no assertions hidden in code, move into unit test
    assert train_u_1 == train_v_1
    assert train_u_2 == train_v_2
    assert wp_u_1.position == wp_u_2.position

    # exit event to start event
    return (v_2, u_1)


def sanity_check_disjunctive_graph(disjunctive_graph: DisjunctiveGraph):
    """Checks structural properties of a disjunctive graph. Not very general.
    Mainly for debugging and development. We should make proper unit tests
    instead.

    Parameters
    ----------
    disjunctive_graph
    """
    edges_without_data = disjunctive_graph.edges(data=False)
    for (v, u_, d) in disjunctive_graph.edges(data=True):
        edge = (v, u_)
        if d['type'] == DisjunctiveGraphEdgeType.DISJUNCTIVE:
            train1, _ = v
            train2, _ = u_

            # verify disjunctive nodes only between different trains
            assert train1 != train2

            # verify mate is in disjunctive graph as well
            mate = get_mate(disjunctive_graph, edge)
            assert mate in edges_without_data, f"expected mate {mate} of {edge} to be in disjunctive graph"
    for (u, v, d) in disjunctive_graph.edges(data=True):
        if d['type'] == DisjunctiveGraphEdgeType.CONJUNCTIVE and u is not None:
            train1, _ = u
            train2, _ = v

            # we currently have conjunctive edges only between
            assert train1 == train2


def make_schedule_from_conjunctive_graph(conjunctive_graph=DisjunctiveGraph, debug: bool = False) -> Schedule:
    """Propagate summed up weights along conjunctive arcs, take the max over
    all incoming weight sums.

    Parameters
    ----------
    conjunctive_graph
    debug

    Returns
    -------
    """
    updated = {None}
    schedule_dict: Dict[DisjunctiveGraphNode, int] = {None: 0}
    trains = set()
    while len(updated) > 0:
        if debug:
            print(f"****** {updated}")
        future_updated = set()
        for node in updated:
            if debug:
                print(f"   {node}")
            for neighbor in conjunctive_graph.neighbors(node):
                edge_data = conjunctive_graph.get_edge_data(node, neighbor)
                if node is None:
                    schedule_dict[neighbor] = edge_data['weight']
                else:
                    schedule_dict[neighbor] = max(schedule_dict.get(neighbor, -np.inf),
                                                  schedule_dict[node] + edge_data['weight'])
                if debug:
                    print(f"{node}->{neighbor}: updating to {schedule_dict[neighbor]}")
                future_updated.add(neighbor)
                train, _ = neighbor
                trains.add(train)
        updated = future_updated
    del schedule_dict[None]
    trainrun_dict = {}
    for (train, waypoint), scheduled_at in schedule_dict.items():
        trainrun_dict[train] = trainrun_dict.get(train, [])
        trainrun_dict[train].append(TrainrunWaypoint(scheduled_at=scheduled_at, waypoint=waypoint))
    for unsorted_trainrun_waypoints in trainrun_dict.values():
        unsorted_trainrun_waypoints.sort(key=lambda trwp: trwp.scheduled_at)
    return trainrun_dict
