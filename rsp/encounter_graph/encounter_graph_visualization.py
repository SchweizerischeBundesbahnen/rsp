from typing import Dict
from typing import Optional

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


def _plot_encounter_graph_directed(weights_matrix: np.ndarray,
                                   title: str,
                                   file_name: Optional[str] = None,
                                   pos: Optional[dict] = None,
                                   highlights: Optional[dict] = None,
                                   changed_agents: Dict[int, bool] = None
                                   ):
    """This method plots the encounter graph and the heatmap of the distance
    matrix into one file.

    Parameters
    ----------
    weights_matrix
        matrix of weights to be rendered as encounter graph
    title
        title of plot
    file_name
        string of filename if saving is required
    pos [Optional]
        fixed positions of nodes in encountergraph
    highlights [Optional]
        dict containing the nodes that need to be highlighted

    Returns
    -------
        dict containing the positions of the nodes
    """
    dt = [('weight', float)]
    distance_matrix_as_weight = np.copy(weights_matrix)
    distance_matrix_as_weight.dtype = dt

    graph = nx.from_numpy_array(distance_matrix_as_weight, create_using=nx.DiGraph)
    print(f"nb edges={len(graph.edges)}, nodes={graph.number_of_nodes()}, "
          f"expected nb of edges={graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2} "
          "(in diff matrix, <= is ok since the zeros are those without change)")

    # Color the nodes
    node_color = ['lightblue' if (not changed_agents[i]) else 'red' for i in range(graph.number_of_nodes())]
    if highlights is not None:
        for node_idx in highlights:
            if highlights[node_idx]:
                node_color[node_idx] = 'r'

    fig = plt.figure(figsize=(18, 12), dpi=80)
    fig.suptitle(title, fontsize=16)
    plt.subplot(121)

    # draw nodes
    plt.gca().invert_yaxis()
    nx.draw_networkx_nodes(graph, pos, node_color=node_color)

    # draw edges with corresponding weights
    for edge_with_data in graph.edges(data=True):
        edge_weight = edge_with_data[2]['weight']
        edge = edge_with_data[:2]
        # TODO could we use colors instead? same as from heatmap below?
        nx.draw_networkx_edges(graph, pos, edgelist=[edge], width=edge_weight)

    # draw labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family='sans-serif')

    # visualize distance matrix as heat plot
    plt.subplot(122)

    plt.imshow(weights_matrix, cmap='hot', interpolation='nearest')

    if file_name is not None:
        fig.savefig(file_name)
        plt.close(fig)
    else:
        plt.show()

    return pos


def _plot_encounter_graph_undirected(distance_matrix: np.ndarray,
                                     title: str,
                                     file_name: Optional[str],
                                     pos: Optional[dict] = None,
                                     highlights: Optional[dict] = None):
    """This method plots the encounter graph and the heatmap of the distance
    matrix into one file.

    Parameters
    ----------
    distance_matrix
        matrix to be rendered as encounter graph
    title
        title of plot
    file_name
        string of filename if saving is required
    pos [Optional]
        fixed positions of nodes in encountergraph
    highlights [Optional]
        dict containing the nodes that need to be highlighted

    Returns
    -------
        dict containing the positions of the nodes
    """
    dt = [('weight', float)]
    distance_matrix_as_weight = np.copy(distance_matrix)
    distance_matrix_as_weight.dtype = dt

    graph = nx.from_numpy_array(distance_matrix_as_weight)
    print(f"nb edges={len(graph.edges)}, nodes={graph.number_of_nodes()}, "
          f"expected nb of edges={graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2} "
          "(in diff matrix, <= is ok since the zeros are those without change)")

    # position of nodes
    if pos is None:
        # Position nodes using Fruchterman-Reingold force-directed algorithm
        # https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html
        pos = nx.spring_layout(graph, seed=42)
    else:
        fixed_nodes = pos.keys()
        pos = nx.spring_layout(graph, seed=42, pos=pos, fixed=fixed_nodes)

    # Color the nodes
    node_color = ['lightblue' for i in range(graph.number_of_nodes())]
    if highlights is not None:
        for node_idx in highlights:
            if highlights[node_idx]:
                node_color[node_idx] = 'r'

    fig = plt.figure(figsize=(18, 12), dpi=80)
    fig.suptitle(title, fontsize=16)
    plt.subplot(121)

    # draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color=node_color)

    # draw edges with corresponding weights
    for edge_with_data in graph.edges(data=True):
        edge_weight = edge_with_data[2]['weight']
        edge = edge_with_data[:2]
        # TODO could we use colors instead? same as from heatmap below?
        nx.draw_networkx_edges(graph, pos, edgelist=[edge], width=edge_weight)

    # draw labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family='sans-serif')

    # visualize distance matrix as heat plot
    plt.subplot(122)

    plt.imshow(distance_matrix, cmap='hot', interpolation='nearest')

    if file_name is not None:
        fig.savefig(file_name)
        plt.close(fig)

    return pos
