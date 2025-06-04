import networkx as nx
import numpy as np
from typing import Iterable


def causallearn_to_dag(amat: np.ndarray, nodes: Iterable) -> nx.DiGraph:
    """Convert a causallearn adjacency matrix to a NetworkX ``DiGraph``.

    Parameters
    ----------
    amat : np.ndarray
        Adjacency matrix where 1/-1 pairs encode edge direction.
    nodes : Iterable
        Node labels in the desired order.
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(range(len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if amat[i, j] == 1 and amat[j, i] == -1:
                dag.add_edge(i, j)
            elif amat[i, j] == -1 and amat[j, i] == 1:
                dag.add_edge(j, i)
    return nx.relabel_nodes(dag, {i: n for i, n in enumerate(nodes)})

