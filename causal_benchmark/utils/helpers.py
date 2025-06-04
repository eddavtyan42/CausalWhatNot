import networkx as nx
import numpy as np
from typing import Iterable, Tuple, Set


def causallearn_to_dag(amat: np.ndarray, nodes: Iterable) -> nx.DiGraph:
    """Convert a causal-learn adjacency matrix to a NetworkX ``DiGraph``.

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


def edge_differences(
    pred: nx.DiGraph, true: nx.DiGraph
) -> Tuple[Set[tuple], Set[tuple], Set[tuple]]:
    """Return extra, missing and reversed edges between two DAGs.

    Parameters
    ----------
    pred : nx.DiGraph
        Predicted graph.
    true : nx.DiGraph
        Ground truth graph.

    Returns
    -------
    tuple of sets
        ``(extra, missing, reversed)`` edges represented as ``(u, v)`` tuples.
    """
    pred_edges = set(pred.edges())
    true_edges = set(true.edges())

    pred_pairs = {frozenset(e) for e in pred_edges}
    true_pairs = {frozenset(e) for e in true_edges}

    extra = {e for e in pred_edges if frozenset(e) not in true_pairs}
    missing = {e for e in true_edges if frozenset(e) not in pred_pairs}
    reversed_edges = {(u, v) for (u, v) in pred_edges if (v, u) in true_edges}

    return extra, missing, reversed_edges

