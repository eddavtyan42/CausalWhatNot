import networkx as nx
import numpy as np
from typing import Iterable, Tuple, Set, Dict
import json


def causallearn_to_dag(amat: np.ndarray, nodes: Iterable) -> Tuple[nx.DiGraph, Dict[str, object]]:
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
    n = len(nodes)
    undirected_edges: Set[Tuple[int, int]] = set()
    for i in range(n):
        for j in range(n):
            if amat[i, j] == 1 and amat[j, i] == -1:
                dag.add_edge(i, j)
            elif amat[i, j] == -1 and amat[j, i] == 1:
                dag.add_edge(j, i)
            elif amat[i, j] == 1 and amat[j, i] == 1 and i < j:
                dag.add_edge(i, j)
                undirected_edges.add((i, j))
    name_map = {i: n for i, n in enumerate(nodes)}
    dag = nx.relabel_nodes(dag, name_map)
    undirected_named = {(name_map[i], name_map[j]) for (i, j) in undirected_edges}
    return dag, {"undirected_edges": undirected_named}


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


def dump_edge_differences_json(
    extra: Iterable[Tuple[str, str]],
    missing: Iterable[Tuple[str, str]],
    reversed_edges: Iterable[Tuple[str, str]],
    path: str,
) -> None:
    """Write edge difference lists to ``path`` in JSON format.

    Parameters
    ----------
    extra, missing, reversed_edges : iterable of edge tuples
        Edge lists to write out.
    path : str
        Destination file path.
    """
    data = {
        "extra": [[u, v] for u, v in sorted(extra)],
        "missing": [[u, v] for u, v in sorted(missing)],
        "reversed": [[u, v] for u, v in sorted(reversed_edges)],
    }
    with open(path, "w") as f:
        json.dump(data, f)

