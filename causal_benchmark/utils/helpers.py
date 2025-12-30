import networkx as nx
import numpy as np
from typing import Iterable, Tuple, Set, Dict
import json
import logging


def causallearn_to_dag(amat: np.ndarray, nodes: Iterable) -> Tuple[nx.DiGraph, Dict[str, object]]:
    """Convert a causal-learn adjacency matrix to a NetworkX ``DiGraph``.

    Causal-learn's PC and GES algorithms output CPDAGs (Completed Partially
    Directed Acyclic Graphs) which contain both directed and undirected edges.
    This function converts them to DAGs by:
    1. Adding all directed edges first
    2. Orienting undirected edges in a way that doesn't create cycles

    Parameters
    ----------
    amat : np.ndarray
        Adjacency matrix where 1/-1 pairs encode edge direction.
        - (1, -1) at (i,j), (j,i) means i -> j
        - (-1, 1) at (i,j), (j,i) means j -> i  
        - (1, 1) or (-1, -1) means undirected edge
    nodes : Iterable
        Node labels in the desired order.
    """
    nodes_list = list(nodes)
    n = len(nodes_list)
    
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n))
    
    # First pass: add all directed edges
    undirected_pairs = []
    for i in range(n):
        for j in range(n):
            if amat[i, j] == 1 and amat[j, i] == -1:
                dag.add_edge(i, j)
            elif amat[i, j] == -1 and amat[j, i] == 1:
                dag.add_edge(j, i)
            elif i < j:  # Check for undirected edges (only once per pair)
                if (amat[i, j] == 1 and amat[j, i] == 1) or \
                   (amat[i, j] == -1 and amat[j, i] == -1):
                    undirected_pairs.append((i, j))
    
    # Second pass: orient undirected edges to avoid cycles
    # Use topological hints from existing directed edges when possible
    undirected_edges: Set[Tuple[int, int]] = set()
    for i, j in undirected_pairs:
        # Try to orient based on existing graph structure
        # Prefer orientation that respects existing partial order
        i_ancestors = nx.ancestors(dag, i) if i in dag else set()
        j_ancestors = nx.ancestors(dag, j) if j in dag else set()
        
        # If j is ancestor of i, orient j -> i; if i is ancestor of j, orient i -> j
        if j in i_ancestors:
            dag.add_edge(j, i)
            undirected_edges.add((j, i))
        elif i in j_ancestors:
            dag.add_edge(i, j)
            undirected_edges.add((i, j))
        else:
            # No clear preference, try i -> j first
            dag.add_edge(i, j)
            if not nx.is_directed_acyclic_graph(dag):
                dag.remove_edge(i, j)
                dag.add_edge(j, i)
                if not nx.is_directed_acyclic_graph(dag):
                    # Neither orientation works, skip this edge
                    dag.remove_edge(j, i)
                    logger = logging.getLogger("benchmark")
                    logger.warning(
                        "Could not orient edge (%s, %s) without creating cycle, skipping",
                        nodes_list[i], nodes_list[j]
                    )
                else:
                    undirected_edges.add((j, i))
            else:
                undirected_edges.add((i, j))
    
    name_map = {i: n for i, n in enumerate(nodes_list)}
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

    logger = logging.getLogger("benchmark")
    logger.info(
        "Edge diffs computed: pred_edges=%d true_edges=%d extra=%d missing=%d reversed=%d",
        pred.number_of_edges(), true.number_of_edges(), len(extra), len(missing), len(reversed_edges)
    )
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
    logging.getLogger("benchmark").info("Edge diffs JSON saved: file=%s", str(path))

