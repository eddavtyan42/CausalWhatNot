import networkx as nx
import numpy as np


def cpdag_to_dag(cpdag: nx.DiGraph) -> nx.DiGraph:
    if nx.is_directed_acyclic_graph(cpdag):
        return cpdag.copy()
    raise RuntimeError("Input graph has cycles")


def adjacency_to_nx(adj: np.ndarray, nodes) -> nx.DiGraph:
    G = nx.DiGraph(adj)
    return nx.relabel_nodes(G, {i: n for i, n in enumerate(nodes)})


def create_mis_specified_graph(true_graph: nx.DiGraph, error_type: str, u: str, v: str) -> nx.DiGraph:
    """Create a mis-specified version of a graph by modifying a single edge.

    Parameters
    ----------
    true_graph:
        Original directed acyclic graph.
    error_type:
        Either ``"missing"`` to remove an existing edge or ``"spurious"`` to add
        a new edge.
    u, v:
        Source and target nodes of the edge to modify.

    Returns
    -------
    nx.DiGraph
        A copy of ``true_graph`` with the specified edge removed or added.

    Raises
    ------
    ValueError
        If ``error_type`` is not recognized, if the nodes do not exist in the
        graph, or if the requested edge already satisfies/violates the
        specified error type.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.DiGraph([('X', 'Y')])
    >>> create_mis_specified_graph(G, 'missing', 'X', 'Y').has_edge('X', 'Y')
    False
    >>> create_mis_specified_graph(G, 'spurious', 'Y', 'X').has_edge('Y', 'X')
    True
    """

    if error_type not in {"missing", "spurious"}:
        raise ValueError("error_type must be either 'missing' or 'spurious'")

    if u not in true_graph.nodes or v not in true_graph.nodes:
        raise ValueError("Both u and v must be nodes in true_graph")

    mis_graph = true_graph.copy()

    if error_type == "missing":
        if not true_graph.has_edge(u, v):
            raise ValueError(f"Edge ({u}, {v}) does not exist in true_graph")
        mis_graph.remove_edge(u, v)
    else:  # error_type == "spurious"
        if true_graph.has_edge(u, v):
            raise ValueError(f"Edge ({u}, {v}) already exists in true_graph")
        mis_graph.add_edge(u, v)

    return mis_graph
