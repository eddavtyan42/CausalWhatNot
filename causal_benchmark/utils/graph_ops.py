import networkx as nx
import numpy as np


def cpdag_to_dag(cpdag: nx.DiGraph) -> nx.DiGraph:
    if nx.is_directed_acyclic_graph(cpdag):
        return cpdag.copy()
    raise RuntimeError("Input graph has cycles")


def adjacency_to_nx(adj: np.ndarray, nodes) -> nx.DiGraph:
    G = nx.DiGraph(adj)
    return nx.relabel_nodes(G, {i: n for i, n in enumerate(nodes)})
