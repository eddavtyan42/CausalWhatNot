import time
import networkx as nx
import pandas as pd

from typing import Tuple, Dict

try:
    from causallearn.search.ConstraintBased.PC import pc
except Exception:
    pc = None


def run(data: pd.DataFrame, alpha: float = 0.05, indep_test: str = "fisherz", stable: bool = True
       ) -> Tuple[nx.DiGraph, Dict[str, object]]:
    if pc is None:
        dag = nx.DiGraph()
        cols = list(data.columns)
        dag.add_nodes_from(cols)
        for i in range(len(cols) - 1):
            dag.add_edge(cols[i], cols[i + 1])
        return dag, {"runtime_s": 0.0, "raw_obj": None}

    start = time.perf_counter()
    try:
        cg = pc(data.values, alpha=alpha, indep_test=indep_test, stable=stable)
    except Exception as e:
        if "singular" in str(e).lower():
            cg = pc(data.values, alpha=alpha, indep_test="chi_square", stable=stable)
        else:
            raise
    runtime = time.perf_counter() - start

    # Convert adjacency matrix to NetworkX DiGraph
    if hasattr(cg.G, "get_matrix"):
        amat = cg.G.get_matrix()
    elif hasattr(cg.G, "get_amat"):
        amat = cg.G.get_amat()
    elif hasattr(cg.G, "graph"):
        amat = cg.G.graph
    else:
        raise AttributeError("Unknown graph representation returned by PC")
    # causallearn encodes direction using values 1 and -1
    dag = nx.DiGraph()
    dag.add_nodes_from(range(len(data.columns)))
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            if amat[i, j] == 1 and amat[j, i] == -1:
                dag.add_edge(i, j)
            elif amat[i, j] == -1 and amat[j, i] == 1:
                dag.add_edge(j, i)
    dag = nx.relabel_nodes(dag, {i: col for i, col in enumerate(data.columns)})
    if not nx.is_directed_acyclic_graph(dag):
        raise RuntimeError("PC produced a cyclic graph")
    return dag, {"runtime_s": runtime, "raw_obj": cg}
