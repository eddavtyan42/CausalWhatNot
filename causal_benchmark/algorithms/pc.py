import time
import networkx as nx
import pandas as pd

from typing import Tuple, Dict

from utils.helpers import causallearn_to_dag

try:
    from causallearn.search.ConstraintBased.PC import pc
except Exception:
    pc = None


def run(
    data: pd.DataFrame,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    stable: bool = True,
) -> Tuple[nx.DiGraph, Dict[str, object]]:
    if pc is None:
        raise ImportError(
            "causal-learn is required for the PC algorithm. Install via pip install causal-learn."
        )

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

    dag = causallearn_to_dag(amat, data.columns)
    if not nx.is_directed_acyclic_graph(dag):
        raise RuntimeError("PC produced a cyclic graph")
    return dag, {"runtime_s": runtime, "raw_obj": cg}
