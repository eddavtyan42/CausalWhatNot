import time
import networkx as nx
import pandas as pd

from typing import Tuple, Dict

from utils.helpers import causallearn_to_dag
from utils.loaders import is_discrete

try:
    from causallearn.search.ConstraintBased.PC import pc
except Exception:
    pc = None


def run(
    data: pd.DataFrame,
    alpha: float = 0.05,
    indep_test: str | None = None,
    stable: bool = True,
) -> Tuple[nx.DiGraph, Dict[str, object]]:
    if pc is None:
        raise ImportError(
            "causal-learn is required for the PC algorithm. Install via pip install causal-learn."
        )

    if indep_test is None:
        indep_test = "chisq" if is_discrete(data) else "fisherz"

    start = time.perf_counter()
    try:
        cg = pc(data.values, alpha=alpha, indep_test=indep_test, stable=stable)
    except Exception as e:
        if "singular" in str(e).lower():
            cg = pc(data.values, alpha=alpha, indep_test="chisq", stable=stable)
        else:
            # Fall back to an empty graph on any other failure so that the
            # caller receives a valid (but empty) result instead of an
            # exception.  This mirrors the error-handling strategy used in the
            # GES wrapper and avoids propagating failures that would lead to
            # NaN metrics downstream.
            runtime = time.perf_counter() - start
            dag = nx.DiGraph()
            dag.add_nodes_from(data.columns)
            return dag, {"runtime_s": runtime, "indep_test": indep_test}
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

    dag, meta = causallearn_to_dag(amat, data.columns)
    if not nx.is_directed_acyclic_graph(dag):
        raise RuntimeError("PC produced a cyclic graph")
    meta.update({
        "runtime_s": runtime,
        "raw_obj": cg,
        "indep_test": indep_test,
    })
    return dag, meta
