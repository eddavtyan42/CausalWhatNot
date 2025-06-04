import time
import networkx as nx
import pandas as pd
from typing import Tuple, Dict


def run(data: pd.DataFrame, backend: str = "causalnex", **kwargs) -> Tuple[nx.DiGraph, Dict[str, object]]:
    start = time.perf_counter()
    if backend == "causalnex":
        try:
            from causalnex.structure.notears import from_numpy
        except Exception:
            raise RuntimeError("NOTEARS with causalnex is not installed")
        adj = from_numpy(data.values, **kwargs)
    elif backend == "gcastle":
        try:
            from gcastle.algorithms.dag.notears import Notears
        except Exception:
            raise RuntimeError("gCastle Notears is not installed")
        model = Notears(**kwargs)
        model.learn(data.values)
        adj = model.adj_mat
    else:
        raise RuntimeError(f"Unknown backend {backend}")

    runtime = time.perf_counter() - start
    dag = nx.DiGraph(adj)
    dag = nx.relabel_nodes(dag, {i: col for i, col in enumerate(data.columns)})
    if not nx.is_directed_acyclic_graph(dag):
        raise RuntimeError("NOTEARS produced a cyclic graph")
    return dag, {"runtime_s": runtime, "raw_obj": None}
