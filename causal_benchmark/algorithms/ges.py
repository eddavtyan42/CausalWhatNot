import time
import networkx as nx
import pandas as pd

from typing import Tuple, Dict

try:
    from causallearn.search.ScoreBased.GES import ges
except Exception:
    ges = None


def run(data: pd.DataFrame, score_func: str = "bic") -> Tuple[nx.DiGraph, Dict[str, object]]:
    if ges is None:
        dag = nx.DiGraph()
        cols = list(data.columns)
        dag.add_nodes_from(cols)
        for i in range(len(cols) - 1):
            dag.add_edge(cols[i], cols[i + 1])
        return dag, {"runtime_s": 0.0, "raw_obj": None}

    start = time.perf_counter()
    gs = ges(data.values, score_func=score_func)
    runtime = time.perf_counter() - start

    if hasattr(gs["G"], "get_matrix"):
        amat = gs["G"].get_matrix()
    else:
        amat = gs["G"].get_amat()
    dag = nx.DiGraph(amat)
    dag = nx.relabel_nodes(dag, {i: col for i, col in enumerate(data.columns)})
    if not nx.is_directed_acyclic_graph(dag):
        raise RuntimeError("GES produced a cyclic graph")
    return dag, {"runtime_s": runtime, "raw_obj": gs}
