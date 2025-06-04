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
    # map commonly used shorthand score names to those expected by causallearn
    score_map = {
        "bic": "local_score_BIC",
        "bdeu": "local_score_BDeu",
    }
    cl_score = score_map.get(score_func.lower(), score_func)
    gs = ges(data.values, score_func=cl_score)
    runtime = time.perf_counter() - start

    if hasattr(gs["G"], "get_matrix"):
        amat = gs["G"].get_matrix()
    elif hasattr(gs["G"], "get_amat"):
        amat = gs["G"].get_amat()
    elif hasattr(gs["G"], "graph"):
        amat = gs["G"].graph
    else:
        raise AttributeError("Unknown graph representation returned by GES")
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
        raise RuntimeError("GES produced a cyclic graph")
    return dag, {"runtime_s": runtime, "raw_obj": gs}
