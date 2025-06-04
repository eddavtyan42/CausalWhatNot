import time
import networkx as nx
import pandas as pd

from typing import Tuple, Dict

from utils.helpers import causallearn_to_dag

try:
    from causallearn.search.ScoreBased.GES import ges
except Exception:
    ges = None


def run(data: pd.DataFrame, score_func: str = "bic") -> Tuple[nx.DiGraph, Dict[str, object]]:
    if ges is None:
        raise ImportError(
            "causallearn is required for the GES algorithm. Install via `pip install causallearn`."
        )

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

    dag = causallearn_to_dag(amat, data.columns)
    if not nx.is_directed_acyclic_graph(dag):
        raise RuntimeError("GES produced a cyclic graph")
    return dag, {"runtime_s": runtime, "raw_obj": gs}
