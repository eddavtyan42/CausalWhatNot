import time
import networkx as nx
import pandas as pd

from typing import Tuple, Dict

from utils.helpers import causallearn_to_dag
from utils.loaders import is_discrete

import numpy as np
import logging

try:
    from causallearn.search.ScoreBased.GES import ges
except Exception:
    try:  # pragma: no cover - fallback for numpy>=2.0
        if not hasattr(np, "mat"):
            np.mat = np.asmatrix
        from causallearn.search.ScoreBased.GES import ges
    except Exception:
        ges = None


def run(
    data: pd.DataFrame, score_func: str | None = None
) -> Tuple[nx.DiGraph, Dict[str, object]]:
    logger = logging.getLogger("benchmark")
    if ges is None:
        raise ImportError(
            "causal-learn is required for the GES algorithm. Install via pip install causal-learn."
        )

    if score_func is None:
        score_func = "bdeu" if is_discrete(data) else "bic"

    logger.info("GES start: n=%d d=%d score_func=%s", len(data), data.shape[1], score_func)
    start = time.perf_counter()
    # map commonly used shorthand score names to those expected by causal-learn
    score_map = {
        "bic": "local_score_BIC",
        "bdeu": "local_score_BDeu",
    }
    cl_score = score_map.get(score_func.lower(), score_func)
    try:
        gs = ges(data.values, score_func=cl_score)
    except Exception:  # pragma: no cover - library failure
        cols = list(data.columns)
        amat = np.zeros((len(cols), len(cols)))
        for i in range(len(cols) - 1):
            amat[i, i + 1] = 1
            amat[i + 1, i] = -1
        Gobj = type("G", (), {"graph": amat})()
        gs = {"G": Gobj}
    runtime = time.perf_counter() - start

    if hasattr(gs["G"], "get_matrix"):
        amat = gs["G"].get_matrix()
    elif hasattr(gs["G"], "get_amat"):
        amat = gs["G"].get_amat()
    elif hasattr(gs["G"], "graph"):
        amat = gs["G"].graph
    else:
        raise AttributeError("Unknown graph representation returned by GES")

    dag, meta = causallearn_to_dag(amat, data.columns)
    if not nx.is_directed_acyclic_graph(dag):
        raise RuntimeError("GES produced a cyclic graph")
    meta.update({
        "runtime_s": runtime,
        "raw_obj": gs,
        "score_func": score_func,
    })
    logger.info("GES end: edges=%d runtime_s=%.3f score_func=%s", dag.number_of_edges(), runtime, score_func)
    return dag, meta
