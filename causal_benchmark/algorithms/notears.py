"""NOTEARS causal discovery algorithm via the CausalNex backend."""

from __future__ import annotations

__all__ = ["run"]

import sys
import time
from typing import Tuple, Dict

import networkx as nx
import numpy as np
import pandas as pd


def run(
    data: pd.DataFrame,
    threshold: float = 0.1,
    **kwargs,
) -> Tuple[nx.DiGraph, Dict[str, object]]:
    """Run NOTEARS on a dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        Numeric dataframe containing observational samples.

    Other Parameters
    ----------------
    max_iter : int, optional
        Maximum number of iterations (default from causalnex).
    lambda1 : float, optional
        L1 penalty parameter.
    lambda2 : float, optional
        L2 penalty parameter.

    Returns
    -------
    nx.DiGraph
        Estimated directed acyclic graph.
    Dict[str, object]
        Dictionary with keys ``runtime_s`` (float) and ``weights`` (np.ndarray) containing the learned
        weighted adjacency matrix.
    """

    if data.isna().any().any():
        raise ValueError("NOTEARS cannot handle missing values.")

    if sys.version_info >= (3, 11):
        raise ImportError("NOTEARS via CausalNex only supports Python <3.11")

    try:
        from causalnex.structure.notears import from_pandas
    except Exception as e:  # pragma: no cover - import failure tested via runtime
        raise ImportError(
            "NOTEARS requires causalnex>=0.12 and torch. Install or remove 'notears' from config."
        ) from e

    # Ignore unsupported legacy parameter if present
    kwargs.pop("backend", None)

    start = time.perf_counter()
    sm = from_pandas(data, w_threshold=threshold, **kwargs)
    runtime = time.perf_counter() - start

    # `sm` is a StructureModel (a DiGraph) with weight attributes
    G = nx.DiGraph()
    G.add_nodes_from(data.columns)
    for u, v, w in sm.edges(data="weight"):
        if abs(w) > 1e-8:
            G.add_edge(u, v, weight=w)

    # Build weight matrix in the same ordering as data.columns
    cols = list(data.columns)
    W = np.zeros((len(cols), len(cols)))
    for u, v, w in sm.edges(data="weight"):
        i, j = cols.index(u), cols.index(v)
        W[i, j] = w

    if not nx.is_directed_acyclic_graph(G):
        raise RuntimeError("NOTEARS produced a cyclic graph")

    return G, {"runtime_s": runtime, "weights": W}
