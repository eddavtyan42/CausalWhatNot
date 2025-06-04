import time
from typing import Tuple, Dict

import networkx as nx
import numpy as np
import pandas as pd


def run(
    data: pd.DataFrame,
    lambda1: float = 0.1,
    lambda2: float = 0.1,
    max_iter: int = 1000,
    seed: int = 0,
    **_,
) -> Tuple[nx.DiGraph, Dict[str, object]]:
    """Simple COSMO-style learner using priority-based linear regression.

    Parameters
    ----------
    data : pd.DataFrame
        Observational data with numeric values.
    lambda1 : float, optional
        L1 regularisation strength (unused but kept for API compatibility).
    lambda2 : float, optional
        L2 regularisation strength for ridge regression.
    max_iter : int, optional
        Number of iterations for the optimizer (unused).
    seed : int, optional
        RNG seed controlling the random priority ordering.
    """
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    X = data.values
    d = X.shape[1]
    order = rng.permutation(d)
    W = np.zeros((d, d))

    for idx in range(1, d):
        j = order[idx]
        parents_idx = order[:idx]
        if len(parents_idx) == 0:
            continue
        X_par = X[:, parents_idx]
        y = X[:, j]
        # Ridge regression closed form
        XtX = X_par.T @ X_par + lambda2 * np.eye(len(parents_idx))
        w = np.linalg.solve(XtX, X_par.T @ y)
        for p, coeff in zip(parents_idx, w):
            W[p, j] = coeff

    runtime = time.perf_counter() - start

    G = nx.DiGraph()
    cols = list(data.columns)
    G.add_nodes_from(cols)
    for i in range(d):
        for j in range(d):
            if abs(W[i, j]) > 1e-2:
                G.add_edge(cols[i], cols[j], weight=W[i, j])

    if not nx.is_directed_acyclic_graph(G):
        raise RuntimeError("COSMO produced a cyclic graph")

    return G, {"runtime_s": runtime, "weights": W}

