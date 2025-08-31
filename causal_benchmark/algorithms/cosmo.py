import time
from typing import Tuple, Dict

import networkx as nx
import numpy as np
import pandas as pd
import logging


def run(
    data: pd.DataFrame,
    lambda1: float = 0.1,
    lambda2: float = 0.1,
    max_iter: int = 1000,
    seed: int = 0,
    n_restarts: int = 10,
    **_,
) -> Tuple[nx.DiGraph, Dict[str, object]]:
    """Simple COSMO-style learner using priority-based linear regression.

    Parameters
    ----------
    data : pd.DataFrame
        Observational data with numeric values.
    lambda1 : float, optional
        L1 regularisation strength. When >0, Lasso regression is used.
    lambda2 : float, optional
        L2 regularisation strength for ridge regression.
    max_iter : int, optional
        Number of iterations for the optimizer.
    seed : int, optional
        RNG seed controlling the random priority ordering.
    n_restarts : int, optional
        Number of random orderings to try (default 10); the best by BIC is returned.
    """
    logger = logging.getLogger("benchmark")
    logger.info(
        "COSMO start: n=%d d=%d lambda1=%.3f lambda2=%.3f max_iter=%d seed=%d restarts=%d",
        len(data), data.shape[1], lambda1, lambda2, max_iter, seed, n_restarts,
    )
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    X = data.values
    n, d = X.shape

    best_bic = np.inf
    best_W: np.ndarray | None = None
    best_order: np.ndarray | None = None

    for _ in range(max(n_restarts, 1)):
        order = rng.permutation(d)
        W = np.zeros((d, d))
        bic = 0.0

        for idx in range(d):
            j = order[idx]
            parents_idx = order[:idx]
            if len(parents_idx) == 0:
                # No parents; only contributes residual variance to BIC
                var = X[:, j].var(ddof=1)
                bic += n * np.log(var + 1e-12)
                continue

            X_par = X[:, parents_idx]
            y = X[:, j]
            if lambda1 > 0:
                try:
                    from sklearn.linear_model import Lasso
                except Exception as e:  # pragma: no cover - optional dep
                    raise ImportError(
                        "scikit-learn is required for L1 regularisation"
                    ) from e
                model = Lasso(
                    alpha=lambda1, fit_intercept=False, max_iter=max_iter
                )
                model.fit(X_par, y)
                w = model.coef_
            else:
                XtX = X_par.T @ X_par + lambda2 * np.eye(len(parents_idx))
                w = np.linalg.solve(XtX, X_par.T @ y)

            pred = X_par @ w
            res = y - pred
            var = res.var(ddof=1)
            bic += n * np.log(var + 1e-12) + len(parents_idx) * np.log(n)
            for p, coeff in zip(parents_idx, w):
                W[p, j] = coeff

        if bic < best_bic:
            best_bic = bic
            best_W = W
            best_order = order

    assert best_W is not None and best_order is not None

    runtime = time.perf_counter() - start

    G = nx.DiGraph()
    cols = list(data.columns)
    G.add_nodes_from(cols)
    for i in range(d):
        for j in range(d):
            if abs(best_W[i, j]) > 1e-2:
                G.add_edge(cols[i], cols[j], weight=best_W[i, j])

    if not nx.is_directed_acyclic_graph(G):
        raise RuntimeError("COSMO produced a cyclic graph")

    logger.info("COSMO end: edges=%d runtime_s=%.3f bic=%.3f", G.number_of_edges(), runtime, float(best_bic))
    return G, {
        "runtime_s": runtime,
        "weights": best_W,
        "ordering": [cols[i] for i in best_order],
        "bic": float(best_bic),
    }

