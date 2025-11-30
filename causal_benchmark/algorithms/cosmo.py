import time
from typing import Tuple, Dict, Optional, List

import networkx as nx
import numpy as np
import pandas as pd
import logging


def _is_discrete(data: pd.DataFrame, max_unique: int = 10) -> bool:
    """Check if data appears to be discrete (few unique values per column)."""
    for col in data.columns:
        if data[col].nunique() > max_unique:
            return False
    return True


def _fit_ordering(
    X: np.ndarray,
    order: np.ndarray,
    lambda1: float,
    lambda2: float,
    max_iter: int,
) -> Tuple[np.ndarray, float]:
    """Fit edge weights for a given variable ordering and compute BIC.
    
    Returns (W, bic) where W is the weighted adjacency matrix.
    """
    n, d = X.shape
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

    return W, bic


def _select_lambda(
    X: np.ndarray,
    rng: np.random.Generator,
    lambda_candidates: List[float],
    lambda2: float,
    max_iter: int,
    edge_threshold: float,
    n_cv_restarts: int = 5,
) -> float:
    """Select lambda1 using modified BIC that penalizes edge count.
    
    Standard BIC tends to favor dense graphs. We add an extra penalty
    proportional to edge count to encourage sparser solutions, which
    aligns better with causal discovery goals.
    """
    n, d = X.shape
    best_lambda = lambda_candidates[len(lambda_candidates) // 2]  # default to middle
    best_score = np.inf

    for lam in lambda_candidates:
        scores = []
        for _ in range(n_cv_restarts):
            order = rng.permutation(d)
            W, bic = _fit_ordering(X, order, lam, lambda2, max_iter)
            # Count edges above threshold
            n_edges = np.sum(np.abs(W) > edge_threshold)
            # Expected edges for a sparse DAG: roughly 1-2 edges per node
            expected_edges = d * 1.5
            excess_edges = max(0, n_edges - expected_edges)
            # Strong penalty for excess edges
            sparsity_penalty = excess_edges * np.log(n) * 5
            score = bic + sparsity_penalty
            scores.append(score)
        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_lambda = lam

    return best_lambda


def run(
    data: pd.DataFrame,
    lambda1: Optional[float] = None,
    lambda2: float = 0.1,
    max_iter: int = 1000,
    seed: int = 0,
    n_restarts: int = 25,
    edge_threshold: float = 0.05,
    auto_lambda: bool = True,
    **_,
) -> Tuple[nx.DiGraph, Dict[str, object]]:
    """Simple COSMO-style learner using priority-based linear regression.

    This implementation searches over random variable orderings and uses
    Lasso regression to identify parent sets. The best ordering by BIC
    is returned.

    Parameters
    ----------
    data : pd.DataFrame
        Observational data with numeric values.
    lambda1 : float, optional
        L1 regularisation strength. When >0, Lasso regression is used.
        If None and auto_lambda=True, lambda is selected automatically.
        Default for discrete data: 0.3, for continuous: 0.1.
    lambda2 : float, optional
        L2 regularisation strength for ridge regression (used when lambda1=0).
    max_iter : int, optional
        Number of iterations for the Lasso optimizer.
    seed : int, optional
        RNG seed controlling the random priority ordering.
    n_restarts : int, optional
        Number of random orderings to try (default 25); the best by BIC is returned.
        More restarts improve quality but increase runtime.
    edge_threshold : float, optional
        Minimum absolute weight to include an edge (default 0.05).
        Helps filter spurious weak edges.
    auto_lambda : bool, optional
        If True and lambda1 is None, automatically select lambda1 using BIC.
    """
    logger = logging.getLogger("benchmark")
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    X = data.values
    n, d = X.shape
    discrete = _is_discrete(data)

    # Determine lambda1
    if lambda1 is None:
        if auto_lambda:
            # Auto-select lambda using modified BIC with sparsity penalty
            if discrete:
                # Discrete data needs stronger regularization
                candidates = [0.2, 0.25, 0.3, 0.35, 0.4]
            else:
                candidates = [0.1, 0.15, 0.2, 0.25, 0.3]
            lambda1 = _select_lambda(X, rng, candidates, lambda2, max_iter, edge_threshold)
            logger.info("COSMO auto-selected lambda1=%.3f (discrete=%s)", lambda1, discrete)
        else:
            # Use sensible defaults based on data type
            lambda1 = 0.3 if discrete else 0.15

    logger.info(
        "COSMO start: n=%d d=%d lambda1=%.3f lambda2=%.3f max_iter=%d seed=%d restarts=%d discrete=%s",
        n, d, lambda1, lambda2, max_iter, seed, n_restarts, discrete,
    )

    best_bic = np.inf
    best_W: np.ndarray | None = None
    best_order: np.ndarray | None = None

    for _ in range(max(n_restarts, 1)):
        order = rng.permutation(d)
        W, bic = _fit_ordering(X, order, lambda1, lambda2, max_iter)

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
            if abs(best_W[i, j]) > edge_threshold:
                G.add_edge(cols[i], cols[j], weight=best_W[i, j])

    if not nx.is_directed_acyclic_graph(G):
        raise RuntimeError("COSMO produced a cyclic graph")

    logger.info(
        "COSMO end: edges=%d runtime_s=%.3f bic=%.3f lambda1=%.3f",
        G.number_of_edges(), runtime, float(best_bic), lambda1,
    )
    return G, {
        "runtime_s": runtime,
        "weights": best_W,
        "ordering": [cols[i] for i in best_order],
        "bic": float(best_bic),
        "lambda1": lambda1,
        "discrete": discrete,
    }

