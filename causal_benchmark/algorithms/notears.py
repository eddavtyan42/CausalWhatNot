"""NOTEARS causal discovery algorithm via the CausalNex backend."""

from __future__ import annotations

__all__ = ["run"]

import sys
import time
from typing import Tuple, Dict, Optional

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


def run(
    data: pd.DataFrame,
    threshold: Optional[float] = None,
    torch_seed: int | None = None,
    standardize: Optional[bool] = None,
    **kwargs,
) -> Tuple[nx.DiGraph, Dict[str, object]]:
    """Run NOTEARS on a dataframe.

    NOTEARS assumes linear-Gaussian relationships, so it works best on
    continuous data. For discrete data, results may be less reliable.

    Parameters
    ----------
    data : pd.DataFrame
        Numeric dataframe containing observational samples.
    threshold : float, optional
        Edge weight threshold. If None, automatically selected based on
        data type: 0.1 for continuous, 0.25 for discrete data.
    torch_seed : int, optional
        If provided, sets the PyTorch RNG seed and enables deterministic
        operations for reproducible results.
    standardize : bool, optional
        If True, standardize the data before running NOTEARS. 
        If None (default), auto-selects: True for discrete, False for continuous.
        Standardization helps discrete data but can hurt continuous data.

    Other Parameters
    ----------------
    max_iter : int, optional
        Maximum number of iterations (default from causalnex).

    Returns
    -------
    nx.DiGraph
        Estimated directed acyclic graph.
    Dict[str, object]
        Dictionary with keys ``runtime_s`` (float) and ``weights`` (np.ndarray) containing the learned
        weighted adjacency matrix.
    
    Notes
    -----
    NOTEARS was designed for continuous data with linear relationships.
    On discrete/binary data:
    - Use higher thresholds (0.25-0.35) to reduce false positives
    - Results represent linear approximations, not true causal effects
    - Consider using PC or GES instead for discrete data
    """

    logger = logging.getLogger("benchmark")
    
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

    # Detect data type
    discrete = _is_discrete(data)
    
    # Auto-select threshold based on data type if not specified
    if threshold is None:
        threshold = 0.25 if discrete else 0.1
        logger.info("NOTEARS auto-selected threshold=%.2f (discrete=%s)", threshold, discrete)
    
    # Auto-select standardization: helps discrete, hurts continuous
    if standardize is None:
        standardize = discrete
        logger.info("NOTEARS auto-selected standardize=%s (discrete=%s)", standardize, discrete)

    logger.info(
        "NOTEARS start: n=%d d=%d threshold=%.3f torch_seed=%s discrete=%s standardize=%s",
        len(data), data.shape[1], threshold, str(torch_seed), discrete, standardize
    )

    if torch_seed is not None:
        try:
            import torch
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "PyTorch is required to set torch_seed"
            ) from e
        torch.manual_seed(torch_seed)
        torch.use_deterministic_algorithms(True)

    # Optionally standardize data for better numerical stability
    data_processed = data
    if standardize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_processed = pd.DataFrame(
            scaler.fit_transform(data),
            columns=data.columns
        )

    # Ignore unsupported legacy parameter if present
    kwargs.pop("backend", None)

    start = time.perf_counter()
    # Allow optional policy to handle rare cyclic outputs from backend
    cycle_policy = kwargs.pop("cycle_policy", "repair")  # one of {"repair", "raise"}

    sm = from_pandas(data_processed, w_threshold=threshold, **kwargs)
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

    meta: Dict[str, object] = {
        "runtime_s": runtime, 
        "weights": W,
        "threshold": threshold,
        "discrete": discrete,
        "standardized": standardize,
    }

    # Very rarely backend can yield cycles (due to thresholding/rounding).
    # Respect policy: either raise or repair by removing weakest edges until DAG.
    if not nx.is_directed_acyclic_graph(G):
        if cycle_policy == "raise":
            raise RuntimeError("NOTEARS produced a cyclic graph")
        # Repair: iteratively remove the smallest-absolute-weight edge among all detected cycles
        removed: list[tuple[str, str, float]] = []
        # Cap iterations to number of edges to avoid infinite loops
        max_iters = max(1, G.number_of_edges())
        iters = 0
        while not nx.is_directed_acyclic_graph(G) and iters < max_iters:
            cycles = list(nx.simple_cycles(G))
            if not cycles:
                break
            candidates: list[tuple[float, tuple[str, str]]] = []
            for cyc in cycles:
                cyc_edges = list(zip(cyc, cyc[1:] + [cyc[0]]))
                for (u, v) in cyc_edges:
                    w = abs(G[u][v].get("weight", 0.0))
                    candidates.append((w, (u, v)))
            if not candidates:
                break
            _, (u_min, v_min) = min(candidates, key=lambda x: x[0])
            w_min = G[u_min][v_min].get("weight", 0.0)
            G.remove_edge(u_min, v_min)
            removed.append((u_min, v_min, float(w_min)))
            # Zero-out corresponding entry in weight matrix for consistency
            i, j = cols.index(u_min), cols.index(v_min)
            W[i, j] = 0.0
            iters += 1

        meta["cycle_repaired"] = True
        meta["cycles_removed"] = len(removed)
        meta["removed_edges"] = removed

    logger.info(
        "NOTEARS end: edges=%d runtime_s=%.3f threshold=%.3f cycles_fixed=%s", 
        G.number_of_edges(), runtime, threshold, str(meta.get("cycle_repaired", False))
    )
    return G, meta
