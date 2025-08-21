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
    torch_seed: int | None = None,
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
    torch_seed : int, optional
        If provided, sets the PyTorch RNG seed and enables deterministic
        operations for reproducible results.

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

    if torch_seed is not None:
        try:
            import torch
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "PyTorch is required to set torch_seed"
            ) from e
        torch.manual_seed(torch_seed)
        torch.use_deterministic_algorithms(True)

    # Ignore unsupported legacy parameter if present
    kwargs.pop("backend", None)

    start = time.perf_counter()
    # Allow optional policy to handle rare cyclic outputs from backend
    cycle_policy = kwargs.pop("cycle_policy", "repair")  # one of {"repair", "raise"}

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

    meta: Dict[str, object] = {"runtime_s": runtime, "weights": W}

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

    return G, meta
