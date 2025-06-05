import time
from functools import wraps
from typing import Callable, Dict, Set, Tuple
import networkx as nx
import numpy as np


def runtime_sec(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return wrapper


def shd(
    pred_graph: nx.DiGraph,
    true_graph: nx.DiGraph,
    cpdag_mode: bool = True,
    pred_undirected: Set[Tuple[str, str]] | None = None,
) -> int:
    nodes = list(true_graph.nodes())
    adj_pred = nx.to_numpy_array(pred_graph, nodelist=nodes)
    adj_true = nx.to_numpy_array(true_graph, nodelist=nodes)
    if cpdag_mode:
        pred_ug = ((adj_pred + adj_pred.T) > 0).astype(int)
        true_ug = ((adj_true + adj_true.T) > 0).astype(int)
        skeleton_diff = int(((pred_ug != true_ug).sum()) // 2)
        orient_mism = 0
        pred_undirected = pred_undirected or set()
        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                if pred_ug[i, j] and true_ug[i, j]:
                    if adj_pred[i, j] != adj_true[i, j] and adj_pred[j, i] != adj_true[j, i]:
                        node_i, node_j = nodes[i], nodes[j]
                        if (node_i, node_j) not in pred_undirected and (node_j, node_i) not in pred_undirected:
                            orient_mism += 1
        return skeleton_diff + orient_mism
    else:
        return int((adj_pred != adj_true).sum())


def precision_recall_f1(pred_graph: nx.DiGraph, true_graph: nx.DiGraph, undirected_ok: bool = True) -> Dict[str, float]:
    nodes = list(true_graph.nodes())
    adj_pred = nx.to_numpy_array(pred_graph, nodelist=nodes)
    adj_true = nx.to_numpy_array(true_graph, nodelist=nodes)
    if undirected_ok:
        pred_edges = ((adj_pred + adj_pred.T) > 0).astype(int)
        true_edges = ((adj_true + adj_true.T) > 0).astype(int)
    else:
        pred_edges = adj_pred
        true_edges = adj_true
    tp = np.sum((pred_edges == 1) & (true_edges == 1))
    fp = np.sum((pred_edges == 1) & (true_edges == 0))
    fn = np.sum((pred_edges == 0) & (true_edges == 1))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def directed_precision_recall_f1(
    pred_graph: nx.DiGraph, true_graph: nx.DiGraph
) -> Dict[str, float]:
    """Precision/recall/F1 considering edge orientation."""
    base = precision_recall_f1(pred_graph, true_graph, undirected_ok=False)
    return {
        "directed_precision": base["precision"],
        "directed_recall": base["recall"],
        "directed_f1": base["f1"],
    }


def shd_dir(pred_graph: nx.DiGraph, true_graph: nx.DiGraph) -> int:
    """Orientation-sensitive structural Hamming distance."""
    return shd(pred_graph, true_graph, cpdag_mode=False)
