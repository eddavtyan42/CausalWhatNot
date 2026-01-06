import time
from functools import wraps
from typing import Callable, Dict, Set, Tuple
import networkx as nx
import numpy as np
import signal
import logging

try:
    from causaldag import DAG as CausalDAG
    from causaldag import structural_intervention_distance as compute_sid
except ImportError:
    CausalDAG = None
    compute_sid = None


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
    # Use weight=None to get binary adjacency (handles weighted graphs like NOTEARS)
    adj_pred = nx.to_numpy_array(pred_graph, nodelist=nodes, weight=None)
    adj_true = nx.to_numpy_array(true_graph, nodelist=nodes, weight=None)
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
    # Use weight=None to get binary adjacency (handles weighted graphs like NOTEARS)
    adj_pred = nx.to_numpy_array(pred_graph, nodelist=nodes, weight=None)
    adj_true = nx.to_numpy_array(true_graph, nodelist=nodes, weight=None)
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


def sid(
    pred_graph: nx.DiGraph,
    true_graph: nx.DiGraph,
    has_undirected_edges: bool = False,
    timeout_seconds: float = 30.0,
) -> float:
    """
    Compute Structural Intervention Distance (SID) between predicted and true DAGs.
    
    SID measures the number of interventional distribution mismatches between two
    causal graphs. Unlike SHD (which counts structural edits), SID quantifies how
    many causal effect queries would yield different answers under the two graphs.
    
    A graph with low SHD but high SID indicates errors on critical confounding paths
    that would lead to incorrect causal effect estimates.
    
    Parameters
    ----------
    pred_graph : nx.DiGraph or np.ndarray
        Predicted causal graph (must be a DAG). Can be a networkx DiGraph or adjacency matrix.
    true_graph : nx.DiGraph or np.ndarray
        True causal graph (must be a DAG). Can be a networkx DiGraph or adjacency matrix.
    has_undirected_edges : bool, optional
        If True, returns NaN because SID requires fully directed graphs.
        CPDAGs (with undirected edges) represent equivalence classes where
        interventional predictions are ambiguous.
    timeout_seconds : float, optional
        Maximum computation time in seconds. Returns NaN on timeout.
        Default is 30 seconds for large graphs.
    
    Returns
    -------
    float
        SID value (integer cast to float for consistency with error handling).
        Returns NaN if:
        - has_undirected_edges is True
        - Either graph contains cycles
        - Computation exceeds timeout
    
    References
    ----------
    Peters, J., & Bühlmann, P. (2015). Structural intervention distance (SID)
    for evaluating causal graphs. Neural Computation, 27(3), 771-799.
    
    Notes
    -----
    - SID is symmetric for the lower bound formulation (counts mismatches in ancestor sets).
    - SID is more sensitive to errors near graph roots (affecting many descendants)
      compared to errors on leaf nodes.
    - For large graphs (>30 nodes), SID computation can be slow due to ancestor
      set comparisons.
    - CAUTION: This implementation strictly evaluates the input graph as a DAG.
      If applied to a CPDAG (e.g., from PC/GES) that happens to be represented 
      as a specific DAG instance, it measures the distance for that specific 
      orientation. This should be interpreted as the "analyst's default"  
      performance, not the optimal distance for the entire equivalence class.
    
    Examples
    --------
    >>> import networkx as nx
    >>> true_g = nx.DiGraph([('X', 'Y'), ('Y', 'Z')])
    >>> pred_g = nx.DiGraph([('X', 'Y'), ('Z', 'Y')])  # wrong orientation of Y-Z
    >>> sid(pred_g, true_g)
    2.0  # Both do(Y) and do(Z) queries would differ
    """
    logger = logging.getLogger("benchmark")
    
    # Skip SID for CPDAGs (theoretically inappropriate)
    if has_undirected_edges:
        logger.debug("SID computation skipped: graph contains undirected edges (CPDAG)")
        return float('nan')
    
    # Convert adjacency matrices to networkx if needed
    if isinstance(pred_graph, np.ndarray):
        n = pred_graph.shape[0]
        pred_nx = nx.DiGraph()
        pred_nx.add_nodes_from(range(n))
        for i in range(n):
            for j in range(n):
                if pred_graph[i, j] != 0:
                    pred_nx.add_edge(i, j)
        pred_graph = pred_nx
    
    if isinstance(true_graph, np.ndarray):
        n = true_graph.shape[0]
        true_nx = nx.DiGraph()
        true_nx.add_nodes_from(range(n))
        for i in range(n):
            for j in range(n):
                if true_graph[i, j] != 0:
                    true_nx.add_edge(i, j)
        true_graph = true_nx
    
    # Check for cycles (SID requires DAGs)
    if not nx.is_directed_acyclic_graph(pred_graph):
        logger.warning("SID computation skipped: predicted graph contains cycles")
        return float('nan')
    if not nx.is_directed_acyclic_graph(true_graph):
        logger.warning("SID computation skipped: true graph contains cycles")
        return float('nan')
    
    # Timeout handler
    class TimeoutError(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutError()
    
    use_signal = False
    try:
        # Set alarm for timeout (Unix-like systems only)
        if hasattr(signal, 'SIGALRM'):
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout_seconds))
                use_signal = True
            except ValueError:
                logger.debug("SID timeout protection disabled: signal only works in main thread")
        
        # Get common node set (graphs should have same nodes)
        nodes = list(true_graph.nodes())
        if set(nodes) != set(pred_graph.nodes()):
            logger.warning("SID computation skipped: graphs have different node sets")
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            return float('nan')
        
        # Compute SID: count nodes with different ancestor sets
        # SID = |{v : An_pred(v) != An_true(v)}|
        # where An(v) is the set of ancestors of v
        sid_count = 0
        for node in nodes:
            anc_pred = nx.ancestors(pred_graph, node)
            anc_true = nx.ancestors(true_graph, node)
            if anc_pred != anc_true:
                sid_count += 1
        
        # Cancel alarm
        if use_signal:
            signal.alarm(0)
        
        return float(sid_count)
    
    except TimeoutError:
        logger.warning(
            "SID computation timeout after %.1fs (nodes=%d, edges_pred=%d, edges_true=%d)",
            timeout_seconds,
            pred_graph.number_of_nodes(),
            pred_graph.number_of_edges(),
            true_graph.number_of_edges(),
        )
        if use_signal:
            signal.alarm(0)
        return float('nan')
    
    except Exception as e:
        logger.warning("SID computation failed: %s", str(e))
        if use_signal:
            signal.alarm(0)
        return float('nan')
