import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import networkx as nx
from metrics.metrics import (
    shd,
    precision_recall_f1,
    directed_precision_recall_f1,
    shd_dir,
)


def test_shd_precision_recall():
    true = nx.DiGraph([(0, 1), (1, 2)])
    pred = nx.DiGraph([(0, 1), (2, 1)])
    assert shd(pred, true, cpdag_mode=True) == 1
    metrics = precision_recall_f1(pred, true)
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1


def test_shd_with_undirected_pred():
    true = nx.DiGraph([(0, 1), (1, 2)])
    pred = nx.DiGraph([(0, 1), (2, 1)])
    undirected = {(2, 1)}
    # Orientation of edge (2,1) is uncertain; shd should ignore its direction
    assert shd(pred, true, cpdag_mode=True, pred_undirected=undirected) == 0
    metrics = precision_recall_f1(pred, true, undirected_ok=True)
    assert metrics['precision'] == 1
    assert metrics['recall'] == 1


def test_directed_metrics_orientation_error():
    true = nx.DiGraph([(0, 1)])
    pred = nx.DiGraph([(1, 0)])
    d_metrics = directed_precision_recall_f1(pred, true)
    assert d_metrics['directed_precision'] == 0
    assert d_metrics['directed_recall'] == 0
    assert d_metrics['directed_f1'] == 0
    assert shd_dir(pred, true) == 2
