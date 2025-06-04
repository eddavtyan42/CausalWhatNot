import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import networkx as nx
from metrics.metrics import shd, precision_recall_f1


def test_shd_precision_recall():
    true = nx.DiGraph([(0, 1), (1, 2)])
    pred = nx.DiGraph([(0, 1), (2, 1)])
    assert shd(pred, true, cpdag_mode=True) == 1
    metrics = precision_recall_f1(pred, true)
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1
