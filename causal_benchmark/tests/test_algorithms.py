import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import networkx as nx
import pytest

from utils.loaders import load_dataset
from algorithms import pc, ges
from metrics.metrics import shd


@pytest.mark.parametrize('algo_module', [pc, ges])
def test_algorithms_asia(algo_module):
    data, true_graph = load_dataset('asia', n_samples=1000, force=True)
    pred_graph, _ = algo_module.run(data)
    assert shd(pred_graph, true_graph) <= 2
