import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import networkx as nx
import pytest

from utils.loaders import load_dataset
from algorithms import pc, ges, notears
from metrics.metrics import shd


algorithms_to_test = [pc, ges]
if notears is not None:
    algorithms_to_test.append(notears)

@pytest.mark.parametrize('algo_module', algorithms_to_test)
def test_algorithms_asia(algo_module):
    data, true_graph = load_dataset('asia', n_samples=1000, force=True)
    pred_graph, _ = algo_module.run(data)
    assert shd(pred_graph, true_graph) <= 2
