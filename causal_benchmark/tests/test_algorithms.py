import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import networkx as nx
import pytest

from utils.loaders import load_dataset
from algorithms import pc, ges, cosmo
try:
    from algorithms import notears
except ImportError:  # causalnex not installed
    notears = None

ALGOS = [pc, ges, cosmo]
if notears is not None:
    ALGOS.append(notears)


@pytest.mark.parametrize('algo_module', ALGOS)
def test_algorithms_asia(algo_module):
    data, true_graph = load_dataset('asia', n_samples=200, force=True)
    pred_graph, _ = algo_module.run(data)
    assert set(pred_graph.nodes()) == set(true_graph.nodes())
    assert nx.is_directed_acyclic_graph(pred_graph)


def test_notears_small():
    if notears is None:
        pytest.skip('causalnex not installed')
    df = pd.DataFrame(np.random.randn(200, 5), columns=[f'x{i}' for i in range(5)])
    g, _ = notears.run(df)
    assert set(g.nodes()) == set(df.columns)
    assert nx.is_directed_acyclic_graph(g)


def test_cosmo_small():
    df = pd.DataFrame(np.random.randn(100, 3), columns=list('ABC'))
    g, _ = cosmo.run(df, seed=0)
    assert set(g.nodes()) == set(df.columns)
    assert nx.is_directed_acyclic_graph(g)
