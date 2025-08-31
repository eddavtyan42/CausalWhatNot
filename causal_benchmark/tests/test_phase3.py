import os
import sys

import networkx as nx
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import create_mis_specified_graph
from experiments.phase3_sensitivity_analysis import compare_graphs


def test_create_mis_specified_graph_missing():
    g = nx.DiGraph([('A', 'B'), ('B', 'C')])
    mis = create_mis_specified_graph(g, 'missing', 'A', 'B')
    assert mis.number_of_edges() == g.number_of_edges() - 1
    assert not mis.has_edge('A', 'B')


def test_create_mis_specified_graph_spurious():
    g = nx.DiGraph([('A', 'B'), ('B', 'C')])
    mis = create_mis_specified_graph(g, 'spurious', 'A', 'C')
    assert mis.number_of_edges() == g.number_of_edges() + 1
    assert mis.has_edge('A', 'C')
    assert nx.is_directed_acyclic_graph(mis)


def test_compare_graphs_metrics_and_diffs():
    true_graph = nx.DiGraph([('A', 'B'), ('B', 'C')])
    pred_graph = nx.DiGraph([('A', 'B'), ('C', 'B')])
    metrics, extra, missing, reversed_edges = compare_graphs(pred_graph, true_graph)
    assert metrics['precision'] == pytest.approx(1.0)
    assert metrics['recall'] == pytest.approx(1.0)
    assert metrics['f1'] == pytest.approx(1.0)
    assert metrics['shd'] == 1
    assert extra == set()
    assert missing == set()
    assert reversed_edges == {('C', 'B')}
