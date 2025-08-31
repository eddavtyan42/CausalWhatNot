import networkx as nx
import pytest

from utils import create_mis_specified_graph


def test_create_mis_specified_graph_missing():
    g = nx.DiGraph([('A', 'B'), ('B', 'C')])
    mis = create_mis_specified_graph(g, 'missing', 'A', 'B')
    assert set(mis.edges()) == {('B', 'C')}


def test_create_mis_specified_graph_spurious():
    g = nx.DiGraph([('A', 'B')])
    mis = create_mis_specified_graph(g, 'spurious', 'B', 'A')
    assert set(mis.edges()) == {('A', 'B'), ('B', 'A')}


def test_create_mis_specified_graph_errors():
    g = nx.DiGraph([('A', 'B')])
    with pytest.raises(ValueError):
        create_mis_specified_graph(g, 'missing', 'B', 'A')
    with pytest.raises(ValueError):
        create_mis_specified_graph(g, 'spurious', 'A', 'B')
    with pytest.raises(ValueError):
        create_mis_specified_graph(g, 'invalid', 'A', 'B')

