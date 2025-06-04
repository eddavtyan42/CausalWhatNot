import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import networkx as nx
from metrics.bootstrap import bootstrap_edge_stability


def dummy_algo(df: pd.DataFrame):
    g = nx.DiGraph()
    g.add_nodes_from(df.columns)
    cols = list(df.columns)
    for i in range(len(cols)-1):
        g.add_edge(cols[i], cols[i+1])
    return g, {}


def test_bootstrap_edge_stability():
    df = pd.DataFrame({"A": [0, 1, 1], "B": [0, 1, 0], "C": [1, 0, 1]})
    freqs = bootstrap_edge_stability(dummy_algo, df, b=5, seed=0, n_jobs=1)
    assert freqs.get(("A", "B")) == 1.0
    assert freqs.get(("B", "C")) == 1.0
