import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import networkx as nx
from metrics.bootstrap import bootstrap_edge_stability
from experiments import run_benchmark
from utils.loaders import load_dataset
import yaml
import pytest


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


def test_bootstrap_edge_stability_parallel_diversity():
    df = pd.DataFrame({"A": range(5)})

    def sample_signature_algo(sample_df: pd.DataFrame):
        order = tuple(sample_df.index.tolist())
        g = nx.DiGraph()
        g.add_nodes_from(["source", "sink"])
        signature_node = f"sample_{order}"
        g.add_node(signature_node)
        g.add_edge("source", signature_node)
        return g, {}

    freqs = bootstrap_edge_stability(sample_signature_algo, df, b=8, seed=123, n_jobs=2)
    sample_edges = [edge for edge in freqs if edge[0] == "source"]
    assert len(sample_edges) > 1


@pytest.mark.timeout(30)
def test_record_edge_stability_benchmark(tmp_path):
    cfg = {
        'datasets': [{'name': 'asia', 'n_samples': 100}],
        'algorithms': {'ges': {}},
        'bootstrap_runs': 2,
        'record_edge_stability': True,
    }
    cfg_path = tmp_path / 'cfg.yaml'
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f)

    load_dataset('asia', n_samples=100, force=True)

    run_benchmark.run(str(cfg_path), output_dir=tmp_path)

    stab_file = tmp_path / 'logs' / 'asia_ges_stability.csv'
    assert stab_file.exists()
    df = pd.read_csv(stab_file)
    assert df['frequency'].between(0, 1).all()
