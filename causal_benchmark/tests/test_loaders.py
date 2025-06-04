import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.loaders import load_dataset


def test_asia_loader():
    df, G = load_dataset("asia", n_samples=500, force=True)
    assert df.shape == (500, len(G.nodes()))
    assert len(G.nodes()) == 8
    assert len(G.edges()) == 8


def test_sachs_loader():
    df, G = load_dataset("sachs", n_samples=500, force=True)
    assert df.shape == (500, len(G.nodes()))
    assert len(G.nodes()) == 11
    assert len(G.edges()) == 17


def test_alarm_loader():
    df, G = load_dataset("alarm", n_samples=1000, force=True)
    assert df.shape == (1000, len(G.nodes()))
    assert len(G.nodes()) == 37


def test_child_loader():
    df, G = load_dataset("child", n_samples=1000, force=True)
    assert df.shape == (1000, len(G.nodes()))
    assert len(G.nodes()) == 20

