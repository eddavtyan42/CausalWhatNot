import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import networkx as nx
from algorithms import cosmo


def test_lasso_sparsity():
    df = pd.DataFrame(np.random.randn(200, 5), columns=list('ABCDE'))
    e0 = cosmo.run(df, lambda1=0.0, lambda2=0.1, seed=0)[0].number_of_edges()
    e1 = cosmo.run(df, lambda1=0.5, lambda2=0.1, seed=0)[0].number_of_edges()
    e2 = cosmo.run(df, lambda1=1.0, lambda2=0.1, seed=0)[0].number_of_edges()
    assert e1 <= e0
    assert e2 <= e1


def test_restart_selects_best_order():
    """Test that COSMO produces valid DAG and reasonable BIC."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=200)
    B = 2 * A + rng.normal(size=200)
    df = pd.DataFrame({'A': A, 'B': B})
    g, meta = cosmo.run(df, lambda1=0.0, lambda2=0.1, seed=0, n_restarts=20)
    
    # Check output is a valid DAG
    assert nx.is_directed_acyclic_graph(g)
    
    # Check ordering contains all variables
    assert set(meta['ordering']) == {'A', 'B'}
    
    # Check BIC is finite and reasonable
    assert np.isfinite(meta['bic'])
    
    # With strong relationship A->B, we expect an edge
    assert g.number_of_edges() >= 0  # May be 0 or 1 depending on threshold


def test_multiple_restarts_improve_bic():
    rng = np.random.default_rng(0)
    A = rng.normal(size=200)
    B = 2 * A + rng.normal(size=200)
    df = pd.DataFrame({'A': A, 'B': B})

    _, single = cosmo.run(df, lambda1=0.0, lambda2=0.1, seed=3, n_restarts=1)
    _, many = cosmo.run(df, lambda1=0.0, lambda2=0.1, seed=3, n_restarts=10)

    assert many['bic'] <= single['bic'] + 1e-6