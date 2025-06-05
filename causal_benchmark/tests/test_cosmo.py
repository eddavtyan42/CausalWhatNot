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
    rng = np.random.default_rng(0)
    A = rng.normal(size=200)
    B = 2 * A + rng.normal(size=200)
    df = pd.DataFrame({'A': A, 'B': B})
    g, meta = cosmo.run(df, lambda1=0.0, lambda2=0.1, seed=0, n_restarts=20)
    # compute expected best order via BIC
    n = len(df)
    X = df.values
    def bic(order):
        Xo = X[:, order]
        bicv = 0.0
        for idx in range(len(order)):
            if idx == 0:
                var = Xo[:, 0].var(ddof=1)
                bicv += n * np.log(var + 1e-12)
                continue
            X_par = Xo[:, :idx]
            y = Xo[:, idx]
            w = np.linalg.solve(
                X_par.T @ X_par + 0.1 * np.eye(idx), X_par.T @ y
            )
            res = y - X_par @ w
            bicv += n * np.log(res.var(ddof=1) + 1e-12) + idx * np.log(n)
        return bicv
    bic_ab = bic([0, 1])
    bic_ba = bic([1, 0])
    best = ['A', 'B'] if bic_ab < bic_ba else ['B', 'A']
    assert meta['ordering'] == best
    assert np.isclose(meta['bic'], min(bic_ab, bic_ba))
    assert nx.is_directed_acyclic_graph(g)


def test_multiple_restarts_improve_bic():
    rng = np.random.default_rng(0)
    A = rng.normal(size=200)
    B = 2 * A + rng.normal(size=200)
    df = pd.DataFrame({'A': A, 'B': B})

    _, single = cosmo.run(df, lambda1=0.0, lambda2=0.1, seed=3, n_restarts=1)
    _, many = cosmo.run(df, lambda1=0.0, lambda2=0.1, seed=3, n_restarts=10)

    assert many['bic'] <= single['bic'] + 1e-6

