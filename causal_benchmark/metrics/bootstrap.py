from typing import Dict, Tuple
import pandas as pd
import networkx as nx
import numpy as np
from joblib import Parallel, delayed


def bootstrap_edge_stability(learn_fn, data_df: pd.DataFrame, b: int = 100, seed: int = 0, n_jobs: int = -1) -> Dict[Tuple[str,str], float]:
    rng = np.random.default_rng(seed)
    max_seed = np.iinfo(np.int32).max
    if b > max_seed:
        raise ValueError(f"Number of bootstrap samples {b} exceeds available unique seeds")
    seeds = rng.choice(max_seed, size=b, replace=False)

    def single_run(idx, seed_value):
        sample = data_df.sample(frac=1.0, replace=True, random_state=int(seed_value))
        g, _ = learn_fn(sample)
        edges = set(g.edges())
        return edges

    results = Parallel(n_jobs=n_jobs)(
        delayed(single_run)(idx, rs) for idx, rs in enumerate(seeds)
    )
    counts = {}
    for edges in results:
        for e in edges:
            counts[e] = counts.get(e, 0) + 1
    return {e: c / b for e, c in counts.items()}
