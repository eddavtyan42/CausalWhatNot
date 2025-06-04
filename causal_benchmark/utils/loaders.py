from pathlib import Path
from typing import Tuple
import pandas as pd
import networkx as nx
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1] / 'data'


def load_dataset(name: str, n_samples: int = 10000, force: bool = False) -> Tuple[pd.DataFrame, nx.DiGraph]:
    data_dir = BASE_DIR / name
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / f"{name}_data.csv"
    edges_path = data_dir / f"{name}_edges.txt"

    if data_path.exists() and not force:
        df = pd.read_csv(data_path)
    else:
        if name.lower() == 'asia':
            rng = np.random.default_rng(0)
            A = rng.normal(size=n_samples)
            B = A + rng.normal(size=n_samples)
            C = B + A + rng.normal(size=n_samples)
            df = pd.DataFrame({'A': A, 'B': B, 'C': C})
        else:
            raise ValueError(f"Unknown dataset {name}")
        df.to_csv(data_path, index=False)
        with open(edges_path, 'w') as f:
            f.write('A,B\nA,C\nB,C\n')

    G = nx.DiGraph()
    with open(edges_path) as f:
        for line in f:
            u, v = line.strip().split(',')
            G.add_edge(u, v)
    return df, G
