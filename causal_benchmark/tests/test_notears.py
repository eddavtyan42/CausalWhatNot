
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'causal_benchmark'))
import pandas as pd
import networkx as nx
from causal_benchmark.algorithms.notears import run
from causal_benchmark.utils.loaders import load_dataset

print(f"Python version: {sys.version}")

try:
    data, _ = load_dataset("sachs", n_samples=100)
    print("Loaded sachs data")
    graph, info = run(data)
    print("NOTEARS run successful")
    print(f"Edges: {graph.number_of_edges()}")
except Exception as e:
    print(f"NOTEARS failed: {e}")
