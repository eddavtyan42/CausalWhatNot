#!/usr/bin/env python
"""Quick test to verify all algorithms work on all datasets."""
import sys
import os
os.environ['TQDM_DISABLE'] = '1'  # Disable tqdm progress bars
sys.path.insert(0, '.')
from utils.loaders import load_dataset
import algorithms.pc as pc
import algorithms.ges as ges
import algorithms.notears as notears
import algorithms.cosmo as cosmo
import networkx as nx
from metrics.metrics import shd, precision_recall_f1
import warnings
warnings.filterwarnings('ignore')

datasets = ['asia', 'sachs', 'alarm', 'child', 'insurance']
algos = [
    ('PC', pc.run),
    ('GES', ges.run),
    ('NOTEARS', notears.run),
    ('COSMO', cosmo.run)
]

results = []

for ds_name in datasets:
    data, true_dag = load_dataset(ds_name, n_samples=500)
    true_edges = true_dag.number_of_edges()
    
    for algo_name, algo_fn in algos:
        try:
            pred_dag, meta = algo_fn(data)
            pred_edges = pred_dag.number_of_edges()
            is_dag = nx.is_directed_acyclic_graph(pred_dag)
            
            if is_dag:
                shd_val = shd(true_dag, pred_dag)
                metrics = precision_recall_f1(true_dag, pred_dag)
                f = metrics['f1']
                results.append((ds_name, algo_name, true_edges, pred_edges, shd_val, f, 'OK'))
            else:
                results.append((ds_name, algo_name, true_edges, pred_edges, -1, 0.0, 'CYCLIC'))
        except Exception as e:
            err = str(e).replace('\n', ' ')[:20]
            results.append((ds_name, algo_name, true_edges, '?', '?', '?', 'ERR:' + err))

# Print clean results
print()
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"{'Dataset':<12} {'Algo':<10} {'True':>5} {'Pred':>5} {'SHD':>5} {'F1':>6} {'Status':<15}")
print("-" * 70)
for r in results:
    ds, algo, true_e, pred_e, s, f, status = r
    if status == 'OK':
        print(f"{ds:<12} {algo:<10} {int(true_e):5d} {int(pred_e):5d} {int(s):5d} {float(f):6.2f} {status:<15}")
    elif status == 'CYCLIC':
        print(f"{ds:<12} {algo:<10} {int(true_e):5d} {int(pred_e):5d}    -1   0.00 {status:<15}")
    else:
        print(f"{ds:<12} {algo:<10} {int(true_e):5d}     ?     ?      ? {status:<15}")
print("=" * 70)
