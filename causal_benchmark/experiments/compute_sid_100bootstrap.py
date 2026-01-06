#!/usr/bin/env python3
"""
Compute SID retroactively for the 100 bootstrap run results.
This script loads each bootstrap run's adjacency matrix, computes SID,
and then aggregates mean and std across all runs.
"""
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
import sys
import logging

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from metrics.metrics import sid
    from utils.loaders import load_dataset
except ImportError:
    sys.path.append(str(Path.cwd() / "causal_benchmark"))
    from metrics.metrics import sid
    from utils.loaders import load_dataset

def main():
    project_root = Path(__file__).resolve().parents[2]
    results_base = project_root / "causal_benchmark" / "results_100btstrp" / "benchmark"
    
    summary_path = results_base / "summary_metrics.csv"
    outputs_dir = results_base / "outputs"
    
    if not summary_path.exists():
        print(f"Error: {summary_path} not found.")
        return

    print(f"Loading summary from {summary_path}")
    df = pd.read_csv(summary_path)
    
    # Add SID columns if they don't exist
    if 'sid' not in df.columns:
        df['sid'] = np.nan
        df['sid_std'] = np.nan

    # Load ground truths once
    ground_truths = {}
    datasets = df['dataset'].unique()
    
    print("Loading ground truth graphs...")
    for ds in datasets:
        try:
            print(f"Loading {ds}...")
            _, true_graph = load_dataset(ds, n_samples=1000)
            ground_truths[ds] = true_graph
        except Exception as e:
            print(f"Failed to load ground truth for {ds}: {e}")

    # Process each dataset-algorithm pair
    for idx, row in df.iterrows():
        ds = row['dataset']
        algo = row['algorithm']
        n_runs = int(row['n_runs'])
        
        print(f"\nProcessing {ds}_{algo} ({n_runs} runs)...")
        
        true_graph = ground_truths.get(ds)
        if true_graph is None:
            print(f"  Skipping: no ground truth for {ds}")
            continue
        
        sid_values = []
        
        # Load each bootstrap run
        for run_idx in range(n_runs):
            adj_path = outputs_dir / f"{ds}_{algo}_run{run_idx}_adj.csv"
            
            if not adj_path.exists():
                print(f"  Warning: {adj_path} not found")
                continue
                
            try:
                # Load prediction
                pred_df = pd.read_csv(adj_path, index_col=0)
                pred_adj = pred_df.values
                pred_nodes = pred_df.columns.tolist()
                
                # Reconstruct NetworkX
                pred_graph = nx.DiGraph()
                pred_graph.add_nodes_from(pred_nodes)
                rows, cols = pred_adj.nonzero()
                for r, c in zip(rows, cols):
                    pred_graph.add_edge(pred_nodes[r], pred_nodes[c])
                
                # Compute SID (bypass has_undirected_edges check)
                val = sid(pred_graph, true_graph, has_undirected_edges=False, timeout_seconds=30.0)
                
                if not np.isnan(val):
                    sid_values.append(val)
                    
            except Exception as e:
                print(f"  Error processing run {run_idx}: {e}")
        
        # Compute statistics
        if len(sid_values) > 0:
            sid_mean = np.mean(sid_values)
            sid_std = np.std(sid_values, ddof=1) if len(sid_values) > 1 else 0.0
            
            df.at[idx, 'sid'] = sid_mean
            df.at[idx, 'sid_std'] = sid_std
            
            print(f"  SID: {sid_mean:.2f} ± {sid_std:.2f} (n={len(sid_values)})")
        else:
            print(f"  No valid SID values computed")

    # Save updated summary
    df.to_csv(summary_path, index=False)
    print(f"\n✓ Updated {summary_path}")

if __name__ == "__main__":
    main()
