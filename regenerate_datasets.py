#!/usr/bin/env python3
"""
Regenerate benchmark datasets with fixes applied.
Run this after code changes to data generation procedures.
"""

import sys
from pathlib import Path

# Add causal_benchmark to path
sys.path.insert(0, str(Path(__file__).parent / "causal_benchmark"))

from utils.loaders import load_dataset
import pandas as pd


def regenerate_all_datasets(n_samples=1000):
    """Regenerate all benchmark datasets with current generation code."""
    datasets = ["asia", "sachs", "alarm", "child", "insurance"]
    
    print("=" * 70)
    print("REGENERATING BENCHMARK DATASETS")
    print("=" * 70)
    print()
    
    for name in datasets:
        print(f"Regenerating {name}...")
        data, graph = load_dataset(name, n_samples=n_samples, force=True)
        
        print(f"  ✓ {name}: {data.shape[0]} samples, {data.shape[1]} variables, {graph.number_of_edges()} edges")
        
        # Check data properties
        if name == "sachs":
            # Should be continuous and standardized
            means = data.mean()
            stds = data.std()
            print(f"    Mean range: [{means.min():.3f}, {means.max():.3f}] (should be near 0)")
            print(f"    Std range:  [{stds.min():.3f}, {stds.max():.3f}] (should be near 1)")
            
            if abs(means.abs().max()) > 0.1 or abs(stds.mean() - 1.0) > 0.1:
                print(f"    ⚠ WARNING: Sachs not properly standardized!")
        else:
            # Should be discrete
            print(f"    Data type: {data.dtypes[0]}")
            nunique = data.nunique()
            print(f"    Unique values per column: {nunique.min()}-{nunique.max()}")
            
            # Check for uniform marginals
            for col in data.columns[:3]:  # Check first 3 columns
                value_counts = data[col].value_counts()
                balance = value_counts.std() / value_counts.mean()
                if balance > 0.05:
                    print(f"    ⚠ WARNING: Column {col} not uniformly distributed (balance={balance:.3f})")
        
        print()
    
    print("=" * 70)
    print("REGENERATION COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review the statistics above to verify data properties")
    print("2. Run: python causal_benchmark/experiments/run_benchmark.py")
    print("3. Compare results using the checklist in DATA_REGENERATION_CHECKLIST.md")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate benchmark datasets")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples per dataset")
    args = parser.parse_args()
    
    regenerate_all_datasets(n_samples=args.n_samples)
