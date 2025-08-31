import argparse
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Allow running as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from algorithms import pc, ges, notears, cosmo
from utils import create_mis_specified_graph
from utils.loaders import load_dataset, is_discrete
from utils.helpers import edge_differences
from metrics.metrics import precision_recall_f1, shd
from experiments.phase3_scenarios import SCENARIOS
from causallearn.utils.cit import FisherZ, Chisq_or_Gsq


def compare_graphs(pred: nx.DiGraph, ref: nx.DiGraph):
    """Compute comparison metrics and edge differences.

    Returns
    -------
    tuple
        ``(metrics, extra, missing, reversed_edges)`` where ``metrics`` is a
        dictionary containing precision/recall/F1 and SHD, and the remaining
        elements are sets of edge tuples.
    """

    metrics = precision_recall_f1(pred, ref)
    metrics["shd"] = shd(pred, ref)
    extra, missing, reversed_edges = edge_differences(pred, ref)
    return metrics, extra, missing, reversed_edges


def run(sample_size: int | None = None, bootstrap: int = 0):
    """Run sensitivity analysis across predefined scenarios.

    Parameters
    ----------
    sample_size:
        Optional number of samples to load from each dataset.
    bootstrap:
        Number of bootstrap resamples per algorithm. If 0, a single run on the
        original data is executed.
    """
    rows: list[dict] = []
    for dataset, scenario in SCENARIOS.items():
        data, true_graph = load_dataset(
            dataset, n_samples=sample_size, force=sample_size is not None
        )

        # Build mis-specified analyst graph by removing a key edge and adding a spurious edge
        analyst_graph = true_graph
        missing_u, missing_v = scenario["missing"]["edge"]
        spurious_u, spurious_v = scenario["spurious"]["edge"]
        analyst_graph = create_mis_specified_graph(
            analyst_graph, "missing", missing_u, missing_v
        )
        analyst_graph = create_mis_specified_graph(
            analyst_graph, "spurious", spurious_u, spurious_v
        )

        # Baseline: analyst vs truth
        metrics, extra, missing, reversed_edges = compare_graphs(
            analyst_graph, true_graph
        )
        rows.append(
            {
                "dataset": dataset,
                "method": "analyst",
                "reference": "truth",
                **metrics,
                "extra": sorted(list(extra)),
                "missing": sorted(list(missing)),
                "reversed": sorted(list(reversed_edges)),
            }
        )

        # CI tests for scenario edges
        discrete = is_discrete(data)
        alpha = 0.05
        column_index = {c: i for i, c in enumerate(data.columns)}
        for edge_type, edge in ("missing", scenario["missing"]["edge"]), (
            "spurious",
            scenario["spurious"]["edge"],
        ):
            u, v = edge
            parents_v = set(true_graph.predecessors(v)) - {u}
            cond_idx = [column_index[p] for p in parents_v]
            u_idx = column_index[u]
            v_idx = column_index[v]
            if discrete:
                tester = Chisq_or_Gsq(data.values, method_name="chisq")
                p_val = tester(u_idx, v_idx, tuple(cond_idx))
                if parents_v:
                    total_stat = 0.0
                    total_df = 0
                    for _, g in data.groupby(list(parents_v)):
                        table = pd.crosstab(g[u], g[v])
                        if table.shape[0] > 1 and table.shape[1] > 1:
                            stat, _, dof, _ = chi2_contingency(table, correction=False)
                            total_stat += stat
                            total_df += dof
                    stat = total_stat
                else:
                    table = pd.crosstab(data[u], data[v])
                    stat, _, _, _ = chi2_contingency(table, correction=False)
                test_name = "chi_square"
            else:
                tester = FisherZ(data.values)
                p_val = tester(u_idx, v_idx, tuple(cond_idx))
                var = [u_idx, v_idx] + cond_idx
                sub_corr = np.corrcoef(data.values[:, var].T)
                inv = np.linalg.inv(sub_corr)
                r = -inv[0, 1] / np.sqrt(abs(inv[0, 0] * inv[1, 1]))
                if abs(r) >= 1:
                    r = (1 - np.finfo(float).eps) * np.sign(r)
                Z = 0.5 * np.log((1 + r) / (1 - r))
                stat = np.sqrt(len(data) - len(cond_idx) - 3) * abs(Z)
                test_name = "fisher_z"
            rows.append(
                {
                    "dataset": dataset,
                    "method": "ci_test",
                    "edge_type": edge_type,
                    "edge": [u, v],
                    "ci_test": test_name,
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "reject": bool(p_val < alpha),
                }
            )

        algorithms = {
            "pc": pc.run,
            "ges": ges.run,
            "notears": notears.run,
            "cosmo": cosmo.run,
        }
        for name, func in algorithms.items():
            n_runs = bootstrap if bootstrap > 0 else 1
            for b in range(n_runs):
                d_run = (
                    data.sample(len(data), replace=True, random_state=b)
                    if bootstrap > 0
                    else data
                )
                try:
                    graph, info = func(d_run.copy())
                    runtime = info.get("runtime_s", float("nan"))
                except Exception as e:  # pragma: no cover - safeguard against optional deps
                    graph = nx.DiGraph()
                    graph.add_nodes_from(data.columns)
                    runtime = float("nan")

                # Algorithm vs truth
                metrics_t, extra_t, missing_t, reversed_t = compare_graphs(
                    graph, true_graph
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "method": name,
                        "reference": "truth",
                        "bootstrap": b,
                        "runtime_s": runtime,
                        **metrics_t,
                        "extra": sorted(list(extra_t)),
                        "missing": sorted(list(missing_t)),
                        "reversed": sorted(list(reversed_t)),
                    }
                )

                # Algorithm vs analyst
                metrics_a, extra_a, missing_a, reversed_a = compare_graphs(
                    graph, analyst_graph
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "method": name,
                        "reference": "analyst",
                        "bootstrap": b,
                        "runtime_s": runtime,
                        **metrics_a,
                        "extra": sorted(list(extra_a)),
                        "missing": sorted(list(missing_a)),
                        "reversed": sorted(list(reversed_a)),
                    }
                )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Phase 3 sensitivity analysis")
    parser.add_argument("--n-samples", type=int, default=None, help="Sample size for each dataset")
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap runs")
    parser.add_argument("--out", type=str, default=None, help="Output path for JSON results")
    args = parser.parse_args()

    results = run(sample_size=args.n_samples, bootstrap=args.bootstrap)
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
