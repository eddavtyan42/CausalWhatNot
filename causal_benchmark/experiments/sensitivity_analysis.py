import argparse
import json
import time
import yaml
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

# Allow running as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from algorithms import pc, ges, notears, cosmo
from utils import create_mis_specified_graph
from utils.loaders import load_dataset, is_discrete
from utils.helpers import edge_differences
from metrics.metrics import precision_recall_f1, shd
from metrics.bootstrap import bootstrap_edge_stability
from experiments.perturbation_scenarios import SCENARIOS
from causallearn.utils.cit import FisherZ, Chisq_or_Gsq
from utils.provenance import save_run_metadata, save_graph_artifacts


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


def run(
    sample_size: int | None = None,
    bootstrap_runs: int = 0,
    n_jobs: int = -1,
    diff_dir: Path | None = None,
    output_dir: Path | None = None,
    benchmark_dir: Path | None = None,
):
    """Run sensitivity analysis across predefined scenarios.

    Parameters
    ----------
    sample_size:
        Optional number of samples to load from each dataset.
    bootstrap_runs:
        Number of bootstrap resamples per algorithm for edge stability.
    n_jobs:
        Parallel jobs for bootstrap resampling. ``-1`` uses all cores.
    diff_dir:
        Directory to write diff logs.
    output_dir:
        Base directory for outputs (artifacts will be in output_dir/outputs).
    benchmark_dir:
        Optional directory containing existing benchmark results to reuse.
    """
    # Load config
    config_path = Path(__file__).resolve().parents[0] / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Parse dataset configs
    ds_configs = {}
    for ds_cfg in cfg.get("datasets", []):
        if isinstance(ds_cfg, dict):
            ds_configs[ds_cfg.get("name")] = ds_cfg
        elif isinstance(ds_cfg, str):
            ds_configs[ds_cfg] = {"name": ds_cfg}

    # Parse algo configs
    algo_configs = cfg.get("algorithms", {})

    # Setup artifacts directory
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "results_new" / "sensitivity"
    artifacts_dir = output_dir / "outputs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for dataset, scenario in SCENARIOS.items():
        # Determine n_samples from config if not provided via CLI
        n_samples = sample_size
        if n_samples is None:
            ds_cfg = ds_configs.get(dataset, {})
            n_samples = ds_cfg.get("n_samples")

        data, true_graph = load_dataset(
            dataset, n_samples=n_samples, force=True
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
                # Compute log p-value for extreme tails (two-sided test)
                # p = 2 * (1 - CDF(|Z|)) = 2 * SF(|Z|)
                # log(p) = log(2) + log(SF(|Z|))
                log_p_val = np.log(2) + norm.logsf(stat)

            rows.append(
                {
                    "dataset": dataset,
                    "method": "ci_test",
                    "edge_type": edge_type,
                    "edge": [u, v],
                    "ci_test": test_name,
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "log_p_value": float(log_p_val) if test_name == "fisher_z" else np.nan,
                    "reject": bool(p_val < alpha),
                }
            )

        algorithms = {
            "pc": pc.run,
            "ges": ges.run,
            "notears": notears.run,
            "cosmo": cosmo.run,
        }
        key_edge = tuple(scenario["missing"]["edge"])
        for name, func in algorithms.items():
            # Get params from config
            params = algo_configs.get(name, {}).copy()
            # Handle per-dataset overrides
            per_dataset = params.pop("per_dataset", {})
            if dataset in per_dataset:
                params.update(per_dataset[dataset])

            graph = None
            info = {}
            runtime = float("nan")
            
            # Try to load from benchmark artifacts if available
            if benchmark_dir is not None:
                # Look for {dataset}_{name}_run0_edges.json or similar
                # We prefer loading the graph directly.
                # The benchmark saves: outputs/{dataset}_{name}_run0_edges.json
                # Or outputs/{dataset}_{name}_edges.json (if bootstrap=0)
                # But benchmark usually runs with bootstrap > 0, so run0 is standard.
                
                # Try run0 first (standard for bootstrap runs)
                artifact_path = benchmark_dir / "outputs" / f"{dataset}_{name}_run0_edges.json"
                if not artifact_path.exists():
                     # Try non-bootstrap name
                     artifact_path = benchmark_dir / "outputs" / f"{dataset}_{name}_edges.json"
                
                if artifact_path.exists():
                    try:
                        with open(artifact_path, "r") as f:
                            edge_list = json.load(f)
                        graph = nx.DiGraph()
                        graph.add_nodes_from(data.columns)
                        for e in edge_list:
                            graph.add_edge(e["source"], e["target"])
                        # We don't have runtime or info easily available without parsing meta, 
                        # but we don't strictly need them for sensitivity analysis metrics.
                        print(f"Loaded cached graph for {dataset} {name} from {artifact_path}")
                    except Exception as e:
                        print(f"Failed to load cached graph {artifact_path}: {e}")
                        graph = None

            if graph is None:
                try:
                    graph, info = func(data.copy(), **params)
                    runtime = info.get("runtime_s", float("nan"))
                except Exception:  # pragma: no cover - safeguard against optional deps
                    graph = nx.DiGraph()
                    graph.add_nodes_from(data.columns)
                    runtime = float("nan")
                    info = {}

            # Save provenance
            base_filename = f"{dataset}_{name}"
            dataset_path = Path(__file__).resolve().parents[1] / "data" / dataset / f"{dataset}_data.csv"
            
            save_run_metadata(
                output_path=artifacts_dir / f"{base_filename}_meta.json",
                dataset_name=dataset,
                dataset_path=dataset_path,
                n_samples=len(data),
                algorithm_name=name,
                algorithm_params=params,
                random_seed=0, 
                preprocessing_info={"scenario": scenario}
            )
            
            save_graph_artifacts(
                output_dir=artifacts_dir,
                base_filename=base_filename,
                graph=graph,
                nodes=list(data.columns),
                raw_adjacency=None
            )

            # Stability: Try to load from benchmark logs if available
            freqs = {}
            loaded_stability = False
            if benchmark_dir is not None:
                stab_path = benchmark_dir / "logs" / f"{dataset}_{name}_stability.csv"
                if stab_path.exists():
                    try:
                        stab_df = pd.read_csv(stab_path)
                        # Convert to dict: (source, target) -> frequency
                        for _, row in stab_df.iterrows():
                            freqs[(row["source"], row["target"])] = row["frequency"]
                        loaded_stability = True
                        print(f"Loaded stability metrics for {dataset} {name} from {stab_path}")
                    except Exception as e:
                        print(f"Failed to load stability {stab_path}: {e}")

            if not loaded_stability:
                freqs = bootstrap_edge_stability(
                    lambda d: func(d.copy(), **params),
                    data,
                    b=bootstrap_runs,
                    seed=0,
                    n_jobs=n_jobs,
                )
            
            key_freq = freqs.get(key_edge, 0.0)

            bic = info.get("bic", float("nan"))

            # Algorithm vs truth
            metrics_t, extra_t, missing_t, reversed_t = compare_graphs(
                graph, true_graph
            )
            rows.append(
                {
                    "dataset": dataset,
                    "method": name,
                    "reference": "truth",
                    "runtime_s": runtime,
                    "bic": bic,
                    **metrics_t,
                    "extra": sorted(list(extra_t)),
                    "missing": sorted(list(missing_t)),
                    "reversed": sorted(list(reversed_t)),
                    "key_edge_freq": key_freq,
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
                    "runtime_s": runtime,
                    "bic": bic,
                    **metrics_a,
                    "extra": sorted(list(extra_a)),
                    "missing": sorted(list(missing_a)),
                    "reversed": sorted(list(reversed_a)),
                    "key_edge_freq": key_freq,
                }
            )

            if diff_dir is not None:
                scenario_dir = Path(diff_dir) / dataset
                scenario_dir.mkdir(parents=True, exist_ok=True)
                log_path = scenario_dir / f"{name}.log"
                with open(log_path, "w") as f:
                    f.write(f"algorithm: {name}\n")
                    f.write("vs truth:\n")
                    f.write(f"  extra: {sorted(list(extra_t))}\n")
                    f.write(f"  missing: {sorted(list(missing_t))}\n")
                    f.write("vs analyst:\n")
                    f.write(f"  extra: {sorted(list(extra_a))}\n")
                    f.write(f"  missing: {sorted(list(missing_a))}\n")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Sensitivity Analysis")
    parser.add_argument("--n-samples", type=int, default=None, help="Sample size for each dataset")
    parser.add_argument(
        "--bootstrap-runs", type=int, default=0, help="Number of bootstrap runs"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=4, help="Parallel jobs for bootstrap runs (default: 4, use -1 for all cores)"
    )
    parser.add_argument("--out", type=str, default=None, help="Output path for JSON results")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write sensitivity_analysis_results.csv",
    )
    parser.add_argument(
        "--diff-logs",
        action="store_true",
        help="Write per-algorithm diff logs to the output directory",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
        help="Directory containing benchmark results (outputs/ and logs/) to reuse instead of re-running algorithms.",
    )
    args = parser.parse_args()

    start_time = time.time()

    diff_dir = None
    if args.diff_logs and args.out_dir is not None:
        diff_dir = Path(args.out_dir) / "diff_logs"

    # Determine output directory (use provided or default)
    if args.out_dir is not None:
        results_dir = Path(args.out_dir)
    else:
        results_dir = Path(__file__).resolve().parents[1] / "results_new" / "sensitivity"
    
    results = run(
        sample_size=args.n_samples,
        bootstrap_runs=args.bootstrap_runs,
        n_jobs=args.n_jobs,
        diff_dir=diff_dir,
        output_dir=results_dir,
        benchmark_dir=Path(args.benchmark_dir) if args.benchmark_dir else None,
    )
    
    # Always save results to CSV
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "sensitivity_analysis_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    # Also save as JSON if requested
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results.to_dict(orient="records"), f, indent=2)
        print(f"Saved JSON results to {out_path}")

    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
