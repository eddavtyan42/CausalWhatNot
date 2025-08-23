import argparse
import importlib
import yaml
from pathlib import Path
import pandas as pd
import warnings
import networkx as nx
import numpy as np
import concurrent.futures
import joblib
import sys, os

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.loaders import load_dataset
from utils.helpers import edge_differences, dump_edge_differences_json
from metrics.metrics import (
    shd,
    precision_recall_f1,
    directed_precision_recall_f1,
    shd_dir,
)
from metrics.bootstrap import bootstrap_edge_stability


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUTPUTS_DIR = RESULTS_DIR / "outputs"
LOGS_DIR = RESULTS_DIR / "logs"
RESULTS_DIR.mkdir(exist_ok=True)
# Suppress noisy deprecation warnings from NumPy's legacy matrix/matlib used by dependencies
warnings.filterwarnings(
    "ignore", category=PendingDeprecationWarning, module=r"numpy\.matlib"
)
warnings.filterwarnings(
    "ignore", category=PendingDeprecationWarning, module=r"numpy\.matrixlib\.defmatrix"
)
# Also filter by message to catch warnings emitted via third-party modules (e.g., causallearn)
warnings.filterwarnings(
    "ignore",
    category=PendingDeprecationWarning,
    message=r".*Importing from numpy\.matlib is deprecated.*",
)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def run(
    config_path: str,
    output_dir: str | Path | None = None,
    parallel_jobs: int | None = None,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(output_dir) if output_dir is not None else RESULTS_DIR
    outputs_dir = base_dir / "outputs"
    logs_dir = base_dir / "logs"
    base_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    bootstrap = int(cfg.get("bootstrap_runs", 0))
    record_stability = bool(cfg.get("record_edge_stability", False))
    orient_metrics = bool(cfg.get("orientation_metrics", False))
    parallel_jobs = int(parallel_jobs if parallel_jobs is not None else cfg.get("parallel_jobs", 1))

    summary_rows = []

    dataset_infos = []
    for ds_cfg in cfg.get("datasets", []):
        if isinstance(ds_cfg, dict):
            dataset = ds_cfg.get("name")
            alias = ds_cfg.get("alias", dataset)
            n_samples = ds_cfg.get("n_samples")
        elif isinstance(ds_cfg, str):
            dataset = ds_cfg
            alias = dataset
            n_samples = None
        else:
            raise ValueError(f"Invalid dataset entry: {ds_cfg}")

        if n_samples is not None:
            data, true_graph = load_dataset(dataset, n_samples=n_samples, force=True)
        else:
            data, true_graph = load_dataset(dataset)

        dataset_infos.append((alias, data, true_graph))

    algo_items = [
        (name, dict(params or {}))
        for name, params in cfg.get("algorithms", {}).items()
    ]

    def process_pair(alias: str, data: pd.DataFrame, true_graph: nx.DiGraph, algo_name: str, params: dict):
        mod = importlib.import_module(f"algorithms.{algo_name}")
        params = dict(params)
        timeout_s = params.pop("timeout_s", None)

        run_metrics = []
        run_times = []
        errors = []
        diff_path = logs_dir / f"{alias}_{algo_name}_diff.txt"
        with open(diff_path, "w"):
            pass
        for b in range(bootstrap if bootstrap > 0 else 1):
            d_run = (
                data.sample(len(data), replace=True, random_state=b)
                if bootstrap > 0
                else data
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(mod.run, d_run.copy(), **params)
                try:
                    if timeout_s is None:
                        graph, info = fut.result()
                    else:
                        graph, info = fut.result(timeout=timeout_s)
                    err = ""
                except concurrent.futures.TimeoutError:
                    fut.cancel()
                    graph = None
                    info = {"runtime_s": timeout_s or 0}
                    err = "timeout"
                except Exception as e:
                    fut.cancel()
                    graph = None
                    info = {"runtime_s": 0}
                    err = str(e)

            if graph is not None:
                metrics = precision_recall_f1(graph, true_graph)
                metrics["shd"] = shd(
                    graph,
                    true_graph,
                    pred_undirected=set(info.get("undirected_edges", [])),
                )
                if orient_metrics:
                    metrics.update(directed_precision_recall_f1(graph, true_graph))
                    metrics["shd_dir"] = shd_dir(graph, true_graph)

                extra, missing, rev = edge_differences(graph, true_graph)
                with open(diff_path, "a") as df:
                    df.write(f"run{b}:\n")
                    for e in extra:
                        df.write(f"extra {e[0]}->{e[1]}\n")
                    for e in missing:
                        df.write(f"missing {e[0]}->{e[1]}\n")
                    for e in rev:
                        df.write(f"reversed {e[0]}->{e[1]}\n")
                diff_json_path = logs_dir / f"{alias}_{algo_name}_diff_run{b}.json"
                dump_edge_differences_json(extra, missing, rev, diff_json_path)
                if bootstrap == 0:
                    adj_path = outputs_dir / f"{alias}_{algo_name}.csv"
                    mat = nx.to_numpy_array(graph, nodelist=data.columns)
                    pd.DataFrame(mat, index=data.columns, columns=data.columns).to_csv(adj_path)
            else:
                # Algorithm failed or timed out.  Treat the output as an empty
                # graph so that downstream metrics are well defined instead of
                # yielding NaNs when all runs fail.
                empty = nx.DiGraph()
                empty.add_nodes_from(data.columns)
                metrics = precision_recall_f1(empty, true_graph)
                metrics["shd"] = shd(empty, true_graph)
                if orient_metrics:
                    metrics.update(directed_precision_recall_f1(empty, true_graph))
                    metrics["shd_dir"] = shd_dir(empty, true_graph)

            run_metrics.append(metrics)
            run_times.append(info["runtime_s"])
            errors.append(err)

        ok_metrics = [m for m, e in zip(run_metrics, errors) if not e]
        ok_times = [t for t, e in zip(run_times, errors) if not e]
        if len(ok_metrics) == 0:
            # If every run failed, fall back to the metrics computed for the
            # empty graphs recorded above.  This provides defined precision,
            # recall and SHD values (all zero except SHD) instead of NaNs.
            prec = np.array([m["precision"] for m in run_metrics], dtype=float)
            rec = np.array([m["recall"] for m in run_metrics], dtype=float)
            f1 = np.array([m["f1"] for m in run_metrics], dtype=float)
            shd_vals = np.array([m["shd"] for m in run_metrics], dtype=float)
            if orient_metrics:
                d_prec = np.array([m["directed_precision"] for m in run_metrics], dtype=float)
                d_rec = np.array([m["directed_recall"] for m in run_metrics], dtype=float)
                d_f1 = np.array([m["directed_f1"] for m in run_metrics], dtype=float)
                shd_dir_vals = np.array([m["shd_dir"] for m in run_metrics], dtype=float)
            times = np.array(run_times, dtype=float)
        else:
            prec = np.array([m["precision"] for m in ok_metrics], dtype=float)
            rec = np.array([m["recall"] for m in ok_metrics], dtype=float)
            f1 = np.array([m["f1"] for m in ok_metrics], dtype=float)
            shd_vals = np.array([m["shd"] for m in ok_metrics], dtype=float)
            if orient_metrics:
                d_prec = np.array([m["directed_precision"] for m in ok_metrics], dtype=float)
                d_rec = np.array([m["directed_recall"] for m in ok_metrics], dtype=float)
                d_f1 = np.array([m["directed_f1"] for m in ok_metrics], dtype=float)
                shd_dir_vals = np.array([m["shd_dir"] for m in ok_metrics], dtype=float)
            times = np.array(ok_times, dtype=float)
        n_fail = sum(1 for e in errors if e and e != "timeout")
        n_timeout = sum(1 for e in errors if e == "timeout")
        all_failed = len(ok_metrics) == 0

        log_path = logs_dir / f"{alias}_{algo_name}.log"
        with open(log_path, "w") as f:
            for i, (m, err) in enumerate(zip(run_metrics, errors)):
                if err:
                    f.write(f"run{i}: {err}\n")
                else:
                    line = f"run{i}: precision={m['precision']:.3f}, recall={m['recall']:.3f}, f1={m['f1']:.3f}, shd={m['shd']}"
                    if orient_metrics:
                        line += (
                            f", directed_precision={m['directed_precision']:.3f},"
                            f" directed_recall={m['directed_recall']:.3f},"
                            f" directed_f1={m['directed_f1']:.3f},"
                            f" shd_dir={m['shd_dir']}"
                        )
                    f.write(line + "\n")
            summary_line = (
                "summary: "
                f"precision={prec.mean():.3f}±{prec.std(ddof=0):.3f}, "
                f"recall={rec.mean():.3f}±{rec.std(ddof=0):.3f}, "
                f"f1={f1.mean():.3f}±{f1.std(ddof=0):.3f}, "
                f"shd={shd_vals.mean():.3f}±{shd_vals.std(ddof=0):.3f}, "
            )
            if orient_metrics:
                summary_line += (
                    f"directed_precision={d_prec.mean():.3f}±{d_prec.std(ddof=0):.3f}, "
                    f"directed_recall={d_rec.mean():.3f}±{d_rec.std(ddof=0):.3f}, "
                    f"directed_f1={d_f1.mean():.3f}±{d_f1.std(ddof=0):.3f}, "
                    f"shd_dir={shd_dir_vals.mean():.3f}±{shd_dir_vals.std(ddof=0):.3f}, "
                )
            summary_line += f"runtime_s={times.mean():.2f}±{times.std(ddof=0):.2f}\n"
            if all_failed:
                f.write("summary_status: ALL_RUNS_FAILED\n")
            f.write(summary_line)

        row = {
            "dataset": alias,
            "algorithm": algo_name,
            "precision": prec.mean() if prec.size else np.nan,
            "precision_std": prec.std(ddof=0) if prec.size else np.nan,
            "recall": rec.mean() if rec.size else np.nan,
            "recall_std": rec.std(ddof=0) if rec.size else np.nan,
            "f1": f1.mean() if f1.size else np.nan,
            "f1_std": f1.std(ddof=0) if f1.size else np.nan,
            "shd": shd_vals.mean() if shd_vals.size else np.nan,
            "shd_std": shd_vals.std(ddof=0) if shd_vals.size else np.nan,
            "runtime_s": times.mean() if times.size else np.nan,
            "runtime_s_std": times.std(ddof=0) if times.size else np.nan,
            "n_fail": n_fail,
            "n_timeout": n_timeout,
            "all_failed": all_failed,
        }
        if orient_metrics:
            row.update(
                {
                    "directed_precision": d_prec.mean() if d_prec.size else np.nan,
                    "directed_precision_std": d_prec.std(ddof=0) if d_prec.size else np.nan,
                    "directed_recall": d_rec.mean() if d_rec.size else np.nan,
                    "directed_recall_std": d_rec.std(ddof=0) if d_rec.size else np.nan,
                    "directed_f1": d_f1.mean() if d_f1.size else np.nan,
                    "directed_f1_std": d_f1.std(ddof=0) if d_f1.size else np.nan,
                    "shd_dir": shd_dir_vals.mean() if shd_dir_vals.size else np.nan,
                    "shd_dir_std": shd_dir_vals.std(ddof=0) if shd_dir_vals.size else np.nan,
                }
            )
        if record_stability and bootstrap > 0:
            freqs = bootstrap_edge_stability(
                lambda d: mod.run(d.copy(), **params),
                data,
                b=bootstrap,
                seed=0,
                n_jobs=-1,
            )
            stab_df = pd.DataFrame([
                {"source": s, "target": t, "frequency": f} for (s, t), f in freqs.items()
            ])
            stab_df.to_csv(logs_dir / f"{alias}_{algo_name}_stability.csv", index=False)

        return row

    tasks = [
        joblib.delayed(process_pair)(alias, data, true_graph, algo_name, params)
        for alias, data, true_graph in dataset_infos
        for algo_name, params in algo_items
    ]

    summary_rows.extend(joblib.Parallel(n_jobs=parallel_jobs, prefer="threads")(tasks))

    df = pd.DataFrame(summary_rows)
    df.to_csv(base_dir / "summary_metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(Path(__file__).with_name("config.yaml"))
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--parallel-jobs", type=int, default=None)
    args = parser.parse_args()
    run(args.config, args.out_dir, parallel_jobs=args.parallel_jobs)


if __name__ == "__main__":
    main()
