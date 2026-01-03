import argparse
import importlib
import yaml
import time
from pathlib import Path
import pandas as pd
import warnings
import networkx as nx
import numpy as np
import concurrent.futures
import joblib
import sys, os
import logging

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.loaders import load_dataset
from utils.helpers import edge_differences, dump_edge_differences_json
from utils.logging_utils import setup_logging
from utils.provenance import save_run_metadata, save_graph_artifacts
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
    start_time = time.time()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(output_dir) if output_dir is not None else RESULTS_DIR
    outputs_dir = base_dir / "outputs"
    logs_dir = base_dir / "logs"
    base_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Initialize centralized benchmark logger to a session log file
    session_log = logs_dir / "benchmark_session.log"
    logger = setup_logging(session_log)
    logger.info("Benchmark run started: config=%s, out_dir=%s", config_path, str(base_dir))

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

        logger.info("Loading dataset: name=%s alias=%s n_samples=%s", dataset, alias, str(n_samples))
        if n_samples is not None:
            data, true_graph = load_dataset(dataset, n_samples=n_samples, force=True)
        else:
            data, true_graph = load_dataset(dataset)
        logger.info("Loaded dataset: alias=%s shape=%s nodes=%d edges=%d", alias, str(data.shape), true_graph.number_of_nodes(), true_graph.number_of_edges())

        dataset_infos.append((dataset, alias, data, true_graph))

    # Algorithms configuration can include optional per-dataset overrides, e.g.:
    # algorithms:
    #   ges:
    #     per_dataset:
    #       sachs: { score_func: bdeu }
    #     score_func: bic  # default
    cfg_algorithms = cfg.get("algorithms", {})

    def process_pair(dataset_name: str, alias: str, data: pd.DataFrame, true_graph: nx.DiGraph, algo_name: str, params: dict):
        # Skip if not in our target list for this partial run
        # This is a hack to filter the cross-product of datasets x algorithms
        target_pairs = {
            ("alarm", "ges"),
            ("insurance", "ges"),
            ("child", "pc")
        }
        if (alias, algo_name) not in target_pairs:
            return None

        logger.info("Starting algorithm: dataset=%s algo=%s params=%s", alias, algo_name, params)
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
            logger.info("Run start: dataset=%s algo=%s bootstrap=%d timeout_s=%s", alias, algo_name, b, str(timeout_s))
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
                    logger.warning("Run timeout: dataset=%s algo=%s bootstrap=%d after %ss", alias, algo_name, b, str(timeout_s))
                except Exception as e:
                    fut.cancel()
                    graph = None
                    info = {"runtime_s": 0}
                    err = str(e)
                    logger.exception("Run error: dataset=%s algo=%s bootstrap=%d error=%s", alias, algo_name, b, err)

            if graph is not None:
                logger.info("Run success: dataset=%s algo=%s bootstrap=%d runtime_s=%.3f", alias, algo_name, b, info.get("runtime_s", float('nan')))
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
                logger.info("Edge diffs written: dataset=%s algo=%s bootstrap=%d file=%s", alias, algo_name, b, str(diff_json_path))
                
                # Save provenance metadata and artifacts
                base_filename = f"{alias}_{algo_name}_run{b}" if bootstrap > 0 else f"{alias}_{algo_name}"
                
                # Save metadata
                meta_path = outputs_dir / f"{base_filename}_meta.json"
                # We need to reconstruct the dataset path. load_dataset uses BASE_DIR internally.
                # Assuming standard location for now or we could expose it from load_dataset.
                # For now, we'll use a placeholder or try to infer.
                # Actually, load_dataset returns data, true_graph. It doesn't return the path.
                # We can construct the expected path.
                dataset_path = Path(__file__).resolve().parents[1] / "data" / dataset_name / f"{dataset_name}_data.csv"
                
                if not dataset_path.exists():
                    logger.error(f"Computed dataset path {dataset_path} does not exist! Provenance metadata will be incorrect.")

                save_run_metadata(
                    output_path=meta_path,
                    dataset_name=alias,
                    dataset_path=dataset_path,
                    n_samples=len(d_run),
                    algorithm_name=algo_name,
                    algorithm_params=params,
                    random_seed=b, # Using bootstrap index as seed proxy
                    preprocessing_info={"bootstrap_iteration": b} if bootstrap > 0 else {},
                    include_environment_snapshot=True
                )
                
                # Save graph artifacts
                save_graph_artifacts(
                    output_dir=outputs_dir,
                    base_filename=base_filename,
                    graph=graph,
                    nodes=list(data.columns),
                    raw_adjacency=None # We don't have raw weighted adj easily available here without modifying algos
                )
                logger.info("Artifacts saved: dataset=%s algo=%s base=%s", alias, algo_name, base_filename)

                if bootstrap == 0:
                    # Legacy CSV save (kept for compatibility, though artifacts cover it)
                    adj_path = outputs_dir / f"{alias}_{algo_name}.csv"
                    mat = nx.to_numpy_array(graph, nodelist=data.columns)
                    pd.DataFrame(mat, index=data.columns, columns=data.columns).to_csv(adj_path)
                    logger.info("Adjacency saved: dataset=%s algo=%s file=%s", alias, algo_name, str(adj_path))
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
                logger.info("Run produced empty graph: dataset=%s algo=%s bootstrap=%d", alias, algo_name, b)

            run_metrics.append(metrics)
            run_times.append(info["runtime_s"])
            errors.append(err)

        ok_metrics = [m for m, e in zip(run_metrics, errors) if not e]
        ok_times = [t for t, e in zip(run_times, errors) if not e]
        # Initialize orientation metric arrays to avoid UnboundLocalError when orient_metrics is False
        d_prec = np.array([], dtype=float)
        d_rec = np.array([], dtype=float)
        d_f1 = np.array([], dtype=float)
        shd_dir_vals = np.array([], dtype=float)
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
            # Helper to format mean±std without emitting warnings on empty arrays
            def fmt(arr: np.ndarray, p: int = 3) -> str:
                if arr.size == 0:
                    return "nan±nan"
                return f"{arr.mean():.{p}f}±{arr.std(ddof=0):.{p}f}"

            summary_line = (
                "summary: "
                f"precision={fmt(prec)}, "
                f"recall={fmt(rec)}, "
                f"f1={fmt(f1)}, "
                f"shd={fmt(shd_vals)}, "
            )
            if orient_metrics:
                summary_line += (
                    f"directed_precision={fmt(d_prec)}, "
                    f"directed_recall={fmt(d_rec)}, "
                    f"directed_f1={fmt(d_f1)}, "
                    f"shd_dir={fmt(shd_dir_vals)}, "
                )
            # Runtime with 2 decimals
            def fmt_time(arr: np.ndarray) -> str:
                if arr.size == 0:
                    return "nan±nan"
                return f"{arr.mean():.2f}±{arr.std(ddof=0):.2f}"
            summary_line += f"runtime_s={fmt_time(times)}\n"
            if all_failed:
                f.write("summary_status: ALL_RUNS_FAILED\n")
            f.write(summary_line)
        logger.info(
            "Algorithm summary written: dataset=%s algo=%s file=%s n_runs=%d n_fail=%d n_timeout=%d",
            alias,
            algo_name,
            str(log_path),
            len(run_metrics),
            n_fail,
            n_timeout,
        )

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
            logger.info("Edge stability saved: dataset=%s algo=%s", alias, algo_name)

        return row

    def _merge_algo_params(algo_params: dict | None, alias: str) -> dict:
        base = dict(algo_params or {})
        # Extract and apply per-dataset overrides if present
        per_ds = base.pop("per_dataset", {}) if isinstance(base, dict) else {}
        if isinstance(per_ds, dict):
            override = per_ds.get(alias, {})
            if isinstance(override, dict):
                base.update(override)
        return base

    tasks = []
    for dataset, alias, data, true_graph in dataset_infos:
        for algo_name, algo_params in cfg_algorithms.items():
            params = _merge_algo_params(algo_params if isinstance(algo_params, dict) else {}, alias)
            tasks.append(joblib.delayed(process_pair)(dataset, alias, data, true_graph, algo_name, params))
    logger.info("Launching parallel runs: tasks=%d parallel_jobs=%d", len(tasks), parallel_jobs)
    summary_rows.extend(joblib.Parallel(n_jobs=parallel_jobs, prefer="threads")(tasks))
    logger.info("Parallel runs finished: gathered_rows=%d", len(summary_rows))

    df = pd.DataFrame(summary_rows)
    summary_csv = base_dir / "summary_metrics.csv"
    df.to_csv(summary_csv, index=False)
    logger.info("Benchmark completed: summary=%s rows=%d", str(summary_csv), len(df))

    elapsed = time.time() - start_time
    logger.info(f"Total execution time: {elapsed:.2f} seconds")
    print(f"Total execution time: {elapsed:.2f} seconds")


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
