import argparse
import importlib
import yaml
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
import concurrent.futures
import sys, os
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.loaders import load_dataset
from utils.helpers import edge_differences
from metrics.metrics import (
    shd,
    precision_recall_f1,
    directed_precision_recall_f1,
    shd_dir,
)
from metrics.bootstrap import bootstrap_edge_stability


RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
OUTPUTS_DIR = RESULTS_DIR / 'outputs'
LOGS_DIR = RESULTS_DIR / 'logs'
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def run(config_path: str, output_dir: str | Path | None = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(output_dir) if output_dir is not None else RESULTS_DIR
    outputs_dir = base_dir / 'outputs'
    logs_dir = base_dir / 'logs'
    base_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    bootstrap = int(cfg.get('bootstrap_runs', 0))
    record_stability = bool(cfg.get('record_edge_stability', False))
    orient_metrics = bool(cfg.get('orientation_metrics', False))

    summary_rows = []

    for ds_cfg in cfg.get('datasets', []):
        if isinstance(ds_cfg, str):
            dataset = ds_cfg
            n_samples = None
        elif isinstance(ds_cfg, dict):
            dataset = ds_cfg.get('name')
            n_samples = ds_cfg.get('n_samples')
        else:
            raise ValueError(f'Invalid dataset entry: {ds_cfg}')

        if n_samples is not None:
            data, true_graph = load_dataset(dataset, n_samples=n_samples, force=True)
        else:
            data, true_graph = load_dataset(dataset)

        for algo_name, params in cfg.get('algorithms', {}).items():
            mod = importlib.import_module(f'algorithms.{algo_name}')
            # individual algorithm modules may raise ImportError inside `run()`;
            # those errors are captured below and logged per bootstrap run.
            params = dict(params or {})
            timeout_s = params.pop('timeout_s', None)

            run_metrics = []
            run_times = []
            errors = []
            diff_path = logs_dir / f'{dataset}_{algo_name}_diff.txt'
            # start a fresh diff file for this dataset/algorithm
            with open(diff_path, 'w'):
                pass
            for b in range(bootstrap if bootstrap > 0 else 1):
                d_run = data.sample(len(data), replace=True, random_state=b) if bootstrap > 0 else data
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(mod.run, d_run.copy(), **params)
                    try:
                        if timeout_s is None:
                            graph, info = fut.result()
                        else:
                            graph, info = fut.result(timeout=timeout_s)
                        err = ''
                    except concurrent.futures.TimeoutError:
                        fut.cancel()
                        graph = None
                        info = {'runtime_s': timeout_s or 0}
                        err = 'timeout'
                    except Exception as e:
                        fut.cancel()
                        graph = None
                        info = {'runtime_s': 0}
                        err = str(e)

                if graph is not None:
                    metrics = precision_recall_f1(graph, true_graph)
                    metrics['shd'] = shd(
                        graph,
                        true_graph,
                        pred_undirected=set(info.get('undirected_edges', [])),
                    )
                    if orient_metrics:
                        metrics.update(directed_precision_recall_f1(graph, true_graph))
                        metrics['shd_dir'] = shd_dir(graph, true_graph)
                    extra, missing, rev = edge_differences(graph, true_graph)
                    with open(diff_path, 'a') as df:
                        df.write(f'run{b}:\n')
                        for e in extra:
                            df.write(f'extra {e[0]}->{e[1]}\n')
                        for e in missing:
                            df.write(f'missing {e[0]}->{e[1]}\n')
                        for e in rev:
                            df.write(f'reversed {e[0]}->{e[1]}\n')
                    if bootstrap == 0:
                        adj_path = outputs_dir / f'{dataset}_{algo_name}.csv'
                        mat = nx.to_numpy_array(graph, nodelist=data.columns)
                        pd.DataFrame(mat, index=data.columns, columns=data.columns).to_csv(adj_path)
                else:
                    metrics = {
                        'precision': 0,
                        'recall': 0,
                        'f1': 0,
                        'shd': -1,
                    }
                    if orient_metrics:
                        metrics.update(
                            {
                                'directed_precision': 0,
                                'directed_recall': 0,
                                'directed_f1': 0,
                                'shd_dir': -1,
                            }
                        )

                run_metrics.append(metrics)
                run_times.append(info['runtime_s'])
                errors.append(err)

            # Logging for all runs
            prec = np.array([m['precision'] for m in run_metrics])
            rec = np.array([m['recall'] for m in run_metrics])
            f1 = np.array([m['f1'] for m in run_metrics])
            shd_vals = np.array([m['shd'] for m in run_metrics])
            if orient_metrics:
                d_prec = np.array([m['directed_precision'] for m in run_metrics])
                d_rec = np.array([m['directed_recall'] for m in run_metrics])
                d_f1 = np.array([m['directed_f1'] for m in run_metrics])
                shd_dir_vals = np.array([m['shd_dir'] for m in run_metrics])
            times = np.array(run_times)

            log_path = logs_dir / f'{dataset}_{algo_name}.log'
            with open(log_path, 'w') as f:
                for i, (m, err) in enumerate(zip(run_metrics, errors)):
                    if err:
                        f.write(f'run{i}: {err}\n')
                    else:
                        line = (
                            f"run{i}: precision={m['precision']:.3f}, recall={m['recall']:.3f}, f1={m['f1']:.3f}, shd={m['shd']}"
                        )
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
                f.write(summary_line)

            row = {
                'dataset': dataset,
                'algorithm': algo_name,
                'precision': prec.mean(),
                'precision_std': prec.std(ddof=0),
                'recall': rec.mean(),
                'recall_std': rec.std(ddof=0),
                'f1': f1.mean(),
                'f1_std': f1.std(ddof=0),
                'shd': shd_vals.mean(),
                'shd_std': shd_vals.std(ddof=0),
                'runtime_s': times.mean(),
                'runtime_s_std': times.std(ddof=0),
            }
            if orient_metrics:
                row.update({
                    'directed_precision': d_prec.mean(),
                    'directed_precision_std': d_prec.std(ddof=0),
                    'directed_recall': d_rec.mean(),
                    'directed_recall_std': d_rec.std(ddof=0),
                    'directed_f1': d_f1.mean(),
                    'directed_f1_std': d_f1.std(ddof=0),
                    'shd_dir': shd_dir_vals.mean(),
                    'shd_dir_std': shd_dir_vals.std(ddof=0),
                })
            summary_rows.append(row)

            if record_stability and bootstrap > 0:
                freqs = bootstrap_edge_stability(
                    lambda d: mod.run(d.copy(), **params),
                    data,
                    b=bootstrap,
                    seed=0,
                    n_jobs=-1,
                )
                stab_df = pd.DataFrame(
                    [
                        {
                            'source': s,
                            'target': t,
                            'frequency': f,
                        }
                        for (s, t), f in freqs.items()
                    ]
                )
                stab_df.to_csv(
                    logs_dir / f'{dataset}_{algo_name}_stability.csv',
                    index=False,
                )

    df = pd.DataFrame(summary_rows)
    df.to_csv(base_dir / 'summary_metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()
    run(args.config, args.out_dir)


if __name__ == '__main__':
    main()
