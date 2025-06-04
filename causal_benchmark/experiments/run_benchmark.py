import argparse
import importlib
import yaml
from pathlib import Path
import pandas as pd
import networkx as nx
import sys, os
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.loaders import load_dataset
from metrics.metrics import shd, precision_recall_f1


RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
OUTPUTS_DIR = RESULTS_DIR / 'outputs'
LOGS_DIR = RESULTS_DIR / 'logs'
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def run(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    summary_rows = []

    for dataset in cfg.get('datasets', []):
        data, true_graph = load_dataset(dataset)

        for algo_name, params in cfg.get('algorithms', {}).items():
            mod = importlib.import_module(f'algorithms.{algo_name}')
            try:
                graph, info = mod.run(data.copy(), **params)
                err = ''
            except Exception as e:
                graph = None
                info = {'runtime_s': 0}
                err = str(e)

            if graph is not None:
                metrics = precision_recall_f1(graph, true_graph)
                metrics['shd'] = shd(graph, true_graph)
                adj_path = OUTPUTS_DIR / f'{dataset}_{algo_name}.csv'
                pd.DataFrame(nx.to_numpy_array(graph, nodelist=data.columns)).to_csv(adj_path, index=False)
            else:
                metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'shd': -1}

            log_path = LOGS_DIR / f'{dataset}_{algo_name}.log'
            with open(log_path, 'w') as f:
                f.write(err or 'OK')

            row = {'dataset': dataset, 'algorithm': algo_name, **metrics, 'runtime_s': info['runtime_s']}
            summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    df.to_csv(RESULTS_DIR / 'summary_metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
