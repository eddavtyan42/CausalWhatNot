import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.loaders import load_dataset

# Ensure the Asia dataset exists for the benchmark tests. This avoids
# interference from previous tests that may remove the generated file.
load_dataset('asia', n_samples=500, force=True)

from pathlib import Path
import yaml
import pandas as pd
import pytest

from experiments import run_benchmark
from utils.loaders import load_dataset


@pytest.mark.timeout(30)
def test_run_benchmark(tmp_path):
    cfg = {
        'datasets': ['asia'],
        'algorithms': {'pc': {}, 'ges': {}},
        'bootstrap_runs': 0,
    }
    cfg_path = tmp_path / 'cfg.yaml'
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f)

    # Ensure Asia data is generated with a small sample size for the test
    load_dataset('asia', n_samples=200, force=True)

    run_benchmark.run(str(cfg_path), output_dir=tmp_path)

    summary = pd.read_csv(tmp_path / 'summary_metrics.csv')
    assert set(summary['algorithm']) == {'pc', 'ges'}
    assert (tmp_path / 'logs' / 'asia_pc.log').exists()
    assert (tmp_path / 'logs' / 'asia_ges.log').exists()
    assert summary['precision'].between(0, 1).all()
    assert summary['recall'].between(0, 1).all()


@pytest.mark.timeout(60)
def test_run_benchmark_notears(tmp_path):
    try:
        import algorithms.notears  # noqa: F401
    except Exception:
        pytest.skip('causalnex not installed')

    cfg = {
        'datasets': ['asia'],
        'algorithms': {'notears': {}},
        'bootstrap_runs': 0,
    }
    cfg_path = tmp_path / 'cfg.yaml'
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f)

    load_dataset('asia', n_samples=200, force=True)

    run_benchmark.run(str(cfg_path), output_dir=tmp_path)

    summary = pd.read_csv(tmp_path / 'summary_metrics.csv')
    assert set(summary['algorithm']) == {'notears'}
    assert (tmp_path / 'logs' / 'asia_notears.log').exists()
    assert summary['precision'].between(0, 1).all()
    assert summary['recall'].between(0, 1).all()
