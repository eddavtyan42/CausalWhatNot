import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import yaml
import pandas as pd
import pytest

from experiments import run_benchmark


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

    run_benchmark.run(str(cfg_path), output_dir=tmp_path)

    summary = pd.read_csv(tmp_path / 'summary_metrics.csv')
    assert set(summary['algorithm']) == {'pc', 'ges'}
    assert (tmp_path / 'logs' / 'asia_pc.log').exists()
    assert (tmp_path / 'logs' / 'asia_ges.log').exists()
    assert summary['precision'].between(0, 1).all()
    assert summary['recall'].between(0, 1).all()
