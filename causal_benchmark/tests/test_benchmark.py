import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.loaders import load_dataset

# Ensure the Asia dataset exists for the benchmark tests. This avoids
# interference from previous tests that may remove the generated file.
load_dataset("asia", n_samples=500, force=True)

from pathlib import Path
import yaml
import json
import pandas as pd
import pytest

from experiments import run_benchmark
from utils.loaders import load_dataset


@pytest.mark.timeout(30)
def test_run_benchmark(tmp_path):
    cfg = {
        "datasets": [{"name": "asia", "n_samples": 200}],
        "algorithms": {"pc": {}, "ges": {}},
        "bootstrap_runs": 0,
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    # Ensure Asia data is generated with a small sample size for the test
    load_dataset("asia", n_samples=200, force=True)

    run_benchmark.run(str(cfg_path), output_dir=tmp_path)

    summary = pd.read_csv(tmp_path / "summary_metrics.csv")
    assert set(summary["algorithm"]) == {"pc", "ges"}
    assert (tmp_path / "logs" / "asia_pc.log").exists()
    assert (tmp_path / "logs" / "asia_ges.log").exists()
    assert (tmp_path / "logs" / "asia_pc_diff.txt").exists()
    assert (tmp_path / "logs" / "asia_ges_diff.txt").exists()
    assert (tmp_path / "logs" / "asia_pc_diff_run0.json").exists()
    assert (tmp_path / "logs" / "asia_ges_diff_run0.json").exists()
    for p in [tmp_path / "logs" / "asia_pc_diff_run0.json", tmp_path / "logs" / "asia_ges_diff_run0.json"]:
        data = json.loads(p.read_text())
        assert set(data) == {"extra", "missing", "reversed"}
        for lst in data.values():
            assert isinstance(lst, list)
    assert summary["precision"].between(0, 1).all()
    assert summary["recall"].between(0, 1).all()


@pytest.mark.timeout(60)
def test_run_benchmark_notears(tmp_path):
    if sys.version_info >= (3, 11):
    # pandas already imported at module level; avoid redefining to prevent
    # UnboundLocalError on Python <3.11 where this branch is skipped.
        import numpy as np
        import algorithms.notears as notears

        with pytest.raises(ImportError):
            notears.run(pd.DataFrame(np.zeros((10, 2)), columns=["a", "b"]))
        return
    try:
        import algorithms.notears as notears  # noqa: F401
    except Exception:
        pytest.skip("causalnex not installed")

    cfg = {
        "datasets": [{"name": "asia", "n_samples": 200}],
        "algorithms": {"notears": {}},
        "bootstrap_runs": 0,
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    load_dataset("asia", n_samples=200, force=True)

    run_benchmark.run(str(cfg_path), output_dir=tmp_path)

    summary = pd.read_csv(tmp_path / "summary_metrics.csv")
    assert set(summary["algorithm"]) == {"notears"}
    assert (tmp_path / "logs" / "asia_notears.log").exists()
    assert (tmp_path / "logs" / "asia_notears_diff.txt").exists()
    assert (tmp_path / "logs" / "asia_notears_diff_run0.json").exists()
    data = json.loads((tmp_path / "logs" / "asia_notears_diff_run0.json").read_text())
    assert set(data) == {"extra", "missing", "reversed"}
    assert summary["precision"].between(0, 1).all()
    assert summary["recall"].between(0, 1).all()


@pytest.mark.timeout(30)
def test_dataset_n_samples(tmp_path):
    # Start with a dataset of a different size to verify overwrite
    load_dataset("asia", n_samples=50, force=True)

    cfg = {
        "datasets": [{"name": "asia", "n_samples": 123}],
        "algorithms": {"ges": {}},
        "bootstrap_runs": 0,
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    run_benchmark.run(str(cfg_path), output_dir=tmp_path)

    data_path = Path(__file__).resolve().parents[1] / "data" / "asia" / "asia_data.csv"
    df = pd.read_csv(data_path)
    assert len(df) == 123


@pytest.mark.timeout(30)
def test_algorithm_timeout(tmp_path, monkeypatch):
    from algorithms import cosmo
    import time
    import networkx as nx

    def slow_run(data, **_):
        time.sleep(1)
        g = nx.DiGraph()
        g.add_nodes_from(data.columns)
        return g, {"runtime_s": 1}

    monkeypatch.setattr(cosmo, "run", slow_run)

    cfg = {
        "datasets": ["asia"],
        "algorithms": {"cosmo": {"timeout_s": 0.1}},
        "bootstrap_runs": 0,
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    load_dataset("asia", n_samples=100, force=True)

    run_benchmark.run(str(cfg_path), output_dir=tmp_path)

    summary = pd.read_csv(tmp_path / "summary_metrics.csv")
    assert summary["n_timeout"].iloc[0] == 1
    assert summary["n_fail"].iloc[0] == 0
    # A timeout yields an implicit empty graph; metrics are defined and equal to
    # zero rather than NaN.
    assert summary["precision"].iloc[0] == 0.0
    log_text = (tmp_path / "logs" / "asia_cosmo.log").read_text()
    assert "timeout" in log_text


@pytest.mark.timeout(30)
def test_algorithm_exception(tmp_path, monkeypatch):
    from algorithms import cosmo
    def bad_run(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cosmo, "run", bad_run)

    cfg = {
        "datasets": ["asia"],
        "algorithms": {"cosmo": {}},
        "bootstrap_runs": 0,
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    load_dataset("asia", n_samples=100, force=True)

    run_benchmark.run(str(cfg_path), output_dir=tmp_path)

    summary = pd.read_csv(tmp_path / "summary_metrics.csv")
    assert summary["n_fail"].iloc[0] == 1
    assert summary["n_timeout"].iloc[0] == 0
    # Failed runs now contribute an empty graph, resulting in zero-valued metrics.
    assert summary["precision"].iloc[0] == 0.0
    log_text = (tmp_path / "logs" / "asia_cosmo.log").read_text()
    assert "boom" in log_text


@pytest.mark.timeout(30)
def test_diff_file_run_headers(tmp_path):
    cfg = {
        "datasets": ["asia"],
        "algorithms": {"pc": {}},
        "bootstrap_runs": 2,
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    load_dataset("asia", n_samples=100, force=True)

    run_benchmark.run(str(cfg_path), output_dir=tmp_path)

    diff_lines = (tmp_path / "logs" / "asia_pc_diff.txt").read_text().splitlines()
    assert any(line.startswith("run0:") for line in diff_lines)
    assert any(line.startswith("run1:") for line in diff_lines)
    for i in (0, 1):
        p = tmp_path / "logs" / f"asia_pc_diff_run{i}.json"
        assert p.exists()
        data = json.loads(p.read_text())
        assert set(data) == {"extra", "missing", "reversed"}


@pytest.mark.timeout(30)
def test_orientation_metrics_summary(tmp_path):
    cfg = {
        "datasets": [{"name": "asia", "n_samples": 200}],
        "algorithms": {"pc": {}},
        "bootstrap_runs": 0,
        "orientation_metrics": True,
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    load_dataset("asia", n_samples=200, force=True)

    run_benchmark.run(str(cfg_path), output_dir=tmp_path)

    summary = pd.read_csv(tmp_path / "summary_metrics.csv")
    assert "directed_precision" in summary.columns
    assert summary["directed_precision"].between(0, 1).all()


@pytest.mark.timeout(30)
def test_dataset_alias_files(tmp_path):
    cfg = {
        'datasets': [
            {'name': 'asia', 'n_samples': 100, 'alias': 'asia_a'},
            {'name': 'asia', 'n_samples': 200, 'alias': 'asia_b'},
        ],
        'algorithms': {'pc': {}},
        'bootstrap_runs': 0,
    }
    cfg_path = tmp_path / 'cfg.yaml'
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f)

    load_dataset('asia', n_samples=200, force=True)

    run_benchmark.run(str(cfg_path), output_dir=tmp_path)

    assert (tmp_path / 'logs' / 'asia_a_pc.log').exists()
    assert (tmp_path / 'logs' / 'asia_b_pc.log').exists()
    assert (tmp_path / 'logs' / 'asia_a_pc_diff_run0.json').exists()
    assert (tmp_path / 'logs' / 'asia_b_pc_diff_run0.json').exists()
    assert (tmp_path / 'outputs' / 'asia_a_pc.csv').exists()
    assert (tmp_path / 'outputs' / 'asia_b_pc.csv').exists()
    summary = pd.read_csv(tmp_path / 'summary_metrics.csv')
    assert set(summary['dataset']) == {'asia_a', 'asia_b'}


@pytest.mark.timeout(30)
def test_parallel_jobs_speedup(tmp_path, monkeypatch):
    from algorithms import cosmo
    import time
    import networkx as nx

    def slow_run(data, **_):
        time.sleep(1)
        g = nx.DiGraph()
        g.add_nodes_from(data.columns)
        return g, {"runtime_s": 1.0}

    monkeypatch.setattr(cosmo, "run", slow_run)

    cfg = {
        'datasets': [
            {'name': 'asia', 'n_samples': 10, 'alias': 'd1'},
            {'name': 'asia', 'n_samples': 10, 'alias': 'd2'},
        ],
        'algorithms': {'cosmo': {}},
        'bootstrap_runs': 0,
    }
    cfg_path = tmp_path / 'cfg.yaml'
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f)

    load_dataset('asia', n_samples=10, force=True)

    start = time.perf_counter()
    run_benchmark.run(str(cfg_path), output_dir=tmp_path / 'seq', parallel_jobs=1)
    t1 = time.perf_counter() - start

    start = time.perf_counter()
    run_benchmark.run(str(cfg_path), output_dir=tmp_path / 'par', parallel_jobs=2)
    t2 = time.perf_counter() - start

    assert t1 - t2 > 0.5
