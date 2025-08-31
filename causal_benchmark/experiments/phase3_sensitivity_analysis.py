import argparse
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np

# Allow running as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from algorithms import pc, ges, notears, cosmo
from utils import create_mis_specified_graph
from utils.loaders import load_dataset
from experiments.phase3_scenarios import SCENARIOS


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
    results: dict[str, dict] = {}
    for dataset, scenario in SCENARIOS.items():
        data, true_graph = load_dataset(dataset, n_samples=sample_size, force=sample_size is not None)

        # Build mis-specified analyst graph by removing a key edge and adding a spurious edge
        analyst_graph = true_graph
        missing_u, missing_v = scenario["missing"]["edge"]
        spurious_u, spurious_v = scenario["spurious"]["edge"]
        analyst_graph = create_mis_specified_graph(analyst_graph, "missing", missing_u, missing_v)
        analyst_graph = create_mis_specified_graph(analyst_graph, "spurious", spurious_u, spurious_v)

        algo_results: dict[str, list[dict]] = {}
        algorithms = {"pc": pc.run, "ges": ges.run, "notears": notears.run, "cosmo": cosmo.run}
        for name, func in algorithms.items():
            runs: list[dict] = []
            n_runs = bootstrap if bootstrap > 0 else 1
            for b in range(n_runs):
                d_run = data.sample(len(data), replace=True, random_state=b) if bootstrap > 0 else data
                try:
                    graph, info = func(d_run.copy())
                    edges = list(graph.edges())
                    runtime = info.get("runtime_s", float("nan"))
                except Exception as e:  # pragma: no cover - safeguard against optional deps
                    G = nx.DiGraph()
                    G.add_nodes_from(data.columns)
                    edges = list(G.edges())
                    runtime = float("nan")
                    info = {"error": str(e)}

                def _sanitize(val):
                    if isinstance(val, (str, int, float, bool)) or val is None:
                        return val
                    if isinstance(val, (list, tuple)):
                        return [_sanitize(v) for v in val]
                    if hasattr(val, "tolist"):
                        try:
                            return val.tolist()
                        except Exception:
                            pass
                    return str(val)

                info_clean = {k: _sanitize(v) for k, v in info.items()}
                runs.append({"edges": edges, "runtime_s": runtime, "info": info_clean})
            algo_results[name] = runs
        results[dataset] = {
            "true_graph": list(true_graph.edges()),
            "analyst_graph": list(analyst_graph.edges()),
            "algorithms": algo_results,
        }
    return results


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
