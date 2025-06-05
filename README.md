# CausalWhatNot

![python](https://img.shields.io/badge/python-3.10-blue)

A reproducible benchmarking framework for causal discovery algorithms. CausalWhatNot allows researchers and practitioners to compare multiple structure learning methods on standard benchmark datasets using a consistent set of metrics.

## Overview

The project orchestrates data generation, algorithm execution and metric computation in a single Python package. It includes canonical networks such as Asia, Sachs, ALARM and Child and currently supports four algorithms:

* **PC** – constraint-based search using conditional independence tests
* **GES** – greedy equivalence search with a BIC score
* **NOTEARS** – continuous optimization approach implemented with CausalNex
* **COSMO** – priority-based regression implementation inspired by smooth acyclic orientations

Results are saved as adjacency matrices and summary metrics so experiments can be reproduced exactly.

## Features

* Runs PC, GES, NOTEARS and COSMO on common benchmark datasets
* Bootstrap evaluation with precision, recall, F1 and structural hamming distance (SHD)
* Optional recording of edge stability frequencies across bootstrap runs
* Easily extensible for new algorithms or datasets
* Deterministic sampling with fixed seeds for reproducibility
* Developed and tested with **Python 3.10**. NOTEARS currently requires Python <3.11 due to the CausalNex dependency.

## Installation

Clone the repository and install the dependencies using either `pip` or `conda`.

NOTEARS relies on the CausalNex library which currently only supports **Python <3.11**
and requires PyTorch (installed automatically with CausalNex). A Python 3.10
environment is therefore recommended for full functionality. The
`environment.yml` provided in `causal_benchmark/` pins `python=3.10`.

To create a Python 3.10 virtual environment with `venv`:

```bash
git clone https://github.com/EDavtyan/CausalWhatNot.git
cd CausalWhatNot
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r causal_benchmark/requirements.txt
```

Or with conda:

```bash
conda env create -f causal_benchmark/environment.yml
conda activate causal_benchmark
```

Before running experiments you can generate the sample datasets:

```bash
python -m causal_benchmark.utils.download_datasets
```

## Usage

Run the benchmark with the default configuration:

```bash
python -m causal_benchmark.experiments.run_benchmark --config causal_benchmark/experiments/config.yaml
```

This will evaluate each algorithm on all datasets listed in the YAML config. Outputs are written to `causal_benchmark/results/` or to a custom directory via the `--out-dir` option. Each run produces:

* `outputs/{dataset}_{algorithm}.csv` – learned adjacency matrices with node labels
* `logs/{dataset}_{algorithm}.log` – per-run status and metrics
* `logs/{dataset}_{algorithm}_diff.txt` – edge discrepancies (extra/missing/reversed)
* `logs/{dataset}_{algorithm}_stability.csv` – bootstrap edge stability frequencies when enabled
* `summary_metrics.csv` – aggregate metrics (mean and std if bootstrapping)

## Configuration

Edit `experiments/config.yaml` to select datasets, algorithms and the number of `bootstrap_runs`.
Set `record_edge_stability: true` to save edge frequencies computed over the bootstrap samples.
Datasets may be listed as just the name or as a mapping with optional `n_samples`:

```yaml
datasets:
  - asia            # uses the default number of samples
  - name: alarm
    n_samples: 2000
```

Algorithm parameters are specified in the `algorithms` section. A `timeout_s` option can be set to abort a run if it exceeds the given number of seconds:

```yaml
algorithms:
  pc:
    timeout_s: 30
```

To compute edge stability across bootstraps use:

```yaml
bootstrap_runs: 20
record_edge_stability: true
```

When a timeout occurs the run is marked as failed and the error is logged.

## Datasets

The framework generates data from four classical networks:

| Name   | Variables | Domain                           | Notes              |
|-------|-----------|----------------------------------|--------------------|
| Asia  | 8         | medical diagnostic toy example   | binary variables   |
| Sachs | 11        | protein signaling (real data)    | continuous         |
| ALARM | 37        | medical monitoring (synthetic)   | discretised        |
| Child | 20        | pediatric diagnosis (synthetic)  | discretised        |

All datasets are generated programmatically so no large files are required.

## Algorithms

| Algorithm | Characteristics | Implementation |
|-----------|----------------|----------------|
| **PC**    | Constraint-based search | `causal-learn` |
| **GES**   | Greedy equivalence search | `causal-learn` |
| **NOTEARS** | Continuous optimization with acyclicity constraint | CausalNex |
| **COSMO** | Regression-based approach enforcing an ordering | numpy / networkx |

PC and GES require the `causal-learn` package. Install it with `pip install causal-learn` or these algorithms will raise an `ImportError` when run.

Each `run()` function returns a networkx `DiGraph` and timing information. Algorithms raise an error if a cycle is detected or required dependencies are missing.

## Evaluation Metrics

The benchmark reports:

* **Precision**, **Recall** and **F1** of predicted edges (orientation ignored by default)
* **Structural Hamming Distance (SHD)** comparing the predicted and true graphs
* When `orientation_metrics: true` in the config, **directed precision/recall/F1** and **SHD_dir** which treat edge orientation as essential
* Mean and standard deviation of metrics when `bootstrap_runs` is greater than zero

## Reproducibility

Sampling functions and algorithms use fixed random seeds so repeated runs yield identical datasets and results. All intermediate artifacts are stored in `results/` (or the directory specified by `--out-dir`) enabling full experiment replay.

## Development & Contribution

Contributions are welcome! To add a new algorithm, implement a `run(data: pd.DataFrame, **params)` function in `causal_benchmark/algorithms/` that returns a `networkx.DiGraph` and an info dict. Add tests under `causal_benchmark/tests/` and update `requirements.txt` if additional dependencies are required.

Run the test suite with:

```bash
pytest -q
```

Feel free to open issues or pull requests with improvements or questions.

## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

This framework builds on excellent open-source projects including:

* [causal-learn](https://github.com/cmu-phil/causal-learn) for PC and GES
* [CausalNex](https://github.com/Microsoft/causalnex) powering NOTEARS
* [NetworkX](https://networkx.org/) for graph structures

We thank the authors of these libraries for making this benchmark possible.

