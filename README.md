# CausalWhatNot

![python](https://img.shields.io/badge/python-3.10%2B-blue)

A reproducible benchmarking framework for causal discovery algorithms. CausalWhatNot allows researchers and practitioners to compare multiple structure learning methods on standard benchmark datasets using a consistent set of metrics.

## Overview

The project orchestrates data generation, algorithm execution and metric computation in a single Python package. It includes canonical networks such as Asia, Sachs, ALARM and Child and currently supports four algorithms:

* **PC** – constraint-based search using conditional independence tests
* **GES** – greedy equivalence search with a BIC score
* **NOTEARS** – continuous optimization approach via the CausalNex backend
* **COSMO** – priority-based regression implementation inspired by smooth acyclic orientations

Results are saved as adjacency matrices and summary metrics so experiments can be reproduced exactly.

## Features

* Runs PC, GES, NOTEARS and COSMO on common benchmark datasets
* Bootstrap evaluation with precision, recall, F1 and structural hamming distance (SHD)
* Easily extensible for new algorithms or datasets
* Deterministic sampling with fixed seeds for reproducibility
* Python 3.10+ compatible and tested with `pytest`

## Installation

Clone the repository and install the dependencies using either `pip` or `conda`:

```bash
git clone https://github.com/EDavtyan/CausalWhatNot.git
cd CausalWhatNot
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

* `outputs/{dataset}_{algorithm}.csv` – learned adjacency matrices
* `logs/{dataset}_{algorithm}.log` – per-run status and metrics
* `summary_metrics.csv` – aggregate metrics (mean and std if bootstrapping)

Edit `experiments/config.yaml` to select datasets, algorithms, algorithm parameters or the number of `bootstrap_runs`.

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
| **PC**    | Constraint-based search | `causallearn` |
| **GES**   | Greedy equivalence search | `causallearn` |
| **NOTEARS** | Continuous optimization with acyclicity constraint | `causalnex` backend |
| **COSMO** | Regression-based approach enforcing an ordering | numpy / networkx |

Each `run()` function returns a networkx `DiGraph` and timing information. Algorithms raise an error if a cycle is detected or required dependencies are missing.

## Evaluation Metrics

The benchmark reports:

* **Precision**, **Recall** and **F1** of predicted edges (orientation ignored by default)
* **Structural Hamming Distance (SHD)** comparing the predicted and true graphs
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

* [causallearn](https://github.com/cmu-phil/causal-learn) for PC and GES
* [CausalNex](https://github.com/Microsoft/causalnex) powering NOTEARS
* [NetworkX](https://networkx.org/) for graph structures

We thank the authors of these libraries for making this benchmark possible.

