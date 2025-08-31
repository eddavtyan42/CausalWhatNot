# Causal Benchmark

Quick start:

```bash
pip install -r requirements.txt
# NOTEARS requires the optional `causalnex` dependency included in requirements
python utils/download_datasets.py
python experiments/run_benchmark.py --config experiments/config.yaml
```

To run the phase 3 sensitivity analysis and write a summary CSV:

```bash
python experiments/phase3_sensitivity_analysis.py \
    --n-samples 100 \
    --bootstrap-runs 10 \
    --out-dir results \
    --diff-logs
```

This creates `results/phase3_results.csv` and, when `--diff-logs` is set,
per-algorithm difference logs under `results/diff_logs/`.
