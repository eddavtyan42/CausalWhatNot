import json
import hashlib
import importlib
import subprocess
import platform
import sys
import pandas as pd
import numpy as np
import networkx as nx
from importlib import metadata
from pathlib import Path
from typing import Dict, Any, List, Tuple

def calculate_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_library_versions() -> Dict[str, str]:
    """Get versions of key libraries."""
    libs = {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "networkx": nx.__version__,
    }
    optional_packages = {
        "scikit-learn": ("sklearn", "scikit-learn"),
        "causal-learn": ("causallearn", "causal-learn"),
        "causalnex": ("causalnex", "causalnex"),
        "torch": ("torch", "torch"),
        "statsmodels": ("statsmodels", "statsmodels"),
        "scipy": ("scipy", "scipy"),
        "joblib": ("joblib", "joblib"),
    }
    for display_name, (module_name, package_name) in optional_packages.items():
        version_value = _get_optional_version(module_name, package_name)
        if version_value is not None:
            libs[display_name] = version_value
    return libs

def _get_optional_version(module_name: str, package_name: str) -> str | None:
    """Return module version if available, otherwise fallback to package metadata."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None
    module_version = getattr(module, "__version__", None)
    if module_version:
        return module_version
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"

def capture_environment_snapshot() -> str | None:
    """Capture a pip freeze snapshot for provenance."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()

def save_run_metadata(
    output_path: Path,
    dataset_name: str,
    dataset_path: Path,
    n_samples: int,
    algorithm_name: str,
    algorithm_params: Dict[str, Any],
    random_seed: int | None = None,
    preprocessing_info: Dict[str, Any] | None = None,
    include_environment_snapshot: bool = False,
):
    """Save metadata for a single experiment run."""
    environment_snapshot = None
    if include_environment_snapshot:
        environment_snapshot = capture_environment_snapshot()
    metadata = {
        "dataset": {
            "name": dataset_name,
            "path": str(dataset_path),
            "sha256": calculate_file_hash(dataset_path) if dataset_path.exists() else None,
            "n_samples_used": n_samples,
        },
        "algorithm": {
            "name": algorithm_name,
            "parameters": algorithm_params,
        },
        "environment": {
            "random_seed": random_seed,
            "libraries": get_library_versions(),
            "environment_snapshot": environment_snapshot,
        },
        "preprocessing": preprocessing_info or {},
    }
    
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

def save_graph_artifacts(
    output_dir: Path,
    base_filename: str,
    graph: nx.DiGraph,
    nodes: List[str],
    raw_adjacency: np.ndarray | None = None,
):
    """Save graph artifacts: raw adjacency, binarized adjacency, and edge list."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save raw weighted adjacency (high precision)
    if raw_adjacency is not None:
        raw_df = pd.DataFrame(raw_adjacency, index=nodes, columns=nodes)
        raw_df.to_csv(output_dir / f"{base_filename}_raw_adj.csv", float_format="%.10g")
    
    # 2. Save final binarized adjacency
    bin_adj = nx.to_numpy_array(graph, nodelist=nodes, weight=None)
    bin_df = pd.DataFrame(bin_adj, index=nodes, columns=nodes)
    bin_df.to_csv(output_dir / f"{base_filename}_adj.csv")
    
    # 3. Save explicit edge list
    edge_list = []
    for u, v in graph.edges():
        edge_list.append({"source": u, "target": v})
    
    with open(output_dir / f"{base_filename}_edges.json", "w") as f:
        json.dump(edge_list, f, indent=2)
