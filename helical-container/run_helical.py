import os
import time
from datetime import datetime

import anndata as ad
import scanpy as sc
import helical

DATA_DIR = os.environ.get("DATA_DIR", "/data")


def find_h5ad_files():
    print(f"[{datetime.now()}] Looking for .h5ad files in {DATA_DIR}")
    if not os.path.isdir(DATA_DIR):
        print("DATA_DIR does not exist, skipping local files.")
        return []

    files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".h5ad")
    ]
    if not files:
        print("No .h5ad files found in DATA_DIR.")
    else:
        for f in files:
            print(f"  - {f}")
    return files


def load_first_dataset(files):
    """Try to load the first .h5ad; if it fails, return None."""
    if not files:
        return None

    path = files[0]
    print(f"[{datetime.now()}] Loading dataset from disk: {path}")
    try:
        adata = ad.read_h5ad(path)
        print(f"Loaded AnnData from disk with shape: {adata.shape}")
        return adata
    except Exception as e:
        print(f"[{datetime.now()}] Failed to read {path} as .h5ad: {e!r}")
        print("Will fall back to a demo dataset instead.")
        return None


def load_demo_dataset():
    """Use a real small public dataset from scanpy."""
    print(f"[{datetime.now()}] Loading demo PBMC3k dataset via scanpy.datasets.pbmc3k()")
    adata = sc.datasets.pbmc3k()
    print(f"Demo AnnData loaded with shape: {adata.shape}")
    return adata


def run_demo_model(adata):
    print(f"[{datetime.now()}] Starting Helical demo run...")
    start = time.time()

    # Here you can plug in a real Helical model; for now we simulate work
    if adata is not None:
        print(f"AnnData is available with shape: {adata.shape}")
        print("Simulating Helical model embedding/annotation...")
        time.sleep(2.0)
    else:
        print("No AnnData available, running in metadata-only mode.")
        time.sleep(1.0)

    duration = time.time() - start
    print(f"[{datetime.now()}] Helical demo completed in {duration:.2f}s")

    version = getattr(helical, "__version__", "unknown")
    print(f"Helical package version: {version}")

    # Simple Prometheus metric
    os.makedirs("/tmp/metrics", exist_ok=True)
    metrics_path = "/tmp/metrics/helical_metrics.prom"
    with open(metrics_path, "w") as f:
        f.write(f"helical_workflow_duration_seconds {duration}\n")
    print(f"Metrics written to: {metrics_path}")


if __name__ == "__main__":
    files = find_h5ad_files()
    adata = load_first_dataset(files)

    if adata is None:
        # sample1.h5ad empty/invalid → use PBMC3k
        adata = load_demo_dataset()

    run_demo_model(adata)
