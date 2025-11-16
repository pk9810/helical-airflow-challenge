import os
import time
from datetime import datetime

import anndata as ad
import scanpy as sc
import helical

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import one_hot
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

from helical.models.scgpt import scGPT, scGPTConfig

# Where Airflow mounts the data folder (in Docker: ./data -> /opt/data)
DATA_DIR = os.environ.get("DATA_DIR", "/data")


# ---------------------------------------------------------
# Utility helpers: small wrappers for logging and file discovery
# ---------------------------------------------------------

def log(msg: str):
    """
    Print a log message with timestamp, similar to Airflow task logs.

    Example:
        [2025-01-10 12:00:00] Loading dataset...
    """
    print(f"[{datetime.now()}] {msg}")


def find_h5ad_files():
    """
    Look inside the mounted /data directory for any .h5ad files.
    This allows users to drop in their own datasets without modifying code.

    Returns:
        A sorted list of full paths to .h5ad files.
    """
    log(f"Looking for .h5ad files in {DATA_DIR}")

    if not os.path.isdir(DATA_DIR):
        log("DATA_DIR does not exist — skipping local file search.")
        return []

    files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".h5ad")
    ]

    if not files:
        log("No .h5ad files found — will fall back to demo dataset.")
    else:
        for f in files:
            log(f"  - Found: {f}")

    return sorted(files)


def load_first_dataset(files):
    """
    Load the first .h5ad file from disk.
    If anything fails (wrong format, partial file, etc.), return None
    so the pipeline falls back to using a public Scanpy dataset.
    """
    if not files:
        log("No dataset paths provided — skipping file load.")
        return None

    path = files[0]
    log(f"Loading dataset from: {path}")

    try:
        adata = ad.read_h5ad(path)
        log(f"Loaded AnnData: shape={adata.shape}")
        return adata
    except Exception as e:
        log(f"Failed to read .h5ad ({e!r}) — using demo dataset instead.")
        return None


def load_demo_dataset():
    """
    Load Scanpy's built-in PBMC3k dataset.
    This ensures the pipeline always works — even with no local data.
    """
    log("Loading demo PBMC3k dataset via scanpy.datasets.pbmc3k()")
    adata = sc.datasets.pbmc3k()
    log(f"PBMC3k loaded: shape={adata.shape}")
    return adata


def shrink_adata(
    adata: ad.AnnData,
    max_cells: int = 300,
    max_genes: int = 2000,
) -> ad.AnnData:
    """
    Reduce dataset size to make CPU execution fast enough for Airflow/Docker.

    - Randomly keep up to `max_cells` observations (rows)
    - Randomly keep up to `max_genes` genes (columns)

    This avoids expensive Scanpy steps like HVG selection or normalization.
    """
    log(f"Original dataset shape: {adata.shape}")

    # Subsample rows (cells)
    if adata.n_obs > max_cells:
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[idx].copy()
        log(f"Subsampled cells → {adata.n_obs}")
    else:
        log(f"Cells <= {max_cells}; keeping all.")

    # Subsample columns (genes)
    if adata.n_vars > max_genes:
        rng = np.random.default_rng(42)
        gene_idx = rng.choice(adata.n_vars, size=max_genes, replace=False)
        adata = adata[:, gene_idx].copy()
        log(f"Subsampled genes → {adata.n_vars}")
    else:
        log(f"Genes <= {max_genes}; keeping all.")

    # Ensure scGPT-compatible gene name field
    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var_names.astype(str).str.upper()

    log(f"Final reduced dataset shape: {adata.shape}")
    return adata


# ---------------------------------------------------------
# Helical + scGPT embedding section
# ---------------------------------------------------------

def get_scgpt_embeddings(adata: ad.AnnData) -> np.ndarray:
    """
    Run scGPT embedding generation on the AnnData object.

    The embeddings provide a compact numerical representation of each cell.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[helical] Using device: {device}")

    scgpt_config = scGPTConfig(batch_size=32, device=device)
    scgpt = scGPT(configurer=scgpt_config)

    log("[helical] Preparing input data...")
    data = scgpt.process_data(adata, gene_names="gene_name")

    log("[helical] Computing embeddings...")
    x_scgpt = scgpt.get_embeddings(data)
    log(f"[helical] Embeddings shape: {x_scgpt.shape}")

    return x_scgpt


def maybe_get_labels(adata: ad.AnnData):
    """
    Attempt to extract labels for a quick classifier head.

    Priority:
      1. adata.obs["LVL1"]
      2. fallback → adata.obs["cell_type"]

    If neither exists or dataset has <2 classes: skip classification entirely.
    """
    if "LVL1" not in adata.obs.columns:
        if "cell_type" in adata.obs.columns:
            log("[helical] LVL1 missing — using cell_type instead.")
            adata.obs["LVL1"] = adata.obs["cell_type"].astype(str)
        else:
            log("[helical] No label column found; embeddings only.")
            return None, None

    labels = np.array(adata.obs["LVL1"].tolist())
    unique = np.unique(labels)

    if len(unique) < 2:
        log("[helical] Not enough classes for classification (<2).")
        return None, None

    # Fit sklearn label encoder
    encoder = LabelEncoder()
    y_int = encoder.fit_transform(labels)

    # One-hot encode class labels for a lightweight NN
    y_encoded = one_hot(torch.tensor(y_int), num_classes=len(encoder.classes_)).float()

    id2type = {i: cls for i, cls in enumerate(encoder.classes_)}
    log(f"[helical] Identified {len(id2type)} cell types: {id2type}")

    return y_encoded, id2type


def train_small_head(
    input_dim: int,
    num_classes: int,
    X: np.ndarray,
    y_encoded: torch.Tensor,
    num_epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> nn.Sequential:
    """
    Train a tiny feedforward classifier on scGPT embeddings.

    The goal isn't SOTA performance — the goal is:
        • fast execution in Airflow
        • proof that embeddings → cell type prediction pipeline works
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, num_classes),
    )

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(torch.from_numpy(X_train), y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.from_numpy(X_val), y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Basic training loop — very quick
    model.train()
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Quick validation pass
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_outputs = model(val_X)
                val_losses.append(loss_fn(val_outputs, val_y).item())
        avg_val = np.mean(val_losses)

        log(f"[helical] Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val:.4f}")
        model.train()

    model.eval()
    return model


def evaluate_metrics(name: str, y_true, y_pred):
    """
    Print + return accuracy, precision, f1, recall.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")

    log(f"[helical] {name} accuracy:  {acc*100:.2f}%")
    log(f"[helical] {name} precision: {prec*100:.2f}%")
    log(f"[helical] {name} f1-score:  {f1*100:.2f}%")
    log(f"[helical] {name} recall:    {rec*100:.2f}%")

    return {"accuracy": acc, "precision": prec, "f1": f1, "recall": rec}


def write_prometheus_metrics(
    duration: float,
    results: dict | None,
    n_cells: int,
    n_genes: int,
):
    """
    Write workflow metrics for Airflow → StatsD → Prometheus → Grafana pipeline.

    This produces a .prom file that Prometheus textfile_exporter can read.
    """
    os.makedirs("/tmp/metrics", exist_ok=True)
    metrics_path = "/tmp/metrics/helical_metrics.prom"

    with open(metrics_path, "w") as f:
        f.write(f"helical_workflow_duration_seconds {duration}\n")
        f.write(f"helical_dataset_cells {n_cells}\n")
        f.write(f"helical_dataset_genes {n_genes}\n")

        if results is not None:
            f.write(f"helical_scgpt_celltype_accuracy {results['accuracy']}\n")
            f.write(f"helical_scgpt_celltype_precision {results['precision']}\n")
            f.write(f"helical_scgpt_celltype_f1 {results['f1']}\n")
            f.write(f"helical_scgpt_celltype_recall {results['recall']}\n")

    log(f"Prometheus metrics written to: {metrics_path}")


# ---------------------------------------------------------
# Main pipeline function — this is what your Airflow DAG calls
# ---------------------------------------------------------

def run_helical_small_pipeline():
    """
    End-to-end lightweight Helical pipeline:

    1. Check /data for .h5ad files
    2. Load first one found, or fall back to PBMC3k
    3. Shrink AnnData massively (fast CPU runs)
    4. Compute scGPT embeddings
    5. Optionally train a tiny classifier head if labels exist
    6. Write summary metrics for Prometheus/Grafana

    This is intentionally optimized for:
        • MacBooks / CPUs
        • Airflow's DockerOperator
        • Completion within ~2–15 minutes
    """
    log("Starting Helical small pipeline...")
    t0 = time.time()

    # Step 1: Look for local .h5ad dataset
    files = find_h5ad_files()
    adata = load_first_dataset(files)

    # Step 2: Use fallback if loading fails
    if adata is None:
        adata = load_demo_dataset()

    # Step 3: Shrink dataset dramatically
    adata = shrink_adata(adata, max_cells=300, max_genes=2000)
    n_cells, n_genes = adata.shape

    # Step 4: scGPT embedding generation
    x_scgpt = get_scgpt_embeddings(adata)

    # Step 5: Optional classifier
    results = None
    y_encoded, id2type = maybe_get_labels(adata)

    if y_encoded is not None:
        input_dim = x_scgpt.shape[1]
        num_classes = y_encoded.shape[1]

        head_model = train_small_head(
            input_dim=input_dim,
            num_classes=num_classes,
            X=x_scgpt,
            y_encoded=y_encoded,
            num_epochs=3,
            batch_size=64,
            lr=1e-3,
        )

        logits = head_model(torch.from_numpy(x_scgpt))
        y_pred_idx = torch.argmax(logits, dim=1).numpy()
        y_pred = [id2type[i] for i in y_pred_idx]
        y_true = np.array(adata.obs["LVL1"].tolist())

        results = evaluate_metrics("Full tiny set", y_true, y_pred)
    else:
        log("[helical] No labels detected — skipping classifier head.")

    duration = time.time() - t0
    log(f"Helical pipeline completed in {duration:.2f}s")

    version = getattr(helical, "__version__", "unknown")
    log(f"Helical package version: {version}")

    # Step 6: Export metrics for monitoring
    write_prometheus_metrics(duration, results, n_cells, n_genes)


if __name__ == "__main__":
    # Allow developer to run this locally as: python script.py
    run_helical_small_pipeline()
