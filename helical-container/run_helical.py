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

DATA_DIR = os.environ.get("DATA_DIR", "/data")


# -------------------------
# Utility helpers
# -------------------------

def log(msg: str):
    print(f"[{datetime.now()}] {msg}")


def find_h5ad_files():
    """
    Look for .h5ad files in the mounted DATA_DIR.
    We only use the first one found to keep things simple.
    """
    log(f"Looking for .h5ad files in {DATA_DIR}")
    if not os.path.isdir(DATA_DIR):
        log("DATA_DIR does not exist, skipping local files.")
        return []

    files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".h5ad")
    ]
    if not files:
        log("No .h5ad files found in DATA_DIR.")
    else:
        for f in files:
            log(f"  - {f}")
    return sorted(files)


def load_first_dataset(files):
    """
    Try to load the first .h5ad from the mounted /data.
    If it fails, return None (we'll fallback to PBMC3k).
    """
    if not files:
        log("No .h5ad files passed in; nothing to load from /data.")
        return None

    path = files[0]
    log(f"Loading dataset from disk: {path}")
    try:
        adata = ad.read_h5ad(path)
        log(f"Loaded AnnData from disk with shape: {adata.shape}")
        return adata
    except Exception as e:
        log(f"Failed to read {path} as .h5ad: {e!r}")
        log("Will fall back to a demo dataset instead.")
        return None


def load_demo_dataset():
    """
    PBMC3k fallback (small public dataset from scanpy).
    This is purely to make the pipeline robust + fast.
    """
    log("Loading demo PBMC3k dataset via scanpy.datasets.pbmc3k()")
    adata = sc.datasets.pbmc3k()
    log(f"Demo AnnData loaded with shape: {adata.shape}")
    return adata


def shrink_adata(
    adata: ad.AnnData,
    max_cells: int = 300,
    max_genes: int = 2000,
) -> ad.AnnData:
    """
    Downsample the AnnData object so everything runs quickly on CPU.

    - Randomly sample up to max_cells cells
    - Randomly sample up to max_genes genes (no HVG / scanpy magic)

    We avoid scanpy.pp.highly_variable_genes to prevent inf/bins errors.
    """
    log(f"Original AnnData shape: {adata.shape}")

    # Subsample cells
    if adata.n_obs > max_cells:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[idx].copy()
        log(f"Subsampled to {adata.n_obs} cells.")
    else:
        log(f"n_obs <= {max_cells}; keeping all cells ({adata.n_obs}).")

    # Subsample genes by random choice (no HVG / no scanpy.cut)
    if adata.n_vars > max_genes:
        rng = np.random.default_rng(seed=42)
        gene_idx = rng.choice(adata.n_vars, size=max_genes, replace=False)
        adata = adata[:, gene_idx].copy()
        log(f"Subsampled genes to {adata.n_vars} genes.")
    else:
        log(f"n_vars <= {max_genes}; keeping all genes ({adata.n_vars}).")

    # Ensure we have a gene_name column for Helical
    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var_names.astype(str).str.upper()

    log(f"Final shrunk AnnData shape: {adata.shape}")
    return adata


# -------------------------
# Helical + scGPT bits
# -------------------------

def get_scgpt_embeddings(adata: ad.AnnData) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[helical] Using device: {device}")

    scgpt_config = scGPTConfig(batch_size=32, device=device)
    scgpt = scGPT(configurer=scgpt_config)

    log("[helical] Processing data for scGPT...")
    data = scgpt.process_data(adata, gene_names="gene_name")

    log("[helical] Computing embeddings...")
    x_scgpt = scgpt.get_embeddings(data)
    log(f"[helical] Embeddings shape: {x_scgpt.shape}")
    return x_scgpt


def maybe_get_labels(adata: ad.AnnData):
    """
    Try to extract LVL1 labels if present.
    If LVL1 missing but 'cell_type' exists, use that as LVL1.
    If not enough classes, return (None, None) and skip classification.
    """
    # If LVL1 not present, but a common label column exists, promote it.
    if "LVL1" not in adata.obs.columns:
        if "cell_type" in adata.obs.columns:
            log("[helical] LVL1 not found; using 'cell_type' as LVL1.")
            adata.obs["LVL1"] = adata.obs["cell_type"].astype(str)
        else:
            log("[helical] No LVL1 column found in obs; skipping classification.")
            return None, None

    labels = np.array(adata.obs["LVL1"].tolist())
    unique = np.unique(labels)
    if unique.shape[0] < 2:
        log("[helical] LVL1 has <2 unique labels; skipping classification.")
        return None, None

    # Encode labels, using encoder.classes_ as canonical mapping
    encoder = LabelEncoder()
    y_int = encoder.fit_transform(labels)
    num_types = len(encoder.classes_)
    y_encoded = one_hot(torch.tensor(y_int), num_types).float()

    # id2type mapping aligned with encoder ordering
    id2type = {i: cls for i, cls in enumerate(encoder.classes_)}

    log(f"[helical] Found {num_types} LVL1 cell types: {id2type}")
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
    Very small classifier head, few epochs only.
    Enough to prove "cell type annotation" without killing the CPU.
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, num_classes),
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(torch.from_numpy(X_train), y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.from_numpy(X_val), y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # quick validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_outputs = model(val_X)
                val_loss = loss_fn(val_outputs, val_y)
                val_losses.append(val_loss.item())
        avg_val = sum(val_losses) / len(val_losses)
        log(f"[helical] Epoch {epoch + 1}/{num_epochs} - Val loss: {avg_val:.4f}")
        model.train()

    model.eval()
    return model


def evaluate_metrics(name: str, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")

    log(f"[helical] {name} accuracy:  {acc * 100:.2f}%")
    log(f"[helical] {name} precision: {prec * 100:.2f}%")
    log(f"[helical] {name} f1:        {f1 * 100:.2f}%")
    log(f"[helical] {name} recall:    {rec * 100:.2f}%")

    return {"accuracy": acc, "precision": prec, "f1": f1, "recall": rec}


def write_prometheus_metrics(
    duration: float,
    results: dict | None,
    n_cells: int,
    n_genes: int,
):
    """
    Export simple metrics in Prometheus textfile format.
    If results is None → only duration + dataset stats.
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
    log(f"Metrics written to: {metrics_path}")


# -------------------------
# Main pipeline
# -------------------------

def run_helical_small_pipeline():
    """
    Small, CPU-friendly Helical pipeline:

    1. Look for mounted .h5ad in /data
    2. Try to load it; if it fails, use PBMC3k
    3. Shrink to very small AnnData (e.g. 300 x 2000; random genes)
    4. Run scGPT embeddings
    5. OPTIONAL: if LVL1 (or cell_type) present → tiny classifier head
    6. Export duration + metrics to /tmp/metrics/helical_metrics.prom
    """
    log("Starting Helical small cell type demo...")
    t0 = time.time()

    # 1) Use mounted data if possible
    files = find_h5ad_files()
    adata = load_first_dataset(files)

    # 2) Fallback to PBMC3k if local file invalid / missing
    if adata is None:
        adata = load_demo_dataset()

    # 3) Shrink dataset heavily for Mac CPU
    adata = shrink_adata(adata, max_cells=300, max_genes=2000)
    n_cells, n_genes = adata.shape

    # 4) Run scGPT embeddings on this tiny dataset
    x_scgpt = get_scgpt_embeddings(adata)

    # 5) Optional classification, only if we can infer LVL1
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
            num_epochs=3,   # very short
            batch_size=64,
            lr=1e-3,
        )

        logits = head_model(torch.from_numpy(x_scgpt))
        y_pred_idx = np.array(torch.argmax(logits, dim=1))
        y_pred = [id2type[i] for i in y_pred_idx]
        y_true = np.array(adata.obs["LVL1"].tolist())

        results = evaluate_metrics("Full tiny set", y_true, y_pred)
    else:
        log("[helical] Skipping classifier head; only embeddings were computed.")

    duration = time.time() - t0
    log(f"Helical small demo completed in {duration:.2f}s")

    version = getattr(helical, "__version__", "unknown")
    log(f"Helical package version: {version}")

    write_prometheus_metrics(duration, results, n_cells=n_cells, n_genes=n_genes)


if __name__ == "__main__":
    # This satisfies:
    #  - "mount a local folder containing data" → we scan /data for .h5ad
    #  - run Helical model on a *small* dataset so it finishes within ~15 min on CPU
    run_helical_small_pipeline()



