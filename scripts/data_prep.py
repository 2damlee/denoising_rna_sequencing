# scripts/data_prep.py
# Utilities to inspect AnnData, invert log2(1+CPM) to counts, and prepare scVI-ready datasets.

from __future__ import annotations
import os
from typing import Dict, Any, Tuple

import numpy as np
import anndata as ad
import scanpy as sc
import scipy.sparse as sp


# ---------- Basic helpers ----------
def _x_to_dense(X):
    """Return a dense ndarray view of X (safe for small slices/printing)."""
    return X.A if hasattr(X, "A") else (X.toarray() if sp.issparse(X) else X)

def _zero_fraction(X) -> float:
    """Compute fraction of zeros in X without densifying large matrices."""
    if sp.issparse(X):
        total = X.shape[0] * X.shape[1]
        return float(1.0 - (X.nnz / max(1, total)))
    # dense
    return float(np.mean(X == 0))


# ---------- 1) Quick summary ----------
def print_adata_summary(adata: ad.AnnData, n_preview: int = 5) -> None:
    """Print shape, dtype, first-row sample, layers, obs/var heads, zero-fraction, min/max."""
    print(f"Shape: {adata.shape}")  # (n_cells, n_genes)

    # X dtype and example values
    X = adata.X
    print("X dtype:", X.dtype)
    try:
        row0 = _x_to_dense(X[0:1, :min(10, adata.n_vars)])[0]
    except Exception:
        row0 = _x_to_dense(X)[:1, :min(10, adata.n_vars)].reshape(-1)
    print("Example values from X:", row0)

    # Layers
    print("Layers:", list(adata.layers.keys()))

    # obs / var
    print("\n.obs columns:", adata.obs.columns.tolist())
    print("\n.obs sample (first 5 rows):")
    print(adata.obs.head(n_preview))
    print("\n.var columns:", adata.var.columns.tolist())
    print("\n.var sample (first 5 rows):")
    print(adata.var.head(n_preview))

    # Zero fraction and min/max
    zfrac = _zero_fraction(X)
    print(f"\nFraction of zeros in X: {zfrac:.2%}")
    try:
        # Min/Max on sparse uses data; on dense uses full array
        if sp.issparse(X):
            xmin = X.data.min() if X.nnz > 0 else 0.0
            xmax = X.data.max() if X.nnz > 0 else 0.0
        else:
            xmin, xmax = float(np.min(X)), float(np.max(X))
        print("Min value in X:", xmin)
        print("Max value in X:", xmax)
    except Exception as e:
        print("Min/Max check skipped:", repr(e))


# ---------- 2) Invert log2(1+CPM) -> integer counts ----------
def invert_log2cpm_to_counts(
    adata: ad.AnnData,
    totals_col: str = "total_counts_after_preprocessing",
    out_layer: str = "counts",
    clip_negative: bool = True
) -> np.ndarray:
    """
    Invert X ≈ log2(1 + CPM) back to integer counts using per-cell totals in adata.obs[totals_col].
    Writes integer counts to adata.layers[out_layer] and returns the counts array (dense).
    """
    if totals_col not in adata.obs.columns:
        raise ValueError(f"'{totals_col}' not found in adata.obs")

    X = _x_to_dense(adata.X).astype(float)
    totals = adata.obs[totals_col].to_numpy().astype(float)

    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        raise ValueError("Invalid totals: must be positive and finite per cell.")

    CPM = np.power(2.0, X) - 1.0
    if clip_negative:
        CPM[CPM < 0] = 0.0

    counts_float = CPM * (totals[:, None] / 1e6)
    counts_float = np.nan_to_num(counts_float, nan=0.0, posinf=0.0, neginf=0.0)

    counts_int = np.rint(counts_float).astype(np.int64)
    if clip_negative:
        counts_int[counts_int < 0] = 0

    adata.layers[out_layer] = counts_int

    # Quick QC
    row_sums = counts_int.sum(axis=1)
    corr = float(np.corrcoef(row_sums, totals)[0, 1]) if len(totals) > 1 else float("nan")
    print("Zero fraction (counts):", float(np.mean(counts_int == 0)))
    print("Row-sum corr vs totals:", corr)
    print("Median abs diff:", float(np.median(np.abs(row_sums - totals))))
    print("Min/Max counts:", int(counts_int.min()), int(counts_int.max()))
    return counts_int


# ---------- 3) Prepare scVI datasets ----------
def prepare_scvi_datasets(
    adata: ad.AnnData,
    out_dir: str = "../data/processed/scvi_prepared",
    n_hvg: int = 4000,
    n_mean: int = 4000,
    totals_col: str = "total_counts_after_preprocessing",
    batch_key: str = "cohort",
    min_counts_filter: int = 10,
    prefix: str = "dataset",
    save: bool = True
) -> Dict[str, Any]:
    """
    Prepare datasets for scVI training from AnnData with X ≈ log2(1+CPM).
    Creates:
      - RAW counts dataset
      - HVG dataset (Seurat v3 HVG selection)
      - Top-Mean dataset (top genes by mean counts)
    Returns paths and in-memory AnnData objects.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Reconstruct counts into a raw AnnData
    counts_int = invert_log2cpm_to_counts(adata, totals_col=totals_col, out_layer="counts")
    X = _x_to_dense(adata.X)

    row_sums = counts_int.sum(axis=1)
    qc = {
        "zero_fraction_counts": float(np.mean(counts_int == 0)),
        "row_sum_corr_vs_totals": float(np.corrcoef(row_sums, adata.obs[totals_col].to_numpy().astype(float))[0, 1]) if adata.n_obs > 1 else float("nan"),
        "median_abs_diff": float(np.median(np.abs(row_sums - adata.obs[totals_col].to_numpy().astype(float)))),
        "min_counts": int(counts_int.min()),
        "max_counts": int(counts_int.max()),
    }

    adata_raw = ad.AnnData(
        X=counts_int,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        uns=adata.uns.copy()
    )
    adata_raw.layers["log2_1p_CPM_original"] = X
    adata_raw.layers["counts"] = counts_int
    adata_raw.uns["scvi_preprocess"] = {
        "totals_col": totals_col,
        "batch_key_hint": batch_key,
        "qc": qc,
    }

    raw_path = os.path.join(out_dir, f"{prefix}_raw_counts.h5ad")
    if save:
        adata_raw.write(raw_path, compression="gzip")

    # HVG dataset (on counts)
    adata_for_hvg = adata_raw.copy()
    if min_counts_filter > 0:
        sc.pp.filter_genes(adata_for_hvg, min_counts=min_counts_filter)

    n_hvg_eff = int(min(n_hvg, adata_for_hvg.n_vars))
    sc.pp.highly_variable_genes(adata_for_hvg, n_top_genes=n_hvg_eff, flavor="seurat_v3")
    hvg_mask = adata_for_hvg.var["highly_variable"].to_numpy()
    adata_hvg = adata_for_hvg[:, hvg_mask].copy()
    adata_hvg.X = adata_hvg.layers["counts"].copy()
    hvg_path = os.path.join(out_dir, f"{prefix}_hvg{n_hvg_eff}.h5ad")
    if save:
        adata_hvg.write(hvg_path, compression="gzip")

    # Top-Mean dataset (on counts)
    counts_raw = adata_raw.layers["counts"]
    counts_dense = counts_raw.A if hasattr(counts_raw, "A") else counts_raw
    n_mean_eff = int(min(n_mean, counts_dense.shape[1]))
    mean_per_gene = counts_dense.mean(axis=0)
    top_idx = np.argsort(mean_per_gene)[-n_mean_eff:]
    adata_mean = adata_raw[:, top_idx].copy()
    adata_mean.X = adata_mean.layers["counts"].copy()
    mean_path = os.path.join(out_dir, f"{prefix}_topmean{n_mean_eff}.h5ad")
    if save:
        adata_mean.write(mean_path, compression="gzip")

    return {
        "qc": qc,
        "raw": {"adata": adata_raw, "path": raw_path if save else None},
        "hvg": {"adata": adata_hvg, "path": hvg_path if save else None},
        "topmean": {"adata": adata_mean, "path": mean_path if save else None},
    }


# ---------- 4) Small pretty-printer ----------
def show_sample(adata: ad.AnnData, name: str, n_cells: int = 5, n_genes: int = 5) -> None:
    """Print a small slice of X, and head of obs/var."""
    X_dense = _x_to_dense(adata.X)
    print(f"\n{name} shape:", adata.shape)
    print("--- X sample (first n_cells x n_genes) ---")
    print(np.asarray(X_dense[:n_cells, :n_genes]))
    print("--- obs sample ---")
    print(adata.obs.head(n_cells))
    print("--- var sample ---")
    print(adata.var.head(n_genes))


__all__ = [
    "print_adata_summary",
    "invert_log2cpm_to_counts",
    "prepare_scvi_datasets",
    "show_sample",
]
