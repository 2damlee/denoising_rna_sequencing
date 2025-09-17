from __future__ import annotations
from typing import Dict, Optional, Tuple
from scipy import sparse
import time
import numpy as np
import os

DEBUG = True
_LOG_FILE: Optional[str] = os.environ.get("DCA_LOG_FILE", None)

def set_debug(flag: bool = True):
    """Enable/disable console debug prints."""
    global DEBUG
    DEBUG = bool(flag)

def set_log_file(path: Optional[str]):
    """Append logs to a file. Set None to disable file logging. (tmux)"""
    global _LOG_FILE
    _LOG_FILE = path

def _append_line(path: str, line: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(line + "\n")

def log(msg: str):
    """Lightweight logger: prints when DEBUG and optionally appends to _LOG_FILE."""
    line = f"{time.strftime('%F %T')} | {msg}"
    if DEBUG:
        print(line, flush=True)
    if _LOG_FILE:
        _append_line(_LOG_FILE, line)

def heartbeat(path: str, msg: str):
    """Write progress line to a file so you can `tail -f` it."""
    _append_line(path, f"{time.strftime('%F %T')} {msg}")

class Timer:
    """Context manager to measure wall time."""
    def __init__(self, label: str):
        self.label, self.t0 = label, None
    def __enter__(self):
        self.t0 = time.time()
        log(f"[time] {self.label} start")
        return self
    def __exit__(self, *exc):
        dt = time.time() - self.t0
        log(f"[time] {self.label} end in {dt:.2f}s")

def shape_dtype(name: str, X):
    """Print basic shape/dtype + sparsity."""
    if sparse.issparse(X):
        total = X.shape[0] * X.shape[1]
        dens = (X.nnz / total) if total else 0.0
        log(f"{name}: sparse {X.shape} dtype={X.dtype} nnz={X.nnz} ({dens:.4%} dense)")
    else:
        log(f"{name}: dense  {X.shape} dtype={X.dtype}")

def sanity_counts(name: str, X):
    """Raise early if matrix looks broken."""
    Xd = X.A if sparse.issparse(X) else X
    if Xd.size == 0:
        raise ValueError(f"{name} is empty")
    if np.isnan(Xd).any():
        raise ValueError(f"{name} has NaNs")
    if (Xd < 0).any():
        raise ValueError(f"{name} has negative values")
    approx_int = np.allclose(Xd, np.rint(Xd))
    log(f"{name}: zeros={(Xd==0).mean():.2%}, approx_int={approx_int}")

def to_int_if_close(X: np.ndarray) -> np.ndarray:
    """If X is approximately integer, round to int64; else return X unchanged."""
    X = np.asarray(X)
    if np.allclose(X, np.rint(X)):
        return np.rint(X).astype(np.int64)
    return X
    
# ===================== Masking helpers =====================

def make_mask_nonzero_by_gene(X: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """
    Mask ~frac of NONZERO entries per gene (column).
    Returns a boolean mask M with True where values are hidden.
    """
    n_cells, n_genes = X.shape
    M = np.zeros_like(X, dtype=bool)
    print(f"DEBUG, make_mask_nonzero_by_gene in utilities.py: {M}")
    for g in range(n_genes):
        nz = np.where(X[:, g] > 0)[0]
        if len(nz) == 0:
            continue
        k = max(1, int(np.ceil(frac * len(nz))))
        sel = rng.choice(nz, size=min(k, len(nz)), replace=False)
        M[sel, g] = True
    return M

def binomial_thinning_split(X: np.ndarray, p_hold: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each entry x, draw x_hold ~ Binom(x, p_hold), x_keep = x - x_hold.
    Returns (X_keep, X_hold) with same dtype as X (int-like).
    """
    X = np.asarray(X)
    if not np.issubdtype(X.dtype, np.integer):
        X_int = np.rint(X).astype(np.int64)
    else:
        X_int = X.astype(np.int64, copy=False)
    p = float(np.clip(p_hold, 0.0, 1.0))
    X_hold = rng.binomial(n=X_int, p=p)
    X_keep = X_int - X_hold
    return X_keep.astype(X.dtype), X_hold.astype(X.dtype)

# ===================== Metric helpers =====================

def _basic_errors(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    return {
        "MAE": float(np.mean(np.abs(err))),
        "MSE": float(np.mean(err**2)),
        "MedianL1": float(np.median(np.abs(err))),
    }

def nb_logpmf(x: np.ndarray, mu: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    NB log PMF with mean 'mu' and dispersion 'theta' (r=theta, p=theta/(theta+mu)).
    """
    from scipy.special import gammaln
    x = x.astype(np.float64)
    mu = np.maximum(mu.astype(np.float64), 1e-12)
    theta = np.maximum(theta.astype(np.float64), 1e-12)
    r = theta
    p = theta / (theta + mu)
    return (gammaln(x + r) - gammaln(r) - gammaln(x + 1.0)
            + r * np.log(p) + x * np.log1p(-p))

def nb_deviance(x: np.ndarray, mu: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    NB deviance per entry: 2*[ x*log(x/mu) - (x+theta)*log((x+theta)/(mu+theta)) ].
    """
    x = x.astype(np.float64)
    mu = np.maximum(mu.astype(np.float64), 1e-12)
    theta = np.maximum(theta.astype(np.float64), 1e-12)
    x_safe = np.maximum(x, 1e-12)
    term1 = x * (np.log(x_safe) - np.log(mu))
    term2 = (x + theta) * (np.log(x + theta) - np.log(mu + theta))
    return 2.0 * (term1 - term2)

def _nb_scores(y_true: np.ndarray, mu: np.ndarray, theta: np.ndarray) -> Dict[str, float]:
    """
    NB metrics averaged over entries. 'theta' should be broadcastable per entry (e.g., per-gene).
    """
    return {
        "NB_ll": float(nb_logpmf(y_true, mu, theta).mean()),
        "NB_dev": float(nb_deviance(y_true, mu, theta).mean()),
    }

def _predict_full_matrix(mdl, X_eval: np.ndarray, mask: Optional[np.ndarray], X_train_ref: Optional[np.ndarray]) -> np.ndarray:
    """
    Uniformly call predict_mean for all baselines; KNN needs a mask (bool matrix).
    """
    try:
        return mdl.predict_mean(X_eval, mask=mask, X_train_ref=X_train_ref)
    except TypeError:
        return mdl.predict_mean(X_eval)

# ===================== Biological structure (Silhouette only) =====================

def libsize_normalize_for_hvg(X: np.ndarray) -> np.ndarray:
    """
    Size-factor normalize to median library size then log1p.
    """
    lib = X.sum(axis=1, keepdims=True).astype(np.float64)
    med = np.median(lib[lib > 0]) if np.any(lib > 0) else 1.0
    sf = np.divide(lib, med, out=np.ones_like(lib), where=(lib != 0))
    Xn = X / np.maximum(sf, 1e-12)
    return np.log1p(Xn)

def bio_silhouette_score(mu_val: np.ndarray, labels_val: Optional[np.ndarray]) -> Optional[float]:
    """
    Silhouette on PCA(≤50) of libsize-normalized log1p(mu_val).
    Returns None if labels missing or not computable.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    if labels_val is None:
        return None
    y = np.asarray(labels_val)
    if len(np.unique(y)) < 2 or mu_val.shape[0] < 3:
        return None

    Xn = libsize_normalize_for_hvg(mu_val)
    n_comp = min(50, Xn.shape[1], max(2, Xn.shape[0] - 1))
    if n_comp < 2:
        return None

    Z = PCA(n_components=n_comp, random_state=0).fit_transform(Xn)
    try:
        return float(silhouette_score(Z, y))
    except Exception:
        return None

# ===================== HVG + NB dispersion =====================

def rank_hvg_by_variance(X_train_counts: np.ndarray) -> np.ndarray:
    """
    Return column indices sorted by variance on log1p(size-factor-normalized) counts.
    """
    Xlog = libsize_normalize_for_hvg(X_train_counts)
    var = Xlog.var(axis=0)
    return np.argsort(var)[::-1]

def estimate_nb_theta_moments(X_train_counts: np.ndarray) -> np.ndarray:
    """
    Per-gene NB dispersion via method-of-moments (clipped to a stable range).
    """
    mu = X_train_counts.mean(axis=0).astype(np.float64)
    var = X_train_counts.var(axis=0, ddof=1).astype(np.float64)
    theta = np.where(var > mu, (mu**2) / np.maximum(var - mu, 1e-12), 1e6)
    return np.clip(theta, 1e-6, 1e9)

# --------------- Seurat v3 HVG (batch-aware) helper ----------------

def seurat_v3_hvg_indices_for_train_split(
    X_train_counts: np.ndarray,
    gene_names: np.ndarray,
    batch_labels_train: np.ndarray,
    normalized_train_matrix: np.ndarray,  # same samples x genes as X_train_counts
    n_top_genes: int,
    layer_name: str = "log2_1p_CPM_original",
    batch_key: str = "batch",
) -> np.ndarray:
    """
    Returns indices of highly variable genes using Scanpy's seurat_v3 flavor.
    """
    try:
        import scanpy as sc
        import anndata as ad
    except Exception as e:
        raise ImportError("Seurat v3 HVG requires scanpy and anndata.") from e

    ad_tr = ad.AnnData(X=X_train_counts)
    ad_tr.var_names = np.array(gene_names, dtype=str)
    ad_tr.obs[batch_key] = np.array(batch_labels_train, dtype=str)
    ad_tr.layers[layer_name] = normalized_train_matrix
    sc.pp.highly_variable_genes(
        ad_tr,
        flavor="seurat_v3",
        n_top_genes=int(n_top_genes),
        batch_key=batch_key,
        layer=layer_name,
    )
    return np.where(ad_tr.var["highly_variable"].to_numpy())[0]
