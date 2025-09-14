from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np
import pandas as pd
import json, os
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.impute import KNNImputer

import scripts.utilities as ut

np.random.seed(123)


# ---------------- Baseline model wrappers ----------------
class _MeanModel:
    name = "MEAN"
    def fit(self, X: np.ndarray):
        self.col_ = X.mean(axis=0)
        return self
    def predict_mean(self, X: np.ndarray, mask: Optional[np.ndarray] = None, X_train_ref: Optional[np.ndarray] = None) -> np.ndarray:
        return np.broadcast_to(self.col_, X.shape)

class _MedianModel:
    name = "MEDIAN"
    def fit(self, X: np.ndarray):
        self.col_ = np.median(X, axis=0)
        return self
    def predict_mean(self, X: np.ndarray, mask: Optional[np.ndarray] = None, X_train_ref: Optional[np.ndarray] = None) -> np.ndarray:
        return np.broadcast_to(self.col_, X.shape)

class _KNNModel:
    name = "KNN"
    def __init__(self, n_neighbors: int = 15, weights: str = "uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.imp = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric="nan_euclidean")
        self.X_train_ = None
    def fit(self, X: np.ndarray):
        self.X_train_ = X
        return self
    def predict_mean(self, X: np.ndarray, mask: Optional[np.ndarray] = None, X_train_ref: Optional[np.ndarray] = None) -> np.ndarray:
        if mask is None:
            raise ValueError("KNN baseline requires a boolean mask to mark hidden entries.")
        if X_train_ref is None:
            X_train_ref = self.X_train_
        X_eval = X.copy().astype(np.float64)
        X_eval[mask] = np.nan
        X_concat = np.vstack([X_train_ref, X_eval])
        X_imputed = self.imp.fit_transform(X_concat)
        return X_imputed[-X.shape[0]:, :]

class _MAGICModel:
    name = "MAGIC"
    def __init__(self, n_pca=None, t=3, knn=5):
        self.n_pca = n_pca
        self.t = t
        self.knn = knn
        self.op_ = None
    def fit(self, X: np.ndarray):
        from magic import MAGIC
        self.op_ = MAGIC(n_pca=self.n_pca, t=self.t, knn=self.knn)
        self.op_.fit(np.asarray(X, dtype=np.float64))
        return self
    def predict_mean(self, X: np.ndarray, mask: Optional[np.ndarray] = None, X_train_ref: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Prefer inductive transform if available; otherwise fall back to concat-fit
        using the provided training reference (no leakage into metrics).
        """
        from magic import MAGIC
        if self.op_ is None:
            raise RuntimeError("Call fit() before predict_mean().")
        X = np.asarray(X, dtype=np.float64)
        if hasattr(self.op_, "transform"):
            try:
                return np.asarray(self.op_.transform(X), dtype=np.float64)
            except Exception:
                pass
        if X_train_ref is not None:
            X_tr = np.asarray(X_train_ref, dtype=np.float64)
            X_concat = np.vstack([X_tr, X])
            tmp = MAGIC(n_pca=self.n_pca, t=self.t, knn=self.knn)
            X_imp = tmp.fit_transform(X_concat)
            return np.asarray(X_imp[-X.shape[0]:, :], dtype=np.float64)
        tmp = MAGIC(n_pca=self.n_pca, t=self.t, knn=self.knn)
        return np.asarray(tmp.fit_transform(X), dtype=np.float64)

def build_baseline(name: str, **params) -> Any:
    if name.upper() == "MEAN":   return _MeanModel()
    if name.upper() == "MEDIAN": return _MedianModel()
    if name.upper() == "KNN":    return _KNNModel(n_neighbors=int(params.get("n_neighbors", 15)),
                                                 weights=params.get("weights", "uniform"))
    if name.upper() == "MAGIC":  return _MAGICModel(**params)
    raise ValueError(f"Unknown baseline: {name}")


# ---------- Minimal DCA wrapper (counts -> expected counts mu_hat) ----------
class DCAPredictWrapper:
    """
    DCA predictor that avoids leakage by concatenating [train || eval] and marking
    train rows for optimization. It does not rely on legacy TF/Keras patches.
    If tensorflow-metal is installed on Apple Silicon, TF will use the GPU (MPS) automatically.
    """
    name = "DCA"

    def __init__(self, hidden_size=[64, 32, 64], epochs=300, batch_size=128, random_state: int = 0):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.X_train_ = None

    def fit(self, X_train: np.ndarray):
        # We cache the train matrix; actual optimization happens during predict via concat,
        # so the model always trains only on the train rows (no leakage).
        X_train = np.asarray(X_train)
        if not np.issubdtype(X_train.dtype, np.floating):
            X_train = X_train.astype(np.float32, copy=False)
        self.X_train_ = X_train
        return self

    def predict_mean(self, X_eval: np.ndarray) -> np.ndarray:
        import anndata as ad
        from dca.api import dca

        if self.X_train is None and self.X_train_ is None:
            raise RuntimeError("Call fit(X_train) before predict_mean.")
        X_eval = np.asarray(X_eval)
        if not np.issubdtype(X_eval.dtype, np.floating):
            X_eval = X_eval.astype(np.float32, copy=False)

        # Concat and mark only train rows for optimization
        X_concat = np.vstack([self.X_train_, X_eval]).astype(np.float32, copy=False)
        n_tr = self.X_train_.shape[0]
        n_ev = X_eval.shape[0]

        ad_all = ad.AnnData(X_concat)
        split = np.array(["train"] * n_tr + ["test"] * n_ev, dtype=object)
        ad_all.obs["dca_split"] = split

        # Train with return_model=False; DCA writes results into adata
        dca(ad_all,
            hidden_size=self.hidden_size,
            epochs=self.epochs,
            batch_size=self.batch_size,
            return_model=False,
            copy=False,
            random_state=self.random_state)

        # DCA typically writes to layers["X_dca"]; fallback to X if absent
        if "X_dca" in ad_all.layers:
            return np.asarray(ad_all.layers["X_dca"][-n_ev:, :])
        return np.asarray(ad_all.X[-n_ev:, :])

# ---------- scVI wrapper (counts -> expected counts mu_hat) ----------
class SCVIWrapper:
    """
    Thin wrapper around scvi-tools' SCVI model.
    - Uses NB likelihood.
    - Auto-selects Lightning accelerator ('mps' on Apple Silicon if available).
    """
    name = "scVI"

    def __init__(self,
                 n_latent=20,
                 n_hidden=128,
                 n_layers=2,
                 dropout_rate=0.1,
                 max_epochs=400,
                 lr=1e-3,
                 weight_decay=1e-6,
                 batch_size: Optional[int] = None,
                 batch_key: str = "batch",
                 use_gpu: bool = True,
                 gene_names: Optional[np.ndarray] = None,
                 batches: Optional[np.ndarray] = None):
        self.hp = dict(n_latent=n_latent, n_hidden=n_hidden, n_layers=n_layers,
                       dropout_rate=dropout_rate, max_epochs=max_epochs, lr=lr,
                       weight_decay=weight_decay, batch_size=batch_size,
                       batch_key=batch_key, use_gpu=use_gpu)
        self._gene_names = gene_names
        self._batches = batches
        self.model_ = None
        self.train_genes_ = None

    @staticmethod
    def _accel_devices(use_gpu_flag: bool) -> Tuple[str, Any]:
        if not use_gpu_flag:
            return "cpu", "auto"
        try:
            import torch
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps", 1
        except Exception:
            pass
        # CPU fallback if no CUDA/MPS
        return "cpu", "auto"

    def fit(self, X_train: np.ndarray):
        import anndata as ad
        import scvi
        # Ensure gene names length matches columns; if absent, synthesize names
        if self._gene_names is None or len(self._gene_names) != X_train.shape[1]:
            self._gene_names = np.array([f"g{i}" for i in range(X_train.shape[1])], dtype=str)

        adata = ad.AnnData(X=X_train)
        adata.var_names = self._gene_names
        if self._batches is not None:
            adata.obs[self.hp["batch_key"]] = np.array(self._batches, dtype=str)

        scvi.model.SCVI.setup_anndata(
            adata,
            batch_key=(self.hp["batch_key"] if self._batches is not None else None)
        )

        self.model_ = scvi.model.SCVI(
            adata,
            n_latent=self.hp["n_latent"],
            n_hidden=self.hp["n_hidden"],
            n_layers=self.hp["n_layers"],
            dropout_rate=self.hp["dropout_rate"],
            gene_likelihood="nb",
        )

        accelerator, devices = self._accel_devices(self.hp["use_gpu"])
        train_kwargs = dict(
            max_epochs=self.hp["max_epochs"],
            plan_kwargs={"lr": self.hp["lr"], "weight_decay": self.hp["weight_decay"]},
            accelerator=accelerator,
            devices=devices,
        )
        if self.hp["batch_size"] is not None:
            train_kwargs["batch_size"] = int(self.hp["batch_size"])
        self.model_.train(**train_kwargs)

        self.train_genes_ = np.array(adata.var_names)
        return self

    def predict_mean(self, X_eval: np.ndarray, eval_batches: Optional[np.ndarray] = None) -> np.ndarray:
        import anndata as ad
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        adata_eval = ad.AnnData(X=X_eval)
        adata_eval.var_names = self.train_genes_
        if self._batches is not None:
            if eval_batches is None:
                raise KeyError(
                    f"Expected eval_batches for batch_key='{self.hp['batch_key']}' "
                    f"with length {X_eval.shape[0]}"
                )
            adata_eval.obs[self.hp["batch_key"]] = np.array(eval_batches, dtype=str)

        # Prefer model likelihood parameters; fall back to normalized expression
        try:
            px = self.model_.get_likelihood_parameters(adata=adata_eval)
            mu = np.asarray(px.get("mean", px.get("mu")))
            if mu is None:
                raise KeyError
            return mu
        except Exception:
            lib_eval = X_eval.sum(axis=1, keepdims=True).astype(np.float64)
            norm = np.asarray(self.model_.get_normalized_expression(adata=adata_eval, library_size=1.0))
            return norm * np.maximum(lib_eval, 1e-12)


# ---------- DCA 5-fold CV with both masking protocols ----------
def cv_dca_5fold(
    X_counts: np.ndarray,
    k: int = 5,
    dca_params: Optional[Dict[str, Any]] = None,
    n_hvg: int = 2000,
    R: int = 3,
    mask_frac: float = 0.10,
    thinning_p: float = 0.10,
    random_state: int = 123,
    # HVG
    hvg_mode: str = "variance",
    gene_names: Optional[np.ndarray] = None,
    batches: Optional[np.ndarray] = None,
    norm_layer: Optional[np.ndarray] = None,
    batch_key: str = "batch",
    seurat_layer_name: str = "log2_1p_CPM_original",
    # labels for Silhouette
    labels: Optional[np.ndarray] = None,
    # CSV export
    save_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    5-fold CV for DCA with two masking protocols (Nonzero Zeroing + Binomial Thinning).
    Metrics: MAE, MSE, MedianL1, NB log-likelihood, NB deviance, Silhouette.
    HVG is computed per train fold (fixed n_hvg).
    """
    from sklearn.model_selection import KFold

    if dca_params is None:
        dca_params = dict(hidden_size=[64, 32, 64], epochs=300, batch_size=128)

    X_counts = np.asarray(X_counts)
    n_samples, n_genes = X_counts.shape
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # seurat_v3 inputs if selected
    if hvg_mode == "seurat_v3":
        if gene_names is None or len(gene_names) != n_genes:
            raise ValueError("gene_names (len == n_genes) required for seurat_v3 HVG.")
        if batches is None or len(batches) != n_samples:
            raise ValueError("batches (len == n_samples) required for seurat_v3 HVG.")
        if norm_layer is None or norm_layer.shape != X_counts.shape:
            raise ValueError("norm_layer must match X_counts shape for seurat_v3 HVG.")

    detailed: List[Dict[str, Any]] = []

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_counts), start=1):
        X_tr, X_va = X_counts[tr_idx], X_counts[va_idx]

        # HVG on train fold
        if hvg_mode == "variance":
            rank = ut.rank_hvg_by_variance(X_tr)
            G = rank[:min(n_hvg, X_tr.shape[1])]
        else:
            G = ut.seurat_v3_hvg_indices_for_train_split(
                X_train_counts=X_tr,
                gene_names=np.asarray(gene_names),
                batch_labels_train=batches[tr_idx],
                normalized_train_matrix=norm_layer[tr_idx, :],
                n_top_genes=n_hvg,
                layer_name=seurat_layer_name,
                batch_key=batch_key,
            )

        X_tr_G = X_tr[:, G]
        X_va_G = X_va[:, G]
        theta_G = ut.estimate_nb_theta_moments(X_tr_G)

        # Build + "fit" (cache train) DCA
        dca = DCAPredictWrapper(**dca_params, random_state=random_state + fold_id).fit(X_tr_G)

        # Full mean for Silhouette
        mu_full = dca.predict_mean(X_va_G)
        sil = ut.bio_silhouette_score(mu_full, labels[va_idx] if labels is not None else None)

        # Nonzero Zeroing (repeat R times)
        zero_errs, zero_nbs = [], []
        for r in range(R):
            rng = np.random.default_rng(random_state + fold_id * 1000 + r)
            M_zero = ut.make_mask_nonzero_by_gene(X_va_G, frac=mask_frac, rng=rng)
            # DCA predicts full mean on the original matrix; metric uses only masked entries
            mu_zero = dca.predict_mean(X_va_G)

            rows, cols = np.where(M_zero)
            y_true = X_va_G[rows, cols]
            y_pred = mu_zero[rows, cols]
            th = theta_G[cols]
            zero_errs.append(ut._basic_errors(y_true, y_pred))
            zero_nbs.append(ut._nb_scores(y_true, y_pred, th))

        Z_mae = float(np.mean([d["MAE"] for d in zero_errs]))
        Z_mse = float(np.mean([d["MSE"] for d in zero_errs]))
        Z_med = float(np.mean([d["MedianL1"] for d in zero_errs]))
        Z_ll  = float(np.mean([d["NB_ll"] for d in zero_nbs]))
        Z_dev = float(np.mean([d["NB_dev"] for d in zero_nbs]))

        # Binomial Thinning (repeat R times)
        thin_errs, thin_nbs = [], []
        for r in range(R):
            rng = np.random.default_rng(random_state + 7777 + fold_id * 1000 + r)
            X_keep, X_hold = ut.binomial_thinning_split(X_va_G, p_hold=thinning_p, rng=rng)
            mu_thin = dca.predict_mean(X_keep)
            y_true = X_hold
            y_pred = mu_thin * float(thinning_p)  # expected holdout mean
            thin_errs.append(ut._basic_errors(y_true, y_pred))
            thin_nbs.append(ut._nb_scores(y_true, y_pred, theta_G))

        T_mae = float(np.mean([d["MAE"] for d in thin_errs]))
        T_mse = float(np.mean([d["MSE"] for d in thin_errs]))
        T_med = float(np.mean([d["MedianL1"] for d in thin_errs]))
        T_ll  = float(np.mean([d["NB_ll"] for d in thin_nbs]))
        T_dev = float(np.mean([d["NB_dev"] for d in thin_nbs]))

        detailed.append({
            "fold": fold_id,
            "model": "DCA",
            "params": dca_params,
            "n_hvg": int(len(G)),
            # Nonzero Zeroing
            "MAE_zero": Z_mae, "MSE_zero": Z_mse, "MedianL1_zero": Z_med,
            "NB_ll_zero": Z_ll, "NB_dev_zero": Z_dev,
            # Binomial Thinning
            "MAE_thin": T_mae, "MSE_thin": T_mse, "MedianL1_thin": T_med,
            "NB_ll_thin": T_ll, "NB_dev_thin": T_dev,
            # Biological structure
            "Silhouette": sil,
        })

    detailed_df = pd.DataFrame(detailed)

    # Summary over folds
    agg_cols = [c for c in detailed_df.columns if c not in ("fold", "model", "params", "n_hvg")]
    summary = (detailed_df
               .groupby(["model", detailed_df["params"].apply(lambda d: tuple(sorted(d.items()))), "n_hvg"])
               [agg_cols].mean().reset_index())
    summary = summary.rename(columns={"params": "params_tuple"})
    summary["params"] = summary["params_tuple"].apply(lambda t: dict(t))
    summary = summary.drop(columns=["params_tuple"])

    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        detailed_df.to_csv(os.path.join(save_dir, "dca_detailed.csv"), index=False)
        summary.to_csv(os.path.join(save_dir, "dca_summary.csv"), index=False)

    return summary, detailed_df

# ---------------- Core nested CV runner ----------------

@dataclass(frozen=True)
class Config:
    model: str
    params: Tuple[Tuple[str, Any], ...]
    n_hvg: int

def _cfg(model: str, params: Dict[str, Any], n_hvg: int) -> Config:
    return Config(model=model, params=tuple(sorted(params.items())), n_hvg=n_hvg)

@dataclass
class FoldScore:
    ll: float
    dev: float
    mae: float

# ---------------- 5-fold CV runner for baselines ----------------
def cv_baselines_5fold(
    X_counts: np.ndarray,
    k: int = 5,
    model_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
    n_hvg: int = 2000,
    R: int = 3,                       # repeats per masking protocol
    mask_frac: float = 0.10,          # Nonzero Zeroing rate
    thinning_p: float = 0.10,         # Binomial Thinning hold-out rate
    random_state: int = 42,
    # HVG mode & metadata
    hvg_mode: str = "variance",       # "variance" or "seurat_v3"
    gene_names: Optional[np.ndarray] = None,
    batches: Optional[np.ndarray] = None,
    norm_layer: Optional[np.ndarray] = None,
    batch_key: str = "batch",
    seurat_layer_name: str = "log2_1p_CPM_original",
    # Biological labels (optional) for Silhouette
    labels: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Plain 5-fold CV for baseline models with two masking protocols:
      (A) Nonzero Zeroing (mask ~mask_frac of nonzero entries per gene)
      (B) Binomial Thinning (two-part split with hold-out probability thinning_p)

    HVG is computed on each train fold only (fixed n_hvg), and NB dispersion θ
    is estimated from the train fold. Silhouette is computed on the imputed
    validation matrix if labels are provided.
    """
    if model_grids is None:
        model_grids = {
            "MEAN": {},
            "MEDIAN": {},
            "KNN": {"n_neighbors": [5, 15, 30], "weights": ["uniform", "distance"]},
            "MAGIC": {"n_pca": [None, 50], "t": [3], "knn": [5, 10]},
        }

    X_counts = np.asarray(X_counts)
    n_samples, n_genes = X_counts.shape
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # Optional checks only when using seurat_v3 HVG
    if hvg_mode == "seurat_v3":
        if gene_names is None or len(gene_names) != n_genes:
            raise ValueError("gene_names (len == n_genes) required for seurat_v3 HVG.")
        if batches is None or len(batches) != n_samples:
            raise ValueError("batches (len == n_samples) required for seurat_v3 HVG.")
        if norm_layer is None or norm_layer.shape != X_counts.shape:
            raise ValueError("norm_layer must match X_counts shape for seurat_v3 HVG.")

    detailed_rows: List[Dict[str, Any]] = []

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_counts), start=1):
        X_tr, X_va = X_counts[tr_idx], X_counts[va_idx]

        # HVG selection on train fold
        if hvg_mode == "variance":
            rank = ut.rank_hvg_by_variance(X_tr)
            G = rank[:min(n_hvg, X_tr.shape[1])]
        else:
            G = ut.seurat_v3_hvg_indices_for_train_split(
                X_train_counts=X_tr,
                gene_names=np.asarray(gene_names),
                batch_labels_train=batches[tr_idx],
                normalized_train_matrix=norm_layer[tr_idx, :],
                n_top_genes=n_hvg,
                layer_name=seurat_layer_name,
                batch_key=batch_key,
            )

        X_tr_G = X_tr[:, G]
        X_va_G = X_va[:, G]
        theta_G = ut.estimate_nb_theta_moments(X_tr_G)  # per-gene θ from train fold

        for model_name, grid in model_grids.items():
            param_list = list(ParameterGrid(grid)) if grid else [dict()]
            for params in param_list:
                # Fit model on train fold (HVG subset)
                mdl = build_baseline(model_name, **params).fit(X_tr_G)

                # --- Biological structure (Silhouette) on full imputed validation matrix
                try:
                    mu_full = ut._predict_full_matrix(
                        mdl, X_va_G,
                        mask=np.zeros_like(X_va_G, dtype=bool),  # KNN needs a mask, but no entries hidden
                        X_train_ref=X_tr_G
                    )
                except Exception:
                    mu_full = X_va_G  # safe fallback if model fails
                sil = ut.bio_silhouette_score(mu_full, labels[va_idx] if labels is not None else None)

                # --- Nonzero Zeroing protocol (repeat R times)
                zero_errs, zero_nbs = [], []
                for r in range(R):
                    rng = np.random.default_rng(random_state + fold_id * 1000 + r)
                    M_zero = ut.make_mask_nonzero_by_gene(X_va_G, frac=mask_frac, rng=rng)
                    mu_zero = ut._predict_full_matrix(mdl, X_va_G.copy(), mask=M_zero, X_train_ref=X_tr_G)

                    rows, cols = np.where(M_zero)
                    y_true = X_va_G[rows, cols]
                    y_pred = mu_zero[rows, cols]
                    th = theta_G[cols]

                    zero_errs.append(ut._basic_errors(y_true, y_pred))
                    zero_nbs.append(ut._nb_scores(y_true, y_pred, th))

                Z_mae = float(np.mean([d["MAE"] for d in zero_errs])) if zero_errs else None
                Z_mse = float(np.mean([d["MSE"] for d in zero_errs])) if zero_errs else None
                Z_med = float(np.mean([d["MedianL1"] for d in zero_errs])) if zero_errs else None
                Z_ll  = float(np.mean([d["NB_ll"] for d in zero_nbs])) if zero_nbs else None
                Z_dev = float(np.mean([d["NB_dev"] for d in zero_nbs])) if zero_nbs else None

                # --- Binomial Thinning protocol (repeat R times)
                thin_errs, thin_nbs = [], []
                for r in range(R):
                    rng = np.random.default_rng(random_state + 7777 + fold_id * 1000 + r)
                    X_keep, X_hold = ut.binomial_thinning_split(X_va_G, p_hold=thinning_p, rng=rng)
                    # Mask all entries to force genuine prediction (esp. for KNN)
                    M_all = np.ones_like(X_keep, dtype=bool)
                    mu_thin = ut._predict_full_matrix(mdl, X_keep, mask=M_all, X_train_ref=X_tr_G)

                    y_true = X_hold
                    y_pred = mu_thin * float(thinning_p)  # expected hold-out mean
                    thin_errs.append(ut._basic_errors(y_true, y_pred))
                    thin_nbs.append(ut._nb_scores(y_true, y_pred, theta_G))

                T_mae = float(np.mean([d["MAE"] for d in thin_errs])) if thin_errs else None
                T_mse = float(np.mean([d["MSE"] for d in thin_errs])) if thin_errs else None
                T_med = float(np.mean([d["MedianL1"] for d in thin_errs])) if thin_errs else None
                T_ll  = float(np.mean([d["NB_ll"] for d in thin_nbs])) if thin_nbs else None
                T_dev = float(np.mean([d["NB_dev"] for d in thin_nbs])) if thin_nbs else None

                detailed_rows.append({
                    "fold": fold_id,
                    "model": model_name,
                    "params": params,
                    "n_hvg": int(len(G)),
                    # Nonzero Zeroing metrics
                    "MAE_zero": Z_mae, "MSE_zero": Z_mse, "MedianL1_zero": Z_med,
                    "NB_ll_zero": Z_ll, "NB_dev_zero": Z_dev,
                    # Binomial Thinning metrics
                    "MAE_thin": T_mae, "MSE_thin": T_mse, "MedianL1_thin": T_med,
                    "NB_ll_thin": T_ll, "NB_dev_thin": T_dev,
                    # Biological structure
                    "Silhouette": sil,
                })

    detailed_df = pd.DataFrame(detailed_rows)

    # Summary: mean over folds for each (model, params)
    agg_cols = [c for c in detailed_df.columns if c not in ("fold", "model", "params", "n_hvg")]
    summary = (detailed_df
               .groupby(["model", detailed_df["params"].apply(lambda d: tuple(sorted(d.items()))), "n_hvg"])
               [agg_cols].mean().reset_index())
    summary = summary.rename(columns={"params": "params_tuple"})
    # restore params as dict for readability
    summary["params"] = summary["params_tuple"].apply(lambda t: dict(t))
    summary = summary.drop(columns=["params_tuple"])
    
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        detailed_df.to_csv(os.path.join(save_dir, "baselines_detailed.csv"), index=False)
        summary.to_csv(os.path.join(save_dir, "baselines_summary.csv"), index=False)

    return summary, detailed_df

def run_dca_nested(
    X_counts: np.ndarray,
    hvg_mode: str,                   # "variance" or "seurat_v3"
    hvg_grid: List[int],
    batches: Optional[np.ndarray] = None,
    norm_layer: Optional[np.ndarray] = None,
    gene_names: Optional[np.ndarray] = None,
    batch_key: str = "batch",
    seurat_layer_name: str = "log2_1p_CPM_original",
    outer_k: int = 5,
    inner_k: int = 3,
    R: int = 3,                      # stochastic masking repeats
    mask_frac: float = 0.1,
    random_state: int = 123,
    dca_grid: Optional[Dict[str, List[Any]]] = None,
    one_se_rule: bool = True,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fully fair DCA nested CV (no leakage):
      - Inner: HVG search × DCA hyperparam grid; train on Train_inner; predict on Val_inner
      - Select by NB log-likelihood (primary), deviance & MAE tie-breakers; 1-SE rule to prefer smaller HVG
      - Outer: retrain on Outer-Train with chosen HVG + HP; evaluate on Outer-Test with R masks
    Requires a DCA build that returns a Keras model supporting .predict on new matrices.
    """
    import numpy as np, pandas as pd
    from sklearn.model_selection import KFold, ParameterGrid

    if dca_grid is None:
        # Keep the grid modest on CPU
        dca_grid = {
            "hidden_size": [[64,32,64]],
            "epochs": [300],
            "batch_size": [128],
            # Optional knobs if your DCA exposes them:
            # "learning_rate": [1e-3],
            # "dropout_rate": [0.1, 0.3],
            # "l2": [None, 1e-4],
        }

    X_counts = np.asarray(X_counts)
    n_samples, n_genes = X_counts.shape

    # --- HVG mode checks
    if hvg_mode not in ("variance", "seurat_v3"):
        raise ValueError("hvg_mode must be 'variance' or 'seurat_v3'.")
    if hvg_mode == "seurat_v3":
        if gene_names is None or len(gene_names) != n_genes:
            raise ValueError("Provide gene_names (len == n_genes) for seurat_v3 HVG.")
        if batches is None or len(batches) != n_samples:
            raise ValueError("Provide batches (len == n_samples) for seurat_v3 HVG.")
        if norm_layer is None or norm_layer.shape != X_counts.shape:
            raise ValueError("Provide norm_layer with same shape as X_counts for seurat_v3 HVG.")

    # ---- KFold splitters
    outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=random_state)
    records = []

    for ofold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_counts), start=1):
        X_tr_out, X_te_out = X_counts[tr_idx], X_counts[te_idx]

        inner_cv = KFold(n_splits=inner_k, shuffle=True, random_state=random_state + 7)
        score_rows = []  # list of tuples ((params_dict, n_hvg), ll_mean, dev_mean, mae_mean, ll_std)

        for ifold, (tr_in_idx, val_in_idx) in enumerate(inner_cv.split(X_tr_out), start=1):
            X_tr_in, X_val_in = X_tr_out[tr_in_idx], X_tr_out[val_in_idx]

            # --- Build HVG indices per n
            rank_slices: Dict[int, np.ndarray] = {}
            theta_map:   Dict[int, np.ndarray] = {}

            if hvg_mode == "variance":
                rank = rank_hvg_by_variance(X_tr_in)
                for n in hvg_grid:
                    idx = rank[:min(n, X_tr_in.shape[1])]
                    rank_slices[n] = idx
                    theta_map[n]   = estimate_nb_theta_moments(X_tr_in[:, idx])
            else:
                b_tr_in   = batches[tr_idx][tr_in_idx]
                norm_in   = norm_layer[tr_idx][tr_in_idx, :]
                for n in hvg_grid:
                    idx = seurat_v3_hvg_indices_for_train_split(
                        X_train_counts=X_tr_in,
                        gene_names=gene_names,
                        batch_labels_train=b_tr_in,
                        normalized_train_matrix=norm_in,
                        n_top_genes=n,
                        layer_name=seurat_layer_name,
                        batch_key=batch_key,
                    )
                    rank_slices[n] = idx
                    theta_map[n]   = estimate_nb_theta_moments(X_tr_in[:, idx])

            # --- Grid over DCA HP × HVG
            for hp in ParameterGrid(dca_grid):
                for n in hvg_grid:
                    G = rank_slices[n]
                    X_tr_G  = X_tr_in[:, G]
                    X_val_G = X_val_in[:, G]
                    theta   = theta_map[n]

                    # Train DCA on Train_inner only
                    dca = DCAPredictWrapper(**hp, random_state=random_state + ofold*100 + ifold).fit(X_tr_G)

                    # Evaluate on Val_inner with R different masks
                    ll_list, dev_list, mae_list = [], [], []
                    for r in range(R):
                        rng = np.random.default_rng(random_state + ofold*1000 + ifold*100 + r)
                        M = make_mask_stratified_by_gene(X_val_G, frac=mask_frac, rng=rng)
                        mu = dca.predict_mean(X_val_G)

                        rows, cols = np.where(M)
                        y_true, y_pred = X_val_G[rows, cols], mu[rows, cols]
                        th = theta[cols]
                        ll_list.append(nb_logpmf(y_true, y_pred, th).mean())
                        dev_list.append(nb_deviance(y_true, y_pred, th).mean())
                        mae_list.append(np.mean(np.abs(y_true - y_pred)))

                    score_rows.append(((hp, n),
                                       float(np.mean(ll_list)),
                                       float(np.mean(dev_list)),
                                       float(np.mean(mae_list)),
                                       float(np.std(ll_list, ddof=1) if len(ll_list) > 1 else 0.0)))

        # --- Select best by NB log-lik; 1-SE prefers smaller HVG
        score_rows.sort(key=lambda t: (t[1], -t[2], -t[3]), reverse=True)
        (best_hp, best_n), best_ll, _, _, best_std = score_rows[0]
        if one_se_rule:
            thr = best_ll - best_std
            cands = [s for s in score_rows if s[1] >= thr]
            cands.sort(key=lambda t: (t[0][1], -t[1], t[2], t[3]))  # smaller HVG first
            (best_hp, best_n) = cands[0][0]

        # --- Outer retrain on Outer-Train; evaluate on Outer-Test
        if hvg_mode == "variance":
            rank_out = rank_hvg_by_variance(X_tr_out)
            G_out = rank_out[:min(best_n, X_tr_out.shape[1])]
        else:
            b_tr_out = batches[tr_idx]
            norm_out = norm_layer[tr_idx, :]
            G_out = seurat_v3_hvg_indices_for_train_split(
                X_train_counts=X_tr_out,
                gene_names=gene_names,
                batch_labels_train=b_tr_out,
                normalized_train_matrix=norm_out,
                n_top_genes=best_n,
                layer_name=seurat_layer_name,
                batch_key=batch_key,
            )

        X_tr_G = X_tr_out[:, G_out]
        X_te_G = X_te_out[:, G_out]
        theta_out = estimate_nb_theta_moments(X_tr_G)

        dca_final = DCAPredictWrapper(**best_hp, random_state=random_state + ofold*777).fit(X_tr_G)

        ll_list, dev_list, mae_list = [], [], []
        for r in range(R):
            rng = np.random.default_rng(random_state + 9999 + ofold*100 + r)
            M_test = make_mask_stratified_by_gene(X_te_G, frac=mask_frac, rng=rng)
            mu_test = dca_final.predict_mean(X_te_G)

            rows_, cols_ = np.where(M_test)
            y_true, y_pred = X_te_G[rows_, cols_], mu_test[rows_, cols_]
            th = theta_out[cols_]
            ll_list.append(nb_logpmf(y_true, y_pred, th).mean())
            dev_list.append(nb_deviance(y_true, y_pred, th).mean())
            mae_list.append(np.mean(np.abs(y_true - y_pred)))

        records.append({
            "outer_fold": ofold,
            "model": "DCA",
            "best_params": best_hp,
            "best_n_hvg": int(best_n),
            "NB_loglik_mean": float(np.mean(ll_list)),
            "NB_deviance_mean": float(np.mean(dev_list)),
            "MAE_mean": float(np.mean(mae_list)),
        })

    df = pd.DataFrame(records)
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        # --- util 기반 CSV 저장 (overwrite) ---
        fp = f"{save_dir}/results.csv"
        if os.path.exists(fp):
            os.remove(fp)
        header_order = list(df.columns)
        for row in df.to_dict(orient="records"):
            safe_row = {}
            for k, v in row.items():
                if isinstance(v, (dict, list, tuple)):
                    safe_row[k] = json.dumps(v, ensure_ascii=False)
                else:
                    safe_row[k] = v
            _append_csv(fp, safe_row, header_order=header_order)
    return df


def cv_scvi_5fold(
    X_counts: np.ndarray,
    k: int = 5,
    scvi_grid: Optional[Dict[str, List[Any]]] = None,
    n_hvg: int = 2000,
    R: int = 3,
    mask_frac: float = 0.10,
    thinning_p: float = 0.10,
    random_state: int = 123,
    # HVG
    hvg_mode: str = "variance",
    gene_names: Optional[np.ndarray] = None,
    batches: Optional[np.ndarray] = None,
    norm_layer: Optional[np.ndarray] = None,
    batch_key: str = "batch",
    seurat_layer_name: str = "log2_1p_CPM_original",
    # labels for Silhouette
    labels: Optional[np.ndarray] = None,
    # CSV export
    save_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    5-fold CV for scVI with two masking protocols (Nonzero Zeroing + Binomial Thinning).
    Metrics: MAE, MSE, MedianL1, NB log-likelihood, NB deviance, Silhouette.
    HVG is computed per train fold (fixed n_hvg).
    """
    from sklearn.model_selection import ParameterGrid

    # Sensible default grid for dropout-imputation benchmarking
    if scvi_grid is None:
        scvi_grid = {
            "n_latent":      [8, 16, 32],
            "n_hidden":      [64, 128, 256],
            "n_layers":      [1, 2, 3],
            "dropout_rate":  [0.0, 0.1, 0.2],
            "max_epochs":    [200, 400],
            "lr":            [1e-3, 5e-4, 1e-4],
            "weight_decay":  [1e-6, 1e-5, 1e-4],
            "batch_size":    [256, 1024],   # reduce if you hit OOM on M1
            "use_gpu":       [True],
        }

    X_counts = np.asarray(X_counts)
    n_samples, n_genes = X_counts.shape
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # Check seurat_v3 inputs if selected
    if hvg_mode == "seurat_v3":
        if gene_names is None or len(gene_names) != n_genes:
            raise ValueError("gene_names (len == n_genes) required for seurat_v3 HVG.")
        if batches is None or len(batches) != n_samples:
            raise ValueError("batches (len == n_samples) required for seurat_v3 HVG.")
        if norm_layer is None or norm_layer.shape != X_counts.shape:
            raise ValueError("norm_layer must match X_counts shape for seurat_v3 HVG.")

    detailed_rows: List[Dict[str, Any]] = []

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_counts), start=1):
        X_tr, X_va = X_counts[tr_idx], X_counts[va_idx]

        # HVG on train fold
        if hvg_mode == "variance":
            rank = ut.rank_hvg_by_variance(X_tr)
            G = rank[:min(n_hvg, X_tr.shape[1])]
        else:
            G = ut.seurat_v3_hvg_indices_for_train_split(
                X_train_counts=X_tr,
                gene_names=np.asarray(gene_names),
                batch_labels_train=batches[tr_idx],
                normalized_train_matrix=norm_layer[tr_idx, :],
                n_top_genes=n_hvg,
                layer_name=seurat_layer_name,
                batch_key=batch_key,
            )

        X_tr_G = X_tr[:, G]
        X_va_G = X_va[:, G]
        theta_G = ut.estimate_nb_theta_moments(X_tr_G)

        for p in ParameterGrid(scvi_grid):
            # Build and fit scVI on train fold
            mdl = SCVIWrapper(
                n_latent=p["n_latent"],
                n_hidden=p["n_hidden"],
                n_layers=p["n_layers"],
                dropout_rate=p["dropout_rate"],
                max_epochs=p["max_epochs"],
                lr=p["lr"],
                weight_decay=p["weight_decay"],
                batch_size=p.get("batch_size", None),
                batch_key=batch_key,
                use_gpu=p.get("use_gpu", True),
                gene_names=(gene_names[G] if gene_names is not None else None),
                batches=(batches[tr_idx] if batches is not None else None),
            ).fit(X_tr_G)

            # Full mean for Silhouette
            mu_full = mdl.predict_mean(
                X_va_G,
                eval_batches=(batches[va_idx] if batches is not None else None),
            )
            sil = ut.bio_silhouette_score(mu_full, labels[va_idx] if labels is not None else None)

            # Nonzero Zeroing
            zero_errs, zero_nbs = [], []
            for r in range(R):
                rng = np.random.default_rng(random_state + fold_id * 1000 + r)
                M_zero = ut.make_mask_nonzero_by_gene(X_va_G, frac=mask_frac, rng=rng)
                # For scVI we don't need to feed a mask; we always predict full mu from X_va_G as-is.
                mu_zero = mdl.predict_mean(
                    X_va_G,
                    eval_batches=(batches[va_idx] if batches is not None else None),
                )
                rows, cols = np.where(M_zero)
                y_true = X_va_G[rows, cols]
                y_pred = mu_zero[rows, cols]
                th = theta_G[cols]
                zero_errs.append(ut._basic_errors(y_true, y_pred))
                zero_nbs.append(ut._nb_scores(y_true, y_pred, th))

            Z_mae = float(np.mean([d["MAE"] for d in zero_errs])) if zero_errs else None
            Z_mse = float(np.mean([d["MSE"] for d in zero_errs])) if zero_errs else None
            Z_med = float(np.mean([d["MedianL1"] for d in zero_errs])) if zero_errs else None
            Z_ll  = float(np.mean([d["NB_ll"] for d in zero_nbs])) if zero_nbs else None
            Z_dev = float(np.mean([d["NB_dev"] for d in zero_nbs])) if zero_nbs else None

            # Binomial Thinning
            thin_errs, thin_nbs = [], []
            for r in range(R):
                rng = np.random.default_rng(random_state + 7777 + fold_id * 1000 + r)
                X_keep, X_hold = ut.binomial_thinning_split(X_va_G, p_hold=thinning_p, rng=rng)
                mu_thin = mdl.predict_mean(
                    X_keep,
                    eval_batches=(batches[va_idx] if batches is not None else None),
                )
                y_true = X_hold
                y_pred = mu_thin * float(thinning_p)
                thin_errs.append(ut._basic_errors(y_true, y_pred))
                thin_nbs.append(ut._nb_scores(y_true, y_pred, theta_G))

            T_mae = float(np.mean([d["MAE"] for d in thin_errs])) if thin_errs else None
            T_mse = float(np.mean([d["MSE"] for d in thin_errs])) if thin_errs else None
            T_med = float(np.mean([d["MedianL1"] for d in thin_errs])) if thin_errs else None
            T_ll  = float(np.mean([d["NB_ll"] for d in thin_nbs])) if thin_nbs else None
            T_dev = float(np.mean([d["NB_dev"] for d in thin_nbs])) if thin_nbs else None

            detailed_rows.append({
                "fold": fold_id,
                "model": "scVI",
                "params": {k: p[k] for k in p},  # copy
                "n_hvg": int(len(G)),
                # Nonzero Zeroing
                "MAE_zero": Z_mae, "MSE_zero": Z_mse, "MedianL1_zero": Z_med,
                "NB_ll_zero": Z_ll, "NB_dev_zero": Z_dev,
                # Binomial Thinning
                "MAE_thin": T_mae, "MSE_thin": T_mse, "MedianL1_thin": T_med,
                "NB_ll_thin": T_ll, "NB_dev_thin": T_dev,
                # Biological structure
                "Silhouette": sil,
            })

    detailed_df = pd.DataFrame(detailed_rows)

    # Summary over folds per (params, n_hvg)
    agg_cols = [c for c in detailed_df.columns if c not in ("fold", "model", "params", "n_hvg")]
    summary = (detailed_df
               .groupby([detailed_df["params"].apply(lambda d: tuple(sorted(d.items()))), "n_hvg"])
               [agg_cols].mean().reset_index())
    summary = summary.rename(columns={"params": "params_tuple"})
    summary["params"] = summary["params_tuple"].apply(lambda t: dict(t))
    summary = summary.drop(columns=["params_tuple"])
    summary.insert(0, "model", "scVI")

    # CSV export
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        detailed_df.to_csv(os.path.join(save_dir, "scvi_detailed.csv"), index=False)
        summary.to_csv(os.path.join(save_dir, "scvi_summary.csv"), index=False)

    return summary, detailed_df