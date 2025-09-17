# scripts/runner_models.py (logging-enabled)
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import os, json, math
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.impute import KNNImputer

sys.path.append("..")
import scripts.utilities as ut  # << uses ut.log, ut.Timer, ut.heartbeat, etc.

np.random.seed(123)

# ======== small helpers ========
def _hb(path: Optional[str], msg: str):
    if path:
        ut.heartbeat(path, msg)

def _maybe_flush_csv(df: pd.DataFrame, path: Optional[str], fname: str):
    if path is None: 
        return
    os.makedirs(path, exist_ok=True)
    tmp = os.path.join(path, f"_{fname}.tmp.csv")
    df.to_csv(tmp, index=False)

# ---------------- Baseline model wrappers ----------------
class _MeanModel:
    name = "MEAN"
    def fit(self, X: np.ndarray):
        ut.log("[MEAN] fit")
        ut.shape_dtype("X_train", X)
        self.col_ = X.mean(axis=0)
        return self
    def predict_mean(self, X: np.ndarray, mask: Optional[np.ndarray] = None, X_train_ref: Optional[np.ndarray] = None) -> np.ndarray:
        ut.log("[MEAN] predict_mean")
        return np.broadcast_to(self.col_, X.shape)

class _MedianModel:
    name = "MEDIAN"
    def fit(self, X: np.ndarray):
        ut.log("[MEDIAN] fit")
        ut.shape_dtype("X_train", X)
        self.col_ = np.median(X, axis=0)
        return self
    def predict_mean(self, X: np.ndarray, mask: Optional[np.ndarray] = None, X_train_ref: Optional[np.ndarray] = None) -> np.ndarray:
        ut.log("[MEDIAN] predict_mean")
        return np.broadcast_to(self.col_, X.shape)

class _KNNModel:
    name = "KNN"
    def __init__(self, n_neighbors: int = 15, weights: str = "uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.imp = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric="nan_euclidean")
        self.X_train_ = None
    def fit(self, X: np.ndarray):
        ut.log(f"[KNN] fit n_neighbors={self.n_neighbors} weights={self.weights}")
        ut.shape_dtype("X_train", X)
        self.X_train_ = X
        return self
    def predict_mean(self, X: np.ndarray, mask: Optional[np.ndarray] = None, X_train_ref: Optional[np.ndarray] = None) -> np.ndarray:
        ut.log("[KNN] predict_mean")
        if mask is None:
            raise ValueError("KNN baseline requires a boolean mask to mark hidden entries.")
        if X_train_ref is None:
            X_train_ref = self.X_train_
        X_eval = X.copy().astype(np.float64)
        X_eval[mask] = np.nan
        X_concat = np.vstack([X_train_ref, X_eval])
        with ut.Timer("[KNN] impute"):
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
        ut.log(f"[MAGIC] fit n_pca={self.n_pca} t={self.t} knn={self.knn}")
        ut.shape_dtype("X_train", X)
        from magic import MAGIC
        self.op_ = MAGIC(n_pca=self.n_pca, t=self.t, knn=self.knn)
        with ut.Timer("[MAGIC] fit"):
            self.op_.fit(np.asarray(X, dtype=np.float64))
        return self
    def predict_mean(self, X, mask=None, X_train_ref=None):
        from magic import MAGIC
        if self.op_ is None:
            raise RuntimeError("Call fit() before predict_mean().")
        X = np.asarray(X, dtype=np.float64)

        # If mask is given, hide those entries in the eval matrix
        X_eval = X.copy()
        if mask is not None:
            X_eval[mask] = 0.0   # (or np.nan, but 0.0 is the safer default here)

        # Prefer out-of-sample transform
        if hasattr(self.op_, "transform"):
            try:
                with ut.Timer("[MAGIC] transform"):
                    return np.asarray(self.op_.transform(X_eval), dtype=np.float64)
            except Exception as e:
                ut.log(f"[MAGIC] transform failed, refit fallback: {e}")

        # Fallback: transductive fit on [train || eval_masked]
        if X_train_ref is not None:
            X_concat = np.vstack([np.asarray(X_train_ref, np.float64), X_eval])
            tmp = MAGIC(n_pca=self.n_pca, t=self.t, knn=self.knn)
            with ut.Timer("[MAGIC] concat fit_transform"):
                X_imp = tmp.fit_transform(X_concat)
            return np.asarray(X_imp[-X.shape[0]:, :], dtype=np.float64)

        tmp = MAGIC(n_pca=self.n_pca, t=self.t, knn=self.knn)
        with ut.Timer("[MAGIC] fit_transform"):
            return np.asarray(tmp.fit_transform(X_eval), dtype=np.float64)

def build_baseline(name: str, **params) -> Any:
    nm = name.upper()
    ut.log(f"[build_baseline] {nm} params={params}")
    if nm == "MEAN":   return _MeanModel()
    if nm == "MEDIAN": return _MedianModel()
    if nm == "KNN":    return _KNNModel(n_neighbors=int(params.get("n_neighbors", 15)),
                                       weights=params.get("weights", "uniform"))
    if nm == "MAGIC":  return _MAGICModel(**params)
    raise ValueError(f"Unknown baseline: {name}")

# ---------- Minimal DCA wrapper ----------
class DCAPredictWrapper:
    """
    DCA predictor that avoids leakage by concatenating [train || eval] and marking
    train rows for optimization. If tensorflow-metal is installed, TF may use MPS.
    """
    name = "DCA"

    def __init__(self, hidden_size=[64, 32, 64], epochs=300, batch_size=128, random_state: int = 0):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.X_train_ = None

    def fit(self, X_train: np.ndarray):
        ut.log(f"[DCA] fit hidden={self.hidden_size} epochs={self.epochs} bs={self.batch_size}")
        ut.shape_dtype("X_train", X_train)
        X_train = np.asarray(X_train)
        if not np.issubdtype(X_train.dtype, np.floating):
            X_train = X_train.astype(np.float32, copy=False)
        self.X_train_ = X_train
        return self

    def predict_mean(self, X_eval: np.ndarray) -> np.ndarray:
        import anndata as ad
        from dca.api import dca

        ut.log("[DCA] predict_mean")
        if self.X_train_ is None:
            raise RuntimeError("Call fit(X_train) before predict_mean.")
        X_eval = np.asarray(X_eval)
        if not np.issubdtype(X_eval.dtype, np.floating):
            X_eval = X_eval.astype(np.float32, copy=False)

        X_concat = np.vstack([self.X_train_, X_eval]).astype(np.float32, copy=False)
        n_tr = self.X_train_.shape[0]
        n_ev = X_eval.shape[0]

        ad_all = ad.AnnData(X_concat)
        ad_all.obs["dca_split"] = np.array(["train"] * n_tr + ["test"] * n_ev, dtype=object)
        
        from keras import backend as K
        K.clear_session()

        import dca.train as dca_train
        from keras import optimizers as _k_optim
        assert hasattr(_k_optim, "legacy"), "Your Keras build must expose `keras.optimizers.legacy`."
        dca_train.opt = _k_optim.legacy       # <-- IMPORTANT
        opt_name = "RMSprop"

        import tensorflow as tf
        with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session().as_default() as _sess:
            # Bind this session to any available backends
            try:
                tf.compat.v1.keras.backend.set_session(_sess)
            except Exception:
                pass
            try:
                if hasattr(K, "set_session"):
                    K.set_session(_sess)
            except Exception:
                pass

            with ut.Timer("[DCA] train(dca.api.dca)"):
                dca(
                    ad_all,
                    hidden_size=self.hidden_size,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    optimizer=opt_name,    # resolved through dca.train.opt (legacy table)
                    return_model=False,
                    copy=False,
                    random_state=self.random_state,
                )

        if "X_dca" in ad_all.layers:
            return np.asarray(ad_all.layers["X_dca"][-n_ev:, :])
        return np.asarray(ad_all.X[-n_ev:, :])


# ---------- scVI wrapper ----------
class SCVIWrapper:
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
        return "cpu", "auto"

    def fit(self, X_train: np.ndarray):
        import anndata as ad
        import scvi
        ut.log(f"[scVI] fit hp={self.hp}")
        ut.shape_dtype("X_train", X_train)

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

        with ut.Timer("[scVI] train"):
            self.model_.train(**train_kwargs)

        self.train_genes_ = np.array(adata.var_names)
        return self

    def predict_mean(self, X_eval: np.ndarray, eval_batches: Optional[np.ndarray] = None) -> np.ndarray:
        import anndata as ad
        ut.log("[scVI] predict_mean")
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        adata_eval = ad.AnnData(X=X_eval)
        adata_eval.var_names = self.train_genes_
        if self._batches is not None:
            if eval_batches is None:
                raise KeyError(
                    f"Expected eval_batches for batch_key='{self.hp['batch_key']}' with length {X_eval.shape[0]}"
                )
            adata_eval.obs[self.hp["batch_key"]] = np.array(eval_batches, dtype=str)

        try:
            with ut.Timer("[scVI] get_likelihood_parameters"):
                px = self.model_.get_likelihood_parameters(adata=adata_eval)
            mu = np.asarray(px.get("mean", px.get("mu")))
            if mu is None:
                raise KeyError("likelihood parameters missing mean/mu")
            return mu
        except Exception as e:
            ut.log(f"[scVI] fallback to normalized_expression: {e}")
            lib_eval = X_eval.sum(axis=1, keepdims=True).astype(np.float64)
            norm = np.asarray(self.model_.get_normalized_expression(adata=adata_eval, library_size=1.0))
            return norm * np.maximum(lib_eval, 1e-12)

    def get_latent(self, X_eval: np.ndarray, eval_batches: Optional[np.ndarray] = None) -> np.ndarray:
        import anndata as ad
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        adata_eval = ad.AnnData(X=X_eval)
        adata_eval.var_names = self.train_genes_
        if self._batches is not None:
            if eval_batches is None:
                raise KeyError(f"Expected eval_batches for batch_key='{self.hp['batch_key']}' with length {X_eval.shape[0]}")
            adata_eval.obs[self.hp["batch_key"]] = np.array(eval_batches, dtype=str)
        Z = self.model_.get_latent_representation(adata=adata_eval)  # (n_cells, n_latent)
        return np.asarray(Z)


# ---------- DCA 5-fold CV ----------
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
    # labels
    labels: Optional[np.ndarray] = None,
    # IO
    save_dir: Optional[str] = None,
    heartbeat_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    5-fold CV for DCA with two masking protocols. Logs progress & shapes.
    """

    # --- local helper: make params hashable for pandas groupby (lists -> tuples, numpy scalars -> py scalars)
    def _to_hashable(obj: Any) -> Any:
        """Convert nested structures to hashable (dict values become tuples if sequences)."""
        import numpy as _np
        if isinstance(obj, dict):
            return {k: _to_hashable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, _np.ndarray)):
            return tuple(_to_hashable(x) for x in list(obj))
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
        return obj

    ut.log(f"[cv_dca_5fold] start k={k} n_hvg={n_hvg} mode={hvg_mode} R={R} "
           f"mask_frac={mask_frac} thinning_p={thinning_p}")
    if dca_params is None:
        dca_params = dict(hidden_size=[64, 32, 64], epochs=300, batch_size=128)

    X_counts = np.asarray(X_counts)
    ut.shape_dtype("X_counts", X_counts)
    ut.sanity_counts("X_counts", X_counts)
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
        _hb(heartbeat_path, f"DCA fold {fold_id}/{k} start")
        with ut.Timer(f"[DCA] fold {fold_id}/{k}"):
            X_tr, X_va = X_counts[tr_idx], X_counts[va_idx]
            ut.shape_dtype("X_tr", X_tr)
            ut.shape_dtype("X_va", X_va)

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
            ut.log(f"[DCA] fold {fold_id} | HVG={len(G)}")

            X_tr_G = X_tr[:, G]
            X_va_G = X_va[:, G]
            theta_G = ut.estimate_nb_theta_moments(X_tr_G)

            dca = DCAPredictWrapper(**dca_params, random_state=random_state + fold_id).fit(X_tr_G)

            with ut.Timer(f"[DCA] fold {fold_id} predict mu_va"):
                mu_va = dca.predict_mean(X_va_G)

            sil = ut.bio_silhouette_score(mu_va, labels[va_idx] if labels is not None else None)

            # Nonzero Zeroing (reuse mu_va)
            zero_errs, zero_nbs = [], []
            for r in range(R):
                # separate RNG for zeroing
                rng_zero = np.random.default_rng(random_state + 1111 + fold_id * 1000 + r)
                M_zero = ut.make_mask_nonzero_by_gene(X_va_G, frac=mask_frac, rng=rng_zero) 

                X_va_masked = X_va_G.copy()
                X_va_masked[M_zero] = 0  # hide the masked entries in the input

                mu_zero = dca.predict_mean(X_va_masked)

                rows, cols = np.where(M_zero)
                y_true = X_va_G[rows, cols]
                y_pred = mu_zero[rows, cols]
                th = theta_G[cols]

                zero_errs.append(ut._basic_errors(y_true, y_pred))
                zero_nbs.append(ut._nb_scores(y_true, y_pred, th))

            # Binomial Thinning (reuse mu_va scaled)
            thin_errs, thin_nbs = [], []
            for r in range(R):
                # separate RNG for thinning
                rng_thin = np.random.default_rng(random_state + 7777 + fold_id * 1000 + r)

                X_keep, X_hold = ut.binomial_thinning_split(X_va_G, p_hold=thinning_p, rng=rng_thin)

                # predict mean from the kept counts only
                mu_keep = dca.predict_mean(X_keep)

                # target is the held-out mass; scale prediction accordingly
                y_true = X_hold
                y_pred = (thinning_p / (1.0 - thinning_p)) * mu_keep

                thin_errs.append(ut._basic_errors(y_true, y_pred))
                thin_nbs.append(ut._nb_scores(y_true, y_pred, theta_G))

            Z_mae = float(np.mean([d["MAE"] for d in zero_errs])); Z_mse = float(np.mean([d["MSE"] for d in zero_errs]))
            Z_med = float(np.mean([d["MedianL1"] for d in zero_errs]))
            Z_ll  = float(np.mean([d["NB_ll"] for d in zero_nbs])); Z_dev = float(np.mean([d["NB_dev"] for d in zero_nbs]))

            T_mae = float(np.mean([d["MAE"] for d in thin_errs])); T_mse = float(np.mean([d["MSE"] for d in thin_errs]))
            T_med = float(np.mean([d["MedianL1"] for d in thin_errs]))
            T_ll  = float(np.mean([d["NB_ll"] for d in thin_nbs])); T_dev = float(np.mean([d["NB_dev"] for d in thin_nbs]))

            # Make params hashable before storing into the DataFrame (so groupby won't fail)
            params_clean = _to_hashable(dca_params)

            row = {
                "fold": fold_id,
                "model": "DCA",
                "params": params_clean,  # <-- hashable dict (lists converted to tuples)
                "n_hvg": int(len(G)),
                "MAE_zero": Z_mae, "MSE_zero": Z_mse, "MedianL1_zero": Z_med,
                "NB_ll_zero": Z_ll, "NB_dev_zero": Z_dev,
                "MAE_thin": T_mae, "MSE_thin": T_mse, "MedianL1_thin": T_med,
                "NB_ll_thin": T_ll, "NB_dev_thin": T_dev,
                "Silhouette": sil,
            }
            detailed.append(row)
            _hb(heartbeat_path, f"DCA fold {fold_id}/{k} done: NB_ll_zero={Z_ll:.4f}, NB_ll_thin={T_ll:.4f}")
            # periodic flush
            _maybe_flush_csv(pd.DataFrame(detailed), save_dir, "dca_detailed")

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
        os.makedirs(save_dir, exist_ok=True)
        detailed_df.to_csv(os.path.join(save_dir, "dca_detailed.csv"), index=False)
        summary.to_csv(os.path.join(save_dir, "dca_summary.csv"), index=False)
        ut.log(f"[cv_dca_5fold] wrote CSVs to {save_dir}")

    ut.log("[cv_dca_5fold] done")
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
    R: int = 3,
    mask_frac: float = 0.10,
    thinning_p: float = 0.10,
    random_state: int = 42,
    hvg_mode: str = "variance",
    gene_names: Optional[np.ndarray] = None,
    batches: Optional[np.ndarray] = None,
    norm_layer: Optional[np.ndarray] = None,
    batch_key: str = "batch",
    seurat_layer_name: str = "log2_1p_CPM_original",
    labels: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None,
    heartbeat_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ut.log(f"[cv_baselines_5fold] start k={k} n_hvg={n_hvg} mode={hvg_mode}")
    if model_grids is None:
        model_grids = {
            "MEAN": {},
            "MEDIAN": {},
            "KNN": {"n_neighbors": [5, 15, 30], "weights": ["uniform", "distance"]},
            "MAGIC": {"n_pca": [None, 50], "t": [3], "knn": [5, 10]},
        }

    X_counts = np.asarray(X_counts)
    ut.shape_dtype("X_counts", X_counts)
    ut.sanity_counts("X_counts", X_counts)
    n_samples, n_genes = X_counts.shape
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    if hvg_mode == "seurat_v3":
        if gene_names is None or len(gene_names) != n_genes:
            raise ValueError("gene_names (len == n_genes) required for seurat_v3 HVG.")
        if batches is None or len(batches) != n_samples:
            raise ValueError("batches (len == n_samples) required for seurat_v3 HVG.")
        if norm_layer is None or norm_layer.shape != X_counts.shape:
            raise ValueError("norm_layer must match X_counts shape for seurat_v3 HVG.")

    detailed_rows: List[Dict[str, Any]] = []

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_counts), start=1):
        _hb(heartbeat_path, f"BASE fold {fold_id}/{k} start")
        with ut.Timer(f"[BASE] fold {fold_id}/{k}"):
            X_tr, X_va = X_counts[tr_idx], X_counts[va_idx]

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
            ut.log(f"[BASE] fold {fold_id} | HVG={len(G)}")
            
            X_tr_G = X_tr[:, G]
            X_va_G = X_va[:, G]
            theta_G = ut.estimate_nb_theta_moments(X_tr_G)

            for model_name, grid in model_grids.items():
                param_list = list(ParameterGrid(grid)) if grid else [dict()]
                for params in param_list:
                    ut.log(f"[BASE] fold {fold_id} model={model_name} params={params}")
                    mdl = build_baseline(model_name, **params).fit(X_tr_G)

                    try:
                        mu_full = ut._predict_full_matrix(
                            mdl, X_va_G,
                            mask=np.zeros_like(X_va_G, dtype=bool),
                            X_train_ref=X_tr_G
                        )
                    except Exception as e:
                        ut.log(f"[BASE] mu_full failed: {e}")
                        mu_full = X_va_G
                    sil = ut.bio_silhouette_score(mu_full, labels[va_idx] if labels is not None else None)

                    # 1. Nonzero Zeroing
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

                    # 2. Binomial Thinning
                    thin_errs, thin_nbs = [], []
                    for r in range(R):
                        rng_thin = np.random.default_rng(random_state + 7777 + fold_id * 1000 + r)

                        X_keep, X_hold = ut.binomial_thinning_split(X_va_G, p_hold=thinning_p, rng=rng_thin)

                        # For KNN, pass an all-False mask so we do NOT inject NaNs.
                        mask_all_false = np.zeros_like(X_keep, dtype=bool)

                        mu_keep = ut._predict_full_matrix(
                            mdl, X_keep, mask=mask_all_false, X_train_ref=X_tr_G
                        )

                        y_true = X_hold
                        y_pred = (thinning_p / (1.0 - thinning_p)) * mu_keep

                        thin_errs.append(ut._basic_errors(y_true, y_pred))
                        thin_nbs.append(ut._nb_scores(y_true, y_pred, theta_G))

                    row = {
                        "fold": fold_id,
                        "model": model_name,
                        "params": params,
                        "n_hvg": int(len(G)),
                        "MAE_zero": float(np.mean([d["MAE"] for d in zero_errs])) if zero_errs else None,
                        "MSE_zero": float(np.mean([d["MSE"] for d in zero_errs])) if zero_errs else None,
                        "MedianL1_zero": float(np.mean([d["MedianL1"] for d in zero_errs])) if zero_errs else None,
                        "NB_ll_zero": float(np.mean([d["NB_ll"] for d in zero_nbs])) if zero_nbs else None,
                        "NB_dev_zero": float(np.mean([d["NB_dev"] for d in zero_nbs])) if zero_nbs else None,
                        "MAE_thin": float(np.mean([d["MAE"] for d in thin_errs])) if thin_errs else None,
                        "MSE_thin": float(np.mean([d["MSE"] for d in thin_errs])) if thin_errs else None,
                        "MedianL1_thin": float(np.mean([d["MedianL1"] for d in thin_errs])) if thin_errs else None,
                        "NB_ll_thin": float(np.mean([d["NB_ll"] for d in thin_nbs])) if thin_nbs else None,
                        "NB_dev_thin": float(np.mean([d["NB_dev"] for d in thin_nbs])) if thin_nbs else None,
                        "Silhouette": sil,
                    }
                    detailed_rows.append(row)
                    _maybe_flush_csv(pd.DataFrame(detailed_rows), save_dir, "baselines_detailed")

        _hb(heartbeat_path, f"BASE fold {fold_id}/{k} done")

    detailed_df = pd.DataFrame(detailed_rows)
    agg_cols = [c for c in detailed_df.columns if c not in ("fold", "model", "params", "n_hvg")]
    summary = (detailed_df
               .groupby(["model", detailed_df["params"].apply(lambda d: tuple(sorted(d.items()))), "n_hvg"])
               [agg_cols].mean().reset_index())
    summary = summary.rename(columns={"params": "params_tuple"})
    summary["params"] = summary["params_tuple"].apply(lambda t: dict(t))
    summary = summary.drop(columns=["params_tuple"])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        detailed_df.to_csv(os.path.join(save_dir, "baselines_detailed.csv"), index=False)
        summary.to_csv(os.path.join(save_dir, "baselines_summary.csv"), index=False)
        ut.log(f"[cv_baselines_5fold] wrote CSVs to {save_dir}")

    ut.log("[cv_baselines_5fold] done")
    return summary, detailed_df


def cv_scvi_5fold(
    X_counts: np.ndarray,
    k: int = 5,
    scvi_grid: Optional[Dict[str, List[Any]]] = None,
    n_hvg: int = 2000,
    R: int = 3,
    mask_frac: float = 0.10,
    thinning_p: float = 0.10,
    random_state: int = 123,
    hvg_mode: str = "variance",
    gene_names: Optional[np.ndarray] = None,
    batches: Optional[np.ndarray] = None,
    norm_layer: Optional[np.ndarray] = None,
    batch_key: str = "batch",
    seurat_layer_name: str = "log2_1p_CPM_original",
    labels: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None,
    heartbeat_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ut.log(f"[cv_scvi_5fold] start k={k} n_hvg={n_hvg} mode={hvg_mode}")
    from sklearn.model_selection import ParameterGrid

    if scvi_grid is None:
        scvi_grid = {
            "n_latent": [8, 16, 32],
            "n_hidden": [64, 128, 256],
            "n_layers": [1, 2, 3],
            "dropout_rate": [0.0, 0.1, 0.2],
            "max_epochs": [200, 400],
            "lr": [1e-3, 5e-4, 1e-4],
            "weight_decay": [1e-6, 1e-5, 1e-4],
            "batch_size": [256, 1024],
            "use_gpu": [True],
        }

    X_counts = np.asarray(X_counts)
    ut.shape_dtype("X_counts", X_counts)
    ut.sanity_counts("X_counts", X_counts)
    n_samples, n_genes = X_counts.shape
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    if hvg_mode == "seurat_v3":
        if gene_names is None or len(gene_names) != n_genes:
            raise ValueError("gene_names (len == n_genes) required for seurat_v3 HVG.")
        if batches is None or len(batches) != n_samples:
            raise ValueError("batches (len == n_samples) required for seurat_v3 HVG.")
        if norm_layer is None or norm_layer.shape != X_counts.shape:
            raise ValueError("norm_layer must match X_counts shape for seurat_v3 HVG.")

    detailed_rows: List[Dict[str, Any]] = []

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_counts), start=1):
        _hb(heartbeat_path, f"SCVI fold {fold_id}/{k} start")
        with ut.Timer(f"[scVI] fold {fold_id}/{k}"):
            X_tr, X_va = X_counts[tr_idx], X_counts[va_idx]

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
            ut.log(f"[scVI] fold {fold_id} | HVG={len(G)}")

            X_tr_G = X_tr[:, G]
            X_va_G = X_va[:, G]
            theta_G = ut.estimate_nb_theta_moments(X_tr_G)

            for p in ParameterGrid(scvi_grid):
                ut.log(f"[scVI] fold {fold_id} params={p}")
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

                mu_full = mdl.predict_mean(
                    X_va_G,
                    eval_batches=(batches[va_idx] if batches is not None else None),
                )
                from sklearn.metrics import silhouette_score
                
                ASW_batch = None
                if batches is not None:
                    Z = mdl.get_latent(X_va_G, eval_batches=batches[va_idx])
                    b = np.asarray(batches[va_idx])
                    if len(np.unique(b)) >= 2 and Z.shape[0] >= 3:
                        try:
                            ASW_batch = float(silhouette_score(Z, b))
                        except Exception:
                            ASW_batch = None
                sil = ut.bio_silhouette_score(mu_full, labels[va_idx] if labels is not None else None)

                zero_errs, zero_nbs = [], []
                for r in range(R):
                    rng_zero = np.random.default_rng(random_state + 1111 + fold_id * 1000 + r)

                    M_zero = ut.make_mask_nonzero_by_gene(X_va_G, frac=mask_frac, rng=rng_zero)
                    X_va_masked = X_va_G.copy()
                    X_va_masked[M_zero] = 0

                    mu_zero = mdl.predict_mean(
                        X_va_masked,
                        eval_batches=(batches[va_idx] if batches is not None else None),
                    )

                    rows, cols = np.where(M_zero)
                    y_true = X_va_G[rows, cols]
                    y_pred = mu_zero[rows, cols]
                    th = theta_G[cols]

                    zero_errs.append(ut._basic_errors(y_true, y_pred))
                    zero_nbs.append(ut._nb_scores(y_true, y_pred, th))

                thin_errs, thin_nbs = [], []
                for r in range(R):
                    rng_thin = np.random.default_rng(random_state + 7777 + fold_id * 1000 + r)

                    X_keep, X_hold = ut.binomial_thinning_split(X_va_G, p_hold=thinning_p, rng=rng_thin)

                    mu_keep = mdl.predict_mean(
                        X_keep,
                        eval_batches=(batches[va_idx] if batches is not None else None),
                    )

                    y_true = X_hold
                    y_pred = (thinning_p / (1.0 - thinning_p)) * mu_keep

                    thin_errs.append(ut._basic_errors(y_true, y_pred))
                    thin_nbs.append(ut._nb_scores(y_true, y_pred, theta_G))

                detailed_rows.append({
                    "fold": fold_id,
                    "model": "scVI",
                    "params": {k_: p[k_] for k_ in p},
                    "n_hvg": int(len(G)),
                    "MAE_zero": float(np.mean([d["MAE"] for d in zero_errs])) if zero_errs else None,
                    "MSE_zero": float(np.mean([d["MSE"] for d in zero_errs])) if zero_errs else None,
                    "MedianL1_zero": float(np.mean([d["MedianL1"] for d in zero_errs])) if zero_errs else None,
                    "NB_ll_zero": float(np.mean([d["NB_ll"] for d in zero_nbs])) if zero_nbs else None,
                    "NB_dev_zero": float(np.mean([d["NB_dev"] for d in zero_nbs])) if zero_nbs else None,
                    "MAE_thin": float(np.mean([d["MAE"] for d in thin_errs])) if thin_errs else None,
                    "MSE_thin": float(np.mean([d["MSE"] for d in thin_errs])) if thin_errs else None,
                    "MedianL1_thin": float(np.mean([d["MedianL1"] for d in thin_errs])) if thin_errs else None,
                    "NB_ll_thin": float(np.mean([d["NB_ll"] for d in thin_nbs])) if thin_nbs else None,
                    "NB_dev_thin": float(np.mean([d["NB_dev"] for d in thin_nbs])) if thin_nbs else None,
                    "Silhouette": sil,
                })
                _maybe_flush_csv(pd.DataFrame(detailed_rows), save_dir, "scvi_detailed")

        _hb(heartbeat_path, f"SCVI fold {fold_id}/{k} done")

    detailed_df = pd.DataFrame(detailed_rows)
    agg_cols = [c for c in detailed_df.columns if c not in ("fold", "model", "params", "n_hvg")]
    summary = (detailed_df
               .groupby([detailed_df["params"].apply(lambda d: tuple(sorted(d.items()))), "n_hvg"])
               [agg_cols].mean().reset_index())
    summary = summary.rename(columns={"params": "params_tuple"})
    summary["params"] = summary["params_tuple"].apply(lambda t: dict(t))
    summary = summary.drop(columns=["params_tuple"])
    summary.insert(0, "model", "scVI")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        detailed_df.to_csv(os.path.join(save_dir, "scvi_detailed.csv"), index=False)
        summary.to_csv(os.path.join(save_dir, "scvi_summary.csv"), index=False)
        ut.log(f"[cv_scvi_5fold] wrote CSVs to {save_dir}")

    ut.log("[cv_scvi_5fold] done")
    return summary, detailed_df
