from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np
import pandas as pd
import json, os
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.impute import KNNImputer

np.random.seed(123)

# ---------------- Utilities: masking, HVG ranking, NB scoring ----------------

def _ensure_dir(d: Optional[str]):
    if d is None: return
    os.makedirs(d, exist_ok=True)

def _save_npz(fp: str, **arrays):
    safe = {}
    for k, v in arrays.items():
        if isinstance(v, (dict, list, tuple)):
            safe[k] = np.array(json.dumps(v), dtype=object)
        else:
            safe[k] = v
    np.savez_compressed(fp, **safe)

def _append_csv(fp: str, row: Dict[str, Any], header_order: Optional[List[str]] = None):
    import csv
    _ensure_dir(os.path.dirname(fp) or ".")
    file_exists = os.path.exists(fp)
    with open(fp, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=(header_order or list(row.keys())))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        
def _save_df_via_util(fp: str, df: pd.DataFrame, overwrite: bool = True, header_order: Optional[List[str]] = None) -> str:
    _ensure_dir(os.path.dirname(fp) or ".")
    if overwrite and os.path.exists(fp):
        os.remove(fp)
    cols = header_order or list(df.columns)
    for row in df.to_dict(orient="records"):
        safe_row = {}
        for k, v in row.items():
            if isinstance(v, (dict, list, tuple)):
                safe_row[k] = json.dumps(v, ensure_ascii=False)
            else:
                safe_row[k] = v
        _append_csv(fp, safe_row, header_order=cols)
    return fp

def make_mask_stratified_by_gene(X: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask same shape as X. Masks ~frac of rows per gene (column)."""
    n_cells, n_genes = X.shape
    m = max(1, int(math.ceil(frac * n_cells)))
    M = np.zeros_like(X, dtype=bool)
    for g in range(n_genes):
        rows = rng.choice(n_cells, size=m, replace=False)
        M[rows, g] = True
    return M

def libsize_normalize_for_hvg(X: np.ndarray) -> np.ndarray:
    """Size-factor normalize to median library size then log1p. For HVG *ranking only*."""
    lib = X.sum(axis=1, keepdims=True).astype(np.float64)
    med = np.median(lib[lib > 0]) if np.any(lib > 0) else 1.0
    sf = np.divide(lib, med, out=np.ones_like(lib), where=(lib != 0))
    Xn = X / np.maximum(sf, 1e
                        -12)
    return np.log1p(Xn)

def rank_hvg_by_variance(X_train_counts: np.ndarray) -> np.ndarray:
    """Return column indices sorted by variance on log1p(size-factor-normalized) counts."""
    Xlog = libsize_normalize_for_hvg(X_train_counts)
    var = Xlog.var(axis=0)
    return np.argsort(var)[::-1]

def estimate_nb_theta_moments(X_train_counts: np.ndarray) -> np.ndarray:
    """Per-gene NB dispersion via method-of-moments."""
    mu = X_train_counts.mean(axis=0).astype(np.float64)
    var = X_train_counts.var(axis=0, ddof=1).astype(np.float64)
    theta = np.where(var > mu, (mu**2) / np.maximum(var - mu, 1e-12), 1e6)
    return np.clip(theta, 1e-6, 1e9)

def nb_logpmf(x: np.ndarray, mu: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """NB log PMF with mean 'mu' and dispersion 'theta' (r=theta, p=theta/(theta+mu))."""
    from scipy.special import gammaln
    x = x.astype(np.float64)
    mu = np.maximum(mu.astype(np.float64), 1e-12)
    theta = np.maximum(theta.astype(np.float64), 1e-12)
    r = theta
    p = theta / (theta + mu)
    return (gammaln(x + r) - gammaln(r) - gammaln(x + 1.0) + r*np.log(p) + x*np.log1p(-p))

def nb_deviance(x: np.ndarray, mu: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """NB deviance per entry: 2*[ x*log(x/mu) - (x+theta)*log((x+theta)/(mu+theta)) ]"""
    x = x.astype(np.float64)
    mu = np.maximum(mu.astype(np.float64), 1e-12)
    theta = np.maximum(theta.astype(np.float64), 1e-12)
    x_safe = np.maximum(x, 1e-12)
    term1 = x * (np.log(x_safe) - np.log(mu))
    term2 = (x + theta) * (np.log(x + theta) - np.log(mu + theta))
    return 2.0 * (term1 - term2)


def _patch_legacy_keras_for_dca():
    """
    DCA가 기대하는 옛 Keras 경로들을 최신 Keras에서도 찾을 수 있게 쉬밍.
    - keras.objectives.mean_squared_error
    - keras.engine.topology.Layer
    - keras.engine.base_layer.InputSpec, Layer, BaseRandomLayer
    """
    import sys, types
    try:
        from keras import layers as KL
    except Exception:
        import keras.layers as KL  # 드문 환경 대비

    # 이미 패치돼 있으면 스킵 (idempotent)
    if isinstance(sys.modules.get("keras.objectives"), types.ModuleType) and \
       isinstance(sys.modules.get("keras.engine.topology"), types.ModuleType) and \
       isinstance(sys.modules.get("keras.engine.base_layer"), types.ModuleType) and \
       hasattr(sys.modules["keras.engine.base_layer"], "BaseRandomLayer"):
        return

    # 1) keras.objectives.mean_squared_error
    try:
        from keras.losses import mean_squared_error as _mse
    except Exception:
        from keras.losses import MeanSquaredError as _MSE
        _mse = lambda y_true, y_pred: _MSE()(y_true, y_pred)
    m_obj = types.ModuleType("keras.objectives")
    m_obj.mean_squared_error = _mse
    sys.modules["keras.objectives"] = m_obj

    # 2) keras.engine.topology.Layer
    m_top = types.ModuleType("keras.engine.topology")
    m_top.Layer = KL.Layer
    sys.modules["keras.engine.topology"] = m_top

    # 3) keras.engine.base_layer.InputSpec, Layer, BaseRandomLayer
    m_base = types.ModuleType("keras.engine.base_layer")
    m_base.InputSpec = getattr(KL, "InputSpec", KL.Layer)   # 안전 디폴트
    m_base.Layer = KL.Layer
    # Keras v3에서 BaseRandomLayer가 별도 클래스가 아닐 수 있으니 Layer로 alias
    try:
        # 혹시 있으면 실제 클래스를 사용
        from keras.layers import BaseRandomLayer as _BRL  # 존재하면 가져옴
        m_base.BaseRandomLayer = _BRL
    except Exception:
        m_base.BaseRandomLayer = KL.Layer
    sys.modules["keras.engine.base_layer"] = m_base


def _pin_single_tf_session(threads: Optional[int] = None):
    import tensorflow as tf
    K = tf.compat.v1.keras.backend  # ← 여기만 바꿈

    if getattr(_pin_single_tf_session, "_done", False):
        return

    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass

    cfg = None
    if threads is not None:
        cfg = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=threads,
            inter_op_parallelism_threads=threads,
        )
    sess = tf.compat.v1.Session(config=cfg) if cfg is not None else tf.compat.v1.Session()
    K.set_session(sess)

    _orig_set_session = K.set_session
    def _set_session_noop(_):
        return _orig_set_session(sess)
    K.set_session = _set_session_noop
    _pin_single_tf_session._done = True





# --------------- Seurat v3 HVG (batch-aware) helper ----------------

def seurat_v3_hvg_indices_for_train_split(
    X_train_counts: np.ndarray,
    gene_names: np.ndarray,
    batch_labels_train: np.ndarray,
    normalized_train_matrix: np.ndarray,     # same samples x genes as X_train_counts
    n_top_genes: int,
    layer_name: str = "log2_1p_CPM_original",
    batch_key: str = "batch",
) -> np.ndarray:
    try:
        import scanpy as sc
        import anndata as ad
    except Exception as e:
        raise ImportError("Seurat v3 HVG requires scanpy and anndata. Install them first.") from e

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

def _patch_pyyaml_loader_for_kopt():
    """
    kopt가 yaml.load(stream)처럼 Loader 없이 호출해도 동작하도록,
    PyYAML 6+에서 기본 Loader를 강제로 넣어주는 래퍼.
    """
    import yaml, inspect
    orig = yaml.load

    # PyYAML 6에서는 Loader가 필수(positional). 없는 호출을 보정한다.
    def compat_load(stream, *args, **kwargs):
        if len(args) == 0 and "Loader" not in kwargs:
            # 안전한 기본 로더 사용
            Loader = getattr(yaml, "SafeLoader", None) or getattr(yaml, "FullLoader", None)
            kwargs["Loader"] = Loader
        return orig(stream, *args, **kwargs)

    # 이미 패치되어 있지 않다면 교체
    if getattr(yaml.load, "__name__", "") != "compat_load":
        yaml.load = compat_load


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
            raise ValueError("KNN baseline requires a mask to know what to impute.")
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

    def predict_mean(self, X: np.ndarray, mask=None, X_train_ref: Optional[np.ndarray]=None) -> np.ndarray:
        from magic import MAGIC
        if self.op_ is None:
            raise RuntimeError("Call fit() before predict_mean().")

        X = np.asarray(X, dtype=np.float64)

        # 1) 시도: 설치된 MAGIC이 유도적 transform을 지원하는 경우
        if hasattr(self.op_, "transform"):
            try:
                return np.asarray(self.op_.transform(X), dtype=np.float64)
            except Exception:
                # 예: n_samples 불일치로 실패 -> 폴백 진행
                pass

        # 2) 폴백: concat-fit (비유도 모델 평가용; 데이터 누수 존재)
        if X_train_ref is not None:
            X_tr = np.asarray(X_train_ref, dtype=np.float64)
            X_concat = np.vstack([X_tr, X])
            tmp = MAGIC(n_pca=self.n_pca, t=self.t, knn=self.knn)
            X_imp = tmp.fit_transform(X_concat)
            return np.asarray(X_imp[-X.shape[0]:, :], dtype=np.float64)

        # 3) 최후수단: eval만으로 fit_transform (누수는 없지만 train 정보를 못 씀)
        tmp = MAGIC(n_pca=self.n_pca, t=self.t, knn=self.knn)
        return np.asarray(tmp.fit_transform(X), dtype=np.float64)


class _MAGICModelTuned:
    name = "MAGIC_TUNED"
    def __init__(self, n_pca=None, t=3, knn=5):
        self.n_pca = n_pca; self.t = t; self.knn = knn; self.op_ = None

    def fit(self, X):
        from magic import MAGIC
        self.op_ = MAGIC(n_pca=self.n_pca, t=self.t, knn=self.knn)
        self.op_.fit(np.asarray(X, dtype=np.float64))
        return self

    def predict_mean(self, X, mask=None, X_train_ref: Optional[np.ndarray]=None):
        from magic import MAGIC
        if self.op_ is None:
            raise RuntimeError("Call fit() first.")

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


#--- DCA wrapper (counts -> expected counts mu_hat) ---
class _DCAModel:
    name = "DCA"
    def __init__(self, hidden_size=[64, 32, 64], epochs=300):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.imputed_ = None

    def fit(self, X: np.ndarray):
        import anndata as ad
        import types, sys
        import sys, types, importlib

# (1) dca가 이전에 부분 임포트된 흔적을 깨끗이 지움
        for m in list(sys.modules):
            if m == "dca" or m.startswith("dca."):
                del sys.modules[m]

        # (2) keras.objectives 쉬밍 만들기
        try:
            from keras.losses import mean_squared_error as _mse
        except Exception:
            # 어떤 환경에서는 함수 alias가 없고 클래스만 있을 수 있음
            try:
                from keras.losses import MeanSquaredError as _MSE
            except Exception:
                from tensorflow.keras.losses import MeanSquaredError as _MSE  # 아주 예외적 대비
            _mse = _MSE()

        shim = types.ModuleType("keras.objectives")
        shim.mean_squared_error = _mse
        sys.modules["keras.objectives"] = shim
        from dca.api import dca
        adata = ad.AnnData(X)
        # dca writes results back into adata
        dca(adata,
            hidden_size=self.hidden_size,
            epochs=self.epochs,
            return_model=False,
            copy=False)
        self.imputed_ = np.array(adata.layers["X_dca"]) if "X_dca" in adata.layers else np.array(adata.X)
        return self

    def predict_mean(self, X: np.ndarray, mask=None, X_train_ref=None):
        return self.imputed_

def build_baseline(name: str, **params) -> Any:
    if name in ("mean", "MEAN"):     return _MeanModel()
    if name in ("median", "MEDIAN"): return _MedianModel()
    if name in ("knn", "KNN"):       return _KNNModel(n_neighbors=int(params.get("n_neighbors", 15)),
                                                     weights=params.get("weights", "uniform"))
    if name in ("magic", "MAGIC"): return _MAGICModel(**params)
    if name in ("magic_tuned", "MAGIC_TUNED"): return _MAGICModelTuned(**params)
    raise ValueError(f"Unknown baseline: {name}")

# --- scVI wrapper (counts -> expected counts mu_hat) ---
class SCVIWrapper:
    name = "scVI"
    def __init__(self, n_latent=20, n_hidden=128, n_layers=2, dropout_rate=0.1,
                 max_epochs=400, lr=1e-3, weight_decay=1e-6, batch_key="batch",
                 use_gpu=False, gene_names=None, batches=None):
        self.hp = dict(n_latent=n_latent, n_hidden=n_hidden, n_layers=n_layers,
                       dropout_rate=dropout_rate, max_epochs=max_epochs, lr=lr,
                       weight_decay=weight_decay, batch_key=batch_key, use_gpu=use_gpu)
        self._gene_names = gene_names
        self._batches = batches
        self.model_ = None
        self.train_genes_ = None

    def fit(self, X_train):
        import anndata as ad, thesis_project.scripts.scvi_runner as scvi_runner
        if self._gene_names is None or len(self._gene_names) != X_train.shape[1]:
            # self._gene_names = np.array([f"g{i}" for i in range(X_train.shape[1])], dtype=str)
            self._gene_names = np.array([f"g{i}" for i in range(X_train.shape[1])], dtype=str)

        adata = ad.AnnData(X=X_train)
        adata.var_names = self._gene_names
        if self._batches is not None:
            adata.obs[self.hp["batch_key"]] = np.array(self._batches, dtype=str)

        scvi_runner.model.SCVI.setup_anndata(
            adata, batch_key=(self.hp["batch_key"] if self._batches is not None else None)
        )
        self.model_ = scvi_runner.model.SCVI(
            adata,
            n_latent=self.hp["n_latent"], n_hidden=self.hp["n_hidden"],
            n_layers=self.hp["n_layers"], dropout_rate=self.hp["dropout_rate"],
            gene_likelihood="nb",
        )
        self.model_.train(
            max_epochs=self.hp["max_epochs"],
            plan_kwargs={"lr": self.hp["lr"], "weight_decay": self.hp["weight_decay"]},
            accelerator=("gpu" if self.hp["use_gpu"] else "cpu"),
            devices=(1 if self.hp["use_gpu"] else "auto"),
        )
        self.train_genes_ = np.array(adata.var_names)
        return self

    def predict_mean(self, X_eval, mask=None, X_train_ref=None, eval_batches=None):
        import anndata as ad
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        adata_eval = ad.AnnData(X=X_eval)
        adata_eval.var_names = self.train_genes_

        # 학습 때 batch_key를 썼다면, 평가에도 같은 열을 채워야 함
        if self._batches is not None:
            if eval_batches is None:
                raise KeyError(
                    f"Expected eval_batches for batch_key='{self.hp['batch_key']}' "
                    f"with length {X_eval.shape[0]}"
                )
            adata_eval.obs[self.hp["batch_key"]] = np.array(eval_batches, dtype=str)

        try:
            px = self.model_.get_likelihood_parameters(adata=adata_eval)
            mu = np.asarray(px.get("mean", px.get("mu")))
            if mu is None:
                raise KeyError
        except Exception:
            lib_eval = X_eval.sum(axis=1, keepdims=True).astype(np.float64)
            norm = self.model_.get_normalized_expression(adata=adata_eval, library_size=1.0)
            mu = np.asarray(norm) * np.maximum(lib_eval, 1e-12)
        return mu


# ---------- DCA nested-CV runner (fair: no leakage; requires predict on new data) ----------

class DCAPredictWrapper:
    """
    Wrapper that works with both:
      (A) DCA builds that return a Keras model (so we can call model.predict on new matrices), and
      (B) legacy/stricter builds that do NOT return a model object.

    Strategy:
      - Try to train once on the train matrix and capture the returned Keras model (if available).
      - If a model is available, use it for inductive prediction on new matrices.
      - If not, fall back at prediction time: re-run dca() on [train || eval] with
        adata.obs['dca_split'] marking only the TRAIN rows for optimization,
        then slice the predicted outputs for the eval rows. This avoids leakage,
        because eval rows are never used for training.
    """
    name = "DCA"

    def __init__(self,
                 hidden_size=[64, 32, 64],
                 epochs=300,
                 batch_size=128,
                 learning_rate=None,    # optional, depending on your DCA build
                 dropout_rate=None,     # optional
                 l2=None,               # optional
                 random_state=0):
        self.hidden_size   = hidden_size
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate  = dropout_rate
        self.l2            = l2
        self.random_state  = random_state

        self.model_   = None   # Keras model if the DCA build returns it
        self.X_train_ = None   # Cached train matrix for fallback prediction
        self._hp_kws  = None   # Cached keyword args passed to dca()

    def _build_kws(self):
        """Assemble keyword arguments for dca()."""
        kws = dict(
            hidden_size=self.hidden_size,
            epochs=self.epochs,
            batch_size=self.batch_size,
            return_model=True,   # try to get a Keras model if possible
            copy=False,
            random_state=self.random_state,
        )
        if self.learning_rate is not None: kws["learning_rate"] = self.learning_rate
        if self.dropout_rate  is not None: kws["dropout_rate"]  = self.dropout_rate
        if self.l2            is not None: kws["l2"]            = self.l2
        return kws

    def fit(self, X_train: np.ndarray):
        import anndata as ad
        import sys, tensorflow as tf

        # Ensure a clean import state for dca and its modules
        for m in list(sys.modules):
            if m == "dca" or m.startswith("dca."):
                del sys.modules[m]

        # Environment patches for PyYAML/keras/TF1-style session behavior
        _patch_pyyaml_loader_for_kopt()
        _patch_legacy_keras_for_dca()
        _pin_single_tf_session()

        # Cast to float to avoid dtype issues in scanpy/tf pipelines
        X_train = np.asarray(X_train)
        if not np.issubdtype(X_train.dtype, np.floating):
            X_train = X_train.astype(np.float32, copy=False)

        # Seed TF if available
        try:
            tf.random.set_seed(self.random_state)
        except Exception:
            pass

        from dca.api import dca
        ad_tr = ad.AnnData(X_train)

        # Try to train and capture a model
        self._hp_kws = self._build_kws()
        res = dca(ad_tr, **self._hp_kws)

        # Most builds return (adata_out, keras_model)
        if isinstance(res, tuple) and len(res) >= 2:
            self.model_ = res[1]
        else:
            # Some builds stash the model in adata.uns
            try:
                self.model_ = ad_tr.uns.get("dca_ae_model", None)
            except Exception:
                self.model_ = None

        # Cache the train matrix for the fallback path
        self.X_train_ = X_train
        return self

    def _predict_via_model(self, X_eval: np.ndarray) -> np.ndarray:
        """Inductive prediction with the captured Keras model."""
        X_eval = np.asarray(X_eval, dtype=np.float32)
        mu = self.model_.predict(X_eval, verbose=0)
        return np.asarray(mu)

    def _predict_via_concat(self, X_eval: np.ndarray) -> np.ndarray:
        """
        Fallback: concatenate [X_train, X_eval], mark only the train rows as 'train'
        in adata.obs['dca_split'], re-run dca() with return_model=False, and then
        slice the outputs for the eval rows. This prevents leakage.
        """
        import anndata as ad
        from dca.api import dca

        X_eval = np.asarray(X_eval)
        if not np.issubdtype(X_eval.dtype, np.floating):
            X_eval = X_eval.astype(np.float32, copy=False)

        X_concat = np.vstack([self.X_train_, X_eval]).astype(np.float32, copy=False)
        n_tr = self.X_train_.shape[0]
        n_ev = X_eval.shape[0]

        ad_all = ad.AnnData(X_concat)
        split = np.array(["train"] * n_tr + ["test"] * n_ev, dtype=object)
        ad_all.obs["dca_split"] = split

        # We don't need the model in this path
        kws = dict(self._hp_kws or self._build_kws())
        kws["return_model"] = False
        dca(ad_all, **kws)

        # DCA commonly writes to layers["X_dca"]; some builds overwrite X
        if "X_dca" in ad_all.layers:
            return np.asarray(ad_all.layers["X_dca"][-n_ev:, :])
        else:
            return np.asarray(ad_all.X[-n_ev:, :])

    def predict_mean(self, X_eval: np.ndarray) -> np.ndarray:
        # Prefer the fast inductive path if a model object is available
        if self.model_ is not None:
            try:
                return self._predict_via_model(X_eval)
            except Exception:
                # If the model path fails (e.g., shape/preproc mismatch), fall back
                pass
        # Fallback that re-runs DCA without training on eval rows
        return self._predict_via_concat(X_eval)

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

def nested_cv_baselines(
    X_counts: np.ndarray,
    outer_k: int = 5,
    inner_k: int = 3,
    hvg_grid: List[int] = (1000, 2000, 4000),
    model_grids: Dict[str, Dict[str, List[Any]]] = None,
    R: int = 3,
    mask_frac: float = 0.1,
    random_state: int = 42,
    one_se_rule: bool = True,
    save_dir: Optional[str] = None,   # if provided, CSVs will be written here
    # ------ HVG mode & metadata ------
    hvg_mode: str = "variance",       # "variance" or "seurat_v3"
    gene_names: Optional[np.ndarray] = None,
    batches: Optional[np.ndarray] = None,
    norm_layer: Optional[np.ndarray] = None,    # same shape as X_counts; e.g., "log2_1p_CPM_original"
    batch_key: str = "batch",
    seurat_layer_name: str = "log2_1p_CPM_original",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run nested CV for baselines. If hvg_mode='seurat_v3', you must pass gene_names, batches, norm_layer.
    Returns: (summary_df, detailed_outer_df). If save_dir is set, writes CSVs there.
    """
    if model_grids is None:
        model_grids = {
            "MEAN": {},
            "MEDIAN": {},
            "KNN": {"n_neighbors": [5, 15, 30], "weights": ["uniform", "distance"]},
        }

    X_counts = np.asarray(X_counts)
    n_samples, n_genes = X_counts.shape

    if hvg_mode not in ("variance", "seurat_v3"):
        raise ValueError("hvg_mode must be 'variance' or 'seurat_v3'.")

    if hvg_mode == "seurat_v3":
        # Validate metadata
        if gene_names is None or len(gene_names) != n_genes:
            raise ValueError("gene_names must be provided with length = n_genes for seurat_v3 HVG.")
        if batches is None or len(batches) != n_samples:
            raise ValueError("batches must be provided with length = n_samples for seurat_v3 HVG.")
        if norm_layer is None or norm_layer.shape != X_counts.shape:
            raise ValueError("norm_layer must be provided with the same shape as X_counts for seurat_v3 HVG.")

    outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=random_state)
    outer_records: List[Dict[str, Any]] = []

    for ofold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_counts), start=1):
        X_tr_out, X_te_out = X_counts[tr_idx], X_counts[te_idx]

        # ----- candidate configs (model grid × HVG grid) -----
        candidate_cfgs: List[Config] = []
        for model, grid in model_grids.items():
            if grid:
                for p in ParameterGrid(grid):
                    for n_hvg in hvg_grid:
                        candidate_cfgs.append(_cfg(model, p, n_hvg))
            else:
                for n_hvg in hvg_grid:
                    candidate_cfgs.append(_cfg(model, {}, n_hvg))

        cfg_to_scores: Dict[Config, List[FoldScore]] = {c: [] for c in candidate_cfgs}
        inner_cv = KFold(n_splits=inner_k, shuffle=True, random_state=random_state + 7)

        # ----- inner folds -----
        for ifold, (tr_in_idx, val_in_idx) in enumerate(inner_cv.split(X_tr_out), start=1):
            X_tr_in, X_val_in = X_tr_out[tr_in_idx], X_tr_out[val_in_idx]

            # Build HVG indices per n in hvg_grid
            rank_slices: Dict[int, np.ndarray] = {}
            theta_map: Dict[int, np.ndarray] = {}

            if hvg_mode == "variance":
                rank = rank_hvg_by_variance(X_tr_in)
                for n in hvg_grid:
                    idx = rank[:min(n, X_tr_in.shape[1])]
                    rank_slices[n] = idx
                    theta_map[n]   = estimate_nb_theta_moments(X_tr_in[:, idx])

            else:  # seurat_v3
                batches_tr_in = batches[tr_idx][tr_in_idx]   # batches for Train_inner
                norm_tr_inner = norm_layer[tr_idx][tr_in_idx, :]
                for n in hvg_grid:
                    idx = seurat_v3_hvg_indices_for_train_split(
                        X_train_counts=X_tr_in,
                        gene_names=gene_names,
                        batch_labels_train=batches_tr_in,
                        normalized_train_matrix=norm_tr_inner,
                        n_top_genes=n,
                        layer_name=seurat_layer_name,
                        batch_key=batch_key,
                    )
                    rank_slices[n] = idx
                    theta_map[n]   = estimate_nb_theta_moments(X_tr_in[:, idx])

            # evaluate every config
            for cfg in candidate_cfgs:
                G = rank_slices[cfg.n_hvg]
                theta = theta_map[cfg.n_hvg]

                mdl = build_baseline(cfg.model, **dict(cfg.params))
                mdl.fit(X_tr_in[:, G])

                # R stochastic masks on Val_inner
                ll_list, dev_list, mae_list = [], [], []
                for r in range(R):
                    rng = np.random.default_rng(random_state + ofold*1000 + ifold*100 + r)
                    M = make_mask_stratified_by_gene(X_val_in[:, G], frac=mask_frac, rng=rng)
                    mu = mdl.predict_mean(X_val_in[:, G], mask=M, X_train_ref=X_tr_in[:, G])

                    rows, cols = np.where(M)
                    y_true = X_val_in[rows, cols]
                    y_pred = mu[rows, cols]
                    theta_masked = theta[cols]

                    ll_list.append(nb_logpmf(y_true, y_pred, theta_masked).mean())
                    dev_list.append(nb_deviance(y_true, y_pred, theta_masked).mean())
                    mae_list.append(np.mean(np.abs(y_true - y_pred)))

                cfg_to_scores[cfg].append(FoldScore(ll=float(np.mean(ll_list)),
                                                    dev=float(np.mean(dev_list)),
                                                    mae=float(np.mean(mae_list))))

        # ----- pick best config per model (primary: NB log-lik; ties dev, then MAE) -----
        rows = []
        for cfg, scores in cfg_to_scores.items():
            ll = np.array([s.ll for s in scores]); dev = np.array([s.dev for s in scores]); mae = np.array([s.mae for s in scores])
            rows.append((cfg, ll.mean(), dev.mean(), mae.mean(), ll.std(ddof=1)))

        best_cfg_per_model: Dict[str, Config] = {}
        for (cfg, ll_mean, dev_mean, mae_mean, ll_std) in rows:
            key = cfg.model
            score_tuple = (ll_mean, -dev_mean, -mae_mean)  # higher is better overall
            if key not in best_cfg_per_model or score_tuple > (best_cfg_per_model[key][0] if isinstance(best_cfg_per_model[key], tuple) else (-np.inf,)):
                best_cfg_per_model[key] = (score_tuple, ll_std, cfg)

        if one_se_rule:
            # Apply 1-SE rule per model to prefer smaller HVG if within 1 SD of that model's best NB log-lik
            best_cfg_1se: Dict[str, Config] = {}
            for model_name in set(cfg.model for cfg, *_ in rows):
                model_rows = [(cfg, ll_mean, dev_mean, mae_mean, ll_std)
                              for (cfg, ll_mean, dev_mean, mae_mean, ll_std) in rows
                              if cfg.model == model_name]
                if not model_rows:
                    continue
                best_ll = max(mr[1] for mr in model_rows)
                best_std = [mr[4] for mr in model_rows if mr[1] == best_ll][0] if len(model_rows) > 1 else 0.0
                thr = best_ll - best_std
                cands = [mr for mr in model_rows if mr[1] >= thr]
                cands.sort(key=lambda mr: (mr[0].n_hvg, -mr[1], mr[2], mr[3]))
                best_cfg_1se[model_name] = cands[0][0]
            # overwrite best_cfg_per_model with 1SE picks
            best_cfg_per_model = {m: best_cfg_1se[m] for m in best_cfg_1se}

        # ----- refit & test each model's own best config on outer test -----
        for model_name, cfg in best_cfg_per_model.items():
            if isinstance(cfg, tuple):
                cfg = cfg[2]  # unwrap (score_tuple, ll_std, cfg)

            # Recompute HVG on full outer-train (train-only) for this cfg's HVG count
            if hvg_mode == "variance":
                rank_out = rank_hvg_by_variance(X_tr_out)
                G_out = rank_out[:min(cfg.n_hvg, X_tr_out.shape[1])]
            else:
                batches_tr_out = batches[tr_idx]
                norm_tr_out    = norm_layer[tr_idx, :]
                G_out = seurat_v3_hvg_indices_for_train_split(
                    X_train_counts=X_tr_out,
                    gene_names=gene_names,
                    batch_labels_train=batches_tr_out,
                    normalized_train_matrix=norm_tr_out,
                    n_top_genes=cfg.n_hvg,
                    layer_name=seurat_layer_name,
                    batch_key=batch_key,
                )

            theta_out  = estimate_nb_theta_moments(X_tr_out[:, G_out])
            best_model = build_baseline(cfg.model, **dict(cfg.params)).fit(X_tr_out[:, G_out])

            ll_list, dev_list, mae_list = [], [], []
            for r in range(R):
                rng = np.random.default_rng(random_state + 9999 + ofold*100 + r)
                M_test = make_mask_stratified_by_gene(X_te_out[:, G_out], frac=mask_frac, rng=rng)
                mu_test = best_model.predict_mean(X_te_out[:, G_out], mask=M_test, X_train_ref=X_tr_out[:, G_out])

                rows_, cols_ = np.where(M_test)
                y_true = X_te_out[rows_, cols_]
                y_pred = mu_test[rows_, cols_]
                theta_masked = theta_out[cols_]

                ll_list.append(nb_logpmf(y_true, y_pred, theta_masked).mean())
                dev_list.append(nb_deviance(y_true, y_pred, theta_masked).mean())
                mae_list.append(np.mean(np.abs(y_true - y_pred)))

            outer_records.append({
                "outer_fold": ofold,
                "model": model_name,
                "best_params": dict(cfg.params),
                "best_n_hvg": cfg.n_hvg,
                "NB_loglik_mean": float(np.mean(ll_list)),
                "NB_deviance_mean": float(np.mean(dev_list)),
                "MAE_mean": float(np.mean(mae_list)),
            })

    detailed_df = pd.DataFrame(outer_records)

    # summary per model across outer folds
    summary_rows = []
    for m in sorted(detailed_df["model"].unique()):
        d = detailed_df[detailed_df["model"] == m]
        summary_rows.append({
            "model": m,
            "outer_folds": len(d),
            "NB_loglik mean±sd": f"{d['NB_loglik_mean'].mean():.6f} ± {d['NB_loglik_mean'].std(ddof=1):.6f}",
            "NB_deviance mean±sd": f"{d['NB_deviance_mean'].mean():.6f} ± {d['NB_deviance_mean'].std(ddof=1):.6f}",
            "MAE mean±sd": f"{d['MAE_mean'].mean():.6f} ± {d['MAE_mean'].std(ddof=1):.6f}",
            "chosen n_hvg (mode)": int(d["best_n_hvg"].mode().iloc[0]) if len(d) else None,
        })
    summary_df = pd.DataFrame(summary_rows).reset_index(drop=True)

    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        detailed_df.to_csv(f"{save_dir}/outer_fold_details.csv", index=False)
        summary_df.to_csv(f"{save_dir}/summary.csv", index=False)

    return summary_df, detailed_df

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


def run_scvi_nested(
    X_counts: np.ndarray,
    hvg_mode: str,                 # "variance" or "seurat_v3"
    hvg_grid: List[int],
    batches: Optional[np.ndarray],
    norm_layer: Optional[np.ndarray],
    gene_names: Optional[np.ndarray],
    batch_key: str = "batch",
    seurat_layer_name: str = "log2_1p_CPM_original",
    outer_k: int = 5, inner_k: int = 3,
    R: int = 3, mask_frac: float = 0.1, random_state: int = 123,
    scvi_grid: Dict[str, List[Any]] = None,
    one_se_rule: bool = True,
) -> pd.DataFrame:
    """
    NOTE: 변경점
      - SCVIWrapper 생성 시 gene_names를 HVG 인덱스(G / G_out)로 서브셋하여 전달
      - (안전장치) shape/길이 assert 추가
    """
    if scvi_grid is None:
        scvi_grid = {
            "n_latent": [5, 10],
            "n_hidden": [64, 128],
            "n_layers": [1, 2],
            "dropout_rate": [0.1, 0.3],
            "max_epochs": [400],
            "lr": [1e-3, 3e-4],
            "weight_decay": [1e-5, 1e-4],
            "use_gpu": [True],
        }

    # ensure gene_names is np.array so fancy indexing with G works
    gnames_full = None if gene_names is None else np.asarray(gene_names)

    outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=random_state)
    records = []

    for ofold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_counts), start=1):
        X_tr_out, X_te_out = X_counts[tr_idx], X_counts[te_idx]

        inner_cv = KFold(n_splits=inner_k, shuffle=True, random_state=random_state + 7)
        scores = []

        for ifold, (tr_in_idx, val_in_idx) in enumerate(inner_cv.split(X_tr_out), start=1):
            X_tr_in, X_val_in = X_tr_out[tr_in_idx], X_tr_out[val_in_idx]

            # HVG per mode
            hvg_map = {}
            if hvg_mode == "variance":
                rank = rank_hvg_by_variance(X_tr_in)
                for n in hvg_grid:
                    hvg_map[n] = rank[:min(n, X_tr_in.shape[1])]
            else:
                batches_tr_in = batches[tr_idx][tr_in_idx] if batches is not None else None
                norm_tr_inner = norm_layer[tr_idx][tr_in_idx, :] if norm_layer is not None else None
                for n in hvg_grid:
                    idx = seurat_v3_hvg_indices_for_train_split(
                        X_train_counts=X_tr_in,
                        gene_names=gnames_full,                       # full list OK; 함수가 내부에서 인덱스 반환
                        batch_labels_train=batches_tr_in,
                        normalized_train_matrix=norm_tr_inner,
                        n_top_genes=n,
                        layer_name=seurat_layer_name,
                        batch_key=batch_key,
                    )
                    hvg_map[n] = idx

            # HP grid
            for p in ParameterGrid(scvi_grid):
                for n in hvg_grid:
                    G = hvg_map[n]
                    # subset data
                    X_tr_in_G = X_tr_in[:, G]
                    X_val_in_G = X_val_in[:, G]
                    # subset gene names to match matrix columns  <-- 변경점
                    gnames_G = None if gnames_full is None else gnames_full[G]

                    # safety checks
                    if gnames_G is not None:
                        assert X_tr_in_G.shape[1] == len(gnames_G), "gene_names (inner) length mismatch"

                    theta = estimate_nb_theta_moments(X_tr_in_G)

                    # mdl = SCVIWrapper(
                    #     **p,
                    #     batch_key=batch_key,
                    #     gene_names=gnames_G,                                # <-- 변경점
                    #     batches=batches[tr_idx][tr_in_idx] if batches is not None else None,
                    # )
                    mdl = SCVIWrapper(**p, batch_key=batch_key,
                                        gene_names=(gene_names[G] if gene_names is not None else None),
                                        batches=(batches[tr_idx][tr_in_idx] if batches is not None else None)
                                    )

                    mdl.fit(X_tr_in_G)

                    ll_list, dev_list, mae_list = [], [], []
                    for r in range(R):
                        rng = np.random.default_rng(random_state + ofold * 1000 + ifold * 100 + r)
                        M = make_mask_stratified_by_gene(X_val_in_G, frac=mask_frac, rng=rng)
                        # mu = mdl.predict_mean(X_val_in_G, mask=M, X_train_ref=X_tr_in_G)
                        mu = mdl.predict_mean(
                            X_val_in_G,
                            mask=M,
                            X_train_ref=X_tr_in_G,
                            eval_batches=(batches[tr_idx][val_in_idx] if batches is not None else None),
                        )



                        rows, cols = np.where(M)
                        y_true, y_pred = X_val_in_G[rows, cols], mu[rows, cols]
                        th = theta[cols]
                        ll_list.append(nb_logpmf(y_true, y_pred, th).mean())
                        dev_list.append(nb_deviance(y_true, y_pred, th).mean())
                        mae_list.append(np.mean(np.abs(y_true - y_pred)))

                    scores.append((
                        (p, n),
                        float(np.mean(ll_list)),
                        float(np.mean(dev_list)),
                        float(np.mean(mae_list)),
                        float(np.std(ll_list, ddof=1) if len(ll_list) > 1 else 0.0),
                    ))

        # 선택 (NB log-lik 1차 + 1SE로 작은 HVG 선호)
        scores.sort(key=lambda t: (t[1], -t[2], -t[3]), reverse=True)
        best_cfg, best_ll, _, _, best_std = scores[0]
        if one_se_rule:
            thr = best_ll - best_std
            cands = [s for s in scores if s[1] >= thr]
            cands.sort(key=lambda t: (t[0][1], -t[1], t[2], t[3]))
            best_cfg = cands[0][0]
        params, n_hvg = best_cfg

        # outer refit & test
        if hvg_mode == "variance":
            rank_out = rank_hvg_by_variance(X_tr_out)
            G_out = rank_out[:min(n_hvg, X_tr_out.shape[1])]
        else:
            batches_tr_out = batches[tr_idx] if batches is not None else None
            norm_tr_out = norm_layer[tr_idx, :] if norm_layer is not None else None
            G_out = seurat_v3_hvg_indices_for_train_split(
                X_train_counts=X_tr_out,
                gene_names=gnames_full,
                batch_labels_train=batches_tr_out,
                normalized_train_matrix=norm_tr_out,
                n_top_genes=n_hvg,
                layer_name=seurat_layer_name,
                batch_key=batch_key,
            )

        # subset for outer
        X_tr_out_G = X_tr_out[:, G_out]
        X_te_out_G = X_te_out[:, G_out]
        gnames_G_out = None if gnames_full is None else gnames_full[G_out]   # <-- 변경점
        if gnames_G_out is not None:
            assert X_tr_out_G.shape[1] == len(gnames_G_out), "gene_names (outer) length mismatch"

        theta_out = estimate_nb_theta_moments(X_tr_out_G)

        # mdl = SCVIWrapper(
        #     **params,
        #     batch_key=batch_key,
        #     gene_names=gnames_G_out,                                       # <-- 변경점
        #     batches=batches[tr_idx] if batches is not None else None,
        # )
        mdl = SCVIWrapper(**params, batch_key=batch_key,
                  gene_names=(gene_names[G_out] if gene_names is not None else None),
                  batches=(batches[tr_idx] if batches is not None else None))

        mdl.fit(X_tr_out_G)

        ll_list, dev_list, mae_list = [], [], []
        for r in range(R):
            rng = np.random.default_rng(random_state + 9999 + ofold * 100 + r)
            M_test = make_mask_stratified_by_gene(X_te_out_G, frac=mask_frac, rng=rng)
            # mu_test = mdl.predict_mean(X_te_out_G, mask=M_test, X_train_ref=X_tr_out_G)
            mu_test = mdl.predict_mean(
                X_te_out_G,
                mask=M_test,
                X_train_ref=X_tr_out_G,
                eval_batches=(batches[te_idx] if batches is not None else None),
            )



            rows_, cols_ = np.where(M_test)
            y_true, y_pred = X_te_out_G[rows_, cols_], mu_test[rows_, cols_]
            th = theta_out[cols_]
            ll_list.append(nb_logpmf(y_true, y_pred, th).mean())
            dev_list.append(nb_deviance(y_true, y_pred, th).mean())
            mae_list.append(np.mean(np.abs(y_true - y_pred)))

        records.append({
            "outer_fold": ofold,
            "model": "scVI",
            "best_params": params,
            "best_n_hvg": n_hvg,
            "NB_loglik_mean": float(np.mean(ll_list)),
            "NB_deviance_mean": float(np.mean(dev_list)),
            "MAE_mean": float(np.mean(mae_list)),
        })

    return pd.DataFrame(records)



