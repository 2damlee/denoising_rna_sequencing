#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run DCA CV + Optuna HPO headlessly (tmux friendly).

- Patch modern .h5ad to be readable by anndata 0.7.x (with backup)
- Load base_legacy.h5ad + attach layers from .npz
- Run Optuna search (lightweight, pruned, with timeout & checkpoints)
- Retrain with fuller settings and save CSVs/JSONs

Usage:
  conda activate dca_legacy
  python scripts/run_dca_hpo.py \
    --neu-root ../data/raw_count/converted/GSE169569 \
    --k-search 2 --trials 16 --timeout-hours 12 --jobs 1 \
    --out results/dca_full
"""

import os
# --- Thread caps (good on M1/CPU) ---
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS","4")
os.environ.setdefault("NUMEXPR_NUM_THREADS","4")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")

import sys
import json
import argparse
from pathlib import Path
import signal
import time
import platform
import subprocess

import numpy as np
import pandas as pd
import anndata as ad
import h5py
from scipy import sparse

from runner_models import cv_dca_5fold
import utilities as ut

# --------------- graceful shutdown ---------------
_SHOULD_STOP = False
def _handle_sigterm(signum, frame):
    global _SHOULD_STOP
    _SHOULD_STOP = True
    print("[SIGNAL] Received termination signal; will stop after current step.", flush=True)

for _sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(_sig, _handle_sigterm)

# --------------- logging ---------------
def init_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ut.set_log_file(str(out_dir / "run.log"))
    ut.set_debug(True)
    # dump minimal env info
    (out_dir / "env.txt").write_text(
        "\n".join([
            f"python: {sys.version.split()[0]}",
            f"platform: {platform.platform()}",
            f"cwd: {os.getcwd()}",
            f"OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS')}",
            f"MKL_NUM_THREADS={os.getenv('MKL_NUM_THREADS')}",
            f"VECLIB_MAXIMUM_THREADS={os.getenv('VECLIB_MAXIMUM_THREADS')}",
            f"NUMEXPR_NUM_THREADS={os.getenv('NUMEXPR_NUM_THREADS')}",
        ]) + "\n"
    )

# ---------------- H5AD patching for anndata 0.7.x ----------------
def _b2s(x):
    import numpy as _np
    return x.decode("utf-8") if isinstance(x, (bytes, _np.bytes_)) else x

def _bytes_to_str_arr(a):
    import numpy as _np
    return _np.array([_b2s(v) for v in a], dtype=object)

def _flatten_categoricals(table_group):
    """Turn pandas-categorical groups into plain utf-8 string datasets."""
    to_convert = []
    for k, v in list(table_group.items()):
        if isinstance(v, h5py.Group) and "codes" in v and "categories" in v:
            to_convert.append(k)
    for k in to_convert:
        g = table_group[k]
        codes = g["codes"][()]
        cats  = _bytes_to_str_arr(g["categories"][()])
        out   = np.empty(codes.shape[0], dtype=object)
        mask  = codes >= 0
        out[mask]  = cats[codes[mask]]
        out[~mask] = ""
        del table_group[k]
        dt = h5py.string_dtype(encoding="utf-8")
        table_group.create_dataset(k, data=out.astype(dt), dtype=dt)

def _purge_dict_groups(h5, path="/"):
    """Recursively delete any group with attrs['encoding-type']=='dict'."""
    grp = h5[path]
    for name in list(grp.keys()):
        obj = grp[name]
        if isinstance(obj, h5py.Group):
            enc = _b2s(obj.attrs.get("encoding-type", None))
            if enc == "dict":
                del grp[name]
            else:
                _purge_dict_groups(h5, obj.name)

def fix_h5ad_for_anndata07(base_path: Path):
    # backup once
    bak = base_path.with_suffix(base_path.suffix + ".bak")
    if not bak.exists():
        import shutil
        shutil.copy2(base_path, bak)
        print(f"[fix] Backup -> {bak}")
    print(f"[fix] Patching {base_path}")
    with h5py.File(base_path, "a") as f:
        for g in ("layers", "obsp", "varp", "obsm", "varm"):
            if g in f:
                print(f" - delete /{g}")
                del f[g]
        if "raw" in f and isinstance(f["raw"], h5py.Group) and "layers" in f["raw"]:
            print(" - delete /raw/layers")
            del f["raw"]["layers"]
        _purge_dict_groups(f, "/")
        if "obs" in f: _flatten_categoricals(f["obs"])
        if "var" in f: _flatten_categoricals(f["var"])
    print("[fix] Done.")

# ---------------- Loading base + attaching layers ----------------
def load_converted_base_then_layers(root_dir: Path) -> ad.AnnData:
    root = Path(root_dir)
    base = root / "base_legacy.h5ad"
    if not base.exists():
        raise FileNotFoundError(f"Missing {base}")
    fix_h5ad_for_anndata07(base)

    adata = ad.read_h5ad(base)

    mf_path = root / "layers_manifest.json"
    if not mf_path.exists():
        raise FileNotFoundError(f"Missing manifest: {mf_path}")
    with open(mf_path) as f:
        mf = json.load(f)

    for layer in mf["layers"]:
        lname = layer["name"]
        npz_path = root / layer["path"]
        M = sparse.load_npz(npz_path).tocsr()
        if M.shape != adata.shape:
            raise ValueError(f"Layer {lname} shape {M.shape} != {adata.shape}")
        adata.layers[lname] = M
    return adata

# ---------------- Optuna (v2.10.1) search ----------------
def run_optuna_search(X_counts, gene_names, batches, norm_layer,
                      k_search: int, n_trials: int, seed: int,
                      batch_key: str, layer_name: str,
                      work_dir: Path, timeout_hours: int, n_jobs: int):
    import optuna
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)

    def objective(trial: "optuna.Trial"):
        hidden_size = [
            trial.suggest_categorical("h1", [32, 64, 128]),
            trial.suggest_categorical("h2", [16, 32, 64]),
            trial.suggest_categorical("h3", [32, 64, 128]),
        ]
        # lighter search to save time
        epochs     = trial.suggest_categorical("epochs", [30, 60, 90])
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        n_hvg      = trial.suggest_categorical("n_hvg", [800, 1000])

        summary, _ = cv_dca_5fold(
            X_counts, k=k_search,
            dca_params=dict(hidden_size=hidden_size, epochs=epochs, batch_size=batch_size),
            n_hvg=int(n_hvg), R=1,
            mask_frac=0.10, thinning_p=0.10, random_state=seed,
            hvg_mode="seurat_v3",
            gene_names=gene_names, batches=batches,
            norm_layer=norm_layer, batch_key=batch_key,
            seurat_layer_name=layer_name,
            save_dir=None
        )
        value = float(summary["NB_ll_zero"].mean())
        trial.report(value, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if _SHOULD_STOP:
            raise RuntimeError("Received stop signal")
        return value

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def _dump(study: "optuna.Study", trial):
        df = study.trials_dataframe()
        df.to_csv(work_dir / "optuna_trials.csv", index=False)
        with open(work_dir / "optuna_best.json", "w") as f:
            json.dump(dict(value=study.best_value, params=study.best_params), f, indent=2)

    study.optimize(
        objective,
        n_trials=n_trials,
        gc_after_trial=True,
        timeout=int(timeout_hours * 3600),
        callbacks=[_dump],
        n_jobs=max(1, int(n_jobs))
    )

    print("[HPO] Best value:", study.best_value)
    print("[HPO] Best params:", study.best_params)
    _dump(study, None)
    return study.best_params

# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--neu-root", type=str, required=True,
                   help="Path to converted/GSE169569 (or your dataset) directory")
    p.add_argument("--cov-root", type=str, default=None,
                   help="Optional second dataset root (unused)")
    p.add_argument("--k-search", type=int, default=2)
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", type=str, default="results/dca_full")
    p.add_argument("--batch-key", type=str, default="BioProject")
    p.add_argument("--seurat-layer", type=str, default="log2_1p_CPM_original")
    p.add_argument("--final-k", type=int, default=5)
    p.add_argument("--final-epochs", type=int, default=300)
    p.add_argument("--final-R", type=int, default=3)
    p.add_argument("--timeout-hours", type=int, default=12,
                   help="Wall-clock timeout for HPO")
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallel trials for Optuna (watch memory!)")
    args = p.parse_args()

    out_dir = Path(args.out)
    init_logging(out_dir)

    # reproducibility-ish
    np.random.seed(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

    # --- Load dataset (neu)
    adata = load_converted_base_then_layers(Path(args.neu_root))
    print(adata)
    print("Layers:", list(adata.layers.keys()))

    # --- Prepare matrices (dense + proper dtypes)
    X_counts = adata.layers["counts"]
    norm_layer = adata.layers[args.seurat_layer]
    if sparse.issparse(X_counts):   X_counts = X_counts.A
    if sparse.issparse(norm_layer): norm_layer = norm_layer.A
    X_counts  = np.rint(X_counts).astype(np.int64, copy=False)       # counts must be int
    norm_layer = np.asarray(norm_layer, dtype=np.float32, order="C") # normalized float

    gene_names = np.array(adata.var_names, dtype=str)
    batches    = np.array(adata.obs[args.batch_key], dtype=str)

    # --- HPO (Optuna 2.10.1 recommended on this legacy env)
    try:
        best = run_optuna_search(
            X_counts, gene_names, batches, norm_layer,
            k_search=args.k_search, n_trials=args.trials, seed=args.seed,
            batch_key=args.batch_key, layer_name=args.seurat_layer,
            work_dir=out_dir, timeout_hours=args.timeout_hours, n_jobs=args.jobs
        )
    except Exception as e:
        print("[HPO] Optuna failed or not available; using defaults.")
        print("[HPO] Error:", repr(e))
        best = dict(h1=64, h2=32, h3=64, batch_size=64, n_hvg=1000)

    # --- Final retrain (fuller settings)
    hidden_size = [best.get("h1", 64), best.get("h2", 32), best.get("h3", 64)]
    final_n_hvg = int(best.get("n_hvg", 1000))
    final_bs    = int(best.get("batch_size", 64))

    final_params = dict(hidden_size=hidden_size,
                        n_hvg=final_n_hvg,
                        batch_size=final_bs,
                        epochs=args.final_epochs,
                        k=args.final_k,
                        R=args.final_R)
    with open(out_dir / "final_params.json", "w") as f:
        json.dump(final_params, f, indent=2)

    print("[FINAL]", final_params)

    summary, details = cv_dca_5fold(
        X_counts, k=args.final_k,
        dca_params=dict(hidden_size=hidden_size,
                        epochs=args.final_epochs,
                        batch_size=final_bs),
        n_hvg=final_n_hvg, R=args.final_R,
        mask_frac=0.10, thinning_p=0.10, random_state=args.seed,
        hvg_mode="seurat_v3",
        gene_names=gene_names, batches=batches,
        norm_layer=norm_layer, batch_key=args.batch_key,
        seurat_layer_name=args.seurat_layer,
        save_dir=str(out_dir)
    )

    summary.to_csv(out_dir / "dca_summary.csv", index=False)
    details.to_csv(out_dir / "dca_detailed.csv", index=False)
    print("[DONE] Wrote results to:", out_dir)

if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    finally:
        print(f"[TIME] Elapsed: {(time.time()-t0)/3600:.2f} h")
