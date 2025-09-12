import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt




# ---------- Utilities ----------
def _to_csr(M):
    """Return a CSR matrix (convert if dense or other sparse format)."""
    return M.tocsr() if sp.issparse(M) else sp.csr_matrix(M)

def make_masks(
    adata,
    layer="counts",
    mode="uniform",            # "uniform" or "expr_weighted"
    frac=0.10,                 # float or (low, high) for per-sample masking rates
    per_sample=True,
    by_batch_key=None,         # e.g., "cohort" to stratify rates by batch (metadata only)
    weight_alpha=1.0,
    rng_seed=7,
    restrict_genes=None,       # boolean array of shape (n_vars,) to limit maskable genes
    return_sparse=True
):
    """rmsep 
    Build a boolean mask (True = will be modified) but only over nonzero entries.
    Implemented to be memory-safe on large sparse count matrices.
    """
    rng = np.random.default_rng(rng_seed)
    X_in = adata.layers[layer] if layer in adata.layers else adata.X
    X = _to_csr(X_in)

    n, g = X.shape
    if isinstance(frac, (list, tuple, np.ndarray)):
        low, high = float(frac[0]), float(frac[1])
        per_sample_frac = rng.uniform(low, high, size=n) if per_sample else np.full(n, rng.uniform(low, high))
    else:
        per_sample_frac = np.full(n, float(frac))

    # Batch groups usable for stratification logic if you extend per-batch behavior later
    groups = (adata.obs[by_batch_key].astype("category").cat.codes.to_numpy()
              if by_batch_key is not None else np.zeros(n, dtype=int))

    feature_mask = np.ones(g, dtype=bool) if restrict_genes is None else restrict_genes.astype(bool)

    # Construct a CSR boolean mask explicitly
    data, indices, indptr = [], [], [0]
    for i in range(n):
        # Nonzero columns in i-th row
        row = X.getrow(i)
        nz_cols = row.indices
        if feature_mask is not None:
            nz_cols = nz_cols[feature_mask[nz_cols]]

        chosen_cols = np.empty(0, dtype=int)
        if nz_cols.size > 0:
            k = max(1, int(np.floor(per_sample_frac[i] * nz_cols.size)))
            k = min(k, nz_cols.size)

            if mode == "uniform":
                chosen_cols = rng.choice(nz_cols, size=k, replace=False)

            elif mode == "expr_weighted":
                # Only materialize the small slice to compute weights
                v = row[:, nz_cols].toarray().ravel().astype(float)
                w = 1.0 / (1.0 + np.log1p(v))
                if weight_alpha != 1.0:
                    w = np.power(w, weight_alpha)
                w_sum = w.sum()
                if w_sum <= 0:
                    chosen_cols = rng.choice(nz_cols, size=k, replace=False)
                else:
                    p = w / w_sum
                    chosen_cols = rng.choice(nz_cols, size=k, replace=False, p=p)
            else:
                raise ValueError("mode must be 'uniform' or 'expr_weighted'")

        # Store True entries for this row
        indices.extend(sorted(chosen_cols))
        data.extend([True] * chosen_cols.size)
        indptr.append(len(indices))

    mask_csr = sp.csr_matrix(
        (np.array(data, dtype=bool), np.array(indices, dtype=int), np.array(indptr, dtype=int)),
        shape=(n, g)
    )
    return mask_csr if return_sparse else mask_csr.toarray().astype(bool)

def apply_mask(adata, mask, layer="counts", to="counts_masked", strategy="zero", thin_p=0.2, rng_seed=7):
    """
    Apply a given mask to produce a masked layer.
    - strategy="zero": set masked nonzero entries to 0
    - strategy="thin": binomial thinning on masked nonzero entries with probability thin_p
    Returns the modified AnnData (in-place layer assignment).
    """
    rng = np.random.default_rng(rng_seed)
    X_in = adata.layers[layer] if layer in adata.layers else adata.X
    X = _to_csr(X_in).copy()

    M = mask if sp.issparse(mask) else sp.csr_matrix(mask)

    X.sort_indices()
    M.sort_indices()

    for i in range(X.shape[0]):
        r0, r1 = X.indptr[i], X.indptr[i + 1]
        cols = X.indices[r0:r1]
        vals = X.data[r0:r1]

        mrow = M.getrow(i)
        if mrow.nnz == 0:
            continue
        mcols = mrow.indices

        # Mask only at the intersection of existing nonzeros
        sel = np.isin(cols, mcols)
        if not sel.any():
            continue

        if strategy == "zero":
            vals[sel] = 0
        elif strategy == "thin":
            k = vals[sel].astype(np.int64)
            vals[sel] = rng.binomial(k, thin_p)
        else:
            raise ValueError('strategy must be "zero" or "thin"')

        X.data[r0:r1] = vals

    X.eliminate_zeros()
    adata.layers[to] = X
    return adata


# ---------- Visualization ----------
def _to_dense_subset(X, n_cells=50, n_genes=50, random_state=0):
    """Convert a random subset of sparse matrix to dense for plotting."""
    rng = np.random.default_rng(random_state)
    cells = rng.choice(X.shape[0], size=min(n_cells, X.shape[0]), replace=False)
    genes = rng.choice(X.shape[1], size=min(n_genes, X.shape[1]), replace=False)
    return X[cells][:, genes].toarray()

def compare_original_masked(adata, orig_layer="counts_orig", masked_layer="counts_masked", n_cells=50, n_genes=50):
    """Plot original vs masked vs difference for a random subset."""
    X0 = adata.layers[orig_layer]
    Xm = adata.layers[masked_layer]
    if sp.issparse(X0): X0 = X0.tocsr()
    if sp.issparse(Xm): Xm = Xm.tocsr()

    sub_orig = _to_dense_subset(X0, n_cells, n_genes)
    sub_mask = _to_dense_subset(Xm, n_cells, n_genes)
    sub_diff = sub_orig - sub_mask  # >0 means masked/reduced

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(sub_orig, aspect="auto", cmap="viridis")
    axes[0].set_title("Original counts")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(sub_mask, aspect="auto", cmap="viridis")
    axes[1].set_title("Masked counts")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(sub_diff != 0, aspect="auto", cmap="Reds")
    axes[2].set_title("Masked positions (diff!=0)")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("Genes")
        ax.set_ylabel("Cells")

    plt.tight_layout()
    plt.show()
