"""Per-group sufficient statistics computed in one pass over the expression matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

# Target number of stored values processed per chunk. Bounds temporary memory
# (a few float64/int64 arrays of this length) independent of matrix size.
_CHUNK_NNZ = 10_000_000


@dataclass
class GroupStats:
    """Sufficient statistics per group code (rows) and gene (columns).

    Only values strictly above the expression threshold contribute.
    """

    n_cells: NDArray[np.float64]  # (n_codes,)
    n_expr: NDArray[np.float64]  # (n_codes, n_genes) count of expressing cells
    sum_expr: NDArray[np.float64]  # (n_codes, n_genes) sum of values among expressors
    sumsq_expr: NDArray[np.float64]  # (n_codes, n_genes) sum of squared values among expressors


def group_expr_stats(X, codes: NDArray[np.integer], n_codes: int, threshold: float) -> GroupStats:
    """Aggregate expressor counts, sums and sums of squares per group and gene.

    Runs a single pass over the stored values of ``X`` (never densifies a
    sparse matrix), accumulating with ``np.bincount`` in float64.

    Parameters
    ----------
    X
        Cells x genes matrix, sparse or dense. Values are used as given.
    codes
        Group code for each cell, in ``[0, n_codes)``.
    n_codes
        Number of distinct codes.
    threshold
        A value counts as expressing when it is strictly greater than this.
        Must be >= 0 so that implicit zeros are never expressing.
    """
    n_cells, n_genes = X.shape
    codes = np.asarray(codes, dtype=np.int64)
    size = n_codes * n_genes

    n_expr = np.zeros(size, dtype=np.float64)
    sum_expr = np.zeros(size, dtype=np.float64)
    sumsq_expr = np.zeros(size, dtype=np.float64)

    def _accumulate(row_codes, cols, vals):
        keep = vals > threshold
        vals = vals[keep].astype(np.float64)
        idx = row_codes[keep] * n_genes + cols[keep]
        n_expr[:] += np.bincount(idx, minlength=size)
        sum_expr[:] += np.bincount(idx, weights=vals, minlength=size)
        sumsq_expr[:] += np.bincount(idx, weights=vals * vals, minlength=size)

    if sp.issparse(X):
        X = sp.csr_matrix(X)  # no copy if already CSR
        if not X.has_canonical_format:
            X = X.copy()
            X.sum_duplicates()
        indptr = X.indptr
        avg_nnz = max(X.nnz / max(n_cells, 1), 1.0)
        step = max(int(_CHUNK_NNZ / avg_nnz), 1)
        for start in range(0, n_cells, step):
            end = min(start + step, n_cells)
            lo, hi = indptr[start], indptr[end]
            row_codes = np.repeat(codes[start:end], np.diff(indptr[start : end + 1]))
            _accumulate(row_codes, X.indices[lo:hi].astype(np.int64), X.data[lo:hi])
    else:
        step = max(int(_CHUNK_NNZ / max(n_genes, 1)), 1)
        for start in range(0, n_cells, step):
            block = np.asarray(X[start : start + step])
            rows, cols = np.nonzero(block > threshold)
            _accumulate(codes[start + rows], cols.astype(np.int64), block[rows, cols])

    shape = (n_codes, n_genes)
    return GroupStats(
        n_cells=np.bincount(codes, minlength=n_codes).astype(np.float64),
        n_expr=n_expr.reshape(shape),
        sum_expr=sum_expr.reshape(shape),
        sumsq_expr=sumsq_expr.reshape(shape),
    )


def benjamini_hochberg(pvals: NDArray[np.floating]) -> NDArray[np.float64]:
    """Benjamini-Hochberg adjusted p-values. Non-finite inputs are returned as 1."""
    pvals = np.asarray(pvals, dtype=np.float64)
    adj = np.ones_like(pvals)
    valid = np.isfinite(pvals)
    p = pvals[valid]
    m = p.size
    if m == 0:
        return adj
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0.0, 1.0)
    adj[valid] = out
    return adj
