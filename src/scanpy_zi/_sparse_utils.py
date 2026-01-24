"""Sparse-native utility functions for memory-efficient computation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sparse_expr_stats(
    X_csc, threshold: float
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Compute mean and SS residuals directly from CSC sparse matrix.

    This function processes sparse matrices column-by-column without
    materializing dense arrays, enabling memory-efficient computation
    on large single-cell datasets.

    Expects input to already be normalized and log-transformed.

    Parameters
    ----------
    X_csc
        Sparse matrix in CSC format (cells x genes). Should contain
        normalized, log-transformed expression values.
    threshold
        Expression threshold above which a cell is considered expressing.

    Returns
    -------
    mean_expr
        Mean expression for x > threshold, per gene.
    ss_resid
        Sum of squared residuals from mean, per gene.
    n_expr
        Count of expressors per gene.
    """
    n_genes = X_csc.shape[1]
    mean_expr = np.zeros(n_genes, dtype=np.float32)
    ss_resid = np.zeros(n_genes, dtype=np.float32)
    n_expr = np.zeros(n_genes, dtype=np.float32)

    for j in range(n_genes):
        start, end = X_csc.indptr[j], X_csc.indptr[j + 1]
        col_data = X_csc.data[start:end]

        # Filter by threshold
        expr_mask = col_data > threshold
        expr_vals = col_data[expr_mask].astype(np.float32)
        n = len(expr_vals)
        n_expr[j] = n

        if n > 0:
            mean_expr[j] = np.mean(expr_vals)
            if n > 1:
                ss_resid[j] = np.sum((expr_vals - mean_expr[j]) ** 2)

    return mean_expr, ss_resid, n_expr


def sparse_influence_covariance(
    X_csc,
    theta: NDArray[np.float32],
    mean_expr: NDArray[np.float32],
    n_cells: int,
    n_expr: NDArray[np.float32],
    threshold: float,
) -> NDArray[np.float32]:
    """Compute influence function covariance from CSC sparse matrix.

    For each cell i and gene j:
        ψ_π[i,j] = (express[i,j] - θ[j]) / (θ[j]*(1-θ[j])) / n_cells
        ψ_μ[i,j] = (expr[i,j] - mean[j]) / n_expr[j] for expressors, 0 otherwise

    Expects input to already be normalized and log-transformed.

    Parameters
    ----------
    X_csc
        Sparse matrix in CSC format (cells x genes). Should contain
        normalized, log-transformed expression values.
    theta
        Fraction of expressing cells per gene.
    mean_expr
        Mean expression among expressors per gene.
    n_cells
        Total number of cells.
    n_expr
        Number of expressing cells per gene.
    threshold
        Expression threshold.

    Returns
    -------
    cov_pi_mu
        Covariance = sum_i(ψ_π * ψ_μ) per gene.
    """
    n_genes = X_csc.shape[1]
    cov_pi_mu = np.zeros(n_genes, dtype=np.float32)

    for j in range(n_genes):
        if theta[j] <= 0 or theta[j] >= 1 or n_expr[j] < 1:
            continue

        inv_var = 1.0 / (theta[j] * (1 - theta[j]) + 1e-10)
        start, end = X_csc.indptr[j], X_csc.indptr[j + 1]
        col_data = X_csc.data[start:end]

        # For expressors: both ψ_π and ψ_μ are non-zero
        expr_mask = col_data > threshold
        expr_vals = col_data[expr_mask].astype(np.float32)
        n = len(expr_vals)

        if n > 0:
            # ψ_π for expressors: (1 - θ) * inv_var / n_cells
            psi_pi_expr = (1 - theta[j]) * inv_var / n_cells
            # ψ_μ for expressors: (expr_val - mean) / n_expr
            psi_mu_expr = (expr_vals - mean_expr[j]) / (n_expr[j] + 1e-10)
            cov_pi_mu[j] = np.sum(psi_pi_expr * psi_mu_expr) * n_cells

        # Non-expressors contribute to ψ_π but ψ_μ = 0, so no contribution to cov

    return cov_pi_mu
