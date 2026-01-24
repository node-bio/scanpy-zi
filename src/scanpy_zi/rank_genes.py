"""Zero-inflated decomposition for differential expression analysis.

This module implements the ZI decomposition method for identifying differentially
expressed genes in single-cell RNA-seq data. Unlike traditional methods that
conflate changes in the fraction of expressing cells (penetrance) with changes
in expression level among expressors (amplitude), ZI decomposition separates
these two distinct biological effects.

The method computes:
- Penetrance (Δπ): Log-odds ratio comparing fraction expressing between groups
- Amplitude (Δμ): Mean log2 expression difference among expressors (OLS)
- Combined Wald test accounting for correlation between estimators
- Joint z-score combining both effects

References
----------
This approach is inspired by:
- Townes et al. (2019) - Feature selection and dimension reduction for scRNA-seq
- Sarkar & Stephens (2021) - Separating measurement and expression models
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
from anndata import AnnData
from numpy.typing import NDArray
from statsmodels.stats.multitest import multipletests

from ._sparse_utils import sparse_expr_stats, sparse_influence_covariance


def rank_genes_groups_zi(
    adata: AnnData,
    groupby: str,
    groups: str | Sequence[str] | None = None,
    reference: str = "rest",
    *,
    use_raw: bool = True,
    expression_threshold: float = 0.0,
    min_cells: int = 10,
    confidence_level: float = 0.95,
    key_added: str = "rank_genes_groups_zi",
    copy: bool = False,
) -> pd.DataFrame | None:
    """Rank genes for characterizing groups using zero-inflated decomposition.

    Decomposes differential expression into penetrance (fraction expressing)
    and amplitude (expression level among expressors) effects. This is
    particularly valuable for single-cell data where zero-inflation is common.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        The key of the observations grouping to consider.
    groups
        Subset of groups to compare. If None, compares the first group.
        For two-group comparison, specify a single group name.
    reference
        Reference group for comparison. Use "rest" to compare against all
        other cells, or specify a group name.
    use_raw
        Use raw attribute of adata if present.
    expression_threshold
        Threshold above which a cell is considered expressing a gene.
        Default 0.0 works well for log-normalized data.
    min_cells
        Minimum number of expressing cells required in each group to compute
        penetrance statistics. For genes with fewer than min_cells expressors
        in either group, penetrance values are set to NaN and p-values to 1.0.
    confidence_level
        Confidence level for intervals (default 0.95 for 95% CI).
    key_added
        Key under which results are stored in adata.uns.
    copy
        Whether to return a copy of results instead of modifying adata.

    Returns
    -------
    If copy=True, returns DataFrame with results. Otherwise, stores results
    in adata.uns[key_added] and returns None.

    The results DataFrame contains:

    Penetrance columns:
    - delta_pi: Penetrance effect (log-odds ratio)
    - se_delta_pi: Standard error of penetrance
    - delta_pi_ci_lower: CI lower bound for penetrance
    - delta_pi_ci_upper: CI upper bound for penetrance
    - z_pi: Z-score for penetrance
    - pvals_pi: P-value for penetrance
    - pvals_pi_adj: FDR-adjusted p-value for penetrance

    Amplitude columns:
    - delta_mu: Amplitude effect (log2 FC among expressors)
    - se_delta_mu: Standard error of amplitude
    - delta_mu_ci_lower: CI lower bound for amplitude
    - delta_mu_ci_upper: CI upper bound for amplitude
    - z_mu: Z-score for amplitude
    - pvals_mu: P-value for amplitude
    - pvals_mu_adj: FDR-adjusted p-value for amplitude

    Joint test columns:
    - wald_statistic: Combined Wald statistic (χ²)
    - combined_z: Sum of z-scores (z_pi + z_mu), estimates combined standardized effect
    - se_combined_z: Standard error of combined_z = sqrt(2 * (1 + rho))
    - combined_z_ci_lower: CI lower bound for combined z-score
    - combined_z_ci_upper: CI upper bound for combined z-score
    - pvals: Combined p-value from multivariate Wald test
    - pvals_adj: Benjamini-Hochberg FDR
    - rho_estimated: Estimated correlation between estimators

    Compatibility columns:
    - names: Gene names
    - logfoldchanges: Amplitude effect (Δμ) for scanpy compatibility
    - pct_nz_group: Fraction expressing in test group
    - pct_nz_reference: Fraction expressing in reference

    Examples
    --------
    >>> import scanpy as sc
    >>> import scanpy_zi
    >>> adata = sc.datasets.pbmc3k_processed()
    >>> results = scanpy_zi.rank_genes_groups_zi(
    ...     adata,
    ...     groupby="louvain",
    ...     groups=["CD4 T cells"],
    ...     reference="rest",
    ...     copy=True,
    ... )
    >>> # View top genes by joint z-score
    >>> print(results.nsmallest(10, "pvals")[["names", "delta_pi", "delta_mu", "pvals_adj"]])
    """
    if copy:
        adata = adata.copy()

    # Validate inputs
    if groupby not in adata.obs.columns:
        raise ValueError(f"groupby '{groupby}' not found in adata.obs")

    # Determine test groups
    all_groups = adata.obs[groupby].unique()
    if groups is None:
        test_groups = [str(all_groups[0])]
    elif isinstance(groups, str):
        test_groups = [groups]
    else:
        test_groups = [str(g) for g in groups]

    # Validate groups exist
    for g in test_groups:
        if g not in all_groups:
            raise ValueError(f"Group '{g}' not found in adata.obs['{groupby}']")

    if reference != "rest" and reference not in all_groups:
        raise ValueError(f"Reference '{reference}' not found in adata.obs['{groupby}']")

    # Get expression matrix
    if use_raw and adata.raw is not None:
        X = adata.raw.X
        gene_names = adata.raw.var_names.tolist()
    else:
        X = adata.X
        gene_names = adata.var_names.tolist()

    n_genes = len(gene_names)

    z_alpha = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    # Process all groups and collect results
    all_results = []
    uns_data: dict = {
        "params": {
            "groupby": groupby,
            "reference": reference,
            "method": "zi_decomposition",
            "use_raw": use_raw,
            "confidence_level": confidence_level,
        },
    }

    for group_a in test_groups:
        group_b = reference

        # Get cell masks
        mask_a = (adata.obs[groupby] == group_a).values
        if group_b == "rest":
            mask_b = ~mask_a
        else:
            mask_b = (adata.obs[groupby] == group_b).values

        n_cells_a = mask_a.sum()
        n_cells_b = mask_b.sum()

        if n_cells_a < 3:
            raise ValueError(f"Group '{group_a}' has fewer than 3 cells ({n_cells_a})")
        if n_cells_b < 3:
            raise ValueError(f"Reference '{group_b}' has fewer than 3 cells ({n_cells_b})")

        # Extract submatrices
        X_a = X[mask_a, :]
        X_b = X[mask_b, :]

        threshold = expression_threshold

        # Convert to CSC sparse format for efficient column-wise operations
        # This is optimal for zero-inflated scRNA-seq data
        if sp.issparse(X_a):
            X_a_csc = X_a.tocsc()
            X_b_csc = X_b.tocsc()
        else:
            X_a_csc = sp.csc_matrix(X_a, dtype=np.float32)
            X_b_csc = sp.csc_matrix(X_b, dtype=np.float32)

        # Compute number of expressors per gene
        n_expressors_a = np.asarray((X_a_csc > threshold).sum(axis=0)).flatten()
        n_expressors_b = np.asarray((X_b_csc > threshold).sum(axis=0)).flatten()

        n_expressors_a = np.asarray(n_expressors_a, dtype=np.float32)
        n_expressors_b = np.asarray(n_expressors_b, dtype=np.float32)

        # =====================================================================
        # PENETRANCE: Log-odds ratio from 2x2 contingency table
        # =====================================================================
        a = n_expressors_a
        b = n_cells_a - n_expressors_a
        c = n_expressors_b
        d = n_cells_b - n_expressors_b

        # Haldane correction for zero counts
        needs_correction = (a == 0) | (b == 0) | (c == 0) | (d == 0)
        a_adj = np.where(needs_correction, a + 0.5, a)
        b_adj = np.where(needs_correction, b + 0.5, b)
        c_adj = np.where(needs_correction, c + 0.5, c)
        d_adj = np.where(needs_correction, d + 0.5, d)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_odds_ratio = np.log(a_adj * d_adj / (b_adj * c_adj))
            se_log_odds = np.sqrt(1 / a_adj + 1 / b_adj + 1 / c_adj + 1 / d_adj)

        delta_pi = log_odds_ratio
        se_delta_pi = se_log_odds

        theta_a = n_expressors_a / n_cells_a
        theta_b = n_expressors_b / n_cells_b

        with np.errstate(divide="ignore", invalid="ignore"):
            z_pi = delta_pi / se_delta_pi
        z_pi = np.where(np.isfinite(z_pi), z_pi, 0.0)
        pvals_pi = 2 * (1 - stats.norm.cdf(np.abs(z_pi)))

        # Apply min_cells filter for penetrance
        penetrance_insufficient = (n_expressors_a < min_cells) | (n_expressors_b < min_cells)
        delta_pi[penetrance_insufficient] = np.nan
        se_delta_pi[penetrance_insufficient] = np.inf
        z_pi[penetrance_insufficient] = 0.0
        pvals_pi[penetrance_insufficient] = 1.0

        # =====================================================================
        # AMPLITUDE: OLS regression among expressors
        # =====================================================================
        MIN_CELLS_FOR_TEST = 3

        n_expr_a = n_expressors_a.astype(np.float32)
        n_expr_b = n_expressors_b.astype(np.float32)
        n_expr_total = n_expr_a + n_expr_b

        insufficient_mask = (n_expr_a < MIN_CELLS_FOR_TEST) | (n_expr_b < MIN_CELLS_FOR_TEST)

        # Compute mean and SS residuals using sparse-optimized functions
        mean_a_vec, ss_resid_a, _ = sparse_expr_stats(X_a_csc, threshold)
        mean_b_vec, ss_resid_b, _ = sparse_expr_stats(X_b_csc, threshold)

        ss_resid = ss_resid_a + ss_resid_b
        delta_mu = mean_a_vec - mean_b_vec

        df_resid = n_expr_total - 2
        df_resid = np.maximum(df_resid, 1)
        mse = ss_resid / df_resid

        with np.errstate(divide="ignore", invalid="ignore"):
            se_delta_mu = np.sqrt(mse * (1 / n_expr_a + 1 / n_expr_b))

        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = delta_mu / se_delta_mu
        t_stat = np.where(np.isfinite(t_stat), t_stat, 0.0)
        pvals_mu = 2 * (1 - stats.t.cdf(np.abs(t_stat), df_resid))

        delta_mu[insufficient_mask] = np.nan
        se_delta_mu[insufficient_mask] = np.inf
        pvals_mu[insufficient_mask] = 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            z_mu = delta_mu / se_delta_mu
        z_mu = np.where(np.isfinite(z_mu), z_mu, 0.0)

        # =====================================================================
        # COMBINED: Multivariate Wald test with covariance
        # =====================================================================
        # Compute influence function covariance using sparse-optimized functions
        cov_a = sparse_influence_covariance(
            X_a_csc, theta_a, mean_a_vec, n_cells_a, n_expr_a, threshold
        )
        cov_b = sparse_influence_covariance(
            X_b_csc, theta_b, mean_b_vec, n_cells_b, n_expr_b, threshold
        )
        cov_pi_mu = cov_a + cov_b

        with np.errstate(divide="ignore", invalid="ignore"):
            rho_estimated = cov_pi_mu / (se_delta_pi * se_delta_mu + 1e-6)
        rho_estimated = np.clip(rho_estimated, -0.99, 0.99)
        rho_estimated = np.where(np.isfinite(rho_estimated), rho_estimated, 0.0)

        var_pi = se_delta_pi**2
        var_mu = se_delta_mu**2
        cov_term = rho_estimated * se_delta_pi * se_delta_mu

        det = var_pi * var_mu - cov_term**2
        det = np.maximum(det, 1e-20)

        with np.errstate(divide="ignore", invalid="ignore"):
            wald_statistic = (
                delta_pi**2 * var_mu - 2 * delta_pi * delta_mu * cov_term + delta_mu**2 * var_pi
            ) / det

        has_pi = np.isfinite(z_pi) & (se_delta_pi > 0) & (se_delta_pi < np.inf)
        has_mu = np.isfinite(z_mu) & (se_delta_mu > 0) & (se_delta_mu < np.inf)
        both_valid = has_pi & has_mu

        pi_only = has_pi & ~has_mu
        mu_only = ~has_pi & has_mu

        wald_statistic[pi_only] = z_pi[pi_only] ** 2
        wald_statistic[mu_only] = z_mu[mu_only] ** 2
        wald_statistic[~(has_pi | has_mu)] = np.nan

        pvals_combined = np.full(n_genes, 1.0)
        pvals_combined[both_valid] = 1 - stats.chi2.cdf(wald_statistic[both_valid], df=2)
        pvals_combined[pi_only] = 1 - stats.chi2.cdf(wald_statistic[pi_only], df=1)
        pvals_combined[mu_only] = 1 - stats.chi2.cdf(wald_statistic[mu_only], df=1)
        pvals_combined = np.clip(pvals_combined, 0, 1)

        # =====================================================================
        # COMBINED Z-SCORE
        # =====================================================================
        # Combined z-score is the sum of individual z-scores: combined_z = z_pi + z_mu
        # This estimates the true sum of standardized effects.
        # Its standard error is sqrt(Var(z_pi) + Var(z_mu) + 2*Cov) = sqrt(2*(1+rho))
        combined_z = z_pi + z_mu
        se_combined_z = np.sqrt(2 * (1 + rho_estimated))
        se_combined_z = np.where(se_combined_z > 0, se_combined_z, 1.0)  # Avoid zero SE

        # Handle cases where only one component is valid
        combined_z[pi_only] = z_pi[pi_only]
        combined_z[mu_only] = z_mu[mu_only]
        combined_z[~(has_pi | has_mu)] = np.nan

        # SE for single-component cases
        se_combined_z[pi_only] = 1.0  # z_pi alone has SE = 1
        se_combined_z[mu_only] = 1.0  # z_mu alone has SE = 1

        # =====================================================================
        # CONFIDENCE INTERVALS
        # =====================================================================
        # Normal-based CI for individual effects
        delta_pi_ci_lower = delta_pi - z_alpha * se_delta_pi
        delta_pi_ci_upper = delta_pi + z_alpha * se_delta_pi
        delta_mu_ci_lower = delta_mu - z_alpha * se_delta_mu
        delta_mu_ci_upper = delta_mu + z_alpha * se_delta_mu

        # Combined z CI: proper CI for sum of standardized effects
        combined_z_ci_lower = combined_z - z_alpha * se_combined_z
        combined_z_ci_upper = combined_z + z_alpha * se_combined_z

        # =====================================================================
        # FDR CORRECTION
        # =====================================================================
        def _apply_fdr(pvals: NDArray[np.floating]) -> NDArray[np.floating]:
            valid = np.isfinite(pvals) & (pvals >= 0) & (pvals <= 1)
            adj = np.ones(len(pvals))
            if valid.sum() > 0:
                _, adj[valid], _, _ = multipletests(pvals[valid], method="fdr_bh")
            return adj

        pvals_adj = _apply_fdr(pvals_combined)
        pvals_pi_adj = _apply_fdr(pvals_pi)
        pvals_mu_adj = _apply_fdr(pvals_mu)

        # =====================================================================
        # BUILD RESULTS
        # =====================================================================
        result_df = pd.DataFrame(
            {
                "group": group_a,
                "names": gene_names,
                # Penetrance
                "delta_pi": delta_pi,
                "se_delta_pi": se_delta_pi,
                "delta_pi_ci_lower": delta_pi_ci_lower,
                "delta_pi_ci_upper": delta_pi_ci_upper,
                "z_pi": z_pi,
                "pvals_pi": pvals_pi,
                "pvals_pi_adj": pvals_pi_adj,
                # Amplitude
                "delta_mu": delta_mu,
                "se_delta_mu": se_delta_mu,
                "delta_mu_ci_lower": delta_mu_ci_lower,
                "delta_mu_ci_upper": delta_mu_ci_upper,
                "z_mu": z_mu,
                "pvals_mu": pvals_mu,
                "pvals_mu_adj": pvals_mu_adj,
                # Joint
                "wald_statistic": wald_statistic,
                "combined_z": combined_z,
                "se_combined_z": se_combined_z,
                "combined_z_ci_lower": combined_z_ci_lower,
                "combined_z_ci_upper": combined_z_ci_upper,
                "pvals": pvals_combined,
                "pvals_adj": pvals_adj,
                "rho_estimated": rho_estimated,
                # Compatibility
                "logfoldchanges": delta_mu,
                "pct_nz_group": theta_a,
                "pct_nz_reference": theta_b,
            }
        )

        result_df["pvals_adj"] = result_df["pvals_adj"].clip(0.0, 1.0)
        result_df = result_df.sort_values("pvals", ascending=True)

        # Collect results for this group
        all_results.append(result_df)

        # Store per-group data in uns_data (scanpy-compatible format)
        def _add_to_uns(key: str, values, grp: str = group_a):
            if key not in uns_data:
                uns_data[key] = {}
            uns_data[key][grp] = values

        _add_to_uns("names", result_df["names"].values)
        _add_to_uns("scores", result_df["wald_statistic"].values)
        _add_to_uns("pvals", result_df["pvals"].values)
        _add_to_uns("pvals_adj", result_df["pvals_adj"].values)
        _add_to_uns("logfoldchanges", result_df["logfoldchanges"].values)
        _add_to_uns("pts", result_df["pct_nz_group"].values)
        _add_to_uns("pts_rest", result_df["pct_nz_reference"].values)
        _add_to_uns("delta_pi", result_df["delta_pi"].values)
        _add_to_uns("se_delta_pi", result_df["se_delta_pi"].values)
        _add_to_uns("delta_pi_ci_lower", result_df["delta_pi_ci_lower"].values)
        _add_to_uns("delta_pi_ci_upper", result_df["delta_pi_ci_upper"].values)
        _add_to_uns("z_pi", result_df["z_pi"].values)
        _add_to_uns("pvals_pi", result_df["pvals_pi"].values)
        _add_to_uns("pvals_pi_adj", result_df["pvals_pi_adj"].values)
        _add_to_uns("delta_mu", result_df["delta_mu"].values)
        _add_to_uns("se_delta_mu", result_df["se_delta_mu"].values)
        _add_to_uns("delta_mu_ci_lower", result_df["delta_mu_ci_lower"].values)
        _add_to_uns("delta_mu_ci_upper", result_df["delta_mu_ci_upper"].values)
        _add_to_uns("z_mu", result_df["z_mu"].values)
        _add_to_uns("pvals_mu", result_df["pvals_mu"].values)
        _add_to_uns("pvals_mu_adj", result_df["pvals_mu_adj"].values)
        _add_to_uns("wald_statistic", result_df["wald_statistic"].values)
        _add_to_uns("combined_z", result_df["combined_z"].values)
        _add_to_uns("se_combined_z", result_df["se_combined_z"].values)
        _add_to_uns("combined_z_ci_lower", result_df["combined_z_ci_lower"].values)
        _add_to_uns("combined_z_ci_upper", result_df["combined_z_ci_upper"].values)
        _add_to_uns("rho_estimated", result_df["rho_estimated"].values)

    # Store results in adata.uns
    adata.uns[key_added] = uns_data

    # Concatenate all group results
    combined_df = pd.concat(all_results, ignore_index=True)

    if copy:
        return combined_df
    return None
