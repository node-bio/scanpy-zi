"""Zero-inflated decomposition for differential expression analysis.

This module implements the ZI decomposition method for identifying differentially
expressed genes in single-cell RNA-seq data. Unlike traditional methods that
conflate changes in the fraction of expressing cells (penetrance) with changes
in expression level among expressors (amplitude), ZI decomposition separates
these two distinct biological effects.

The method computes:
- Penetrance (Δπ): Log-odds ratio comparing fraction expressing between groups
- Amplitude (Δμ): Difference in mean input value among expressors (pooled-variance t-test)
- Combined Wald test of both effects
- Joint z-score combining both effects

Independence of the two estimators
----------------------------------
Within a group, the penetrance estimate depends only on the number of expressing
cells N, and given N the mean among expressors is unbiased for the true mean.
By the law of total covariance their covariance is therefore exactly zero, and
groups are independent samples, so Cov(Δπ̂, Δμ̂) = 0. The Wald statistic is
z_π² + z_μ² with no correlation term.

References
----------
This approach is inspired by:
- Townes et al. (2019) - Feature selection and dimension reduction for scRNA-seq
- Sarkar & Stephens (2021) - Separating measurement and expression models
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
from anndata import AnnData

from ._sparse_utils import benjamini_hochberg, group_expr_stats

# Columns stored per group in adata.uns[key_added], in addition to the
# scanpy-compatible names/scores/pvals/pvals_adj/logfoldchanges/pts/pts_rest.
_UNS_COLUMNS = (
    "delta_pi",
    "se_delta_pi",
    "delta_pi_ci_lower",
    "delta_pi_ci_upper",
    "z_pi",
    "pvals_pi",
    "pvals_pi_adj",
    "delta_mu",
    "se_delta_mu",
    "delta_mu_ci_lower",
    "delta_mu_ci_upper",
    "z_mu",
    "pvals_mu",
    "pvals_mu_adj",
    "wald_statistic",
    "combined_z",
    "se_combined_z",
    "combined_z_ci_lower",
    "combined_z_ci_upper",
)


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

    Input expectations
    ------------------
    The expression values are used exactly as given; no transform is applied.
    Pass normalized, log-transformed data (e.g. ``sc.pp.normalize_total`` then
    ``sc.pp.log1p``). Do not log-transform again before calling this function.
    ``delta_mu`` is in the units of the input: for ``sc.pp.log1p`` output it is
    a difference in natural-log(1 + x) among expressors; divide by ``ln(2)``
    to express it in log2 units.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        The key of the observations grouping to consider.
    groups
        Groups to test. If None, every group except ``reference`` is tested.
    reference
        Reference group for comparison. Use "rest" to compare against all
        other cells, or specify a group name.
    use_raw
        Use raw attribute of adata if present.
    expression_threshold
        A cell expresses a gene when its value is strictly greater than this.
        Must be >= 0. Default 0.0 works well for log-normalized data.
    min_cells
        Minimum number of expressing cells. Amplitude requires at least
        ``min_cells`` expressors in each of the two groups. Penetrance requires
        at least ``min_cells`` expressors in the two groups combined, so genes
        that are on in one group and off in the other are still tested.
        Untestable components get p-value 1.0 and NaN or infinite SE.
    confidence_level
        Confidence level for intervals (default 0.95 for 95% CI).
    key_added
        Key under which results are stored in adata.uns.
    copy
        If True, return the results as a DataFrame and leave ``adata``
        unmodified. If False, store results in ``adata.uns[key_added]``.

    Returns
    -------
    If copy=True, returns DataFrame with results. Otherwise, stores results
    in adata.uns[key_added] and returns None.

    Rows are sorted within each group by ``pvals`` (ties broken by larger
    ``wald_statistic``). The results DataFrame contains:

    Penetrance columns:
    - delta_pi: Penetrance effect (log-odds ratio, natural log)
    - se_delta_pi: Standard error of penetrance
    - delta_pi_ci_lower / delta_pi_ci_upper: CI bounds
    - z_pi: Z-score for penetrance
    - pvals_pi / pvals_pi_adj: P-value and BH-adjusted p-value

    Amplitude columns:
    - delta_mu: Difference in mean input value among expressors (input units)
    - se_delta_mu: Standard error of amplitude (pooled variance)
    - delta_mu_ci_lower / delta_mu_ci_upper: CI bounds
    - z_mu: t-statistic for amplitude
    - pvals_mu / pvals_mu_adj: P-value (t-distribution) and BH-adjusted p-value

    Joint test columns:
    - wald_statistic: z_pi² + z_mu² (χ² with 2 df; 1 df if only one component is testable)
    - combined_z: z_pi + z_mu
    - se_combined_z: sqrt(2) when both components are testable, 1 otherwise
    - combined_z_ci_lower / combined_z_ci_upper: CI bounds for combined_z
    - pvals / pvals_adj: Combined Wald p-value and BH-adjusted p-value

    Compatibility columns:
    - names: Gene names
    - logfoldchanges: Same as delta_mu (input units, not log2 of a mean ratio)
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
    >>> print(results.head(10)[["names", "delta_pi", "delta_mu", "pvals_adj"]])
    """
    # Validate inputs
    if groupby not in adata.obs.columns:
        raise ValueError(f"groupby '{groupby}' not found in adata.obs")
    if expression_threshold < 0:
        raise ValueError("expression_threshold must be >= 0")
    if min_cells < 1:
        raise ValueError("min_cells must be >= 1")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    col = adata.obs[groupby]
    labels = col.astype(str).to_numpy()
    if isinstance(col.dtype, pd.CategoricalDtype):
        all_groups = [str(c) for c in col.cat.categories if (col == c).any()]
    else:
        all_groups = list(pd.unique(labels))

    if reference != "rest" and reference not in all_groups:
        raise ValueError(f"Reference '{reference}' not found in adata.obs['{groupby}']")

    if groups is None:
        test_groups = [g for g in all_groups if g != reference]
    elif isinstance(groups, str):
        test_groups = [groups]
    else:
        test_groups = [str(g) for g in groups]

    for g in test_groups:
        if g not in all_groups:
            raise ValueError(f"Group '{g}' not found in adata.obs['{groupby}']")
        if g == reference:
            raise ValueError(f"Group '{g}' cannot be tested against itself")

    # Get expression matrix
    if use_raw and adata.raw is not None:
        X = adata.raw.X
        gene_names = np.asarray(adata.raw.var_names, dtype=object)
    else:
        X = adata.X
        gene_names = np.asarray(adata.var_names, dtype=object)
    _warn_if_counts(X)

    # One code per group we need, plus a final code for every other cell.
    needed = list(test_groups) + ([reference] if reference != "rest" else [])
    code_of = {g: i for i, g in enumerate(needed)}
    other = len(needed)
    codes = np.array([code_of.get(lab, other) for lab in labels], dtype=np.int64)
    gs = group_expr_stats(X, codes, other + 1, expression_threshold)

    total_cells = gs.n_cells.sum()
    total = (gs.n_expr.sum(axis=0), gs.sum_expr.sum(axis=0), gs.sumsq_expr.sum(axis=0))

    z_alpha = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    uns_data: dict = {
        "params": {
            "groupby": groupby,
            "reference": reference,
            "method": "zi_decomposition",
            "use_raw": use_raw,
            "expression_threshold": expression_threshold,
            "min_cells": min_cells,
            "confidence_level": confidence_level,
        },
    }
    all_results = []

    for group_a in test_groups:
        ia = code_of[group_a]
        n_cells_a = gs.n_cells[ia]
        stats_a = (gs.n_expr[ia], gs.sum_expr[ia], gs.sumsq_expr[ia])
        if reference == "rest":
            n_cells_b = total_cells - n_cells_a
            stats_b = tuple(t - s for t, s in zip(total, stats_a))
        else:
            ib = code_of[reference]
            n_cells_b = gs.n_cells[ib]
            stats_b = (gs.n_expr[ib], gs.sum_expr[ib], gs.sumsq_expr[ib])

        if n_cells_a < 3:
            raise ValueError(f"Group '{group_a}' has fewer than 3 cells ({int(n_cells_a)})")
        if n_cells_b < 3:
            raise ValueError(f"Reference '{reference}' has fewer than 3 cells ({int(n_cells_b)})")

        res = _compare(n_cells_a, stats_a, n_cells_b, stats_b, min_cells, z_alpha)

        result_df = pd.DataFrame({"group": group_a, "names": gene_names, **res})
        result_df["logfoldchanges"] = result_df["delta_mu"]
        order = np.lexsort((-np.nan_to_num(res["wald_statistic"], nan=-1.0), res["pvals"]))
        result_df = result_df.iloc[order].reset_index(drop=True)
        all_results.append(result_df)

        # Per-group arrays in scanpy-compatible layout
        per_group = {
            "names": "names",
            "scores": "wald_statistic",
            "pvals": "pvals",
            "pvals_adj": "pvals_adj",
            "logfoldchanges": "logfoldchanges",
            "pts": "pct_nz_group",
            "pts_rest": "pct_nz_reference",
            **{c: c for c in _UNS_COLUMNS},
        }
        for key, column in per_group.items():
            uns_data.setdefault(key, {})[group_a] = result_df[column].to_numpy()

    combined_df = pd.concat(all_results, ignore_index=True)

    if copy:
        return combined_df
    adata.uns[key_added] = uns_data
    return None


def _compare(n_cells_a, stats_a, n_cells_b, stats_b, min_cells, z_alpha) -> dict:
    """Penetrance, amplitude and combined statistics for one group pair."""
    n_expr_a, sum_a, sumsq_a = stats_a
    n_expr_b, sum_b, sumsq_b = stats_b
    n_genes = n_expr_a.shape[0]

    # =====================================================================
    # PENETRANCE: log-odds ratio from the 2x2 table, Haldane correction
    # =====================================================================
    a, b = n_expr_a, n_cells_a - n_expr_a
    c, d = n_expr_b, n_cells_b - n_expr_b
    corr = np.where((a == 0) | (b == 0) | (c == 0) | (d == 0), 0.5, 0.0)
    a_, b_, c_, d_ = a + corr, b + corr, c + corr, d + corr

    delta_pi = np.log(a_) + np.log(d_) - np.log(b_) - np.log(c_)
    se_delta_pi = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    has_pi = (a + c) >= min_cells

    delta_pi = np.where(has_pi, delta_pi, np.nan)
    se_delta_pi = np.where(has_pi, se_delta_pi, np.inf)
    z_pi = np.where(has_pi, delta_pi / np.where(has_pi, se_delta_pi, 1.0), 0.0)
    pvals_pi = np.where(has_pi, 2 * stats.norm.sf(np.abs(z_pi)), 1.0)

    # =====================================================================
    # AMPLITUDE: pooled-variance t-test on expressors only
    # =====================================================================
    enough = (n_expr_a >= min_cells) & (n_expr_b >= min_cells)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_a = sum_a / n_expr_a
        mean_b = sum_b / n_expr_b
        ss_a = _sum_sq_dev(n_expr_a, sum_a, sumsq_a)
        ss_b = _sum_sq_dev(n_expr_b, sum_b, sumsq_b)
        df_resid = n_expr_a + n_expr_b - 2
        mse = (ss_a + ss_b) / df_resid
        se_delta_mu = np.sqrt(mse * (1 / n_expr_a + 1 / n_expr_b))
    delta_mu = np.where(enough, mean_a - mean_b, np.nan)

    # Zero within-group variance (e.g. every expressor has the same value)
    # leaves no basis for a test; report the estimate but no inference.
    has_mu = enough & (df_resid >= 1) & np.isfinite(se_delta_mu) & (se_delta_mu > 0)
    se_delta_mu = np.where(has_mu, se_delta_mu, np.inf)
    z_mu = np.where(has_mu, delta_mu / np.where(has_mu, se_delta_mu, 1.0), 0.0)
    pvals_mu = np.where(has_mu, 2 * stats.t.sf(np.abs(z_mu), np.maximum(df_resid, 1)), 1.0)

    # =====================================================================
    # COMBINED: Wald test. Cov(Δπ̂, Δμ̂) = 0 (see module docstring).
    # =====================================================================
    both = has_pi & has_mu
    n_tested = has_pi.astype(int) + has_mu.astype(int)
    wald_statistic = np.where(n_tested > 0, z_pi**2 + z_mu**2, np.nan)
    pvals = np.ones(n_genes)
    for k in (1, 2):
        m = n_tested == k
        pvals[m] = stats.chi2.sf(wald_statistic[m], df=k)

    combined_z = np.where(n_tested > 0, z_pi + z_mu, np.nan)
    se_combined_z = np.where(both, np.sqrt(2.0), 1.0)

    return {
        "delta_pi": delta_pi,
        "se_delta_pi": se_delta_pi,
        "delta_pi_ci_lower": delta_pi - z_alpha * se_delta_pi,
        "delta_pi_ci_upper": delta_pi + z_alpha * se_delta_pi,
        "z_pi": z_pi,
        "pvals_pi": pvals_pi,
        "pvals_pi_adj": benjamini_hochberg(pvals_pi),
        "delta_mu": delta_mu,
        "se_delta_mu": se_delta_mu,
        "delta_mu_ci_lower": delta_mu - z_alpha * se_delta_mu,
        "delta_mu_ci_upper": delta_mu + z_alpha * se_delta_mu,
        "z_mu": z_mu,
        "pvals_mu": pvals_mu,
        "pvals_mu_adj": benjamini_hochberg(pvals_mu),
        "wald_statistic": wald_statistic,
        "combined_z": combined_z,
        "se_combined_z": se_combined_z,
        "combined_z_ci_lower": combined_z - z_alpha * se_combined_z,
        "combined_z_ci_upper": combined_z + z_alpha * se_combined_z,
        "pvals": pvals,
        "pvals_adj": benjamini_hochberg(pvals),
        "pct_nz_group": n_expr_a / n_cells_a,
        "pct_nz_reference": n_expr_b / n_cells_b,
    }


def _sum_sq_dev(n, s, ss):
    """Sum of squared deviations from the mean, from count, sum and sum of squares.

    Values below the floating-point cancellation error of ``ss - s²/n`` are
    rounded to zero so that constant columns give exactly zero variance.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = ss - s * s / n
    dev = np.where(n > 0, dev, 0.0)
    tol = 64 * np.finfo(np.float64).eps * ss
    return np.where(dev > tol, dev, 0.0)


def _warn_if_counts(X) -> None:
    """Warn when the input looks like raw integer counts rather than log values."""
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    sample = np.asarray(data[:100_000])
    sample = sample[sample > 0]
    if sample.size and np.all(sample == np.round(sample)) and sample.max() > 1:
        warnings.warn(
            "Expression values look like raw integer counts. rank_genes_groups_zi uses the "
            "input as given; normalize and log-transform first (sc.pp.normalize_total, "
            "sc.pp.log1p).",
            UserWarning,
            stacklevel=3,
        )
