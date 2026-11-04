# Changelog

## 0.2.0

Correctness, speed, and API cleanup ahead of adoption as the shared ZI implementation.

### Breaking changes

- `rho_estimated` is removed from the results DataFrame and `adata.uns`. The influence-function
  covariance it came from summed deviations from a mean over the same cells the mean was
  computed on, so it was zero up to floating-point rounding. That rounding noise could reach
  the ±0.99 clip and distort `wald_statistic`, `pvals` and `se_combined_z`. The penetrance and
  amplitude estimators are uncorrelated by construction, so the Wald statistic is now
  `z_pi² + z_mu²` and `se_combined_z` is `sqrt(2)` (1 when only one component is testable).
- `min_cells` for penetrance now applies to expressing cells in both groups combined, not to
  each group. Previously a gene expressed in 0 cells of one group (an on/off marker) had its
  penetrance set to NaN and p-value 1. Amplitude still requires `min_cells` expressors in
  each group.
- `groups=None` now tests every group except `reference` (as in `sc.tl.rank_genes_groups`)
  instead of only the first group.
- `copy=True` no longer copies the AnnData object; it returns the DataFrame and leaves `adata`
  untouched (same observable behavior, without the memory cost).
- Output columns are float64.

### Fixed

- P-values use survival functions (`sf`) instead of `1 - cdf`, so strong markers no longer
  collapse to p = 0 ties. Within each group, rows are sorted by p-value with ties broken by
  the Wald statistic.
- Genes whose expressing cells all have the same value (zero variance) are reported as
  untestable for amplitude (p = 1) instead of producing near-infinite statistics from
  rounding error.
- Group labels are compared as strings, so integer or other non-string categories work.
- `delta_mu` / `logfoldchanges` are documented as being in input units (natural log for
  `sc.pp.log1p` data), not log2. The plot axis labels are updated to match. A warning is
  raised when the input looks like raw integer counts.
- `expression_threshold < 0`, `min_cells < 1`, and out-of-range `confidence_level` now raise
  `ValueError`.

### Performance

- Statistics for all requested groups come from a single pass over the stored values of the
  matrix (`np.bincount` in float64). "Rest" is computed as totals minus the group. All
  per-gene work is vectorized; the per-gene Python loops are gone.
- 46,148 cells × 16,648 genes (73M non-zeros), 35 groups vs rest: about 2.3 s total, versus
  about 3.1 s per group (about 110 s total) in 0.1.0.

### Dependencies

- `statsmodels` is no longer a runtime dependency; Benjamini-Hochberg is implemented
  directly. It remains a dev dependency for a cross-check test.

- Amplitude uses the `min_cells` parameter instead of a hardcoded 3 (PR #1, merged after
  0.1.0 and first released here).

## 0.1.0

- Initial release.
