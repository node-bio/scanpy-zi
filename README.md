# scanpy-zi

Decompose single-cell differential expression into penetrance and amplitude effects.

[![PyPI](https://img.shields.io/pypi/v/scanpy-zi.svg)](https://pypi.org/project/scanpy-zi/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Motivation

Standard differential expression methods report a single fold-change per gene. In single-cell data, this conflates two sources of variation:

1. **Fraction of cells expressing**: changes in the proportion of cells with detectable transcripts
2. **Expression level among expressors**: changes in abundance conditional on expression

### Zeros in scRNA-seq

Zeros in scRNA-seq arise from a mixture of technical and biological sources. Technical zeros (dropout) occur when a transcript is present but not captured due to inefficient lysis, reverse transcription, or amplification. Biological zeros occur when a gene is not transcribed in a given cell, whether due to chromatin state, absence of required transcription factors, or cell type identity.

The relative contribution of these sources varies by gene and technology. However, for genes with high zero rates, the zeros are predominantly biological. If dropout were the dominant source, we would expect zeros to be roughly uniform across genes at a given expression level. Instead, zero rates vary dramatically across genes even at similar mean expression, suggesting biological non-expression as a major component (Sarkar & Stephens 2021).

This has implications for differential expression. When comparing two conditions, a change in the fraction of zeros may reflect:

- More cells initiating transcription (biological)
- Differences in capture efficiency between conditions (technical)
- Some combination of both

For highly sparse genes, the biological interpretation is more defensible: if 95% of cells show zero expression, the 5% with detectable transcripts likely represent a distinct subpopulation rather than lucky capture events.

### The decomposition

scanpy-zi separates differential expression into two components:

| Effect | Measures | Statistic |
|--------|----------|-----------|
| **Δπ** (penetrance) | Change in fraction expressing | Log-odds ratio |
| **Δμ** (amplitude) | Change in expression level | Log fold change among expressors |

Each component has a separate effect size, confidence interval, and p-value. A joint Wald test provides overall significance while preserving the decomposition.

The terminology follows a biological framing:

- **Penetrance**: in how many cells is this gene active?
- **Amplitude**: given that a gene is active, how strongly is it expressed?

### Zero-inflated models

Zero-inflated regression models have been applied across many domains with excess zeros: ecology (species counts), insurance (claim frequency), public health (disease incidence), and manufacturing (defect counts). The statistical framework is well established. scanpy-zi applies these methods to single-cell transcriptomics, where the structure of the data (sparse counts, biological non-expression) fits the modeling assumptions.

## Installation
```bash
pip install scanpy-zi
```

## Usage
```python
import scanpy as sc
import scanpy_zi

# Load and normalize data
adata = sc.read_h5ad("your_data.h5ad")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

results = scanpy_zi.rank_genes_groups_zi(
    adata,
    groupby="celltype",
    groups=["T cells"],
    reference="rest",
    use_raw=False,
    copy=True,
)

# Top genes by combined significance
results.nsmallest(20, "pvals_adj")

# Genes with significant penetrance (FDR < 0.05)
results.query("pvals_pi_adj < 0.05")

# Genes with significant amplitude (FDR < 0.05)
results.query("pvals_mu_adj < 0.05")

# Visualization
scanpy_zi.pl.zi_decomposition(adata)
```

![ZI Decomposition](zi_decomposition.png)

## Output

| Column | Description |
|--------|-------------|
| `delta_pi` | Penetrance effect (log-odds ratio) |
| `se_delta_pi` | Standard error of penetrance |
| `delta_pi_ci_lower` | 95% CI lower bound |
| `delta_pi_ci_upper` | 95% CI upper bound |
| `pvals_pi_adj` | FDR-adjusted p-value for penetrance |
| `delta_mu` | Amplitude effect (log FC among expressors) |
| `se_delta_mu` | Standard error of amplitude |
| `delta_mu_ci_lower` | 95% CI lower bound |
| `delta_mu_ci_upper` | 95% CI upper bound |
| `pvals_mu_adj` | FDR-adjusted p-value for amplitude |
| `pvals_adj` | Combined FDR-adjusted p-value (Wald test) |
| `combined_z` | Combined z-score: z_π + z_μ (signed) |

## Group-vs-rest comparisons

Some implementations of zero-inflated models restrict analysis to pairwise group comparisons, on the grounds that the "rest" category is heterogeneous.

The log-odds ratio and mean difference remain well-defined statistics regardless of reference group heterogeneity. They estimate the aggregate difference between your group of interest and the background population.

The limitation is interpretive: group-vs-rest cannot distinguish whether a marker is unique to your group or shared with a subset of the reference. Pairwise comparisons are necessary to establish specificity.

## Method

**Penetrance**: Log-odds ratio from the 2×2 contingency table (expressing/non-expressing × group/reference). Haldane correction (+0.5) applied when any cell count is zero.

**Amplitude**: Mean difference among expressing cells. Input should be log-normalized, so this is a log fold change.

**Combined test**: Penetrance and amplitude estimators are correlated. We estimate their joint covariance using influence functions and compute a multivariate Wald statistic (χ² with 2 df).

**Implementation**: Sparse-native throughout. The expression matrix is never densified.

### Limitations

- Amplitude is undefined for genes with no expressors in either group (reported as `NaN`).
- Covariance estimation is unstable for genes with penetrance near 0 or 1. Correlation is clipped to [-0.99, 0.99].
- Input should be log-normalized. Count data should be normalized before analysis.
- The decomposition assumes zeros represent non-expression. For genes with substantial technical dropout, penetrance estimates will be attenuated.

## References

- Townes FW, Hicks SC, Aryee MJ, Irizarry RA (2019). Feature selection and dimension reduction for single-cell RNA-Seq based on a multinomial model. *Genome Biology* 20:295. [doi:10.1186/s13059-019-1861-6](https://doi.org/10.1186/s13059-019-1861-6)

- Sarkar A, Stephens M (2021). Separating measurement and expression models clarifies confusion in single-cell RNA sequencing analysis. *Nature Genetics* 53:770-777. [doi:10.1038/s41588-021-00873-4](https://doi.org/10.1038/s41588-021-00873-4)

## Citation
```bibtex
@software{scanpy_zi,
  author = {Bryan Downie},
  organization = {Node Biosciences},
  title = {scanpy-zi: Zero-inflated decomposition for single-cell differential expression},
  url = {https://github.com/node-bio/scanpy-zi}
}
```

## License

MIT