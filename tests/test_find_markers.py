"""Tests for ZI decomposition."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from scanpy_zi import rank_genes_groups_zi


@pytest.fixture
def simple_adata():
    """Create a simple AnnData for testing."""
    np.random.seed(42)

    # Create two groups with different expression patterns
    n_cells_a = 100
    n_cells_b = 100
    n_genes = 50

    # Group A: higher penetrance for gene 0, higher amplitude for gene 1
    X_a = np.random.poisson(0.5, (n_cells_a, n_genes)).astype(np.float32)
    X_a[:80, 0] = np.random.poisson(2, 80)  # Gene 0: 80% expressing
    X_a[:50, 1] = np.random.poisson(5, 50)  # Gene 1: high amplitude

    # Group B: lower expression
    X_b = np.random.poisson(0.5, (n_cells_b, n_genes)).astype(np.float32)
    X_b[:30, 0] = np.random.poisson(2, 30)  # Gene 0: 30% expressing
    X_b[:50, 1] = np.random.poisson(1, 50)  # Gene 1: low amplitude

    X = np.vstack([X_a, X_b])
    X = np.log1p(X)  # Log-normalize

    obs = pd.DataFrame(
        {"group": ["A"] * n_cells_a + ["B"] * n_cells_b},
        index=[f"cell_{i}" for i in range(n_cells_a + n_cells_b)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    return AnnData(X, obs=obs, var=var)


@pytest.fixture
def sparse_adata(simple_adata):
    """Convert simple_adata to sparse format."""
    adata = simple_adata.copy()
    adata.X = sp.csr_matrix(adata.X)
    return adata


class TestRankGenesGroupsZI:
    """Tests for rank_genes_groups_zi function."""

    def test_basic_usage(self, simple_adata):
        """Test basic function call."""
        result = rank_genes_groups_zi(
            simple_adata,
            groupby="group",
            groups=["A"],
            reference="B",
            use_raw=False,
            copy=True,
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == simple_adata.n_vars
        assert "delta_pi" in result.columns
        assert "delta_mu" in result.columns
        assert "combined_z" in result.columns

    def test_stores_in_uns(self, simple_adata):
        """Test that results are stored in adata.uns."""
        rank_genes_groups_zi(
            simple_adata,
            groupby="group",
            groups=["A"],
            reference="B",
            use_raw=False,
        )

        assert "rank_genes_groups_zi" in simple_adata.uns
        assert "delta_pi" in simple_adata.uns["rank_genes_groups_zi"]
        assert "delta_mu" in simple_adata.uns["rank_genes_groups_zi"]
        assert "combined_z" in simple_adata.uns["rank_genes_groups_zi"]

    def test_sparse_input(self, sparse_adata):
        """Test with sparse matrix input."""
        result = rank_genes_groups_zi(
            sparse_adata,
            groupby="group",
            groups=["A"],
            reference="B",
            use_raw=False,
            copy=True,
        )

        assert result is not None
        assert len(result) == sparse_adata.n_vars

    def test_rest_reference(self, simple_adata):
        """Test using 'rest' as reference."""
        result = rank_genes_groups_zi(
            simple_adata,
            groupby="group",
            groups=["A"],
            reference="rest",
            use_raw=False,
            copy=True,
        )

        assert result is not None
        assert len(result) == simple_adata.n_vars

    def test_pvalues_valid(self, simple_adata):
        """Test that p-values are in valid range."""
        result = rank_genes_groups_zi(
            simple_adata,
            groupby="group",
            groups=["A"],
            reference="B",
            use_raw=False,
            copy=True,
        )

        assert (result["pvals"] >= 0).all()
        assert (result["pvals"] <= 1).all()
        assert (result["pvals_adj"] >= 0).all()
        assert (result["pvals_adj"] <= 1).all()

    def test_invalid_groupby(self, simple_adata):
        """Test error on invalid groupby column."""
        with pytest.raises(ValueError, match="not found"):
            rank_genes_groups_zi(
                simple_adata,
                groupby="nonexistent",
                groups=["A"],
                reference="B",
                use_raw=False,
            )

    def test_invalid_group(self, simple_adata):
        """Test error on invalid group name."""
        with pytest.raises(ValueError, match="not found"):
            rank_genes_groups_zi(
                simple_adata,
                groupby="group",
                groups=["nonexistent"],
                reference="B",
                use_raw=False,
            )

    def test_custom_key_added(self, simple_adata):
        """Test custom key_added parameter."""
        rank_genes_groups_zi(
            simple_adata,
            groupby="group",
            groups=["A"],
            reference="B",
            use_raw=False,
            key_added="my_custom_key",
        )

        assert "my_custom_key" in simple_adata.uns
        assert "rank_genes_groups_zi" not in simple_adata.uns

    def test_ci_columns_present(self, simple_adata):
        """Test that CI columns are present in output."""
        result = rank_genes_groups_zi(
            simple_adata,
            groupby="group",
            groups=["A"],
            reference="B",
            use_raw=False,
            copy=True,
        )

        # Check penetrance CIs
        assert "delta_pi_ci_lower" in result.columns
        assert "delta_pi_ci_upper" in result.columns

        # Check amplitude CIs
        assert "delta_mu_ci_lower" in result.columns
        assert "delta_mu_ci_upper" in result.columns

        # Check joint z CIs
        assert "combined_z_ci_lower" in result.columns
        assert "combined_z_ci_upper" in result.columns

    def test_ci_values_valid(self, simple_adata):
        """Test that CI values are properly ordered (lower < upper)."""
        result = rank_genes_groups_zi(
            simple_adata,
            groupby="group",
            groups=["A"],
            reference="B",
            use_raw=False,
            copy=True,
        )

        # For valid (non-NaN) entries, lower should be <= upper
        valid_pi = result["delta_pi"].notna()
        if valid_pi.any():
            assert (
                result.loc[valid_pi, "delta_pi_ci_lower"]
                <= result.loc[valid_pi, "delta_pi_ci_upper"]
            ).all()

        valid_mu = result["delta_mu"].notna()
        if valid_mu.any():
            assert (
                result.loc[valid_mu, "delta_mu_ci_lower"]
                <= result.loc[valid_mu, "delta_mu_ci_upper"]
            ).all()

        valid_joint = result["combined_z"].notna()
        if valid_joint.any():
            assert (
                result.loc[valid_joint, "combined_z_ci_lower"]
                <= result.loc[valid_joint, "combined_z_ci_upper"]
            ).all()


class TestMultiGroup:
    """Tests for multi-group processing."""

    @pytest.fixture
    def three_group_adata(self):
        """Create an AnnData with three groups."""
        np.random.seed(42)

        n_cells_per_group = 50
        n_genes = 30

        X_a = np.random.poisson(1, (n_cells_per_group, n_genes)).astype(np.float32)
        X_b = np.random.poisson(2, (n_cells_per_group, n_genes)).astype(np.float32)
        X_c = np.random.poisson(3, (n_cells_per_group, n_genes)).astype(np.float32)

        X = np.vstack([X_a, X_b, X_c])
        X = np.log1p(X)

        obs = pd.DataFrame(
            {
                "cluster": ["A"] * n_cells_per_group
                + ["B"] * n_cells_per_group
                + ["C"] * n_cells_per_group
            },
            index=[f"cell_{i}" for i in range(3 * n_cells_per_group)],
        )
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

        return AnnData(X, obs=obs, var=var)

    def test_multiple_groups(self, three_group_adata):
        """Test processing multiple groups at once."""
        result = rank_genes_groups_zi(
            three_group_adata,
            groupby="cluster",
            groups=["A", "B"],
            reference="rest",
            use_raw=False,
            copy=True,
        )

        assert result is not None
        # Should have results for both groups
        assert set(result["group"].unique()) == {"A", "B"}
        # Should have all genes for each group
        assert len(result) == three_group_adata.n_vars * 2

    def test_multiple_groups_in_uns(self, three_group_adata):
        """Test that multiple groups are stored in adata.uns."""
        rank_genes_groups_zi(
            three_group_adata,
            groupby="cluster",
            groups=["A", "B"],
            reference="rest",
            use_raw=False,
        )

        assert "rank_genes_groups_zi" in three_group_adata.uns
        # Check both groups are in the results
        assert "A" in three_group_adata.uns["rank_genes_groups_zi"]["names"]
        assert "B" in three_group_adata.uns["rank_genes_groups_zi"]["names"]


class TestMinCellsFilter:
    """Tests for min_cells penetrance filter."""

    @pytest.fixture
    def low_expressing_adata(self):
        """Create an AnnData with some genes having few expressing cells."""
        np.random.seed(42)

        n_cells_a = 100
        n_cells_b = 100
        n_genes = 20

        # Most genes are zero (no expression)
        X_a = np.zeros((n_cells_a, n_genes), dtype=np.float32)
        X_b = np.zeros((n_cells_b, n_genes), dtype=np.float32)

        # Gene 0: well expressed in both groups
        X_a[:50, 0] = np.random.uniform(1, 3, 50)
        X_b[:40, 0] = np.random.uniform(1, 3, 40)

        # Gene 1: only 5 cells express in group A (below default min_cells=10)
        X_a[:5, 1] = np.random.uniform(1, 3, 5)
        X_b[:15, 1] = np.random.uniform(1, 3, 15)

        # Gene 2: 6 expressors in total (below min_cells=10 combined)
        X_a[:3, 2] = np.random.uniform(1, 3, 3)
        X_b[:3, 2] = np.random.uniform(1, 3, 3)

        X = np.vstack([X_a, X_b])

        obs = pd.DataFrame(
            {"group": ["A"] * n_cells_a + ["B"] * n_cells_b},
            index=[f"cell_{i}" for i in range(n_cells_a + n_cells_b)],
        )
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

        return AnnData(X, obs=obs, var=var)

    def test_min_cells_filters_penetrance(self, low_expressing_adata):
        """Test that genes with few expressing cells have filtered penetrance."""
        result = rank_genes_groups_zi(
            low_expressing_adata,
            groupby="group",
            groups=["A"],
            reference="B",
            use_raw=False,
            min_cells=10,
            copy=True,
        )

        # Gene 2 has 6 expressors in total: penetrance is not tested
        gene_2 = result[result["names"] == "gene_2"].iloc[0]
        assert np.isnan(gene_2["delta_pi"])
        assert gene_2["pvals_pi"] == 1.0

        # Gene 1 has 20 expressors in total, so penetrance is tested, but only
        # 5 in group A, so amplitude is not
        gene_1 = result[result["names"] == "gene_1"].iloc[0]
        assert not np.isnan(gene_1["delta_pi"])
        assert np.isnan(gene_1["delta_mu"])
        assert gene_1["pvals_mu"] == 1.0

        # Gene 0 should have valid penetrance and amplitude
        gene_0 = result[result["names"] == "gene_0"].iloc[0]
        assert not np.isnan(gene_0["delta_pi"])
        assert not np.isnan(gene_0["delta_mu"])

    def test_custom_min_cells(self, low_expressing_adata):
        """Test that custom min_cells parameter works for both components."""
        result = rank_genes_groups_zi(
            low_expressing_adata,
            groupby="group",
            groups=["A"],
            reference="B",
            use_raw=False,
            min_cells=3,
            copy=True,
        )

        gene_1 = result[result["names"] == "gene_1"].iloc[0]
        assert not np.isnan(gene_1["delta_pi"])
        assert not np.isnan(gene_1["delta_mu"])
        gene_2 = result[result["names"] == "gene_2"].iloc[0]
        assert not np.isnan(gene_2["delta_pi"])
        assert not np.isnan(gene_2["delta_mu"])

    def test_on_off_marker_is_tested(self):
        """A gene on in one group and off in the other must not be dropped."""
        rng = np.random.default_rng(0)
        X = np.zeros((200, 2), dtype=np.float32)
        X[:80, 0] = rng.uniform(1, 3, 80)  # 80% of A, 0% of B
        X[:, 1] = rng.uniform(1, 3, 200)
        obs = pd.DataFrame({"g": ["A"] * 100 + ["B"] * 100}, index=[str(i) for i in range(200)])
        adata = AnnData(X, obs=obs)
        res = rank_genes_groups_zi(adata, "g", groups="A", reference="B", use_raw=False, copy=True)
        top = res.iloc[0]
        assert top["names"] == "0"
        assert top["delta_pi"] > 0
        # Haldane-corrected SE is wide when a cell is zero (z ~ 4.6 here)
        assert top["pvals_pi"] < 1e-4
        assert np.isnan(top["delta_mu"])  # no expressors in B
        np.testing.assert_allclose(top["pvals"], top["pvals_pi"], rtol=1e-10)


class TestSparseUtils:
    """Tests for sparse utility functions."""

    def test_sparse_vs_dense_consistency(self, simple_adata, sparse_adata):
        """Test that sparse and dense produce similar results."""
        result_dense = rank_genes_groups_zi(
            simple_adata,
            groupby="group",
            groups=["A"],
            reference="B",
            use_raw=False,
            copy=True,
        )

        result_sparse = rank_genes_groups_zi(
            sparse_adata,
            groupby="group",
            groups=["A"],
            reference="B",
            use_raw=False,
            copy=True,
        )

        # Results should be very close
        np.testing.assert_allclose(
            result_dense["delta_pi"].values,
            result_sparse["delta_pi"].values,
            rtol=1e-5,
        )

        # delta_mu may have NaN, so handle that
        dense_mu = result_dense["delta_mu"].fillna(0).values
        sparse_mu = result_sparse["delta_mu"].fillna(0).values
        np.testing.assert_allclose(dense_mu, sparse_mu, rtol=1e-2, atol=1e-6)


class TestConfidenceLevel:
    """Tests for confidence level parameter."""

    def test_custom_confidence_level(self, simple_adata):
        """Test that custom confidence level is recorded."""
        # Use copy=False so results are stored in adata.uns
        rank_genes_groups_zi(
            simple_adata,
            groupby="group",
            groups=["A"],
            reference="B",
            use_raw=False,
            confidence_level=0.90,
        )

        assert simple_adata.uns["rank_genes_groups_zi"]["params"]["confidence_level"] == 0.90

    def test_narrower_ci_at_lower_confidence(self, simple_adata):
        """Test that 90% CI is narrower than 95% CI."""
        result_95 = rank_genes_groups_zi(
            simple_adata.copy(),
            groupby="group",
            groups=["A"],
            reference="B",
            use_raw=False,
            confidence_level=0.95,
            copy=True,
        )

        result_90 = rank_genes_groups_zi(
            simple_adata.copy(),
            groupby="group",
            groups=["A"],
            reference="B",
            use_raw=False,
            confidence_level=0.90,
            copy=True,
        )

        # 90% CI should be narrower than 95% CI (on average for valid entries)
        valid = result_95["delta_pi"].notna() & result_90["delta_pi"].notna()
        if valid.any():
            width_95 = (
                result_95.loc[valid, "delta_pi_ci_upper"]
                - result_95.loc[valid, "delta_pi_ci_lower"]
            )
            width_90 = (
                result_90.loc[valid, "delta_pi_ci_upper"]
                - result_90.loc[valid, "delta_pi_ci_lower"]
            )
            assert (width_90 < width_95).all()


def _reference_stats(xa, xb, threshold=0.0):
    """Slow per-gene reference implementation for one gene."""
    from scipy import stats

    xa, xb = xa.astype(np.float64), xb.astype(np.float64)
    ea, eb = xa[xa > threshold], xb[xb > threshold]
    a, b, c, d = len(ea), len(xa) - len(ea), len(eb), len(xb) - len(eb)
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    lor = np.log(a * d / (b * c))
    se_lor = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    t = stats.ttest_ind(ea.astype(np.float64), eb.astype(np.float64), equal_var=True)
    return lor, se_lor, ea.mean() - eb.mean(), t.statistic, t.pvalue


class TestStatistics:
    """Check the vectorized statistics against a per-gene reference."""

    @pytest.fixture
    def adata(self):
        rng = np.random.default_rng(1)
        n, g = 300, 40
        dens = rng.uniform(0.1, 0.9, g)
        mask = rng.random((n, g)) < dens
        X = np.where(mask, np.log1p(rng.gamma(2.0, 2.0, (n, g))), 0.0).astype(np.float32)
        X[:100, 0] *= 2.0  # amplitude effect in group A
        X[:100, 1] = np.where(rng.random(100) < 0.95, X[:100, 1] + 1, 0)  # penetrance
        labels = np.array(["A"] * 100 + ["B"] * 120 + ["C"] * 80)
        obs = pd.DataFrame({"g": pd.Categorical(labels)}, index=[str(i) for i in range(n)])
        return AnnData(sp.csr_matrix(X), obs=obs)

    @pytest.mark.parametrize("reference", ["rest", "B"])
    def test_matches_reference(self, adata, reference):
        res = rank_genes_groups_zi(
            adata, "g", groups="A", reference=reference, use_raw=False, min_cells=2, copy=True
        ).set_index("names")
        X = adata.X.toarray()
        ma = (adata.obs["g"] == "A").to_numpy()
        mb = ~ma if reference == "rest" else (adata.obs["g"] == "B").to_numpy()
        for j, name in enumerate(adata.var_names):
            assert min((X[ma, j] > 0).sum(), (X[mb, j] > 0).sum()) >= 2
            lor, se_lor, dmu, t, p = _reference_stats(X[ma, j], X[mb, j])
            row = res.loc[name]
            np.testing.assert_allclose(row["delta_pi"], lor, rtol=1e-10)
            np.testing.assert_allclose(row["se_delta_pi"], se_lor, rtol=1e-10)
            np.testing.assert_allclose(row["delta_mu"], dmu, rtol=1e-6, atol=1e-9)
            np.testing.assert_allclose(row["z_mu"], t, rtol=1e-6)
            np.testing.assert_allclose(row["pvals_mu"], p, rtol=1e-6, atol=1e-300)

    def test_wald_has_no_correlation_term(self, adata):
        res = rank_genes_groups_zi(adata, "g", groups="A", use_raw=False, copy=True)
        assert "rho_estimated" not in res.columns
        np.testing.assert_allclose(res["wald_statistic"], res["z_pi"] ** 2 + res["z_mu"] ** 2)
        both = res["delta_pi"].notna() & np.isfinite(res["se_delta_mu"])
        np.testing.assert_allclose(res.loc[both, "se_combined_z"], np.sqrt(2))

    def test_all_groups_matches_one_at_a_time(self, adata):
        allres = rank_genes_groups_zi(adata, "g", use_raw=False, copy=True)
        assert set(allres["group"]) == {"A", "B", "C"}
        for grp in ["A", "B", "C"]:
            one = rank_genes_groups_zi(adata, "g", groups=grp, use_raw=False, copy=True)
            sub = allres[allres["group"] == grp].reset_index(drop=True)
            pd.testing.assert_frame_equal(sub, one)

    def test_dense_and_sparse_identical(self, adata):
        dense = adata.copy()
        dense.X = adata.X.toarray()
        r1 = rank_genes_groups_zi(adata, "g", use_raw=False, copy=True)
        r2 = rank_genes_groups_zi(dense, "g", use_raw=False, copy=True)
        pd.testing.assert_frame_equal(r1, r2)

    def test_non_canonical_sparse(self, adata):
        X = adata.X
        dup = sp.csr_matrix(
            (np.repeat(X.data / 2, 2), np.repeat(X.indices, 2), X.indptr * 2), shape=X.shape
        )
        assert not dup.has_canonical_format
        other = adata.copy()
        other.X = dup
        r1 = rank_genes_groups_zi(adata, "g", use_raw=False, copy=True)
        r2 = rank_genes_groups_zi(other, "g", use_raw=False, copy=True)
        pd.testing.assert_frame_equal(r1, r2, rtol=1e-5)

    def test_expression_threshold(self, adata):
        res = rank_genes_groups_zi(
            adata,
            "g",
            groups="A",
            reference="B",
            use_raw=False,
            expression_threshold=1.0,
            copy=True,
        ).set_index("names")
        X = adata.X.toarray()
        ma = (adata.obs["g"] == "A").to_numpy()
        mb = (adata.obs["g"] == "B").to_numpy()
        lor, _, dmu, _, _ = _reference_stats(X[ma, 3], X[mb, 3], threshold=1.0)
        np.testing.assert_allclose(res.loc["3", "delta_pi"], lor, rtol=1e-10)
        np.testing.assert_allclose(res.loc["3", "delta_mu"], dmu, rtol=1e-6)

    def test_pvalues_do_not_underflow(self):
        rng = np.random.default_rng(0)
        n = 2000
        X = np.zeros((2 * n, 1), dtype=np.float32)
        X[: n * 7 // 10, 0] = rng.uniform(1, 2, n * 7 // 10)  # 70% of A
        X[n : n + n * 3 // 10, 0] = rng.uniform(1, 2, n * 3 // 10)  # 30% of B
        obs = pd.DataFrame({"g": ["A"] * n + ["B"] * n}, index=[str(i) for i in range(2 * n)])
        res = rank_genes_groups_zi(AnnData(X, obs=obs), "g", groups="A", use_raw=False, copy=True)
        # |z_pi| is ~25 here; 1 - cdf rounds to exactly 0 beyond z ~ 8.3
        assert abs(res["z_pi"].iloc[0]) > 20
        assert 0 < res["pvals_pi"].iloc[0] < 1e-100

    def test_zero_variance_amplitude_is_untested(self):
        X = np.zeros((40, 1), dtype=np.float32)
        X[:15, 0] = np.log(2)  # every expressor has the same value
        X[20:32, 0] = np.log(2)
        obs = pd.DataFrame({"g": ["A"] * 20 + ["B"] * 20}, index=[str(i) for i in range(40)])
        res = rank_genes_groups_zi(AnnData(X, obs=obs), "g", groups="A", use_raw=False, copy=True)
        row = res.iloc[0]
        assert row["pvals_mu"] == 1.0
        assert row["z_mu"] == 0.0
        assert row["pvals"] > 0.01


class TestApi:
    def test_copy_does_not_modify_adata(self, simple_adata):
        before = set(simple_adata.uns)
        rank_genes_groups_zi(simple_adata, "group", groups="A", use_raw=False, copy=True)
        assert set(simple_adata.uns) == before

    def test_groups_none_tests_all_but_reference(self, simple_adata):
        res = rank_genes_groups_zi(simple_adata, "group", reference="B", use_raw=False, copy=True)
        assert set(res["group"]) == {"A"}

    def test_integer_labels(self, simple_adata):
        simple_adata.obs["num"] = pd.Categorical([0] * 100 + [1] * 100)
        res = rank_genes_groups_zi(simple_adata, "num", groups=[0], use_raw=False, copy=True)
        assert set(res["group"]) == {"0"}

    def test_uns_layout(self, simple_adata):
        rank_genes_groups_zi(simple_adata, "group", use_raw=False)
        uns = simple_adata.uns["rank_genes_groups_zi"]
        assert "rho_estimated" not in uns
        assert uns["params"]["min_cells"] == 10
        assert len(uns["names"]["A"]) == simple_adata.n_vars

    def test_warns_on_counts(self, simple_adata):
        simple_adata.X = np.round(np.expm1(simple_adata.X))
        with pytest.warns(UserWarning, match="raw integer counts"):
            rank_genes_groups_zi(simple_adata, "group", groups="A", use_raw=False, copy=True)

    def test_invalid_threshold(self, simple_adata):
        with pytest.raises(ValueError, match="expression_threshold"):
            rank_genes_groups_zi(simple_adata, "group", use_raw=False, expression_threshold=-1.0)


def test_benjamini_hochberg_matches_statsmodels():
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    from scanpy_zi._sparse_utils import benjamini_hochberg

    rng = np.random.default_rng(0)
    p = np.concatenate([rng.random(500), rng.random(50) * 1e-4, [np.nan, 1.0, 0.0]])
    expected = np.ones_like(p)
    ok = np.isfinite(p)
    expected[ok] = multitest.multipletests(p[ok], method="fdr_bh")[1]
    np.testing.assert_allclose(benjamini_hochberg(p), expected, rtol=1e-12)


def test_plot_smoke(simple_adata, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import scanpy_zi

    rank_genes_groups_zi(simple_adata, "group", groups="A", use_raw=False)
    fig = scanpy_zi.pl.zi_decomposition(simple_adata, show=False, save=tmp_path / "zi.png")
    assert fig is not None
    assert (tmp_path / "zi.png").exists()
