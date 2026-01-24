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

        # Gene 1 should have NaN delta_pi due to insufficient cells
        gene_1 = result[result["names"] == "gene_1"].iloc[0]
        assert np.isnan(gene_1["delta_pi"])
        assert gene_1["pvals_pi"] == 1.0

        # Gene 0 should have valid penetrance
        gene_0 = result[result["names"] == "gene_0"].iloc[0]
        assert not np.isnan(gene_0["delta_pi"])

    def test_custom_min_cells(self, low_expressing_adata):
        """Test that custom min_cells parameter works."""
        # With min_cells=3, gene_1 should pass
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
