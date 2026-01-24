"""Visualization functions for ZI decomposition results."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData


def zi_decomposition(
    adata: AnnData,
    results_key: str = "rank_genes_groups_zi",
    *,
    highlight_genes: Sequence[str] | None = None,
    n_top_genes: int = 30,
    figsize: tuple[float, float] = (18, 12),
    dpi: int = 150,
    save: str | Path | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """Plot zero-inflated decomposition results.

    Creates a multi-panel figure showing:
    - Scatter plot of amplitude (Δμ) vs penetrance (Δπ)
    - Forest plot for penetrance effects
    - Forest plot for amplitude effects
    - Forest plot for joint effect size

    Parameters
    ----------
    adata
        AnnData object with ZI decomposition results.
    results_key
        Key in adata.uns containing ZI results.
    highlight_genes
        Genes to highlight regardless of significance.
    n_top_genes
        Number of top genes to show in forest plots.
    figsize
        Figure size (width, height).
    dpi
        Figure resolution.
    save
        Path to save figure. If None, figure is not saved.
    show
        Whether to show the figure.

    Returns
    -------
    Figure object if show=False, otherwise None.

    Examples
    --------
    >>> import scanpy_zi
    >>> scanpy_zi.rank_genes_groups_zi(adata, groupby="cluster", groups=["A"])
    >>> scanpy_zi.pl.zi_decomposition(adata)
    """
    if results_key not in adata.uns:
        raise ValueError(
            f"ZI decomposition results not found at adata.uns['{results_key}']. "
            f"Run rank_genes_groups_zi first. Available keys: {list(adata.uns.keys())}"
        )

    results = adata.uns[results_key]

    if "delta_pi" not in results:
        raise ValueError(
            f"Results in '{results_key}' do not contain ZI decomposition data. "
            f"Keys found: {list(results.keys())}"
        )

    # Extract group name
    if isinstance(results["names"], dict):
        group_names = list(results["names"].keys())
        group = group_names[0]
    else:
        group = "group_0"

    def _extract_data(data, grp):
        if isinstance(data, dict):
            return np.array(data[grp])
        elif hasattr(data, "dtype") and data.dtype.names:
            return np.array(data[grp])
        else:
            return np.array(data)

    n_genes_total = len(_extract_data(results["names"], group))

    df = pd.DataFrame(
        {
            "gene": _extract_data(results["names"], group),
            "delta_pi": _extract_data(results["delta_pi"], group),
            "se_delta_pi": _extract_data(results.get("se_delta_pi", {}), group)
            if "se_delta_pi" in results
            else np.full(n_genes_total, np.nan),
            "z_pi": _extract_data(results.get("z_pi", {}), group)
            if "z_pi" in results
            else np.zeros(n_genes_total),
            "pvals_pi_adj": _extract_data(results.get("pvals_pi_adj", {}), group)
            if "pvals_pi_adj" in results
            else np.ones(n_genes_total),
            "delta_mu": _extract_data(
                results.get("delta_mu", results.get("logfoldchanges", {})), group
            ),
            "se_delta_mu": _extract_data(results.get("se_delta_mu", {}), group)
            if "se_delta_mu" in results
            else np.full(n_genes_total, np.nan),
            "z_mu": _extract_data(results.get("z_mu", {}), group)
            if "z_mu" in results
            else np.zeros(n_genes_total),
            "pvals_mu_adj": _extract_data(results.get("pvals_mu_adj", {}), group)
            if "pvals_mu_adj" in results
            else np.ones(n_genes_total),
            "combined_z": _extract_data(results.get("combined_z", {}), group)
            if "combined_z" in results
            else np.zeros(n_genes_total),
            "combined_z_ci_lower": _extract_data(results.get("combined_z_ci_lower", {}), group)
            if "combined_z_ci_lower" in results
            else np.full(n_genes_total, np.nan),
            "combined_z_ci_upper": _extract_data(results.get("combined_z_ci_upper", {}), group)
            if "combined_z_ci_upper" in results
            else np.full(n_genes_total, np.nan),
        }
    )

    # Compute effect_type for coloring based on FDR-corrected p-values
    fdr_threshold = 0.05
    sig_pi = df["pvals_pi_adj"] < fdr_threshold
    sig_mu = df["pvals_mu_adj"] < fdr_threshold
    df["effect_type"] = np.where(
        sig_pi & sig_mu,
        "both",
        np.where(sig_pi, "penetrance", np.where(sig_mu, "amplitude", "none")),
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    color_map = {
        "penetrance": "#E74C3C",
        "amplitude": "#3498DB",
        "both": "#9B59B6",
        "none": "#BDC3C7",
    }

    df_plot = df.copy()
    df_plot["delta_mu_plot"] = df_plot["delta_mu"].fillna(0)

    valid_mask = df_plot["delta_pi"].notna()
    df_valid = df_plot[valid_mask].copy()

    df_valid["delta_mu_display"] = df_valid["delta_mu"]
    df_valid["delta_mu"] = df_valid["delta_mu_plot"]

    df_valid["abs_combined_z"] = np.abs(df_valid["combined_z"].fillna(0))

    top_genes_df = df_valid.nlargest(min(n_top_genes, len(df_valid)), "abs_combined_z")

    if highlight_genes:
        highlight_mask = df_valid["gene"].isin(highlight_genes)
        for _, row in df_valid[highlight_mask].iterrows():
            if row["gene"] not in top_genes_df["gene"].values:
                top_genes_df = pd.concat([top_genes_df, pd.DataFrame([row])], ignore_index=True)

    forest_genes_df = top_genes_df.head(min(20, len(top_genes_df)))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1.2, 1], height_ratios=[1, 1, 1], hspace=0.35, wspace=0.25
    )

    # Scatter plot
    ax_scatter = fig.add_subplot(gs[:, 0])

    if len(df_valid) == 0:
        ax_scatter.text(0.5, 0.5, "No valid data points", ha="center", va="center", fontsize=12)
    else:
        has_se = df_valid["se_delta_pi"].notna() & df_valid["se_delta_mu"].notna()
        if has_se.any():
            combined_se = np.sqrt(df_valid["se_delta_pi"] ** 2 + df_valid["se_delta_mu"] ** 2)
            precision = 1.0 / (combined_se + 0.01)
            prec_min = np.nanmin(precision)
            prec_max = np.nanmax(precision)
            if np.isnan(prec_min) or np.isnan(prec_max) or prec_min == prec_max:
                sizes = pd.Series(30, index=df_valid.index)
            else:
                precision_norm = (precision - prec_min) / (prec_max - prec_min + 1e-10)
                sizes = 15 + 80 * precision_norm
                sizes = sizes.fillna(30)
        else:
            sizes = pd.Series(30, index=df_valid.index)

        # Plot in order: none first (background), then significant categories on top
        plot_order = ["none", "penetrance", "amplitude", "both"]
        for effect_type in plot_order:
            color = color_map[effect_type]
            mask = df_valid["effect_type"] == effect_type
            n_points = mask.sum()
            if n_points > 0:
                point_sizes = sizes[mask] if isinstance(sizes, pd.Series) else sizes
                if isinstance(point_sizes, pd.Series):
                    point_sizes = point_sizes.fillna(30).values
                zorder = 1 if effect_type == "none" else 2
                ax_scatter.scatter(
                    df_valid.loc[mask, "delta_mu"],
                    df_valid.loc[mask, "delta_pi"],
                    c=color,
                    s=point_sizes,
                    alpha=0.6,
                    label=f"{effect_type.capitalize()} ({n_points})",
                    edgecolors="white",
                    linewidths=0.3,
                    zorder=zorder,
                )

        ax_scatter.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)
        ax_scatter.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)

        # Annotate top genes - labels on top of everything
        for _, row in forest_genes_df.head(20).iterrows():
            ax_scatter.annotate(
                row["gene"],
                (row["delta_mu"], row["delta_pi"]),
                fontsize=7,
                alpha=0.9,
                xytext=(3, 3),
                textcoords="offset points",
                zorder=10,
            )

        ax_scatter.set_xlabel("Δμ (Amplitude: log2 FC among expressors)", fontsize=11)
        ax_scatter.set_ylabel("Log-Odds Ratio (Penetrance)", fontsize=11)
        ax_scatter.set_title(
            "Zero-Inflated Decomposition\n(point size ∝ precision)", fontsize=13, fontweight="bold"
        )
        ax_scatter.legend(loc="upper left", fontsize=9)
        ax_scatter.grid(alpha=0.3)

    # Forest plot: Penetrance
    ax_forest_pi = fig.add_subplot(gs[0, 1])

    if len(forest_genes_df) > 0:
        y_positions = np.arange(len(forest_genes_df))
        gene_names = forest_genes_df["gene"].values[::-1]
        delta_pi_vals = forest_genes_df["delta_pi"].values[::-1]
        se_pi_vals = forest_genes_df["se_delta_pi"].values[::-1]
        effect_types = forest_genes_df["effect_type"].values[::-1]

        ci_95 = 1.96

        for i, (_gene, effect, d_pi, se_pi) in enumerate(
            zip(gene_names, effect_types, delta_pi_vals, se_pi_vals)
        ):
            color = color_map.get(effect, "#BDC3C7")

            if not np.isnan(se_pi):
                ci_low = d_pi - ci_95 * se_pi
                ci_high = d_pi + ci_95 * se_pi
                ax_forest_pi.plot([ci_low, ci_high], [i, i], color=color, linewidth=2, alpha=0.7)

            ax_forest_pi.scatter(
                [d_pi], [i], color=color, s=60, zorder=5, edgecolors="black", linewidths=0.5
            )

        ax_forest_pi.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.7)
        ax_forest_pi.set_yticks(y_positions)
        ax_forest_pi.set_yticklabels(gene_names, fontsize=8)
        ax_forest_pi.set_xlabel("Log-Odds Ratio ± 95% CI", fontsize=10)
        ax_forest_pi.set_title("Penetrance Forest Plot", fontsize=12, fontweight="bold")
        ax_forest_pi.grid(axis="x", alpha=0.3)

    # Forest plot: Amplitude
    ax_forest_mu = fig.add_subplot(gs[1, 1])

    if len(forest_genes_df) > 0:
        y_positions = np.arange(len(forest_genes_df))
        gene_names = forest_genes_df["gene"].values[::-1]
        delta_mu_vals = forest_genes_df["delta_mu_display"].values[::-1]
        se_mu_vals = forest_genes_df["se_delta_mu"].values[::-1]
        effect_types = forest_genes_df["effect_type"].values[::-1]

        for i, (_gene, effect, dmu, se_mu) in enumerate(
            zip(gene_names, effect_types, delta_mu_vals, se_mu_vals)
        ):
            color = color_map.get(effect, "#BDC3C7")

            if np.isnan(dmu):
                ax_forest_mu.scatter([0], [i], color="#CCCCCC", s=30, marker="x", alpha=0.5)
                continue

            if not np.isnan(se_mu):
                ci_low = dmu - ci_95 * se_mu
                ci_high = dmu + ci_95 * se_mu
                ax_forest_mu.plot([ci_low, ci_high], [i, i], color=color, linewidth=2, alpha=0.7)

            ax_forest_mu.scatter(
                [dmu],
                [i],
                color=color,
                s=60,
                marker="o",
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
            )

        ax_forest_mu.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.7)
        ax_forest_mu.set_yticks(y_positions)
        ax_forest_mu.set_yticklabels(gene_names, fontsize=8)
        ax_forest_mu.set_xlabel("Δμ (log FC) ± 95% CI", fontsize=10)
        ax_forest_mu.set_title("Amplitude Forest Plot", fontsize=12, fontweight="bold")
        ax_forest_mu.grid(axis="x", alpha=0.3)

    # Forest plot: Combined effect (signed)
    ax_forest_joint = fig.add_subplot(gs[2, 1])

    if len(forest_genes_df) > 0:
        y_positions = np.arange(len(forest_genes_df))
        gene_names = forest_genes_df["gene"].values[::-1]
        combined_z_vals = forest_genes_df["combined_z"].fillna(0).values[::-1]
        combined_z_ci_lower = forest_genes_df["combined_z_ci_lower"].values[::-1]
        combined_z_ci_upper = forest_genes_df["combined_z_ci_upper"].values[::-1]
        effect_types = forest_genes_df["effect_type"].values[::-1]

        for i, (_gene, effect, cz, ci_l, ci_h) in enumerate(
            zip(gene_names, effect_types, combined_z_vals, combined_z_ci_lower, combined_z_ci_upper)
        ):
            color = color_map.get(effect, "#BDC3C7")
            if not np.isnan(ci_l) and not np.isnan(ci_h):
                ax_forest_joint.plot([ci_l, ci_h], [i, i], color=color, linewidth=2, alpha=0.7)
            ax_forest_joint.scatter(
                [cz], [i], color=color, s=60, zorder=5, edgecolors="black", linewidths=0.5
            )

        ax_forest_joint.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.7)
        # Reference lines at ±2.77 ≈ 1.96*sqrt(2) for independent z-scores (rho=0)
        ax_forest_joint.axvline(
            2.77, color="green", linestyle="--", linewidth=1, alpha=0.5, label="p≈0.05 (ρ=0)"
        )
        ax_forest_joint.axvline(-2.77, color="green", linestyle="--", linewidth=1, alpha=0.5)

        ax_forest_joint.set_yticks(y_positions)
        ax_forest_joint.set_yticklabels(gene_names, fontsize=8)
        ax_forest_joint.set_xlabel("Combined Z (z_π + z_μ) ± 95% CI", fontsize=10)
        ax_forest_joint.set_title("Combined Effect Forest Plot", fontsize=12, fontweight="bold")
        ax_forest_joint.grid(axis="x", alpha=0.3)
        ax_forest_joint.legend(loc="lower right", fontsize=7)

    # Summary
    summary_text = "Summary: "
    for effect_type in ["penetrance", "amplitude", "both", "none"]:
        count = (df_valid["effect_type"] == effect_type).sum()
        summary_text += f"{effect_type.capitalize()}: {count}  "

    fig.text(0.5, 0.02, summary_text, ha="center", fontsize=9, style="italic")

    plt.suptitle(f"Zero-Inflated Decomposition: {group}", fontsize=16, fontweight="bold", y=0.98)

    if save:
        plt.savefig(save, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
        return None

    return fig
