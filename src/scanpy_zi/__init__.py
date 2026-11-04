"""scanpy-zi: Zero-inflated decomposition for single-cell differential expression.

This package provides methods to decompose differential gene expression into
penetrance (fraction of cells expressing) and amplitude (expression level among
expressors) effects.

Example
-------
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
>>> scanpy_zi.pl.zi_decomposition(adata)
"""

from __future__ import annotations

from .rank_genes import rank_genes_groups_zi

__version__ = "0.2.0"
__all__ = ["rank_genes_groups_zi", "pl", "__version__"]

# Lazy loading for plotting module to avoid matplotlib import on package load
_pl = None


def __getattr__(name: str):
    """Lazy load the plotting module."""
    global _pl
    if name == "pl":
        if _pl is None:
            from . import plotting as _pl_module

            _pl = _pl_module
        return _pl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_scanpy_extension() -> None:
    """Register ZI decomposition as a scanpy extension.

    After calling this function, you can use:
        >>> sc.tl.rank_genes_groups_zi(adata, groupby="cluster")
        >>> sc.pl.zi_decomposition(adata)

    Note: This modifies scanpy's namespace at runtime.
    """
    try:
        import scanpy as sc

        # Register tool
        sc.tl.rank_genes_groups_zi = rank_genes_groups_zi

        # Register plotting (this will trigger lazy load)
        from . import plotting as pl

        sc.pl.zi_decomposition = pl.zi_decomposition

    except ImportError as e:
        raise ImportError(
            "scanpy is required to register the extension. Install with: pip install scanpy"
        ) from e
