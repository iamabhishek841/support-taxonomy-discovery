"""UMAP dimensionality reduction and cluster scatter plots.

UMAP is the primary/preferred method. reduce_to_2d() falls back to
scikit-learn's t-SNE only if UMAP genuinely cannot be imported in the
current environment (e.g. its numba dependency being unavailable) --
this is reported honestly via the returned `method` string rather than
silently substituted, so nothing downstream claims a UMAP projection
that didn't actually happen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def reduce_to_2d(
    embeddings: np.ndarray,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> tuple[np.ndarray, str]:
    """Reduce high-dimensional embeddings to 2D.

    Tries UMAP first. Returns (coords_2d, method) where method is "umap"
    or "tsne" (fallback, used only when UMAP can't be imported/run here).
    """
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        return reducer.fit_transform(embeddings), "umap"
    except Exception as e:  # pragma: no cover - environment-dependent path
        import warnings

        from sklearn.manifold import TSNE

        warnings.warn(
            f"UMAP unavailable ({type(e).__name__}: {e}); falling back to sklearn TSNE for 2D projection.",
            stacklevel=2,
        )
        perplexity = min(30, max(5, embeddings.shape[0] // 100))
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
        return reducer.fit_transform(embeddings), "tsne"


def plot_clusters(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "Discovered clusters",
    save_path: str | Path | None = None,
    method: str = "umap",
):
    """Scatter-plot 2D coordinates colored by cluster label."""
    import matplotlib.pyplot as plt

    axis_prefix = "UMAP" if method == "umap" else "t-SNE"
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1], c=labels, cmap="tab20", s=4, alpha=0.6
    )
    ax.set_title(title)
    ax.set_xlabel(f"{axis_prefix}-1")
    ax.set_ylabel(f"{axis_prefix}-2")
    fig.colorbar(scatter, ax=ax, label="cluster")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    import time

    embeddings = np.load(RESULTS_DIR / "embeddings.npy")
    hdbscan_labels = np.load(RESULTS_DIR / "hdbscan_labels.npy")

    print(f"Reducing {embeddings.shape[0]:,} embeddings to 2D...")
    start = time.time()
    coords_2d, method = reduce_to_2d(embeddings)
    print(f"2D projection ({method}) done in {time.time() - start:.1f}s")

    np.save(RESULTS_DIR / "umap_coords.npy", coords_2d)
    plot_clusters(
        coords_2d,
        hdbscan_labels,
        title=f"Discovered support-issue clusters (HDBSCAN, {method.upper()} projection)",
        save_path=RESULTS_DIR / "umap_clusters.png",
        method=method,
    )
    print(f"Saved 2D coords and plot to {RESULTS_DIR} (method={method})")
