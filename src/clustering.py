"""Cluster support-utterance embeddings with K-Means and HDBSCAN.

Both methods are run and compared -- K-Means requires picking k up front
(done here via silhouette-score search over a range of k), HDBSCAN infers
the number of clusters from density and can mark points as noise (-1).
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DEFAULT_K_RANGE = range(4, 71, 4)
DEFAULT_SILHOUETTE_SAMPLE_SIZE = 5000


def select_k_by_silhouette(
    embeddings: np.ndarray,
    k_range: range = DEFAULT_K_RANGE,
    sample_size: int = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
    random_state: int = 42,
) -> tuple[int, dict[int, float]]:
    """Fit K-Means for each k in k_range, score with a sampled silhouette score.

    Returns (best_k, {k: silhouette_score, ...}).
    """
    n = embeddings.shape[0]
    scores: dict[int, float] = {}
    for k in k_range:
        if k >= n:
            continue
        labels = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit_predict(embeddings)
        sample = min(sample_size, n)
        scores[k] = float(
            silhouette_score(embeddings, labels, sample_size=sample, random_state=random_state)
        )
    if not scores:
        raise ValueError(f"No valid k in {k_range} for {n} samples")
    best_k = max(scores, key=scores.get)
    return best_k, scores


def run_kmeans(
    embeddings: np.ndarray,
    k: int | None = None,
    k_range: range = DEFAULT_K_RANGE,
    random_state: int = 42,
) -> tuple[np.ndarray, int, dict[int, float] | None]:
    """Run K-Means, selecting k by silhouette search if not given explicitly.

    Returns (labels, chosen_k, k_search_scores_or_None).
    """
    scores = None
    if k is None:
        k, scores = select_k_by_silhouette(embeddings, k_range=k_range, random_state=random_state)
    labels = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit_predict(embeddings)
    return labels, k, scores


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 30,
    min_samples: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """Run HDBSCAN. Returns (labels, n_clusters_found, n_noise_points).

    Noise points are labeled -1 by HDBSCAN and excluded from n_clusters_found.
    """
    import hdbscan

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    return labels, n_clusters, n_noise


if __name__ == "__main__":
    import json
    import time
    from pathlib import Path

    from src.data_loader import load_bitext_dataset
    from src.embeddings import generate_embeddings

    RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_bitext_dataset()
    embeddings = generate_embeddings(dataset.texts["text"].tolist())
    print(f"Loaded {embeddings.shape[0]:,} embeddings (dim={embeddings.shape[1]})")

    print("Running K-Means with silhouette-based k selection...")
    start = time.time()
    kmeans_labels, kmeans_k, k_scores = run_kmeans(embeddings)
    kmeans_silhouette = k_scores[kmeans_k]
    print(f"K-Means: chosen k={kmeans_k}, silhouette={kmeans_silhouette:.4f} ({time.time() - start:.1f}s)")
    print(f"  k search scores: { {k: round(v, 4) for k, v in k_scores.items()} }")

    print("Running HDBSCAN...")
    start = time.time()
    hdbscan_labels, hdbscan_n_clusters, hdbscan_n_noise = run_hdbscan(embeddings)
    print(
        f"HDBSCAN: found {hdbscan_n_clusters} clusters, "
        f"{hdbscan_n_noise:,} noise points ({time.time() - start:.1f}s)"
    )

    np.save(RESULTS_DIR / "kmeans_labels.npy", kmeans_labels)
    np.save(RESULTS_DIR / "hdbscan_labels.npy", hdbscan_labels)
    np.save(RESULTS_DIR / "embeddings.npy", embeddings)
    dataset.texts.to_parquet(RESULTS_DIR / "texts.parquet")
    dataset.labels.to_parquet(RESULTS_DIR / "labels.parquet")

    summary = {
        "n_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "kmeans": {
            "k": kmeans_k,
            "silhouette": kmeans_silhouette,
            "k_search_scores": {str(k): v for k, v in k_scores.items()},
        },
        "hdbscan": {
            "n_clusters": hdbscan_n_clusters,
            "n_noise": hdbscan_n_noise,
        },
    }
    with open(RESULTS_DIR / "clustering_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved cluster labels, embeddings, and summary to {RESULTS_DIR}")
