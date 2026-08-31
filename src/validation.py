"""Statistical validation of a clustering: internal (silhouette) and external.

External validation compares the discovered clusters against the Bitext
dataset's real `category`/`intent` labels -- held out during clustering and
brought back here ONLY to check, honestly, whether the unsupervised result
resembles human-defined structure. A modest score is reported as-is; this
metric is a sanity check, not something the pipeline optimizes for.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

DEFAULT_SILHOUETTE_SAMPLE_SIZE = 5000


def compute_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_size: int = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
    random_state: int = 42,
    exclude_noise: bool = True,
) -> float:
    """Silhouette score for a clustering. Returns NaN if fewer than 2 clusters."""
    labels = np.asarray(labels)
    mask = labels != -1 if exclude_noise else np.ones(len(labels), dtype=bool)

    if mask.sum() < 2 or len(set(labels[mask].tolist())) < 2:
        return float("nan")

    sample = min(sample_size, int(mask.sum()))
    return float(
        silhouette_score(embeddings[mask], labels[mask], sample_size=sample, random_state=random_state)
    )


def external_validation(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    exclude_noise: bool = True,
) -> dict:
    """Compare discovered clusters to real (held-out) labels.

    Uses Adjusted Rand Index and Normalized Mutual Information -- both are
    appropriate for comparing partitions with different label sets/counts
    and both are invariant to label permutation.
    """
    cluster_labels = np.asarray(cluster_labels)
    true_labels = np.asarray(true_labels)

    if exclude_noise:
        mask = cluster_labels != -1
        cluster_labels = cluster_labels[mask]
        true_labels = true_labels[mask]

    return {
        "ari": float(adjusted_rand_score(true_labels, cluster_labels)),
        "nmi": float(normalized_mutual_info_score(true_labels, cluster_labels)),
        "n_evaluated": int(len(cluster_labels)),
        "n_total": int(len(np.asarray(cluster_labels)) if not exclude_noise else len(mask)),
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

    with open(RESULTS_DIR / "clustering_summary.json") as f:
        summary = json.load(f)

    embeddings = np.load(RESULTS_DIR / "embeddings.npy")
    kmeans_labels = np.load(RESULTS_DIR / "kmeans_labels.npy")
    hdbscan_labels = np.load(RESULTS_DIR / "hdbscan_labels.npy")

    import pandas as pd

    labels_df = pd.read_parquet(RESULTS_DIR / "labels.parquet")
    category = labels_df["category"].to_numpy()
    intent = labels_df["intent"].to_numpy()

    for method_name, labels in [("kmeans", kmeans_labels), ("hdbscan", hdbscan_labels)]:
        silhouette = compute_silhouette(embeddings, labels)
        vs_category = external_validation(labels, category)
        vs_intent = external_validation(labels, intent)
        print(f"\n=== {method_name} ===")
        print(f"silhouette: {silhouette:.4f}")
        print(f"vs category (11 classes): ARI={vs_category['ari']:.4f} NMI={vs_category['nmi']:.4f} "
              f"(n={vs_category['n_evaluated']:,})")
        print(f"vs intent (27 classes):   ARI={vs_intent['ari']:.4f} NMI={vs_intent['nmi']:.4f} "
              f"(n={vs_intent['n_evaluated']:,})")

        summary[method_name]["silhouette_full"] = silhouette
        summary[method_name]["external_validation"] = {"vs_category": vs_category, "vs_intent": vs_intent}

    with open(RESULTS_DIR / "clustering_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nUpdated {RESULTS_DIR / 'clustering_summary.json'}")
