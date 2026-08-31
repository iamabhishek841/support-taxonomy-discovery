"""Compare HDBSCAN parameter sets to check whether the original choice
(min_cluster_size=30, default min_samples) over-segments the data.

Motivation: the original HDBSCAN result (65 clusters, 25% noise) had a much
lower ARI (0.510) than NMI (0.780) against the held-out category labels.
ARI penalizes partition-count/granularity mismatches harshly, so a large
ARI/NMI gap combined with a high noise fraction is a signal -- not proof --
of over-segmentation: many small, tight clusters of near-duplicate phrasing
that a human taxonomy would lump into one category. This script tests
coarser parameter sets (larger min_cluster_size) to see whether a less
fine-grained clustering narrows that gap and reduces noise while still
beating the K-Means baseline, then reports the real numbers for each so the
choice is auditable rather than asserted.
"""

from __future__ import annotations

import time

from src.clustering import run_hdbscan
from src.validation import compute_silhouette, external_validation

PARAM_SETS = [
    {"name": "baseline (min_cluster_size=30)", "min_cluster_size": 30, "min_samples": None},
    {"name": "coarser A (min_cluster_size=50, min_samples=15)", "min_cluster_size": 50, "min_samples": 15},
    {"name": "coarser B (min_cluster_size=75, min_samples=20)", "min_cluster_size": 75, "min_samples": 20},
    {"name": "coarser C (min_cluster_size=100, min_samples=30)", "min_cluster_size": 100, "min_samples": 30},
]


def tune(embeddings, category_labels, param_sets: list[dict] = PARAM_SETS) -> list[dict]:
    """Run HDBSCAN for each param set, return a list of result dicts."""
    results = []
    for params in param_sets:
        start = time.time()
        labels, n_clusters, n_noise = run_hdbscan(
            embeddings,
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
        )
        elapsed = time.time() - start

        silhouette = compute_silhouette(embeddings, labels)
        ext = external_validation(labels, category_labels)

        results.append(
            {
                "name": params["name"],
                "min_cluster_size": params["min_cluster_size"],
                "min_samples": params["min_samples"],
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_fraction": n_noise / len(labels),
                "silhouette": silhouette,
                "ari": ext["ari"],
                "nmi": ext["nmi"],
                "elapsed_s": elapsed,
                "labels": labels,
            }
        )
    return results


if __name__ == "__main__":
    import pandas as pd

    from src.data_loader import load_bitext_dataset
    from src.embeddings import generate_embeddings

    dataset = load_bitext_dataset()
    embeddings = generate_embeddings(dataset.texts["text"].tolist())
    category = dataset.labels["category"].to_numpy()

    print(f"Tuning HDBSCAN on {embeddings.shape[0]:,} embeddings across {len(PARAM_SETS)} param sets...\n")
    results = tune(embeddings, category)

    header = f"{'config':<45} {'k':>4} {'noise%':>8} {'silhouette':>11} {'ARI':>7} {'NMI':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<45} {r['n_clusters']:>4} {100 * r['noise_fraction']:>7.1f}% "
            f"{r['silhouette']:>11.4f} {r['ari']:>7.4f} {r['nmi']:>7.4f}"
        )

    out_rows = [{k: v for k, v in r.items() if k != "labels"} for r in results]
    pd.DataFrame(out_rows).to_csv("results/hdbscan_tuning.csv", index=False)
    print("\nSaved comparison table to results/hdbscan_tuning.csv")
