import numpy as np

from src.clustering import run_hdbscan, run_kmeans, select_k_by_silhouette


def _blob_embeddings(n_per_cluster=20, n_clusters=3, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10, 10, size=(n_clusters, dim))
    points = []
    for c in centers:
        points.append(c + rng.normal(scale=0.3, size=(n_per_cluster, dim)))
    return np.vstack(points).astype(np.float32)


def test_select_k_by_silhouette_finds_true_cluster_count():
    embeddings = _blob_embeddings(n_clusters=3)
    best_k, scores = select_k_by_silhouette(embeddings, k_range=range(2, 8), sample_size=1000)

    assert best_k == 3
    assert set(scores.keys()) == set(range(2, 8))
    assert all(-1.0 <= v <= 1.0 for v in scores.values())


def test_run_kmeans_with_explicit_k_skips_search():
    embeddings = _blob_embeddings(n_clusters=3)
    labels, k, scores = run_kmeans(embeddings, k=3)

    assert k == 3
    assert scores is None
    assert len(set(labels.tolist())) == 3
    assert len(labels) == len(embeddings)


def test_run_kmeans_without_k_runs_selection():
    embeddings = _blob_embeddings(n_clusters=3)
    labels, k, scores = run_kmeans(embeddings, k_range=range(2, 8))

    assert k == 3
    assert scores is not None
    assert len(labels) == len(embeddings)


def test_run_hdbscan_finds_dense_clusters():
    embeddings = _blob_embeddings(n_clusters=3, n_per_cluster=30)
    labels, n_clusters, n_noise = run_hdbscan(embeddings, min_cluster_size=10)

    assert n_clusters >= 1
    assert n_noise >= 0
    assert len(labels) == len(embeddings)
