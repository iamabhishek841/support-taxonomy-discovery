import numpy as np

from src.validation import compute_silhouette, external_validation


def _blob_embeddings(n_per_cluster=20, n_clusters=3, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10, 10, size=(n_clusters, dim))
    points = []
    for c in centers:
        points.append(c + rng.normal(scale=0.3, size=(n_per_cluster, dim)))
    return np.vstack(points).astype(np.float32)


def test_compute_silhouette_high_for_well_separated_clusters():
    embeddings = _blob_embeddings(n_clusters=3)
    labels = np.repeat([0, 1, 2], 20)

    score = compute_silhouette(embeddings, labels, sample_size=60)
    assert score > 0.5


def test_compute_silhouette_returns_nan_for_single_cluster():
    embeddings = _blob_embeddings(n_clusters=1)
    labels = np.zeros(20, dtype=int)

    score = compute_silhouette(embeddings, labels)
    assert np.isnan(score)


def test_compute_silhouette_excludes_noise_by_default():
    embeddings = _blob_embeddings(n_clusters=2)
    labels = np.repeat([0, 1], 20)
    labels_with_noise = labels.copy()
    labels_with_noise[:5] = -1

    score_clean = compute_silhouette(embeddings, labels, sample_size=40)
    score_noisy = compute_silhouette(embeddings, labels_with_noise, sample_size=40)
    assert abs(score_clean - score_noisy) < 0.3


def test_external_validation_perfect_agreement():
    cluster_labels = np.array([0, 0, 1, 1, 2, 2])
    true_labels = np.array(["a", "a", "b", "b", "c", "c"])

    result = external_validation(cluster_labels, true_labels)
    assert result["ari"] == 1.0
    assert result["nmi"] == 1.0
    assert result["n_evaluated"] == 6


def test_external_validation_no_agreement_is_near_zero():
    rng = np.random.default_rng(1)
    cluster_labels = rng.integers(0, 5, size=200)
    true_labels = rng.integers(0, 5, size=200)

    result = external_validation(cluster_labels, true_labels)
    assert result["ari"] < 0.2


def test_external_validation_excludes_noise_points():
    cluster_labels = np.array([-1, -1, 0, 0, 1, 1])
    true_labels = np.array(["x", "y", "a", "a", "b", "b"])

    result = external_validation(cluster_labels, true_labels)
    assert result["n_evaluated"] == 4
    assert result["ari"] == 1.0
