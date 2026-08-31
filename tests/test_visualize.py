import numpy as np

from src.visualize import plot_clusters, reduce_to_2d


def test_reduce_to_2d_shape():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(60, 16)).astype(np.float32)

    coords, method = reduce_to_2d(embeddings, n_neighbors=5)
    assert coords.shape == (60, 2)
    assert method in ("umap", "tsne")


def test_plot_clusters_saves_file(tmp_path):
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(30, 2))
    labels = rng.integers(0, 3, size=30)
    save_path = tmp_path / "plot.png"

    fig = plot_clusters(coords, labels, save_path=save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0
    import matplotlib.pyplot as plt

    plt.close(fig)
