"""Generate and cache sentence-transformer embeddings for support utterances."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_CACHE_DIR = REPO_ROOT / "embeddings_cache"


def _cache_key(texts: list[str], model_name: str) -> str:
    """Content hash so a changed dataset/model never silently reuses a stale cache."""
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(str(len(texts)).encode("utf-8"))
    for t in texts:
        hasher.update(t.encode("utf-8"))
    return hasher.hexdigest()[:16]


def generate_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: Path = EMBEDDINGS_CACHE_DIR,
    batch_size: int = 64,
    use_cache: bool = True,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Embed a list of texts, caching the result to disk keyed by content hash.

    Returns an (n_texts, embedding_dim) float32 array.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    cache_dir = Path(cache_dir)
    key = _cache_key(texts, model_name)
    cache_path = cache_dir / f"{key}.npy"

    if use_cache and cache_path.exists():
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    ).astype(np.float32)

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)

    return embeddings


if __name__ == "__main__":
    import time

    from src.data_loader import load_bitext_dataset

    dataset = load_bitext_dataset()
    texts = dataset.texts["text"].tolist()

    start = time.time()
    embeddings = generate_embeddings(texts, show_progress_bar=True)
    elapsed = time.time() - start

    print(f"Embedded {len(texts):,} utterances with '{DEFAULT_MODEL_NAME}' in {elapsed:.1f}s")
    print(f"Embedding shape: {embeddings.shape} (dim={embeddings.shape[1]})")
