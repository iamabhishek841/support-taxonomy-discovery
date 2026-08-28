import numpy as np
import pytest

from src.embeddings import _cache_key, generate_embeddings


class _FakeSentenceTransformer:
    """Stand-in for sentence_transformers.SentenceTransformer in tests -- no
    model download, no torch inference. Encodes deterministically from text
    length so we can assert on shape/values without a real model."""

    call_count = 0

    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True):
        _FakeSentenceTransformer.call_count += 1
        return np.array([[float(len(t)), 1.0, 2.0] for t in texts])


@pytest.fixture(autouse=True)
def _reset_call_count():
    _FakeSentenceTransformer.call_count = 0
    yield


@pytest.fixture
def fake_model(monkeypatch):
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)


def test_generate_embeddings_shape(fake_model, tmp_path):
    texts = ["hello", "a longer piece of text"]
    emb = generate_embeddings(texts, cache_dir=tmp_path)

    assert emb.shape == (2, 3)
    assert emb.dtype == np.float32
    assert _FakeSentenceTransformer.call_count == 1


def test_generate_embeddings_uses_cache_on_second_call(fake_model, tmp_path):
    texts = ["hello", "world"]
    generate_embeddings(texts, cache_dir=tmp_path)
    generate_embeddings(texts, cache_dir=tmp_path)

    assert _FakeSentenceTransformer.call_count == 1


def test_generate_embeddings_empty_list_returns_empty_array(fake_model, tmp_path):
    emb = generate_embeddings([], cache_dir=tmp_path)
    assert emb.shape == (0, 0)
    assert _FakeSentenceTransformer.call_count == 0


def test_cache_key_differs_for_different_texts_or_model():
    key_a = _cache_key(["hello"], "model-a")
    key_b = _cache_key(["world"], "model-a")
    key_c = _cache_key(["hello"], "model-b")

    assert key_a != key_b
    assert key_a != key_c
