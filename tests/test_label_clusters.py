import json

import numpy as np
import pytest

from src.label_clusters import (
    label_all_clusters,
    label_cluster,
    select_representative_examples,
)


def _synthetic_clustered_data():
    rng = np.random.default_rng(0)
    cluster_0 = rng.normal(loc=[0, 0], scale=0.1, size=(5, 2))
    cluster_1 = rng.normal(loc=[10, 10], scale=0.1, size=(5, 2))
    noise = rng.normal(loc=[5, -5], scale=0.1, size=(2, 2))
    embeddings = np.vstack([cluster_0, cluster_1, noise]).astype(np.float32)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1, -1])
    texts = [f"utterance {i}" for i in range(len(labels))]
    return embeddings, labels, texts


def test_select_representative_examples_returns_closest_to_centroid():
    embeddings, labels, texts = _synthetic_clustered_data()

    examples = select_representative_examples(embeddings, labels, texts, cluster_id=0, n_examples=3)
    assert len(examples) == 3
    assert all(e in texts[:5] for e in examples)


def test_select_representative_examples_handles_missing_cluster():
    embeddings, labels, texts = _synthetic_clustered_data()
    examples = select_representative_examples(embeddings, labels, texts, cluster_id=99)
    assert examples == []


class _FakeOllamaClient:
    """Stand-in for ollama.Client -- no real Ollama server needed for tests."""

    responses = None  # list of dicts or exceptions, consumed in order
    calls = 0

    def __init__(self, host):
        self.host = host

    def chat(self, model, messages, format):
        response = _FakeOllamaClient.responses[_FakeOllamaClient.calls]
        _FakeOllamaClient.calls += 1
        if isinstance(response, Exception):
            raise response
        return {"message": {"content": json.dumps(response)}}


@pytest.fixture(autouse=True)
def _reset_fake_client():
    _FakeOllamaClient.calls = 0
    _FakeOllamaClient.responses = None
    yield


@pytest.fixture
def fake_ollama(monkeypatch):
    import ollama

    monkeypatch.setattr(ollama, "Client", _FakeOllamaClient)
    return _FakeOllamaClient


def test_label_cluster_success(fake_ollama):
    fake_ollama.responses = [{"label": "Refund requests", "description": "Customers asking about refunds."}]

    result = label_cluster(["I want a refund"], max_retries=3, retry_delay=0)

    assert result["label"] == "Refund requests"
    assert "refund" in result["description"].lower()
    assert fake_ollama.calls == 1


def test_label_cluster_retries_then_succeeds(fake_ollama):
    fake_ollama.responses = [
        ConnectionError("ollama not reachable"),
        {"label": "Order cancellation", "description": "Customers cancelling an order."},
    ]

    result = label_cluster(["cancel my order"], max_retries=3, retry_delay=0)

    assert result["label"] == "Order cancellation"
    assert fake_ollama.calls == 2


def test_label_cluster_raises_clear_error_after_exhausting_retries(fake_ollama):
    fake_ollama.responses = [ConnectionError("ollama not reachable")] * 3

    with pytest.raises(RuntimeError, match="Could not get a label from Ollama"):
        label_cluster(["cancel my order"], max_retries=3, retry_delay=0)

    assert fake_ollama.calls == 3


def test_label_all_clusters_excludes_noise_by_default(fake_ollama):
    embeddings, labels, texts = _synthetic_clustered_data()
    fake_ollama.responses = [
        {"label": "Cluster A", "description": "First cluster."},
        {"label": "Cluster B", "description": "Second cluster."},
    ]

    result = label_all_clusters(embeddings, labels, texts, retry_delay=0)

    assert set(result.keys()) == {0, 1}
    assert result[0]["label"] == "Cluster A"
    assert result[0]["size"] == 5
    assert result[1]["label"] == "Cluster B"
