"""Generate human-readable labels for discovered clusters via a local Ollama model.

For each cluster, the utterances closest to the cluster centroid (in
embedding space) are sampled and sent to a local Ollama model, which is
asked to return a short label and one-sentence description. Ollama being
slow to respond is handled with retries; Ollama being unreachable entirely
raises a clear RuntimeError rather than silently producing empty labels.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "llama3.2"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_N_EXAMPLES = 8
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0


def select_representative_examples(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: list[str],
    cluster_id: int,
    n_examples: int = DEFAULT_N_EXAMPLES,
) -> list[str]:
    """Pick the n_examples utterances closest to the cluster's centroid."""
    mask = labels == cluster_id
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []

    cluster_embeddings = embeddings[idx]
    centroid = cluster_embeddings.mean(axis=0)
    dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    order = np.argsort(dists)[: min(n_examples, len(idx))]
    chosen = idx[order]
    return [texts[i] for i in chosen]


def _build_prompt(examples: list[str]) -> str:
    bullet_examples = "\n".join(f"- {e}" for e in examples)
    return (
        "You are analyzing real customer-support messages that a clustering "
        "algorithm has grouped together because they are semantically similar. "
        "Here are representative examples from one cluster:\n\n"
        f"{bullet_examples}\n\n"
        "Respond with ONLY a JSON object with two keys: "
        '"label" (a short 3-6 word category name) and '
        '"description" (one sentence describing what this cluster of issues is about). '
        "Do not include any other text."
    )


def label_cluster(
    examples: list[str],
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> dict:
    """Ask a local Ollama model to label one cluster from its representative examples.

    Raises RuntimeError with a clear message if Ollama is unreachable/fails
    to return a usable response after max_retries attempts.
    """
    import ollama

    client = ollama.Client(host=host)
    prompt = _build_prompt(examples)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            content = response["message"]["content"]
            parsed = json.loads(content)
            return {
                "label": str(parsed.get("label", "")).strip() or "Unlabeled",
                "description": str(parsed.get("description", "")).strip(),
            }
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(retry_delay)

    raise RuntimeError(
        f"Could not get a label from Ollama model '{model}' at {host} after "
        f"{max_retries} attempts. Is Ollama running (`ollama serve`) and is the "
        f"model pulled (`ollama pull {model}`)? Last error: {last_error}"
    )


def label_all_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: list[str],
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    n_examples: int = DEFAULT_N_EXAMPLES,
    exclude_noise: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    on_cluster_labeled=None,
) -> dict[int, dict]:
    """Label every cluster in `labels`.

    Returns {cluster_id: {"label": ..., "description": ..., "size": ..., "examples": [...]}}.
    If given, on_cluster_labeled(cluster_id, info) is called after each cluster is
    labeled, so callers can report progress or checkpoint results incrementally.
    """
    unique_ids = sorted(set(int(x) for x in labels.tolist()))
    if exclude_noise:
        unique_ids = [c for c in unique_ids if c != -1]

    results = {}
    for cluster_id in unique_ids:
        examples = select_representative_examples(embeddings, labels, texts, cluster_id, n_examples)
        info = label_cluster(
            examples, model=model, host=host, max_retries=max_retries, retry_delay=retry_delay
        )
        size = int((labels == cluster_id).sum())
        results[cluster_id] = {
            "label": info["label"],
            "description": info["description"],
            "size": size,
            "examples": examples,
        }
        if on_cluster_labeled is not None:
            on_cluster_labeled(cluster_id, results[cluster_id])
    return results


if __name__ == "__main__":
    import pandas as pd

    from src.embeddings import generate_embeddings

    RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

    texts_df = pd.read_parquet(RESULTS_DIR / "texts.parquet")
    texts = texts_df["text"].tolist()
    hdbscan_labels = np.load(RESULTS_DIR / "hdbscan_labels.npy")
    embeddings = generate_embeddings(texts)  # hits the disk cache from earlier branches

    n_clusters = len(set(int(x) for x in hdbscan_labels.tolist()) - {-1})
    print(f"Labeling {n_clusters} HDBSCAN clusters via Ollama ({DEFAULT_MODEL})...", flush=True)
    start = time.time()

    cluster_info_partial: dict[int, dict] = {}

    def _checkpoint(cluster_id, info):
        cluster_info_partial[cluster_id] = info
        elapsed = time.time() - start
        print(
            f"  [{elapsed:6.1f}s] cluster {cluster_id} (n={info['size']}): "
            f"{info['label']} -- {info['description']}",
            flush=True,
        )
        current = {str(k): v for k, v in sorted(cluster_info_partial.items())}
        with open(RESULTS_DIR / "cluster_labels.json", "w") as f:
            json.dump(current, f, indent=2)

    cluster_info = label_all_clusters(
        embeddings, hdbscan_labels, texts, on_cluster_labeled=_checkpoint
    )
    print(f"Done in {time.time() - start:.1f}s", flush=True)
    print(f"Saved cluster labels to {RESULTS_DIR / 'cluster_labels.json'}")
    print(f"Saved cluster labels to {RESULTS_DIR / 'cluster_labels.json'}")
