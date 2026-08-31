"""Load precomputed pipeline results for the dashboard.

Deliberately does NOT re-run any part of the discovery pipeline or require
Ollama -- everything here reads artifacts already committed under
results/, produced by branches 1-3 (data-and-embeddings,
clustering-and-validation, llm-cluster-labeling).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


@dataclass
class DashboardData:
    texts: pd.DataFrame  # id, text
    hdbscan_labels: np.ndarray
    umap_coords: np.ndarray
    clustering_summary: dict
    cluster_labels: dict  # {cluster_id (int): {label, description, size, examples}}


def results_available() -> bool:
    required = [
        "texts.parquet",
        "hdbscan_labels.npy",
        "umap_coords.npy",
        "clustering_summary.json",
        "cluster_labels.json",
    ]
    return all((RESULTS_DIR / name).exists() for name in required)


def load_dashboard_data() -> DashboardData:
    texts = pd.read_parquet(RESULTS_DIR / "texts.parquet")
    hdbscan_labels = np.load(RESULTS_DIR / "hdbscan_labels.npy")
    umap_coords = np.load(RESULTS_DIR / "umap_coords.npy")

    with open(RESULTS_DIR / "clustering_summary.json") as f:
        clustering_summary = json.load(f)

    with open(RESULTS_DIR / "cluster_labels.json") as f:
        raw_labels = json.load(f)
    cluster_labels = {int(k): v for k, v in raw_labels.items()}

    return DashboardData(
        texts=texts,
        hdbscan_labels=hdbscan_labels,
        umap_coords=umap_coords,
        clustering_summary=clustering_summary,
        cluster_labels=cluster_labels,
    )
