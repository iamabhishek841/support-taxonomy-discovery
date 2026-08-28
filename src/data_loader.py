"""Load and clean the Bitext customer-support dataset.

The dataset ships with `category` / `intent` labels. Those labels are
held out from the clustering pipeline entirely and only reattached at the
very end for external validation -- see src/validation.py. Nothing in this
module or in src/embeddings.py / src/clustering.py should read the label
columns for anything other than that final honesty check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_CACHE_PATH = DATA_DIR / "bitext_raw.parquet"
CLEAN_CACHE_PATH = DATA_DIR / "bitext_clean.parquet"


@dataclass
class SupportDataset:
    """Text to cluster, kept separate from the held-out labels."""

    texts: pd.DataFrame  # columns: id, text
    labels: pd.DataFrame  # columns: id, category, intent -- held out, validation only


def _download_raw(cache_path: Path = RAW_CACHE_PATH) -> pd.DataFrame:
    """Download the dataset from Hugging Face (or load from local cache)."""
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split="train")
    df = ds.to_pandas()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: normalize whitespace, drop empty/duplicate utterances."""
    df = df.copy()
    df["instruction"] = df["instruction"].astype(str).str.strip()
    df = df[df["instruction"].str.len() > 0]
    df = df.drop_duplicates(subset="instruction").reset_index(drop=True)
    df["id"] = df.index
    return df


def load_bitext_dataset(
    sample_size: int | None = None,
    seed: int = 42,
    force_download: bool = False,
    use_cache: bool = True,
) -> SupportDataset:
    """Load the Bitext dataset, clean it, and split text from held-out labels.

    Args:
        sample_size: if set, return a random sample of this many rows
            (useful for fast local iteration / tests).
        seed: random seed for sampling.
        force_download: bypass the raw-download cache and re-fetch from HF.
        use_cache: if True (default) reuse a previously cleaned parquet
            cache instead of re-cleaning every call.
    """
    if use_cache and not force_download and CLEAN_CACHE_PATH.exists() and sample_size is None:
        clean = pd.read_parquet(CLEAN_CACHE_PATH)
    else:
        raw = _download_raw(RAW_CACHE_PATH if not force_download else DATA_DIR / "_never_cache.parquet")
        clean = _clean(raw)
        if sample_size is None:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            clean.to_parquet(CLEAN_CACHE_PATH)

    if sample_size is not None and sample_size < len(clean):
        clean = clean.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        clean["id"] = clean.index

    texts = clean[["id", "instruction"]].rename(columns={"instruction": "text"})
    labels = clean[["id", "category", "intent"]]
    return SupportDataset(texts=texts, labels=labels)


if __name__ == "__main__":
    import time

    start = time.time()
    dataset = load_bitext_dataset()
    elapsed = time.time() - start

    print(f"Loaded {len(dataset.texts):,} cleaned utterances in {elapsed:.1f}s")
    print(f"Categories ({dataset.labels['category'].nunique()}): "
          f"{sorted(dataset.labels['category'].unique().tolist())}")
    print(f"Intents: {dataset.labels['intent'].nunique()} unique")
