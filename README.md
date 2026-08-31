# support-taxonomy-discovery

Unsupervised discovery of natural issue-categories in customer-support
conversations -- no predefined taxonomy, no labels used during discovery.
Transformer embeddings + clustering (K-Means and HDBSCAN) + statistical
validation + LLM-generated cluster summaries via a local Ollama model.

## Why unsupervised discovery?

Support taxonomies are usually hand-defined by whoever set up the helpdesk,
and they drift out of date as products and customer language change. This
project asks a different question: if you throw out the existing labels
entirely and only look at the raw text of what customers actually wrote,
what categories does the data itself suggest? The dataset used here
(Bitext's customer-support dataset) happens to ship with human-defined
`category` / `intent` labels. Those labels are held out completely during
embedding and clustering, and are only brought back at the very end as an
external sanity check on whether the discovered structure resembles
something a human taxonomy would recognize -- not as a target to fit.

## Architecture

```
data_loader.py   -> download + clean the Bitext dataset, split out held-out labels
embeddings.py    -> sentence-transformer embeddings (all-MiniLM-L6-v2), cached to disk
clustering.py    -> K-Means (silhouette-selected k) and HDBSCAN, run and compared
validation.py    -> silhouette score + ARI/NMI against held-out labels (honest, not optimized for)
visualize.py     -> UMAP projection to 2D for plotting
label_clusters.py -> local Ollama model summarizes each cluster from representative examples
dashboard/       -> Streamlit app over precomputed results/
```

## How to run locally

### Prerequisites
- Python 3.11+ (this repo was developed against 3.14)
- [Ollama](https://ollama.com) installed and running locally, with a small
  model pulled (e.g. `ollama pull llama3.2`)
- **Windows users:** run the pipeline from **WSL** (or another Linux/macOS
  environment), not a native Windows Python install. `numba`
  (a dependency of `umap-learn`) ships compiled `.pyd` binaries that Windows
  **Smart App Control**, when enabled, blocks as unrecognized/low-reputation
  code -- this is a Windows OS security feature, not a bug in this project.
  `src/visualize.py` falls back to scikit-learn's t-SNE if UMAP can't be
  imported, so the pipeline still runs and clearly reports which method it
  used, but native WSL is the tested, reliable path.

### Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Run the pipeline
```bash
python -m src.data_loader      # downloads + caches the dataset (first run only)
python -m src.embeddings       # generates + caches embeddings
python -m src.clustering       # runs K-Means and HDBSCAN, picks a clustering
python -m src.validation       # reports silhouette + external validation
python -m src.label_clusters   # labels each cluster via local Ollama
```

### Run the dashboard
```bash
streamlit run dashboard/app.py
```

## Results

_Filled in as each pipeline stage is actually run -- see individual PRs for
the real numbers from real runs on the real dataset._

**Data & embeddings (branch `data-and-embeddings`):**
- Raw dataset: 26,872 rows (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`)
- After cleaning (whitespace-normalize, drop empty, dedupe exact-duplicate utterances): **24,635 unique utterances**
- Held-out labels: 11 `category` values, 27 `intent` values (never used until final validation)
- Embedding model: `all-MiniLM-L6-v2`, 384 dimensions
- Full-dataset embedding generation: ~45s of actual encoding time (13-14 it/s over 385 batches on CPU); first run also pays a one-time model download

**Clustering & validation (branch `clustering-and-validation`):**

K-Means silhouette-based k-search was swept over k = 4..68 (step 4) on the
full 24,635-utterance embedding set (silhouette scored on a 5,000-point
sample for speed):

| method  | clusters found | silhouette | ARI vs category (11) | NMI vs category | ARI vs intent (27) | NMI vs intent |
|---------|----------------|------------|-----------------------|------------------|----------------------|-----------------|
| K-Means | k=52 (chosen by silhouette peak) | 0.188 | 0.291 | 0.730 | 0.562 | 0.808 |
| HDBSCAN | 65 clusters, 6,153 noise pts (25.0%) | 0.256 (on the 18,482 non-noise points) | 0.510 | 0.780 | 0.715 | 0.860 |

**HDBSCAN is the chosen clustering** -- it beats K-Means on both internal
(silhouette) and external (ARI/NMI vs. the held-out human labels) metrics,
at the cost of leaving a quarter of the data unclustered as noise. The
external validation numbers say something real and worth being honest
about: NMI in the 0.78-0.86 range indicates the *discovered* clusters share
substantial mutual information with the *human-defined* categories/intents
-- i.e. the unsupervised pipeline is finding structure that meaningfully
resembles what a human taxonomy already captured, without ever seeing those
labels. ARI is lower (0.29-0.71) because ARI penalizes differences in
cluster *count* and *granularity* much more harshly than NMI does, and
HDBSCAN's 65 clusters vs. 11 human categories is a real granularity
mismatch, not necessarily a failure -- see Limitations below.

UMAP ran successfully on native WSL (Windows blocked it via Smart App
Control -- see Prerequisites) in 27.9s. The generated plot
(`results/umap_clusters.png`) shows many visually distinct, well-separated
clusters plus a diffuse "noise" region, consistent with the HDBSCAN numbers
above.

**LLM cluster labeling (branch `llm-cluster-labeling`):**

All 65 HDBSCAN clusters were labeled by a local `llama3.2` model via Ollama,
using the utterances closest to each cluster's centroid as context. Real run:
23.2s total (0.3-0.5s/cluster once the model was warm on GPU). A few actual
generated labels, largest clusters first:

| cluster | size | LLM-generated label | description |
|---|---|---|---|
| 23 | 1,743 | Delivery Address Support | Customer issues related to updating delivery address |
| 1  | 1,738 | Lost Invoice Inquiry | Customer seeks assistance locating a lost invoice |
| 0  | 975   | Newsletter Unsubscribing Assistance | Customer requests help with unsubscribing from company newsletter |
| 12 | 922   | Payment Notification Issues | Customer seeking assistance with notification of payment problems |
| 13 | 921   | Payment Method Inquiry | Customer seeking information on accepted payment options |
| 8  | 870   | Recovering User Account PIN | Customer support inquiries about recovering forgotten PIN codes |
| 17 | 768   | Tracking and Shipment | Customer inquiries about shipment status and timelines |

Full labels for all 65 clusters are in `results/cluster_labels.json`. Several
labels land on genuinely finer-grained distinctions than the original 11
categories (e.g. splitting "early termination fees" from "early exit
penalty" from "withdrawal charges" as separate clusters) -- plausible
evidence that the discovered taxonomy captures real structure the flat
human categories don't.

**Note on the local environment for this branch:** the machine's default
Ollama install (native WSL, not the Windows one) turned out to have an
incomplete/corrupted binary (missing the `llama-server` inference engine --
confirmed via a `500` error and directory listing), likely from an
interrupted install. It was repaired by re-downloading the official Ollama
release tarball to a user-owned directory (no `sudo` required) and running
it directly. This is an environment quirk specific to this machine, not a
code issue -- a normal `curl -fsSL https://ollama.com/install.sh | sh`
install works fine in general.

## Limitations

- **k-selection is data-driven but not sharply peaked.** The silhouette
  curve for K-Means is fairly flat across k=32..68 (0.174-0.188) -- there
  isn't one obviously "correct" k in that range, just a mild maximum at
  k=52. A different k in that band would be almost as statistically
  defensible. This is a real property of the embedding space, not a bug.
- **HDBSCAN's cluster count (65) is much finer-grained than the human
  taxonomy (11 categories / 27 intents).** That's plausibly *interesting*
  (the raw text may support finer distinctions than the hand-built
  taxonomy draws) but it also mechanically drags ARI down relative to NMI,
  since ARI is sensitive to partition-size mismatches. Don't read the ARI
  numbers alone as "the clustering is wrong" -- read them alongside NMI and
  the granularity mismatch.
- **25% of points are HDBSCAN noise.** Those utterances didn't fall into any
  dense region at the `min_cluster_size=30` setting used here; a looser
  setting would trade fewer noise points for coarser/less pure clusters.
- **`all-MiniLM-L6-v2` is a small, general-purpose sentence embedding
  model**, not fine-tuned on support-ticket language. A domain-tuned or
  larger embedding model would likely change both the clusters found and
  the external validation scores, in either direction.
- **What external validation does and doesn't tell you:** a high NMI means
  the discovered partition shares information with the human labels -- it
  does *not* mean the discovered categories are "correct" in any absolute
  sense, since the human taxonomy is itself just one particular way of
  carving up the same underlying issues. Treat it as a sanity check that
  the pipeline is finding *real*, human-recognizable structure, not as a
  score to be maximized.
