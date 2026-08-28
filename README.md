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

## Limitations

_Filled in progressively -- covers sentence-transformer model choice,
k-selection sensitivity, and what the external validation score does and
doesn't tell you._
