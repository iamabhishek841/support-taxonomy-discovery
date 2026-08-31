"""Support Taxonomy Discovery -- Streamlit dashboard.

Loads precomputed results from results/ (produced by branches 1-3 of the
pipeline) and never re-runs embeddings, clustering, or Ollama labeling
itself -- see dashboard/data.py. This is what makes it deployable to
Streamlit Community Cloud without any local model or GPU dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from theme import (  # noqa: E402
    ACCENT_EMERALD,
    BG_SURFACE,
    CSS,
    FONT_BODY,
    NOISE_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    qualitative_cluster_colors,
    score_accent,
)

from data import load_dashboard_data, results_available  # noqa: E402

st.set_page_config(
    page_title="Support Taxonomy Discovery",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


def kpi_card(
    label: str, value: str, accent: str = "neutral", sub: str | None = None, sub_accent: str = "neutral"
) -> str:
    sub_html = f'<div class="kpi-sub {sub_accent}">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {accent}">{value}</div>{sub_html}</div>'
    )


if not results_available():
    st.markdown(CSS, unsafe_allow_html=True)
    missing_panel = st.container(border=True)
    missing_panel.markdown("## Results not found")
    missing_panel.markdown(
        '<p style="color:#9AA3B8;">This dashboard reads precomputed artifacts from '
        "<code>results/</code> (dataset, embeddings, cluster labels, LLM-generated "
        "summaries). Run the pipeline first:</p>",
        unsafe_allow_html=True,
    )
    missing_panel.code(
        "python -m src.data_loader\n"
        "python -m src.embeddings\n"
        "python -m src.clustering\n"
        "python -m src.validation\n"
        "python -m src.visualize\n"
        "python -m src.label_clusters",
        language="bash",
    )
    st.stop()

data = load_dashboard_data()
summary = data.clustering_summary
hdb = summary["hdbscan"]
ext_cat = hdb["external_validation"]["vs_category"]
ext_intent = hdb["external_validation"]["vs_intent"]

n_samples = summary["n_samples"]
n_clusters = hdb["n_clusters"]
n_noise = hdb["n_noise"]
noise_pct = 100 * n_noise / n_samples
silhouette = hdb["silhouette_full"]
nmi_category = ext_cat["nmi"]

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.markdown("### Support Taxonomy Discovery")
    st.markdown(
        '<p style="color:#9AA3B8; font-size:0.85rem;">'
        "Unsupervised issue-category discovery from raw support text -- "
        "no predefined taxonomy, no labels used during clustering."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**Method**")
    st.markdown(
        f'<span class="mono-num" style="color:{TEXT_PRIMARY};">HDBSCAN</span> '
        f'<span style="color:{TEXT_SECONDARY};">(chosen over K-Means -- see README)</span>',
        unsafe_allow_html=True,
    )
    st.markdown("**Embedding model**")
    st.markdown(
        f'<span class="mono-num" style="color:{TEXT_PRIMARY};">all-MiniLM-L6-v2</span> '
        f'<span class="mono-num" style="color:{TEXT_SECONDARY};">({summary["embedding_dim"]}d)</span>',
        unsafe_allow_html=True,
    )
    st.markdown("**Cluster labels**")
    st.markdown(
        '<span style="color:#E8EBF2;">Local Ollama (llama3.2)</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        f'<p style="color:{TEXT_SECONDARY}; font-size:0.8rem;">'
        "External validation compares discovered clusters to the dataset's "
        "held-out human category/intent labels -- as a sanity check only, "
        "never used during clustering."
        "</p>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ header --
st.markdown("# Support Taxonomy Discovery")
st.markdown(
    '<div class="subtitle">Clusters discovered from raw customer-support text via '
    "transformer embeddings + HDBSCAN -- no predefined taxonomy.</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------- KPI row --
sil_accent = score_accent(silhouette, warn_below=0.15, good_above=0.5)
nmi_accent = score_accent(nmi_category, warn_below=0.4, good_above=0.7)

kpi_html = '<div class="kpi-row">'
kpi_html += kpi_card("Conversations Processed", f"{n_samples:,}", "neutral")
kpi_html += kpi_card(
    "Clusters Discovered",
    f"{n_clusters}",
    "accent-indigo",
    sub=f"{n_noise:,} noise pts ({noise_pct:.1f}%)",
    sub_accent="accent-amber" if noise_pct > 15 else "neutral",
)
kpi_html += kpi_card(
    "Silhouette Score",
    f"{silhouette:.3f}",
    sil_accent,
    sub="internal cluster quality",
    sub_accent="neutral",
)
kpi_html += kpi_card(
    "External Validation (NMI)",
    f"{nmi_category:.3f}",
    nmi_accent,
    sub="vs. 11 held-out categories",
    sub_accent="neutral",
)
kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)

# ------------------------------------------------------- cluster explorer --
# Build once, used by both the scatter highlight and the explorer panel below.
sorted_clusters = sorted(data.cluster_labels.items(), key=lambda kv: -kv[1]["size"])
cluster_ids_sorted = [cid for cid, _ in sorted_clusters]

if "selected_cluster" not in st.session_state:
    st.session_state.selected_cluster = cluster_ids_sorted[0]

# ----------------------------------------------------------- UMAP scatter --
umap_panel = st.container(border=True)
umap_panel.markdown("### Cluster map (UMAP projection)")
umap_panel.markdown(
    f'<p style="color:{TEXT_SECONDARY}; font-size:0.85rem; margin-top:-8px;">'
    "Each point is one support conversation. Select a cluster below to highlight it here."
    "</p>",
    unsafe_allow_html=True,
)

unique_cluster_ids = sorted(c for c in set(int(x) for x in data.hdbscan_labels.tolist()) if c != -1)
palette = qualitative_cluster_colors(len(unique_cluster_ids))
color_map = dict(zip(unique_cluster_ids, palette))

selected = st.session_state.selected_cluster
labels_arr = data.hdbscan_labels
coords = data.umap_coords

point_colors = []
point_opacity = []
for lab in labels_arr:
    lab = int(lab)
    if lab == selected:
        point_colors.append(ACCENT_EMERALD)
        point_opacity.append(0.9)
    elif lab == -1:
        point_colors.append(NOISE_COLOR)
        point_opacity.append(0.5)
    else:
        point_colors.append(color_map.get(lab, NOISE_COLOR))
        point_opacity.append(0.35)

hover_text = [
    f"cluster {int(lab)}: {data.cluster_labels.get(int(lab), {}).get('label', 'noise')}<br>"
    f"{txt[:80]}{'...' if len(txt) > 80 else ''}"
    for lab, txt in zip(labels_arr, data.texts["text"].tolist())
]

fig = go.Figure()
fig.add_trace(
    go.Scattergl(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers",
        marker=dict(
            color=point_colors,
            size=[7 if int(lab) == selected else 4 for lab in labels_arr],
            opacity=point_opacity,
            line=dict(width=0),
        ),
        text=hover_text,
        hoverinfo="text",
    )
)
fig.update_layout(
    paper_bgcolor=BG_SURFACE,
    plot_bgcolor=BG_SURFACE,
    font=dict(family=FONT_BODY, color=TEXT_PRIMARY),
    height=560,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=None),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=None),
    showlegend=False,
)
umap_panel.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# --------------------------------------------------------- explorer panel --
explorer_panel = st.container(border=True)
explorer_panel.markdown("### Cluster explorer")

options = [
    f"{info['label']}  ({info['size']:,} conversations)" for cid, info in sorted_clusters
]
current_index = cluster_ids_sorted.index(st.session_state.selected_cluster)
choice = explorer_panel.selectbox("Select a discovered cluster", options, index=current_index)
chosen_idx = options.index(choice)
st.session_state.selected_cluster = cluster_ids_sorted[chosen_idx]

info = data.cluster_labels[st.session_state.selected_cluster]
explorer_panel.markdown(f'<div class="cluster-label">{info["label"]}</div>', unsafe_allow_html=True)
explorer_panel.markdown(f'<div class="cluster-desc">{info["description"]}</div>', unsafe_allow_html=True)
explorer_panel.markdown(
    f'<p style="color:{TEXT_SECONDARY};">Size: '
    f'<span class="mono-num" style="color:{TEXT_PRIMARY};">{info["size"]:,}</span> conversations '
    f'(<span class="mono-num" style="color:{TEXT_PRIMARY};">'
    f'{100 * info["size"] / n_samples:.1f}%</span> of dataset)</p>',
    unsafe_allow_html=True,
)

explorer_panel.markdown(
    f'<p style="color:{TEXT_SECONDARY}; margin-top:16px;">Example conversations from this cluster:</p>',
    unsafe_allow_html=True,
)
for example in info["examples"][:5]:
    explorer_panel.markdown(f'<div class="example-item">{example}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------ methodology --
kmeans_sil = summary["kmeans"]["silhouette_full"]
kmeans_nmi = summary["kmeans"]["external_validation"]["vs_category"]["nmi"]

with st.expander("Methodology & honest limitations"):
    st.markdown(
        f'K-Means was also run for comparison (k={summary["kmeans"]["k"]} chosen by '
        f'silhouette search): silhouette <span class="mono-num">{kmeans_sil:.3f}</span>, '
        f'NMI vs. category <span class="mono-num">{kmeans_nmi:.3f}</span>. '
        "HDBSCAN was chosen as it scored higher on both.",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'Full HDBSCAN external validation: ARI <span class="mono-num">{ext_cat["ari"]:.3f}</span> / '
        f'NMI <span class="mono-num">{ext_cat["nmi"]:.3f}</span> vs. the 11 held-out categories, '
        f'and ARI <span class="mono-num">{ext_intent["ari"]:.3f}</span> / '
        f'NMI <span class="mono-num">{ext_intent["nmi"]:.3f}</span> vs. the 27 held-out intents.',
        unsafe_allow_html=True,
    )
    st.markdown(
        "External validation (ARI/NMI) compares discovered clusters to the dataset's "
        "real, human-defined `category`/`intent` labels -- held out entirely during "
        "clustering, used here only as an honest sanity check, not a target the "
        "pipeline was optimized against. See the README for the full discussion of "
        "what these numbers do and don't mean.",
        unsafe_allow_html=True,
    )
