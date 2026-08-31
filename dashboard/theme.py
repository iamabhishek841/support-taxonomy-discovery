"""Design tokens and CSS injection for the dashboard.

Color tokens, typography, and spacing are fixed per the project's design
system -- see README for the rationale. Every numeric value rendered in the
app (counts, scores, percentages) uses JetBrains Mono with tabular figures;
that's the one rule applied without exception throughout app.py.
"""

from __future__ import annotations

BG_PRIMARY = "#0B0E14"
BG_SURFACE = "#131826"
BG_SURFACE_ALT = "#171D2E"
BORDER_SUBTLE = "#232A3D"
TEXT_PRIMARY = "#E8EBF2"
TEXT_SECONDARY = "#9AA3B8"
ACCENT_INDIGO = "#6366F1"
ACCENT_EMERALD = "#10B981"
ACCENT_AMBER = "#F59E0B"

FONT_HEADING = "'Space Grotesk', sans-serif"
FONT_BODY = "'Inter', sans-serif"
FONT_MONO = "'JetBrains Mono', monospace"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: {FONT_BODY};
}}

.stApp {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
}}

section[data-testid="stSidebar"] {{
    background-color: {BG_SURFACE_ALT};
    border-right: 1px solid {BORDER_SUBTLE};
}}

section[data-testid="stSidebar"] * {{
    color: {TEXT_PRIMARY};
}}

h1, h2, h3, h4 {{
    font-family: {FONT_HEADING} !important;
    color: {TEXT_PRIMARY} !important;
    letter-spacing: -0.01em;
}}

p, span, label, div {{
    color: {TEXT_PRIMARY};
}}

.subtitle {{
    color: {TEXT_SECONDARY};
    font-size: 0.95rem;
    margin-top: -8px;
    margin-bottom: 24px;
}}

/* ---- numeric values: JetBrains Mono, tabular figures, no exceptions ---- */
.mono-num {{
    font-family: {FONT_MONO};
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum" 1;
}}

/* ---- KPI cards ---- */
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}}

.kpi-card {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
}}

.kpi-label {{
    font-family: {FONT_BODY};
    color: {TEXT_SECONDARY};
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
}}

.kpi-value {{
    font-family: {FONT_MONO};
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum" 1;
    font-size: 2rem;
    font-weight: 600;
    line-height: 1.1;
}}

.kpi-value.accent-indigo {{ color: {ACCENT_INDIGO}; }}
.kpi-value.accent-emerald {{ color: {ACCENT_EMERALD}; }}
.kpi-value.accent-amber {{ color: {ACCENT_AMBER}; }}
.kpi-value.neutral {{ color: {TEXT_PRIMARY}; }}

.kpi-sub {{
    font-family: {FONT_MONO};
    font-variant-numeric: tabular-nums;
    font-size: 0.8rem;
    margin-top: 6px;
}}

.kpi-sub.accent-amber {{ color: {ACCENT_AMBER}; }}
.kpi-sub.accent-emerald {{ color: {ACCENT_EMERALD}; }}
.kpi-sub.neutral {{ color: {TEXT_SECONDARY}; }}

/* Native st.container(border=True) restyled to match the card design tokens */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER_SUBTLE} !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    padding: 8px 8px 16px 8px;
    margin-bottom: 24px;
}}

.cluster-label {{
    font-family: {FONT_HEADING};
    font-size: 1.4rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    margin-bottom: 4px;
}}

.cluster-desc {{
    color: {TEXT_SECONDARY};
    font-size: 0.95rem;
    margin-bottom: 16px;
}}

.example-item {{
    background-color: {BG_SURFACE_ALT};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.9rem;
    color: {TEXT_PRIMARY};
}}

/* Streamlit widget restyling to match the surface language */
div[data-baseweb="select"] > div {{
    background-color: {BG_SURFACE_ALT};
    border-color: {BORDER_SUBTLE};
}}

hr {{
    border-color: {BORDER_SUBTLE};
}}

/* Hide default Streamlit chrome for a cleaner portfolio look */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
</style>
"""


def score_accent(value: float, warn_below: float, good_above: float) -> str:
    """Map a metric value to a semantic accent class name."""
    if value >= good_above:
        return "accent-emerald"
    if value < warn_below:
        return "accent-amber"
    return "accent-indigo"


def qualitative_cluster_colors(n: int) -> list[str]:
    """Generate n visually distinct HSL colors for cluster identity coloring.

    Not drawn from the brand accent set -- with up to dozens of discovered
    clusters, a small fixed categorical theme can't provide enough distinct
    hues, so we space hues evenly around the wheel with alternating
    lightness bands, tuned for legibility on the dark surface. Cluster
    identity is never conveyed by color alone: hover tooltips and the
    cluster-explorer selection both label points by text as well.
    """
    colors = []
    for i in range(n):
        hue = (i * 137.508) % 360  # golden-angle spacing avoids near-duplicate hues
        lightness = 55 + (10 if i % 2 == 0 else -8)
        saturation = 65
        colors.append(f"hsl({hue:.1f}, {saturation}%, {lightness}%)")
    return colors


NOISE_COLOR = "rgba(154, 163, 184, 0.35)"  # muted text-secondary, low opacity
