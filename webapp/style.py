# Visual layer on top of Streamlit's defaults. Palette/tokens below are the validated set
# (dataviz skill's color-blindness/contrast check) rather than eyeballed: hairline borders
# instead of drop shadows, no gradients, no emoji, a capped left-aligned content column, a
# system-sans font stack throughout (including stat numbers - no serif anywhere), and a
# small set of chart color constants used explicitly on every Plotly figure instead of the
# library's default rainbow palette.

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
BORDER = "rgba(11,11,11,0.10)"
ACCENT = "#2a78d6"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
WARNING = "#fab219"
GRIDLINE = "#e1e0d9"
FONT_UI = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Inter, sans-serif'

# Multi-series categorical palette - validated all-pairs colorblind-safe up to 3 concurrent
# lines. A 4th series should be a small multiple, not a 4th color in this same list.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]

# Diverging (e.g. a returns/correlation heatmap) and sequential (magnitude-only heatmaps)
# scales, both anchored on the same accent blue so they read as one system.
DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS = "#e34948", "#f0efec", "#2a78d6"
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]

CSS = f"""
<style>
:root {{
  --ink-primary: {INK_PRIMARY};
  --ink-secondary: {INK_SECONDARY};
  --ink-muted: {INK_MUTED};
  --surface: {SURFACE};
  --page: {PAGE};
  --border: {BORDER};
  --accent: {ACCENT};
  --good: {GOOD};
  --critical: {CRITICAL};
  --gridline: {GRIDLINE};
  --font-ui: {FONT_UI};
}}

html, body, [class*="css"], .stMarkdown, .stText {{
  font-family: var(--font-ui) !important;
}}

.stApp {{
  background-color: var(--page);
}}

/* capped, left-aligned content column - never centered-everything */
.block-container {{
  padding-top: 3rem;
  padding-left: 2rem;
  padding-right: 2rem;
  padding-bottom: 3rem;
  max-width: 1160px;
  margin-left: 0;
  margin-right: auto;
}}

h1, h2, h3 {{
  color: var(--ink-primary);
  font-weight: 600;
  letter-spacing: -0.01em;
  font-family: var(--font-ui);
}}

h1 {{ font-size: 1.6rem; margin-bottom: 0.25rem; }}
h2 {{ font-size: 1.15rem; margin-top: 1.75rem; }}
h3 {{ font-size: 1.0rem; color: var(--ink-secondary); font-weight: 500; text-transform: none; }}

p, li, label, .stMarkdown p {{
  color: var(--ink-primary);
}}

/* small-caps muted letter-spaced eyebrow label above a section/group of controls or stats */
.eyebrow {{
  display: block;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin: 1.1rem 0 0.4rem 0;
}}

hr {{
  border: none;
  border-top: 1px solid var(--border);
  margin: 1.25rem 0;
}}

/* sidebar: quiet, structured, no heavy background */
section[data-testid="stSidebar"] {{
  background-color: var(--surface);
  border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] .block-container {{
  padding-top: 1.5rem;
}}

/* primary button - flat, hairline border, accent reserved for the primary action only */
.stButton > button, .stDownloadButton > button {{
  background-color: var(--accent);
  color: #ffffff;
  border: 1px solid var(--accent);
  border-radius: 8px;
  padding: 0.55rem 1.1rem;
  font-weight: 500;
  box-shadow: none !important;
  transition: opacity 0.15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  background-color: var(--accent);
  opacity: 0.85;
}}
.stButton > button:focus:not(:active) {{
  border-color: var(--accent);
}}

/* card-like containers (metric strip, bordered blocks): hairline border, 12px radius,
   NEVER a drop shadow - the single most common "generic dashboard" tell */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  box-shadow: none !important;
  background-color: var(--surface);
}}
div[data-testid="stMetric"] {{
  background: transparent;
  box-shadow: none !important;
}}
div[data-testid="stMetricLabel"] {{
  color: var(--ink-muted);
  font-weight: 400;
}}
div[data-testid="stMetricValue"] {{
  color: var(--ink-primary);
  font-weight: 600;
  font-variant-numeric: proportional-nums;
}}

/* tabs - understated underline style, not filled pill buttons */
.stTabs [data-baseweb="tab-list"] {{
  gap: 1.75rem;
  border-bottom: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
  height: 2.4rem;
  color: var(--ink-muted);
  font-weight: 500;
  background: transparent;
}}
.stTabs [aria-selected="true"] {{
  color: var(--ink-primary) !important;
  border-bottom: 2px solid var(--accent) !important;
}}

/* dataframes/tables: hairline border, no shadow, no zebra rainbow. Table cells use
   tabular (monospaced-width) digits for column alignment - the one place proportional
   figures are wrong, unlike the stat-tile numbers above. */
[data-testid="stDataFrame"] {{
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: none !important;
}}
[data-testid="stDataFrame"] * {{
  font-variant-numeric: tabular-nums;
}}

/* remove Streamlit's default footer/hamburger clutter for a local personal tool */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

/* caption text, muted and small - used for caveats instead of alert boxes everywhere */
.stCaption, [data-testid="stCaptionContainer"] {{
  color: var(--ink-muted) !important;
}}

/* alert boxes (st.info/st.error/etc): hairline border, no shadow, no gradient */
div[data-testid="stAlert"] {{
  box-shadow: none !important;
  border: 1px solid var(--border);
}}
</style>
"""


def eyebrow(text):
    """A small-caps muted letter-spaced label marking a section/group, e.g. 'RESULTS
    SUMMARY' above a metrics row - used instead of a raw, unlabeled st.metric() strip.
    Returns HTML for st.markdown(..., unsafe_allow_html=True)."""
    return f'<span class="eyebrow">{text}</span>'


PLOTLY_LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(family=FONT_UI, color=INK_PRIMARY),
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    colorway=CATEGORICAL,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis=dict(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE),
    yaxis=dict(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE),
)
