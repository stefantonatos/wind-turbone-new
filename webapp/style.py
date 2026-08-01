# Visual layer on top of Streamlit's defaults - dark trading-terminal / HUD panel look
# (TradingView-dark base + glowing cyan accents, "Jarvis" energy), not the earlier minimal
# light theme. Same exported names as before (color constants, CSS, PLOTLY_LAYOUT_DEFAULTS,
# eyebrow()) so app.py/optimization.py don't need import changes - only the values and the
# CSS content changed.

INK_PRIMARY = "#e6edf3"
INK_SECONDARY = "#9aa7b8"
INK_MUTED = "#5c6b7f"
SURFACE = "#141a26"
PAGE = "#0a0e17"
BORDER = "rgba(0, 212, 255, 0.18)"
ACCENT = "#00c2ff"
GOOD = "#00e6a0"
CRITICAL = "#ff4d6a"
WARNING = "#ffb800"
GRIDLINE = "rgba(255, 255, 255, 0.06)"
FONT_UI = '"Inter", -apple-system, "Segoe UI", sans-serif'
FONT_MONO = '"JetBrains Mono", "SF Mono", "Cascadia Code", ui-monospace, monospace'

# Multi-series categorical palette - cyan/amber/violet, dark-background-legible, capped at 3
# concurrent lines (a 4th series should be a small multiple, not a 4th color in this list).
CATEGORICAL = ["#00c2ff", "#ffb800", "#a06bff"]

# Diverging (returns/correlation heatmaps) and sequential (magnitude-only heatmaps), both
# anchored on the same accent cyan so they read as one system against the dark base.
DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS = "#ff4d6a", "#1c2333", "#00c2ff"
SEQUENTIAL_BLUE = ["#0a2a3d", "#0d3a54", "#10507a", "#1470a3", "#1795d4", "#3ab4f0", "#7dd3ff"]

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
  --font-mono: {FONT_MONO};
}}

html, body, [class*="css"], .stMarkdown, .stText {{
  font-family: var(--font-ui) !important;
}}

.stApp {{
  background: radial-gradient(ellipse at top, #0d1420 0%, var(--page) 55%);
  color: var(--ink-primary);
}}

/* wide panel/terminal layout, not a narrow blog column - HUD dashboards run wide */
.block-container {{
  padding-top: 2.25rem;
  padding-left: 2rem;
  padding-right: 2rem;
  padding-bottom: 3rem;
  max-width: 1560px;
  margin-left: auto;
  margin-right: auto;
}}

h1, h2, h3 {{
  color: var(--ink-primary);
  font-weight: 600;
  letter-spacing: -0.01em;
  font-family: var(--font-ui);
}}

h1 {{
  font-size: 1.6rem;
  margin-bottom: 0.25rem;
  text-shadow: 0 0 18px rgba(0, 194, 255, 0.35);
}}
h2 {{ font-size: 1.15rem; margin-top: 1.75rem; }}
h3 {{ font-size: 1.0rem; color: var(--ink-secondary); font-weight: 500; text-transform: none; }}

p, li, label, .stMarkdown p {{
  color: var(--ink-primary);
}}

/* HUD-style eyebrow label - glowing cyan, wide tracking, monospace, small corner ticks either
   side (a lightweight nod to sci-fi panel framing without relying on external icon assets) */
.eyebrow {{
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
  text-shadow: 0 0 8px rgba(0, 194, 255, 0.45);
  margin: 1.4rem 0 0.6rem 0;
}}
.eyebrow::before {{
  content: "";
  width: 0.85rem;
  height: 1px;
  background: var(--accent);
  box-shadow: 0 0 6px rgba(0, 194, 255, 0.7);
  display: inline-block;
}}
.eyebrow::after {{
  content: "";
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, rgba(0, 194, 255, 0.35), transparent);
}}

hr {{
  border: none;
  border-top: 1px solid var(--border);
  margin: 1.25rem 0;
}}

/* top brand bar + page nav - replaces the old left sidebar entirely, HUD console strip
   instead of a 2006-era left-rail layout */
.brand {{
  font-family: var(--font-mono);
  font-size: 1.35rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--ink-primary);
  text-shadow: 0 0 18px rgba(0, 194, 255, 0.35);
  padding-top: 0.4rem;
  white-space: nowrap;
}}
.brand-rule {{
  border: none;
  border-top: 1px solid var(--border);
  margin: 0.9rem 0 1.6rem 0;
  box-shadow: 0 1px 14px rgba(0, 194, 255, 0.08);
}}
div[data-testid="stSegmentedControl"] label {{
  font-family: var(--font-mono) !important;
  letter-spacing: 0.05em;
}}

/* config console - the bordered panel holding strategy/instrument/date controls that used
   to live in the sidebar, now a horizontal strip at the top of the page instead */
.console-label {{
  font-family: var(--font-mono);
  font-size: 0.7rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin-bottom: 0.15rem;
}}

/* primary button - glowing HUD action button */
.stButton > button, .stDownloadButton > button {{
  background: linear-gradient(180deg, rgba(0, 194, 255, 0.18), rgba(0, 194, 255, 0.08));
  color: var(--accent);
  border: 1px solid rgba(0, 194, 255, 0.55);
  border-radius: 4px;
  padding: 0.55rem 1.1rem;
  font-weight: 600;
  font-family: var(--font-mono);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-size: 0.82rem;
  box-shadow: 0 0 0 rgba(0, 194, 255, 0);
  transition: box-shadow 0.2s ease, background 0.2s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  background: linear-gradient(180deg, rgba(0, 194, 255, 0.3), rgba(0, 194, 255, 0.12));
  box-shadow: 0 0 18px rgba(0, 194, 255, 0.35);
  color: #ffffff;
}}
.stButton > button:focus:not(:active) {{
  border-color: var(--accent);
}}
.stButton > button[kind="primary"] {{
  background: linear-gradient(180deg, rgba(0, 194, 255, 0.4), rgba(0, 194, 255, 0.18));
  color: #ffffff;
  box-shadow: 0 0 14px rgba(0, 194, 255, 0.3);
}}

/* card-like containers (metric strip, bordered blocks): dark panel, glowing hairline border,
   soft outer glow + inset highlight - the HUD-panel look, deliberately not flat/shadowless */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  background: linear-gradient(180deg, #141a26 0%, #10151f 100%) !important;
  box-shadow: 0 0 24px rgba(0, 194, 255, 0.05), inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
}}
div[data-testid="stMetric"] {{
  background: transparent;
  box-shadow: none !important;
}}
div[data-testid="stMetricLabel"] {{
  color: var(--ink-muted);
  font-weight: 500;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}}
div[data-testid="stMetricValue"] {{
  color: var(--ink-primary);
  font-weight: 600;
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  text-shadow: 0 0 10px rgba(230, 237, 243, 0.15);
}}

/* tabs - glowing underline on the active tab instead of a plain line */
.stTabs [data-baseweb="tab-list"] {{
  gap: 1.75rem;
  border-bottom: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
  height: 2.4rem;
  color: var(--ink-muted);
  font-weight: 500;
  font-family: var(--font-mono);
  letter-spacing: 0.04em;
  background: transparent;
}}
.stTabs [aria-selected="true"] {{
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
  text-shadow: 0 0 8px rgba(0, 194, 255, 0.45);
}}

/* dataframes/tables: dark panel, glowing hairline border, monospace tabular digits */
[data-testid="stDataFrame"] {{
  border: 1px solid var(--border);
  border-radius: 6px;
  box-shadow: 0 0 18px rgba(0, 194, 255, 0.04);
}}
[data-testid="stDataFrame"] * {{
  font-variant-numeric: tabular-nums;
  font-family: var(--font-mono) !important;
}}

/* inputs - dark fields with a glowing focus ring */
.stTextInput input, .stNumberInput input, .stDateInput input, .stSelectbox div[data-baseweb="select"] > div {{
  background-color: #0d1420 !important;
  border-color: var(--border) !important;
  color: var(--ink-primary) !important;
  font-family: var(--font-mono);
}}

/* remove Streamlit's default footer/hamburger clutter for a local personal tool */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

/* caption text, muted mono, used for caveats instead of alert boxes everywhere */
.stCaption, [data-testid="stCaptionContainer"] {{
  color: var(--ink-muted) !important;
  font-family: var(--font-mono);
  font-size: 0.78rem;
}}

/* alert boxes (st.info/st.error/etc): dark panel, glowing hairline border */
div[data-testid="stAlert"] {{
  box-shadow: 0 0 16px rgba(0, 194, 255, 0.05);
  border: 1px solid var(--border);
  background: #10151f;
}}

/* thin, dark, glowing scrollbar - a small but real HUD-terminal tell */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: var(--page); }}
::-webkit-scrollbar-thumb {{
  background: rgba(0, 194, 255, 0.25);
  border-radius: 5px;
}}
::-webkit-scrollbar-thumb:hover {{ background: rgba(0, 194, 255, 0.45); }}
</style>
"""


def eyebrow(text):
    """A glowing cyan, wide-tracked, monospace HUD-style label marking a section/group, e.g.
    'RESULTS SUMMARY' above a metrics row - used instead of a raw, unlabeled st.metric() strip.
    Returns HTML for st.markdown(..., unsafe_allow_html=True)."""
    return f'<span class="eyebrow">{text}</span>'


PLOTLY_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    font=dict(family=FONT_MONO, color=INK_PRIMARY),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=CATEGORICAL,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis=dict(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, linecolor=GRIDLINE),
    yaxis=dict(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, linecolor=GRIDLINE),
)
