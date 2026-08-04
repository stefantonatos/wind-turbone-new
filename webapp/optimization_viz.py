# Visual layer for the 4-step deep optimization pipeline.
#
# WHY THIS EXISTS: the optimization tab used to be a spinner followed by a stack of
# st.dataframe() tables. That is technically complete and practically unreadable - you
# cannot tell from four tables whether the pipeline is halfway done, which step just
# failed, or whether the walk-forward folds actually held up. This module renders the
# same numbers as a pipeline you can watch and charts you can read at a glance. The
# tables stay: every chart here has a table-view twin already on the page, so no value
# is reachable ONLY through a hover tooltip.
#
# DESIGN RULES FOLLOWED (webapp/style.py supplies the tokens, this file consumes them):
#   - Diverging scales (avg R/trade has real polarity around zero) use a warm pole, a
#     NEUTRAL GRAY midpoint, and a cool pole. Never a hue at the midpoint, never a
#     rainbow ramp for magnitude.
#   - One axis per chart. No dual-axis anywhere.
#   - Status colors (GOOD/WARNING/CRITICAL) mean state, never series identity, and always
#     ship with a text label so state is never encoded by color alone.
#   - Thin marks, hairline solid gridlines, no number printed on every data point.
#
# Everything here is pure: it takes DataFrames/dicts and returns HTML strings or plotly
# figures. No streamlit import, so it is testable without a running app.

import math

import plotly.graph_objects as go

from style import (ACCENT, BORDER, CRITICAL, FONT_MONO, GOOD, GRIDLINE, INK_MUTED,
                   INK_PRIMARY, INK_SECONDARY, PLOTLY_LAYOUT_DEFAULTS, SURFACE, WARNING)

# Neutral gray midpoint for every diverging scale below. Deliberately a desaturated
# surface tone, not a hue - the midpoint has to read as "nothing here", and any hue at
# the midpoint makes zero look like a value.
DIVERGING_MID = "#33415a"

# warm pole -> neutral -> cool pole. CRITICAL is warm (red/pink), GOOD is cool
# (green/teal), so the two poles read as genuinely opposite rather than as two
# shades of the same temperature.
DIVERGING_SCALE = [
    [0.0, CRITICAL],
    [0.25, "#8f3550"],
    [0.5, DIVERGING_MID],
    [0.75, "#009e77"],
    [1.0, GOOD],
]

STAGES = [
    ("01", "GRID SEARCH", "every parameter combination, scored in R"),
    ("02", "MONTE CARLO", "resample each cell to see what was luck"),
    ("03", "CLUSTER CHECK", "is the best cell a plateau or a spike?"),
    ("04", "WALK-FORWARD", "refit rolling, score only out-of-sample"),
    ("05", "LOCKBOX", "one-shot sealed window, opened once ever"),
]

_STATE_STYLES = {
    "done":    (GOOD, "COMPLETE"),
    "running": (ACCENT, "RUNNING"),
    "pending": (INK_MUTED, "WAITING"),
    "sealed":  (WARNING, "SEALED"),
    "failed":  (CRITICAL, "FAILED"),
    "skipped": (INK_MUTED, "SKIPPED"),
}


def _esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def pipeline_css():
    """One <style> block for the pipeline diagram. Kept beside the markup it styles
    rather than in style.py's global CSS so this component stays self-contained and
    can be unit-tested without importing the whole app."""
    return f"""
<style>
.optviz-rail {{
  display: flex; gap: 0; align-items: stretch;
  margin: 0.35rem 0 1.1rem 0; flex-wrap: wrap;
}}
.optviz-stage {{
  flex: 1 1 150px; min-width: 150px; position: relative;
  padding: 0.7rem 0.85rem 0.8rem 0.85rem;
  background: {SURFACE}; border: 1px solid {BORDER};
  border-left-width: 3px; border-radius: 6px; margin-right: 26px;
}}
.optviz-stage:last-child {{ margin-right: 0; }}
/* the connector between stages - a hairline, not an arrow glyph, so it stays
   recessive next to the data it is introducing */
.optviz-stage:not(:last-child)::after {{
  content: ""; position: absolute; top: 50%; right: -26px; width: 26px;
  height: 1px; background: {GRIDLINE};
}}
.optviz-num {{
  font-family: {FONT_MONO}; font-size: 0.62rem; letter-spacing: 0.14em;
  color: {INK_MUTED};
}}
.optviz-title {{
  font-family: {FONT_MONO}; font-size: 0.78rem; letter-spacing: 0.06em;
  color: {INK_PRIMARY}; margin: 0.15rem 0 0.3rem 0; font-weight: 600;
}}
.optviz-state {{
  font-family: {FONT_MONO}; font-size: 0.6rem; letter-spacing: 0.12em;
  display: inline-flex; align-items: center; gap: 0.35rem;
}}
.optviz-dot {{ width: 6px; height: 6px; border-radius: 50%; display: inline-block; }}
.optviz-stat {{
  font-family: {FONT_MONO}; font-size: 0.72rem; color: {INK_SECONDARY};
  margin-top: 0.4rem; line-height: 1.35;
}}
.optviz-hint {{
  font-family: {FONT_MONO}; font-size: 0.6rem; color: {INK_MUTED};
  margin-top: 0.3rem; line-height: 1.3;
}}
@keyframes optviz-pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.35; }} }}
.optviz-live .optviz-dot {{ animation: optviz-pulse 1.1s ease-in-out infinite; }}
</style>
"""


def render_pipeline(states, stats=None):
    """The headline component: the 4 methodology steps plus the lockbox, as a rail you
    can read mid-run.

    `states` maps stage index (0-4) -> one of _STATE_STYLES' keys. `stats` optionally
    maps the same index -> a short string shown under the title (e.g. "196 cells").
    State is rendered as a colored dot AND a word ("COMPLETE"/"RUNNING"/...) so it is
    never communicated by color alone."""
    stats = stats or {}
    cards = []
    for i, (num, title, hint) in enumerate(STAGES):
        state = states.get(i, "pending")
        color, label = _STATE_STYLES.get(state, _STATE_STYLES["pending"])
        live = " optviz-live" if state == "running" else ""
        stat = stats.get(i)
        stat_html = f'<div class="optviz-stat">{_esc(stat)}</div>' if stat else ""
        cards.append(
            f'<div class="optviz-stage{live}" style="border-left-color:{color};">'
            f'<div class="optviz-num">STEP {num}</div>'
            f'<div class="optviz-title">{_esc(title)}</div>'
            f'<div class="optviz-state" style="color:{color};">'
            f'<span class="optviz-dot" style="background:{color};"></span>{label}</div>'
            f'{stat_html}'
            f'<div class="optviz-hint">{_esc(hint)}</div>'
            f'</div>')
    return pipeline_css() + '<div class="optviz-rail">' + "".join(cards) + "</div>"


def states_from_result(result, lockbox_status=None):
    """Derive the rail's per-stage state from a finished OptimizationResult, so the
    same component renders both the live run and the completed one."""
    def has(df):
        if df is None:
            return False
        empty = getattr(df, "empty", None)
        return not empty if empty is not None else True

    states = {
        0: "done" if has(getattr(result, "heatmap", None)) else "skipped",
        1: "done" if has(getattr(result, "monte_carlo", None)) else "skipped",
        2: "done" if getattr(result, "cluster_verdict", None) else "skipped",
        3: "done" if has(getattr(result, "walk_forward", None)) else "skipped",
        4: "done" if lockbox_status else "sealed",
    }
    return states


# ---------------------------------------------------------------------------
# STEP 01 - parameter grid
# ---------------------------------------------------------------------------

def heatmap_figure(heatmap_df, title="avg R / trade per parameter combination"):
    """The grid search as an actual heatmap instead of a coloured table.

    Job: compare magnitude across a grid, where the value has polarity (profitable vs
    not) - so this is a DIVERGING scale pinned so that zero always lands on the neutral
    gray midpoint, never wherever the data happens to centre. Without that pin, an
    all-losing grid would still render half-green and read as a mixed result."""
    if heatmap_df is None or getattr(heatmap_df, "empty", True):
        return None
    values = heatmap_df.values.tolist()
    finite = [v for row in values for v in row if v is not None and not _isnan(v)]
    if not finite:
        return None
    # symmetric bound so the gray midpoint sits exactly on zero
    bound = max(abs(min(finite)), abs(max(finite))) or 1e-9

    fig = go.Figure(go.Heatmap(
        z=values,
        x=[str(c) for c in heatmap_df.columns],
        y=[str(i) for i in heatmap_df.index],
        colorscale=DIVERGING_SCALE, zmid=0.0, zmin=-bound, zmax=bound,
        xgap=2, ygap=2,          # 2px surface gap between cells, not a drawn border
        hovertemplate="%{y} · %{x}<br>avg R/trade %{z:+.4f}<extra></extra>",
        colorbar=dict(title=dict(text="avg R", side="right"), thickness=10,
                      outlinewidth=0, tickfont=dict(size=10, color=INK_SECONDARY)),
    ))
    layout = dict(PLOTLY_LAYOUT_DEFAULTS)
    layout.update(title=dict(text=title, font=dict(size=12, color=INK_SECONDARY)),
                  height=max(220, 30 * len(heatmap_df.index) + 100),
                  xaxis=dict(showgrid=False, linecolor=GRIDLINE, tickfont=dict(size=10)),
                  yaxis=dict(showgrid=False, linecolor=GRIDLINE, tickfont=dict(size=10)))
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# STEP 02 - Monte Carlo
# ---------------------------------------------------------------------------

def monte_carlo_figure(mc_df, top_n=12):
    """Bootstrap p5-p95 range per grid cell, with the zero line as the reference.

    This is the chart that answers the question a raw table cannot: does this cell's
    confidence band CLEAR zero, or merely straddle it? A cell whose p5 is still above
    zero survived resampling; one whose band crosses zero did not, however good its
    point estimate looks. Single series, so no legend - the title names it."""
    if mc_df is None or getattr(mc_df, "empty", True):
        return None
    need = {"boot_total_r_p5", "boot_total_r_p50", "boot_total_r_p95"}
    if not need.issubset(set(mc_df.columns)):
        return None
    df = mc_df.head(top_n).iloc[::-1]          # best at the top once plotted
    labels, p5s, p50s, p95s = [], [], [], []
    for pos, (_, row) in enumerate(df.iterrows()):
        # Compact axis labels. The full parameter names are already in the table view
        # below the chart, so spelling out "range_minutes=15 · reward_risk=1" on every
        # row just crowds the plot area without adding anything reachable.
        parts = [f"{_short(c)} {row[c]:g}" for c in df.columns
                 if c not in ("n_trades", "avg_r", "p_total_r_le_0") and not c.startswith("boot_")]
        labels.append(" · ".join(parts) or f"cell {pos + 1}")
        p5s.append(_safe(row["boot_total_r_p5"]))
        p50s.append(_safe(row["boot_total_r_p50"]))
        p95s.append(_safe(row["boot_total_r_p95"]))

    fig = go.Figure()
    for label, lo, hi in zip(labels, p5s, p95s):
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[label, label], mode="lines",
            line=dict(color=ACCENT, width=2), opacity=0.55,
            hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=p50s, y=labels, mode="markers",
        marker=dict(color=ACCENT, size=9, line=dict(color=SURFACE, width=2)),
        customdata=list(zip(p5s, p95s)),
        hovertemplate="%{y}<br>p5 %{customdata[0]:+.1f}R · median %{x:+.1f}R · "
                      "p95 %{customdata[1]:+.1f}R<extra></extra>",
        showlegend=False))
    fig.add_vline(x=0, line=dict(color=CRITICAL, width=1))

    layout = dict(PLOTLY_LAYOUT_DEFAULTS)
    layout.update(
        title=dict(text="bootstrap p5 - median - p95 total R (red line = break-even)",
                   font=dict(size=12, color=INK_SECONDARY)),
        height=max(260, 26 * len(labels) + 120),
        xaxis=dict(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, linecolor=GRIDLINE,
                   title=dict(text="total R", font=dict(size=10, color=INK_MUTED))),
        yaxis=dict(showgrid=False, linecolor=GRIDLINE, tickfont=dict(size=9)))
    fig.update_layout(**layout)
    return fig


def monte_carlo_headline(mc_df):
    """The one number the Monte Carlo section leads with: how many cells actually
    cleared zero at the 5th percentile. Deliberately a stat, not another chart - and
    deliberately the honest framing, because "every path was positive" is what any
    profitable backtest produces and says nothing on its own."""
    if mc_df is None or getattr(mc_df, "empty", True):
        return None
    if "boot_total_r_p5" not in mc_df.columns:
        return None
    p5 = [_safe(v) for v in mc_df["boot_total_r_p5"].tolist()]
    p5 = [v for v in p5 if v is not None]
    if not p5:
        return None
    cleared = sum(1 for v in p5 if v > 0)
    return {"cleared": cleared, "total": len(p5), "share": cleared / len(p5)}


# ---------------------------------------------------------------------------
# STEP 04 - walk-forward
# ---------------------------------------------------------------------------

def walk_forward_figure(wf_df):
    """In-sample vs out-of-sample avg R per fold, as a dumbbell.

    Walk-forward's whole point is the GAP between the two, so the chart draws the gap
    itself: one dot for what the fit achieved in-sample, one for what it then earned on
    data it never saw, joined by a line. Two series, so a legend is present. One axis -
    both dots are the same unit (avg R/trade), which is exactly why they belong on one
    chart and a dual axis would be wrong."""
    if wf_df is None or getattr(wf_df, "empty", True):
        return None
    if not {"is_avg_r", "oos_avg_r"}.issubset(set(wf_df.columns)):
        return None

    labels, is_vals, oos_vals = [], [], []
    for _, row in wf_df.iloc[::-1].iterrows():
        fold = row.get("fold", "?")
        oos_s, oos_e = row.get("oos_start"), row.get("oos_end")
        labels.append(f"fold {fold}  {oos_s}→{oos_e}" if oos_s is not None else f"fold {fold}")
        is_vals.append(_safe(row["is_avg_r"]))
        oos_vals.append(_safe(row["oos_avg_r"]))

    fig = go.Figure()
    for label, a, b in zip(labels, is_vals, oos_vals):
        if a is None or b is None:
            continue
        fig.add_trace(go.Scatter(x=[a, b], y=[label, label], mode="lines",
                                 line=dict(color=INK_MUTED, width=1),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=is_vals, y=labels, mode="markers", name="in-sample (fitted)",
        marker=dict(color=INK_SECONDARY, size=8, line=dict(color=SURFACE, width=2)),
        hovertemplate="%{y}<br>in-sample %{x:+.4f} R/trade<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=oos_vals, y=labels, mode="markers", name="out-of-sample (real)",
        marker=dict(color=ACCENT, size=10, line=dict(color=SURFACE, width=2)),
        hovertemplate="%{y}<br>out-of-sample %{x:+.4f} R/trade<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=CRITICAL, width=1))

    layout = dict(PLOTLY_LAYOUT_DEFAULTS)
    # NO plotly title on this one. The shared layout defaults park the legend at y=1.02,
    # which lands exactly on top of a title string - confirmed by screenshot, the two
    # overlapped and both became unreadable. This chart needs the legend (two series, so
    # identity must not be colour-alone), and the section's own eyebrow heading plus the
    # caption underneath already say what it is, so the legend gets the top strip.
    layout.update(
        height=max(240, 34 * len(labels) + 130),
        xaxis=dict(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, linecolor=GRIDLINE,
                   title=dict(text="avg R / trade", font=dict(size=10, color=INK_MUTED))),
        yaxis=dict(showgrid=False, linecolor=GRIDLINE, tickfont=dict(size=9)))
    fig.update_layout(**layout)
    return fig


def walk_forward_headline(wf_df):
    """How many folds actually made money out-of-sample. The honest headline for a
    walk-forward: in-sample results are fitted by construction and mean nothing."""
    if wf_df is None or getattr(wf_df, "empty", True) or "oos_avg_r" not in wf_df.columns:
        return None
    vals = [_safe(v) for v in wf_df["oos_avg_r"].tolist()]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    positive = sum(1 for v in vals if v > 0)
    return {"positive": positive, "total": len(vals), "share": positive / len(vals)}


# ---------------------------------------------------------------------------
# STEP 05 - lockbox
# ---------------------------------------------------------------------------

def lockbox_panel(sealed, window=None, prior=None):
    """The lockbox as a state you can see. Sealed is the DEFAULT and the good state -
    an unused lockbox is an asset, not an incomplete task, which the old copy-only
    version made no attempt to convey."""
    if sealed:
        color, label, headline = WARNING, "SEALED", "Never opened"
        body = ("This window has never been touched by any search. It can be opened "
                "exactly once, ever, and the result is written to a permanent ledger.")
    else:
        color, label = INK_MUTED, "OPENED"
        headline = f"Used {prior.get('timestamp', '?')}" if prior else "Used"
        body = ("This strategy's one-shot confirmation has already been spent. Any further "
                "run against this window is in-sample by definition.")
    window_html = (f'<div class="optviz-hint">window {_esc(window)}</div>' if window else "")
    return (pipeline_css() +
            f'<div class="optviz-stage" style="border-left-color:{color}; margin-right:0;">'
            f'<div class="optviz-num">STEP 05</div>'
            f'<div class="optviz-title">LOCKBOX · {_esc(headline)}</div>'
            f'<div class="optviz-state" style="color:{color};">'
            f'<span class="optviz-dot" style="background:{color};"></span>{label}</div>'
            f'<div class="optviz-stat">{body}</div>{window_html}</div>')


# ---------------------------------------------------------------------------

_SHORT_NAMES = {
    "range_minutes": "range", "reward_risk": "RR", "stop_buffer_pct": "buffer",
    "fallback_reward_risk": "fallback RR", "confirmation_candles": "confirm",
}


def _short(column):
    return _SHORT_NAMES.get(column, column.replace("_", " "))


def _isnan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _safe(v):
    """None for anything that isn't a real finite number, so plotly leaves a gap
    instead of silently plotting a NaN as zero."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f
