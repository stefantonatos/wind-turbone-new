# Streamlit entry point for a personal, local-only strategy backtesting dashboard.
#
# Every backtest here runs for real against Dukascopy (via the research/*.py modules'
# own fetch functions - nothing mocked or precomputed). research/ is never imported for
# anything other than calling its existing functions; see registry.py for exactly which
# module attributes/functions each strategy relies on.
#
# Run with:  streamlit run webapp/app.py

import datetime
import importlib
import math
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # webapp/ itself (registry, stats, ...)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (research/ package)

import github_storage
import optimization
import run_history
import stats as stats_mod
import style
from registry import STRATEGIES, STRATEGIES_BY_ID, _instrument_labels
from style import ACCENT, CRITICAL, GOOD, WARNING, INK_MUTED, CSS, PLOTLY_LAYOUT_DEFAULTS, eyebrow

st.set_page_config(page_title="Strategy Backtests", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

if "last_run" not in st.session_state:
    st.session_state.last_run = None


# --------------------------------------------------------------------------------------
# table styling helpers - color AND a +/- sign together for any R-multiple/P&L column
# (never color alone), a diverging blue/red background for the optimization heatmap.
# No matplotlib dependency - colors are interpolated by hand so webapp/requirements.txt
# stays to the 5 packages the task called for.
# --------------------------------------------------------------------------------------

def _signed_color(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if v > 0:
        return f"color: {GOOD}"
    if v < 0:
        return f"color: {CRITICAL}"
    return f"color: {INK_MUTED}"


def style_signed_columns(df, cols, fmt="{:+.4f}"):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    fmt_map = fmt if isinstance(fmt, dict) else {c: fmt for c in cols}
    styler = df.style
    styler = styler.map(_signed_color, subset=cols) if hasattr(styler, "map") else styler.applymap(_signed_color, subset=cols)
    return styler.format(fmt_map, na_rep="-")


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _mix_hex(c1, c2, t):
    a, b = _hex_to_rgb(c1), _hex_to_rgb(c2)
    mixed = tuple(round(a[k] + (b[k] - a[k]) * t) for k in range(3))
    return f"#{mixed[0]:02x}{mixed[1]:02x}{mixed[2]:02x}"


def style_diverging_heatmap(df):
    numeric = df.select_dtypes("number")
    vmax = float(numeric.abs().max().max()) if not numeric.empty else 0.0

    def bg(v):
        if v is None or (isinstance(v, float) and pd.isna(v)) or vmax == 0:
            return ""
        t = min(abs(v) / vmax, 1.0)
        color = _mix_hex(style.DIVERGING_MID, style.DIVERGING_POS, t) if v >= 0 else _mix_hex(style.DIVERGING_MID, style.DIVERGING_NEG, t)
        return f"background-color: {color}"

    styler = df.style
    styler = styler.map(bg) if hasattr(styler, "map") else styler.applymap(bg)
    return styler.format("{:+.4f}", na_rep="-")


def render_dollar_equity_chart(xs, equity, chronological, key, height=320, compact=False,
                                 starting_balance=10000.0):
    """Account-equity-in-dollars chart, green above `starting_balance` and red below it, as a
    filled area - the standard "clip to baseline, fill twice" Plotly technique for a
    threshold-relative color split (Plotly has no native per-segment line coloring). Two
    baseline+fill trace pairs: one clipped to show only the ABOVE-baseline excursions (green),
    one clipped to show only the BELOW-baseline excursions (red) - their line stroke doubles as
    the visible equity path since raw fill traces alone have no crisp edge. `compact=True` drops
    axis labels/ticks for gallery thumbnails."""
    y_pos = [max(v, starting_balance) for v in equity]
    y_neg = [min(v, starting_balance) for v in equity]
    baseline = [starting_balance] * len(xs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=baseline, mode="lines", line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=y_pos, mode="lines", line=dict(width=2, color=GOOD),
                              fill="tonexty", fillcolor="rgba(0, 230, 160, 0.20)",
                              showlegend=False, hoverinfo="skip" if compact else None,
                              name="Above $10,000"))
    fig.add_trace(go.Scatter(x=xs, y=baseline, mode="lines", line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=y_neg, mode="lines", line=dict(width=2, color=CRITICAL),
                              fill="tonexty", fillcolor="rgba(255, 77, 106, 0.20)",
                              showlegend=False, hoverinfo="skip" if compact else None,
                              name="Below $10,000"))
    layout_kwargs = dict(PLOTLY_LAYOUT_DEFAULTS)
    if compact:
        layout_kwargs["xaxis"] = dict(visible=False)
        layout_kwargs["yaxis"] = dict(visible=False)
        layout_kwargs["margin"] = dict(l=0, r=0, t=0, b=0)
    fig.update_layout(
        height=height,
        xaxis_title=None if compact else ("Trade sequence (chronological)" if chronological else "Trade sequence"),
        yaxis_title=None if compact else "Account equity ($)",
        showlegend=False,
        **layout_kwargs,
    )
    st.plotly_chart(fig, use_container_width=True, key=key,
                     config={"displayModeBar": False} if compact else None)


def render_performance_calendar(trades, key, unit_label="R", unit_fmt="{:+.3f}R", height=230):
    """GitHub-contribution-style calendar: one column per week, one row per weekday, colored by
    that day's total return (diverging colorscale centered on zero - not activity intensity like
    a real GitHub graph). Answers a question the equity curve can't: is the edge spread evenly
    across time, or is it a few clustered days carrying the whole result? `trades` should already
    be in whichever unit (R or %) the rest of the page is displaying, matching `unit_fmt`."""
    daily = stats_mod.daily_pnl(trades)
    if not daily:
        st.caption("No dated trades to build a calendar from.")
        return
    dates = sorted(daily)
    start, end = dates[0], dates[-1]
    grid_start = start - datetime.timedelta(days=start.weekday())
    grid_end = end + datetime.timedelta(days=6 - end.weekday())
    all_days = [grid_start + datetime.timedelta(days=i) for i in range((grid_end - grid_start).days + 1)]
    week_starts = sorted({d - datetime.timedelta(days=d.weekday()) for d in all_days})
    week_index = {w: i for i, w in enumerate(week_starts)}
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    z = [[None] * len(week_starts) for _ in range(7)]
    hover = [[""] * len(week_starts) for _ in range(7)]
    for d in all_days:
        col = week_index[d - datetime.timedelta(days=d.weekday())]
        row = d.weekday()
        val = daily.get(d)
        z[row][col] = val
        hover[row][col] = f"{d.isoformat()}<br>{unit_fmt.format(val)}" if val is not None else f"{d.isoformat()}<br>no trades"

    fig = go.Figure(data=go.Heatmap(
        z=z, y=weekday_labels, text=hover, hoverinfo="text",
        colorscale=[[0.0, CRITICAL], [0.5, "#1c2333"], [1.0, GOOD]], zmid=0,
        xgap=3, ygap=3, showscale=True, colorbar=dict(title=unit_label, thickness=12),
    ))
    layout_kwargs = dict(PLOTLY_LAYOUT_DEFAULTS)
    layout_kwargs["xaxis"] = dict(visible=False)
    layout_kwargs["margin"] = dict(l=10, r=10, t=10, b=10)
    fig.update_layout(height=height, **layout_kwargs)
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})
    st.caption("Each cell is one calendar day's total return across every instrument - green "
               "net-positive, red net-negative, blank = no trade that day.")


# --------------------------------------------------------------------------------------
# shared results rendering - used for a fresh run AND for re-viewing a past run from
# History, so the filter/stats/chart/table behavior is identical either way.
# --------------------------------------------------------------------------------------

def render_filterable_results(trades, strategy, key_prefix):
    facets = strategy.facets if strategy else []
    outcomes = sorted({t.get("outcome") for t in trades if t.get("outcome")})
    instruments = sorted({t.get("instrument") for t in trades if t.get("instrument")})
    sides = sorted({t.get("side") for t in trades if t.get("side")})
    has_dates = all(t.get("date") for t in trades)

    # Everything that changes HOW results are filtered/computed lives in one hidden-by-default
    # section - sensible defaults (no filters, typical costs applied, % of account, 1% risk)
    # apply automatically, same "don't make people decide things to see a result" philosophy as
    # the removed manual-parameter sliders. Nothing below this needs opening it to work.
    with st.expander("Filters & display settings", expanded=False):
        n_cols = 3 + len(facets)
        cols = st.columns(n_cols)
        sel_outcomes = cols[0].multiselect("Outcome", outcomes, default=outcomes, key=f"{key_prefix}_f_outcome")
        sel_instruments = cols[1].multiselect("Instrument", instruments, default=instruments,
                                                key=f"{key_prefix}_f_instrument")
        sel_sides = cols[2].multiselect("Side", sides, default=sides, key=f"{key_prefix}_f_side")

        facet_selections = {}
        for i, facet in enumerate(facets):
            values = sorted({t.get(facet) for t in trades if t.get(facet)})
            facet_selections[facet] = cols[3 + i].multiselect(facet.capitalize(), values, default=values,
                                                                 key=f"{key_prefix}_f_{facet}")

        date_range_filter = None
        if has_dates:
            all_dates = sorted({t["date"] for t in trades})
            min_d, max_d = all_dates[0], all_dates[-1]
            if min_d != max_d:
                date_range_filter = st.slider("Date sub-range", min_value=min_d, max_value=max_d,
                                                value=(min_d, max_d), key=f"{key_prefix}_f_daterange")

        st.markdown("---")
        set_cols = st.columns([1.5, 1.1, 1.1, 2.3])
        apply_costs = set_cols[0].checkbox("Apply typical trading costs", value=True,
                                             key=f"{key_prefix}_apply_costs",
                                             help="Deducts a typical PROP FIRM round-trip cost (raw spread + "
                                                  "per-lot commission, averaged across FTMO/FundedNext/The5ers "
                                                  "- see stats.py's TYPICAL_COST_PCT_BY_INSTRUMENT) per "
                                                  "instrument from every trade - every backtest here runs with "
                                                  "ZERO cost modeled by default, so this is on by default to "
                                                  "avoid overstating what's achievable. Turn off to see the "
                                                  "underlying research script's raw numbers.")
        display_mode = set_cols[1].radio("Units", ["% of account", "R-multiples"], index=0,
                                          key=f"{key_prefix}_display_mode", label_visibility="collapsed")
        risk_pct = set_cols[2].number_input("Risk/trade %", min_value=0.05, max_value=10.0, value=1.0,
                                              step=0.25, key=f"{key_prefix}_risk_pct",
                                              help="Assumed % of account risked per trade - converts R into "
                                                   "an account % and into the dollar equity curve. A "
                                                   "display assumption only; the underlying trades never "
                                                   "change.")
        set_cols[3].caption("Costs applied (typical, not live broker data)" if apply_costs
                             else "⚠ RAW - no costs deducted")

    filtered = []
    for t in trades:
        if sel_outcomes and t.get("outcome") not in sel_outcomes:
            continue
        if sel_instruments and t.get("instrument") not in sel_instruments:
            continue
        if sel_sides and t.get("side") not in sel_sides:
            continue
        skip = False
        for facet, sel in facet_selections.items():
            if sel and t.get(facet) not in sel:
                skip = True
                break
        if skip:
            continue
        if date_range_filter is not None and t.get("date"):
            if not (date_range_filter[0] <= t["date"] <= date_range_filter[1]):
                continue
        filtered.append(t)

    if not filtered:
        st.warning("No trades match the current filters.")
        return
    n_unadjusted = 0
    if apply_costs:
        filtered, n_unadjusted = stats_mod.apply_cost_adjustment(filtered)
    caption_bits = []
    if len(filtered) != len(trades):
        caption_bits.append(f"{len(filtered)} of {len(trades)} trades match the current filters")
    if n_unadjusted:
        caption_bits.append(f"{n_unadjusted} trade(s) had no stop distance and were left cost-unadjusted")
    if caption_bits:
        st.caption(" · ".join(caption_bits) + ".")

    s = stats_mod.compute_stats(filtered)   # filtered is non-empty and cost adjustment never drops trades
    display_trades = stats_mod.scale_trades_r(filtered, risk_pct) if display_mode == "% of account" else filtered
    unit_label = "%" if display_mode == "% of account" else "R"
    unit_fmt = "{:+.2f}%" if display_mode == "% of account" else "{:+.3f}R"
    s_display = stats_mod.compute_stats(display_trades)

    # "Total" is COMPOUNDED for the "% of account" unit, not the naive sum compute_stats returns
    # (which treats every trade as risking a fixed dollar amount off the STARTING balance
    # forever - fine for R-multiples themselves, which are meant to sum, but it can show an
    # impossible return past -100% once losses accumulate over enough trades). R-multiples stay
    # additive (that IS what a raw R total means) - see stats.compounded_return_pct's own
    # docstring for the full reasoning.
    if unit_label == "%":
        total_display = stats_mod.compounded_return_pct(filtered, risk_pct)
        total_ci_low, total_ci_high = stats_mod.compounded_return_ci(
            s["avg_r_ci_low"], s["avg_r_ci_high"], risk_pct, s["n_trades"])
        max_dd_display = stats_mod.compounded_max_drawdown_pct(filtered, risk_pct)
    else:
        total_display = s_display["total_r"]
        total_ci_low, total_ci_high = s_display["total_r_ci_low"], s_display["total_r_ci_high"]
        max_dd_display = s_display["max_drawdown_r"]

    result_tab_labels = ["Overview", "Breakdown", "Prop Firm Fit"]
    if has_dates:
        result_tab_labels.append("Calendar")
    chart_capable = bool(strategy and strategy.chart_fetcher)
    if chart_capable:
        result_tab_labels.append("Trade Chart")
    result_tabs = dict(zip(result_tab_labels, st.tabs(result_tab_labels)))

    with result_tabs["Overview"]:
        with st.container(border=True):
            metric_cols = st.columns(7)
            metric_cols[0].metric("Trades", s_display["n_trades"])
            metric_cols[1].metric(f"Total {unit_label}", unit_fmt.format(total_display),
                                   help=f"95% CI: {unit_fmt.format(total_ci_low)} to "
                                        f"{unit_fmt.format(total_ci_high)} (normal "
                                        f"approximation - wide on a small sample, not a guarantee)."
                                        + (" Compounded (risking % of current balance each trade), "
                                           "not a naive sum - can approach but never cross -100%."
                                           if unit_label == "%" else ""))
            avg_fmt = unit_fmt if unit_label == "R" else "{:+.3f}%"
            metric_cols[2].metric(f"Avg {unit_label}/trade", avg_fmt.format(s_display["avg_r"]),
                                   help=f"95% CI: {avg_fmt.format(s_display['avg_r_ci_low'])} to "
                                        f"{avg_fmt.format(s_display['avg_r_ci_high'])}.")
            metric_cols[3].metric("Win rate", f"{s_display['tp_pct']:.1f}%")
            metric_cols[4].metric("Loss rate", f"{s_display['sl_pct']:.1f}%")
            metric_cols[5].metric(f"Max drawdown", unit_fmt.format(-max_dd_display),
                                   help="Largest peak-to-trough decline in the cumulative equity curve "
                                        "below, not the worst single losing trade."
                                        + (" Compounded, same reasoning as Total % above - bounded to "
                                           "[0, 100]%, unlike a naive additive drawdown."
                                           if unit_label == "%" else ""))
            metric_cols[6].metric("z-score", f"{s_display['z_score']:.2f}",
                                   help="Approximate significance vs. zero edge - scale-invariant (same in "
                                        "either unit). Not strong evidence in isolation: this project has "
                                        "shipped many strategies/parameter searches, so data-snooping risk "
                                        "compounds project-wide, not just here.")
        if s_display["n_trades"] < stats_mod.MIN_TRADES_FOR_RANKING:
            st.warning(f"Only {s_display['n_trades']} trades - below the {stats_mod.MIN_TRADES_FOR_RANKING}-"
                       f"trade floor this app uses elsewhere (Compare All, Gallery) before treating a result "
                       f"as rankable. Directional at best, not evidence either way.")

        st.markdown(eyebrow("EQUITY CURVE (ACCOUNT \\$, \\$10,000 START)"), unsafe_allow_html=True)
        xs, equity, chronological = stats_mod.dollar_equity_curve(filtered, risk_pct)
        render_dollar_equity_chart(xs, equity, chronological, key=f"{key_prefix}_equity_chart")
        st.caption(f"\\${risk_pct:.2f}% risked/trade, green above \\$10,000 red below - a sizing assumption "
                   f"for this chart only." + ("" if chronological else " Order shown is backtest sequence, "
                   "not calendar order (no date field upstream)."))

    with result_tabs["Breakdown"]:
        st.markdown(eyebrow("PER-INSTRUMENT BREAKDOWN"), unsafe_allow_html=True)
        rows = stats_mod.per_instrument_breakdown(display_trades)
        breakdown_df = pd.DataFrame(rows)
        breakdown_styler = style_signed_columns(breakdown_df, ["total_r", "avg_r"],
                                                  fmt={"total_r": "{:+.3f}", "avg_r": "{:+.4f}"})
        if "win_pct" in breakdown_df.columns and hasattr(breakdown_styler, "format"):
            breakdown_styler = breakdown_styler.format({"win_pct": "{:.1f}%"})
        st.dataframe(breakdown_styler, use_container_width=True, hide_index=True)

        st.markdown(eyebrow("TRADE LOG"), unsafe_allow_html=True)
        trade_df = pd.DataFrame(display_trades)
        st.dataframe(style_signed_columns(trade_df, ["r"]), use_container_width=True, hide_index=True)

    with result_tabs["Prop Firm Fit"]:
        render_prop_firm_fit_section(filtered, key_prefix)

    if has_dates:
        with result_tabs["Calendar"]:
            calendar_fmt = "{:+.3f}%" if unit_label == "%" else "{:+.3f}R"
            render_performance_calendar(display_trades, key=f"{key_prefix}_calendar",
                                          unit_label=unit_label, unit_fmt=calendar_fmt)

    if chart_capable:
        with result_tabs["Trade Chart"]:
            render_trade_chart_section(strategy, filtered, key_prefix)


def render_trade_chart_section(strategy, trades, key_prefix):
    """Real candlestick chart around one chosen trade, with entry/stop/target as horizontal
    lines and entry/exit as markers - a short, freshly-fetched window (not a slice of the
    full backtest range), gated behind a button since it's a real Dukascopy fetch. Only
    strategies whose trade dicts carry entry/stop/target/exit price+time show this section -
    see registry.py's chart_fetcher field for exactly which ones do so far."""
    if strategy is None or strategy.chart_fetcher is None:
        return
    chartable = [t for t in trades if t.get("entry_time") and t.get("exit_time")
                 and t.get("entry_price") is not None]
    if not chartable:
        return

    def _label(t):
        et = pd.Timestamp(t["entry_time"])
        return (f"{et.strftime('%Y-%m-%d %H:%M')} - {t.get('instrument', '?')} {t.get('side', '?')} "
                f"({t.get('outcome', '?')}, {t.get('r', 0.0):+.2f}R)")

    chosen_idx = st.selectbox("Trade", range(len(chartable)), format_func=lambda i: _label(chartable[i]),
                               key=f"{key_prefix}_chart_trade")
    trade = chartable[chosen_idx]
    const_map = dict(strategy.instruments)
    const = const_map.get(trade.get("instrument"))
    if const is None:
        st.caption("Can't chart this trade - its instrument wasn't found in the strategy's instrument list.")
        return

    entry_time = pd.Timestamp(trade["entry_time"])
    exit_time = pd.Timestamp(trade["exit_time"])
    duration = exit_time - entry_time
    pad = max(duration * 0.25, pd.Timedelta(hours=6))
    window_start, window_end = entry_time - pad, exit_time + pad
    fetch_start = window_start.tz_convert("UTC").tz_localize(None) if window_start.tzinfo else window_start
    fetch_end = window_end.tz_convert("UTC").tz_localize(None) if window_end.tzinfo else window_end

    st.caption("A real, freshly-fetched Dukascopy candlestick window around this trade only (not the whole "
               "backtest range) - gated behind the button below since it's a real fetch, same as the "
               "Optimization tab and Prop Firm Fit sweep above.")
    if not st.button("Load chart", key=f"{key_prefix}_chart_load"):
        return

    module = importlib.import_module(strategy.module_name)
    with st.spinner(f"Fetching {trade.get('instrument')} bars around this trade..."):
        try:
            df = strategy.chart_fetcher(module, trade.get("instrument"), const, fetch_start, fetch_end)
        except Exception as exc:
            st.error(f"Couldn't fetch chart data: {exc}")
            return
    if df is None or df.empty:
        st.warning("No bar data returned for this window.")
        return

    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=GOOD, decreasing_line_color=CRITICAL, name=trade.get("instrument"))])
    for price, price_label, color in (
        (trade.get("entry_price"), "Entry", ACCENT),
        (trade.get("stop_price"), "Stop", CRITICAL),
        (trade.get("target_price"), "Target", GOOD),
    ):
        if price is not None:
            fig.add_hline(y=price, line=dict(color=color, width=1, dash="dot"),
                          annotation_text=price_label, annotation_position="right",
                          annotation_font_color=color)
    exit_color = GOOD if trade.get("outcome") == "TP" else (CRITICAL if trade.get("outcome") == "SL" else WARNING)
    fig.add_trace(go.Scatter(x=[entry_time], y=[trade.get("entry_price")], mode="markers",
                              marker=dict(symbol="triangle-right", size=13, color=ACCENT,
                                          line=dict(width=1, color="#ffffff")),
                              name="Entry", showlegend=True))
    fig.add_trace(go.Scatter(x=[exit_time], y=[trade.get("exit_price")], mode="markers",
                              marker=dict(symbol="x", size=12, color=exit_color,
                                          line=dict(width=1, color="#ffffff")),
                              name=f"Exit ({trade.get('outcome')})", showlegend=True))
    fig.update_layout(height=440, xaxis_rangeslider_visible=False,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                       **{k: v for k, v in PLOTLY_LAYOUT_DEFAULTS.items() if k != "legend"})
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart_fig")


def render_prop_firm_fit_section(trades, key_prefix):
    """% chance of passing a REAL prop firm's evaluation, run against this strategy's actual
    backtested trades (bootstrap-resampled, chained across every real evaluation phase - see
    research/prop_firm_challenge_simulator.py's multi_phase_risk_sweep and
    research/prop_firm_presets.py for the sourced rule sets). Gated behind a button - this is
    real Monte Carlo compute (thousands of simulated multi-phase attempts), not instant."""
    if len(trades) < 10:
        st.caption("Not enough trades in this run (need at least 10) to run a meaningful prop-firm simulation.")
        return

    st.warning("⚠ **Not a guarantee.** This is a probability estimate from resampling one already-completed "
               "backtest, not a prediction of what will happen if you trade this for real. A high \"chance of "
               "passing\" here does not mean you will pass, or that this strategy is safe to fund a real "
               "account with. See the app-wide disclaimer at the bottom of the page.")

    prop_presets = importlib.import_module("research.prop_firm_presets")
    prop_sim = importlib.import_module("research.prop_firm_challenge_simulator")

    preset_options = prop_presets.list_presets()
    preset_labels = {pid: label for pid, label in preset_options}
    cols = st.columns([2, 1])
    chosen_preset_id = cols[0].selectbox("Prop firm", [pid for pid, _ in preset_options],
                                          format_func=lambda pid: preset_labels[pid],
                                          key=f"{key_prefix}_propfirm_preset")
    run_sweep = cols[1].button("Run Prop Firm Simulation", key=f"{key_prefix}_propfirm_run",
                                use_container_width=True)
    st.caption("Uses each firm's REAL, sourced evaluation rules (phases, profit targets, drawdown limits) - "
               "not a generic made-up account. Bootstrap-resamples this run's actual R-multiples across "
               "thousands of simulated attempts at each risk level, chained through every real evaluation "
               "phase. Rules change and vary by account type - see the sourcing note below before relying "
               "on this for a real decision.")

    if not run_sweep:
        return

    preset = prop_presets.get_preset(chosen_preset_id)
    with st.spinner(f"Simulating {preset['display_name']} across risk levels - this runs thousands of "
                     f"multi-phase Monte Carlo attempts, may take a few seconds..."):
        risk_levels = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
        results = prop_sim.multi_phase_risk_sweep(trades, preset, risk_levels_pct=risk_levels, n_iter=2000)

    if not results:
        st.warning("Not enough trades to run the sweep.")
        return

    best = max(results, key=lambda r: r["pass_prob"])
    st.markdown(eyebrow(f"BEST FIT - {best['risk_pct_per_trade']:.2f}% RISK/TRADE"), unsafe_allow_html=True)
    with st.container(border=True):
        headline_cols = st.columns(4)
        headline_cols[0].metric("Chance of passing", f"{best['pass_prob'] * 100:.1f}%")
        if math.isnan(best["median_days_to_pass"]):
            headline_cols[1].metric("Days to pass (median)", "n/a")
        else:
            headline_cols[1].metric("Days to pass (median)", f"{best['median_days_to_pass']:.0f}")
        if math.isnan(best["p25_days_to_pass"]):
            headline_cols[2].metric("Typical range", "n/a")
        else:
            headline_cols[2].metric("Typical range",
                                     f"{best['p25_days_to_pass']:.0f}-{best['p75_days_to_pass']:.0f} days")
        headline_cols[3].metric("Trades to pass (median)",
                                 "n/a" if math.isnan(best["median_trades_to_pass"])
                                 else f"{best['median_trades_to_pass']:.0f}")
    st.caption(f"Among simulated attempts that passed every phase of {preset['display_name']} at this risk "
               f"level, half took {best['median_days_to_pass']:.0f} days or less, and the middle 50% of "
               f"attempts fell between {best['p25_days_to_pass']:.0f} and {best['p75_days_to_pass']:.0f} days - "
               f"median and a percentile range instead of a plain average, since time-to-pass is usually "
               f"right-skewed (a few slow-but-still-passing attempts drag a mean average upward)."
               if not math.isnan(best["median_days_to_pass"]) else
               "No simulated attempts passed every phase at this risk level, so there's no time-to-pass "
               "figure to show - see the fail breakdown below.")

    st.markdown(eyebrow("EVERY RISK LEVEL"), unsafe_allow_html=True)
    phase_names = [p["name"] for p in preset["phases"]]
    rows = []
    for row in results:
        rows.append({
            "risk %/trade": f"{row['risk_pct_per_trade']:.2f}%",
            "pass %": row["pass_prob"] * 100,
            "fail %": row["fail_prob"] * 100,
            "median days to pass": row["median_days_to_pass"],
            "typical range (days)": ("n/a" if math.isnan(row["p25_days_to_pass"])
                                       else f"{row['p25_days_to_pass']:.0f}-{row['p75_days_to_pass']:.0f}"),
            "median trades to pass": row["median_trades_to_pass"],
            **{f"fails in {name}": row["fail_by_phase"][i] for i, name in enumerate(phase_names)},
        })
    sweep_df = pd.DataFrame(rows)
    st.dataframe(sweep_df.style.format({"pass %": "{:.1f}%", "fail %": "{:.1f}%",
                                          "median days to pass": "{:.0f}", "median trades to pass": "{:.0f}"},
                                         na_rep="n/a"),
                 use_container_width=True, hide_index=True)

    with st.expander("Sourcing & caveats for this preset"):
        st.caption(f"Source: {', '.join(preset['source_urls'])}")
        st.caption(preset["source_note"])
        st.caption("Bootstrap resampling of a finite historical trade sample approximates future variance, "
                   "it is not a guarantee - especially at the tails. No commission/spread/slippage modeled "
                   "in the underlying trades. Each phase resets to a fresh account balance (real prop-firm "
                   "convention) while continuing to consume the same simulated trade sequence.")


def _render_optimization_result(result, opt_module_name):
    if not result.available:
        st.info("Optimization & robustness data isn't available yet for this strategy.")
        if opt_module_name:
            st.caption(f"Found {opt_module_name} on disk, but: {result.reason}")
        else:
            st.caption("No companion research/<strategy>_optimization.py script exists on disk yet for this "
                       "strategy - other agents may still be building it. This tab will populate automatically "
                       "once one lands and webapp/optimization.py recognizes its function names.")
        return

    if result.heatmap is not None:
        st.markdown(eyebrow("PARAMETER STABILITY HEATMAP (AVG R/TRADE)"), unsafe_allow_html=True)
        st.dataframe(style_diverging_heatmap(result.heatmap), use_container_width=True)
    if result.monte_carlo is not None and not result.monte_carlo.empty:
        st.markdown(eyebrow("MONTE CARLO PER CELL"), unsafe_allow_html=True)
        # signed color+sign applies to R-multiple/P&L columns only - a probability column
        # (P(total R<=0)) isn't a P&L figure, so "positive=good" coloring would be backwards
        # (a HIGH probability of loss is bad news, not a green number)
        non_signed = {"stop_buffer_pct", "fallback_reward_risk", "confirmation_candles", "n_trades"}
        signed_cols = [c for c in result.monte_carlo.columns
                       if c not in non_signed and "p_total_r" not in c and "p_le_0" not in c]
        prob_cols = [c for c in result.monte_carlo.columns if "p_total_r" in c or "p_le_0" in c]
        mc_styler = style_signed_columns(result.monte_carlo, signed_cols, fmt="{:+.4f}")
        if prob_cols and hasattr(mc_styler, "format"):
            mc_styler = mc_styler.format({c: "{:.1%}" for c in prob_cols})
        st.dataframe(mc_styler, use_container_width=True, hide_index=True)
    if result.cluster_verdict is not None:
        st.markdown(eyebrow("CLUSTER / PLATEAU-VS-SPIKE VERDICT"), unsafe_allow_html=True)
        st.write(result.cluster_verdict)
    if result.walk_forward is not None and not result.walk_forward.empty:
        st.markdown(eyebrow("ROLLING WALK-FORWARD VALIDATION"), unsafe_allow_html=True)
        wf_signed = [c for c in result.walk_forward.columns if "total_r" in c or "avg_r" in c]
        st.dataframe(style_signed_columns(result.walk_forward, wf_signed, fmt="{:+.4f}"),
                     use_container_width=True, hide_index=True)
    if result.decay is not None:
        st.markdown(eyebrow("SIGNAL-DECAY DIAGNOSTIC"), unsafe_allow_html=True)
        d = result.decay
        if d.get("confidence") == "insufficient_folds":
            st.caption(f"Insufficient walk-forward folds ({d.get('n_folds_used', 0)}) to pool a "
                       f"months-since-fit decay estimate yet.")
        else:
            dcols = st.columns(3)
            dcols[0].metric("Slope (per month)", f"{d['slope']:+.5f}")
            dcols[1].metric("Half-life (months)",
                             f"{d['half_life_months']:.1f}" if d.get("half_life_months") is not None else "n/a")
            dcols[2].metric("Folds pooled", d.get("n_folds_used", 0))
            verdict = "decay evidence (negative slope)" if d["slope"] < 0 else "no decay evidence (flat/positive slope)"
            st.caption(f"Pooled by months-since-fit across every walk-forward fold's own out-of-sample "
                       f"trades - {verdict}. This is a diagnostic, not a pass/fail gate; it never changes "
                       f"the walk-forward efficiency verdict above.")
    if result.extra_notes:
        st.caption(result.extra_notes)
    if result.reason == "partial":
        st.caption("Some of the four methodology outputs weren't found on the companion module yet - "
                   "showing what is available.")


def render_lockbox_section(strategy_id):
    mapping = optimization.lockbox_param_mapping(strategy_id)
    if mapping is None:
        return

    st.markdown("---")
    st.markdown(eyebrow("LOCKBOX CONFIRMATION - ONE-SHOT, EVER"), unsafe_allow_html=True)

    prior = optimization.lockbox_ledger_status(strategy_id)
    if prior is not None:
        verdict = "PASSED" if prior.get("passed") else "DID NOT PASS"
        st.warning(f"This strategy's lockbox has already been used in this webapp - {verdict} on "
                   f"{prior.get('timestamp', '?')} (lockbox window {prior.get('lockbox_start', '?')} to "
                   f"{prior.get('lockbox_end', '?')}). A lockbox can only ever be confirmed once per "
                   f"strategy, so no button is shown here - see the recorded attempt below.")
        detail_cols = st.columns(4)
        detail_cols[0].metric("Result", verdict)
        detail_cols[1].metric("Total R", f"{prior.get('total_r', 0):+.2f}")
        detail_cols[2].metric("Avg R / trade", f"{prior.get('avg_r', 0):+.4f}")
        detail_cols[3].metric("Trades", prior.get("n_trades", 0))
        return

    with st.container(border=True):
        st.markdown("**This is a genuine one-shot, ledger-enforced final holdout check - not a "
                    "re-runnable report.** It scores your chosen final parameters on a final held-out "
                    "window that STEP 1-4 above never touched, exactly once, ever, for this strategy. "
                    "A second attempt is refused outright (`LockboxAlreadyUsedError`), not just "
                    "discouraged - do not click the button below unless you genuinely mean to spend "
                    "this strategy's one and only lockbox confirmation right now.")

        strategy = STRATEGIES_BY_ID.get(strategy_id)
        final_params = {}
        param_display = []
        if strategy is not None:
            for attr, lockbox_key in mapping.items():
                widget_key = f"{strategy_id}_{attr}"
                spec = next((p for p in strategy.params if p.attr == attr), None)
                value = st.session_state.get(widget_key, spec.default if spec else None)
                final_params[lockbox_key] = value
                param_display.append(f"{attr}={value}")
        st.caption("Final parameters that will be locked in (currently set in the sidebar): "
                   + ", ".join(param_display))

        confirmed = st.checkbox(
            "I understand this can only be run ONCE per strategy, ever, and cannot be undone.",
            key=f"lockbox_ack_{strategy_id}")
        if confirmed:
            if st.button("Run Lockbox Confirmation (ONE-SHOT)", key=f"lockbox_run_{strategy_id}",
                          type="primary"):
                with st.spinner("Running the one-shot lockbox confirmation against real Dukascopy data..."):
                    outcome = optimization.run_lockbox(strategy_id, final_params)
                if outcome.status == "already_used":
                    st.error("Refused: this strategy's lockbox was already used (a race with another "
                             "run, most likely). " + outcome.detail)
                elif outcome.status == "error":
                    st.error(f"Lockbox run failed: {outcome.detail}")
                else:
                    verdict = "PASSED" if outcome.status == "passed" else "DID NOT PASS"
                    st.success(f"Lockbox confirmation complete - {verdict}") if outcome.status == "passed" \
                        else st.warning(f"Lockbox confirmation complete - {verdict}")
                    rcols = st.columns(4)
                    rcols[0].metric("Result", verdict)
                    rcols[1].metric("Total R", f"{outcome.total_r:+.2f}")
                    rcols[2].metric("Avg R / trade", f"{outcome.avg_r:+.4f}")
                    rcols[3].metric("Trades", outcome.n_trades)
                    st.caption(f"Lockbox window: {outcome.lockbox_start} to {outcome.lockbox_end}. "
                               f"Consistency: {outcome.consistency}. This attempt is now permanently "
                               f"recorded - re-opening this page will show the same result, not a new button.")
        else:
            st.button("Run Lockbox Confirmation (ONE-SHOT)", key=f"lockbox_run_disabled_{strategy_id}",
                      disabled=True)


def _to_date(value):
    """instruments/start_date/end_date arrive as real date objects straight off a live run's
    session_state, but as ISO strings when reloaded from run_history's JSON storage - this
    normalizes either into a date object."""
    if isinstance(value, datetime.date):
        return value
    return datetime.date.fromisoformat(str(value)[:10])


def render_generic_param_sweep(strategy, instruments, start_date, end_date):
    """Automatic parameter search for any strategy that doesn't (yet) have a bespoke
    research/<x>_optimization.py companion script - see optimization.run_generic_param_sweep.
    This is what "Optimization & Robustness" shows INSTEAD of manual parameter sliders: no
    raw number inputs anywhere in this app any more, just a button that searches this
    strategy's own parameter ranges automatically against real data."""
    if not strategy.params:
        st.info("This strategy has no tunable parameters to search - its results already reflect "
                 "its one fixed rule set.")
        return

    st.caption(f"No hand-built 4-step methodology script exists yet for this strategy (that's the heavier "
               f"grid-search + Monte Carlo + walk-forward + cluster-check pipeline a couple of strategies "
               f"have). Instead, this automatically searches {optimization.GENERIC_SWEEP_N_COMBINATIONS} "
               f"random parameter combinations across this strategy's own tunable ranges, each one a real "
               f"backtest over the same {len(instruments)} instrument(s) and {start_date} to {end_date} "
               f"window you already ran - no manual sliders to guess at. Lighter-weight than the full "
               f"methodology (no Monte Carlo resampling, no walk-forward validation, no cluster/plateau "
               f"check) - treat this as a quick automatic scan, not full robustness proof.")

    cache_key = f"generic_sweep_{strategy.id}"
    if st.button("Run automatic parameter search", key=f"generic_sweep_run_{strategy.id}", type="primary"):
        module = importlib.import_module(strategy.module_name)
        start_dt = datetime.datetime.combine(_to_date(start_date), datetime.time.min)
        end_dt = datetime.datetime.combine(_to_date(end_date) + datetime.timedelta(days=1), datetime.time.min)
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0, text="Starting...")

        def progress_cb(done, total, label):
            pct = 0.0 if total == 0 else min(done / total, 1.0)
            progress_bar.progress(pct, text=f"{label}")

        with st.spinner("Searching parameter combinations against real Dukascopy data - the first "
                         "combination fetches fresh data, later ones reuse it from cache and are much "
                         "faster..."):
            st.session_state[cache_key] = optimization.run_generic_param_sweep(
                strategy, module, instruments, start_dt, end_dt, progress_cb=progress_cb)
        progress_placeholder.empty()

    result = st.session_state.get(cache_key)
    if result is None:
        st.info("Not run yet this session - click the button above when you're ready to wait for it.")
        return
    if not result.available:
        st.warning(f"Couldn't run the search: {result.reason}")
        return

    st.markdown(eyebrow(f"BEST COMBINATION FOUND ({result.n_combinations} tried)"), unsafe_allow_html=True)
    if result.best is not None:
        with st.container(border=True):
            best_cols = st.columns(len(strategy.params) + 2)
            for i, p in enumerate(strategy.params):
                best_cols[i].metric(p.label, f"{result.best[p.attr]:g}")
            best_cols[-2].metric("Trades", int(result.best["n_trades"]))
            best_cols[-1].metric("Avg R / trade", f"{result.best['avg_r']:+.4f}")
        if result.best.get("is_default"):
            st.caption("This strategy's own hand-picked defaults came out on top of the combinations tried.")

    st.markdown(eyebrow("EVERY COMBINATION TRIED"), unsafe_allow_html=True)
    signed_cols = [c for c in result.table.columns if c in ("total_r", "avg_r")]
    st.dataframe(style_signed_columns(result.table, signed_cols, fmt={"total_r": "{:+.3f}", "avg_r": "{:+.4f}"}),
                 use_container_width=True, hide_index=True)
    st.caption("Sorted best avg R/trade first. Combinations with too few trades to be meaningful "
               f"(<{optimization.GENERIC_SWEEP_MIN_TRADES}) are still shown here but excluded from picking "
               "the best combination above.")


def _render_known_pipeline_section(strategy, known_module):
    """The heavy, real 4-step optimization/robustness pipeline for the couple of strategies with
    a hand-wired research/<x>_optimization.py companion script (optimization.KNOWN_PIPELINES).
    Shared between the Optimization & Robustness tab (reachable only after a plain "Run Backtest"
    has already produced trades) and a standalone button on the run-config page - the companion
    script has its own fixed instrument list and date range baked in (see its own "THE 4 STEPS"
    header), so running it never actually depended on anything picked in the config console or on
    a plain backtest having run first; this just removes that artificial requirement."""
    strategy_id = strategy.id
    st.caption(f"A real companion script ({known_module.rsplit('.', 1)[-1]}.py) exists for this strategy: a full "
               f"parameter-stability grid, Monte Carlo resampling per cell, a cluster/plateau-vs-spike check, and "
               f"a rolling walk-forward validation, run against real Dukascopy data using this script's own fixed "
               f"instrument list and date range (independent of anything picked above). This is genuinely heavy - "
               f"the script's own header warns 30 minutes to well over an hour end to end - so it only runs when "
               f"you explicitly ask for it below.")

    cache_key = f"opt_result_{strategy_id}"
    if st.button("Run Deep Optimization (4-Step)", key=f"deep_opt_run_{strategy_id}"):
        with st.spinner("Running the full 4-step optimization & robustness pass against real Dukascopy data - "
                         "this can take a long time..."):
            st.session_state[cache_key] = optimization.run_known_pipeline(strategy_id)

    result = st.session_state.get(cache_key)
    if result is None:
        st.info("Not run yet this session - click the button above when you're ready to wait for it.")
    else:
        _render_optimization_result(result, strategy.optimization_module or known_module)


def render_optimization_tab(strategy, instruments, start_date, end_date):
    strategy_id = strategy.id if strategy else None
    known_module = optimization.known_pipeline_module_name(strategy_id)

    if known_module is None:
        if strategy is not None:
            render_generic_param_sweep(strategy, instruments, start_date, end_date)
        else:
            st.info("Optimization & robustness data isn't available for this strategy.")
        render_lockbox_section(strategy_id)
        return

    _render_known_pipeline_section(strategy, known_module)
    render_lockbox_section(strategy_id)


def render_run_context(strategy_name, strategy_id, trades, instruments, start_date, end_date, key_prefix):
    st.markdown(f"## {strategy_name}")
    if not trades:
        st.info("No trades were generated for this selection. Try widening the date range, "
                "picking different instruments, or loosening the parameters.")
        return
    strategy = STRATEGIES_BY_ID.get(strategy_id)
    tabs = st.tabs(["Results", "Optimization & Robustness"])
    with tabs[0]:
        render_filterable_results(trades, strategy, key_prefix)
    with tabs[1]:
        render_optimization_tab(strategy, instruments, start_date, end_date)


# --------------------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------------------

def _compare_all_pct_metrics(cost_trades, risk_pct):
    """Compounded %-of-account metrics for one slice (full period, fit window, or holdout
    window) of a Compare All run - shared so the three slices are computed identically. Returns
    None for an empty slice (e.g. a strategy with zero holdout trades), which callers use as
    their own "not enough data in this slice" signal."""
    s = stats_mod.compute_stats(cost_trades)
    if s is None:
        return None
    total_pct = stats_mod.compounded_return_pct(cost_trades, risk_pct)
    total_pct_ci_low, total_pct_ci_high = stats_mod.compounded_return_ci(
        s["avg_r_ci_low"], s["avg_r_ci_high"], risk_pct, s["n_trades"])
    max_drawdown_pct = stats_mod.compounded_max_drawdown_pct(cost_trades, risk_pct)
    return {
        "n_trades": s["n_trades"], "total_pct": total_pct,
        "total_pct_ci_low": total_pct_ci_low, "total_pct_ci_high": total_pct_ci_high,
        "avg_pct_per_trade": s["avg_r"] * risk_pct, "win_pct": s["tp_pct"],
        "max_drawdown_pct": max_drawdown_pct, "z_score": s["z_score"],
    }


def render_compare_all_section():
    st.markdown(eyebrow("COMPARE ALL STRATEGIES"), unsafe_allow_html=True)
    st.caption(f"Runs every one of the {len(STRATEGIES)} strategies in this catalog over the SAME date "
               f"range (each using its own usual instrument list and its own defaults - no manual "
               f"parameters here either). Comparing this many strategies on the exact same window and "
               f"crowning whichever looks best is itself a form of data snooping - the \"winner\" might "
               f"just be the luckiest strategy on that window, not the best one. So ranking uses only the "
               f"LAST {stats_mod.HOLDOUT_FRACTION * 100:.0f}% of the selected range (never seen by the "
               f"ranking decision itself) - the same in-sample/out-of-sample discipline every research/*.py "
               f"script's own SPLIT_DATE convention already uses, applied one level up. The full-period "
               f"and fit-window numbers are still shown for context, just not used to pick a \"best\". "
               f"Typical per-instrument trading costs are deducted from every trade (same as the Results "
               f"page default - see stats.py for sourcing), and strategies under "
               f"{stats_mod.MIN_TRADES_FOR_RANKING} HOLDOUT trades are excluded from ranking entirely, not "
               f"just caveated.")

    with st.container(border=True):
        today = datetime.date.today()
        latest_available = today - datetime.timedelta(days=1)
        widest_default_days = max(s.default_history_days for s in STRATEGIES)
        default_start = latest_available - datetime.timedelta(days=widest_default_days)
        date_range = st.date_input("Date range", value=(default_start, latest_available),
                                     label_visibility="collapsed", key="compare_all_daterange")

        date_range_error = None
        if not (isinstance(date_range, tuple) and len(date_range) == 2):
            date_range_error = "Pick both a start and end date."
        else:
            start_d, end_d = date_range
            clamped_end = min(end_d, latest_available)
            if start_d >= clamped_end:
                date_range_error = f"Start date must be before {latest_available}."
            else:
                date_range = (start_d, clamped_end)

        if date_range_error:
            st.caption(f"⚠ {date_range_error}")
        else:
            span_days = (date_range[1] - date_range[0]).days
            st.caption(f"Currently set to ~{span_days} days ({date_range[0]} to {date_range[1]}). "
                       f"Swing/position strategies (Donchian, MA Cross) need real multi-year history to "
                       f"produce more than a couple of trades - a short range makes them look "
                       f"artificially empty, not necessarily bad.")

        risk_pct_compare = st.number_input("Risk per trade (%) - for the % column below", min_value=0.05,
                                             max_value=10.0, value=1.0, step=0.25, key="compare_all_risk_pct")

        run_all_clicked = st.button("Run All Strategies", type="primary", use_container_width=True,
                                      disabled=bool(date_range_error))
        st.caption("Real fetches against Dukascopy for every strategy, one at a time - with this many "
                   "strategies this can take a long time, especially on a wide date range or first-time "
                   "fetches of a given instrument/range (later strategies sharing an instrument reuse "
                   "the disk cache, so it does get faster partway through).")

    if run_all_clicked and not date_range_error:
        start_dt = datetime.datetime.combine(date_range[0], datetime.time.min)
        end_dt = datetime.datetime.combine(date_range[1] + datetime.timedelta(days=1), datetime.time.min)
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0, text="Starting...")
        results = []
        n_saved = 0
        for i, strategy in enumerate(STRATEGIES):
            progress_bar.progress(i / len(STRATEGIES), text=f"Running {strategy.name} ({i + 1}/{len(STRATEGIES)})...")
            try:
                module = importlib.import_module(strategy.module_name)
                labels = _instrument_labels(strategy)
                raw_trades = strategy.runner(module, labels, start_dt, end_dt, {}, lambda *a: None)
                raw_trades = stats_mod.normalize_trade_dates(raw_trades)
                fit_trades, holdout_trades, split_is_date_based = stats_mod.split_trades_for_holdout(raw_trades)
                full_cost_trades, _n_unadjusted = stats_mod.apply_cost_adjustment(raw_trades)
                fit_cost_trades, _ = stats_mod.apply_cost_adjustment(fit_trades)
                holdout_cost_trades, _ = stats_mod.apply_cost_adjustment(holdout_trades)
                full_m = _compare_all_pct_metrics(full_cost_trades, risk_pct_compare)
                fit_m = _compare_all_pct_metrics(fit_cost_trades, risk_pct_compare)
                holdout_m = _compare_all_pct_metrics(holdout_cost_trades, risk_pct_compare)
            except Exception as exc:
                results.append({"strategy": strategy.name, "n_trades": 0, "error": str(exc)})
                continue
            if full_m is None:
                results.append({"strategy": strategy.name, "n_trades": 0, "error": "no trades produced"})
                continue

            # Every strategy that produced at least one trade also becomes a real Gallery entry -
            # same raw (pre-cost-adjustment) trades a normal "Run Backtest" would save, so Gallery/
            # History treat it identically (its own cost-adjustment toggle, equity curve, etc. all
            # work the same way). Best-effort: a save failure here must never break the comparison
            # itself, which is why this whole block is wrapped separately from the run above.
            run_id = None
            try:
                run_id = run_history.append_run(
                    strategy_name=strategy.name,
                    instruments=labels,
                    start_date=date_range[0],
                    end_date=date_range[1],
                    params={},
                    trades=stats_mod.trades_to_jsonable(raw_trades),
                    name=f"{strategy.name} - Compare All {date_range[0]}",
                )
                n_saved += 1
            except Exception as exc:
                print(f"Compare All: failed to save {strategy.name} to history/gallery: {exc}")

            row = {"strategy": strategy.name, "error": None, "run_id": run_id,
                   "split_is_date_based": split_is_date_based}
            row.update(full_m)   # n_trades, total_pct, total_pct_ci_low/high, avg_pct_per_trade, win_pct,
                                  # max_drawdown_pct, z_score - full-period, shown for context only
            row["fit_n_trades"] = fit_m["n_trades"] if fit_m else 0
            row["fit_total_pct"] = fit_m["total_pct"] if fit_m else None
            # HOLDOUT numbers are what ranking actually uses - prefixed so they can't be confused
            # with the full-period fields above
            for k, v in (holdout_m or {}).items():
                row[f"holdout_{k}"] = v
            row.setdefault("holdout_n_trades", 0)
            results.append(row)
        progress_bar.progress(1.0, text="Done.")
        progress_placeholder.empty()
        st.session_state["compare_all_results"] = results
        st.session_state["compare_all_risk_pct_used"] = risk_pct_compare
        st.session_state["compare_all_n_saved"] = n_saved

    results = st.session_state.get("compare_all_results")
    if not results:
        return

    n_saved = st.session_state.get("compare_all_n_saved", 0)
    if n_saved:
        saved_cols = st.columns([3, 1])
        saved_cols[0].caption(f"Saved {n_saved} of {len(results)} runs to History/Gallery (search "
                               f"\"Compare All\" in the Gallery to find just these) - each with its own "
                               f"equity curve, exactly like a normal single Run Backtest.")
        if saved_cols[1].button("Open Gallery", key="compare_all_open_gallery", use_container_width=True):
            st.session_state.pending_page_nav = "Gallery"
            st.rerun()

    has_trades = [r for r in results if not r.get("error") and r.get("n_trades", 0) > 0]
    empty_or_failed = [r for r in results if r.get("error") or not r.get("n_trades")]
    # A strategy under the trade-count floor in the HOLDOUT window CANNOT win the headline
    # comparison, however good its full-period or holdout return looks - that would just be
    # crowning noise the ranking itself was never protected against. Ranked by HOLDOUT total %,
    # not full-period - see this function's own top caption for why.
    qualifying = [r for r in has_trades if r["holdout_n_trades"] >= stats_mod.MIN_TRADES_FOR_RANKING]
    thin_sample = [r for r in has_trades if r["holdout_n_trades"] < stats_mod.MIN_TRADES_FOR_RANKING]
    qualifying.sort(key=lambda r: -r["holdout_total_pct"])
    thin_sample.sort(key=lambda r: -r["holdout_n_trades"])

    if qualifying:
        best = qualifying[0]
        st.markdown(eyebrow(f"BEST OF {len(qualifying)} QUALIFYING ON HOLDOUT (of {len(results)} total)"),
                    unsafe_allow_html=True)
        with st.container(border=True):
            best_cols = st.columns(5)
            best_cols[0].metric("Holdout Total %", f"{best['holdout_total_pct']:+.2f}%",
                                 help=f"95% CI: {best['holdout_total_pct_ci_low']:+.2f}% to "
                                      f"{best['holdout_total_pct_ci_high']:+.2f}% (normal approximation). "
                                      f"Computed ONLY on the last {stats_mod.HOLDOUT_FRACTION * 100:.0f}% of "
                                      f"the selected range - see this section's top caption.")
            best_cols[1].metric("Holdout avg % / trade", f"{best['holdout_avg_pct_per_trade']:+.3f}%")
            best_cols[2].metric("Holdout trades", best["holdout_n_trades"])
            best_cols[3].metric("Holdout win rate", f"{best['holdout_win_pct']:.1f}%")
            best_cols[4].metric("Holdout max drawdown", f"-{best['holdout_max_drawdown_pct']:.2f}%")
        fit_note = (f"Fit-window total was {best['fit_total_pct']:+.2f}% ({best['fit_n_trades']} trades) - "
                    f"{'consistent direction, a good sign' if (best['fit_total_pct'] or 0) * best['holdout_total_pct'] > 0 else 'OPPOSITE direction from holdout - a real red flag, not just noise'}."
                    if best.get("fit_total_pct") is not None else "No fit-window trades to compare against.")
        st.caption(f"**{best['strategy']}** - only strategies with at least "
                   f"{stats_mod.MIN_TRADES_FOR_RANKING} HOLDOUT trades are eligible to be ranked \"best\" "
                   f"at all; see \"too few trades to rank\" below for the rest. {fit_note}")
    else:
        st.warning(f"None of the {len(results)} strategies produced at least "
                   f"{stats_mod.MIN_TRADES_FOR_RANKING} trades in the holdout window on this date range, "
                   f"so there's no meaningful \"best\" to highlight - widen the range and re-run (a wider "
                   f"range also means a wider holdout slice, not just a wider fit slice).")

    st.markdown(eyebrow(f"LEADERBOARD (RANKED BY HOLDOUT, ≥{stats_mod.MIN_TRADES_FOR_RANKING} HOLDOUT TRADES)"),
                unsafe_allow_html=True)
    rows = [{
        "strategy": r["strategy"], "holdout trades": r["holdout_n_trades"],
        "holdout total %": r["holdout_total_pct"],
        "holdout 95% CI": f"{r['holdout_total_pct_ci_low']:+.1f}% to {r['holdout_total_pct_ci_high']:+.1f}%",
        "fit total %": r.get("fit_total_pct"), "fit trades": r["fit_n_trades"],
        "full-period total %": r["total_pct"], "holdout win %": r["holdout_win_pct"],
        "holdout max DD %": r["holdout_max_drawdown_pct"], "holdout z-score": r["holdout_z_score"],
    } for r in qualifying]
    if rows:
        leaderboard_df = pd.DataFrame(rows)
        st.dataframe(
            style_signed_columns(leaderboard_df, ["holdout total %", "fit total %", "full-period total %"],
                                  fmt={"holdout total %": "{:+.2f}%", "fit total %": "{:+.2f}%",
                                       "full-period total %": "{:+.2f}%"})
            .format({"holdout win %": "{:.1f}%", "holdout max DD %": "-{:.2f}%",
                     "holdout z-score": "{:.2f}"}, na_rep="-"),
            use_container_width=True, hide_index=True)
        st.caption("\"fit total %\" and \"full-period total %\" are shown for context only - a strategy "
                   "whose fit and holdout numbers point in opposite directions is a red flag even if the "
                   "holdout number alone looks fine, since it suggests the edge isn't stable over time.")
    else:
        st.caption("No strategy qualifies for holdout-ranking on this range yet.")

    if thin_sample:
        st.markdown(eyebrow(f"TOO FEW HOLDOUT TRADES TO RANK (<{stats_mod.MIN_TRADES_FOR_RANKING})"),
                    unsafe_allow_html=True)
        st.caption("Shown for reference only - NOT sorted by return, NOT eligible for \"best of\" above. "
                   "A strong-looking % here (holdout or full-period) is not evidence of anything with this "
                   "few holdout trades. A wider date range gives the holdout slice more room to work with.")
        thin_rows = [{
            "strategy": r["strategy"], "holdout trades": r["holdout_n_trades"],
            "holdout total %": r.get("holdout_total_pct"), "full-period trades": r["n_trades"],
            "full-period total %": r["total_pct"], "full-period win %": r["win_pct"],
        } for r in thin_sample]
        thin_df = pd.DataFrame(thin_rows)
        st.dataframe(
            style_signed_columns(thin_df, ["holdout total %", "full-period total %"],
                                  fmt={"holdout total %": "{:+.2f}%", "full-period total %": "{:+.2f}%"})
            .format({"full-period win %": "{:.1f}%"}, na_rep="-"),
            use_container_width=True, hide_index=True)

    if empty_or_failed:
        with st.expander(f"{len(empty_or_failed)} strategies produced no trades or failed on this range"):
            for r in empty_or_failed:
                st.caption(f"**{r['strategy']}**: {r.get('error') or 'no trades in this date range'}")
    st.caption(f"Leaderboard ranked by HOLDOUT total % return (the last {stats_mod.HOLDOUT_FRACTION * 100:.0f}% "
               f"of the selected range, never used to pick the ranking) at "
               f"{st.session_state.get('compare_all_risk_pct_used', 1.0):.2f}% risk/trade, restricted to "
               f"strategies with at least {stats_mod.MIN_TRADES_FOR_RANKING} holdout trades. Same caveats "
               f"as everywhere else in this app: no commission/spread/slippage modeled. This IS an "
               f"out-of-sample check on which strategy to trust, one level up from the individual research "
               f"scripts' own in-sample/out-of-sample splits - it is NOT a live/forward test, since the "
               f"whole \"holdout\" window is still historical data that already happened.")


def _render_strategy_catalog():
    # This is the actual landing state (no strategy picked yet) - the one thing every first-time
    # visitor sees before anything else, so it carries the only "what is this" explainer in the
    # whole app. Kept to a few sentences on purpose: this is a HUD console, not a marketing page.
    st.markdown(eyebrow("STRATEGY CATALOG"), unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**What this is**")
        st.write(f"A backtesting sandbox for {len(STRATEGIES)} algorithmic trading strategies against "
                 f"real historical Dukascopy data - nothing here is mocked or precomputed. Pick a "
                 f"strategy card below to read what it does, then run it. Already know the strategies "
                 f"and just want to see all {len(STRATEGIES)} ranked against each other? Use "
                 f"**Compare All** in the top nav instead.")
        st.caption("⚠ Not financial advice and no guarantee of future performance - see the full "
                   "disclaimer at the bottom of every page.")

    cols_per_row = 3
    for row_start in range(0, len(STRATEGIES), cols_per_row):
        row = STRATEGIES[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, strategy in zip(cols, row):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{strategy.name}**")
                    st.caption(strategy.granularity)
                    st.write(strategy.notes)
                    badges = []
                    if strategy.chart_fetcher:
                        badges.append("Trade chart")
                    if strategy.optimization_module:
                        badges.append("Optimization & robustness")
                    st.markdown(eyebrow(" · ".join(badges) if badges else "CORE BACKTEST"),
                                unsafe_allow_html=True)
                    st.caption(f"{len(strategy.instruments)} instruments - "
                               f"{len(strategy.params)} parameters searched automatically when optimizing")
                    if st.button("Run this strategy", key=f"catalog_run_{strategy.id}", use_container_width=True):
                        # Plain session_state we fully own (not a widget's own key), so it's safe to
                        # set and immediately rerun on - this is what actually transitions the SAME
                        # "Backtest" page from the catalog into the run-config view below, no separate
                        # page/nav jump needed any more.
                        st.session_state.selected_strategy_id = strategy.id
                        st.rerun()

    st.markdown(eyebrow("ADDING A NEW STRATEGY"), unsafe_allow_html=True)
    st.caption("This catalog is meant to keep growing. New strategies get added in "
               "webapp/registry.py - each one is a single StrategyDef entry pointing at a "
               "research/*.py backtest module that already exists; see the comment at the top "
               "of registry.py for the exact shape, or the auto_param() helper for a quicker "
               "way to wire up its tunable parameters without hand-typing every bound.")


def gallery_page():
    st.markdown(eyebrow("BACKTEST GALLERY"), unsafe_allow_html=True)
    runs = run_history.load_runs()
    if not runs:
        st.info("No runs yet - go run a backtest first.")
        return

    if not github_storage.is_configured():
        st.caption("⚠ GitHub-backed history isn't configured yet - runs are saved locally only and "
                   "will be lost if this app restarts (Streamlit Cloud wipes local disk on redeploys "
                   "and sleep/wake cycles). See webapp/github_storage.py's header for one-time setup.")

    strategies_present = sorted({r["strategy"] for r in runs})
    filter_cols = st.columns([2, 1.6, 1.4])
    sel_strategies = filter_cols[0].multiselect("Strategy", strategies_present, default=strategies_present,
                                                  key="gallery_f_strategy")
    sort_options = {
        "Newest first": ("timestamp", True),
        "Oldest first": ("timestamp", False),
        "Total % / R: high to low": ("total_r", True),
        "Total % / R: low to high": ("total_r", False),
        "Most trades": ("n_trades", True),
        "Worst max drawdown": ("max_drawdown_r", True),
        "Name (A-Z)": ("name", False),
    }
    sort_choice = filter_cols[1].selectbox("Sort by", list(sort_options.keys()), key="gallery_sort")
    search = filter_cols[2].text_input("Search name", key="gallery_search", placeholder="Filter by name...",
                                        label_visibility="visible")

    shown = [r for r in runs if r["strategy"] in sel_strategies]
    if search:
        shown = [r for r in shown if search.lower() in (r.get("name") or "").lower()]
    sort_field, reverse = sort_options[sort_choice]
    if sort_field == "name":
        shown.sort(key=lambda r: (r.get("name") or "").lower(), reverse=reverse)
    else:
        shown.sort(key=lambda r: r.get(sort_field, 0) or 0, reverse=reverse)

    st.caption(f"{len(shown)} of {len(runs)} runs shown.")

    cols_per_row = 3
    for row_start in range(0, len(shown), cols_per_row):
        row = shown[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, r in zip(cols, row):
            with col:
                with st.container(border=True):
                    run_id = r["run_id"]
                    current_name = r.get("name") or f"{r['strategy']} - {r['start_date']}"
                    name_cols = st.columns([4, 1])
                    # help= surfaces the FULL name as a hover tooltip - the input box itself is only
                    # as wide as 1/3 of the page (3 cards/row), so a long strategy name (most of
                    # them, once auto-named "<Strategy> - Compare All <date>") truncates with no way
                    # to read the rest otherwise.
                    new_name = name_cols[0].text_input("Name", value=current_name,
                                                          key=f"gallery_name_input_{run_id}",
                                                          label_visibility="collapsed", help=current_name)
                    if name_cols[1].button("💾", key=f"gallery_name_save_{run_id}", help="Save name",
                                             use_container_width=True):
                        if new_name.strip() and new_name.strip() != current_name:
                            run_history.rename_run(run_id, new_name.strip())
                            st.rerun()

                    when = datetime.datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M")
                    st.caption(f"{r['strategy']} · {', '.join(r.get('instruments') or [])}")
                    st.caption(f"{r['start_date']} to {r['end_date']} · run at {when}")

                    # 2 st.metric columns, not 3 - a card is only ~1/3 of the page wide, and a third
                    # narrow metric column was truncating "Max DD" values with an ellipsis
                    # ("-77.4…") no matter the formatting, since st.metric doesn't wrap. Max DD gets
                    # its own full-width caption line instead, which never truncates.
                    metric_cols = st.columns(2)
                    metric_cols[0].metric("Trades", r["n_trades"])
                    metric_cols[1].metric("Total R", f"{r['total_r']:+.2f}")
                    st.caption(f"Max drawdown: -{r.get('max_drawdown_r', 0.0):.2f}R")
                    if r["n_trades"] < stats_mod.MIN_TRADES_FOR_RANKING:
                        st.caption(f"⚠ Small sample (<{stats_mod.MIN_TRADES_FOR_RANKING} trades) - "
                                   f"directional only, not meaningful evidence either way.")

                    trades = run_history.load_trades_for_run(run_id)
                    if trades:
                        xs, equity, chronological = stats_mod.dollar_equity_curve(trades, risk_pct=1.0)
                        render_dollar_equity_chart(xs, equity, chronological, key=f"gallery_chart_{run_id}",
                                                     height=140, compact=True)
                    else:
                        st.caption("Trade-level detail not found for this run.")

                    if st.button("View full results", key=f"gallery_view_{run_id}", use_container_width=True):
                        st.session_state.pending_history_run_id = run_id
                        st.session_state.pending_page_nav = "History"
                        st.rerun()


def _render_run_config(strategy):
    strategy_names = [s.name for s in STRATEGIES]

    back_l, back_r = st.columns([1, 4])
    with back_l:
        if st.button("← Back to catalog", key="backtest_back_to_catalog", use_container_width=True):
            st.session_state.selected_strategy_id = None
            st.rerun()
    with back_r:
        st.markdown(eyebrow(strategy.name), unsafe_allow_html=True)

    # config console - a single compact row, no captions under each control (detail moved into
    # tooltips) - pick 3 things, hit Run. The Strategy dropdown here still lets you flip between
    # all strategies without going back to the catalog card grid; picking one updates
    # selected_strategy_id too, so "Back to catalog" and this dropdown always agree.
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.3, 1.6, 1.3])
        with c1:
            chosen_name = st.selectbox("Strategy", strategy_names,
                                        index=strategy_names.index(strategy.name))
            strategy = next(s for s in STRATEGIES if s.name == chosen_name)
            st.session_state.selected_strategy_id = strategy.id
        with c2:
            labels = _instrument_labels(strategy)
            selected_instruments = st.multiselect("Instruments", labels, default=labels,
                                                     key=f"{strategy.id}_instruments")
        with c3:
            today = datetime.date.today()
            latest_available = today - datetime.timedelta(days=1)   # today's trading day isn't complete yet
            default_end = latest_available
            default_start = default_end - datetime.timedelta(days=strategy.default_history_days)
            # No max_value cap on the widget itself - a hard cap produces a confusing invalid/red-error
            # state the moment someone picks "today" (a completely natural thing to try), and on that
            # invalid state Streamlit was returning the OLD default range to Python instead of what was
            # actually typed - silently running the wrong window while LOOKING like the app ignored the
            # selection entirely. Instead: accept whatever's picked, then clamp/validate it in plain code
            # below, so the caption and the actual run always agree on exactly the same range.
            date_range = st.date_input("Date range", value=(default_start, default_end),
                                          key=f"{strategy.id}_daterange",
                                          help="Widen or narrow freely - first fetches of a wide range can "
                                               "take many minutes. Capped to yesterday (today's trading day "
                                               "isn't complete yet).")

            date_range_error = None
            if not (isinstance(date_range, tuple) and len(date_range) == 2):
                date_range_error = "Pick both a start and end date."
            else:
                start_d, end_d = date_range
                clamped_end = min(end_d, latest_available)
                if start_d >= clamped_end:
                    date_range_error = (f"Start date must be before {latest_available} (today's trading day "
                                         f"isn't complete yet, so data only goes up to yesterday).")
                else:
                    date_range = (start_d, clamped_end)

            if date_range_error:
                st.caption(f"⚠ {date_range_error}")
            elif date_range[1] != end_d:
                st.caption("⚠ End date clamped to yesterday.")

        # No manual parameter tweaking here any more - every plain Run Backtest uses this
        # strategy's own fixed defaults. Finding a better combination automatically is what the
        # Optimization & Robustness tab (after a run) is for, not hand-guessed sidebar sliders.
        param_values = {}

        run_clicked = st.button("Run Backtest", type="primary", use_container_width=True,
                                 help="Fetches real historical data live from Dukascopy - nothing here is "
                                      "mocked or precomputed.")

    # Standalone entry point into the heavy 4-step pipeline (grid search, Monte Carlo, walk-forward,
    # cluster/plateau check) for the couple of strategies that have one - previously only reachable
    # from the Optimization & Robustness tab, which itself only appeared after a plain "Run Backtest"
    # had already produced trades. That requirement was never real: the companion script uses its own
    # fixed instrument list and date range, not anything picked in the console above, so it never
    # actually depended on a prior run - this just exposes it directly instead of gating it behind one.
    known_module = optimization.known_pipeline_module_name(strategy.id)
    if known_module is not None:
        st.markdown("---")
        st.markdown(eyebrow("DEEP OPTIMIZATION (4-STEP)"), unsafe_allow_html=True)
        _render_known_pipeline_section(strategy, known_module)

    if run_clicked:
        if not selected_instruments:
            st.error("Select at least one instrument above.")
            return
        if date_range_error:
            st.error(f"Fix the date range above before running: {date_range_error}")
            return

        module = importlib.import_module(strategy.module_name)
        start_dt = datetime.datetime.combine(date_range[0], datetime.time.min)
        end_dt = datetime.datetime.combine(date_range[1] + datetime.timedelta(days=1), datetime.time.min)

        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0, text="Starting...")

        def progress_cb(done, total, label):
            pct = 0.0 if total == 0 else min(done / total, 1.0)
            progress_bar.progress(pct, text=f"{label} ({done}/{total} instruments)")

        try:
            with st.spinner("Running backtest - fetching real Dukascopy data and running the strategy. "
                             "This can take a while on wide date ranges or first-time fetches."):
                trades = strategy.runner(module, selected_instruments, start_dt, end_dt, dict(param_values),
                                           progress_cb)
        except Exception as exc:
            progress_placeholder.empty()
            st.error(f"The backtest run failed: {exc}")
            st.caption("This is usually a Dukascopy connectivity problem (network/proxy/rate limit) rather than "
                       "a bug in the strategy itself - check your network connection and try again. The full "
                       "traceback was printed to the terminal running `streamlit run`.")
            import traceback
            print("Backtest run failed:")
            traceback.print_exc()
            return
        progress_placeholder.empty()

        trades = stats_mod.normalize_trade_dates(trades)
        run_id = run_history.append_run(
            strategy_name=strategy.name,
            instruments=selected_instruments,
            start_date=date_range[0],
            end_date=date_range[1],
            params=param_values,
            trades=stats_mod.trades_to_jsonable(trades),
        )
        st.session_state.last_run = {
            "strategy_id": strategy.id,
            "strategy_name": strategy.name,
            "trades": trades,
            "run_id": run_id,
            "instruments": selected_instruments,
            "start_date": date_range[0],
            "end_date": date_range[1],
        }
        st.rerun()

    if st.session_state.last_run:
        st.markdown("---")
        render_run_context(
            st.session_state.last_run["strategy_name"],
            st.session_state.last_run["strategy_id"],
            st.session_state.last_run["trades"],
            st.session_state.last_run["instruments"],
            st.session_state.last_run["start_date"],
            st.session_state.last_run["end_date"],
            key_prefix="live",
        )


def backtest_page():
    # One continuous flow instead of two separate nav pages: land on the catalog (what/why +
    # strategy cards), and picking a strategy transitions this SAME page into the run-config +
    # results view rather than jumping somewhere else. selected_strategy_id is the only thing
    # deciding which half renders.
    selected_id = st.session_state.get("selected_strategy_id")
    strategy = next((s for s in STRATEGIES if s.id == selected_id), None) if selected_id else None
    if strategy is None:
        _render_strategy_catalog()
    else:
        _render_run_config(strategy)


def compare_all_page():
    # Its own top-level nav item rather than bundled at the top of the strategy catalog - this is
    # a distinct, power-user workflow ("I already know these strategies, rank all N at once"), not
    # part of the "what is this / pick one" onboarding path, and it shouldn't compete with that
    # path for the first thing a new visitor sees.
    render_compare_all_section()


def history_page():
    st.markdown(eyebrow("RUN HISTORY"), unsafe_allow_html=True)
    st.caption("Every completed backtest run from this tool, newest first.")

    runs = run_history.load_runs()
    if not runs:
        st.info("No runs yet - go run a backtest first.")
        return

    table_rows = []
    for r in runs:
        table_rows.append({
            "name": r.get("name") or f"{r['strategy']} - {r['start_date']}",
            "when": datetime.datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M"),
            "strategy": r["strategy"],
            "instruments": ", ".join(r.get("instruments") or []),
            "range": f"{r['start_date']} to {r['end_date']}",
            "trades": r["n_trades"],
            "total_r": round(r["total_r"], 2),
            "avg_r": round(r["avg_r"], 4),
            "max_drawdown_r": round(r.get("max_drawdown_r", 0.0), 2),
            "run_id": r["run_id"],
        })
    df = pd.DataFrame(table_rows)
    st.dataframe(df.drop(columns=["run_id"]), use_container_width=True, hide_index=True)

    st.markdown("### Re-view a past run")
    # jump here from a "View full results" click on the Gallery page - pre-selects that run
    # for exactly this rerun, same pattern as Browse Strategies -> Run Backtest
    pending_run_id = st.session_state.pop("pending_history_run_id", None)
    option_labels = [f"{row['name']} ({row['when']}, {row['trades']} trades)" for row in table_rows]
    default_index = 0
    if pending_run_id:
        for i, row in enumerate(table_rows):
            if row["run_id"] == pending_run_id:
                default_index = i
                break
    choice = st.selectbox("Pick a run", option_labels, index=default_index, label_visibility="collapsed")
    if not choice:
        return
    run_id = table_rows[option_labels.index(choice)]["run_id"]
    trades = run_history.load_trades_for_run(run_id)
    if trades is None:
        st.warning("Trade-level detail wasn't found on disk for this run.")
        return
    trades = stats_mod.normalize_trade_dates(trades)
    matching_row = next(r for r in runs if r["run_id"] == run_id)
    strategy = next((s for s in STRATEGIES if s.name == matching_row["strategy"]), None)
    st.markdown("---")
    if not trades:
        st.info("This run produced no trades.")
        return
    tabs = st.tabs(["Results", "Optimization & Robustness"])
    with tabs[0]:
        render_filterable_results(trades, strategy, key_prefix=f"hist_{run_id}")
    with tabs[1]:
        render_optimization_tab(strategy, matching_row.get("instruments") or [],
                                 matching_row["start_date"], matching_row["end_date"])


# --------------------------------------------------------------------------------------
# top bar - brand + page nav, replacing the old left sidebar. A HUD console strip across
# the top instead of a 2006-era left rail; both pages share it.
# --------------------------------------------------------------------------------------

pending_page_nav = st.session_state.pop("pending_page_nav", None)
if pending_page_nav:
    st.session_state.page_nav = pending_page_nav

header_l, header_r = st.columns([2, 1])
with header_l:
    st.markdown('<div class="brand">&#9889; STRATEGY BACKTESTS</div>', unsafe_allow_html=True)
with header_r:
    page = st.segmented_control("Page", ["Backtest", "Compare All", "Gallery", "History"],
                                 default="Backtest", label_visibility="collapsed", key="page_nav")
st.markdown('<hr class="brand-rule"/>', unsafe_allow_html=True)

if page == "History":
    history_page()
elif page == "Compare All":
    compare_all_page()
elif page == "Gallery":
    gallery_page()
else:
    backtest_page()

# Footer, not the top bar - the top bar was deliberately stripped down earlier (explicit
# feedback that it felt cluttered), so this lives at the bottom instead: present on every page,
# out of the way of the actual workflow. DRAFT language, not reviewed by a lawyer - needs real
# legal review before this app is used to make any actual trading or purchase decision, same
# framing as every other "not measured real data" caveat already in this project.
st.markdown("---")
st.caption(
    "Not financial advice. Nothing on this site is a recommendation to buy, sell, or trade any "
    "instrument, strategy, or prop firm challenge. Every number here comes from a backtest or a "
    "resample of a backtest - a model of the past, not a prediction of the future - and none of "
    "it accounts for live spread, slippage, commission, execution risk, or your own broker's "
    "actual conditions. Past performance (real or backtested) does not guarantee future results. "
    "This tool is provided for research and educational purposes only, with no warranty of "
    "accuracy or completeness, and is not a substitute for independent financial or legal "
    "advice. [Draft disclaimer - not reviewed by a lawyer.]"
)
