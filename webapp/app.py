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
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # webapp/ itself (registry, stats, ...)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (research/ package)

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

    st.markdown(eyebrow("FILTERS"), unsafe_allow_html=True)
    n_cols = 3 + len(facets)
    cols = st.columns(n_cols)
    sel_outcomes = cols[0].multiselect("Outcome", outcomes, default=outcomes, key=f"{key_prefix}_f_outcome")
    sel_instruments = cols[1].multiselect("Instrument", instruments, default=instruments, key=f"{key_prefix}_f_instrument")
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

    st.caption(f"{len(filtered)} of {len(trades)} trades match the current filters. "
               f"Filtering never re-fetches data - it only recomputes over the already-run trade list.")

    s = stats_mod.compute_stats(filtered)
    if s is None:
        st.warning("No trades match the current filters.")
        return

    st.markdown(eyebrow("DISPLAY"), unsafe_allow_html=True)
    display_cols = st.columns([1, 1, 2])
    display_mode = display_cols[0].radio("Units", ["% of account", "R-multiples"], index=0,
                                          horizontal=True, key=f"{key_prefix}_display_mode",
                                          label_visibility="collapsed")
    risk_pct = 1.0
    if display_mode == "% of account":
        risk_pct = display_cols[1].number_input("Risk per trade (%)", min_value=0.05, max_value=10.0,
                                                   value=1.0, step=0.25, key=f"{key_prefix}_risk_pct",
                                                   help="Assumed % of account risked per trade - converts each "
                                                        "trade's R-multiple into an account % (r x risk%). This "
                                                        "is a DISPLAY assumption, not something the backtest "
                                                        "itself used - the underlying trades never change.")
        display_cols[2].caption(f"Every number below is r × {risk_pct:.2f}% - purely a unit conversion, "
                                 f"same trades either way.")
    display_trades = stats_mod.scale_trades_r(filtered, risk_pct) if display_mode == "% of account" else filtered
    unit_label = "%" if display_mode == "% of account" else "R"
    unit_fmt = "{:+.2f}%" if display_mode == "% of account" else "{:+.3f}R"
    s_display = stats_mod.compute_stats(display_trades)

    st.markdown(eyebrow("PERFORMANCE METRICS"), unsafe_allow_html=True)
    with st.container(border=True):
        metric_cols = st.columns(6)
        metric_cols[0].metric("Trades", s_display["n_trades"])
        metric_cols[1].metric(f"Total {unit_label}", unit_fmt.format(s_display["total_r"]))
        metric_cols[2].metric(f"Avg {unit_label} / trade", (unit_fmt if unit_label == "R" else "{:+.3f}%")
                               .format(s_display["avg_r"]))
        metric_cols[3].metric("Win rate (TP)", f"{s_display['tp_pct']:.1f}%")
        metric_cols[4].metric("Loss rate (SL)", f"{s_display['sl_pct']:.1f}%")
        metric_cols[5].metric("Approx z-score", f"{s_display['z_score']:.2f}")
    if s_display["n_trades"] < 100:
        st.caption(f"Only {s_display['n_trades']} trades - too few to trust the z-score regardless of its value.")
    st.caption("Multiple comparisons: this project has shipped many strategies, several with their own "
               "parameter grid searches - a single strategy's z-score in isolation isn't strong evidence, "
               "since data-snooping risk compounds across every strategy and parameter combination tried "
               "project-wide, not just this one. (z-score is the same number in either display unit above - "
               "it's scale-invariant.)")

    st.markdown(eyebrow(f"EQUITY CURVE (CUMULATIVE {unit_label})"), unsafe_allow_html=True)
    xs, ys, chronological = stats_mod.equity_curve(display_trades)
    fig = go.Figure()
    # neon-glow line: wide, low-opacity copies of the same trace stacked behind the crisp
    # main line - a standard "HUD glow" trick, not a real visual effect Plotly has natively
    for glow_width, glow_opacity in ((14, 0.06), (8, 0.10), (4, 0.16)):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                  line=dict(width=glow_width, color=ACCENT),
                                  opacity=glow_opacity, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=2, color=ACCENT),
                              name=f"Cumulative {unit_label}"))
    fig.update_layout(
        height=320,
        xaxis_title="Trade sequence (chronological)" if chronological else "Trade sequence",
        yaxis_title=f"Cumulative {unit_label}",
        showlegend=False,
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_equity_chart")
    if not chronological:
        st.caption("This strategy's trade records don't carry a date field upstream - order shown is "
                   "per-instrument backtest sequence, not calendar order.")

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

    render_trade_chart_section(strategy, filtered, key_prefix)
    render_prop_firm_fit_section(filtered, key_prefix)


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

    st.markdown(eyebrow("TRADE CHART"), unsafe_allow_html=True)

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
    st.markdown(eyebrow("PROP FIRM FIT"), unsafe_allow_html=True)
    if len(trades) < 10:
        st.caption("Not enough trades in this run (need at least 10) to run a meaningful prop-firm simulation.")
        return

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

    phase_names = [p["name"] for p in preset["phases"]]
    rows = []
    for row in results:
        rows.append({
            "risk %/trade": f"{row['risk_pct_per_trade']:.2f}%",
            "pass %": row["pass_prob"] * 100,
            "fail %": row["fail_prob"] * 100,
            "avg trades to pass": row["avg_trades_to_pass"],
            "avg days to pass": row["avg_days_to_pass"],
            **{f"fails in {name}": row["fail_by_phase"][i] for i, name in enumerate(phase_names)},
        })
    sweep_df = pd.DataFrame(rows)
    st.dataframe(sweep_df.style.format({"pass %": "{:.1f}%", "fail %": "{:.1f}%",
                                          "avg trades to pass": "{:.0f}", "avg days to pass": "{:.0f}"},
                                         na_rep="n/a"),
                 use_container_width=True, hide_index=True)

    best = max(results, key=lambda r: r["pass_prob"])
    st.markdown(f"**Best risk level: {best['risk_pct_per_trade']:.2f}% per trade -> "
                f"{best['pass_prob'] * 100:.1f}% chance of clearing every phase of {preset['display_name']}.**")

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


def render_optimization_tab(strategy_id, opt_module_name):
    known_module = optimization.known_pipeline_module_name(strategy_id)

    if known_module is None:
        # generic fallback path - lightweight introspection only, safe to run immediately
        result = optimization.load_optimization_result(opt_module_name)
        _render_optimization_result(result, opt_module_name)
        return

    st.caption(f"A real companion script ({known_module.rsplit('.', 1)[-1]}.py) exists for this strategy: a full "
               f"parameter-stability grid, Monte Carlo resampling per cell, a cluster/plateau-vs-spike check, and "
               f"a rolling walk-forward validation, run against real Dukascopy data. This is genuinely heavy - "
               f"the script's own header warns 30 minutes to well over an hour end to end - so it only runs when "
               f"you explicitly ask for it below, never automatically on opening this tab.")

    cache_key = f"opt_result_{strategy_id}"
    if st.button("Run full optimization & robustness pass", key=f"opt_run_{strategy_id}"):
        with st.spinner("Running the full 4-step optimization & robustness pass against real Dukascopy data - "
                         "this can take a long time..."):
            st.session_state[cache_key] = optimization.run_known_pipeline(strategy_id)

    result = st.session_state.get(cache_key)
    if result is None:
        st.info("Not run yet this session - click the button above when you're ready to wait for it.")
    else:
        _render_optimization_result(result, opt_module_name or known_module)

    render_lockbox_section(strategy_id)


def render_run_context(strategy_name, strategy_id, trades, key_prefix):
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
        render_optimization_tab(strategy_id, strategy.optimization_module if strategy else None)


# --------------------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------------------

def run_backtest_page():
    strategy_names = [s.name for s in STRATEGIES]

    # config console - a single horizontal HUD strip replacing the old left sidebar. All
    # run controls live here, top of page, above the results - nothing tucked in a side rail.
    with st.container(border=True):
        st.markdown(eyebrow("BACKTEST CONFIG"), unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1.3, 1.6, 1.3])
        with c1:
            st.markdown('<div class="console-label">Strategy</div>', unsafe_allow_html=True)
            chosen_name = st.selectbox("Strategy", strategy_names, label_visibility="collapsed")
            strategy = next(s for s in STRATEGIES if s.name == chosen_name)
            st.caption(f"{strategy.granularity}. {strategy.notes}")
        with c2:
            st.markdown('<div class="console-label">Instruments</div>', unsafe_allow_html=True)
            labels = _instrument_labels(strategy)
            selected_instruments = st.multiselect("Instruments", labels, default=labels,
                                                     label_visibility="collapsed", key=f"{strategy.id}_instruments")
        with c3:
            st.markdown('<div class="console-label">Date range</div>', unsafe_allow_html=True)
            default_end = datetime.date.today() - datetime.timedelta(days=1)
            default_start = default_end - datetime.timedelta(days=strategy.default_history_days)
            date_range = st.date_input("Date range", value=(default_start, default_end),
                                          max_value=default_end, label_visibility="collapsed",
                                          key=f"{strategy.id}_daterange")
            default_window_label = f"~{strategy.default_history_days / 365:.0f}-year" if strategy.default_history_days >= 365 else "6-month"
            st.caption(f"Defaults to a short {default_window_label} window - widen deliberately, "
                       f"first fetches of a wide range can take many minutes.")

        # Manual parameter tweaking is a power-user feature, not the default flow - most people
        # don't know what STOP_BUFFER_PCT should be and shouldn't have to. This runs with the
        # strategy's own defaults unless deliberately opened and changed; the "find the best
        # combo automatically" path is the Optimization & Robustness tab after a run, not this.
        param_values = {}
        with st.expander("Advanced parameters (optional)", expanded=False):
            st.caption("Leave these alone unless you know what they do. Prefer the Optimization & "
                       "Robustness tab after running once - it searches many combinations "
                       "automatically instead of you guessing values here.")
            param_cols = st.columns(3) if strategy.params else []
            for i, p in enumerate(strategy.params):
                widget_key = f"{strategy.id}_{p.attr}"
                target = param_cols[i % 3]
                if p.kind == "int":
                    param_values[p.attr] = target.number_input(p.label, value=int(p.default), min_value=int(p.min_value),
                                                              max_value=int(p.max_value), step=int(p.step),
                                                              key=widget_key, help=p.help or None)
                else:
                    param_values[p.attr] = target.number_input(p.label, value=float(p.default), min_value=float(p.min_value),
                                                              max_value=float(p.max_value), step=float(p.step),
                                                              key=widget_key, help=p.help or None)

        run_clicked = st.button("Run Backtest", type="primary", use_container_width=True)
        st.caption("Every run fetches real historical data live from Dukascopy - nothing here is mocked or "
                   "precomputed. No commission, spread, or slippage is modeled, matching every underlying "
                   "research script's own caveat.")

    if run_clicked:
        if not selected_instruments:
            st.error("Select at least one instrument in the sidebar.")
            return
        if not isinstance(date_range, tuple) or len(date_range) != 2:
            st.error("Pick a full start and end date in the sidebar.")
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
        }
        st.rerun()

    if st.session_state.last_run:
        st.markdown("---")
        render_run_context(
            st.session_state.last_run["strategy_name"],
            st.session_state.last_run["strategy_id"],
            st.session_state.last_run["trades"],
            key_prefix="live",
        )


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
            "when": datetime.datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M"),
            "strategy": r["strategy"],
            "instruments": ", ".join(r.get("instruments") or []),
            "range": f"{r['start_date']} to {r['end_date']}",
            "trades": r["n_trades"],
            "total_r": round(r["total_r"], 2),
            "avg_r": round(r["avg_r"], 4),
            "run_id": r["run_id"],
        })
    df = pd.DataFrame(table_rows)
    st.dataframe(df.drop(columns=["run_id"]), use_container_width=True, hide_index=True)

    st.markdown("### Re-view a past run")
    options = {f"{row['when']} - {row['strategy']} ({row['trades']} trades)": row["run_id"] for row in table_rows}
    choice = st.selectbox("Pick a run", list(options.keys()), label_visibility="collapsed")
    if not choice:
        return
    run_id = options[choice]
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
        render_optimization_tab(strategy.id if strategy else None, strategy.optimization_module if strategy else None)


# --------------------------------------------------------------------------------------
# top bar - brand + page nav, replacing the old left sidebar. A HUD console strip across
# the top instead of a 2006-era left rail; both pages share it.
# --------------------------------------------------------------------------------------

header_l, header_r = st.columns([2, 1])
with header_l:
    st.markdown('<div class="brand">&#9889; STRATEGY BACKTESTS</div>', unsafe_allow_html=True)
with header_r:
    page = st.segmented_control("Page", ["Run Backtest", "History"], default="Run Backtest",
                                 label_visibility="collapsed", key="page_nav")
st.markdown('<hr class="brand-rule"/>', unsafe_allow_html=True)

if page == "History":
    history_page()
else:
    run_backtest_page()
