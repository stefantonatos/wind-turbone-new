# Strategy registry - the only place that knows how to import and call each
# research/*.py module. Nothing in research/ is ever edited: every entry below
# only imports the module (importlib, a plain module-level import under the hood)
# and calls its already-existing fetch/backtest functions, after overriding a
# handful of its module-level config constants (date range, instrument subset,
# tunable parameters exposed in the sidebar) - Python functions read module
# globals at CALL time, so setting `module.FETCH_START = ...` before calling
# `module.fetch_instrument_data(...)` is enough, no importlib.reload needed.
#
# TO ADD A NEW STRATEGY: add one more StrategyDef to STRATEGIES at the bottom of
# this file. If the new module follows the day_trading_rauf_dukascopy_backtest.py
# shape (own disk cache, fetch_instrument_data(label, const), backtest_instrument
# (label, df) -> trades) - true for 5 of the 9 strategies below so far - just alias
# `runner=_run_with_own_cache`, no new function needed. Otherwise write a small
# `_run_xxx` function following the shape of the other entries below. Nothing else
# in the app needs to change - app.py only ever talks to the StrategyDef objects,
# never to research/*.py directly.
#
# Quick checklist:
#   1. `mod = _load_module("research.<your_module>")`
#   2. `instruments=mod.INSTRUMENTS`
#   3. `params=[auto_param(mod, "SOME_CONST", "Human label"), ...]` for each tunable
#      constant (see auto_param() below - it reads the module's own current value as
#      the default and infers sensible bounds, so you're not hand-typing min/max/step
#      for every one; pass min_value/max_value/step explicitly to override the guess).
#   4. `runner=_run_with_own_cache` (or a bespoke `_run_xxx`, see above).
#   5. `optimization_module=_find_optimization_module("research.<your_module>")` (auto-
#      detects a companion _optimization.py if one exists on disk - safe to always
#      include, it's a no-op if there isn't one yet).
#   6. Trade candlestick charting (the Results tab's TRADE CHART section) is opt-in:
#      only wire up `chart_fetcher=_chart_fetch_po3_shape` or
#      `=_chart_fetch_own_cache_shape` once the module's trades.append(...) calls
#      also record entry_price/stop_price/target_price/exit_price/entry_time/exit_time
#      - see ict_po3_forex_dukascopy_backtest.py for a worked example. Leave it unset
#      (default None) otherwise; the chart section just stays hidden for that strategy.

import datetime
import importlib
from dataclasses import dataclass
from typing import Callable, Optional

from data_cache import cached_dukascopy_fetch


@dataclass
class ParamSpec:
    """One tunable number exposed as a sidebar input. `attr` is the exact
    module-level constant name it overrides (read the module's source - this
    is deliberately not guessed); it is applied via setattr(module, attr, value)
    for every entry except the handful of params documented as function
    arguments instead (those are handled inside that strategy's `_run_*`)."""
    attr: str
    label: str
    kind: str  # "float" or "int"
    default: float
    min_value: float
    max_value: float
    step: float
    help: str = ""


def auto_param(module, attr, label, min_value=None, max_value=None, step=None, help=""):
    """Builds a ParamSpec from a module's OWN current value for `attr` - reads the live
    default straight off the module (so it never drifts out of sync with the research
    script), infers int vs float from that value's type, and fills in a sensible
    min/max/step range scaled off the default's magnitude when not given explicitly.
    Cuts the copy-paste boilerplate of hand-writing every numeric bound when wiring up
    a new strategy; override min_value/max_value/step/help for anything that needs a
    tighter or more meaningful range than the generic guess (e.g. a 0-100 oscillator
    threshold instead of a magnitude-scaled one)."""
    default = getattr(module, attr)
    kind = "int" if isinstance(default, int) else "float"
    if min_value is None:
        min_value = 0 if default >= 0 else default * 3
    if max_value is None:
        max_value = max(default * 5, default + (10 if kind == "int" else 1.0))
    if step is None:
        step = 1 if kind == "int" else round(max(abs(default) * 0.1, 0.01), 4)
    return ParamSpec(attr, label, kind, default, min_value, max_value, step, help)


@dataclass
class StrategyDef:
    id: str
    name: str
    module_name: str          # research.<module>, importlib.import_module target
    granularity: str          # human label for the sidebar caption
    instruments: list         # [(label, ...rest)] straight from the module's own instrument list
    params: list              # list[ParamSpec]
    facets: list              # extra trade-dict keys usable as filter facets beyond side/outcome/instrument/date
    notes: str                # caveats shown under the strategy picker
    runner: Callable          # (module, selected_labels, start_dt, end_dt, overrides, progress_cb) -> list[trade dict]
    optimization_module: Optional[str] = None  # research.<x>_optimization module name, if one exists
    default_history_days: int = 182  # sidebar date-range default width; swing/daily-bar strategies
                                       # need a much wider default than the intraday ones to see any
                                       # signal at all (a 50/200-day SMA cross needs 200+ days of
                                       # history just to warm up, before a rare cross can even fire)
    chart_fetcher: Optional[Callable] = None  # (module, label, const, start_dt, end_dt) -> OHLC df, for
                                       # the trade candlestick chart. None means this strategy's trade
                                       # dicts don't carry entry/stop/target/exit price+time yet - the
                                       # chart section hides itself rather than showing a broken chart.


def _instrument_labels(strategy_def):
    return [row[0] for row in strategy_def.instruments]


def _apply_overrides(module, overrides):
    for attr, value in overrides.items():
        setattr(module, attr, value)


def _run_with_own_cache(module, selected_labels, start_dt, end_dt, overrides, progress_cb):
    """Shared runner for every module that follows the day_trading_rauf_dukascopy_backtest.py
    shape exactly: its own pickle disk cache (CACHE_DIR, redirected below into webapp/cache
    rather than the repo root), fetch_instrument_data(label, instrument_const) -> df, and
    backtest_instrument(label, df) -> trades (keys: side, outcome, r, date). Used by Rauf,
    Donchian/Turtle, MA Golden/Death Cross, Bollinger Band mean-reversion, and RSI
    mean-reversion - all five share this exact shape, confirmed by reading each module's
    source directly, not assumed from the name."""
    import os
    module.FETCH_START, module.FETCH_END = start_dt, end_dt
    _apply_overrides(module, overrides)
    module.CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache",
                                      f"{module.__name__.rsplit('.', 1)[-1]}_dukascopy_cache")
    chosen = [row for row in module.INSTRUMENTS if row[0] in selected_labels]
    all_trades = []
    with cached_dukascopy_fetch():  # backstop only - these modules already cache their combined df themselves
        for done, (label, const) in enumerate(chosen):
            progress_cb(done, len(chosen), label)
            df = module.fetch_instrument_data(label, const)
            if df is None or df.empty:
                continue
            for t in module.backtest_instrument(label, df):
                t["instrument"] = label
                all_trades.append(t)
        progress_cb(len(chosen), len(chosen), "done")
    return all_trades


# ---------------------------------------------------------------------------
# ict_po3_forex_dukascopy_backtest.py (PO3)
# Exposes: INSTRUMENTS [(label, const)], FETCH_START, FETCH_END, STOP_BUFFER_PCT,
# FALLBACK_REWARD_RISK, MIN_RANGE_PCT, ACCUMULATION_START/END, MANIPULATION_START/END,
# fetch_instrument_data(instrument_const) -> df, backtest_instrument(label, df) -> trades
# (trade keys: side, outcome, r, date). No own disk cache -> generic cache used.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# chart fetchers - a SHORT re-fetch (just the padded window around one trade,
# not the whole backtest range) for the per-trade candlestick chart on the
# results page. Mirrors the corresponding runner's fetch_instrument_data call
# shape exactly, just with start_dt/end_dt set to the trade's own window
# instead of the run's full range - a fresh, small, separately-cached lookup,
# not a slice of the (potentially huge) full-history cache entry.
# ---------------------------------------------------------------------------
def _chart_fetch_po3_shape(module, label, const, start_dt, end_dt):
    module.FETCH_START, module.FETCH_END = start_dt, end_dt
    with cached_dukascopy_fetch():
        return module.fetch_instrument_data(const)


def _chart_fetch_own_cache_shape(module, label, const, start_dt, end_dt):
    import os
    module.FETCH_START, module.FETCH_END = start_dt, end_dt
    module.CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache",
                                      f"{module.__name__.rsplit('.', 1)[-1]}_dukascopy_cache")
    with cached_dukascopy_fetch():
        return module.fetch_instrument_data(label, const)


def _run_po3(module, selected_labels, start_dt, end_dt, overrides, progress_cb):
    module.FETCH_START, module.FETCH_END = start_dt, end_dt
    _apply_overrides(module, overrides)
    chosen = [row for row in module.INSTRUMENTS if row[0] in selected_labels]
    all_trades = []
    with cached_dukascopy_fetch():
        for done, (label, const) in enumerate(chosen):
            progress_cb(done, len(chosen), label)
            df = module.fetch_instrument_data(const)
            if df is None or df.empty:
                continue
            for t in module.backtest_instrument(label, df):
                t["instrument"] = label
                all_trades.append(t)
        progress_cb(len(chosen), len(chosen), "done")
    return all_trades


# ---------------------------------------------------------------------------
# day_trading_rauf_dukascopy_backtest.py (Scam or Slam / Rauf)
# Exposes: INSTRUMENTS [(label, const)], FETCH_START, FETCH_END, STOP_BUFFER_PCT,
# CACHE_DIR (own pickle disk cache, redirected into webapp/cache below rather
# than the repo root), fetch_instrument_data(label, instrument_const) -> df,
# backtest_instrument(label, df) -> trades (keys: side, outcome, r, date, range).
# Uses the shared _run_with_own_cache helper - see that function's docstring for
# the full list of modules confirmed to share this exact shape.
# ---------------------------------------------------------------------------
_run_rauf = _run_with_own_cache


# ---------------------------------------------------------------------------
# donchian_turtle_breakout_dukascopy_backtest.py (Donchian / Turtle breakout)
# SWING/POSITION system on daily channels (5-min bars used only for intrabar
# stop/trail checks) - a trade can stay open for days or weeks. Exposes:
# INSTRUMENTS [(label, const)], FETCH_START, FETCH_END, CACHE_DIR (own disk
# cache), DONCHIAN_PERIOD, EXIT_PERIOD, ATR_PERIOD, ATR_STOP_MULT,
# fetch_instrument_data(label, instrument_const) -> df, backtest_instrument(label,
# df) -> trades (keys: side, outcome, r, date). Same shape as Rauf's module.
# ---------------------------------------------------------------------------
_run_donchian = _run_with_own_cache


# ---------------------------------------------------------------------------
# ma_golden_death_cross_dukascopy_backtest.py (MA Golden/Death Cross)
# SWING/POSITION system on a 50/200-day SMA cross (5-min bars used only for
# intrabar ATR-stop checks and a realistic next-day fill) - inherently rare
# signals, a handful of trades per instrument over a 9-year history is normal,
# not a bug. Exposes: INSTRUMENTS [(label, const)], FETCH_START, FETCH_END,
# CACHE_DIR (own disk cache), SMA_FAST, SMA_SLOW, ATR_PERIOD, ATR_STOP_MULT,
# fetch_instrument_data(label, instrument_const) -> df, backtest_instrument(label,
# df) -> trades (keys: side, outcome, r, date). Same shape as Rauf's module.
# ---------------------------------------------------------------------------
_run_ma_cross = _run_with_own_cache


# ---------------------------------------------------------------------------
# bollinger_band_mean_reversion_dukascopy_backtest.py (Bollinger Band fade)
# Intraday, 5-min bars, no session window. Exposes: INSTRUMENTS [(label, const)],
# FETCH_START, FETCH_END, CACHE_DIR (own disk cache), BB_LENGTH, BB_NUM_STD,
# STOP_BUFFER_PCT, MIN_SL_PCT, MAX_HOLD_BARS, fetch_instrument_data(label,
# instrument_const) -> df, backtest_instrument(label, df) -> trades (keys: side,
# outcome, r, date). Same shape as Rauf's module.
# ---------------------------------------------------------------------------
_run_bollinger = _run_with_own_cache


# ---------------------------------------------------------------------------
# rsi_mean_reversion_dukascopy_backtest.py (RSI mean-reversion)
# Intraday, 5-min bars, no session window. Exposes: INSTRUMENTS [(label, const)],
# FETCH_START, FETCH_END, CACHE_DIR (own disk cache), RSI_LENGTH, RSI_OVERSOLD,
# RSI_OVERBOUGHT, STOP_LOOKBACK_BARS, STOP_BUFFER_PCT, MIN_SL_PCT, REWARD_RISK,
# MAX_HOLD_BARS, fetch_instrument_data(label, instrument_const) -> df,
# backtest_instrument(label, df) -> trades (keys: side, outcome, r, date). Same
# shape as Rauf's module.
# ---------------------------------------------------------------------------
_run_rsi = _run_with_own_cache


# ---------------------------------------------------------------------------
# asian_range_breakout_dukascopy_backtest.py (Asian Range Breakout)
# Trades WITH a close-confirmed breakout of the prior evening's Asian session
# range during the London-open window - the mechanical opposite philosophy
# from PO3 (which fades that kind of break). Exposes: INSTRUMENTS
# [(label, const)], FETCH_START, FETCH_END, CACHE_DIR (own disk cache),
# STOP_BUFFER_PCT, TARGET_RANGE_MULT, FALLBACK_REWARD_RISK, MIN_RANGE_PCT,
# fetch_instrument_data(label, instrument_const) -> df, backtest_instrument(label,
# df) -> trades (keys: side, outcome, r, date). Same shape as Rauf's module.
# ---------------------------------------------------------------------------
_run_asian_range_breakout = _run_with_own_cache


# ---------------------------------------------------------------------------
# dow_theory_swing_structure_dukascopy_backtest.py (Dow Theory Swing Structure)
# SWING/POSITION trend-following system on confirmed daily HH/HL (uptrend) or
# LH/LL (downtrend) swing pivots, with a trailing stop that ratchets to each new
# confirmed pivot (reuses Donchian/Turtle's trailing mechanic) - a selective
# filter by design, expect a modest handful to a few dozen trades per instrument
# over 9 years, not hundreds. Exposes: INSTRUMENTS [(label, const)], FETCH_START,
# FETCH_END, CACHE_DIR (own disk cache), SWING_LEN, fetch_instrument_data(label,
# instrument_const) -> df, backtest_instrument(label, df) -> trades (keys: side,
# outcome, r, date). Same shape as Rauf's module.
# ---------------------------------------------------------------------------
_run_dow_theory = _run_with_own_cache


# ---------------------------------------------------------------------------
# bollinger_squeeze_breakout_dukascopy_backtest.py (Bollinger Squeeze Breakout)
# NOT THE SAME STRATEGY AS bollinger_mean_reversion below - this is the
# mechanical OPPOSITE: waits for the bands to visibly contract (a volatility
# "squeeze", a relative/rolling-percentile condition) then trades WITH the
# breakout once price closes outside a band, instead of fading a band touch.
# Exposes: INSTRUMENTS [(label, const)], FETCH_START, FETCH_END, CACHE_DIR (own
# disk cache), BB_LENGTH, BB_NUM_STD, SQUEEZE_LOOKBACK, SQUEEZE_PERCENTILE,
# SQUEEZE_RECENCY_BARS, MIN_SL_PCT, REWARD_RISK, MAX_HOLD_BARS,
# fetch_instrument_data(label, instrument_const) -> df, backtest_instrument(label,
# df) -> trades (keys: side, outcome, r, date). Same shape as Rauf's module.
# ---------------------------------------------------------------------------
_run_bollinger_squeeze = _run_with_own_cache


# ---------------------------------------------------------------------------
# climax_volume_reversal_dukascopy_backtest.py (Climax Volume Reversal)
# Native 15-MIN bars (DUKASCOPY_INTERVAL = INTERVAL_MIN_15, not 5-min like every
# other strategy here - the module's own fetch_instrument_data already fetches
# at its own configured interval, so no special handling needed for that part).
# DOES NOT follow the shared _run_with_own_cache shape: backtest_instrument
# takes a THIRD argument, `use_volume_filter` (bool), which main() decides
# EMPIRICALLY at run time via volume_field_is_usable() - applied only if the
# volume field looks usable (non-zero fraction, enough distinct values) on
# EVERY instrument being tested this run, dropped for the whole run otherwise
# (never a silent per-instrument split). This runner reproduces that exact
# empirical decision (across only the SELECTED instruments, not the module's
# full default list) rather than hardcoding or bypassing it - see the module's
# own header note 4 and main() for the source of this logic.
# Exposes: INSTRUMENTS [(label, const)], FETCH_START, FETCH_END, CACHE_DIR (own
# disk cache), ATR_LENGTH, CLIMAX_RANGE_ATR_MULT, RUN_LOOKBACK_BARS,
# CLIMAX_VOLUME_MULT, PENDING_ORDER_EXPIRY_BARS, REWARD_RISK, MAX_HOLD_HOURS,
# MIN_SL_PCT, fetch_instrument_data(label, instrument_const) -> df,
# volume_field_is_usable(volumes) -> bool, backtest_instrument(label, df,
# use_volume_filter) -> trades (keys: side, outcome, r, date).
# ---------------------------------------------------------------------------
def _run_climax_volume_reversal(module, selected_labels, start_dt, end_dt, overrides, progress_cb):
    import os
    module.FETCH_START, module.FETCH_END = start_dt, end_dt
    _apply_overrides(module, overrides)
    module.CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache",
                                      "climax_volume_reversal_dukascopy_cache")
    chosen = [row for row in module.INSTRUMENTS if row[0] in selected_labels]
    dfs = {}
    with cached_dukascopy_fetch():
        for done, (label, const) in enumerate(chosen):
            progress_cb(done, len(chosen), f"{label} (fetching)")
            df = module.fetch_instrument_data(label, const)
            if df is not None and not df.empty:
                dfs[label] = df

        # empirical volume-usability decision, reproduced exactly from main(): usable only if
        # EVERY fetched instrument's own volume field looks real, dropped for the whole run
        # otherwise - never bypassed, never hardcoded either way.
        per_instrument_usable = {}
        for label, df in dfs.items():
            volumes = df["volume"].tolist() if "volume" in df.columns else []
            per_instrument_usable[label] = module.volume_field_is_usable(volumes)
        use_volume_filter = bool(dfs) and all(per_instrument_usable.values())

        all_trades = []
        for done, (label, df) in enumerate(dfs.items()):
            progress_cb(done, len(dfs), f"{label} (backtesting)")
            for t in module.backtest_instrument(label, df, use_volume_filter):
                t["instrument"] = label
                all_trades.append(t)
        progress_cb(len(dfs), len(dfs), "done")
    return all_trades


# ---------------------------------------------------------------------------
# support_resistance_zone_bounce_dukascopy_backtest.py (S/R zone bounce)
# Exposes: INSTRUMENTS [(label, const)], FETCH_START, FETCH_END, DAILY bars,
# ATR_LEN, ZONE_PIVOT_LOOKBACK, ZONE_WIDTH_ATR_MULT, ZONE_BREAK_BUFFER_ATR_MULT,
# MAX_TARGET_DISTANCE_ATR_MULT, FALLBACK_REWARD_RISK, MAX_HOLD_BARS,
# fetch_daily_ohlc(instrument_const) -> df, backtest_instrument(label, df) -> trades
# (keys: side, outcome, r, date, entry, sl_distance).
# ---------------------------------------------------------------------------
def _run_sr_zone_bounce(module, selected_labels, start_dt, end_dt, overrides, progress_cb):
    module.FETCH_START, module.FETCH_END = start_dt, end_dt
    _apply_overrides(module, overrides)
    chosen = [row for row in module.INSTRUMENTS if row[0] in selected_labels]
    all_trades = []
    with cached_dukascopy_fetch():
        for done, (label, const) in enumerate(chosen):
            progress_cb(done, len(chosen), label)
            df = module.fetch_daily_ohlc(const)
            if df is None or len(df) < 30:
                continue
            for t in module.backtest_instrument(label, df):
                t["instrument"] = label
                all_trades.append(t)
        progress_cb(len(chosen), len(chosen), "done")
    return all_trades


# ---------------------------------------------------------------------------
# ict_silver_bullet_forex_dukascopy_backtest.py (ICT Silver Bullet)
# Exposes: INSTRUMENTS [(label, const)], FETCH_START, FETCH_END, STOP_BUFFER_PCT,
# fetch_instrument_data(instrument_const) -> df, precompute_days(df) -> ind dict,
# run_backtest(ind, reward_risk, split_before=None, split_after=None) -> trades
# (keys: side, outcome, r, date, window). The module's own main() grid-searches
# REWARD_RISK on an in-sample/out-of-sample split; this webapp instead exposes
# reward_risk directly as one sidebar number and runs the FULL selected range in
# one pass (split_before/split_after both left None), since a live UI run is a
# single deliberate choice, not a grid search.
# ---------------------------------------------------------------------------
def _run_silver_bullet(module, selected_labels, start_dt, end_dt, overrides, progress_cb):
    module.FETCH_START, module.FETCH_END = start_dt, end_dt
    reward_risk = overrides.pop("reward_risk", 2.0)
    _apply_overrides(module, overrides)
    chosen = [row for row in module.INSTRUMENTS if row[0] in selected_labels]
    all_trades = []
    with cached_dukascopy_fetch():
        for done, (label, const) in enumerate(chosen):
            progress_cb(done, len(chosen), label)
            df = module.fetch_instrument_data(const)
            if df is None or df.empty:
                continue
            ind = module.precompute_days(df)
            for t in module.run_backtest(ind, reward_risk):
                t["instrument"] = label
                all_trades.append(t)
        progress_cb(len(chosen), len(chosen), "done")
    return all_trades


# ---------------------------------------------------------------------------
# orb_indices_dukascopy_backtest.py (ORB indices)
# Exposes: INDICES [(label, const, tz_name, session_start)], START_DATE, END_DATE
# (note: NOT named FETCH_START/FETCH_END like every other module here), RANGE_MINUTES,
# ENTRY_WINDOW_MINUTES, ENTRY_BUFFER_PCT, REWARD_RISK, MIN_RANGE_PCT, plus several
# boolean filter toggles (RANGE_ATR_FILTER etc). fetch and backtest are NOT
# separable - backtest_index(label, instrument_const, tz_name, session_start)
# fetches internally and returns trades directly (keys: side, outcome, r; NO
# "date" key - the equity curve for this strategy falls back to trade-sequence
# order per instrument, noted in the UI).
# ---------------------------------------------------------------------------
def _run_orb_indices(module, selected_labels, start_dt, end_dt, overrides, progress_cb):
    module.START_DATE, module.END_DATE = start_dt, end_dt
    _apply_overrides(module, overrides)
    chosen = [row for row in module.INDICES if row[0] in selected_labels]
    all_trades = []
    with cached_dukascopy_fetch():
        for done, (label, const, tz_name, session_start) in enumerate(chosen):
            progress_cb(done, len(chosen), label)
            for t in module.backtest_index(label, const, tz_name, session_start):
                t.setdefault("instrument", label)
                t.setdefault("date", None)
                all_trades.append(t)
        progress_cb(len(chosen), len(chosen), "done")
    return all_trades


# ---------------------------------------------------------------------------
# evendyer_vwap_orb_dukascopy_backtest.py (EvenDyer VWAP ORB / "Scam Or Slam" VWAP)
# US index CFDs ONLY (SP500/NASDAQ100/DOWJONES) - the 0930-1000 NY opening range and
# VWAP session windows only make sense on US equity index hours, not forex. Exposes:
# INSTRUMENTS [(label, const)], FETCH_START, FETCH_END, CACHE_DIR (own disk cache),
# STOP_SWING_LEN, TARGET_SWING_LEN, BUFFER_AMOUNT, MAX_TRADES_PER_DAY,
# fetch_instrument_data(label, instrument_const) -> df, backtest_instrument(label, df,
# verbose=True, record_trace=False) -> (trades, used_real_volume, diagnostic_str) - a
# 3-TUPLE, not a bare trade list like the _run_with_own_cache shape, so this needs its
# own runner rather than aliasing that shared one. Trade keys: side, outcome, r, date,
# stop_pct (no entry/stop/target/exit price+time yet, so no chart_fetcher wired).
# ---------------------------------------------------------------------------
def _run_vwap_orb(module, selected_labels, start_dt, end_dt, overrides, progress_cb):
    import os
    module.FETCH_START, module.FETCH_END = start_dt, end_dt
    _apply_overrides(module, overrides)
    module.CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache",
                                      "evendyer_vwap_orb_dukascopy_cache")
    chosen = [row for row in module.INSTRUMENTS if row[0] in selected_labels]
    all_trades = []
    with cached_dukascopy_fetch():
        for done, (label, const) in enumerate(chosen):
            progress_cb(done, len(chosen), label)
            df = module.fetch_instrument_data(label, const)
            if df is None or df.empty:
                continue
            trades, _used_real_volume, _diag = module.backtest_instrument(label, df, verbose=False)
            for t in trades:
                t["instrument"] = label
                all_trades.append(t)
        progress_cb(len(chosen), len(chosen), "done")
    return all_trades


def _load_module(module_name):
    return importlib.import_module(module_name)


def _find_optimization_module(module_name):
    """Companion optimization script naming convention used across this project:
    <backtest module name with '_backtest' replaced by '_optimization'>. Returns
    the module name string if research/<name>.py exists on disk, else None. Other
    agents are building these in parallel - this only detects them, it never
    assumes a specific function signature beyond what optimization.py documents."""
    import os
    if not module_name.endswith("_backtest"):
        return None
    candidate = module_name[: -len("_backtest")] + "_optimization"
    research_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research")
    path = os.path.join(research_dir, candidate.split(".")[-1] + ".py")
    if os.path.isfile(path):
        return f"research.{candidate.split('.')[-1]}"
    return None


def _build_registry():
    entries = []

    po3_mod = _load_module("research.ict_po3_forex_dukascopy_backtest")
    entries.append(StrategyDef(
        id="po3",
        name="ICT Power of Three (PO3)",
        module_name="research.ict_po3_forex_dukascopy_backtest",
        granularity="5-min bars",
        instruments=po3_mod.INSTRUMENTS,
        params=[
            ParamSpec("STOP_BUFFER_PCT", "Stop buffer (% of price)", "float", po3_mod.STOP_BUFFER_PCT, 0.0, 1.0, 0.01),
            ParamSpec("FALLBACK_REWARD_RISK", "Fallback reward:risk", "float", po3_mod.FALLBACK_REWARD_RISK, 0.5, 10.0, 0.5),
            ParamSpec("MIN_RANGE_PCT", "Min accumulation range (% of price)", "float", po3_mod.MIN_RANGE_PCT, 0.0, 1.0, 0.01),
        ],
        facets=[],
        notes="One trade/day/instrument. Trades against the first accumulation-range break each morning.",
        runner=_run_po3,
        optimization_module=_find_optimization_module("research.ict_po3_forex_dukascopy_backtest"),
        chart_fetcher=_chart_fetch_po3_shape,
    ))

    rauf_mod = _load_module("research.day_trading_rauf_dukascopy_backtest")
    entries.append(StrategyDef(
        id="rauf",
        name="Scam or Slam (Day Trading Rauf)",
        module_name="research.day_trading_rauf_dukascopy_backtest",
        granularity="5-min bars",
        instruments=rauf_mod.INSTRUMENTS,
        params=[
            ParamSpec("STOP_BUFFER_PCT", "Stop buffer (% of price)", "float", rauf_mod.STOP_BUFFER_PCT, 0.0, 1.0, 0.01),
        ],
        facets=["range"],
        notes="Sweep + 3-candle-reversal confirmation on two daily opening ranges (London, NY).",
        runner=_run_rauf,
        optimization_module=_find_optimization_module("research.day_trading_rauf_dukascopy_backtest"),
        chart_fetcher=_chart_fetch_own_cache_shape,
    ))

    donchian_mod = _load_module("research.donchian_turtle_breakout_dukascopy_backtest")
    entries.append(StrategyDef(
        id="donchian_turtle",
        name="Donchian / Turtle Breakout",
        module_name="research.donchian_turtle_breakout_dukascopy_backtest",
        granularity="5-min bars driving daily channels (swing/position - trades can stay open days-weeks)",
        instruments=donchian_mod.INSTRUMENTS,
        params=[
            ParamSpec("DONCHIAN_PERIOD", "Entry channel (prior days)", "int", donchian_mod.DONCHIAN_PERIOD, 5, 60, 1),
            ParamSpec("EXIT_PERIOD", "Trailing exit channel (prior days)", "int", donchian_mod.EXIT_PERIOD, 3, 40, 1),
            ParamSpec("ATR_PERIOD", "ATR length (days)", "int", donchian_mod.ATR_PERIOD, 2, 60, 1),
            ParamSpec("ATR_STOP_MULT", "Initial stop (x ATR)", "float", donchian_mod.ATR_STOP_MULT, 0.5, 6.0, 0.5),
        ],
        facets=[],
        notes="Swing/position system on DAILY channels - far fewer, much longer-held trades than the "
              "intraday strategies above. Defaults to a 3-year window (not the app-wide 6 months) since "
              "a 20-day entry channel needs real history to produce more than a couple of breakouts.",
        runner=_run_donchian,
        optimization_module=_find_optimization_module("research.donchian_turtle_breakout_dukascopy_backtest"),
        default_history_days=3 * 365,
        chart_fetcher=_chart_fetch_own_cache_shape,
    ))

    ma_cross_mod = _load_module("research.ma_golden_death_cross_dukascopy_backtest")
    entries.append(StrategyDef(
        id="ma_golden_death_cross",
        name="MA Golden/Death Cross",
        module_name="research.ma_golden_death_cross_dukascopy_backtest",
        granularity="5-min bars driving a daily 50/200 SMA cross (swing/position)",
        instruments=ma_cross_mod.INSTRUMENTS,
        params=[
            ParamSpec("SMA_FAST", "Fast SMA (days)", "int", ma_cross_mod.SMA_FAST, 5, 100, 5),
            ParamSpec("SMA_SLOW", "Slow SMA (days)", "int", ma_cross_mod.SMA_SLOW, 50, 300, 10),
            ParamSpec("ATR_PERIOD", "ATR length (days)", "int", ma_cross_mod.ATR_PERIOD, 2, 60, 1),
            ParamSpec("ATR_STOP_MULT", "Stop / R basis (x ATR)", "float", ma_cross_mod.ATR_STOP_MULT, 0.5, 8.0, 0.5),
        ],
        facets=[],
        notes="A 50/200-day SMA cross is BY DESIGN rare - low single digits to a dozen crosses per "
              "instrument over a 9-year history is normal, not a bug. Defaults to a 3-year window (the "
              "SMA(200) alone needs ~200 trading days just to warm up); 0 trades for some instruments "
              "even at that width is expected - widen further for a better chance of seeing a cross.",
        runner=_run_ma_cross,
        optimization_module=_find_optimization_module("research.ma_golden_death_cross_dukascopy_backtest"),
        default_history_days=3 * 365,
        chart_fetcher=_chart_fetch_own_cache_shape,
    ))

    bollinger_mod = _load_module("research.bollinger_band_mean_reversion_dukascopy_backtest")
    entries.append(StrategyDef(
        id="bollinger_mean_reversion",
        name="Bollinger Band Mean-Reversion",
        module_name="research.bollinger_band_mean_reversion_dukascopy_backtest",
        granularity="5-min bars",
        instruments=bollinger_mod.INSTRUMENTS,
        params=[
            ParamSpec("BB_LENGTH", "Band length (bars)", "int", bollinger_mod.BB_LENGTH, 5, 100, 5),
            ParamSpec("BB_NUM_STD", "Band width (x stddev)", "float", bollinger_mod.BB_NUM_STD, 0.5, 4.0, 0.25),
            ParamSpec("STOP_BUFFER_PCT", "Stop buffer (% of price)", "float", bollinger_mod.STOP_BUFFER_PCT, 0.0, 1.0, 0.01),
            ParamSpec("MIN_SL_PCT", "Min stop distance (% of price)", "float", bollinger_mod.MIN_SL_PCT, 0.0, 1.0, 0.01),
            ParamSpec("MAX_HOLD_BARS", "Max hold (5-min bars)", "int", bollinger_mod.MAX_HOLD_BARS, 6, 288, 6),
        ],
        facets=[],
        notes="Confirmed two-bar re-entry into the bands (not a raw touch), target = the moving average "
              "at entry. Runs continuously, not tied to a session window.",
        runner=_run_bollinger,
        optimization_module=_find_optimization_module("research.bollinger_band_mean_reversion_dukascopy_backtest"),
        chart_fetcher=_chart_fetch_own_cache_shape,
    ))

    rsi_mod = _load_module("research.rsi_mean_reversion_dukascopy_backtest")
    entries.append(StrategyDef(
        id="rsi_mean_reversion",
        name="RSI Mean-Reversion",
        module_name="research.rsi_mean_reversion_dukascopy_backtest",
        granularity="5-min bars",
        instruments=rsi_mod.INSTRUMENTS,
        params=[
            ParamSpec("RSI_LENGTH", "RSI length (bars)", "int", rsi_mod.RSI_LENGTH, 2, 50, 1),
            ParamSpec("RSI_OVERSOLD", "Oversold threshold", "float", rsi_mod.RSI_OVERSOLD, 5.0, 45.0, 1.0),
            ParamSpec("RSI_OVERBOUGHT", "Overbought threshold", "float", rsi_mod.RSI_OVERBOUGHT, 55.0, 95.0, 1.0),
            ParamSpec("STOP_LOOKBACK_BARS", "Stop lookback (bars)", "int", rsi_mod.STOP_LOOKBACK_BARS, 3, 60, 1),
            ParamSpec("STOP_BUFFER_PCT", "Stop buffer (% of price)", "float", rsi_mod.STOP_BUFFER_PCT, 0.0, 1.0, 0.01),
            ParamSpec("MIN_SL_PCT", "Min stop distance (% of price)", "float", rsi_mod.MIN_SL_PCT, 0.0, 1.0, 0.01),
            ParamSpec("REWARD_RISK", "Reward:risk", "float", rsi_mod.REWARD_RISK, 0.5, 6.0, 0.5),
            ParamSpec("MAX_HOLD_BARS", "Max hold (5-min bars)", "int", rsi_mod.MAX_HOLD_BARS, 6, 288, 6),
        ],
        facets=[],
        notes="Confirmed re-entry back through RSI(14) 30/70 (not a raw threshold touch), fixed-R:R "
              "target off a swing-low/high stop. Runs continuously, not tied to a session window.",
        runner=_run_rsi,
        optimization_module=_find_optimization_module("research.rsi_mean_reversion_dukascopy_backtest"),
        chart_fetcher=_chart_fetch_own_cache_shape,
    ))

    asian_mod = _load_module("research.asian_range_breakout_dukascopy_backtest")
    entries.append(StrategyDef(
        id="asian_range_breakout",
        name="Asian Range Breakout",
        module_name="research.asian_range_breakout_dukascopy_backtest",
        granularity="5-min bars",
        instruments=asian_mod.INSTRUMENTS,
        params=[
            ParamSpec("STOP_BUFFER_PCT", "Stop buffer (% of price)", "float", asian_mod.STOP_BUFFER_PCT, 0.0, 1.0, 0.01),
            ParamSpec("TARGET_RANGE_MULT", "Measured-move target (x Asian range)", "float", asian_mod.TARGET_RANGE_MULT, 0.5, 5.0, 0.25),
            ParamSpec("FALLBACK_REWARD_RISK", "Fallback reward:risk", "float", asian_mod.FALLBACK_REWARD_RISK, 0.5, 10.0, 0.5),
            ParamSpec("MIN_RANGE_PCT", "Min Asian range (% of price)", "float", asian_mod.MIN_RANGE_PCT, 0.0, 1.0, 0.01),
        ],
        facets=[],
        notes="Trades WITH a close-confirmed break of the prior evening's Asian session range during "
              "the London open - the mechanical opposite of ICT Power of Three above (which fades that "
              "same kind of break).",
        runner=_run_asian_range_breakout,
        optimization_module=_find_optimization_module("research.asian_range_breakout_dukascopy_backtest"),
    ))

    dow_theory_mod = _load_module("research.dow_theory_swing_structure_dukascopy_backtest")
    entries.append(StrategyDef(
        id="dow_theory_swing_structure",
        name="Dow Theory Swing Structure",
        module_name="research.dow_theory_swing_structure_dukascopy_backtest",
        granularity="5-min bars driving daily swing pivots (swing/position - trades can stay open weeks-months)",
        instruments=dow_theory_mod.INSTRUMENTS,
        params=[
            ParamSpec("SWING_LEN", "Swing pivot confirmation (days each side)", "int", dow_theory_mod.SWING_LEN, 5, 60, 1),
        ],
        facets=[],
        notes="Trend-following on confirmed daily higher-highs/higher-lows (or lower-highs/lower-lows) "
              "swing structure, with a trailing stop that ratchets to each new confirmed pivot - the "
              "same trailing mechanic as Donchian/Turtle above. A selective filter by design: expect a "
              "modest handful to a few dozen trades per instrument over 9 years, not hundreds. Defaults "
              "to a 3-year window like the other swing/daily-bar strategies above.",
        runner=_run_dow_theory,
        optimization_module=_find_optimization_module("research.dow_theory_swing_structure_dukascopy_backtest"),
        default_history_days=3 * 365,
    ))

    bollinger_squeeze_mod = _load_module("research.bollinger_squeeze_breakout_dukascopy_backtest")
    entries.append(StrategyDef(
        id="bollinger_squeeze_breakout",
        name="Bollinger Squeeze Breakout",
        module_name="research.bollinger_squeeze_breakout_dukascopy_backtest",
        granularity="5-min bars",
        instruments=bollinger_squeeze_mod.INSTRUMENTS,
        params=[
            ParamSpec("BB_LENGTH", "Band length (bars)", "int", bollinger_squeeze_mod.BB_LENGTH, 5, 100, 5),
            ParamSpec("BB_NUM_STD", "Band width (x stddev)", "float", bollinger_squeeze_mod.BB_NUM_STD, 0.5, 4.0, 0.25),
            ParamSpec("SQUEEZE_LOOKBACK", "Squeeze lookback (bars)", "int", bollinger_squeeze_mod.SQUEEZE_LOOKBACK, 20, 400, 20),
            ParamSpec("SQUEEZE_PERCENTILE", "Squeeze percentile threshold", "float", bollinger_squeeze_mod.SQUEEZE_PERCENTILE, 1.0, 50.0, 1.0),
            ParamSpec("SQUEEZE_RECENCY_BARS", "Squeeze recency window (bars)", "int", bollinger_squeeze_mod.SQUEEZE_RECENCY_BARS, 1, 30, 1),
            ParamSpec("MIN_SL_PCT", "Min stop distance (% of price)", "float", bollinger_squeeze_mod.MIN_SL_PCT, 0.0, 1.0, 0.01),
            ParamSpec("REWARD_RISK", "Reward:risk", "float", bollinger_squeeze_mod.REWARD_RISK, 0.5, 6.0, 0.5),
            ParamSpec("MAX_HOLD_BARS", "Max hold (5-min bars)", "int", bollinger_squeeze_mod.MAX_HOLD_BARS, 6, 288, 6),
        ],
        facets=[],
        notes="NOT the same strategy as Bollinger Band Mean-Reversion above - this is the mechanical "
              "OPPOSITE: waits for the bands to visibly contract (a volatility squeeze) then trades WITH "
              "the breakout once price closes outside a band, instead of fading a band touch. Compare "
              "the two side by side deliberately, don't conflate them.",
        runner=_run_bollinger_squeeze,
        optimization_module=_find_optimization_module("research.bollinger_squeeze_breakout_dukascopy_backtest"),
    ))

    climax_mod = _load_module("research.climax_volume_reversal_dukascopy_backtest")
    entries.append(StrategyDef(
        id="climax_volume_reversal",
        name="Climax Volume Reversal",
        module_name="research.climax_volume_reversal_dukascopy_backtest",
        granularity="15-min bars (native fetch, not resampled from 5-min like every other strategy here)",
        instruments=climax_mod.INSTRUMENTS,
        params=[
            ParamSpec("ATR_LENGTH", "ATR length (15-min bars)", "int", climax_mod.ATR_LENGTH, 2, 60, 1),
            ParamSpec("CLIMAX_RANGE_ATR_MULT", "Climax range floor (x ATR)", "float", climax_mod.CLIMAX_RANGE_ATR_MULT, 1.0, 8.0, 0.5),
            ParamSpec("RUN_LOOKBACK_BARS", "Prior-run lookback (bars)", "int", climax_mod.RUN_LOOKBACK_BARS, 2, 40, 1),
            ParamSpec("CLIMAX_VOLUME_MULT", "Climax volume floor (x trailing avg)", "float", climax_mod.CLIMAX_VOLUME_MULT, 1.0, 8.0, 0.5,
                       help="Only applied if the volume field looks usable on every selected instrument this run - see the strategy note."),
            ParamSpec("PENDING_ORDER_EXPIRY_BARS", "Pending stop-entry expiry (bars)", "int", climax_mod.PENDING_ORDER_EXPIRY_BARS, 1, 30, 1),
            ParamSpec("REWARD_RISK", "Reward:risk", "float", climax_mod.REWARD_RISK, 0.5, 6.0, 0.5),
            ParamSpec("MIN_SL_PCT", "Min stop distance (% of price)", "float", climax_mod.MIN_SL_PCT, 0.0, 1.0, 0.01),
        ],
        facets=[],
        notes="A volume-and-range exhaustion candle at the end of a directional run, confirmed by the "
              "very next bar closing the opposite way, entered on a stop order (not a market order). "
              "Native 15-min bars. Whether the volume condition is even applied is decided empirically "
              "each run (checked against the actual fetched data for every selected instrument, dropped "
              "for the whole run if any instrument's volume field looks degenerate) - watch for that "
              "diagnostic if results seem to ignore CLIMAX_VOLUME_MULT.",
        runner=_run_climax_volume_reversal,
        optimization_module=_find_optimization_module("research.climax_volume_reversal_dukascopy_backtest"),
    ))

    sr_mod = _load_module("research.support_resistance_zone_bounce_dukascopy_backtest")
    entries.append(StrategyDef(
        id="sr_zone_bounce",
        name="Support/Resistance Zone Bounce",
        module_name="research.support_resistance_zone_bounce_dukascopy_backtest",
        granularity="daily bars",
        instruments=sr_mod.INSTRUMENTS,
        params=[
            ParamSpec("ATR_LEN", "ATR length", "int", sr_mod.ATR_LEN, 2, 60, 1),
            ParamSpec("ZONE_PIVOT_LOOKBACK", "Pivot lookback (bars each side)", "int", sr_mod.ZONE_PIVOT_LOOKBACK, 1, 20, 1),
            ParamSpec("ZONE_WIDTH_ATR_MULT", "Zone half-width (x ATR)", "float", sr_mod.ZONE_WIDTH_ATR_MULT, 0.05, 2.0, 0.05),
            ParamSpec("ZONE_BREAK_BUFFER_ATR_MULT", "Zone break buffer (x ATR)", "float", sr_mod.ZONE_BREAK_BUFFER_ATR_MULT, 0.0, 1.0, 0.05),
            ParamSpec("FALLBACK_REWARD_RISK", "Fallback reward:risk", "float", sr_mod.FALLBACK_REWARD_RISK, 0.5, 10.0, 0.5),
            ParamSpec("MAX_HOLD_BARS", "Max hold (daily bars)", "int", sr_mod.MAX_HOLD_BARS, 5, 250, 5),
        ],
        facets=[],
        notes="Daily bars - a short date range will produce very few (or zero) confirmed swing pivots. Widen the range to see meaningful trade counts.",
        runner=_run_sr_zone_bounce,
        optimization_module=_find_optimization_module("research.support_resistance_zone_bounce_dukascopy_backtest"),
    ))

    # ---------------------------------------------------------------------------
    # parabolic_sar_dukascopy_backtest.py (Parabolic SAR, stop-and-reverse)
    # Exposes: INSTRUMENTS [(label, const)], FETCH_START, FETCH_END, INITIAL_AF, STEP_AF,
    # END_AF, MIN_SL_PCT, fetch_daily_ohlc(instrument_const) -> df, backtest_instrument(label,
    # df) -> trades (keys: side, outcome, r, date, stop_pct). Same shape as
    # support_resistance_zone_bounce - single-arg daily fetch, bare trade-list return - so it
    # reuses _run_sr_zone_bounce directly rather than needing its own runner.
    # ---------------------------------------------------------------------------
    _run_parabolic_sar = _run_sr_zone_bounce
    psar_mod = _load_module("research.parabolic_sar_dukascopy_backtest")
    entries.append(StrategyDef(
        id="parabolic_sar",
        name="Parabolic SAR (Stop-and-Reverse)",
        module_name="research.parabolic_sar_dukascopy_backtest",
        granularity="daily bars (swing/position - always in the market, alternating long/short)",
        instruments=psar_mod.INSTRUMENTS,
        params=[
            ParamSpec("INITIAL_AF", "Initial acceleration factor", "float", psar_mod.INITIAL_AF, 0.005, 0.1, 0.005),
            ParamSpec("STEP_AF", "Acceleration factor step", "float", psar_mod.STEP_AF, 0.005, 0.1, 0.005),
            ParamSpec("END_AF", "Acceleration factor cap", "float", psar_mod.END_AF, 0.05, 0.5, 0.05),
            ParamSpec("MIN_SL_PCT", "Min stop distance floor (% of price)", "float", psar_mod.MIN_SL_PCT, 0.0, 1.0, 0.01),
        ],
        facets=[],
        notes="Sourced from an open-source (Apache 2.0) Python port of Wilder's classic recursive SAR "
              "formula - the trading layer on top (stop-and-reverse, always in the market) is this "
              "project's own, since the source repo's own strategy is a simplified long-only demo with "
              "no stop-loss or R-multiple accounting at all. Expect frequent small whipsaw losses punctuated "
              "by occasional large trend-following wins - far fewer, much longer-held trades than this "
              "catalog's intraday strategies.",
        runner=_run_parabolic_sar,
        optimization_module=_find_optimization_module("research.parabolic_sar_dukascopy_backtest"),
        default_history_days=3 * 365,
    ))

    sb_mod = _load_module("research.ict_silver_bullet_forex_dukascopy_backtest")
    entries.append(StrategyDef(
        id="silver_bullet",
        name="ICT Silver Bullet",
        module_name="research.ict_silver_bullet_forex_dukascopy_backtest",
        granularity="5-min bars",
        instruments=sb_mod.INSTRUMENTS,
        params=[
            ParamSpec("reward_risk", "Reward:risk", "float", 2.0, 0.5, 10.0, 0.5,
                       help="Passed directly to run_backtest() - the module's own script grid-searches this; here it's one deliberate value."),
            ParamSpec("STOP_BUFFER_PCT", "Stop buffer (% of price)", "float", sb_mod.STOP_BUFFER_PCT, 0.0, 1.0, 0.01),
        ],
        facets=["window"],
        notes="Sweep -> market structure shift -> fair value gap -> retest, inside three fixed 1-hour NY kill zones. Narrow setup - short ranges may show 0 trades.",
        runner=_run_silver_bullet,
        optimization_module=_find_optimization_module("research.ict_silver_bullet_forex_dukascopy_backtest"),
    ))

    # ---------------------------------------------------------------------------
    # london_3am_range_reversal_dukascopy_backtest.py (London 3AM Range Reversal)
    # Exposes: INSTRUMENTS [(label, const)], FETCH_START, FETCH_END, STOP_BUFFER_PCT,
    # MIN_RANGE_PCT, MIN_REWARD_RISK, fetch_instrument_data(instrument_const) -> df,
    # backtest_instrument(label, df) -> trades (keys: side, outcome, r, date, stop_pct). Same
    # shape as ict_po3 - single-arg fetch, bare trade-list return - so it reuses _run_po3
    # directly rather than needing its own runner.
    # ---------------------------------------------------------------------------
    _run_london_3am = _run_po3
    l3am_mod = _load_module("research.london_3am_range_reversal_dukascopy_backtest")
    entries.append(StrategyDef(
        id="london_3am_range_reversal",
        name="London 3AM Range Reversal",
        module_name="research.london_3am_range_reversal_dukascopy_backtest",
        granularity="5-min bars",
        instruments=l3am_mod.INSTRUMENTS,
        params=[
            ParamSpec("STOP_BUFFER_PCT", "Stop buffer (% of price)", "float", l3am_mod.STOP_BUFFER_PCT, 0.0, 1.0, 0.01),
            ParamSpec("MIN_RANGE_PCT", "Min dealing-range floor (% of price)", "float", l3am_mod.MIN_RANGE_PCT, 0.0, 1.0, 0.01),
            ParamSpec("MIN_REWARD_RISK", "Min reward:risk to target (else skip)", "float", l3am_mod.MIN_REWARD_RISK, 0.5, 5.0, 0.5),
        ],
        facets=[],
        notes="Sourced from two independent YouTube walkthroughs, not Pine code - unlike PO3/Silver "
              "Bullet, several stated concepts have no numeric definition. One entry condition (SMT) is "
              "required in both sources but never defined in either (no named second instrument, no "
              "divergence rule) and is DELIBERATELY NOT IMPLEMENTED here - this only trades the "
              "mechanical half (00:00-02:00 NY dealing range -> 02:00-04:30 sweep -> post-sweep "
              "displacement -> trade to the FULL opposite range boundary, not the 50% midpoint - the "
              "second source's own \"50%... or the connected range low\" made the full-range target the "
              "one actually used, since it strictly dominates 50% every time it's reachable at all). See "
              "the script's own header for the full reasoning.",
        runner=_run_london_3am,
        optimization_module=_find_optimization_module("research.london_3am_range_reversal_dukascopy_backtest"),
    ))

    orb_mod = _load_module("research.orb_indices_dukascopy_backtest")
    entries.append(StrategyDef(
        id="orb_indices",
        name="ORB (Opening Range Breakout) - Indices",
        module_name="research.orb_indices_dukascopy_backtest",
        granularity="5-min bars",
        instruments=[(row[0],) for row in orb_mod.INDICES],
        params=[
            ParamSpec("RANGE_MINUTES", "Opening range length (min)", "int", orb_mod.RANGE_MINUTES, 5, 120, 5),
            ParamSpec("ENTRY_WINDOW_MINUTES", "Entry window after range (min)", "int", orb_mod.ENTRY_WINDOW_MINUTES, 15, 360, 15),
            ParamSpec("ENTRY_BUFFER_PCT", "Entry buffer (% of price)", "float", orb_mod.ENTRY_BUFFER_PCT, 0.0, 1.0, 0.01),
            ParamSpec("REWARD_RISK", "Reward:risk", "float", orb_mod.REWARD_RISK, 0.5, 5.0, 0.5),
            ParamSpec("MIN_RANGE_PCT", "Min range floor (% of price)", "float", orb_mod.MIN_RANGE_PCT, 0.0, 1.0, 0.01),
        ],
        facets=[],
        notes="Per-index local session opens (own timezone each). One trade per index per day, "
              "close-confirmed breakout with a filter stack (range-vs-ATR, impulsive candle, "
              "relative volume, volatility regime) on top of the bare breakout rule.",
        runner=_run_orb_indices,
        optimization_module=_find_optimization_module("research.orb_indices_dukascopy_backtest"),
    ))

    # ---------------------------------------------------------------------------
    # bdm_orb_reversal_indices_dukascopy_backtest.py ("Big Daddy Max ORB")
    # Same INDICES [(label, const, tz, session_open)] shape and the same
    # backtest_index(label, const, tz, session_start) signature as orb_indices above, so it reuses
    # _run_orb_indices unchanged. Trade keys: side, outcome, r, date, entry, sl_distance, stop_pct,
    # trade_type. Deliberately kept as a SEPARATE entry rather than folded into the ORB entry
    # above as another parameter set - the failed-breakout reversal is a different bet with the
    # opposite directional premise, and averaging the two into one row would hide whichever leg is
    # carrying (or sinking) the result.
    # ---------------------------------------------------------------------------
    bdm_mod = _load_module("research.bdm_orb_reversal_indices_dukascopy_backtest")
    entries.append(StrategyDef(
        id="bdm_orb_reversal_indices",
        name="Big Daddy Max ORB + Failed-Breakout Reversal - Indices",
        module_name="research.bdm_orb_reversal_indices_dukascopy_backtest",
        granularity="5-min bars - opening-range breakout with a reversal leg when it fails",
        instruments=[(row[0],) for row in bdm_mod.INDICES],
        params=[
            ParamSpec("ORB_MINUTES", "Opening range length (min)", "int", bdm_mod.ORB_MINUTES, 5, 120, 5),
            ParamSpec("SESSION_MINUTES", "Trade window from the open (min)", "int",
                      bdm_mod.SESSION_MINUTES, 30, 720, 15),
            ParamSpec("REWARD_RISK", "Reward:risk", "float", bdm_mod.REWARD_RISK, 0.5, 5.0, 0.5),
            ParamSpec("CONTINUATION_STOP_AT_MID", "Continuation stop: 1 = ORB midpoint, 0 = opposite side",
                      "int", bdm_mod.CONTINUATION_STOP_AT_MID, 0, 1, 1,
                      help="The source strategy's default is the midpoint, which halves the stop "
                           "distance versus the opposite side - so it doubles the R-multiple on the "
                           "same price move AND doubles how often it is hit. Not a free improvement."),
            ParamSpec("ENABLE_CONTINUATION", "Take the breakout trade (1/0)", "int",
                      bdm_mod.ENABLE_CONTINUATION, 0, 1, 1,
                      help="Turn off to test the reversal leg on its own. The breakout is still "
                           "detected either way - it is what defines the reversal setup."),
            ParamSpec("ENABLE_REVERSALS", "Take the failed-breakout reversal (1/0)", "int",
                      bdm_mod.ENABLE_REVERSALS, 0, 1, 1),
            ParamSpec("ALLOW_REVERSAL_AFTER_CLOSE", "Reverse even after the breakout trade closed (1/0)",
                      "int", bdm_mod.ALLOW_REVERSAL_AFTER_CLOSE, 0, 1, 1,
                      help="OFF (the source default) means the reversal only fires while the "
                           "breakout trade is STILL OPEN, which with a midpoint stop restricts it to "
                           "closes back inside the range but above the midpoint. ON makes any close "
                           "back inside the range a trade. This single flag changes the strategy's "
                           "character more than any other input here - compare both, don't assume."),
            ParamSpec("MIN_STOP_PCT", "Minimum stop distance (% of price)", "float",
                      bdm_mod.MIN_STOP_PCT, 0.0, 1.0, 0.01,
                      help="Trades with a thinner stop than this are skipped rather than scored. A "
                           "hairline stop produces an enormous R-multiple off a single bar and would "
                           "dominate the average - the source strategy scores in dollars and never "
                           "has to confront this."),
        ],
        facets=["trade_type"],
        notes="Ported from a public TradingView Pine strategy whose own report showed +6.63% on a "
              "FIXED ONE-CONTRACT size while the stop distance varies with each morning's range - "
              "so that result is a sum of unequal bets and cannot say whether the average trade "
              "was profitable per unit of risk. This port scores it in R, applies the same cost "
              "model and out-of-sample split as everything else, and tags each trade as "
              "'continuation' or 'reversal' so the two legs can be read apart with the Trade type "
              "filter. Read those separately first: they are opposite bets sharing one script.",
        runner=_run_orb_indices,
    ))

    vwap_mod = _load_module("research.evendyer_vwap_orb_dukascopy_backtest")
    entries.append(StrategyDef(
        id="evendyer_vwap_orb",
        name="EvenDyer VWAP ORB (Scam Or Slam)",
        module_name="research.evendyer_vwap_orb_dukascopy_backtest",
        granularity="5-min bars, US index CFDs only (SP500/NASDAQ100/DOWJONES)",
        instruments=vwap_mod.INSTRUMENTS,
        params=[
            ParamSpec("STOP_SWING_LEN", "Stop swing pivot lookback (bars)", "int", vwap_mod.STOP_SWING_LEN, 5, 60, 5),
            ParamSpec("TARGET_SWING_LEN", "Target swing pivot lookback (bars)", "int", vwap_mod.TARGET_SWING_LEN, 2, 40, 1),
            ParamSpec("BUFFER_AMOUNT", "Stop buffer beyond swing pivot (price units)", "float", vwap_mod.BUFFER_AMOUNT, 0.0, 10.0, 0.5),
        ],
        facets=[],
        notes="US index CFDs only - its 0930-1000 NY opening range and VWAP session windows only make "
              "sense on US equity index hours, there is no forex reading of these defaults. Enters WITH "
              "an opening-range break's direction, confirmed by a VWAP pullback-then-reclaim (not a "
              "candle pattern or FVG); stop/target come from confirmed swing pivots, not a fixed R:R. "
              "One trade/day/instrument max.",
        runner=_run_vwap_orb,
        optimization_module=_find_optimization_module("research.evendyer_vwap_orb_dukascopy_backtest"),
    ))

    # ---------------------------------------------------------------------------
    # tma_trend_scalper_forex_dukascopy_backtest.py ("TMA Trend Scalper")
    # Follows the _run_with_own_cache shape exactly (own pickle cache, FETCH_START/FETCH_END,
    # fetch_instrument_data(label, const) -> df, backtest_instrument(label, df) -> trades).
    # Trade keys: side, outcome, r, date, entry, sl_distance, stop_pct, pattern. Ported from both
    # the Pine v6 script AND a separate plain-English strategy writeup describing the same system -
    # they disagreed on entry timing and on a daily trade cap, and the writeup (the stated intent)
    # is what this port follows; see the module's own header for the full reasoning.
    # ---------------------------------------------------------------------------
    tma_mod = _load_module("research.tma_trend_scalper_forex_dukascopy_backtest")
    entries.append(StrategyDef(
        id="tma_trend_scalper",
        name="TMA Trend Scalper (triple SMMA + pattern + RSI)",
        module_name="research.tma_trend_scalper_forex_dukascopy_backtest",
        granularity="5-min bars, London session only",
        instruments=tma_mod.INSTRUMENTS,
        params=[
            ParamSpec("SESSION_START_HOUR", "Session start hour (UTC)", "int",
                      tma_mod.SESSION_START_HOUR, 0, 23, 1),
            ParamSpec("SESSION_END_HOUR", "Session end hour (UTC, exclusive)", "int",
                      tma_mod.SESSION_END_HOUR, 1, 24, 1),
            ParamSpec("WEEKDAY_FILTER_MODE", "Weekdays: 1 = Mon-Fri, 0 = source (Sun-Thu)", "int",
                      tma_mod.WEEKDAY_FILTER_MODE, 0, 1, 1,
                      help="Defaults to 1 - real Monday-to-Friday London trading, which both the "
                           "Pine's evident intent and the writeup ('Weekdays Only') agree on. The "
                           "Pine's `dayofweek >= 1 and <= 5` reads like Monday-Friday, but Pine "
                           "numbers Sunday as 1 - so as literally written it trades Sunday to "
                           "THURSDAY and never trades Friday. Set to 0 to reproduce that exactly, "
                           "e.g. to reconcile a result against the source's own TradingView report."),
            ParamSpec("MAX_TRADES_PER_DAY_PER_INSTRUMENT", "Max new trades per instrument per day (0 = unlimited)",
                      "int", tma_mod.MAX_TRADES_PER_DAY_PER_INSTRUMENT, 0, 5, 1,
                      help="Defaults to 1, matching the writeup's explicit, repeated rule ('One "
                           "Trade Per Day (Maximum)... NOT 50 trades, NOT 10 trades, just ONE'). "
                           "The Pine code itself doesn't enforce this - it only blocks a second "
                           "trade while the first is still OPEN, so a fresh signal later the same "
                           "day after the first trade already closed is free to fire again as "
                           "literally coded. Set to 0 to test that Pine-literal, uncapped version."),
            ParamSpec("ADX_MIN", "Minimum ADX", "float", tma_mod.ADX_MIN, 0.0, 60.0, 1.0),
            ParamSpec("ATR_MIN_MULT", "ATR vs its 50-bar average (multiple)", "float",
                      tma_mod.ATR_MIN_MULT, 0.0, 3.0, 0.1),
            ParamSpec("SMMA_FAST_LEN", "Fast SMMA length", "int", tma_mod.SMMA_FAST_LEN, 5, 100, 1),
            ParamSpec("SMMA_MED_LEN", "Medium SMMA length", "int", tma_mod.SMMA_MED_LEN, 10, 200, 5),
            ParamSpec("SMMA_SLOW_LEN", "Slow SMMA length", "int", tma_mod.SMMA_SLOW_LEN, 50, 400, 10),
            ParamSpec("STOP_CANDLE_MULT", "Stop = signal candle range x", "float",
                      tma_mod.STOP_CANDLE_MULT, 0.5, 6.0, 0.5),
            ParamSpec("TARGET_CANDLE_MULT", "Target = signal candle range x", "float",
                      tma_mod.TARGET_CANDLE_MULT, 0.5, 12.0, 0.5),
            ParamSpec("FILL_AT_NEXT_OPEN", "Fill at next bar's open (1) or signal close (0)", "int",
                      tma_mod.FILL_AT_NEXT_OPEN, 0, 1, 1,
                      help="Defaults to 1 - enter at the OPEN of the candle after the signal candle "
                           "CLOSES, exactly as the writeup describes ('wait for the candle to "
                           "close... enter at open of next candle'). The stop/target are still "
                           "computed from the signal bar's close and range, so the realised R:R "
                           "isn't a clean 2:1 as a result (0.30R-4.62R on test data) - that's a real "
                           "property of the strategy as designed, not a bug. Set to 0 to fill at the "
                           "signal close instead and isolate what the next-bar-open timing is worth."),
            ParamSpec("MIN_STOP_PCT", "Minimum stop distance (% of price)", "float",
                      tma_mod.MIN_STOP_PCT, 0.0, 0.5, 0.001),
        ],
        facets=["pattern"],
        notes="Ported from a TradingView Pine script AND a separate plain-English writeup of the "
              "same strategy - they disagreed in two places, and the writeup (the stated intent) "
              "wins both times. It documents entering at the OPEN of the candle AFTER the signal "
              "candle closes (default here); the Pine code produces exactly that by never setting "
              "process_orders_on_close, which the Pine alone looked like an accidental mismatch "
              "before the writeup confirmed it's deliberate. It also states 'One Trade Per Day "
              "(Maximum)' as a hard rule the Pine code itself never enforces - capped here by "
              "default, switchable to unlimited same-day re-entries to test the Pine literally. "
              "Trades EURUSD/GBPUSD/AUDUSD/USDJPY - AUD/USD is the writeup's own recommended pair - "
              "using its own per-pair minimum-SMMA-separation table (0.001 for the first three, "
              "0.10 for USDJPY) rather than one value calibrated for EURUSD alone. There is also NO "
              "time-based exit beyond the daily cap - a position can still be open well into a "
              "later session. Trades are tagged by pattern (3-line strike / engulfing / both) for "
              "the Pattern filter.",
        runner=_run_with_own_cache,
        default_history_days=2 * 365,
    ))

    # RANDOM ENTRY CONTROL - listed LAST on purpose so it reads as the yardstick at the bottom of
    # the catalog rather than as strategy #17. It is the reference line every other row should be
    # compared against: coin-flip entries, zero edge by construction, run through the identical
    # instruments/date range/cost model/holdout split as everything else. If a real strategy can't
    # beat this, it has not been shown to have an edge; if EVERYTHING lands near this line, the
    # leaderboard is measuring the cost model rather than the strategies. See the script's own
    # header for why that distinction is not otherwise recoverable from the results.
    random_control_mod = _load_module("research.random_baseline_control_dukascopy_backtest")
    entries.append(StrategyDef(
        id="random_baseline_control",
        name="⊘ Random Entry (control, not a strategy)",
        module_name="research.random_baseline_control_dukascopy_backtest",
        granularity="5-min bars - coin-flip entries, symmetric 1:1 ATR stop/target",
        instruments=random_control_mod.INSTRUMENTS,
        params=[
            ParamSpec("ATR_MULT", "Stop/target distance (ATR multiples)", "float",
                      random_control_mod.ATR_MULT, 0.5, 6.0, 0.5),
            ParamSpec("BARS_BETWEEN_ENTRIES", "Minimum bars between entries", "int",
                      random_control_mod.BARS_BETWEEN_ENTRIES, 1, 96, 1),
        ],
        facets=[],
        notes="NOT A STRATEGY - this is the control. It flips a coin on each eligible bar and takes a "
              "symmetric 1:1 bet, so it has NO edge by construction and its result is a direct "
              "readout of what trading costs alone do to an account over this date range. Use it as "
              "the bar: a strategy that doesn't clearly beat this line hasn't demonstrated an edge, "
              "and if every strategy clusters around it, the numbers are dominated by the cost "
              "assumption rather than by the rules. Seeded and reproducible so it can't be re-rolled.",
        runner=_run_with_own_cache,
        default_history_days=3 * 365,
    ))

    # --- trend_following_momentum_dukascopy_backtest.py: deliberately NOT included HERE, but it is
    # no longer unreachable - it now has its own "Momentum" page in app.py (momentum_page()), which
    # renders it in the unit it actually reports in. The reasoning below is why it can't live in
    # this registry, not a reason it goes unevaluated.
    # It doesn't produce a list of R-multiple trade dicts at all - its unit of output is a
    # monthly-rebalanced PORTFOLIO return series (NAV, Sharpe, max drawdown, % positive
    # months), built from build_instrument_frame()/portfolio_return_series()/portfolio_stats().
    # Every other strategy in this registry shares one "list of {side, outcome, r, date}
    # trades" shape that the whole results UI (equity curve, R-multiple stats, z-score,
    # trade table) is built around; this script's output is structurally a different kind of
    # result (a monthly NAV curve, not a trade log) and forcing it into the same UI would
    # mean either faking trade dicts that don't exist or building an entirely separate results
    # renderer just for this one strategy. Skipped per the task's own guidance to skip
    # scripts that don't cleanly fit the pattern rather than force it in.

    return entries


STRATEGIES = _build_registry()
STRATEGIES_BY_ID = {s.id: s for s in STRATEGIES}
