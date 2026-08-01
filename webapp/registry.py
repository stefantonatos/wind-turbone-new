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
        notes="Per-index local session opens (own timezone each). Trade dicts carry no date field upstream, so the equity curve here is sequence order, not calendar order.",
        runner=_run_orb_indices,
        optimization_module=_find_optimization_module("research.orb_indices_dukascopy_backtest"),
    ))

    # --- evendyer_vwap_orb_dukascopy_backtest.py: not present in research/ yet as of this
    # build (checked at registry-build time below) - add its own StrategyDef here, following
    # the same shape as the entries above, once it exists and follows the fetch+backtest
    # function-based pattern.
    import os
    research_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research")
    if os.path.isfile(os.path.join(research_dir, "evendyer_vwap_orb_dukascopy_backtest.py")):
        pass  # present but not wired in this pass - read it fully and add a StrategyDef before enabling

    # --- trend_following_momentum_dukascopy_backtest.py: deliberately NOT included.
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
