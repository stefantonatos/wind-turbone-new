# Proper walk-forward parameter optimization + ML scoring for ORB on
# indices - the one candidate out of everything tested in this project
# that's shown real (if not yet proven) promise: +0.012R/trade unfiltered,
# +0.043R/trade with the range/impulse/volume/regime filters, over
# 1500/548 trades on 6 real indices via Dukascopy.
#
# THIS FILE'S RIGOR WAS UPGRADED to match research/ict_po3_forex_dukascopy_optimization.py
# and research/day_trading_rauf_dukascopy_optimization.py, which had since surpassed it:
# both of those scripts now use the shared research/optimization_engine.py (grid/Bayesian/
# genetic search + a selectable objective), per-cell Monte Carlo, cluster/plateau analysis,
# and PROPER ROLLING walk-forward (multiple folds) - this file previously only did a 4x4
# grid search on a SINGLE fixed in-sample/out-of-sample split. Given this is the most
# promising strategy this project has tested, it deserves at least that same rigor,
# arguably more so. See optimization_engine.py's own header for the full reasoning behind
# offering more than exhaustive grid search, and ict_po3_forex_dukascopy_optimization.py for
# the reference implementation this file's new STEP 1-4 structure below is modeled on.
#
# THREE things this file now does:
#
#   1. PARAMETER OPTIMIZATION + ROBUSTNESS PASS (STEPs 1-4 below, narrow default grid):
#      STEP 1 - parameter search (SEARCH_METHOD/OBJECTIVE, research/optimization_engine.py)
#               over RANGE_MINUTES x REWARD_RISK, full 2016-2025 range, all 6 indices per
#               combo. SEARCH_METHOD="grid"/OBJECTIVE="total_r" by default - the EXACT
#               selection rule this script always used (see the regression test in
#               test_orb_indices_optimization_and_ml.py proving the new engine-dispatched
#               search reproduces the OLD hand-written grid loop's numbers exactly).
#      STEP 2 - Monte Carlo (bootstrap + shuffle, 2000 iterations each) on EVERY evaluated
#               cell, not just the best one.
#      STEP 3 - cluster/plateau-vs-spike analysis: sklearn KMeans plus an always-available
#               neighbor check (grid search only - see print_cluster_analysis's docstring).
#      STEP 4 - PROPER ROLLING walk-forward: 3-year in-sample / 1-year out-of-sample,
#               rolled forward 1 year at a time across 2016-2025 (6 folds, same convention
#               as PO3/Rauf) - replaces this file's old SINGLE fixed 2016-2023/2024 split.
#               Each fold re-runs the in-sample search on its own window, applies the
#               winning combo unchanged to the immediately-following out-of-sample window,
#               and chains all folds' OOS trades together - if OOS performance is much
#               worse than in-sample, that is direct evidence of overfitting, not a reason
#               to keep searching for a better number.
#
#   2. OPTIONAL WIDE SEARCH (WIDE_SEARCH=False by default - see the WIDE SEARCH section
#      below for the full reasoning, config, and REQUIRED caveat print). This file's own
#      ORIGINAL header comment (still true, quoted verbatim below) reasoned that only 2
#      structural parameters should be grid-searched, not filter thresholds too:
#
#        "Only two structural parameters are grid-searched (not the filter thresholds
#         too) - keeping the search space small relative to the data is itself a guard
#         against overfitting; a search over 6+ parameters would very likely just find
#         noise that happens to fit 2021-2023."
#
#      That reasoning is correct FOR EXHAUSTIVE GRID SEARCH specifically - a 4x4x4x4x5x5
#      grid over 2 structural + 4 filter-threshold parameters is 6,400 combos, almost all
#      of them noise. Bayesian/genetic search exist precisely to make a wider space
#      tractable WITHOUT exhaustive coverage (a small, fixed evaluation budget instead of
#      combos-multiply-across-every-added-dimension), so WIDE_SEARCH adds an OPTIONAL,
#      OFF-BY-DEFAULT path that searches RANGE_MINUTES/REWARD_RISK plus the ATR-range,
#      impulse, and relative-volume filter thresholds using Bayesian or genetic search
#      SPECIFICALLY (never exhaustive grid, for exactly the overfitting reason this file's
#      own header already gave) - the narrow 2-parameter grid stays the DEFAULT, unchanged,
#      still comparable to every already-shipped number this project has reported for ORB.
#
#   3. ML SCORING (unchanged in approach from before this upgrade) - same idea already
#      proven for forex in quantconnect/orb_datagen.py / train_orb_model.ipynb / orb_ml.py:
#      strip the hand-picked filters, record every raw close-confirmed breakout's features
#      and real outcome across the in-sample period (SPLIT_DATE, unchanged - a single
#      train/test split is the right tool for training one classifier, unlike the parameter
#      SEARCH above which specifically needed rolling folds to avoid overfitting a single
#      lucky window), train a classifier, check its R-multiple performance on the SAME
#      held-out out-of-sample period the STEP-1 full-range search's best RANGE_MINUTES
#      informs - apples-to-apples against the hand-tuned version, not a different window.
#
# NEWS/ECONOMIC-CALENDAR FILTER - INVESTIGATED, NOT IMPLEMENTED. See the "NEWS FILTER"
# section further down for the full write-up of what was tried and why it was not
# feasible to add honestly in this environment (no fabricated event dates were added).

# !pip install --upgrade dukascopy-python scikit-learn optuna -q   # uncomment in Colab
# (optuna is only needed for SEARCH_METHOD/WIDE_SEARCH_METHOD="bayesian" - see
# optimization_engine.py's header for why it's an optional, gracefully-skipped dependency)

import datetime
import os
import sys

import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

# research/optimization_engine.py is a sibling module in this same directory, not a package -
# this sys.path insert makes `import optimization_engine` resolve correctly regardless of HOW
# this file is loaded (run directly, run via `python research/....py`, or loaded by
# importlib.util.spec_from_file_location the way test_orb_indices_optimization_and_ml.py loads
# this module) rather than depending on the caller's own sys.path/cwd - same pattern already
# used by ict_po3_forex_dukascopy_optimization.py / day_trading_rauf_dukascopy_optimization.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optimization_engine as opt_engine

INDICES = [
    ("SP500", dki.INSTRUMENT_IDX_AMERICA_E_SANDP_500, "America/New_York", pd.Timestamp("09:30").time()),
    ("NASDAQ100", dki.INSTRUMENT_IDX_AMERICA_E_NQ_100, "America/New_York", pd.Timestamp("09:30").time()),
    ("DOWJONES", dki.INSTRUMENT_IDX_AMERICA_E_D_J_IND, "America/New_York", pd.Timestamp("09:30").time()),
    ("DAX", dki.INSTRUMENT_IDX_EUROPE_E_DAAX, "Europe/Berlin", pd.Timestamp("09:00").time()),
    ("FTSE100", dki.INSTRUMENT_IDX_EUROPE_E_FUTSEE_100, "Europe/London", pd.Timestamp("08:00").time()),
    ("NIKKEI225", dki.INSTRUMENT_IDX_ASIA_E_N225JAP, "Asia/Tokyo", pd.Timestamp("09:00").time()),
]

FETCH_START = datetime.datetime(2016, 1, 1)   # widened from 2021 in an earlier pass - also happens to
                                               # match PO3/Rauf's own FETCH_START, so the walk-forward
                                               # folds below use the identical 2016-2025/6-fold convention
FETCH_END = datetime.datetime(2025, 1, 1)
SPLIT_DATE = datetime.date(2024, 1, 1)   # used ONLY by the ML SCORING pass below (PART 3) - everything
                                          # before this = ML train, on/after = ML test. The parameter
                                          # SEARCH in PART 1 no longer uses a single fixed split at all;
                                          # it uses the rolling walk-forward folds instead (STEP 4).
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_MIN_5
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

# --- fixed strategy config (the parts NOT being grid-searched, even under WIDE_SEARCH) ---
ENTRY_WINDOW_MINUTES = 180
SESSION_HOLD_HOURS = 8
ENTRY_BUFFER_PCT = 0.02
MIN_RANGE_PCT = 0.05
REVERSE_SIGNALS = False

ATR_LEN = 14
RANGE_ATR_FILTER = True
MIN_RANGE_ATR_MULT = 0.5   # default/narrow-search value - overridable per-combo under WIDE_SEARCH
MAX_RANGE_ATR_MULT = 3.0   # default/narrow-search value - overridable per-combo under WIDE_SEARCH
IMPULSE_FILTER = True
RANGE_AVG_LEN = 20
IMPULSE_RANGE_MULT = 1.3   # default/narrow-search value - overridable per-combo under WIDE_SEARCH
VOLUME_FILTER = True
VOLUME_AVG_LEN = 20
RVOL_MULT = 1.3   # default/narrow-search value - overridable per-combo under WIDE_SEARCH
VOLATILITY_REGIME_FILTER = True
ATR_BASELINE_LEN = 100
LOW_VOL_MULT = 0.7
HIGH_VOL_MULT = 1.5
ALLOWED_REGIMES = {"normal", "high"}

# --- narrow grid (DEFAULT, unchanged) - kept deliberately small, see header comment ---
RANGE_MINUTES_GRID = [10, 15, 20, 30]
REWARD_RISK_GRID = [0.5, 1.0, 1.5, 2.0]

# --- search strategy + objective config (research/optimization_engine.py) ---
# SEARCH_METHOD: "grid" (exhaustive, DEFAULT), "bayesian" (Optuna/TPE), or "genetic" (hand-rolled
# GA). Default is "grid" DELIBERATELY - the narrow RANGE_MINUTES_GRID x REWARD_RISK_GRID space is
# only 16 combos, where exhaustive search is the better choice (deterministic, nothing left
# unexplored). This governs STEP 1's full-range search and STEP 4's per-fold in-sample search;
# WIDE_SEARCH (below) has its own separate WIDE_SEARCH_METHOD, which is NEVER allowed to be "grid".
SEARCH_METHOD = "grid"
# OBJECTIVE: which of optimization_engine.OBJECTIVES the search maximizes by. Default "total_r"
# DELIBERATELY - this is the exact selection rule this script already used before this upgrade
# (every already-shipped, already-verified ORB number this project has reported picked its "best"
# combo by raw total R), so SEARCH_METHOD="grid" + OBJECTIVE="total_r" reproduces that behavior
# exactly, not a new default - see test_orb_indices_optimization_and_ml.py's regression test.
OBJECTIVE = "total_r"

MC_ITERATIONS = 2000   # per cell, per method (bootstrap and shuffle) - same budget as PO3/Rauf

# --- STEP 4: rolling 3yr-in-sample / 1yr-out-of-sample walk-forward, stepped forward 1 year ---
WALK_FORWARD_IS_YEARS = 3
WALK_FORWARD_OOS_YEARS = 1
WALK_FORWARD_STEP_YEARS = 1
WFE_PASS_THRESHOLD = 0.5   # standard rule-of-thumb, not a proof - see print_walk_forward's output

HEATMAP_PNG_PATH = "orb_indices_param_heatmap.png"

# ============================================================================
# WIDE SEARCH (OPTIONAL, OFF BY DEFAULT) - read the module header before flipping this on
# ============================================================================
#
# WIDE_SEARCH=True additionally searches 4 of the currently-fixed filter thresholds alongside
# RANGE_MINUTES/REWARD_RISK - MIN_RANGE_ATR_MULT, MAX_RANGE_ATR_MULT (the ATR-range filter's two
# bounds - the single most structurally important filter per this project's own header claims
# about this strategy), IMPULSE_RANGE_MULT, and RVOL_MULT (relative volume) - using
# WIDE_SEARCH_METHOD (Bayesian or genetic, NEVER grid - enforced at runtime by
# _resolve_wide_search_method() below) instead of exhaustive search. The full cross-product of
# WIDE_PARAM_GRID below is 4x4x4x4x5x5 = 6,400 combos - exactly the "6+ parameters would very
# likely just find noise" scenario this file's own original header warned about for EXHAUSTIVE
# grid search - which is why this path is never allowed to dispatch through grid_search.
WIDE_SEARCH = False
WIDE_SEARCH_METHOD = "bayesian"   # "bayesian" or "genetic" only - see _resolve_wide_search_method()
WIDE_SEARCH_N_TRIALS = 60          # Optuna trial budget, if WIDE_SEARCH_METHOD == "bayesian"
WIDE_SEARCH_POPULATION = 16        # genetic population size, if WIDE_SEARCH_METHOD == "genetic"
WIDE_SEARCH_GENERATIONS = 12       # genetic generations, if WIDE_SEARCH_METHOD == "genetic"

# Ranges chosen so MIN_RANGE_ATR_MULT_GRID's max (1.0) stays below MAX_RANGE_ATR_MULT_GRID's min
# (2.0) - guarantees every sampled (min, max) pair is structurally valid (min < max) without
# needing extra validation code in the eval path.
MIN_RANGE_ATR_MULT_GRID = [0.3, 0.5, 0.7, 1.0]
MAX_RANGE_ATR_MULT_GRID = [2.0, 2.5, 3.0, 4.0]
IMPULSE_RANGE_MULT_GRID = [1.0, 1.15, 1.3, 1.5, 1.75]
RVOL_MULT_GRID = [1.0, 1.15, 1.3, 1.5, 1.75]

WIDE_PARAM_GRID = {
    "range_minutes": RANGE_MINUTES_GRID,
    "reward_risk": REWARD_RISK_GRID,
    "min_range_atr_mult": MIN_RANGE_ATR_MULT_GRID,
    "max_range_atr_mult": MAX_RANGE_ATR_MULT_GRID,
    "impulse_range_mult": IMPULSE_RANGE_MULT_GRID,
    "rvol_mult": RVOL_MULT_GRID,
}

# HONEST CAVEAT (printed verbatim by main() whenever WIDE_SEARCH is used - see
# print_wide_search_caveat() below): as of this upgrade, research/optimization_engine.py does not
# yet have a lockbox/embargoed-holdout mechanism wired in anywhere in this project (checked via
# `grep -n "lockbox\|split_lockbox" research/optimization_engine.py` immediately before writing
# this - no match). If that lands later, the wide-search path here - being the higher-dimensional,
# higher-overfitting-risk search - is exactly the case it should be wired into first. Until then,
# any promising WIDE_SEARCH result should be treated with MORE skepticism than the narrow default
# grid's result, not less, even though it also gets Monte Carlo, cluster analysis, and its own
# rolling walk-forward below (STEP W4) - more searched dimensions over the same finite historical
# data is inherently more overfitting surface, walk-forward or not, without a genuine held-out
# check this project doesn't have wired in yet.

# ============================================================================
# NEWS FILTER - INVESTIGATED, NOT IMPLEMENTED
# ============================================================================
#
# The source material this task is based on describes an ORB variant that adds "the earnings and
# FX calendar" as a fundamental filter on top of the price-action filters already in this script -
# skipping new signals within a window of high-impact scheduled news (NFP, FOMC, CPI, ECB rate
# decisions). This project does not have this yet, and this upgrade pass investigated whether a
# genuinely reliable historical calendar of these events (2016-2025) could be built here.
#
# WHAT WAS TRIED: WebSearch found the correct primary sources (federalreserve.gov's per-year FOMC
# calendars, e.g. federalreserve.gov/monetarypolicy/fomchistorical2016.htm, and bls.gov's per-year
# release schedules, e.g. bls.gov/schedule/2016/home.htm - both are the actual, authoritative
# sources for these dates). Fetching either one to read the actual table of dates - required to
# transcribe REAL dates rather than assume the "usual" schedule - failed with HTTP 403 in this
# sandbox. This was confirmed to be a sandbox-wide network restriction, not a source-specific
# block: a fetch of https://example.com (a plain, unauthenticated, non-adversarial URL) through
# the same tool also returned 403. With page-fetching unavailable, only WebSearch's short result
# snippets remained, and those are not a reliable way to reconstruct a ~190-row historical
# calendar (72 FOMC decisions + ~120 NFP releases across 2016-2025): snippets are fragments, not
# verified against the actual source table, and NFP in particular has real holiday-shift exceptions
# to the "first Friday" rule that cannot be confirmed without reading BLS's own schedule page.
#
# DECISION: per this project's standing rule that a fabricated-but-plausible-looking date is worse
# than no filter at all (it would silently corrupt any backtest that used it), NO event-date CSV
# and NO NEWS_FILTER code were added in this pass - guessing "first Friday of the month" for NFP or
# "roughly every 6 weeks" for FOMC would be exactly the kind of unverified data this project's
# rigor conventions elsewhere (see e.g. optimization_engine.py's Bonferroni section) exist to guard
# against, just applied to input data instead of a statistical test.
#
# WHAT A REAL IMPLEMENTATION WOULD NEED: either (a) network access to actually fetch and read
# federalreserve.gov's and bls.gov's own per-year schedule pages (or any other authoritative
# calendar source) so the dates can be transcribed and spot-checked against the real table, not
# assumed from a pattern, or (b) a proper economic-calendar API/dataset (e.g. FRED's release
# calendar API, a commercial provider, or a maintained CSV export from a site like ForexFactory)
# that ships pre-verified historical dates. Both are out of scope for what is obtainable in this
# sandbox today. A future pass with working network access could build
# research/data/major_news_events_2016_2025.csv (columns: date, time_utc, event, currency,
# impact) and wire an optional NEWS_FILTER=False / NEWS_AVOID_MINUTES=30 config into run_backtest
# the same way min_range_atr_mult/max_range_atr_mult/impulse_range_mult/rvol_mult were made
# overridable below - the backtest loop's per-bar structure already has a natural place to check
# "is this bar's timestamp within NEWS_AVOID_MINUTES of any listed event" right alongside the
# existing filter checks.

FEATURE_COLUMNS = [
    "range_vs_atr", "impulse_ratio", "atr_regime_ratio", "volume_ratio",
    "session_minute", "day_of_week", "direction",
]


def to_local_time(index, tz_name):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(tz_name)


def persisted_avg(values, length, start_index=0):
    n = len(values)
    out = [None] * n
    usable = values[start_index:]
    if len(usable) < length:
        return out
    seed = sum(usable[:length]) / length
    out[start_index + length - 1] = seed
    prev = seed
    for i in range(start_index + length, n):
        prev = (prev * (length - 1) + values[i]) / length
        out[i] = prev
    return out


def compute_atr_series(highs, lows, closes, length):
    n = len(closes)
    atr = [None] * n
    tr_seed = []
    atr_val = None
    prev_close = None
    for i in range(n):
        if prev_close is not None:
            tr = max(highs[i] - lows[i], abs(highs[i] - prev_close), abs(lows[i] - prev_close))
            if atr_val is None:
                tr_seed.append(tr)
                if len(tr_seed) >= length:
                    atr_val = sum(tr_seed) / length
            else:
                atr_val = (atr_val * (length - 1) + tr) / length
        atr[i] = atr_val
        prev_close = closes[i]
    return atr


def volatility_regime(current_atr, baseline):
    if current_atr is None or baseline is None or baseline <= 0:
        return "normal"
    if current_atr > HIGH_VOL_MULT * baseline:
        return "high"
    if current_atr < LOW_VOL_MULT * baseline:
        return "low"
    return "normal"


def fetch_index_data(instrument_const, tz_name):
    """Downloads ONCE for the full FETCH_START-FETCH_END range - grid search, walk-forward, and
    ML all reuse this same in-memory data rather than re-fetching per combo/fold, which would be
    enormously wasteful."""
    df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, FETCH_START, FETCH_END)
    if df.empty:
        return None
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df.index = to_local_time(df.index, tz_name)
    return df


def precompute_indicators(df):
    """Everything that doesn't depend on RANGE_MINUTES/REWARD_RISK/filter thresholds or the
    walk-forward window - computed once per index, reused across every grid combination and
    every fold."""
    highs, lows, closes = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
    volumes = df["volume"].tolist() if "volume" in df.columns else [0] * len(df)

    atr = compute_atr_series(highs, lows, closes, ATR_LEN)
    bar_ranges = [h - l for h, l in zip(highs, lows)]
    range_avg = persisted_avg(bar_ranges, RANGE_AVG_LEN)
    volume_avg = persisted_avg(volumes, VOLUME_AVG_LEN)
    atr_first_valid = next((idx for idx, a in enumerate(atr) if a is not None), len(atr))
    atr_baseline = persisted_avg([a if a is not None else 0.0 for a in atr], ATR_BASELINE_LEN,
                                  start_index=atr_first_valid)
    return {
        "highs": highs, "lows": lows, "closes": closes, "volumes": volumes,
        "atr": atr, "range_avg": range_avg, "volume_avg": volume_avg, "atr_baseline": atr_baseline,
    }


def run_backtest(df, ind, tz_name, session_start, range_minutes, reward_risk,
                  apply_filters=True, split_before=None, split_after=None, collect_candidates=False,
                  min_range_atr_mult=None, max_range_atr_mult=None, impulse_range_mult=None, rvol_mult=None):
    """Core ORB loop, parameterized by range_minutes/reward_risk for the parameter search.
    apply_filters=False + collect_candidates=True switches into ML-datagen mode: every
    close-confirmed breakout is taken and labeled, filters only computed as features, matching
    this project's existing quantconnect/orb_datagen.py approach.

    split_before/split_after restrict which calendar dates are processed (a two-sided
    [split_after, split_before) window when both are given - the exact same convention
    ict_po3_forex_dukascopy_optimization.py's window_start/window_end generalizes from), without
    needing to re-slice or re-download the underlying data.

    min_range_atr_mult/max_range_atr_mult/impulse_range_mult/rvol_mult: NEW in this upgrade -
    override the module-level MIN_RANGE_ATR_MULT/MAX_RANGE_ATR_MULT/IMPULSE_RANGE_MULT/RVOL_MULT
    filter thresholds for THIS call only, defaulting to those module-level constants when left
    None (the original, always-verified behavior). This is what lets WIDE_SEARCH search these
    thresholds per-combo without touching the module-level defaults the narrow search still uses -
    with every override left at None (the default), this function's behavior is byte-for-byte
    identical to before this upgrade; see test_orb_indices_optimization_and_ml.py's regression
    test for a direct check of that claim."""
    min_range_atr_mult = MIN_RANGE_ATR_MULT if min_range_atr_mult is None else min_range_atr_mult
    max_range_atr_mult = MAX_RANGE_ATR_MULT if max_range_atr_mult is None else max_range_atr_mult
    impulse_range_mult = IMPULSE_RANGE_MULT if impulse_range_mult is None else impulse_range_mult
    rvol_mult = RVOL_MULT if rvol_mult is None else rvol_mult

    highs, lows, closes = ind["highs"], ind["lows"], ind["closes"]
    volumes, atr = ind["volumes"], ind["atr"]
    range_avg, volume_avg, atr_baseline = ind["range_avg"], ind["volume_avg"], ind["atr_baseline"]
    times = df.index
    n = len(closes)

    range_end = (datetime.datetime.combine(datetime.date.min, session_start)
                 + datetime.timedelta(minutes=range_minutes)).time()
    entry_end = (datetime.datetime.combine(datetime.date.min, session_start)
                 + datetime.timedelta(minutes=range_minutes + ENTRY_WINDOW_MINUTES)).time()
    session_end = (datetime.datetime.combine(datetime.date.min, session_start)
                   + datetime.timedelta(hours=SESSION_HOLD_HOURS)).time()

    trades = []
    current_day = None
    range_high = range_low = None
    traded_today = False
    session_start_dt = None
    i = 0
    while i < n:
        t = times[i]
        today = t.date()
        tod = t.time()

        if split_before is not None and today >= split_before:
            i += 1
            continue
        if split_after is not None and today < split_after:
            i += 1
            continue

        if today != current_day:
            current_day = today
            range_high = range_low = None
            traded_today = False

        if traded_today:
            i += 1
            continue

        if session_start <= tod < range_end:
            range_high = highs[i] if range_high is None else max(range_high, highs[i])
            range_low = lows[i] if range_low is None else min(range_low, lows[i])
            if session_start_dt is None or today != session_start_dt:
                session_start_dt = today
            i += 1
            continue

        if range_high is None:
            i += 1
            continue

        if tod >= entry_end:
            traded_today = True
            i += 1
            continue

        buffer_price = (ENTRY_BUFFER_PCT / 100.0) * closes[i]
        price = closes[i]
        buy_setup = price > range_high + buffer_price
        sell_setup = price < range_low - buffer_price

        if REVERSE_SIGNALS:
            buy_setup, sell_setup = sell_setup, buy_setup

        if not buy_setup and not sell_setup:
            i += 1
            continue

        range_size = range_high - range_low
        current_atr = atr[i]

        if apply_filters:
            if RANGE_ATR_FILTER and current_atr:
                if range_size < min_range_atr_mult * current_atr or range_size > max_range_atr_mult * current_atr:
                    traded_today = True
                    i += 1
                    continue
            if VOLATILITY_REGIME_FILTER:
                regime = volatility_regime(current_atr, atr_baseline[i])
                if regime not in ALLOWED_REGIMES:
                    traded_today = True
                    i += 1
                    continue
            if IMPULSE_FILTER and range_avg[i]:
                if (highs[i] - lows[i]) < impulse_range_mult * range_avg[i]:
                    traded_today = True
                    i += 1
                    continue
            if VOLUME_FILTER and volume_avg[i] and volume_avg[i] > 0:
                if volumes[i] < rvol_mult * volume_avg[i]:
                    traded_today = True
                    i += 1
                    continue

        sl_distance = max(range_size, (MIN_RANGE_PCT / 100.0) * price)
        tp_distance = sl_distance * reward_risk
        side = "LONG" if buy_setup else "SHORT"
        entry = price
        stop = entry - sl_distance if side == "LONG" else entry + sl_distance
        target = entry + tp_distance if side == "LONG" else entry - tp_distance

        traded_today = True
        outcome, exit_r = None, None
        j = i + 1
        while j < n and times[j].date() == today and times[j].time() < session_end:
            hi, lo = highs[j], lows[j]
            hit_stop = lo <= stop if side == "LONG" else hi >= stop
            hit_target = hi >= target if side == "LONG" else lo <= target
            if hit_stop:
                outcome, exit_r = "SL", -1.0
                break
            if hit_target:
                outcome, exit_r = "TP", reward_risk
                break
            j += 1
        else:
            j = min(j, n - 1)

        if outcome is None:
            last_close = closes[j]
            pnl = (last_close - entry) if side == "LONG" else (entry - last_close)
            outcome, exit_r = "FLAT", pnl / sl_distance

        trade = {"side": side, "outcome": outcome, "r": exit_r, "date": today}

        if collect_candidates:
            minutes_since_open = (datetime.datetime.combine(datetime.date.min, tod)
                                   - datetime.datetime.combine(datetime.date.min, session_start)).total_seconds() / 60
            trade["features"] = {
                "range_vs_atr": (range_size / current_atr) if current_atr else np.nan,
                "impulse_ratio": ((highs[i] - lows[i]) / range_avg[i]) if range_avg[i] else np.nan,
                "atr_regime_ratio": (current_atr / atr_baseline[i]) if (current_atr and atr_baseline[i]) else np.nan,
                "volume_ratio": (volumes[i] / volume_avg[i]) if (volume_avg[i] and volume_avg[i] > 0) else np.nan,
                "session_minute": minutes_since_open,
                "day_of_week": t.weekday(),
                "direction": 1 if side == "LONG" else -1,
            }
            trade["label"] = 1 if exit_r > 0 else 0

        trades.append(trade)
        i = j + 1

    return trades


# ============================= legacy reference implementation =============================

def run_grid_search(data, split_before=None, split_after=None, grid=None, desc="grid search"):
    """The ORIGINAL hand-written grid loop this file's Part 1 used before this upgrade, factored
    into a standalone function (behavior byte-for-byte unchanged) so it can be used as the
    regression-test reference for run_param_search's engine-dispatched default path below - see
    test_orb_indices_optimization_and_ml.py's TestSearchEngineRetrofit. grid defaults to the full
    RANGE_MINUTES_GRID x REWARD_RISK_GRID cross product; split_before/split_after restrict the
    window exactly as run_backtest's own params do."""
    if grid is None:
        grid = [(rm, rr) for rm in RANGE_MINUTES_GRID for rr in REWARD_RISK_GRID]

    results = []
    for range_minutes, reward_risk in grid:
        all_r = []
        per_index = {}
        for label, (df, ind, tz_name, session_start) in data.items():
            trades = run_backtest(df, ind, tz_name, session_start, range_minutes, reward_risk,
                                   apply_filters=True, split_before=split_before, split_after=split_after)
            rs = [t["r"] for t in trades]
            all_r.extend(rs)
            per_index[label] = rs
        n = len(all_r)
        total_r = sum(all_r)
        results.append({
            "params": {"range_minutes": range_minutes, "reward_risk": reward_risk},
            "range_minutes": range_minutes, "reward_risk": reward_risk,
            "total_r": total_r, "n_trades": n, "avg_r": total_r / n if n else 0.0,
            "trades_r": all_r, "per_index": per_index,
        })
    return results


# ==================== SEARCH_METHOD/OBJECTIVE dispatch (research/optimization_engine.py) ====================

def _make_orb_eval_fn(data, window_start=None, window_end=None):
    """Builds an optimization_engine-compatible eval_fn(params) -> list_of_trade_dicts. params
    must have "range_minutes"/"reward_risk"; the 4 filter-threshold keys
    (min_range_atr_mult/max_range_atr_mult/impulse_range_mult/rvol_mult) are optional - when
    absent (the narrow-search path), run_backtest falls back to the module-level filter
    constants, exactly as before this upgrade."""
    def eval_fn(params):
        trades = []
        for label, (df, ind, tz_name, session_start) in data.items():
            trades.extend(run_backtest(
                df, ind, tz_name, session_start, params["range_minutes"], params["reward_risk"],
                apply_filters=True, split_before=window_end, split_after=window_start,
                min_range_atr_mult=params.get("min_range_atr_mult"),
                max_range_atr_mult=params.get("max_range_atr_mult"),
                impulse_range_mult=params.get("impulse_range_mult"),
                rvol_mult=params.get("rvol_mult"),
            ))
        return trades
    return eval_fn


def _search_result_to_grid_cells(search_result):
    """Converts an optimization_engine search result's "all" list into this file's cell-dict
    shape (params, range_minutes, reward_risk, total_r, n_trades, avg_r, trades_r, score) so
    every downstream consumer (Monte Carlo, cluster analysis, print/plot helpers) works
    unchanged regardless of which SEARCH_METHOD produced the results, and regardless of whether
    `params` has 2 keys (narrow) or 6 (wide)."""
    cells = []
    for entry in search_result["all"]:
        params, trades = entry["params"], entry["trades"]
        all_r = [t["r"] for t in trades]
        n = len(all_r)
        total_r_sum = sum(all_r)
        cells.append({
            "params": params,
            "range_minutes": params["range_minutes"],
            "reward_risk": params["reward_risk"],
            "total_r": total_r_sum,
            "n_trades": n,
            "avg_r": total_r_sum / n if n else 0.0,
            "trades_r": all_r,
            "score": entry["score"],
        })
    return cells


def _find_cell(grid_results, params):
    for c in grid_results:
        if c["params"] == params:
            return c
    return None


def run_param_search(data, window_start=None, window_end=None, param_grid=None, method=None, objective=None,
                      desc="param search", **search_kwargs):
    """Dispatches the parameter search (STEP 1's full-range pass, STEP 4's per-fold in-sample
    pass, and the optional WIDE_SEARCH pass) to whichever method is configured (method=None
    resolves to the module-level SEARCH_METHOD; objective=None resolves to OBJECTIVE), via
    optimization_engine.run_search(), then converts the result into this file's cell-dict shape.

    param_grid defaults to the narrow {"range_minutes": RANGE_MINUTES_GRID, "reward_risk":
    REWARD_RISK_GRID} - pass WIDE_PARAM_GRID (or any other dict of param_name -> candidate list)
    to search a different space.

    DEFAULTS REPRODUCE THE ORIGINAL BEHAVIOR EXACTLY: method=None/objective=None/param_grid=None
    resolve to SEARCH_METHOD="grid"/OBJECTIVE="total_r"/the narrow grid, i.e. the exact same
    exhaustive grid search + raw-total-R selection run_grid_search() above already performs - see
    test_orb_indices_optimization_and_ml.py's TestSearchEngineRetrofit regression test for a
    direct, assertion-based comparison against that old hand-written loop."""
    method = method if method is not None else SEARCH_METHOD
    objective = objective if objective is not None else OBJECTIVE
    if param_grid is None:
        param_grid = {"range_minutes": list(RANGE_MINUTES_GRID), "reward_risk": list(REWARD_RISK_GRID)}

    objective_fn = opt_engine.get_objective(objective)
    if window_start is not None and window_end is not None:
        years = (window_end - window_start).days / 365.0
    elif window_start is None and window_end is None:
        years = (FETCH_END - FETCH_START).days / 365.0
    else:
        years = None   # a one-sided window has no well-defined span - sharpe will degrade to its sentinel

    eval_fn = _make_orb_eval_fn(data, window_start, window_end)
    search_result = opt_engine.run_search(method, param_grid, eval_fn, objective_fn, years=years,
                                           **search_kwargs)
    grid_results = _search_result_to_grid_cells(search_result)
    return grid_results, search_result


def _format_params(params):
    """Generic "key=value key=value ..." formatter for a cell's params dict - works for both the
    narrow 2-key grid and WIDE_PARAM_GRID's 6 keys without hardcoding either shape."""
    parts = []
    for k, v in params.items():
        parts.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
    return " ".join(parts)


def build_grid_matrix(grid_results):
    rm_list = sorted(set(r["range_minutes"] for r in grid_results))
    rr_list = sorted(set(r["reward_risk"] for r in grid_results))
    lookup = {(r["range_minutes"], r["reward_risk"]): r for r in grid_results}
    avg_r_matrix = np.zeros((len(rm_list), len(rr_list)))
    total_r_matrix = np.zeros((len(rm_list), len(rr_list)))
    for i, rm in enumerate(rm_list):
        for j, rr in enumerate(rr_list):
            cell = lookup[(rm, rr)]
            avg_r_matrix[i, j] = cell["avg_r"]
            total_r_matrix[i, j] = cell["total_r"]
    return rm_list, rr_list, avg_r_matrix, total_r_matrix


def print_grid_table(grid_results):
    """Always-available text table of avg R/trade per cell - only meaningful for the narrow (2
    parameter) grid; the wide search's 6-dimensional results are reported with
    print_top_wide_results instead (see below)."""
    rm_list, rr_list, avg_r_matrix, total_r_matrix = build_grid_matrix(grid_results)
    best_i, best_j = np.unravel_index(np.argmax(avg_r_matrix), avg_r_matrix.shape)

    print("\nRANGE_MINUTES x REWARD_RISK grid - avg R/trade (best cell marked *):")
    header = "RM\\RR".ljust(10) + "".join(f"{rr:>12.1f}" for rr in rr_list)
    print(header)
    for i, rm in enumerate(rm_list):
        cells = []
        for j in range(len(rr_list)):
            marker = "*" if (i, j) == (best_i, best_j) else " "
            cells.append(f"{avg_r_matrix[i, j]:>+10.4f}{marker}")
        print(f"{rm:<10.0f}" + "".join(cells))

    print("\nSame grid, total R:")
    print(header)
    for i, rm in enumerate(rm_list):
        cells = [f"{total_r_matrix[i, j]:>+11.2f} " for j in range(len(rr_list))]
        print(f"{rm:<10.0f}" + "".join(cells))


def plot_heatmap(grid_results, out_path=HEATMAP_PNG_PATH):
    """Renders the avg-R/trade heatmap to a PNG and calls plt.show() for inline Colab display.
    Wrapped in try/except ImportError so a missing matplotlib never kills the script - the text
    table above is the fallback (and is always printed regardless of this succeeding)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed - skipping heatmap plot (the text table above is the fallback, "
              "not the only path; run '!pip install matplotlib -q' and re-run for the plot too).")
        return

    rm_list, rr_list, avg_r_matrix, total_r_matrix = build_grid_matrix(grid_results)
    best_i, best_j = np.unravel_index(np.argmax(avg_r_matrix), avg_r_matrix.shape)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(avg_r_matrix, cmap="RdYlGn", aspect="auto",
                    vmin=-abs(avg_r_matrix).max() or -1, vmax=abs(avg_r_matrix).max() or 1)
    ax.set_xticks(range(len(rr_list)))
    ax.set_xticklabels(rr_list)
    ax.set_yticks(range(len(rm_list)))
    ax.set_yticklabels(rm_list)
    ax.set_xlabel("REWARD_RISK")
    ax.set_ylabel("RANGE_MINUTES")
    ax.set_title("ORB indices - avg R/trade by parameter (full 2016-2025, all 6 indices)")
    for i in range(len(rm_list)):
        for j in range(len(rr_list)):
            marker = " *" if (i, j) == (best_i, best_j) else ""
            ax.text(j, i, f"{avg_r_matrix[i, j]:+.3f}{marker}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="avg R/trade")
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved heatmap to {out_path}")
    except Exception as exc:
        print(f"\nCould not save heatmap PNG ({exc}) - showing inline only.")
    plt.show()


def print_top_wide_results(grid_results, top_n=10):
    """Text-table analog of print_grid_table for the wide search's higher-dimensional results -
    a heatmap doesn't generalize past 2 axes, so this just prints the top N evaluated combos by
    score, sorted descending."""
    ranked = sorted(grid_results, key=lambda r: r["score"], reverse=True)
    print(f"\nTop {min(top_n, len(ranked))} of {len(ranked)} evaluated WIDE_SEARCH combos (by {OBJECTIVE}):")
    for r in ranked[:top_n]:
        print(f"  score={r['score']:+9.4f}  n_trades={r['n_trades']:5d}  avg_r={r['avg_r']:+.4f}  "
              f"{_format_params(r['params'])}")


# ============================= STEP 2 / STEP W2: Monte Carlo per cell =============================

def monte_carlo_bootstrap(r_values, n_iter=MC_ITERATIONS, rng=None):
    """Bootstrap resample WITH replacement, same N as original trade count, vectorized (one
    (n_iter, n) integer draw + fancy-indexing, not a Python-level loop) - identical implementation
    to ict_po3_forex_dukascopy_optimization.py's / day_trading_rauf_dukascopy_optimization.py's
    own monte_carlo_bootstrap (duplicated here rather than shared, matching this project's existing
    per-script convention)."""
    r = np.asarray(r_values, dtype=float)
    n = len(r)
    if n == 0:
        return {"total_r": np.array([]), "max_drawdown": np.array([])}
    rng = rng if rng is not None else np.random.default_rng(42)
    idx = rng.integers(0, n, size=(n_iter, n))
    sampled = r[idx]
    cum = np.cumsum(sampled, axis=1)
    total_r = cum[:, -1]
    running_max = np.maximum.accumulate(cum, axis=1)
    max_dd = (running_max - cum).max(axis=1)
    return {"total_r": total_r, "max_drawdown": max_dd}


def monte_carlo_shuffle(r_values, n_iter=MC_ITERATIONS, rng=None):
    """Shuffle WITHOUT replacement (pure reordering of the SAME trades), vectorized via a per-row
    random-argsort permutation - total R is invariant under reordering by construction; only
    drawdown carries new information here. Same implementation as the PO3/Rauf companion
    scripts."""
    r = np.asarray(r_values, dtype=float)
    n = len(r)
    if n == 0:
        return {"total_r": np.array([]), "max_drawdown": np.array([])}
    rng = rng if rng is not None else np.random.default_rng(42)
    perm = np.argsort(rng.random((n_iter, n)), axis=1)
    shuffled = r[perm]
    cum = np.cumsum(shuffled, axis=1)
    total_r = cum[:, -1]
    running_max = np.maximum.accumulate(cum, axis=1)
    max_dd = (running_max - cum).max(axis=1)
    return {"total_r": total_r, "max_drawdown": max_dd}


def summarize_mc(mc_result):
    total_r, max_dd = mc_result["total_r"], mc_result["max_drawdown"]
    if len(total_r) == 0:
        return None
    return {
        "total_r_p5": float(np.percentile(total_r, 5)),
        "total_r_p50": float(np.percentile(total_r, 50)),
        "total_r_p95": float(np.percentile(total_r, 95)),
        "dd_p5": float(np.percentile(max_dd, 5)),
        "dd_p50": float(np.percentile(max_dd, 50)),
        "dd_p95": float(np.percentile(max_dd, 95)),
        "p_total_r_le_0": float(np.mean(total_r <= 0)),
    }


def run_monte_carlo_all_cells(grid_results, n_iter=MC_ITERATIONS):
    """Runs BOTH Monte Carlo procedures on EVERY evaluated cell's realized trade-R list (not just
    the best cell) - works identically for narrow-grid and WIDE_SEARCH cell shapes, since it only
    reads "trades_r"."""
    mc_results = []
    for cell_idx, cell in enumerate(grid_results):
        boot = monte_carlo_bootstrap(cell["trades_r"], n_iter=n_iter, rng=np.random.default_rng(10_000 + cell_idx))
        shuf = monte_carlo_shuffle(cell["trades_r"], n_iter=n_iter, rng=np.random.default_rng(20_000 + cell_idx))
        mc_results.append({
            "params": cell["params"],
            "n_trades": cell["n_trades"],
            "bootstrap": summarize_mc(boot),
            "shuffle": summarize_mc(shuf),
        })
    return mc_results


def print_monte_carlo_table(mc_results):
    print(f"\nMonte Carlo per cell ({MC_ITERATIONS} iterations each; bootstrap = resample WITH "
          "replacement, estimates total-R and drawdown variability; shuffle = reorder WITHOUT "
          "replacement, isolates pure sequence/path risk - shuffle's total-R is deterministic by "
          "construction, so only its drawdown percentiles carry new information):")
    for cell in mc_results:
        boot, shuf = cell["bootstrap"], cell["shuffle"]
        print(f"\n  {_format_params(cell['params'])}  ({cell['n_trades']} trades)")
        if boot is None or shuf is None:
            print("    no trades - skipped")
            continue
        print(f"    bootstrap total R   p5={boot['total_r_p5']:+8.2f}  p50={boot['total_r_p50']:+8.2f}  "
              f"p95={boot['total_r_p95']:+8.2f}   P(total R<=0)={boot['p_total_r_le_0']:.3f}")
        print(f"    bootstrap max DD    p5={boot['dd_p5']:8.2f}  p50={boot['dd_p50']:8.2f}  "
              f"p95={boot['dd_p95']:8.2f}")
        print(f"    shuffle   total R   p5={shuf['total_r_p5']:+8.2f}  p50={shuf['total_r_p50']:+8.2f}  "
              f"p95={shuf['total_r_p95']:+8.2f}   P(total R<=0)={shuf['p_total_r_le_0']:.3f}")
        print(f"    shuffle   max DD    p5={shuf['dd_p5']:8.2f}  p50={shuf['dd_p50']:8.2f}  "
              f"p95={shuf['dd_p95']:8.2f}")


# ============================= STEP 3 / STEP W3: cluster analysis =============================

def neighbor_plateau_check(grid_results, rm_grid=None, rr_grid=None):
    """Always-available (no sklearn needed) check: look at the best cell's immediate
    RANGE_MINUTES/REWARD_RISK grid neighbors (up/down/left/right, fewer at grid edges). PLATEAU if
    most neighbors are decent (same sign as the peak and within a reasonable ratio of its
    magnitude) - ISOLATED SPIKE / overfit warning if neighbors are starkly worse or flip sign.
    Only meaningful over a full 2-axis exhaustive grid - see print_cluster_analysis for when this
    is skipped."""
    rm_grid = rm_grid if rm_grid is not None else sorted(set(r["range_minutes"] for r in grid_results))
    rr_grid = rr_grid if rr_grid is not None else sorted(set(r["reward_risk"] for r in grid_results))
    lookup = {(r["range_minutes"], r["reward_risk"]): r for r in grid_results}

    best = max(grid_results, key=lambda r: r["total_r"])
    bi = rm_grid.index(best["range_minutes"])
    bj = rr_grid.index(best["reward_risk"])
    peak_avg = best["avg_r"]

    neighbors = []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = bi + di, bj + dj
        if 0 <= ni < len(rm_grid) and 0 <= nj < len(rr_grid):
            neighbors.append(lookup[(rm_grid[ni], rr_grid[nj])])

    decent_flags = []
    for nb in neighbors:
        nb_avg = nb["avg_r"]
        if peak_avg == 0:
            decent_flags.append(abs(nb_avg) < 1e-9)
        else:
            same_sign = (nb_avg >= 0) == (peak_avg >= 0)
            ratio_ok = abs(nb_avg) >= 0.5 * abs(peak_avg)
            decent_flags.append(same_sign and ratio_ok)

    frac_decent = (sum(decent_flags) / len(decent_flags)) if decent_flags else 0.0
    verdict = "PLATEAU" if frac_decent >= 0.5 else "ISOLATED SPIKE / overfit warning"
    return {
        "best_range_minutes": best["range_minutes"],
        "best_reward_risk": best["reward_risk"],
        "best_avg_r": peak_avg,
        "n_neighbors": len(neighbors),
        "n_decent_neighbors": sum(decent_flags),
        "frac_decent": frac_decent,
        "verdict": verdict,
    }


def sklearn_cluster_analysis(grid_results, param_names=("range_minutes", "reward_risk"), k=3, random_state=42):
    """sklearn KMeans (k=3) over (normalized position along each of `param_names`, avg R/trade)
    feature vectors across every evaluated cell. Generalized (unlike a hardcoded 2-axis version)
    to work over any number of params - the narrow grid's 2 and WIDE_PARAM_GRID's 6 both work
    unchanged. Identifies which cluster the best-in-sample cell (highest total R) falls in, and
    reports that cluster's size and its members' min/mean avg R/trade. Wrapped in try/except
    ImportError - prints a clear skip message and returns None rather than crashing, matching this
    project's existing sklearn-missing handling pattern."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("\nscikit-learn not installed - skipping cluster analysis. Run "
              "'!pip install scikit-learn -q' and re-run.")
        return None

    axis_positions = {}
    for name in param_names:
        vals = sorted(set(r["params"][name] for r in grid_results))
        axis_positions[name] = {v: (i / (len(vals) - 1) if len(vals) > 1 else 0.0) for i, v in enumerate(vals)}

    X = np.array([[axis_positions[name][r["params"][name]] for name in param_names] + [r["avg_r"]]
                  for r in grid_results])
    n_clusters = min(k, len(grid_results))
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    best_idx = max(range(len(grid_results)), key=lambda i: grid_results[i]["total_r"])
    best_cluster = int(labels[best_idx])
    members = [grid_results[i] for i in range(len(grid_results)) if labels[i] == best_cluster]
    avg_rs = [m["avg_r"] for m in members]

    return {
        "labels": labels,
        "best_cluster": best_cluster,
        "cluster_size": len(members),
        "cluster_min_avg_r": min(avg_rs),
        "cluster_mean_avg_r": sum(avg_rs) / len(avg_rs),
        "cluster_members": [m["params"] for m in members],
    }


def print_cluster_analysis(grid_results, search_method=None, param_names=("range_minutes", "reward_risk")):
    """search_method defaults to the module-level SEARCH_METHOD global. The always-available
    neighbor-plateau check requires a full exhaustive 2-axis grid (every neighboring combo must
    actually have been evaluated) - it is skipped (with an explicit message) under any non-"grid"
    SEARCH_METHOD, or when param_names has more than 2 entries (i.e. the WIDE_SEARCH path), for
    the same reason ict_po3_forex_dukascopy_optimization.py's print_cluster_analysis skips it:
    silently running the lookup over a smaller, non-random sample of neighbors than the check was
    designed for would produce a materially misleading PLATEAU/SPIKE verdict, not just an
    incomplete one. The sklearn cluster analysis has no such requirement - it derives its own axis
    positions from whatever distinct parameter values are actually present in grid_results, so it
    stays valid (just over however many points were actually evaluated) under any SEARCH_METHOD or
    param count, and always runs."""
    search_method = search_method if search_method is not None else SEARCH_METHOD

    print("\n" + "-" * 70)
    print("Cluster / plateau-vs-spike analysis")
    print("-" * 70)

    neighbor_result = None
    if search_method == "grid" and len(param_names) == 2 and param_names == ("range_minutes", "reward_risk"):
        neighbor_result = neighbor_plateau_check(grid_results)
        print(f"\nAlways-available neighbor check (best cell = RANGE_MINUTES="
              f"{neighbor_result['best_range_minutes']}, REWARD_RISK="
              f"{neighbor_result['best_reward_risk']:.1f}, avg R/trade="
              f"{neighbor_result['best_avg_r']:+.4f}):")
        print(f"  {neighbor_result['n_decent_neighbors']}/{neighbor_result['n_neighbors']} immediate neighbors "
              f"are 'decent' (same sign, at least half the peak's magnitude) -> {neighbor_result['verdict']}")
    else:
        print(f"\nNeighbor-plateau check: SKIPPED - requires a full exhaustive 2-axis "
              f"(RANGE_MINUTES x REWARD_RISK) grid, not available under SEARCH_METHOD={search_method!r} "
              f"and/or a {len(param_names)}-parameter search space (only a sample of the space was "
              f"evaluated, so an immediate-neighbor combo may never have been tried).")

    cluster_result = sklearn_cluster_analysis(grid_results, param_names=param_names)
    if cluster_result is not None:
        if search_method != "grid" or len(param_names) != 2:
            print(f"\nNOTE: cluster analysis below is over the {len(grid_results)} points actually "
                  f"evaluated (SEARCH_METHOD={search_method!r}, {len(param_names)} parameters), not a "
                  f"full grid - axis positions are relative to those observed points only.")
        print(f"\nsklearn KMeans (k=3) cluster containing the best-in-sample cell: "
              f"{cluster_result['cluster_size']} of {len(grid_results)} cells")
        print(f"  cluster avg R/trade: min={cluster_result['cluster_min_avg_r']:+.4f}  "
              f"mean={cluster_result['cluster_mean_avg_r']:+.4f}")
        print(f"  cluster members: {cluster_result['cluster_members']}")

    return neighbor_result, cluster_result


# ============================= STEP 4 / STEP W4: rolling walk-forward =============================

def generate_walk_forward_folds(fetch_start_year, fetch_end_year,
                                 is_years=WALK_FORWARD_IS_YEARS,
                                 oos_years=WALK_FORWARD_OOS_YEARS,
                                 step_years=WALK_FORWARD_STEP_YEARS):
    """Generates rolling (not anchored) walk-forward fold boundaries as plain datetime.date
    year-starts - identical convention (and, with this file's FETCH_START/FETCH_END, identical
    boundaries) to ict_po3_forex_dukascopy_optimization.py's / day_trading_rauf_dukascopy_optimization.py's
    own generate_walk_forward_folds: IS = [is_start, is_end), OOS = [oos_start, oos_end) =
    [is_end, is_end+oos_years). With the defaults (2016, 2025, 3, 1, 1) this produces exactly 6
    folds."""
    folds = []
    is_start_year = fetch_start_year
    while True:
        is_end_year = is_start_year + is_years
        oos_start_year = is_end_year
        oos_end_year = oos_start_year + oos_years
        if oos_end_year > fetch_end_year:
            break
        folds.append({
            "is_start": datetime.date(is_start_year, 1, 1),
            "is_end": datetime.date(is_end_year, 1, 1),
            "oos_start": datetime.date(oos_start_year, 1, 1),
            "oos_end": datetime.date(oos_end_year, 1, 1),
        })
        is_start_year += step_years
    return folds


def run_walk_forward(data, folds, param_grid=None, method=None, objective=None, **search_kwargs):
    """For each fold: re-run the in-sample parameter search (via run_param_search, dispatching to
    whichever method is configured) restricted to the fold's in-sample window, pick the best combo
    by whichever objective is configured, apply that exact combo unchanged to the immediately-
    following out-of-sample window. Chains all folds' OOS trades together.

    DEFAULTS REPRODUCE THE NARROW-SEARCH BEHAVIOR: method=None/objective=None/param_grid=None
    resolve to the module-level SEARCH_METHOD="grid"/OBJECTIVE="total_r"/narrow grid - pass
    param_grid=WIDE_PARAM_GRID and method=WIDE_SEARCH_METHOD for the wide-search walk-forward
    (STEP W4)."""
    fold_results = []
    combined_oos_r = []

    for fold_idx, fold in enumerate(folds, start=1):
        is_start, is_end = fold["is_start"], fold["is_end"]
        oos_start, oos_end = fold["oos_start"], fold["oos_end"]

        is_grid, is_search_result = run_param_search(
            data, window_start=is_start, window_end=is_end, param_grid=param_grid, method=method,
            objective=objective, desc=f"fold {fold_idx} in-sample search", **search_kwargs)
        best_params = is_search_result["best"]["params"]
        best = _find_cell(is_grid, best_params)

        oos_all_r = []
        for label, (df, ind, tz_name, session_start) in data.items():
            trades = run_backtest(
                df, ind, tz_name, session_start, best_params["range_minutes"], best_params["reward_risk"],
                apply_filters=True, split_before=oos_end, split_after=oos_start,
                min_range_atr_mult=best_params.get("min_range_atr_mult"),
                max_range_atr_mult=best_params.get("max_range_atr_mult"),
                impulse_range_mult=best_params.get("impulse_range_mult"),
                rvol_mult=best_params.get("rvol_mult"),
            )
            oos_all_r.extend(t["r"] for t in trades)
        combined_oos_r.extend(oos_all_r)

        oos_total = sum(oos_all_r)
        oos_n = len(oos_all_r)
        fold_results.append({
            "fold": fold_idx,
            "is_start": is_start, "is_end": is_end, "oos_start": oos_start, "oos_end": oos_end,
            "best_params": best_params,
            "is_total_r": best["total_r"], "is_n_trades": best["n_trades"], "is_avg_r": best["avg_r"],
            "oos_total_r": oos_total, "oos_n_trades": oos_n,
            "oos_avg_r": (oos_total / oos_n) if oos_n else 0.0,
        })

    return fold_results, combined_oos_r


def compute_walk_forward_efficiency(fold_results, combined_oos_r):
    combined_oos_total = sum(combined_oos_r)
    combined_oos_n = len(combined_oos_r)
    combined_oos_avg = (combined_oos_total / combined_oos_n) if combined_oos_n else 0.0

    is_avgs = [f["is_avg_r"] for f in fold_results]
    mean_is_avg = (sum(is_avgs) / len(is_avgs)) if is_avgs else 0.0

    if abs(mean_is_avg) < 1e-9:
        wfe = float("nan")
    else:
        wfe = combined_oos_avg / mean_is_avg

    return {
        "combined_oos_total_r": combined_oos_total,
        "combined_oos_n_trades": combined_oos_n,
        "combined_oos_avg_r": combined_oos_avg,
        "mean_is_avg_r": mean_is_avg,
        "wfe": wfe,
    }


def print_walk_forward(fold_results, wfe_stats, title="rolling walk-forward (3yr in-sample / 1yr out-of-sample, rolled 1yr at a time)"):
    print("\n" + "-" * 70)
    print(f"STEP 4: {title}")
    print("-" * 70)
    for f in fold_results:
        print(f"\nFold {f['fold']}: IS {f['is_start']} to {f['is_end']}   OOS {f['oos_start']} to {f['oos_end']}")
        print(f"  best params: {_format_params(f['best_params'])}")
        print(f"  IS: {f['is_n_trades']:5d} trades, {f['is_total_r']:+8.2f}R, {f['is_avg_r']:+.4f}R/trade    "
              f"OOS: {f['oos_n_trades']:5d} trades, {f['oos_total_r']:+8.2f}R, {f['oos_avg_r']:+.4f}R/trade")

    print(f"\nCombined out-of-sample (all {len(fold_results)} folds chained together): "
          f"{wfe_stats['combined_oos_n_trades']} trades, {wfe_stats['combined_oos_total_r']:+.2f}R, "
          f"{wfe_stats['combined_oos_avg_r']:+.4f}R/trade")
    print(f"Mean in-sample avg R/trade of the selected combo across folds: {wfe_stats['mean_is_avg_r']:+.4f}")

    if wfe_stats["mean_is_avg_r"] == 0 or np.isnan(wfe_stats["wfe"]):
        print("\nWalk-Forward Efficiency: undefined (mean in-sample avg R/trade is ~0) - cannot compute "
              "a meaningful ratio here either way.")
    else:
        wfe = wfe_stats["wfe"]
        verdict = "PASS" if wfe >= WFE_PASS_THRESHOLD else "FAIL"
        print(f"\nWalk-Forward Efficiency (combined OOS avg R/trade / mean IS avg R/trade) = {wfe:.3f} -> {verdict}")
        print(f"CAVEAT: {WFE_PASS_THRESHOLD:.0%} is a standard rule-of-thumb threshold in walk-forward "
              "analysis, not a proof of robustness (a PASS) or non-robustness (a FAIL) on its own.")
        if wfe_stats["mean_is_avg_r"] < 0:
            print("ADDITIONAL CAVEAT: the selected in-sample combo's own avg R/trade is negative - a "
                  "'PASS' here only means out-of-sample decayed by less than half relative to an "
                  "already-losing in-sample baseline, not that the strategy is profitable.")


def _resolve_wide_search_method():
    """WIDE_SEARCH_METHOD must be "bayesian" or "genetic" - NEVER "grid". Exhaustive search over
    WIDE_PARAM_GRID's full 6,400-combo cross product is exactly the "would very likely just find
    noise" scenario this file's own header warns about; forcing a fallback here (rather than
    silently letting a misconfigured "grid" through) keeps that guarantee true regardless of what
    WIDE_SEARCH_METHOD gets set to."""
    if WIDE_SEARCH_METHOD not in ("bayesian", "genetic"):
        print(f"WIDE_SEARCH_METHOD={WIDE_SEARCH_METHOD!r} is not allowed for the wide-search path - "
              "exhaustive grid search over this much wider space is exactly the overfitting risk this "
              "file's own header warns about. Forcing 'bayesian' instead.")
        return "bayesian"
    return WIDE_SEARCH_METHOD


def print_wide_search_caveat():
    print("\n" + "=" * 70)
    print("WIDE_SEARCH CAVEAT - READ BEFORE TRUSTING ANYTHING BELOW")
    print("=" * 70)
    print("This run additionally searched MIN_RANGE_ATR_MULT/MAX_RANGE_ATR_MULT/IMPULSE_RANGE_MULT/"
          "RVOL_MULT alongside RANGE_MINUTES/REWARD_RISK - a much wider space than this project's "
          "narrow, already-verified 2-parameter default. This project's research/optimization_engine.py "
          "does not yet have a lockbox/embargoed-holdout mechanism wired in anywhere (checked via "
          "`grep -n \"lockbox\\|split_lockbox\" research/optimization_engine.py` immediately before this "
          "run - no match). Any promising result from this wide search should be treated with EXTRA "
          "skepticism, more so than the narrow default's result above, even though it also gets its own "
          "Monte Carlo, cluster analysis, and rolling walk-forward below - more searched dimensions over "
          "the same finite historical data is inherently more overfitting surface, walk-forward or not, "
          "without a genuine held-out check this project does not have wired in yet. If a lockbox lands "
          "in optimization_engine.py later, this is exactly the path it should be wired into first.")


# ============================= main =============================

def main():
    years = (FETCH_END - FETCH_START).days / 365
    n_narrow = len(RANGE_MINUTES_GRID) * len(REWARD_RISK_GRID)
    n_folds = len(generate_walk_forward_folds(FETCH_START.year, FETCH_END.year))
    print(f"ORB optimization/robustness pass: {n_narrow}-cell narrow grid search, {MC_ITERATIONS}-iteration "
          f"Monte Carlo x2 per cell, cluster analysis, a {n_folds}-fold rolling walk-forward (each fold "
          f"re-runs the search on its in-sample window), plus ML scoring, over ~{years:.0f} years of "
          f"{len(INDICES)} indices"
          + (f", PLUS an optional WIDE_SEARCH pass ({WIDE_SEARCH_METHOD!r}, {WIDE_SEARCH_N_TRIALS} trials) "
             "over 4 additional filter thresholds" if WIDE_SEARCH else "")
          + " - this can take well over an hour end to end against real data, not a hang.\n")

    print(f"Downloading {len(INDICES)} indices from Dukascopy ({FETCH_START.date()} to {FETCH_END.date()})...")
    data = {}
    for label, instrument_const, tz_name, session_start in INDICES:
        df = fetch_index_data(instrument_const, tz_name)
        if df is None:
            print(f"  {label}: no data")
            continue
        data[label] = (df, precompute_indicators(df), tz_name, session_start)
        print(f"  {label}: {len(df)} bars")

    if not data:
        print("No data downloaded - check output above.")
        return

    # ================= PART 1: parameter optimization + robustness (narrow, default) =================
    print("\n" + "=" * 70)
    print(f"PART 1: parameter search (SEARCH_METHOD={SEARCH_METHOD!r}, OBJECTIVE={OBJECTIVE!r}) - "
          f"full {FETCH_START.date()} to {FETCH_END.date()} range, all {len(data)} indices per combo")
    print("=" * 70)
    grid_results, step1_search_result = run_param_search(data, desc="step 1 full-range search")
    if SEARCH_METHOD != "grid":
        print(f"\n{SEARCH_METHOD} search evaluated {step1_search_result['n_evals']} of {n_narrow} possible "
              f"combos this run (vs grid_search's exhaustive {n_narrow}).")
    print_grid_table(grid_results)
    try:
        plot_heatmap(grid_results)
    except ImportError:
        print("\nmatplotlib not installed - skipping heatmap plot (text table above is the fallback).")

    best_cell = _find_cell(grid_results, step1_search_result["best"]["params"])
    worst_cell = min(grid_results, key=lambda r: r["total_r"])
    best_range_minutes, best_reward_risk = best_cell["range_minutes"], best_cell["reward_risk"]
    print(f"\nBest cell:  RANGE_MINUTES={best_cell['range_minutes']}  REWARD_RISK={best_cell['reward_risk']:.1f}  "
          f"-> {best_cell['total_r']:+.2f}R over {best_cell['n_trades']} trades ({best_cell['avg_r']:+.4f}R/trade)")
    print(f"Worst cell: RANGE_MINUTES={worst_cell['range_minutes']}  REWARD_RISK={worst_cell['reward_risk']:.1f}  "
          f"-> {worst_cell['total_r']:+.2f}R over {worst_cell['n_trades']} trades ({worst_cell['avg_r']:+.4f}R/trade)")
    print(f"Avg-R/trade spread across the whole grid: {best_cell['avg_r'] - worst_cell['avg_r']:.4f}")

    print("\n" + "-" * 70)
    print(f"STEP 2: Monte Carlo per cell ({MC_ITERATIONS} iterations each)")
    print("-" * 70)
    mc_results = run_monte_carlo_all_cells(grid_results)
    print_monte_carlo_table(mc_results)

    neighbor_result, cluster_result = print_cluster_analysis(grid_results, search_method=SEARCH_METHOD)

    folds = generate_walk_forward_folds(FETCH_START.year, FETCH_END.year)
    fold_results, combined_oos_r = run_walk_forward(data, folds)
    wfe_stats = compute_walk_forward_efficiency(fold_results, combined_oos_r)
    print_walk_forward(fold_results, wfe_stats)

    print("\n" + "-" * 70)
    print("CORRECTED SIGNIFICANCE CHECK: best-cell z-score + Bonferroni-adjusted bar")
    print("-" * 70)
    best_cell_trades = [{"r": r} for r in best_cell["trades_r"]]
    best_z = opt_engine.zscore(best_cell_trades)
    n_trials_this_run = step1_search_result["n_evals"]
    z_bar = opt_engine.bonferroni_adjusted_z_threshold(n_trials_this_run)
    clears_adjusted = abs(best_z) >= z_bar
    print(f"Best cell (RANGE_MINUTES={best_cell['range_minutes']}, REWARD_RISK={best_cell['reward_risk']:.1f}) "
          f"corrected z-score: {best_z:+.2f} (sample-std-based, not an avg_r*sqrt(n) shortcut)")
    print(f"Combos tested this run: {n_trials_this_run} -> Bonferroni-adjusted |z| bar = {z_bar:.2f} "
          f"(vs the naive single-test 1.96 rule of thumb)")
    print(f"-> {'CLEARS' if clears_adjusted else 'does NOT clear'} the multiple-testing-adjusted bar")

    print("\n" + "=" * 70)
    print("PART 1 FINAL VERDICT (narrow, default search)")
    print("=" * 70)
    grid_all_negative = all(r["total_r"] <= 0 for r in grid_results)
    best_cell_positive = best_cell["total_r"] > 0
    oos_positive = wfe_stats["combined_oos_avg_r"] > 0
    wfe_pass = (not np.isnan(wfe_stats["wfe"])) and wfe_stats["wfe"] >= WFE_PASS_THRESHOLD

    if grid_all_negative:
        print("Every single cell of the parameter grid was net-negative over the full range - no corner "
              "of this narrow parameter space rescues ORB here.")
    elif best_cell_positive and neighbor_result is not None and neighbor_result["verdict"] == "ISOLATED SPIKE / overfit warning":
        print("The best in-sample cell was positive, but it is an ISOLATED SPIKE - its immediate "
              "neighbors are starkly worse or flip sign. Treat any positive number here with real "
              "suspicion.")
    elif best_cell_positive and not oos_positive:
        print("At least one cell was positive, but the chained out-of-sample result from the rolling "
              "walk-forward was NOT - the winning parameters did not transfer forward.")
    elif best_cell_positive and oos_positive and wfe_pass:
        print("At least one cell was positive, its neighbors were also decent (a real plateau, not a "
              "spike), and the rolling walk-forward's chained out-of-sample result was also positive "
              f"with WFE >= {WFE_PASS_THRESHOLD:.0%} - the strongest evidence this rigor pass can offer "
              "for a real (if not yet proven) edge, not a green light on its own.")
    else:
        print("Results are mixed across the grid, Monte Carlo, cluster, and walk-forward checks - see "
              "the sections above for the specific numbers.")

    # ================= PART 1b: OPTIONAL WIDE SEARCH =================
    if WIDE_SEARCH:
        wide_method = _resolve_wide_search_method()
        print_wide_search_caveat()

        print("\n" + "=" * 70)
        print(f"WIDE SEARCH: {wide_method} search over RANGE_MINUTES/REWARD_RISK + 4 filter thresholds "
              f"(6,400-combo full cross product, NOT exhaustively searched)")
        print("=" * 70)
        wide_kwargs = {"n_trials": WIDE_SEARCH_N_TRIALS} if wide_method == "bayesian" else \
            {"population_size": WIDE_SEARCH_POPULATION, "generations": WIDE_SEARCH_GENERATIONS}
        wide_grid_results, wide_search_result = run_param_search(
            data, param_grid=WIDE_PARAM_GRID, method=wide_method, desc="wide search", **wide_kwargs)
        print(f"\n{wide_method} search evaluated {wide_search_result['n_evals']} distinct combos "
              f"(vs 6,400 in the full cross product).")
        print_top_wide_results(wide_grid_results)

        wide_best = _find_cell(wide_grid_results, wide_search_result["best"]["params"])
        print(f"\nBest wide-search combo: {_format_params(wide_best['params'])} -> {wide_best['total_r']:+.2f}R "
              f"over {wide_best['n_trades']} trades ({wide_best['avg_r']:+.4f}R/trade)")
        print(f"For comparison, the narrow default's best cell: {best_cell['total_r']:+.2f}R over "
              f"{best_cell['n_trades']} trades ({best_cell['avg_r']:+.4f}R/trade)")

        print("\n" + "-" * 70)
        print(f"STEP W2: Monte Carlo per wide-search cell ({MC_ITERATIONS} iterations each)")
        print("-" * 70)
        wide_mc_results = run_monte_carlo_all_cells(wide_grid_results)
        print_monte_carlo_table(wide_mc_results)

        print_cluster_analysis(wide_grid_results, search_method=wide_method,
                                param_names=tuple(WIDE_PARAM_GRID.keys()))

        wide_fold_results, wide_combined_oos_r = run_walk_forward(
            data, folds, param_grid=WIDE_PARAM_GRID, method=wide_method, **wide_kwargs)
        wide_wfe_stats = compute_walk_forward_efficiency(wide_fold_results, wide_combined_oos_r)
        print_walk_forward(wide_fold_results, wide_wfe_stats,
                            title=f"wide-search rolling walk-forward ({wide_method}, 4 extra filter thresholds)")

        print("\nREMINDER: see the WIDE_SEARCH CAVEAT printed above before drawing any conclusion from "
              "this section - extra skepticism, not less, applies here.")

    # ================= PART 2: ML scoring =================
    print("\n" + "=" * 70)
    print("PART 2: ML classifier scoring raw breakouts (filters off, all candidates labeled), "
          "using the best RANGE_MINUTES from Part 1's full-range search")
    print("=" * 70)

    in_sample_candidates, oos_candidates = [], []
    for label, (df, ind, tz_name, session_start) in data.items():
        in_sample_candidates.extend(run_backtest(df, ind, tz_name, session_start, best_range_minutes, 1.0,
                                                   apply_filters=False, split_before=SPLIT_DATE,
                                                   collect_candidates=True))
        oos_candidates.extend(run_backtest(df, ind, tz_name, session_start, best_range_minutes, 1.0,
                                            apply_filters=False, split_after=SPLIT_DATE,
                                            collect_candidates=True))

    print(f"In-sample candidates: {len(in_sample_candidates)}   Out-of-sample candidates: {len(oos_candidates)}")

    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        print("scikit-learn not installed - run '!pip install scikit-learn -q' and re-run. Skipping ML step.")
        return

    train_df = pd.DataFrame([{**c["features"], "label": c["label"]} for c in in_sample_candidates])
    test_df = pd.DataFrame([{**c["features"], "label": c["label"], "r": c["r"]} for c in oos_candidates])

    train_df = train_df.dropna(subset=FEATURE_COLUMNS + ["label"])
    test_clean = test_df.dropna(subset=FEATURE_COLUMNS + ["label"])
    print(f"After dropping incomplete-feature rows: train={len(train_df)}, test={len(test_clean)}")

    if len(train_df) < 30 or len(test_clean) < 10:
        print("Not enough clean candidates to train/evaluate a model meaningfully. Stopping here.")
        return

    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"])

    test_clean = test_clean.copy()
    test_clean["pred_proba"] = model.predict_proba(test_clean[FEATURE_COLUMNS])[:, 1]

    baseline_r = test_clean["r"].sum()
    print(f"\nBaseline (every raw breakout, no ML filtering): {len(test_clean)} trades, "
          f"{baseline_r:+.2f}R, {baseline_r/len(test_clean):+.4f}R/trade")

    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        filtered = test_clean[test_clean["pred_proba"] >= threshold]
        if len(filtered) == 0:
            print(f"  threshold {threshold:.2f}: 0 trades")
            continue
        total_r = filtered["r"].sum()
        print(f"  threshold {threshold:.2f}: {len(filtered):4d} trades, {total_r:+8.2f}R, "
              f"{total_r/len(filtered):+.4f}R/trade")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print(f"\nFeature importances:\n{importances}")

    print(f"\nCompare this out-of-sample R/trade against Part 1's hand-tuned-filter walk-forward "
          f"result ({wfe_stats['combined_oos_avg_r']:+.4f}R/trade combined OOS) - similar test spirit, "
          f"different selection method and (for the ML step specifically) a single fixed split rather "
          f"than rolling folds. Whichever wins here is only a real finding if it's not a razor-thin "
          f"difference on a few hundred out-of-sample trades.")


if __name__ == "__main__":
    main()
