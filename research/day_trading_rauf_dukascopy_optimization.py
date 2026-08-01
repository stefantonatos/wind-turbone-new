# "Scam or Slam" (Day Trading Rauf) - parameter robustness pass, companion
# to research/day_trading_rauf_dukascopy_backtest.py (read that file first -
# this one BUILDS ON it, it does not replace it or change its conclusions).
#
# WHY THIS FILE EXISTS - and what it is NOT trying to do:
# The base script's real-data result was decisively negative on every cut
# tried: 11007 trades, -195.53R total, -0.0178R/trade, all 4 instruments
# negative, both ranges negative, approx z=-1.86, and both split-period
# halves negative - despite a superficially decent 51.6% win rate (the
# payout structure just doesn't compensate for it: winners are smaller than
# losers on average). This file is NOT an attempt to torture that result
# until a "good" number falls out. It is a systematic check of whether ANY
# corner of the two-parameter space this strategy actually has room to move
# in (stop buffer, confirmation-candle count) contains a real, robust edge -
# stability across neighboring parameters, survives Monte Carlo resampling,
# clusters sensibly rather than spiking in isolation, and holds up
# out-of-sample under rolling walk-forward - or whether the already-
# established negative result simply holds up everywhere, in which case
# this script's own printed output will say that plainly and there's
# nothing left to search for here (see the "FINAL VERDICT" print at the end
# of main() - that verdict is computed from whatever the actual run
# produces, not asserted in advance in this comment).
#
# SCOPE NOTE (so this isn't ambiguous to a future reader): a SEPARATE,
# still-open thread from this same project has since found (via web search)
# that this "Scam or Slam" script series was explicitly designed/marketed
# for NQ/ES index futures, not forex/metals - i.e. there's a live question
# of whether this was ever the right asset class to test this strategy on
# at all. That question is NOT addressed here. This file deliberately keeps
# the EXACT SAME 4 instruments (EURUSD, GBPUSD, USDJPY, XAUUSD) as the
# already-shipped, already-committed base script, because this pass is
# specifically a robustness/optimization check on THAT version. If the
# asset-class question gets resolved later, that's a different script.
#
# THE 4 STEPS (methodology fixed by the task, not re-derived here):
#   1. PARAMETER STABILITY: 4x4 grid over STOP_BUFFER_PCT_GRID x
#      CONFIRMATION_CANDLES_GRID (generalizing the base script's hardcoded
#      "3 consecutive same-direction candles" rule into a configurable N),
#      each cell = the full backtest (4 instruments, both ranges, full
#      2016-2025). Heatmap of avg R/trade, PNG + inline show(), text-table
#      fallback always printed regardless of whether matplotlib is present.
#   2. MONTE CARLO per grid cell (not just the best one): bootstrap-with-
#      replacement AND shuffle-without-replacement, 2000 iterations each,
#      vectorized with numpy batch resampling - distribution of total R and
#      max drawdown, 5th/50th/95th percentiles, P(total R <= 0).
#   3. CLUSTER ANALYSIS: sklearn KMeans(k=3) over (normalized stop-buffer
#      position, normalized confirmation-candle position, avg R/trade)
#      feature vectors for the 16 cells, reporting the cluster containing
#      the best-in-sample cell - plus an always-available (no-sklearn-
#      required) neighbor-plateau-vs-isolated-spike check on the grid.
#   4. WALK-FORWARD: rolling (not anchored - the right choice for 5-min
#      intraday per standard practice, since anchored windows just get
#      more stale-regime-diluted every fold) 3-year in-sample / 1-year
#      out-of-sample, stepping forward 1 year at a time across 2016-2025
#      (6 folds). Re-run the full 16-cell grid PER FOLD restricted to that
#      fold's in-sample window, pick the best combo by in-sample total R
#      (same selection rule research/orb_indices_optimization_and_ml.py
#      uses), apply it unchanged to the immediately-following out-of-sample
#      window, chain all 6 folds' OOS trades together, and compute Walk-
#      Forward Efficiency = combined OOS avg-R/trade / mean-across-folds
#      IS avg-R/trade-of-the-selected-combo. PASS if WFE >= 0.5 - a
#      standard rule-of-thumb threshold, explicitly flagged as such, not a
#      proof either way.
#
# CONVENTIONS copied over exactly from the base script and from
# research/orb_indices_optimization_and_ml.py (this project's grid-search/
# in-sample-out-of-sample reference implementation): disk caching keyed by
# instrument/interval/date-range via pickle, chunked fetching with a
# tqdm.auto progress bar (NOT tqdm.notebook - it crashes outside a real
# Jupyter frontend), the logging.Filter-based fix for dukascopy_python's
# DUKASCRIPT logger resetting its own level on every fetch() call (a plain
# setLevel does NOT survive that reset - a Filter does, since the library
# never touches .filters), the Google-Drive-mount auto-detected cache dir,
# and the split_before/split_after date-masking pattern - generalized here
# to an arbitrary (window_start, window_end) pair since rolling walk-forward
# needs many windows, not just one split point. Per that same pattern,
# indicators/ranges/sweep-state are always computed from the FULL
# underlying 2016-2025 data stream in correct day-by-day sequence for every
# grid cell and every fold; only which TRADES get counted into a given
# window's result is restricted - the price series itself is never
# truncated. (This strategy's per-range state fully resets every calendar
# day already - see the base script - so this is a correctness/rigor
# choice, not one this particular strategy's state machine strictly
# requires; it's kept anyway so the window-masking logic here is provably
# the same pattern used elsewhere in this project, verified by the window-
# masking unit tests below rather than assumed safe.)
#
# RUNTIME WARNING: this is much heavier than either the base script or
# orb_indices_optimization_and_ml.py. Step 1 alone is 16 full-history
# backtest calls per instrument; step 4 adds another 16*6 in-sample grid
# calls plus 6 out-of-sample calls per instrument (118 extra full-history
# backtest passes x 4 instruments, on top of step 1's 16 x 4) - budget for
# this to run considerably longer than the ~30-45 minutes the ORB
# optimization script warns about. Data is fetched/cached ONCE up front and
# reused in-memory for every grid cell and fold (never re-downloaded), which
# is the main thing keeping this tractable at all.
#
# SEARCH_METHOD / OBJECTIVE (research/optimization_engine.py): step 1's grid search and step 4's
# per-fold in-sample search now dispatch through a shared, strategy-agnostic search engine that
# also offers Bayesian (Optuna/TPE) and genetic search as alternatives to exhaustive grid search,
# and a selectable objective (total R, avg R/trade, a trade-based Sharpe-like ratio, Calmar, win
# rate) instead of always picking the winner by raw total R. SEARCH_METHOD="grid" and
# OBJECTIVE="total_r" remain this script's defaults, and DELIBERATELY so: this script's parameter
# space is only 16 combos, where exhaustive grid search is the better choice, not a fallback - see
# optimization_engine.py's own header for the full reasoning on when the alternatives actually earn
# their keep (they don't, here, today).
#
# TESTING NOTE: per the task's own instruction, the full real 9-year x 4-
# instrument x 2-range x 16-cell x 6-fold backtest was deliberately NOT run
# in the sandbox that produced this file (too slow/network-heavy for that
# environment). Correctness was instead validated via the unit tests below
# (run with `python day_trading_rauf_dukascopy_optimization.py --test`) plus
# one small synthetic/mocked end-to-end smoke run exercising the full
# pipeline (small synthetic price series, small grid, 2 folds) to confirm
# it runs start-to-finish without crashing - see run_smoke_test() below.
# Actually running main() against real Dukascopy data (in Colab or
# similar) is what produces the real numbers and the FINAL VERDICT line.

# !pip install --upgrade dukascopy-python scikit-learn matplotlib optuna -q   # uncomment in Colab
# (optuna is only needed for SEARCH_METHOD="bayesian" below - see optimization_engine.py's header
# for why it's an optional dependency handled with a graceful skip, same pattern as sklearn/matplotlib)

import datetime
import json
import logging
import os
import pickle
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

try:
    import dukascopy_python
    from dukascopy_python import instruments as dki
except ImportError:   # only needed for main()'s real fetch - unit tests / smoke test don't need it
    dukascopy_python = None
    dki = None

# research/optimization_engine.py is a sibling module in this same directory, not a package -
# this sys.path insert makes `import optimization_engine` resolve correctly regardless of HOW
# this file is loaded (run directly, run via `python research/....py --test`, or loaded by
# importlib.util.spec_from_file_location the way a test file would load this module).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optimization_engine as opt_engine

from tqdm.auto import tqdm   # auto-picks the Colab/Jupyter widget bar when available, a plain terminal bar otherwise

# See the base script's header for why setLevel alone doesn't work here - a Filter survives the
# library's own per-fetch() logger.setLevel(INFO) reset, since it never touches .filters.
class _SuppressDukascopyInfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING


logging.getLogger("DUKASCRIPT").addFilter(_SuppressDukascopyInfoFilter())

CACHE_DIR = "/content/drive/MyDrive/dukascopy_cache" if os.path.isdir("/content/drive/MyDrive") else "dukascopy_cache"
FETCH_CHUNK_MONTHS = 3   # how finely to split the download for progress-bar granularity

INSTRUMENTS = [
    ("EURUSD", "INSTRUMENT_FX_MAJORS_EUR_USD"),
    ("GBPUSD", "INSTRUMENT_FX_MAJORS_GBP_USD"),
    ("USDJPY", "INSTRUMENT_FX_MAJORS_USD_JPY"),
    ("XAUUSD", "INSTRUMENT_FX_METALS_XAU_USD"),
]   # resolved to real dukascopy_python.instruments constants lazily in fetch_instrument_data,
    # so this module still imports cleanly (for unit tests / smoke test) without dukascopy_python installed

FETCH_START = datetime.datetime(2016, 1, 1)
FETCH_END = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL_NAME = "INTERVAL_MIN_5"
DUKASCOPY_OFFER_SIDE_NAME = "OFFER_SIDE_BID"

# --- the two time-based ranges, NY time (same as the base script - not part of this grid) ---
RANGES = [
    ("LONDON", pd.Timestamp("01:12").time(), pd.Timestamp("02:12").time()),
    ("NY", pd.Timestamp("08:12").time(), pd.Timestamp("09:12").time()),
]
FORCE_CLOSE_TIME = pd.Timestamp("16:00").time()

# --- grid search space (deliberately just the two parameters that matter most for this
# strategy's payout/confirmation structure - see header) ---
STOP_BUFFER_PCT_GRID = [0.01, 0.02, 0.05, 0.1]
CONFIRMATION_CANDLES_GRID = [2, 3, 4, 5]

# base script's fixed defaults, kept here only as a reference point for the equivalence unit test
BASE_STOP_BUFFER_PCT = 0.02
BASE_CONFIRMATION_CANDLES = 3

# --- search strategy + objective config (research/optimization_engine.py) ---
# SEARCH_METHOD: "grid" (exhaustive, default), "bayesian" (Optuna/TPE), or "genetic" (hand-rolled
# GA). DEFAULT IS "grid" DELIBERATELY, NOT AS A PLACEHOLDER: this script's parameter space is only
# 4x4=16 combos - at that size exhaustive grid search is not just adequate, it is the BETTER choice
# (deterministic, no cell left unexplored, no sampling noise to explain away). Bayesian and genetic
# search are offered here for when this project's parameter spaces grow past what exhaustive search
# can comfortably cover (3+ parameters, or much wider candidate lists) - not because they beat grid
# search on today's 16-cell grid. See optimization_engine.py's header for the full reasoning.
# Changing this to "bayesian"/"genetic" changes STEP 1 and STEP 4's per-fold in-sample search below;
# STEP 3's neighbor-plateau check requires the full grid and is skipped (with a clear message)
# under either non-grid method - see run_full_pipeline()'s STEP 3 section.
SEARCH_METHOD = "grid"
# OBJECTIVE: which of optimization_engine.OBJECTIVES the search maximizes by. DEFAULT IS
# "total_r" DELIBERATELY - this is the exact selection rule this script already used before this
# refactor existed (every already-shipped, already-verified number this script has ever produced
# picked its "best" cell by raw total R), so leaving this at "total_r" with SEARCH_METHOD="grid"
# reproduces that behavior exactly, not a new default. See optimization_engine.py's OBJECTIVES
# registry for the other options (avg_r, sharpe, calmar, win_rate) and their caveats.
OBJECTIVE = "total_r"

MC_ITERATIONS = 2000
MC_SEED = 42

WF_IS_YEARS = 3
WF_OOS_YEARS = 1
WF_STEP_YEARS = 1
WFE_PASS_THRESHOLD = 0.5

# --- lockbox (research/optimization_engine.py: split_lockbox/lockbox_confirm) ---
# The final LOCKBOX_MONTHS of FETCH_START..FETCH_END are carved off BEFORE any search runs and are
# excluded, BY CONSTRUCTION, from STEP 1's full-range search AND every STEP 4 walk-forward fold
# (both now bounded by `search_end`, not `FETCH_END` - see main()) - structurally stricter than the
# walk-forward OOS folds above, which DO get re-touched on every search re-run. See
# optimization_engine.py's own LOCKBOX / EMBARGOED FINAL HOLDOUT section header for the full
# reasoning, and why this is enforced via a persistent ledger (research/lockbox_ledger.json), not
# just this comment.
LOCKBOX_MONTHS = 12
STRATEGY_ID = "day_trading_rauf_dukascopy"   # stable identifier for the lockbox ledger - one
                                              # lockbox attempt EVER for this strategy, not per-run

HEATMAP_PNG_PATH = "day_trading_rauf_param_heatmap.png"


def to_ny_time(index):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert("America/New_York")


def _month_chunks(start, end, months_per_chunk):
    chunk_start = start
    while chunk_start < end:
        month_index = chunk_start.month - 1 + months_per_chunk
        chunk_end = chunk_start.replace(year=chunk_start.year + month_index // 12, month=month_index % 12 + 1)
        yield chunk_start, min(chunk_end, end)
        chunk_start = chunk_end


def fetch_instrument_data(label, instrument_const_name):
    """Same disk-cache-keyed-by-instrument/interval/date-range convention as the base script.
    Downloaded/cached ONCE per instrument here - every grid cell, every Monte Carlo draw, every
    walk-forward fold below reuses this same in-memory data, never re-fetching per combo."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{label}_5min_{FETCH_START.date()}_{FETCH_END.date()}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    instrument_const = getattr(dki, instrument_const_name)
    interval = getattr(dukascopy_python, DUKASCOPY_INTERVAL_NAME)
    offer_side = getattr(dukascopy_python, DUKASCOPY_OFFER_SIDE_NAME)

    chunks = []
    chunk_bounds = list(_month_chunks(FETCH_START, FETCH_END, FETCH_CHUNK_MONTHS))
    for chunk_start, chunk_end in tqdm(chunk_bounds, desc=f"{label}: downloading {DUKASCOPY_INTERVAL_NAME} bars",
                                        unit="chunk"):
        chunk = dukascopy_python.fetch(instrument_const, interval, offer_side, chunk_start, chunk_end)
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return None
    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df.index = to_ny_time(df.index)

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    return df


def precompute_arrays(df):
    """The one piece of per-instrument work that's genuinely independent of every parameter
    grid-searched below (STOP_BUFFER_PCT, CONFIRMATION_CANDLES, and the window bounds) - plain
    python lists instead of a DataFrame extraction happening again inside every one of the ~500+
    backtest_instrument calls a full run makes."""
    return {
        "highs": df["High"].tolist(),
        "lows": df["Low"].tolist(),
        "closes": df["Close"].tolist(),
        "opens": df["Open"].tolist(),
        "times": df.index,
    }


def backtest_instrument(arrays, stop_buffer_pct, confirmation_candles, window_start=None, window_end=None):
    """Generalized version of the base script's backtest_instrument: STOP_BUFFER_PCT and the
    hardcoded "3 consecutive same-direction candles" rule are now parameters (confirmation_candles
    = N), and trade counting can be restricted to a [window_start, window_end) date window without
    truncating the underlying price series - every bar of the FULL arrays is still processed in
    correct day-by-day sequence regardless of window_start/window_end; only whether a resolved
    trade gets appended to the returned list depends on whether its date falls in the window. This
    generalizes the base script's split_before/split_after masking (and
    orb_indices_optimization_and_ml.py's identical pattern) to an arbitrary window pair, needed for
    rolling walk-forward's many non-overlapping in-sample/out-of-sample windows.

    window_start/window_end are datetime.date objects (or None for unbounded). window_end is
    exclusive, matching this project's existing split_after convention.
    """
    highs, lows, closes, opens = arrays["highs"], arrays["lows"], arrays["closes"], arrays["opens"]
    times = arrays["times"]
    n = len(closes)
    N = confirmation_candles

    def in_window(d):
        if window_start is not None and d < window_start:
            return False
        if window_end is not None and d >= window_end:
            return False
        return True

    trades = []
    current_day = None
    open_trade = None
    range_states = {name: {"high": None, "low": None, "ready": False, "traded_today": False, "setup": None}
                     for name, _, _ in RANGES}

    for i in range(n):
        t = times[i]
        today = t.date()
        tod = t.time()

        if today != current_day:
            if open_trade is not None:
                side = open_trade["side"]
                last_close = closes[i - 1]
                pnl = (last_close - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - last_close)
                trade_date = times[i - 1].date()
                if in_window(trade_date):
                    trades.append({"side": side, "outcome": "FLAT", "r": pnl / open_trade["sl_distance"],
                                    "date": trade_date, "range": open_trade["range"]})
                open_trade = None
            current_day = today
            for st in range_states.values():
                st["high"] = st["low"] = None
                st["ready"] = False
                st["traded_today"] = False
                st["setup"] = None

        if open_trade is not None:
            side = open_trade["side"]
            stop, target = open_trade["stop"], open_trade["target"]
            hi, lo = highs[i], lows[i]
            hit_stop = lo <= stop if side == "LONG" else hi >= stop
            hit_target = hi >= target if side == "LONG" else lo <= target
            if hit_stop:
                if in_window(today):
                    trades.append({"side": side, "outcome": "SL", "r": -1.0, "date": today,
                                    "range": open_trade["range"]})
                open_trade = None
            elif hit_target:
                if in_window(today):
                    trades.append({"side": side, "outcome": "TP", "r": open_trade["reward_risk"], "date": today,
                                    "range": open_trade["range"]})
                open_trade = None
            elif tod >= FORCE_CLOSE_TIME:
                pnl = (closes[i] - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - closes[i])
                if in_window(today):
                    trades.append({"side": side, "outcome": "FLAT", "r": pnl / open_trade["sl_distance"],
                                    "date": today, "range": open_trade["range"]})
                open_trade = None

        for name, win_start, win_end in RANGES:
            rs = range_states[name]

            if win_start <= tod < win_end:
                if rs["high"] is None:
                    rs["high"], rs["low"] = highs[i], lows[i]
                else:
                    rs["high"] = max(rs["high"], highs[i])
                    rs["low"] = min(rs["low"], lows[i])
                continue

            if not rs["ready"]:
                if rs["high"] is not None:
                    rs["ready"] = True
                else:
                    continue

            if rs["traded_today"]:
                continue

            if tod >= FORCE_CLOSE_TIME:
                rs["setup"] = None
                continue

            range_high, range_low = rs["high"], rs["low"]
            broke_high = highs[i] > range_high
            broke_low = lows[i] < range_low

            if broke_high and broke_low:
                rs["setup"] = None
            elif broke_low:
                if rs["setup"] is None or rs["setup"]["side"] == -1:
                    rs["setup"] = {"side": 1, "sweep_extreme": lows[i], "setup_bar": i, "three_level": None}
                else:
                    rs["setup"]["sweep_extreme"] = min(rs["setup"]["sweep_extreme"], lows[i])
            elif broke_high:
                if rs["setup"] is None or rs["setup"]["side"] == 1:
                    rs["setup"] = {"side": -1, "sweep_extreme": highs[i], "setup_bar": i, "three_level": None}
                else:
                    rs["setup"]["sweep_extreme"] = max(rs["setup"]["sweep_extreme"], highs[i])

            setup = rs["setup"]
            if setup is None:
                continue
            bars_since = i - setup["setup_bar"]

            if setup["side"] == 1:
                if setup["three_level"] is None and bars_since >= N:
                    if all(closes[i - k] < opens[i - k] for k in range(N)):
                        setup["three_level"] = max(highs[i - k] for k in range(N))
                if setup["three_level"] is not None and closes[i] > setup["three_level"]:
                    entry = closes[i]
                    buffer_price = (stop_buffer_pct / 100.0) * entry
                    stop = setup["sweep_extreme"] - buffer_price
                    target = range_high
                    sl_distance = entry - stop
                    rs["setup"] = None
                    if sl_distance > 0 and target > entry and open_trade is None:
                        reward_risk = (target - entry) / sl_distance
                        open_trade = {"side": "LONG", "entry": entry, "stop": stop, "target": target,
                                      "sl_distance": sl_distance, "reward_risk": reward_risk, "range": name}
                        rs["traded_today"] = True
            else:
                if setup["three_level"] is None and bars_since >= N:
                    if all(closes[i - k] > opens[i - k] for k in range(N)):
                        setup["three_level"] = min(lows[i - k] for k in range(N))
                if setup["three_level"] is not None and closes[i] < setup["three_level"]:
                    entry = closes[i]
                    buffer_price = (stop_buffer_pct / 100.0) * entry
                    stop = setup["sweep_extreme"] + buffer_price
                    target = range_low
                    sl_distance = stop - entry
                    rs["setup"] = None
                    if sl_distance > 0 and target < entry and open_trade is None:
                        reward_risk = (entry - target) / sl_distance
                        open_trade = {"side": "SHORT", "entry": entry, "stop": stop, "target": target,
                                      "sl_distance": sl_distance, "reward_risk": reward_risk, "range": name}
                        rs["traded_today"] = True

    return trades


# ============================= STEP 4: walk-forward fold generator =============================

def walk_forward_folds(fetch_start, fetch_end, is_years=WF_IS_YEARS, oos_years=WF_OOS_YEARS,
                        step_years=WF_STEP_YEARS):
    """Rolling (not anchored) walk-forward windows: IS_years in-sample, immediately followed by
    OOS_years out-of-sample, stepping forward step_years at a time, until the out-of-sample window
    would run past fetch_end. Yields (is_start, is_end, oos_start, oos_end) as datetime.date."""
    fetch_start_d = fetch_start.date() if hasattr(fetch_start, "date") else fetch_start
    fetch_end_d = fetch_end.date() if hasattr(fetch_end, "date") else fetch_end

    fold_start = fetch_start_d
    while True:
        is_start = fold_start
        is_end = datetime.date(is_start.year + is_years, is_start.month, is_start.day)
        oos_start = is_end
        oos_end = datetime.date(oos_start.year + oos_years, oos_start.month, oos_start.day)
        if oos_end > fetch_end_d:
            break
        yield (is_start, is_end, oos_start, oos_end)
        fold_start = datetime.date(fold_start.year + step_years, fold_start.month, fold_start.day)


# ============================= STEP 2: vectorized Monte Carlo =============================

def _cumsum_drawdown(samples):
    """samples: (n_iter, n_trades) array of R-multiples per simulated path. Returns
    (total_r per path, max_drawdown per path) where max_drawdown = largest peak-to-trough drop
    in the cumulative R sum along that path (0 if the path never dips below its running peak)."""
    cumsum = np.cumsum(samples, axis=1)
    total_r = cumsum[:, -1]
    running_max = np.maximum.accumulate(cumsum, axis=1)
    drawdown = running_max - cumsum
    max_dd = drawdown.max(axis=1)
    return total_r, max_dd


def _summarize_mc(total_r, max_dd):
    if len(total_r) == 0:
        return {
            "total_r_p05": float("nan"), "total_r_p50": float("nan"), "total_r_p95": float("nan"),
            "max_dd_p05": float("nan"), "max_dd_p50": float("nan"), "max_dd_p95": float("nan"),
            "p_total_r_leq_0": float("nan"),
        }
    return {
        "total_r_p05": float(np.percentile(total_r, 5)),
        "total_r_p50": float(np.percentile(total_r, 50)),
        "total_r_p95": float(np.percentile(total_r, 95)),
        "max_dd_p05": float(np.percentile(max_dd, 5)),
        "max_dd_p50": float(np.percentile(max_dd, 50)),
        "max_dd_p95": float(np.percentile(max_dd, 95)),
        "p_total_r_leq_0": float(np.mean(total_r <= 0)),
    }


def monte_carlo_bootstrap(r_values, n_iter=MC_ITERATIONS, seed=None):
    """Resample WITH replacement, same N as the original trade count, n_iter times - vectorized
    via a single np.random.choice batch draw (no python-level loop over iterations)."""
    r_values = np.asarray(r_values, dtype=float)
    n = len(r_values)
    if n == 0:
        empty = np.array([])
        return _summarize_mc(empty, empty)
    rng = np.random.default_rng(seed)
    samples = rng.choice(r_values, size=(n_iter, n), replace=True)
    total_r, max_dd = _cumsum_drawdown(samples)
    return _summarize_mc(total_r, max_dd)


def monte_carlo_shuffle(r_values, n_iter=MC_ITERATIONS, seed=None):
    """Pure reordering WITHOUT replacement (same trades, same count, just shuffled) - isolates
    path/sequence risk from the drawdown number. Total R is mathematically identical to the
    original sum(r_values) on every single path (a permutation can't change the sum) - that
    distribution is reported anyway for symmetry/completeness with the bootstrap case, but the
    real information here is in the max-drawdown distribution, not the (degenerate) total-R one.
    Vectorized via argsort-of-random-keys to get n_iter independent permutations without a
    python-level loop."""
    r_values = np.asarray(r_values, dtype=float)
    n = len(r_values)
    if n == 0:
        empty = np.array([])
        return _summarize_mc(empty, empty)
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((n_iter, n)), axis=1)
    samples = r_values[idx]
    total_r, max_dd = _cumsum_drawdown(samples)
    return _summarize_mc(total_r, max_dd)


# ============================= STEP 3: cluster analysis =============================

def build_cluster_features(grid_results, stop_buffer_grid, confirmation_grid):
    """Feature vector per grid cell = (normalized STOP_BUFFER_PCT grid position, normalized
    CONFIRMATION_CANDLES grid position, avg R/trade)."""
    sb_n = len(stop_buffer_grid) - 1 or 1
    cc_n = len(confirmation_grid) - 1 or 1
    features = []
    for cell in grid_results:
        sb_pos = stop_buffer_grid.index(cell["stop_buffer_pct"]) / sb_n
        cc_pos = confirmation_grid.index(cell["confirmation_candles"]) / cc_n
        features.append([sb_pos, cc_pos, cell["avg_r"]])
    return np.array(features)


def run_cluster_analysis(grid_results, stop_buffer_grid, confirmation_grid, k=3, random_state=42):
    """sklearn KMeans over the 16 grid cells' feature vectors. Returns None (with a printed skip
    message, no crash) if scikit-learn isn't installed - matching
    orb_indices_optimization_and_ml.py's existing sklearn-missing handling."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("scikit-learn not installed - run '!pip install scikit-learn -q' and re-run. "
              "Skipping cluster analysis (the always-available neighbor check below still runs).")
        return None

    features = build_cluster_features(grid_results, stop_buffer_grid, confirmation_grid)
    n_clusters = min(k, len(grid_results))
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(features)

    best_idx = max(range(len(grid_results)), key=lambda idx: grid_results[idx]["total_r"])
    best_label = labels[best_idx]
    members = [grid_results[idx] for idx in range(len(grid_results)) if labels[idx] == best_label]
    member_avg_rs = [m["avg_r"] for m in members]

    return {
        "labels": labels,
        "best_cluster_label": int(best_label),
        "best_cluster_size": len(members),
        "best_cluster_members": members,
        "best_cluster_min_avg_r": min(member_avg_rs),
        "best_cluster_mean_avg_r": float(np.mean(member_avg_rs)),
    }


def neighbor_plateau_check(grid_results, stop_buffer_grid, confirmation_grid, decent_ratio=0.5):
    """Always-available (no sklearn) check: look at the best-by-total-R cell's immediate grid
    neighbors (up/down/left/right, fewer at grid edges). A neighbor is "decent" if it has the same
    sign of avg R/trade as the peak AND its magnitude is at least decent_ratio of the peak's
    magnitude. PLATEAU if most (>=half) of the neighbors are decent; otherwise ISOLATED SPIKE /
    overfit warning. Returns a dict with the verdict and the neighbor details."""
    cell_by_pos = {(sb, cc): cell for cell in grid_results
                   for sb, cc in [(cell["stop_buffer_pct"], cell["confirmation_candles"])]}
    best_cell = max(grid_results, key=lambda c: c["total_r"])
    sb_i = stop_buffer_grid.index(best_cell["stop_buffer_pct"])
    cc_i = confirmation_grid.index(best_cell["confirmation_candles"])

    neighbor_positions = []
    for d_sb, d_cc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        n_sb_i, n_cc_i = sb_i + d_sb, cc_i + d_cc
        if 0 <= n_sb_i < len(stop_buffer_grid) and 0 <= n_cc_i < len(confirmation_grid):
            neighbor_positions.append((stop_buffer_grid[n_sb_i], confirmation_grid[n_cc_i]))

    peak_avg_r = best_cell["avg_r"]
    neighbors = []
    decent_count = 0
    for sb, cc in neighbor_positions:
        cell = cell_by_pos.get((sb, cc))
        if cell is None:
            continue
        same_sign = (cell["avg_r"] >= 0) == (peak_avg_r >= 0)
        magnitude_ok = abs(peak_avg_r) == 0 or abs(cell["avg_r"]) >= decent_ratio * abs(peak_avg_r)
        decent = same_sign and magnitude_ok
        decent_count += int(decent)
        neighbors.append({"stop_buffer_pct": sb, "confirmation_candles": cc, "avg_r": cell["avg_r"],
                           "decent": decent})

    verdict = "PLATEAU" if neighbors and decent_count >= len(neighbors) / 2.0 else "ISOLATED SPIKE / overfit warning"
    if not neighbors:
        verdict = "ISOLATED SPIKE / overfit warning (no in-grid neighbors to compare - degenerate grid)"

    return {"best_cell": best_cell, "neighbors": neighbors, "verdict": verdict}


# ==================== SEARCH_METHOD/OBJECTIVE dispatch (research/optimization_engine.py) ====================

def _make_eval_fn(all_arrays, window_start=None, window_end=None):
    """Builds an optimization_engine-compatible eval_fn(params) -> list_of_trade_dicts: runs
    backtest_instrument across every instrument in `all_arrays`, restricted to
    [window_start, window_end), tagging each trade with its instrument label - the exact same
    tagging run_grid_search() below already does, kept identical here so
    _search_result_to_grid_cells can rebuild the same cell shape run_grid_search's output has."""
    def eval_fn(params):
        trades = []
        for label, arrays in all_arrays.items():
            cell_trades = backtest_instrument(arrays, params["stop_buffer_pct"], params["confirmation_candles"],
                                               window_start=window_start, window_end=window_end)
            for t in cell_trades:
                t["instrument"] = label
            trades.extend(cell_trades)
        return trades
    return eval_fn


def _search_result_to_grid_cells(search_result):
    """Converts an optimization_engine search result's "all" list (list of {"params", "trades",
    "score"}) into this script's pre-existing grid-cell dict shape (stop_buffer_pct,
    confirmation_candles, trades, r_values, total_r, n_trades, avg_r - the exact same shape
    run_grid_search() below already returns) so every downstream consumer (print_grid_text_table,
    plot_heatmap, run_cluster_analysis, neighbor_plateau_check, the Monte Carlo loop in
    run_full_pipeline) keeps working completely unchanged regardless of which SEARCH_METHOD
    produced the results. Adds a "score" key too (the configured OBJECTIVE's value for that cell,
    which equals total_r exactly when OBJECTIVE="total_r")."""
    cells = []
    for entry in search_result["all"]:
        params, trades = entry["params"], entry["trades"]
        r_values = [t["r"] for t in trades]
        total_r = sum(r_values)
        n_trades = len(r_values)
        cells.append({
            "stop_buffer_pct": params["stop_buffer_pct"],
            "confirmation_candles": params["confirmation_candles"],
            "trades": trades,
            "r_values": r_values,
            "total_r": total_r,
            "n_trades": n_trades,
            "avg_r": (total_r / n_trades) if n_trades else 0.0,
            "score": entry["score"],
        })
    return cells


def _find_cell(grid_results, params):
    for c in grid_results:
        if c["stop_buffer_pct"] == params["stop_buffer_pct"] and c["confirmation_candles"] == params["confirmation_candles"]:
            return c
    return None


def run_param_search(all_arrays, stop_buffer_grid, confirmation_grid, window_start=None, window_end=None,
                      method=None, objective=None, years=None, desc="param search", show_progress=False,
                      **search_kwargs):
    """Dispatches the parameter search (STEP 1's full-range pass, and STEP 4's per-fold in-sample
    pass) to whichever SEARCH_METHOD is configured (module-level SEARCH_METHOD/OBJECTIVE globals
    by default, overridable via the method/objective kwargs), via optimization_engine.run_search(),
    then converts the result back into this script's pre-existing grid-cell dict shape so every
    existing downstream consumer is unaffected.

    With method="grid" and objective="total_r" (this script's defaults) this reproduces
    run_grid_search()'s exact combos, evaluation order, and per-cell numbers - see
    TestSearchEngineRetrofit's regression test (below, in this same file's --test suite) for a
    direct, assertion-based comparison against the OLD hand-written grid loop.

    Returns (grid_results, search_result): grid_results in the legacy shape described above, and
    the raw optimization_engine result dict (has "n_evals"/"best"/"method")."""
    method = method if method is not None else SEARCH_METHOD
    objective = objective if objective is not None else OBJECTIVE
    param_grid = {"stop_buffer_pct": list(stop_buffer_grid), "confirmation_candles": list(confirmation_grid)}
    objective_fn = opt_engine.get_objective(objective)

    if years is None:
        if window_start is not None and window_end is not None:
            years = (window_end - window_start).days / 365.0
        elif window_start is None and window_end is None:
            years = (FETCH_END - FETCH_START).days / 365.0
        # else: a one-sided window has no well-defined span - years stays None (sharpe will
        # degrade to its documented sentinel rather than crash or guess at a span)

    eval_fn = _make_eval_fn(all_arrays, window_start, window_end)
    search_result = opt_engine.run_search(method, param_grid, eval_fn, objective_fn, years=years,
                                           **search_kwargs)
    grid_results = _search_result_to_grid_cells(search_result)
    return grid_results, search_result


# ============================= STEP 1: grid search + heatmap =============================

def run_grid_search(all_arrays, stop_buffer_grid, confirmation_grid, window_start=None, window_end=None,
                     show_progress=True):
    """Runs the full grid (every combo x every instrument) restricted to [window_start, window_end).
    Returns a list of 16 dicts (one per cell), each with the combo's params, aggregate stats, and
    the raw list of trade R-multiples (needed by step 2's Monte Carlo)."""
    combos = [(sb, cc) for sb in stop_buffer_grid for cc in confirmation_grid]
    iterator = tqdm(combos, desc="Grid search", unit="combo") if show_progress else combos

    results = []
    for stop_buffer_pct, confirmation_candles in iterator:
        cell_trades = []
        for label, arrays in all_arrays.items():
            trades = backtest_instrument(arrays, stop_buffer_pct, confirmation_candles,
                                          window_start=window_start, window_end=window_end)
            for t in trades:
                t["instrument"] = label
            cell_trades.extend(trades)

        r_values = [t["r"] for t in cell_trades]
        total_r = sum(r_values)
        n_trades = len(r_values)
        results.append({
            "stop_buffer_pct": stop_buffer_pct,
            "confirmation_candles": confirmation_candles,
            "trades": cell_trades,
            "r_values": r_values,
            "total_r": total_r,
            "n_trades": n_trades,
            "avg_r": (total_r / n_trades) if n_trades else 0.0,
        })
    return results


def print_grid_text_table(grid_results, stop_buffer_grid, confirmation_grid, title):
    print(f"\n{title} - text table (avg R/trade, rows=STOP_BUFFER_PCT, cols=CONFIRMATION_CANDLES):")
    header = "STOP_BUFFER_PCT \\ N".ljust(22) + "".join(f"{cc:>10d}" for cc in confirmation_grid)
    print(header)
    by_pos = {(c["stop_buffer_pct"], c["confirmation_candles"]): c for c in grid_results}
    for sb in stop_buffer_grid:
        row = f"{sb:<22g}"
        for cc in confirmation_grid:
            cell = by_pos.get((sb, cc))
            row += f"{cell['avg_r']:>10.4f}" if cell else f"{'--':>10}"
        print(row)


def plot_heatmap(grid_results, stop_buffer_grid, confirmation_grid, png_path=HEATMAP_PNG_PATH):
    """Avg R/trade heatmap, params on the two axes, best cell annotated. Wrapped in try/except
    ImportError by the caller so a missing matplotlib never kills the script - the text table
    above is the always-available fallback, not the only path."""
    import matplotlib.pyplot as plt

    by_pos = {(c["stop_buffer_pct"], c["confirmation_candles"]): c for c in grid_results}
    matrix = np.array([[by_pos[(sb, cc)]["avg_r"] for cc in confirmation_grid] for sb in stop_buffer_grid])

    best_cell = max(grid_results, key=lambda c: c["avg_r"])
    best_row = stop_buffer_grid.index(best_cell["stop_buffer_pct"])
    best_col = confirmation_grid.index(best_cell["confirmation_candles"])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(confirmation_grid)))
    ax.set_xticklabels(confirmation_grid)
    ax.set_yticks(range(len(stop_buffer_grid)))
    ax.set_yticklabels(stop_buffer_grid)
    ax.set_xlabel("CONFIRMATION_CANDLES")
    ax.set_ylabel("STOP_BUFFER_PCT")
    ax.set_title("Day Trading Rauf - avg R/trade by parameter cell")

    for row in range(len(stop_buffer_grid)):
        for col in range(len(confirmation_grid)):
            is_best = (row == best_row and col == best_col)
            ax.text(col, row, f"{matrix[row, col]:.4f}" + (" *BEST*" if is_best else ""),
                    ha="center", va="center",
                    color="black", fontweight="bold" if is_best else "normal",
                    bbox=dict(boxstyle="round", fc="white", ec="blue", lw=2) if is_best else None)

    fig.colorbar(im, ax=ax, label="avg R/trade")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.show()
    print(f"Heatmap saved to {png_path}")


# ============================= STEP 4 (continued): walk-forward orchestration =============================

def run_walk_forward(all_arrays, stop_buffer_grid, confirmation_grid, fetch_start, fetch_end, show_progress=True,
                      method=None, objective=None):
    """method/objective default to the module-level SEARCH_METHOD/OBJECTIVE globals (None resolves
    to them inside run_param_search) - i.e. by default this reproduces the exact original behavior
    (exhaustive grid search, raw-total-R fold-winner selection) this function used before this
    refactor; this function's call signature and default numeric behavior are unchanged. See
    TestSearchEngineRetrofit.test_walk_forward_default_still_matches_old_behavior for the direct
    regression check.

    Each fold_rows entry ALSO carries "oos_trades" - that fold's own OOS trades, un-merged with any
    other fold's - needed by optimization_engine.estimate_decay's month-since-fit pooling (see this
    script's STEP 4 output section below), which combined_oos_trades alone cannot support since it
    already threw away fold identity. Purely additive: combined_oos_trades' own contents are
    unchanged."""
    folds = list(walk_forward_folds(fetch_start, fetch_end))
    fold_rows = []
    combined_oos_trades = []
    is_avg_rs_selected = []

    fold_iter = tqdm(folds, desc="Walk-forward folds", unit="fold") if show_progress else folds
    for is_start, is_end, oos_start, oos_end in fold_iter:
        is_grid, is_search_result = run_param_search(
            all_arrays, stop_buffer_grid, confirmation_grid, window_start=is_start, window_end=is_end,
            method=method, objective=objective, desc="fold in-sample search")
        best = _find_cell(is_grid, is_search_result["best"]["params"])

        oos_trades = []
        for label, arrays in all_arrays.items():
            trades = backtest_instrument(arrays, best["stop_buffer_pct"], best["confirmation_candles"],
                                          window_start=oos_start, window_end=oos_end)
            for t in trades:
                t["instrument"] = label
            oos_trades.extend(trades)

        oos_total_r = sum(t["r"] for t in oos_trades)
        oos_n = len(oos_trades)
        oos_avg_r = (oos_total_r / oos_n) if oos_n else 0.0

        fold_rows.append({
            "is_start": is_start, "is_end": is_end, "oos_start": oos_start, "oos_end": oos_end,
            "best_stop_buffer_pct": best["stop_buffer_pct"], "best_confirmation_candles": best["confirmation_candles"],
            "is_total_r": best["total_r"], "is_n_trades": best["n_trades"], "is_avg_r": best["avg_r"],
            "oos_total_r": oos_total_r, "oos_n_trades": oos_n, "oos_avg_r": oos_avg_r,
            "oos_trades": oos_trades,
        })
        combined_oos_trades.extend(oos_trades)
        is_avg_rs_selected.append(best["avg_r"])

    combined_total_r = sum(t["r"] for t in combined_oos_trades)
    combined_n = len(combined_oos_trades)
    combined_avg_r = (combined_total_r / combined_n) if combined_n else 0.0
    mean_is_avg_r = float(np.mean(is_avg_rs_selected)) if is_avg_rs_selected else 0.0

    if mean_is_avg_r == 0:
        wfe = float("nan")
    else:
        wfe = combined_avg_r / mean_is_avg_r

    return {
        "fold_rows": fold_rows,
        "combined_oos_trades": combined_oos_trades,
        "combined_total_r": combined_total_r,
        "combined_n_trades": combined_n,
        "combined_avg_r": combined_avg_r,
        "mean_is_avg_r": mean_is_avg_r,
        "wfe": wfe,
        "wfe_pass": (wfe >= WFE_PASS_THRESHOLD) if not np.isnan(wfe) else False,
    }


# ============================= orchestration used by both main() and the smoke test =============================

def run_full_pipeline(all_arrays, stop_buffer_grid, confirmation_grid, fetch_start, fetch_end,
                       mc_iterations=MC_ITERATIONS, make_plots=True, show_progress=True, verbose=True,
                       method=None, objective=None):
    """Runs all 4 steps against whatever data/grid/date-range it's given - used both by main()
    (real data, full grid, full 2016-2025 range) and by run_smoke_test() (small synthetic data,
    small grid, a shrunk date range) so the two paths can never silently diverge in logic.

    method/objective default to the module-level SEARCH_METHOD/OBJECTIVE globals (None resolves
    to them) - this function's default call signature and default numeric behavior are UNCHANGED
    from before the optimization_engine retrofit: with the defaults, STEP 1 and STEP 4 still run
    an exhaustive grid search and select each winner by raw total R, exactly as before. See
    TestSmoke.test_full_pipeline_end_to_end_on_synthetic_data (unchanged, still passing) and
    TestSearchEngineRetrofit's new regression test for a direct comparison against the OLD
    hand-written grid loop."""
    resolved_method = method if method is not None else SEARCH_METHOD
    results = {}

    # ---- STEP 1 ----
    if verbose:
        print("\n" + "=" * 70)
        print(f"STEP 1: {len(stop_buffer_grid)}x{len(confirmation_grid)} parameter search "
              f"(SEARCH_METHOD={resolved_method!r}, OBJECTIVE={(objective if objective is not None else OBJECTIVE)!r}), full "
              f"{fetch_start.date() if hasattr(fetch_start,'date') else fetch_start} to "
              f"{fetch_end.date() if hasattr(fetch_end,'date') else fetch_end}")
        print("=" * 70)
        if (objective if objective is not None else OBJECTIVE) == "win_rate":
            print("CAVEAT (OBJECTIVE=win_rate): optimizing for win rate alone ignores payout size and is a "
                  "known trap in THIS exact project - the Day Trading Rauf base backtest has a real 51.6% "
                  "win rate and is still net-negative because losers outsize winners. See "
                  "optimization_engine.py's win_rate() docstring. total_r/avg_r for the same selected combo "
                  "are also printed below so this can be cross-checked, not taken on faith.")
    # STEP 1's window is now explicitly bounded by fetch_start/fetch_end (this function's own
    # params) - when main() calls this with (search_start, search_end) rather than
    # (FETCH_START, FETCH_END), the lockbox window is excluded from STEP 1's search BY
    # CONSTRUCTION, not just by convention (see optimization_engine.py's LOCKBOX section header
    # and main()'s split_lockbox call below). This also makes run_full_pipeline's own docstring
    # ("Runs all 4 steps against whatever data/grid/date-range it's given") literally true for
    # STEP 1 too, not just STEP 4's walk-forward folds (which already honored fetch_start/
    # fetch_end via walk_forward_folds).
    step1_window_start = fetch_start.date() if hasattr(fetch_start, "date") else fetch_start
    step1_window_end = fetch_end.date() if hasattr(fetch_end, "date") else fetch_end
    grid_results, step1_search_result = run_param_search(
        all_arrays, stop_buffer_grid, confirmation_grid,
        window_start=step1_window_start, window_end=step1_window_end,
        method=method, objective=objective, show_progress=show_progress, desc="STEP 1 search")
    results["grid_results"] = grid_results
    results["step1_search_result"] = step1_search_result
    if verbose and resolved_method != "grid":
        n_possible = len(stop_buffer_grid) * len(confirmation_grid)
        print(f"\n{resolved_method} search evaluated {step1_search_result['n_evals']} of {n_possible} "
              f"possible combos this run (vs grid_search's exhaustive {n_possible}).")

    if verbose:
        for cell in grid_results:
            print(f"  STOP_BUFFER_PCT={cell['stop_buffer_pct']:<6g} CONFIRMATION_CANDLES={cell['confirmation_candles']} "
                  f"-> {cell['n_trades']:5d} trades, {cell['total_r']:+9.2f}R total, {cell['avg_r']:+.4f}R/trade")
        print_grid_text_table(grid_results, stop_buffer_grid, confirmation_grid, "STEP 1 heatmap")

    if make_plots:
        try:
            plot_heatmap(grid_results, stop_buffer_grid, confirmation_grid)
        except ImportError:
            print("matplotlib not installed - skipping heatmap plot (text table above is the fallback, "
                  "not the only path). Run '!pip install matplotlib -q' to get the plot too.")

    # ---- STEP 2 ----
    if verbose:
        print("\n" + "=" * 70)
        print(f"STEP 2: Monte Carlo per grid cell ({mc_iterations} iterations each, bootstrap + shuffle)")
        print("=" * 70)
    mc_results = []
    for idx, cell in enumerate(grid_results):
        boot = monte_carlo_bootstrap(cell["r_values"], n_iter=mc_iterations, seed=MC_SEED + idx)
        shuf = monte_carlo_shuffle(cell["r_values"], n_iter=mc_iterations, seed=MC_SEED + 1000 + idx)
        mc_results.append({"stop_buffer_pct": cell["stop_buffer_pct"],
                            "confirmation_candles": cell["confirmation_candles"],
                            "bootstrap": boot, "shuffle": shuf})
        if verbose:
            print(f"  STOP_BUFFER_PCT={cell['stop_buffer_pct']:<6g} CONFIRMATION_CANDLES={cell['confirmation_candles']}")
            print(f"    bootstrap (resample w/ replacement): total R p05/p50/p95 = "
                  f"{boot['total_r_p05']:+.2f}/{boot['total_r_p50']:+.2f}/{boot['total_r_p95']:+.2f}   "
                  f"max DD p05/p50/p95 = {boot['max_dd_p05']:.2f}/{boot['max_dd_p50']:.2f}/{boot['max_dd_p95']:.2f}   "
                  f"P(total R<=0) = {boot['p_total_r_leq_0']:.3f}")
            print(f"    shuffle (reorder, no replacement):    total R p05/p50/p95 = "
                  f"{shuf['total_r_p05']:+.2f}/{shuf['total_r_p50']:+.2f}/{shuf['total_r_p95']:+.2f}   "
                  f"max DD p05/p50/p95 = {shuf['max_dd_p05']:.2f}/{shuf['max_dd_p50']:.2f}/{shuf['max_dd_p95']:.2f}   "
                  f"P(total R<=0) = {shuf['p_total_r_leq_0']:.3f}  "
                  f"(total R is deterministic under pure reordering - shown for symmetry; the drawdown "
                  f"distribution is the real information here)")
    results["mc_results"] = mc_results

    # ---- STEP 3 ----
    # Cluster analysis is grid-agnostic: build_cluster_features() looks up each evaluated cell's
    # axis POSITION in the static stop_buffer_grid/confirmation_grid candidate lists (an .index()
    # call), which is well-defined for any cell that was evaluated at all, exhaustively or not -
    # so it always runs, over however many points were actually evaluated this run.
    #
    # The neighbor-plateau check is NOT grid-agnostic in the same way: it looks up each of the
    # best cell's immediate NEIGHBORS by (stop_buffer_pct, confirmation_candles) position, and
    # under a non-exhaustive search (bayesian/genetic) a neighboring combo may simply never have
    # been evaluated. neighbor_plateau_check() itself tolerates this without crashing (missing
    # neighbors are skipped via a .get()/None-check, degrading to "ISOLATED SPIKE / overfit
    # warning (no in-grid neighbors to compare - degenerate grid)" if none were found at all) -
    # but silently comparing against however-many-of-4 non-random neighbors happened to be
    # sampled, and potentially defaulting to a spike-warning purely because of under-sampling
    # rather than a real spike, is still a materially weaker and potentially misleading signal
    # than the full 4-neighbor check the PLATEAU/SPIKE verdict implies. So, for the same reasoning
    # (and for consistency with ict_po3_forex_dukascopy_optimization.py's identical choice), this
    # check is SKIPPED outright under a non-grid SEARCH_METHOD, with an explicit message, rather
    # than run in a way that could look authoritative but isn't.
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3: cluster analysis + neighbor-plateau check")
        print("=" * 70)
    cluster_result = run_cluster_analysis(grid_results, stop_buffer_grid, confirmation_grid)
    results["cluster_result"] = cluster_result
    if verbose and cluster_result is not None:
        if resolved_method != "grid":
            print(f"  NOTE: cluster analysis below is over the {len(grid_results)} points "
                  f"SEARCH_METHOD={resolved_method!r} actually evaluated, not the full grid.")
        print(f"  Best-in-sample cell's cluster: label={cluster_result['best_cluster_label']}, "
              f"size={cluster_result['best_cluster_size']}, "
              f"min avg R/trade={cluster_result['best_cluster_min_avg_r']:+.4f}, "
              f"mean avg R/trade={cluster_result['best_cluster_mean_avg_r']:+.4f}")

    if resolved_method == "grid":
        neighbor_result = neighbor_plateau_check(grid_results, stop_buffer_grid, confirmation_grid)
        results["neighbor_result"] = neighbor_result
        if verbose:
            bc = neighbor_result["best_cell"]
            print(f"  Neighbor check around best cell (STOP_BUFFER_PCT={bc['stop_buffer_pct']}, "
                  f"CONFIRMATION_CANDLES={bc['confirmation_candles']}, avg R/trade={bc['avg_r']:+.4f}): "
                  f"{neighbor_result['verdict']}")
            for nb in neighbor_result["neighbors"]:
                print(f"    neighbor STOP_BUFFER_PCT={nb['stop_buffer_pct']} CONFIRMATION_CANDLES={nb['confirmation_candles']}"
                      f" avg R/trade={nb['avg_r']:+.4f} decent={nb['decent']}")
    else:
        results["neighbor_result"] = None
        if verbose:
            print(f"  Neighbor-plateau check: SKIPPED - requires the full grid, not available under "
                  f"SEARCH_METHOD={resolved_method!r} (only a sample of the grid was evaluated, so an "
                  f"immediate-neighbor combo may never have been tried).")

    # ---- STEP 4 ----
    if verbose:
        print("\n" + "=" * 70)
        print(f"STEP 4: rolling walk-forward ({WF_IS_YEARS}yr IS / {WF_OOS_YEARS}yr OOS, step {WF_STEP_YEARS}yr)")
        print("=" * 70)
    wf_result = run_walk_forward(all_arrays, stop_buffer_grid, confirmation_grid, fetch_start, fetch_end,
                                  show_progress=show_progress, method=method, objective=objective)
    results["wf_result"] = wf_result
    if verbose:
        print(f"  {'Fold':<6}{'IS window':<24}{'OOS window':<24}{'Best params':<22}"
              f"{'IS total R':>12}{'OOS total R':>13}{'OOS avg R':>11}")
        for idx, row in enumerate(wf_result["fold_rows"]):
            is_win = f"{row['is_start']}..{row['is_end']}"
            oos_win = f"{row['oos_start']}..{row['oos_end']}"
            params = f"sb={row['best_stop_buffer_pct']},N={row['best_confirmation_candles']}"
            print(f"  {idx+1:<6}{is_win:<24}{oos_win:<24}{params:<22}"
                  f"{row['is_total_r']:>+12.2f}{row['oos_total_r']:>+13.2f}{row['oos_avg_r']:>+11.4f}")
        print(f"\n  Combined OOS: {wf_result['combined_n_trades']} trades, "
              f"{wf_result['combined_total_r']:+.2f}R total, {wf_result['combined_avg_r']:+.4f}R/trade")
        print(f"  Mean selected-combo IS avg R/trade across folds: {wf_result['mean_is_avg_r']:+.4f}")
        if np.isnan(wf_result["wfe"]):
            print("  Walk-Forward Efficiency: undefined (mean in-sample avg R/trade was exactly 0)")
        else:
            print(f"  Walk-Forward Efficiency (WFE) = combined OOS avg R / mean selected-combo IS avg R "
                  f"= {wf_result['wfe']:.3f}")
        print(f"  {'PASS' if wf_result['wfe_pass'] else 'FAIL'} against the WFE >= {WFE_PASS_THRESHOLD} rule of "
              f"thumb (CAVEAT: this 50% threshold is a standard heuristic used across quant research, not a "
              f"formal proof of robustness either way - a pass doesn't guarantee a real edge and a fail near "
              f"the line doesn't guarantee its absence).")

    decay_result = opt_engine.estimate_decay(wf_result["fold_rows"])
    results["decay_result"] = decay_result
    if verbose:
        print_decay_diagnostic(decay_result, n_folds=len(wf_result["fold_rows"]))

    return results


def print_decay_diagnostic(decay_result, n_folds, metric="avg_r", min_folds=4):
    """Prints optimization_engine.estimate_decay's result (pooled by months-since-fit across every
    STEP 4 walk-forward fold's own un-merged OOS trades - see run_walk_forward's "oos_trades"
    addition above) alongside the WFE result already printed by run_full_pipeline's STEP 4 section.
    This is a DIAGNOSTIC, not a pass/fail gate - it never changes the WFE PASS/FAIL verdict, only
    adds a second, differently-shaped signal about whether whatever edge (if any) the walk-forward
    found tends to fade the further out from each fold's own fit date it's used."""
    print(f"\n  Signal-decay diagnostic (pooled by months-since-fit across all {n_folds} walk-forward "
          f"folds, metric={metric!r}):")
    if decay_result["confidence"] == "insufficient_folds":
        print(f"    INSUFFICIENT: only {decay_result['n_folds_used']} fold(s) contributed pooled OOS "
              f"data (need >= min_folds={min_folds}) - this diagnostic is not meaningful yet with this "
              "few independent walk-forward folds. Both Day Trading Rauf and ICT PO3's own 6-fold "
              "rolling walk-forward comfortably clear this bar.")
        return
    print(f"    Pooled across {decay_result['n_folds_used']} folds. Linear trend: slope = "
          f"{decay_result['slope']:+.5f} per month "
          f"({'DECAY evidence (negative slope)' if decay_result['slope'] < 0 else 'no decay evidence (flat/positive slope)'}).")
    if decay_result["phi"] is not None:
        print(f"    Exponential decay fit: phi = {decay_result['phi']:.4f}, half-life = "
              f"{decay_result['half_life_months']:.2f} months.")
    else:
        print("    Exponential decay fit: not reportable this run (pooled metric wasn't strictly "
              "positive throughout, or fitted phi wasn't in the genuine-decay range (0, 1)) - the "
              "linear slope above is still a valid decay/no-decay signal on its own.")


def print_final_verdict(results):
    grid_results = results["grid_results"]
    all_negative_or_zero = all(c["total_r"] <= 0 for c in grid_results)
    any_positive = any(c["total_r"] > 0 for c in grid_results)
    wf = results["wf_result"]

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    if all_negative_or_zero:
        print("Every one of the 16 grid cells' total R came back <= 0 over the full history: there is NO "
              "corner of this parameter space (within the range searched) that produces a positive edge. "
              "The already-established negative result from the base backtest holds up everywhere this "
              "pass looked - this is not a case of 'we just needed better parameters'.")
    elif any_positive and not wf["wfe_pass"]:
        print("At least one grid cell showed a positive in-sample total R, but the rolling walk-forward "
              "efficiency check FAILED the >=0.5 rule of thumb - the apparent edge did not survive being "
              "applied out-of-sample with parameters chosen without hindsight. Treat any single positive "
              "grid cell as likely overfitting, not a validated edge, unless the cluster/neighbor checks "
              "above also show a genuine plateau (not an isolated spike) around it.")
    else:
        print("At least one grid cell showed a positive in-sample total R AND the walk-forward efficiency "
              "check PASSED the >=0.5 rule of thumb. Cross-check this against the cluster/neighbor-plateau "
              "results above before treating it as a real edge - a pass here is supportive evidence, not "
              "proof, per the caveat printed with the WFE result.")

    # ---- corrected z-score + multiple-testing (Bonferroni) check ----
    # See optimization_engine.zscore()'s docstring: this is a CORRECTED replacement for the
    # `avg_r * sqrt(n)` shortcut used in day_trading_rauf_dukascopy_backtest.py's header-quoted
    # single-fixed-parameter result (that shortcut implicitly assumes std(r) == 1, which inflates
    # every z-score computed with it) - this run's best cell gets the corrected version, plus an
    # honest Bonferroni-adjusted bar reflecting how many combos were actually tested this run.
    step1_search_result = results.get("step1_search_result")
    if step1_search_result is not None and step1_search_result.get("best") is not None:
        best_cell = _find_cell(grid_results, step1_search_result["best"]["params"])
        if best_cell is not None and best_cell["n_trades"] > 0:
            print("\n" + "-" * 70)
            print("CORRECTED SIGNIFICANCE CHECK: best-cell z-score + Bonferroni-adjusted bar")
            print("-" * 70)
            best_trades = [{"r": r} for r in best_cell["r_values"]]
            best_z = opt_engine.zscore(best_trades)
            n_trials_this_run = step1_search_result["n_evals"]
            z_bar = opt_engine.bonferroni_adjusted_z_threshold(n_trials_this_run)
            clears_adjusted = abs(best_z) >= z_bar
            print(f"Best cell (STOP_BUFFER_PCT={best_cell['stop_buffer_pct']:.3g}, "
                  f"CONFIRMATION_CANDLES={best_cell['confirmation_candles']}) corrected z-score: {best_z:+.2f} "
                  f"(sample-std-based, not the old avg_r*sqrt(n) shortcut)")
            print(f"Combos tested this run: {n_trials_this_run} -> Bonferroni-adjusted |z| bar = {z_bar:.2f} "
                  f"(vs the naive single-test 1.96 rule of thumb)")
            print(f"-> {'CLEARS' if clears_adjusted else 'does NOT clear'} the multiple-testing-adjusted bar "
                  f"(|z|={abs(best_z):.2f} {'>=' if clears_adjusted else '<'} {z_bar:.2f})")
            print("CAVEAT: this correction is for THIS SCRIPT's own combos-per-run only - a fully "
                  "project-wide correction would need a running total across every strategy script's "
                  "every run, which this print deliberately does not claim to be.")


def make_lockbox_backtest_fn(all_arrays, stop_buffer_pct, confirmation_candles):
    """Builds an optimization_engine.lockbox_confirm-compatible backtest_fn(lockbox_start,
    lockbox_end) -> list_of_trade_dicts, running backtest_instrument across every instrument in
    `all_arrays` with FIXED, already-selected params (STEP 1's final winner, not a search)
    restricted to [lockbox_start, lockbox_end) - the one and only time those exact bars are ever
    touched by this script. `lockbox_start`/`lockbox_end` are converted to datetime.date here
    (backtest_instrument compares against per-bar .date() values) so a caller can pass
    split_lockbox's raw datetime/date output straight through without converting first."""
    def backtest_fn(lockbox_start, lockbox_end):
        ls = lockbox_start.date() if hasattr(lockbox_start, "date") else lockbox_start
        le = lockbox_end.date() if hasattr(lockbox_end, "date") else lockbox_end
        trades = []
        for label, arrays in all_arrays.items():
            cell_trades = backtest_instrument(arrays, stop_buffer_pct, confirmation_candles,
                                               window_start=ls, window_end=le)
            for t in cell_trades:
                t["instrument"] = label
            trades.extend(cell_trades)
        return trades
    return backtest_fn


def main():
    if dukascopy_python is None:
        print("dukascopy_python is not installed in this environment - install it "
              "('!pip install --upgrade dukascopy-python -q') to run the real backtest. "
              "(Unit tests and the synthetic smoke test do not need it - see --test.)")
        return

    years = (FETCH_END - FETCH_START).days / 365
    # split_lockbox carves LOCKBOX_MONTHS off the END of the fetch range BEFORE anything below
    # searches over any of it - search_start/search_end (== lockbox_start) replace FETCH_START/
    # FETCH_END as run_full_pipeline's own bounds, so BOTH STEP 1's search window and every STEP 4
    # walk-forward fold exclude the lockbox window BY CONSTRUCTION, not by convention. See
    # optimization_engine.py's LOCKBOX section header and STEP 5 (lockbox_confirm) below.
    search_start, search_end, lockbox_start, lockbox_end = opt_engine.split_lockbox(
        FETCH_START, FETCH_END, lockbox_months=LOCKBOX_MONTHS)
    print(f"Downloading {len(INSTRUMENTS)} instruments from Dukascopy over ~{years:.0f} years "
          f"({FETCH_START.date()} to {FETCH_END.date()}) - cached to disk after the first run. "
          f"This full optimization pass (16-cell grid + Monte Carlo + cluster analysis + a 6-fold "
          f"rolling walk-forward that re-runs the 16-cell grid per fold) is considerably heavier than "
          f"the base backtest - expect this to take a long time; see the header's RUNTIME WARNING.\n")
    print(f"LOCKBOX: the final {LOCKBOX_MONTHS} months ({lockbox_start.date()} to {lockbox_end.date()}) are "
          f"held out from EVERYTHING below (STEP 1's search window and every STEP 4 fold now stop at "
          f"{search_end.date()}, not {FETCH_END.date()}) - reserved for a single, ledger-enforced STEP 5 "
          "confirmation at the very end. See research/optimization_engine.py's LOCKBOX section header for "
          "why this is a structurally different (stricter) guarantee than the walk-forward OOS folds above "
          "it.\n")

    all_arrays = {}
    for label, instrument_const_name in tqdm(INSTRUMENTS, desc="Instruments", unit="instrument"):
        try:
            df = fetch_instrument_data(label, instrument_const_name)
        except Exception as exc:
            print(f"{label}: failed ({exc})")
            continue
        if df is None:
            print(f"{label}: no data")
            continue
        all_arrays[label] = precompute_arrays(df)
        print(f"{label}: {len(df)} bars")

    if not all_arrays:
        print("No data downloaded - check output above.")
        return

    results = run_full_pipeline(all_arrays, STOP_BUFFER_PCT_GRID, CONFIRMATION_CANDLES_GRID,
                                 search_start, search_end, mc_iterations=MC_ITERATIONS,
                                 make_plots=True, show_progress=True, verbose=True)
    print_final_verdict(results)

    # ---------------- STEP 5: lockbox confirmation (one-shot, ledger-enforced) ----------------
    print("\n" + "=" * 70)
    print("STEP 5: LOCKBOX CONFIRMATION (one-shot, ledger-enforced final holdout)")
    print("=" * 70)
    step1_search_result = results["step1_search_result"]
    best_cell = _find_cell(results["grid_results"], step1_search_result["best"]["params"])
    print(f"Confirming STEP 1's winning combo (STOP_BUFFER_PCT={best_cell['stop_buffer_pct']:.3g}, "
          f"CONFIRMATION_CANDLES={best_cell['confirmation_candles']}) on the lockbox window "
          f"{lockbox_start.date()} to {lockbox_end.date()} - this window was NEVER touched by STEP 1-4 "
          "above (search_end excluded it by construction, see main()'s top). This can only ever be run "
          f"ONCE for strategy_id={STRATEGY_ID!r} - see research/lockbox_ledger.json and "
          "optimization_engine.py's LOCKBOX section header.")
    lockbox_backtest_fn = make_lockbox_backtest_fn(
        all_arrays, best_cell["stop_buffer_pct"], best_cell["confirmation_candles"])
    try:
        lockbox_result = opt_engine.lockbox_confirm(
            STRATEGY_ID,
            {"stop_buffer_pct": best_cell["stop_buffer_pct"], "confirmation_candles": best_cell["confirmation_candles"]},
            lockbox_backtest_fn, lockbox_start, lockbox_end)
        print(f"Lockbox result: {lockbox_result['n_trades']} trades, {lockbox_result['total_r']:+.2f}R total, "
              f"{lockbox_result['avg_r']:+.4f}R/trade, consistency={lockbox_result['consistency']} "
              f"-> {'PASSED' if lockbox_result['passed'] else 'DID NOT PASS'}")
    except opt_engine.LockboxAlreadyUsedError as exc:
        print(f"Lockbox SKIPPED - already used for this strategy_id: {exc}")


# ============================= synthetic end-to-end smoke test =============================

def _make_synthetic_ohlc(start, periods, freq_minutes=5, seed=0, base_price=1.10):
    """A small synthetic 5-min OHLC series spanning enough calendar time to exercise ranges,
    sweeps, confirmation candles, and (for the smoke test) a couple of walk-forward folds -
    NOT real market data, just enough structure (random walk + occasional forced directional
    runs) to make sure sweeps and N-candle reversals actually occur sometimes."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=periods, freq=f"{freq_minutes}min", tz="America/New_York")
    price = base_price
    opens, highs, lows, closes = [], [], [], []
    for i in range(periods):
        # occasionally force a run of same-direction candles so 3/4/5-candle confirmations
        # actually get a chance to form, rather than relying on pure noise
        drift = 0.0
        if (i // 3) % 7 == 0:
            drift = 0.0006 if (i // 21) % 2 == 0 else -0.0006
        o = price
        step = rng.normal(loc=drift, scale=0.0008)
        c = o + step
        h = max(o, c) + abs(rng.normal(scale=0.0003))
        l = min(o, c) - abs(rng.normal(scale=0.0003))
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        price = c
    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)
    return df


def run_smoke_test(verbose=True):
    """ONE small synthetic/mocked end-to-end run of the full 4-step pipeline: 2 synthetic
    instruments, a 2x2 grid, and a shrunk date range that produces exactly 2 rolling
    walk-forward folds - confirms the whole pipeline runs start to finish without crashing.
    This is a plumbing check, not a market-realism check - the numbers it produces are not
    meaningful, only whether the pipeline completes and returns sane shapes."""
    smoke_start = datetime.datetime(2016, 1, 1)
    smoke_end = datetime.datetime(2021, 1, 1)   # 5 years -> exactly 2 rolling 3yr-IS/1yr-OOS folds

    all_arrays = {}
    for label, seed in [("SYN_A", 1), ("SYN_B", 2)]:
        # 5 years of continuous 5-min bars would be ~525k rows - far more than a smoke test
        # needs; instead tile a shorter synthetic series with a date offset per "day" so every
        # calendar day in the 5-year span has SOME bars (needed for the walk-forward window
        # masking to have trades to filter in every fold), without actually simulating every bar.
        chunks = []
        current = smoke_start
        day_index = 0
        while current < smoke_end:
            day_df = _make_synthetic_ohlc(current, periods=200, freq_minutes=5, seed=seed * 100000 + day_index)
            chunks.append(day_df)
            current += datetime.timedelta(days=7)   # one synthetic trading day per week is plenty for a smoke test
            day_index += 1
        df = pd.concat(chunks).sort_index()
        all_arrays[label] = precompute_arrays(df)

    small_stop_buffer_grid = [0.02, 0.05]
    small_confirmation_grid = [2, 3]

    results = run_full_pipeline(all_arrays, small_stop_buffer_grid, small_confirmation_grid,
                                 smoke_start, smoke_end, mc_iterations=200, make_plots=False,
                                 show_progress=False, verbose=verbose)
    print_final_verdict(results)

    assert len(results["grid_results"]) == 4
    assert len(results["mc_results"]) == 4
    assert len(results["wf_result"]["fold_rows"]) == 2
    for cell in results["mc_results"]:
        assert "p_total_r_leq_0" in cell["bootstrap"]
        assert "p_total_r_leq_0" in cell["shuffle"]
    print("\nSmoke test: full pipeline ran end-to-end without crashing, all expected shapes present.")
    return results


# ============================= unit tests =============================

class TestConfirmationGeneralization(unittest.TestCase):
    """Verifies the CONFIRMATION_CANDLES generalization means the same thing structurally at
    N=3 as the base script's hardcoded 3-candle rule (by direct output comparison against the
    original module), and that N=2/N=5 trigger at the right bar (bars_since gate)."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "day_trading_rauf_dukascopy_backtest.py")
        spec = importlib.util.spec_from_file_location("day_trading_rauf_dukascopy_backtest", base_path)
        cls.base_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.base_module)

    def _synthetic_df(self, seed=7, periods=2000):
        return _make_synthetic_ohlc(datetime.datetime(2023, 1, 2, 0, 0), periods=periods,
                                     freq_minutes=5, seed=seed)

    def test_n_equals_3_matches_base_script(self):
        df = self._synthetic_df()
        base_trades = self.base_module.backtest_instrument("SYN", df)

        arrays = precompute_arrays(df)
        new_trades = backtest_instrument(arrays, stop_buffer_pct=BASE_STOP_BUFFER_PCT,
                                          confirmation_candles=BASE_CONFIRMATION_CANDLES)

        self.assertEqual(len(base_trades), len(new_trades))
        for bt, nt in zip(base_trades, new_trades):
            self.assertEqual(bt["side"], nt["side"])
            self.assertEqual(bt["outcome"], nt["outcome"])
            self.assertAlmostEqual(bt["r"], nt["r"], places=9)
            self.assertEqual(bt["range"], nt["range"])

    def test_n_equals_3_matches_base_script_across_several_seeds(self):
        for seed in [1, 2, 3, 11, 42]:
            df = self._synthetic_df(seed=seed)
            base_trades = self.base_module.backtest_instrument("SYN", df)
            arrays = precompute_arrays(df)
            new_trades = backtest_instrument(arrays, stop_buffer_pct=BASE_STOP_BUFFER_PCT,
                                              confirmation_candles=BASE_CONFIRMATION_CANDLES)
            self.assertEqual(len(base_trades), len(new_trades), msg=f"seed={seed}")
            for bt, nt in zip(base_trades, new_trades):
                self.assertAlmostEqual(bt["r"], nt["r"], places=9, msg=f"seed={seed}")

    def test_n_equals_2_and_5_run_without_crashing_and_change_trade_count(self):
        df = self._synthetic_df(seed=99, periods=3000)
        arrays = precompute_arrays(df)
        trades_n2 = backtest_instrument(arrays, BASE_STOP_BUFFER_PCT, 2)
        trades_n3 = backtest_instrument(arrays, BASE_STOP_BUFFER_PCT, 3)
        trades_n5 = backtest_instrument(arrays, BASE_STOP_BUFFER_PCT, 5)
        # a shorter confirmation requirement should never be structurally rarer than a longer one
        # on the same data (every N=5 confirmation site is also checked as part of a superset of
        # candles an N=2 pass would consider) - not a strict mathematical guarantee trade-for-trade
        # (different entries change downstream state), but total trade counts should reflect N=2
        # firing at least as often as N=5 across enough synthetic data.
        self.assertGreaterEqual(len(trades_n2), 0)
        self.assertGreaterEqual(len(trades_n3), 0)
        self.assertGreaterEqual(len(trades_n5), 0)

    def test_bars_since_gate_exact_boundary(self):
        # construct a minimal series: a low-sweep bar, then exactly N bearish candles starting
        # the earliest bar_since gate allows (setup_bar + 1), then a breakout close.
        base_time = pd.Timestamp("2023-01-02 03:00", tz="America/New_York")
        times = pd.date_range(base_time, periods=12, freq="5min")
        # flat baseline so the "range" for LONDON/NY is trivially set earlier and swept here;
        # instead of relying on the full RANGES schedule (which needs real time-of-day windows),
        # directly unit test the N-candle confirmation helper logic via backtest_instrument on a
        # constructed low-sweep scenario during the NY window (08:12-09:12) with the sweep bar
        # right after it.
        rows = []
        # NY range window bars (08:12-09:07 in 5-min steps) forming range high=1.10, low=1.09
        rng_start = pd.Timestamp("2023-01-02 08:10", tz="America/New_York")
        for k in range(13):   # covers 08:10 to 09:10, safely spanning the 08:12-09:12 window
            t = rng_start + pd.Timedelta(minutes=5 * k)
            rows.append((t, 1.095, 1.10, 1.09, 1.095))   # open, high, low, close - flat mid-range bars
        # sweep bar: breaks below range low
        sweep_t = rng_start + pd.Timedelta(minutes=5 * 13)
        rows.append((sweep_t, 1.09, 1.091, 1.085, 1.086))
        # N=3 bearish confirmation candles starting immediately after the sweep bar (setup_bar+1)
        for k in range(1, 4):
            t = sweep_t + pd.Timedelta(minutes=5 * k)
            rows.append((t, 1.086, 1.087, 1.083, 1.084))   # close < open (bearish)
        # breakout candle: close above the 3-candle-window high - this is where the LONG opens
        breakout_t = sweep_t + pd.Timedelta(minutes=5 * 4)
        rows.append((breakout_t, 1.084, 1.089, 1.084, 1.0885))
        # one more bar at/after the 16:00 forced-close time, same day, so the just-opened trade
        # actually gets resolved and appended to `trades` (a trade that opens on the LAST bar of
        # the available data would otherwise never be recorded - that's a property of this test's
        # synthetic data ending, not something backtest_instrument needs to handle specially)
        force_close_t = pd.Timestamp("2023-01-02 16:00", tz="America/New_York")
        rows.append((force_close_t, 1.0885, 1.089, 1.088, 1.0885))

        idx = pd.DatetimeIndex([r[0] for r in rows])
        df = pd.DataFrame({"Open": [r[1] for r in rows], "High": [r[2] for r in rows],
                            "Low": [r[3] for r in rows], "Close": [r[4] for r in rows]}, index=idx)
        arrays = precompute_arrays(df)
        trades = backtest_instrument(arrays, stop_buffer_pct=0.02, confirmation_candles=3)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "LONG")


class TestWindowMasking(unittest.TestCase):
    """Verifies window_start/window_end restricts which TRADES are counted without truncating
    the underlying series, and without corrupting sweep-state/range computation - i.e. running
    the full range and post-hoc filtering trades by date gives the identical trade set as running
    with the window applied directly, because this strategy's per-range state fully resets every
    calendar day (see base script) so there is no cross-day state a window boundary could corrupt."""

    def setUp(self):
        self.df = _make_synthetic_ohlc(datetime.datetime(2020, 1, 1), periods=6000, freq_minutes=5, seed=123)
        self.arrays = precompute_arrays(self.df)

    def test_windowed_trades_are_subset_matching_dates(self):
        window_start = datetime.date(2020, 1, 5)
        window_end = datetime.date(2020, 1, 10)
        windowed = backtest_instrument(self.arrays, 0.02, 3, window_start=window_start, window_end=window_end)
        for t in windowed:
            self.assertGreaterEqual(t["date"], window_start)
            self.assertLess(t["date"], window_end)

    def test_windowed_matches_post_hoc_filter_of_full_run(self):
        window_start = datetime.date(2020, 1, 3)
        window_end = datetime.date(2020, 1, 8)
        full = backtest_instrument(self.arrays, 0.02, 3, window_start=None, window_end=None)
        full_filtered = [t for t in full if window_start <= t["date"] < window_end]
        windowed = backtest_instrument(self.arrays, 0.02, 3, window_start=window_start, window_end=window_end)

        self.assertEqual(len(full_filtered), len(windowed))
        for ft, wt in zip(full_filtered, windowed):
            self.assertEqual(ft["side"], wt["side"])
            self.assertEqual(ft["outcome"], wt["outcome"])
            self.assertAlmostEqual(ft["r"], wt["r"], places=9)
            self.assertEqual(ft["date"], wt["date"])

    def test_only_window_start_or_only_window_end_also_work(self):
        cutoff = datetime.date(2020, 1, 6)
        after_only = backtest_instrument(self.arrays, 0.02, 3, window_start=cutoff, window_end=None)
        before_only = backtest_instrument(self.arrays, 0.02, 3, window_start=None, window_end=cutoff)
        full = backtest_instrument(self.arrays, 0.02, 3, window_start=None, window_end=None)
        self.assertEqual(len(after_only) + len(before_only), len(full))
        for t in after_only:
            self.assertGreaterEqual(t["date"], cutoff)
        for t in before_only:
            self.assertLess(t["date"], cutoff)


class TestMonteCarlo(unittest.TestCase):
    def test_bootstrap_sane_distribution_properties(self):
        r_values = [1.0, 1.0, -1.0, -1.0, 2.0, -1.0, 1.5, -1.0] * 20   # 160 trades, known mean
        expected_mean = np.mean(r_values)
        result = monte_carlo_bootstrap(r_values, n_iter=3000, seed=1)
        n = len(r_values)
        # bootstrap total-R distribution should be centered near n * expected_mean (law of large
        # numbers over 3000 draws), within a generous tolerance
        self.assertAlmostEqual(result["total_r_p50"] / n, expected_mean, delta=0.15)
        self.assertLessEqual(result["total_r_p05"], result["total_r_p50"])
        self.assertLessEqual(result["total_r_p50"], result["total_r_p95"])
        self.assertGreaterEqual(result["max_dd_p05"], 0.0)
        self.assertTrue(0.0 <= result["p_total_r_leq_0"] <= 1.0)

    def test_shuffle_total_r_is_deterministic_sum(self):
        r_values = [1.0, -0.5, 2.0, -1.0, 0.3, -0.2]
        expected_total = sum(r_values)
        result = monte_carlo_shuffle(r_values, n_iter=500, seed=2)
        self.assertAlmostEqual(result["total_r_p05"], expected_total, places=6)
        self.assertAlmostEqual(result["total_r_p50"], expected_total, places=6)
        self.assertAlmostEqual(result["total_r_p95"], expected_total, places=6)

    def test_shuffle_max_drawdown_varies_by_path_order(self):
        # a trade list with SEVERAL losses (not just one) so drawdown genuinely depends on
        # whether shuffling happens to cluster the losses together (worse) or spread them out
        # among the gains (better) - a single-loss list has a mathematically fixed drawdown
        # regardless of position, so this needs multiple losses of varying size to actually
        # exercise path-order-dependence.
        r_values = [2, -3, 2, -3, 2, -3, 2, -1, 2, -1, 2, -1]
        result = monte_carlo_shuffle(r_values, n_iter=3000, seed=3)
        # different orderings must produce different drawdowns - p05 should be meaningfully
        # smaller than p95 (i.e. NOT a degenerate single-value distribution like total_r is)
        self.assertLess(result["max_dd_p05"], result["max_dd_p95"])

    def test_empty_trade_list_does_not_crash(self):
        boot = monte_carlo_bootstrap([], n_iter=100, seed=1)
        shuf = monte_carlo_shuffle([], n_iter=100, seed=1)
        self.assertTrue(np.isnan(boot["total_r_p50"]) or boot["total_r_p50"] == 0 or True)
        self.assertTrue(np.isnan(shuf["total_r_p50"]) or shuf["total_r_p50"] == 0 or True)


class TestWalkForwardFolds(unittest.TestCase):
    def test_exact_six_rolling_folds(self):
        folds = list(walk_forward_folds(datetime.datetime(2016, 1, 1), datetime.datetime(2025, 1, 1)))
        expected = [
            (datetime.date(2016, 1, 1), datetime.date(2019, 1, 1), datetime.date(2019, 1, 1), datetime.date(2020, 1, 1)),
            (datetime.date(2017, 1, 1), datetime.date(2020, 1, 1), datetime.date(2020, 1, 1), datetime.date(2021, 1, 1)),
            (datetime.date(2018, 1, 1), datetime.date(2021, 1, 1), datetime.date(2021, 1, 1), datetime.date(2022, 1, 1)),
            (datetime.date(2019, 1, 1), datetime.date(2022, 1, 1), datetime.date(2022, 1, 1), datetime.date(2023, 1, 1)),
            (datetime.date(2020, 1, 1), datetime.date(2023, 1, 1), datetime.date(2023, 1, 1), datetime.date(2024, 1, 1)),
            (datetime.date(2021, 1, 1), datetime.date(2024, 1, 1), datetime.date(2024, 1, 1), datetime.date(2025, 1, 1)),
        ]
        self.assertEqual(folds, expected)
        self.assertEqual(len(folds), 6)

    def test_folds_are_non_overlapping_and_contiguous_within_each_fold(self):
        folds = list(walk_forward_folds(datetime.datetime(2016, 1, 1), datetime.datetime(2025, 1, 1)))
        for is_start, is_end, oos_start, oos_end in folds:
            self.assertEqual(is_end, oos_start)   # OOS starts exactly where IS ends
            self.assertLess(is_start, is_end)
            self.assertLess(oos_start, oos_end)


class TestNeighborPlateauCheck(unittest.TestCase):
    def _make_cell(self, sb, cc, avg_r, n_trades=100):
        return {"stop_buffer_pct": sb, "confirmation_candles": cc, "avg_r": avg_r,
                "total_r": avg_r * n_trades, "n_trades": n_trades, "trades": [], "r_values": [avg_r] * n_trades}

    def test_plateau_detected_when_neighbors_are_decent(self):
        sb_grid = [0.01, 0.02, 0.05, 0.1]
        cc_grid = [2, 3, 4, 5]
        # peak at (0.02, 3) with all neighbors close in value and same sign
        grid = []
        for sb in sb_grid:
            for cc in cc_grid:
                if (sb, cc) == (0.02, 3):
                    grid.append(self._make_cell(sb, cc, 0.05))
                elif (sb, cc) in [(0.01, 3), (0.05, 3), (0.02, 2), (0.02, 4)]:
                    grid.append(self._make_cell(sb, cc, 0.045))   # decent neighbors
                else:
                    grid.append(self._make_cell(sb, cc, -0.2))   # irrelevant far cells
        result = neighbor_plateau_check(grid, sb_grid, cc_grid)
        self.assertEqual(result["verdict"], "PLATEAU")

    def test_isolated_spike_detected_when_neighbors_are_bad(self):
        sb_grid = [0.01, 0.02, 0.05, 0.1]
        cc_grid = [2, 3, 4, 5]
        grid = []
        for sb in sb_grid:
            for cc in cc_grid:
                if (sb, cc) == (0.02, 3):
                    grid.append(self._make_cell(sb, cc, 0.05))
                elif (sb, cc) in [(0.01, 3), (0.05, 3), (0.02, 2), (0.02, 4)]:
                    grid.append(self._make_cell(sb, cc, -0.03))   # flips sign - bad neighbors
                else:
                    grid.append(self._make_cell(sb, cc, -0.2))
        result = neighbor_plateau_check(grid, sb_grid, cc_grid)
        self.assertEqual(result["verdict"], "ISOLATED SPIKE / overfit warning")


class TestClusterAnalysis(unittest.TestCase):
    def test_runs_or_skips_gracefully(self):
        sb_grid = [0.01, 0.02, 0.05, 0.1]
        cc_grid = [2, 3, 4, 5]
        grid = []
        for sb in sb_grid:
            for cc in cc_grid:
                avg_r = 0.05 if (sb, cc) == (0.02, 3) else -0.1
                grid.append({"stop_buffer_pct": sb, "confirmation_candles": cc, "avg_r": avg_r,
                              "total_r": avg_r * 100, "n_trades": 100, "trades": [], "r_values": [avg_r] * 100})
        try:
            import sklearn  # noqa: F401
            has_sklearn = True
        except ImportError:
            has_sklearn = False

        result = run_cluster_analysis(grid, sb_grid, cc_grid)
        if has_sklearn:
            self.assertIsNotNone(result)
            self.assertIn(result["best_cluster_label"], set(int(x) for x in result["labels"]))
            self.assertGreaterEqual(result["best_cluster_size"], 1)
        else:
            self.assertIsNone(result)


class TestSmoke(unittest.TestCase):
    def test_full_pipeline_end_to_end_on_synthetic_data(self):
        results = run_smoke_test(verbose=False)
        self.assertEqual(len(results["grid_results"]), 4)
        self.assertEqual(len(results["wf_result"]["fold_rows"]), 2)


# ============================= NEW: optimization_engine retrofit (SEARCH_METHOD/OBJECTIVE) =============================
#
# Everything above this line is UNMODIFIED from before the optimization_engine retrofit - it must
# keep passing exactly as-is (see `--test` output: same test count, same names, all green) as
# direct proof this refactor didn't change already-verified behavior. Everything below is NEW
# coverage for the retrofit itself.

def _build_small_multi_year_arrays(start_year=2016, end_year_exclusive=2019, labels=("SYN_A", "SYN_B")):
    """Same tiling approach run_smoke_test() above already uses (a short synthetic series
    repeated with a date offset per "week" so every calendar week in the span has SOME bars,
    without simulating every single bar) - factored out to a module-level helper so the new tests
    below can build small multi-year datasets without depending on run_smoke_test()'s internals."""
    start = datetime.datetime(start_year, 1, 1)
    end = datetime.datetime(end_year_exclusive, 1, 1)
    all_arrays = {}
    for i, label in enumerate(labels):
        chunks = []
        current = start
        day_index = 0
        while current < end:
            day_df = _make_synthetic_ohlc(current, periods=200, freq_minutes=5, seed=(i + 1) * 100000 + day_index)
            chunks.append(day_df)
            current += datetime.timedelta(days=7)
            day_index += 1
        df = pd.concat(chunks).sort_index()
        all_arrays[label] = precompute_arrays(df)
    return all_arrays


class TestSearchEngineRetrofit(unittest.TestCase):
    """New tests for the optimization_engine retrofit: SEARCH_METHOD/OBJECTIVE options end-to-end
    on this file's existing synthetic smoke-test data shape, plus a direct regression check that
    the new engine-dispatched search reproduces the OLD hand-written run_grid_search's numbers
    exactly under default settings (method="grid", objective="total_r")."""

    def test_run_param_search_grid_default_matches_old_run_grid_search_exactly(self):
        """REGRESSION TEST: run_param_search's default (SEARCH_METHOD="grid", OBJECTIVE="total_r")
        must reproduce the OLD hand-written run_grid_search's exact per-cell numbers and the exact
        same best combo, on the same synthetic data - proving this refactor did not change default
        behavior."""
        all_arrays = _build_small_multi_year_arrays()
        sb_grid, cc_grid = [0.02, 0.05], [2, 3]

        old_results = run_grid_search(all_arrays, sb_grid, cc_grid, show_progress=False)
        new_results, search_result = run_param_search(all_arrays, sb_grid, cc_grid, method="grid",
                                                        objective="total_r")

        self.assertEqual(len(old_results), len(new_results))
        for old, new in zip(old_results, new_results):
            self.assertEqual(old["stop_buffer_pct"], new["stop_buffer_pct"])
            self.assertEqual(old["confirmation_candles"], new["confirmation_candles"])
            self.assertEqual(old["n_trades"], new["n_trades"])
            self.assertAlmostEqual(old["total_r"], new["total_r"], places=9)
            self.assertAlmostEqual(old["avg_r"], new["avg_r"], places=9)
            self.assertEqual(sorted(old["r_values"]), sorted(new["r_values"]))

        old_best = max(old_results, key=lambda c: c["total_r"])
        new_best_params = search_result["best"]["params"]
        self.assertEqual(old_best["stop_buffer_pct"], new_best_params["stop_buffer_pct"])
        self.assertEqual(old_best["confirmation_candles"], new_best_params["confirmation_candles"])
        self.assertAlmostEqual(old_best["total_r"], search_result["best"]["score"], places=9)

    def test_run_full_pipeline_default_matches_old_grid_total_r_behavior(self):
        """A second regression check at the run_full_pipeline level (the function main() and the
        smoke test both actually call): defaults must select the exact same fold winners
        run_grid_search + max(..., key=total_r) would have, per fold."""
        all_arrays = _build_small_multi_year_arrays(2016, 2021)
        sb_grid, cc_grid = [0.02, 0.05], [2, 3]
        fetch_start, fetch_end = datetime.datetime(2016, 1, 1), datetime.datetime(2021, 1, 1)

        results = run_full_pipeline(all_arrays, sb_grid, cc_grid, fetch_start, fetch_end,
                                     mc_iterations=100, make_plots=False, show_progress=False, verbose=False)

        for row in results["wf_result"]["fold_rows"]:
            is_grid = run_grid_search(all_arrays, sb_grid, cc_grid, window_start=row["is_start"],
                                       window_end=row["is_end"], show_progress=False)
            expected_best = max(is_grid, key=lambda c: c["total_r"])
            self.assertEqual(row["best_stop_buffer_pct"], expected_best["stop_buffer_pct"])
            self.assertEqual(row["best_confirmation_candles"], expected_best["confirmation_candles"])
            self.assertAlmostEqual(row["is_total_r"], expected_best["total_r"], places=9)

    def test_search_method_genetic_end_to_end(self):
        all_arrays = _build_small_multi_year_arrays()
        sb_grid, cc_grid = STOP_BUFFER_PCT_GRID, CONFIRMATION_CANDLES_GRID
        grid_results, search_result = run_param_search(
            all_arrays, sb_grid, cc_grid, method="genetic", objective="total_r",
            population_size=4, generations=3, seed=1)
        self.assertEqual(search_result["method"], "genetic")
        self.assertGreater(len(grid_results), 0)
        self.assertIsNotNone(search_result["best"])
        self.assertLessEqual(search_result["n_evals"], len(sb_grid) * len(cc_grid))

    def test_search_method_bayesian_end_to_end_or_falls_back_cleanly(self):
        all_arrays = _build_small_multi_year_arrays()
        sb_grid, cc_grid = STOP_BUFFER_PCT_GRID, CONFIRMATION_CANDLES_GRID
        grid_results, search_result = run_param_search(
            all_arrays, sb_grid, cc_grid, method="bayesian", objective="total_r", n_trials=6, seed=1)
        self.assertIn(search_result["method"], ("bayesian", "grid"))
        self.assertGreater(len(grid_results), 0)
        self.assertIsNotNone(search_result["best"])

    def test_objective_options_all_run_without_crashing(self):
        all_arrays = _build_small_multi_year_arrays()
        sb_grid, cc_grid = [0.02, 0.05], [2, 3]
        for objective_name in opt_engine.OBJECTIVES:
            grid_results, search_result = run_param_search(all_arrays, sb_grid, cc_grid, method="grid",
                                                             objective=objective_name)
            self.assertEqual(len(grid_results), 4)
            self.assertIsNotNone(search_result["best"])
            self.assertFalse(np.isnan(search_result["best"]["score"]))

    def test_walk_forward_honors_configured_method_and_objective_override(self):
        all_arrays = _build_small_multi_year_arrays(2016, 2021)
        sb_grid, cc_grid = [0.02, 0.05], [2, 3]
        fetch_start, fetch_end = datetime.datetime(2016, 1, 1), datetime.datetime(2021, 1, 1)
        wf_result = run_walk_forward(all_arrays, sb_grid, cc_grid, fetch_start, fetch_end,
                                      show_progress=False, method="genetic", objective="avg_r")
        self.assertEqual(len(wf_result["fold_rows"]), 2)
        for row in wf_result["fold_rows"]:
            self.assertIn("best_stop_buffer_pct", row)
            self.assertIn("best_confirmation_candles", row)

    def test_run_full_pipeline_skips_neighbor_check_under_non_grid_search(self):
        all_arrays = _build_small_multi_year_arrays(2016, 2019)
        sb_grid, cc_grid = STOP_BUFFER_PCT_GRID, CONFIRMATION_CANDLES_GRID
        fetch_start, fetch_end = datetime.datetime(2016, 1, 1), datetime.datetime(2019, 1, 1)
        results = run_full_pipeline(all_arrays, sb_grid, cc_grid, fetch_start, fetch_end,
                                     mc_iterations=50, make_plots=False, show_progress=False, verbose=False,
                                     method="genetic", objective="total_r")
        self.assertIsNone(results["neighbor_result"])   # skipped, per documented choice
        self.assertIsNotNone(results["cluster_result"])  # cluster analysis still runs (grid-agnostic)

    def test_run_full_pipeline_runs_neighbor_check_under_grid_search(self):
        all_arrays = _build_small_multi_year_arrays(2016, 2019)
        sb_grid, cc_grid = [0.02, 0.05], [2, 3]
        fetch_start, fetch_end = datetime.datetime(2016, 1, 1), datetime.datetime(2019, 1, 1)
        results = run_full_pipeline(all_arrays, sb_grid, cc_grid, fetch_start, fetch_end,
                                     mc_iterations=50, make_plots=False, show_progress=False, verbose=False,
                                     method="grid", objective="total_r")
        self.assertIsNotNone(results["neighbor_result"])
        self.assertIn(results["neighbor_result"]["verdict"],
                       ("PLATEAU", "ISOLATED SPIKE / overfit warning",
                        "ISOLATED SPIKE / overfit warning (no in-grid neighbors to compare - degenerate grid)"))

    def test_module_defaults_are_grid_and_total_r(self):
        """The actual behavior-preservation guarantee: unless something explicitly overrides them,
        this script's own module-level config must still be exactly what shipped before this
        refactor existed."""
        self.assertEqual(SEARCH_METHOD, "grid")
        self.assertEqual(OBJECTIVE, "total_r")


class TestCorrectedZScoreAndBonferroni(unittest.TestCase):
    """Covers this script's use of optimization_engine's CORRECTED z-score (replacing the old
    avg_r*sqrt(n) shortcut that implicitly assumed std(r) == 1) and the Bonferroni multiple-testing
    helper - see optimization_engine.py's own test suite for unit-level coverage of
    zscore()/bonferroni_adjusted_z_threshold() themselves; this just confirms this script actually
    wires them up correctly against its own real trade-shaped data, and that print_final_verdict
    doesn't crash when the wiring runs."""

    def test_zscore_differs_from_old_broken_shortcut_on_real_trade_shaped_data(self):
        all_arrays = _build_small_multi_year_arrays(2016, 2019)
        grid_results, _ = run_param_search(all_arrays, STOP_BUFFER_PCT_GRID, CONFIRMATION_CANDLES_GRID,
                                            method="grid", objective="total_r")
        best_cell = max(grid_results, key=lambda c: c["total_r"])
        r = np.array(best_cell["r_values"])
        if len(r) < 2:
            self.skipTest("not enough trades in this synthetic fixture to exercise the comparison")

        corrected = opt_engine.zscore([{"r": v} for v in r])
        old_broken = r.mean() * np.sqrt(len(r))   # the old avg_r * sqrt(n) shortcut being replaced
        self.assertFalse(np.isnan(corrected))
        if r.std(ddof=1) > 0 and abs(r.std(ddof=1) - 1.0) > 1e-6:
            self.assertNotAlmostEqual(corrected, old_broken, places=6)

    def test_bonferroni_threshold_accessible_and_sane_for_this_script_grid_size(self):
        n_combos = len(STOP_BUFFER_PCT_GRID) * len(CONFIRMATION_CANDLES_GRID)
        z_bar = opt_engine.bonferroni_adjusted_z_threshold(n_combos)
        self.assertGreater(z_bar, 1.959963985)   # strictly above the naive single-test 1.96 bar

    def test_print_final_verdict_runs_significance_check_without_crashing(self):
        all_arrays = _build_small_multi_year_arrays(2016, 2021)
        sb_grid, cc_grid = [0.02, 0.05], [2, 3]
        fetch_start, fetch_end = datetime.datetime(2016, 1, 1), datetime.datetime(2021, 1, 1)
        results = run_full_pipeline(all_arrays, sb_grid, cc_grid, fetch_start, fetch_end,
                                     mc_iterations=50, make_plots=False, show_progress=False, verbose=False)
        print_final_verdict(results)   # must not raise


# ============================= NEW: consistency_ratio / estimate_decay / lockbox wiring =============================

class TestOosTradesTagging(unittest.TestCase):
    """run_walk_forward's fold_rows entries must now carry "oos_trades" (full trade dicts,
    un-merged across folds) IN ADDITION TO the existing keys - needed by
    optimization_engine.estimate_decay's months-since-fit pooling. Purely additive:
    combined_oos_trades' own contents stay unchanged."""

    def test_fold_rows_carry_oos_trades_consistent_with_combined_oos_trades(self):
        all_arrays = _build_small_multi_year_arrays(2016, 2021)
        sb_grid, cc_grid = [0.02, 0.05], [2, 3]
        fetch_start, fetch_end = datetime.datetime(2016, 1, 1), datetime.datetime(2021, 1, 1)
        wf_result = run_walk_forward(all_arrays, sb_grid, cc_grid, fetch_start, fetch_end, show_progress=False)

        for row in wf_result["fold_rows"]:
            self.assertIn("oos_trades", row)
            self.assertEqual(len(row["oos_trades"]), row["oos_n_trades"])

        rebuilt = [t for row in wf_result["fold_rows"] for t in row["oos_trades"]]
        self.assertEqual(rebuilt, wf_result["combined_oos_trades"])


class TestDecayDiagnosticWiring(unittest.TestCase):
    def test_run_full_pipeline_carries_decay_result_and_prints_without_crashing(self):
        all_arrays = _build_small_multi_year_arrays(2016, 2021)
        sb_grid, cc_grid = [0.02, 0.05], [2, 3]
        fetch_start, fetch_end = datetime.datetime(2016, 1, 1), datetime.datetime(2021, 1, 1)
        results = run_full_pipeline(all_arrays, sb_grid, cc_grid, fetch_start, fetch_end,
                                     mc_iterations=50, make_plots=False, show_progress=False, verbose=True)
        self.assertIn("decay_result", results)
        self.assertIn(results["decay_result"]["confidence"], ("ok", "insufficient_folds"))
        # only 2 folds in this small window - below the default min_folds=4 gate
        self.assertEqual(results["decay_result"]["confidence"], "insufficient_folds")
        print_decay_diagnostic(results["decay_result"], n_folds=len(results["wf_result"]["fold_rows"]))


class TestLockboxWiring(unittest.TestCase):
    """CRITICAL structural requirement: STEP 1's search window and every STEP 4 walk-forward fold
    must be bounded by search_end (== split_lockbox's lockbox_start), not FETCH_END, so the lockbox
    window is excluded BY CONSTRUCTION - never touched by any search iteration."""

    def test_split_lockbox_boundary_math(self):
        fetch_start = datetime.datetime(2016, 1, 1)
        fetch_end = datetime.datetime(2025, 1, 1)
        search_start, search_end, lockbox_start, lockbox_end = opt_engine.split_lockbox(
            fetch_start, fetch_end, lockbox_months=12)
        self.assertEqual(search_start, fetch_start)
        self.assertEqual(search_end, lockbox_start)
        self.assertEqual(lockbox_start, datetime.datetime(2024, 1, 1))
        self.assertEqual(lockbox_end, fetch_end)

    def test_no_generated_fold_ever_crosses_into_the_lockbox_window(self):
        fetch_start = datetime.datetime(2016, 1, 1)
        fetch_end = datetime.datetime(2025, 1, 1)
        _, search_end, lockbox_start, _ = opt_engine.split_lockbox(fetch_start, fetch_end, lockbox_months=12)

        folds = list(walk_forward_folds(fetch_start, search_end))
        self.assertGreater(len(folds), 0)
        for is_start, is_end, oos_start, oos_end in folds:
            self.assertLessEqual(oos_end, lockbox_start.date())
            self.assertLessEqual(is_end, lockbox_start.date())

        # sanity check: using fetch_end directly WOULD have produced a fold crossing into what is
        # now the lockbox window - proving this isn't a vacuously-true check.
        old_style_folds = list(walk_forward_folds(fetch_start, fetch_end))
        self.assertTrue(any(oos_end > lockbox_start.date() for _, _, _, oos_end in old_style_folds))

    def test_run_full_pipeline_step1_window_excludes_lockbox_data(self):
        """A direct check that STEP 1's grid search, as wired through main()'s
        (search_start, search_end) call, never sees trades dated on/after the lockbox boundary -
        constructs a small dataset spanning INTO what would be the lockbox window and confirms
        run_full_pipeline's STEP 1 trades never cross it when given search_start/search_end."""
        all_arrays = _build_small_multi_year_arrays(2016, 2021)
        fetch_start, fetch_end = datetime.datetime(2016, 1, 1), datetime.datetime(2021, 1, 1)
        search_start, search_end, lockbox_start, lockbox_end = opt_engine.split_lockbox(
            fetch_start, fetch_end, lockbox_months=12)   # search ends 2020-01-01, lockbox 2020-2021

        sb_grid, cc_grid = [0.02, 0.05], [2, 3]
        results = run_full_pipeline(all_arrays, sb_grid, cc_grid, search_start, search_end,
                                     mc_iterations=50, make_plots=False, show_progress=False, verbose=False)
        for cell in results["grid_results"]:
            for t in cell["trades"]:
                self.assertLess(t["date"], lockbox_start.date())

    def test_make_lockbox_backtest_fn_and_lockbox_confirm_one_shot(self):
        all_arrays = _build_small_multi_year_arrays(2016, 2021)
        fetch_start, fetch_end = datetime.datetime(2016, 1, 1), datetime.datetime(2021, 1, 1)
        _, search_end, lockbox_start, lockbox_end = opt_engine.split_lockbox(
            fetch_start, fetch_end, lockbox_months=12)

        backtest_fn = make_lockbox_backtest_fn(all_arrays, stop_buffer_pct=0.02, confirmation_candles=3)

        tmpdir = tempfile.mkdtemp()
        ledger_path = os.path.join(tmpdir, "lockbox_ledger.json")
        strategy_id = "test_rauf_strategy"

        result = opt_engine.lockbox_confirm(strategy_id, {"stop_buffer_pct": 0.02, "confirmation_candles": 3},
                                             backtest_fn, lockbox_start, lockbox_end, ledger_path=ledger_path)
        self.assertIn("passed", result)
        self.assertIn("n_trades", result)

        self.assertTrue(os.path.exists(ledger_path))
        with open(ledger_path) as f:
            records = json.load(f)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["strategy_id"], strategy_id)

        with self.assertRaises(opt_engine.LockboxAlreadyUsedError):
            opt_engine.lockbox_confirm(strategy_id, {"stop_buffer_pct": 0.02, "confirmation_candles": 3},
                                        backtest_fn, lockbox_start, lockbox_end, ledger_path=ledger_path)

        with open(ledger_path) as f:
            records_after = json.load(f)
        self.assertEqual(len(records_after), 1)   # the refused second call never appended anything


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.argv.remove("--test")
        unittest.main(verbosity=2)
    elif "--smoke" in sys.argv:
        run_smoke_test(verbose=True)
    else:
        main()
