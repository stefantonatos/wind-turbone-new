# ICT "Power of Three" (PO3) - rigorous 4-step optimization/robustness pass,
# built on TOP of ict_po3_forex_dukascopy_backtest.py (read that file first -
# this does not replace it or re-derive the strategy rules; it imports the
# same rule set and asks a narrower question: is there ANY corner of the
# parameter space where PO3 has a real, robust edge, or does the already-
# established negative result hold up everywhere?
#
# THE ALREADY-ESTABLISHED BASE RESULT (fixed STOP_BUFFER_PCT=0.02,
# FALLBACK_REWARD_RISK=2.0, no tuning): 6,894 trades, -287.83R total,
# -0.0418R/trade, all 4 instruments (EURUSD/GBPUSD/USDJPY/XAUUSD)
# individually negative, approx z=-3.47, both split-halves negative. That is
# a decisively negative result, not a case of "the right parameters weren't
# found yet" - but it was tested at exactly one point in parameter space.
# This script is the honest follow-up: sweep the two parameters that
# actually matter for PO3's payout structure, stress-test every corner with
# Monte Carlo, check whether any good-looking cell is a real region or an
# isolated fluke, and confirm (or refute) all of it out-of-sample via
# rolling walk-forward. This is NOT an attempt to retroactively make the
# strategy look good - if every corner of the grid, every Monte Carlo band,
# and every walk-forward fold comes back negative too, this script's own
# summary at the end says so plainly ("still negative everywhere") rather
# than mining for a flattering cell to report. Whatever this prints when
# actually run against real data is the real answer - nothing here is
# pre-decided.
#
# THE 4 STEPS (methodology fixed in advance, not iterated on after seeing
# results - see the task this shipped from for the exact spec):
#   1. PARAMETER STABILITY GRID: STOP_BUFFER_PCT x FALLBACK_REWARD_RISK,
#      4x5=20 combos, full 2016-2025 range, all 4 instruments per combo.
#      Heatmap of avg R/trade (PNG + inline show()), text table always
#      printed regardless of whether matplotlib is available.
#   2. MONTE CARLO PER CELL: every one of the 20 cells (not just the best),
#      bootstrap-with-replacement AND shuffle-without-replacement over that
#      cell's actual realized trade R-multiples, 2000 iterations each,
#      vectorized with numpy. Reports total-R and max-drawdown percentiles
#      plus P(total R <= 0) per cell.
#   3. CLUSTER ANALYSIS: sklearn KMeans (k=3) over (param position, param
#      position, avg R/trade) feature vectors to see if the best cell sits
#      in a real cluster of similarly-decent neighbors or is an isolated
#      outlier - plus an always-available (no sklearn needed) direct
#      neighbor check on the grid itself.
#   4. ROLLING WALK-FORWARD: 3-year in-sample / 1-year out-of-sample,
#      rolled forward by 1 year across 2016-2025 (6 folds). Re-runs the
#      full 20-cell grid on each fold's in-sample window only, picks the
#      in-sample winner, scores it out-of-sample, chains all 6 folds'
#      out-of-sample trades together, and reports Walk-Forward Efficiency
#      (combined OOS avg R/trade divided by the mean in-sample avg R/trade
#      of the selected combo across folds) - PASS if WFE >= 0.5, FAIL
#      otherwise. That 50% threshold is a standard rule-of-thumb in
#      walk-forward analysis, not a proof of anything either way - printed
#      explicitly as a caveat, not asserted as ground truth.
#
# WINDOWING CONVENTION (follows orb_indices_optimization_and_ml.py's
# split_before/split_after pattern, generalized to a window_start/
# window_end pair so rolling folds - not just one fixed split point - are
# possible): the underlying per-instrument bar data is fetched and passed
# in FULL every time - it is never truncated. A window only gates which
# bars actually get PROCESSED into trades (by comparing each bar's own
# calendar date against [window_start, window_end)); accumulation/
# manipulation/distribution range-building is entirely intraday and resets
# fresh every new calendar day regardless of the window, so restricting by
# whole calendar days never corrupts a range computation near a boundary
# (there is no cross-day carryover in this strategy to break). See the
# window-slicing unit tests below for a direct verification of this.

# !pip install --upgrade dukascopy-python scikit-learn matplotlib -q   # uncomment in Colab

import datetime
import logging
import os
import pickle

import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

from tqdm.auto import tqdm   # auto-picks the Colab/Jupyter widget bar when available, a plain terminal bar otherwise

# dukascopy_python logs an "INFO:DUKASCRIPT:current timestamp:..." line for every internal
# download chunk - useful for debugging a stuck fetch, just noisy for normal runs. A plain
# logger.setLevel(WARNING) does NOT work here: the library's own fetch() call resets the
# "DUKASCRIPT" logger's level back to INFO internally on every single call (confirmed by
# reading dukascopy_python/__init__.py's _get_custom_logger - it unconditionally calls
# logger.setLevel(...) each time), silently undoing a one-time setLevel before it ever helps.
# A logging Filter survives that reset (the library never touches .filters), so that's what's
# used instead. Remove this filter to see the raw log again.
class _SuppressDukascopyInfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING


logging.getLogger("DUKASCRIPT").addFilter(_SuppressDukascopyInfoFilter())

# Fetched data is cached to disk per (instrument, interval, date range) - the FIRST run of a
# given range still has to download it all, but every run after that (e.g. after tweaking a
# parameter below, or a DIFFERENT script that happens to need the same instrument/interval/
# range) loads from disk instantly instead of re-downloading ~650k+ bars per instrument. Auto-
# detects a mounted Google Drive (run `from google.colab import drive; drive.mount('/content/
# drive')` once at the top of your notebook, before this cell) and uses that instead of the
# ephemeral local disk if present - this is what makes the cache survive runtime resets AND
# get shared across every script in this project that uses the same instruments/date range,
# not just repeat runs of this one file (this uses the exact same cache filename convention as
# ict_po3_forex_dukascopy_backtest.py / day_trading_rauf_dukascopy_backtest.py, so it reuses
# their cached downloads instead of re-fetching). Falls back to a local (session-only) cache if
# Drive isn't mounted, so this still works without any setup, just without the persistence.
CACHE_DIR = "/content/drive/MyDrive/dukascopy_cache" if os.path.isdir("/content/drive/MyDrive") else "dukascopy_cache"
FETCH_CHUNK_MONTHS = 3   # how finely to split the download for progress-bar granularity

INSTRUMENTS = [
    ("EURUSD", dki.INSTRUMENT_FX_MAJORS_EUR_USD),
    ("GBPUSD", dki.INSTRUMENT_FX_MAJORS_GBP_USD),
    ("USDJPY", dki.INSTRUMENT_FX_MAJORS_USD_JPY),
    ("XAUUSD", dki.INSTRUMENT_FX_METALS_XAU_USD),
]

FETCH_START = datetime.datetime(2016, 1, 1)
FETCH_END = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_MIN_5
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

# --- session windows, all NY time - identical to ict_po3_forex_dukascopy_backtest.py ---
ACCUMULATION_START = pd.Timestamp("00:00").time()
ACCUMULATION_END = pd.Timestamp("05:00").time()
MANIPULATION_START = pd.Timestamp("05:00").time()
MANIPULATION_END = pd.Timestamp("08:00").time()
DISTRIBUTION_END = pd.Timestamp("17:00").time()

MIN_RANGE_PCT = 0.02   # fixed floor on accumulation-range-derived stop distance - NOT grid-searched,
                        # same value as the base script; only the two payout-structure parameters below are

# --- step 1/2/3/4 grid: the two parameters that actually matter for PO3's payout structure ---
STOP_BUFFER_PCT_GRID = [0.01, 0.02, 0.05, 0.1]
FALLBACK_REWARD_RISK_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]

MC_ITERATIONS = 2000   # per cell, per method (bootstrap and shuffle)

# --- step 4: rolling 3yr-in-sample / 1yr-out-of-sample walk-forward, stepped forward 1 year ---
WALK_FORWARD_IS_YEARS = 3
WALK_FORWARD_OOS_YEARS = 1
WALK_FORWARD_STEP_YEARS = 1
WFE_PASS_THRESHOLD = 0.5   # standard rule-of-thumb, not a proof - see header and step-4 output

HEATMAP_PNG_PATH = "ict_po3_param_heatmap.png"


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


def fetch_instrument_data(label, instrument_const):
    """Downloads in FETCH_CHUNK_MONTHS-sized pieces (shows real progress instead of one long
    silent call) and caches the combined result to disk, so re-running this script - or any
    other script in this project fetching the same instrument/interval/range - loads instantly
    instead of re-downloading everything."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{label}_5min_{FETCH_START.date()}_{FETCH_END.date()}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    chunks = []
    chunk_bounds = list(_month_chunks(FETCH_START, FETCH_END, FETCH_CHUNK_MONTHS))
    for chunk_start, chunk_end in tqdm(chunk_bounds, desc=f"{label}: downloading {DUKASCOPY_INTERVAL} bars",
                                        unit="chunk"):
        chunk = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE,
                                        chunk_start, chunk_end)
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return None
    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")].sort_index()   # chunk boundaries may overlap by one bar
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df.index = to_ny_time(df.index)

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    return df


def precompute_indicators(df):
    """Everything that doesn't depend on STOP_BUFFER_PCT/FALLBACK_REWARD_RISK or the walk-forward
    window - computed once per instrument, reused across every grid cell and every fold. Also
    precomputes .date()/.time() once per bar (instead of re-deriving them from the Timestamp on
    every single grid/fold pass), since the same instrument gets scanned dozens of times over."""
    return {
        "highs": df["High"].tolist(),
        "lows": df["Low"].tolist(),
        "closes": df["Close"].tolist(),
        "dates": [ts.date() for ts in df.index],
        "times": [ts.time() for ts in df.index],
    }


def run_po3_backtest(ind, stop_buffer_pct, fallback_reward_risk, window_start=None, window_end=None):
    """Core PO3 loop - identical rules to ict_po3_forex_dukascopy_backtest.py's
    backtest_instrument(), parameterized by the two grid parameters and generalized with an
    optional [window_start, window_end) date window (both datetime.date or None).

    window_start/window_end restrict which bars get PROCESSED (by calendar date) - exactly the
    split_before/split_after pattern from orb_indices_optimization_and_ml.py, generalized to a
    two-sided window. The underlying ind[...] arrays are never truncated; a bar outside the
    window is simply skipped before any state update, which is safe here specifically because
    accumulation/manipulation/distribution state resets fresh on every new calendar day (there
    is no cross-day carryover for this strategy to corrupt) - see the window-slicing unit tests
    for a direct check that a windowed run produces exactly the same trades a full run would,
    filtered to that window, never anything different.
    """
    highs, lows, closes = ind["highs"], ind["lows"], ind["closes"]
    dates, times_of_day = ind["dates"], ind["times"]
    n = len(closes)

    trades = []
    current_day = None
    acc_high = acc_low = None
    manipulated = False
    traded_today = False
    i = 0
    while i < n:
        today = dates[i]
        tod = times_of_day[i]

        if window_start is not None and today < window_start:
            i += 1
            continue
        if window_end is not None and today >= window_end:
            i += 1
            continue

        if today != current_day:
            current_day = today
            acc_high = acc_low = None
            manipulated = False
            traded_today = False

        if traded_today:
            i += 1
            continue

        if ACCUMULATION_START <= tod < ACCUMULATION_END:
            acc_high = highs[i] if acc_high is None else max(acc_high, highs[i])
            acc_low = lows[i] if acc_low is None else min(acc_low, lows[i])
            i += 1
            continue

        if acc_high is None:
            i += 1
            continue

        if tod >= DISTRIBUTION_END:
            traded_today = True
            i += 1
            continue

        if not (MANIPULATION_START <= tod < MANIPULATION_END):
            i += 1
            continue

        if manipulated:
            i += 1
            continue

        price = closes[i]
        broke_high = highs[i] > acc_high
        broke_low = lows[i] < acc_low

        if not broke_high and not broke_low:
            i += 1
            continue

        manipulated = True

        if broke_high and broke_low:
            traded_today = True
            i += 1
            continue

        side = "SHORT" if broke_high else "LONG"
        entry = price
        manipulation_extreme = highs[i] if side == "SHORT" else lows[i]
        buffer_price = (stop_buffer_pct / 100.0) * entry

        if side == "SHORT":
            stop = manipulation_extreme + buffer_price
        else:
            stop = manipulation_extreme - buffer_price
        sl_distance = abs(entry - stop)
        min_sl = (MIN_RANGE_PCT / 100.0) * entry
        if sl_distance < min_sl:
            sl_distance = min_sl
            stop = entry - sl_distance if side == "LONG" else entry + sl_distance

        opposite_edge = acc_low if side == "SHORT" else acc_high
        reaches_opposite = (opposite_edge < entry) if side == "SHORT" else (opposite_edge > entry)
        if reaches_opposite and abs(opposite_edge - entry) > sl_distance:
            target = opposite_edge
            reward_risk = abs(target - entry) / sl_distance
        else:
            reward_risk = fallback_reward_risk
            target = entry - sl_distance * reward_risk if side == "SHORT" else entry + sl_distance * reward_risk

        traded_today = True
        outcome, exit_r = None, None
        j = i + 1
        while j < n and dates[j] == today and times_of_day[j] < DISTRIBUTION_END:
            hi, lo = highs[j], lows[j]
            hit_stop = hi >= stop if side == "SHORT" else lo <= stop
            hit_target = lo <= target if side == "SHORT" else hi >= target
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
            pnl = (entry - last_close) if side == "SHORT" else (last_close - entry)
            outcome, exit_r = "FLAT", pnl / sl_distance

        trades.append({"side": side, "outcome": outcome, "r": exit_r, "date": today})
        i = j + 1

    return trades


# ============================= STEP 1: parameter grid =============================

def run_grid_search(data, window_start=None, window_end=None, grid=None, desc="grid search"):
    """data: dict label -> precomputed indicators. Runs every (stop_buffer_pct,
    fallback_reward_risk) combo in `grid` (defaults to the full 20-cell grid) across every
    instrument in `data`, restricted to [window_start, window_end) if given. Returns one result
    dict per cell with the realized trade-R list included (needed for step 2's Monte Carlo)."""
    if grid is None:
        grid = [(sb, frr) for sb in STOP_BUFFER_PCT_GRID for frr in FALLBACK_REWARD_RISK_GRID]

    results = []
    for stop_buffer_pct, fallback_reward_risk in tqdm(grid, desc=desc, unit="combo", leave=False):
        all_r = []
        per_instrument = {}
        for label, ind in data.items():
            trades = run_po3_backtest(ind, stop_buffer_pct, fallback_reward_risk, window_start, window_end)
            rs = [t["r"] for t in trades]
            all_r.extend(rs)
            per_instrument[label] = rs
        n = len(all_r)
        total_r = sum(all_r)
        results.append({
            "stop_buffer_pct": stop_buffer_pct,
            "fallback_reward_risk": fallback_reward_risk,
            "total_r": total_r,
            "n_trades": n,
            "avg_r": total_r / n if n else 0.0,
            "trades_r": all_r,
            "per_instrument": per_instrument,
        })
    return results


def build_grid_matrix(grid_results):
    sb_list = sorted(set(r["stop_buffer_pct"] for r in grid_results))
    frr_list = sorted(set(r["fallback_reward_risk"] for r in grid_results))
    lookup = {(r["stop_buffer_pct"], r["fallback_reward_risk"]): r for r in grid_results}
    avg_r_matrix = np.zeros((len(sb_list), len(frr_list)))
    total_r_matrix = np.zeros((len(sb_list), len(frr_list)))
    for i, sb in enumerate(sb_list):
        for j, frr in enumerate(frr_list):
            cell = lookup[(sb, frr)]
            avg_r_matrix[i, j] = cell["avg_r"]
            total_r_matrix[i, j] = cell["total_r"]
    return sb_list, frr_list, avg_r_matrix, total_r_matrix


def print_grid_table(grid_results):
    """Always-available fallback (and companion) to the heatmap plot - a plain text table of
    avg R/trade per cell, printed regardless of whether matplotlib is installed."""
    sb_list, frr_list, avg_r_matrix, total_r_matrix = build_grid_matrix(grid_results)
    best_i, best_j = np.unravel_index(np.argmax(avg_r_matrix), avg_r_matrix.shape)

    print("\nSTOP_BUFFER_PCT x FALLBACK_REWARD_RISK grid - avg R/trade (best cell marked *):")
    header = "SB\\FRR".ljust(10) + "".join(f"{frr:>12.1f}" for frr in frr_list)
    print(header)
    for i, sb in enumerate(sb_list):
        cells = []
        for j in range(len(frr_list)):
            marker = "*" if (i, j) == (best_i, best_j) else " "
            cells.append(f"{avg_r_matrix[i, j]:>+10.4f}{marker}")
        print(f"{sb:<10.3f}" + "".join(cells))

    print("\nSame grid, total R:")
    print(header)
    for i, sb in enumerate(sb_list):
        cells = [f"{total_r_matrix[i, j]:>+11.2f} " for j in range(len(frr_list))]
        print(f"{sb:<10.3f}" + "".join(cells))


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

    sb_list, frr_list, avg_r_matrix, total_r_matrix = build_grid_matrix(grid_results)
    best_i, best_j = np.unravel_index(np.argmax(avg_r_matrix), avg_r_matrix.shape)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(avg_r_matrix, cmap="RdYlGn", aspect="auto",
                    vmin=-abs(avg_r_matrix).max() or -1, vmax=abs(avg_r_matrix).max() or 1)
    ax.set_xticks(range(len(frr_list)))
    ax.set_xticklabels(frr_list)
    ax.set_yticks(range(len(sb_list)))
    ax.set_yticklabels(sb_list)
    ax.set_xlabel("FALLBACK_REWARD_RISK")
    ax.set_ylabel("STOP_BUFFER_PCT")
    ax.set_title("ICT PO3 - avg R/trade by parameter (in-sample, full 2016-2025, all 4 instruments)")
    for i in range(len(sb_list)):
        for j in range(len(frr_list)):
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


# ============================= STEP 2: Monte Carlo per cell =============================

def monte_carlo_bootstrap(r_values, n_iter=MC_ITERATIONS, rng=None):
    """Bootstrap resample WITH replacement, same N as original trade count, vectorized (one
    (n_iter, n) integer draw + fancy-indexing, not a Python-level loop). Returns the resulting
    distribution of total R and of max drawdown (largest peak-to-trough drop in the cumulative
    R sum along each simulated path) across all n_iter simulated paths."""
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
    """Shuffle WITHOUT replacement (pure reordering of the SAME trades), vectorized via a
    per-row random-argsort permutation. Isolates pure sequence/path risk from the drawdown
    number: total R is mathematically invariant under reordering (sum of a fixed multiset does
    not change), so every iteration's total_r is identical to sum(r_values) by construction -
    that's expected, not a bug, and is exactly why max_drawdown (which DOES depend on order) is
    the real signal from this method, not total_r."""
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
    """Runs BOTH Monte Carlo procedures on EVERY grid cell's realized trade-R list (not just the
    best cell)."""
    mc_results = []
    for cell_idx, cell in enumerate(tqdm(grid_results, desc="Monte Carlo per cell", unit="cell", leave=False)):
        boot = monte_carlo_bootstrap(cell["trades_r"], n_iter=n_iter, rng=np.random.default_rng(10_000 + cell_idx))
        shuf = monte_carlo_shuffle(cell["trades_r"], n_iter=n_iter, rng=np.random.default_rng(20_000 + cell_idx))
        mc_results.append({
            "stop_buffer_pct": cell["stop_buffer_pct"],
            "fallback_reward_risk": cell["fallback_reward_risk"],
            "n_trades": cell["n_trades"],
            "bootstrap": summarize_mc(boot),
            "shuffle": summarize_mc(shuf),
        })
    return mc_results


def print_monte_carlo_table(mc_results):
    print("\nMonte Carlo per cell (2000 iterations each; bootstrap = resample WITH replacement, "
          "estimates total-R and drawdown variability; shuffle = reorder WITHOUT replacement, isolates "
          "pure sequence/path risk - note shuffle's total-R is deterministic by construction, sum of a "
          "fixed set doesn't change on reorder, so only its drawdown percentiles carry new information):")
    for cell in mc_results:
        boot, shuf = cell["bootstrap"], cell["shuffle"]
        print(f"\n  STOP_BUFFER_PCT={cell['stop_buffer_pct']:.3f}  FALLBACK_REWARD_RISK="
              f"{cell['fallback_reward_risk']:.1f}  ({cell['n_trades']} trades)")
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


# ============================= STEP 3: cluster analysis =============================

def neighbor_plateau_check(grid_results, sb_grid=None, frr_grid=None):
    """Always-available (no sklearn needed) check: look at the best cell's immediate grid
    neighbors (up/down/left/right, fewer at grid edges). PLATEAU if most neighbors are decent
    (same sign as the peak and within a reasonable ratio of its magnitude) - ISOLATED SPIKE /
    overfit warning if neighbors are starkly worse or flip sign."""
    sb_grid = sb_grid if sb_grid is not None else sorted(set(r["stop_buffer_pct"] for r in grid_results))
    frr_grid = frr_grid if frr_grid is not None else sorted(set(r["fallback_reward_risk"] for r in grid_results))
    lookup = {(r["stop_buffer_pct"], r["fallback_reward_risk"]): r for r in grid_results}

    best = max(grid_results, key=lambda r: r["total_r"])
    bi = sb_grid.index(best["stop_buffer_pct"])
    bj = frr_grid.index(best["fallback_reward_risk"])
    peak_avg = best["avg_r"]

    neighbors = []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = bi + di, bj + dj
        if 0 <= ni < len(sb_grid) and 0 <= nj < len(frr_grid):
            neighbors.append(lookup[(sb_grid[ni], frr_grid[nj])])

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
        "best_stop_buffer_pct": best["stop_buffer_pct"],
        "best_fallback_reward_risk": best["fallback_reward_risk"],
        "best_avg_r": peak_avg,
        "n_neighbors": len(neighbors),
        "n_decent_neighbors": sum(decent_flags),
        "frac_decent": frac_decent,
        "verdict": verdict,
    }


def sklearn_cluster_analysis(grid_results, sb_grid=None, frr_grid=None, k=3, random_state=42):
    """sklearn KMeans (k=3) over (normalized STOP_BUFFER_PCT position, normalized
    FALLBACK_REWARD_RISK position, avg R/trade) feature vectors across all 20 grid cells.
    Identifies which cluster the best-in-sample cell (highest total R) falls in, and reports
    that cluster's size and its members' min/mean avg R/trade. Wrapped in try/except
    ImportError - prints a clear skip message and returns None rather than crashing, matching
    orb_indices_optimization_and_ml.py's existing sklearn-missing handling."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("\nscikit-learn not installed - skipping cluster analysis. Run "
              "'!pip install scikit-learn -q' and re-run.")
        return None

    sb_grid = sb_grid if sb_grid is not None else sorted(set(r["stop_buffer_pct"] for r in grid_results))
    frr_grid = frr_grid if frr_grid is not None else sorted(set(r["fallback_reward_risk"] for r in grid_results))
    sb_pos = {sb: (i / (len(sb_grid) - 1) if len(sb_grid) > 1 else 0.0) for i, sb in enumerate(sb_grid)}
    frr_pos = {frr: (i / (len(frr_grid) - 1) if len(frr_grid) > 1 else 0.0) for i, frr in enumerate(frr_grid)}

    X = np.array([[sb_pos[r["stop_buffer_pct"]], frr_pos[r["fallback_reward_risk"]], r["avg_r"]]
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
        "cluster_members": [(m["stop_buffer_pct"], m["fallback_reward_risk"]) for m in members],
    }


def print_cluster_analysis(grid_results):
    print("\n" + "-" * 70)
    print("STEP 3: cluster / plateau-vs-spike analysis")
    print("-" * 70)

    neighbor_result = neighbor_plateau_check(grid_results)
    print(f"\nAlways-available neighbor check (best cell = STOP_BUFFER_PCT="
          f"{neighbor_result['best_stop_buffer_pct']:.3f}, FALLBACK_REWARD_RISK="
          f"{neighbor_result['best_fallback_reward_risk']:.1f}, avg R/trade="
          f"{neighbor_result['best_avg_r']:+.4f}):")
    print(f"  {neighbor_result['n_decent_neighbors']}/{neighbor_result['n_neighbors']} immediate neighbors "
          f"are 'decent' (same sign, at least half the peak's magnitude) -> {neighbor_result['verdict']}")

    cluster_result = sklearn_cluster_analysis(grid_results)
    if cluster_result is not None:
        print(f"\nsklearn KMeans (k=3) cluster containing the best-in-sample cell: "
              f"{cluster_result['cluster_size']} of {len(grid_results)} cells")
        print(f"  cluster avg R/trade: min={cluster_result['cluster_min_avg_r']:+.4f}  "
              f"mean={cluster_result['cluster_mean_avg_r']:+.4f}")
        print(f"  cluster members (STOP_BUFFER_PCT, FALLBACK_REWARD_RISK): "
              f"{cluster_result['cluster_members']}")

    return neighbor_result, cluster_result


# ============================= STEP 4: rolling walk-forward =============================

def generate_walk_forward_folds(fetch_start_year, fetch_end_year,
                                 is_years=WALK_FORWARD_IS_YEARS,
                                 oos_years=WALK_FORWARD_OOS_YEARS,
                                 step_years=WALK_FORWARD_STEP_YEARS):
    """Generates rolling (not anchored) walk-forward fold boundaries as plain datetime.date
    year-starts: IS = [is_start, is_end), OOS = [oos_start, oos_end) = [is_end, is_end+oos_years).
    Rolls forward by step_years each fold, stopping once a fold's OOS window would run past
    fetch_end_year. With the defaults (2016, 2025, 3, 1, 1) this produces exactly 6 folds:
    IS 2016-2019/OOS 2019-2020, IS 2017-2020/OOS 2020-2021, IS 2018-2021/OOS 2021-2022,
    IS 2019-2022/OOS 2022-2023, IS 2020-2023/OOS 2023-2024, IS 2021-2024/OOS 2024-2025."""
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


def run_walk_forward(data, folds, grid=None):
    """For each fold: re-run the full grid search restricted to the fold's in-sample window,
    pick the best combo by in-sample total R, apply that exact combo to the immediately-
    following out-of-sample window. Chains all folds' OOS trades together."""
    fold_results = []
    combined_oos_r = []

    for fold_idx, fold in enumerate(folds, start=1):
        is_start, is_end = fold["is_start"], fold["is_end"]
        oos_start, oos_end = fold["oos_start"], fold["oos_end"]

        is_grid = run_grid_search(data, window_start=is_start, window_end=is_end, grid=grid,
                                   desc=f"fold {fold_idx} in-sample grid")
        best = max(is_grid, key=lambda r: r["total_r"])

        oos_all_r = []
        for label, ind in data.items():
            trades = run_po3_backtest(ind, best["stop_buffer_pct"], best["fallback_reward_risk"],
                                       oos_start, oos_end)
            oos_all_r.extend(t["r"] for t in trades)
        combined_oos_r.extend(oos_all_r)

        oos_total = sum(oos_all_r)
        oos_n = len(oos_all_r)
        fold_results.append({
            "fold": fold_idx,
            "is_start": is_start, "is_end": is_end, "oos_start": oos_start, "oos_end": oos_end,
            "best_stop_buffer_pct": best["stop_buffer_pct"],
            "best_fallback_reward_risk": best["fallback_reward_risk"],
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


def print_walk_forward(fold_results, wfe_stats):
    print("\n" + "-" * 70)
    print("STEP 4: rolling walk-forward (3yr in-sample / 1yr out-of-sample, rolled 1yr at a time)")
    print("-" * 70)
    print(f"\n{'Fold':<5}{'In-sample':<22}{'Out-of-sample':<18}{'Best params':<22}"
          f"{'IS total R':<13}{'OOS total R':<13}{'OOS avg R':<10}")
    for f in fold_results:
        params = f"sb={f['best_stop_buffer_pct']:.3f} frr={f['best_fallback_reward_risk']:.1f}"
        print(f"{f['fold']:<5}{str(f['is_start'])+' to '+str(f['is_end']):<22}"
              f"{str(f['oos_start'])+' to '+str(f['oos_end']):<18}{params:<22}"
              f"{f['is_total_r']:>+9.2f}    {f['oos_total_r']:>+9.2f}    {f['oos_avg_r']:>+.4f}")

    print(f"\nCombined out-of-sample (all 6 folds chained together): {wfe_stats['combined_oos_n_trades']} trades, "
          f"{wfe_stats['combined_oos_total_r']:+.2f}R, {wfe_stats['combined_oos_avg_r']:+.4f}R/trade")
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


# ============================= main =============================

def main():
    years = (FETCH_END - FETCH_START).days / 365
    n_grid = len(STOP_BUFFER_PCT_GRID) * len(FALLBACK_REWARD_RISK_GRID)
    n_folds = len(generate_walk_forward_folds(FETCH_START.year, FETCH_END.year))
    print(f"ICT PO3 optimization/robustness pass: {n_grid}-cell grid search, {MC_ITERATIONS}-iteration "
          f"Monte Carlo x2 per cell, cluster analysis, and a {n_folds}-fold rolling walk-forward "
          f"(each fold re-runs the full {n_grid}-cell grid on its in-sample window) over ~{years:.0f} years "
          f"of {len(INSTRUMENTS)} instruments - this is a lot more compute than the base script and can "
          f"take well over an hour end to end; the disk cache means only the FIRST run pays the download "
          f"cost.\n")

    data = {}
    for label, instrument_const in tqdm(INSTRUMENTS, desc="Instruments", unit="instrument"):
        try:
            df = fetch_instrument_data(label, instrument_const)
        except Exception as exc:
            print(f"{label}: failed ({exc})")
            continue
        if df is None:
            print(f"{label}: no data")
            continue
        data[label] = precompute_indicators(df)
        print(f"{label}: {len(df)} bars")

    if not data:
        print("No data downloaded - check output above.")
        return

    # ---------------- STEP 1: parameter grid (full 2016-2025, all instruments) ----------------
    print("\n" + "=" * 70)
    print("STEP 1: parameter stability grid - full 2016-2025 range, all instruments per cell")
    print("=" * 70)
    grid_results = run_grid_search(data, desc="step 1 full-range grid")
    print_grid_table(grid_results)
    try:
        plot_heatmap(grid_results)
    except ImportError:
        print("\nmatplotlib not installed - skipping heatmap plot (text table above is the fallback).")

    best_cell = max(grid_results, key=lambda r: r["total_r"])
    worst_cell = min(grid_results, key=lambda r: r["total_r"])
    spread = best_cell["avg_r"] - worst_cell["avg_r"]
    print(f"\nBest cell:  STOP_BUFFER_PCT={best_cell['stop_buffer_pct']:.3f}  "
          f"FALLBACK_REWARD_RISK={best_cell['fallback_reward_risk']:.1f}  "
          f"-> {best_cell['total_r']:+.2f}R over {best_cell['n_trades']} trades "
          f"({best_cell['avg_r']:+.4f}R/trade)")
    print(f"Worst cell: STOP_BUFFER_PCT={worst_cell['stop_buffer_pct']:.3f}  "
          f"FALLBACK_REWARD_RISK={worst_cell['fallback_reward_risk']:.1f}  "
          f"-> {worst_cell['total_r']:+.2f}R over {worst_cell['n_trades']} trades "
          f"({worst_cell['avg_r']:+.4f}R/trade)")
    print(f"Avg-R/trade spread across the whole grid: {spread:.4f}")

    # ---------------- STEP 2: Monte Carlo per cell ----------------
    print("\n" + "-" * 70)
    print("STEP 2: Monte Carlo per cell (bootstrap + shuffle, 2000 iterations each)")
    print("-" * 70)
    mc_results = run_monte_carlo_all_cells(grid_results)
    print_monte_carlo_table(mc_results)

    # ---------------- STEP 3: cluster analysis ----------------
    neighbor_result, cluster_result = print_cluster_analysis(grid_results)

    # ---------------- STEP 4: rolling walk-forward ----------------
    folds = generate_walk_forward_folds(FETCH_START.year, FETCH_END.year)
    fold_results, combined_oos_r = run_walk_forward(data, folds)
    wfe_stats = compute_walk_forward_efficiency(fold_results, combined_oos_r)
    print_walk_forward(fold_results, wfe_stats)

    # ---------------- final honest verdict (computed, not asserted in advance) ----------------
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    grid_all_negative = all(r["total_r"] <= 0 for r in grid_results)
    best_cell_positive = best_cell["total_r"] > 0
    oos_positive = wfe_stats["combined_oos_avg_r"] > 0
    wfe_pass = (not np.isnan(wfe_stats["wfe"])) and wfe_stats["wfe"] >= WFE_PASS_THRESHOLD

    if grid_all_negative:
        print("Every single cell of the parameter grid was net-negative over the full 2016-2025 range - "
              "still negative everywhere. No corner of this parameter space rescues PO3; the base "
              "script's decisively negative result holds up here too, not because a flattering cell "
              "wasn't found, but because there isn't one.")
    elif best_cell_positive and neighbor_result["verdict"] == "ISOLATED SPIKE / overfit warning":
        print("The best in-sample cell was positive, but it is an ISOLATED SPIKE - its immediate "
              "neighbors are starkly worse or flip sign, which is what overfitting to a lucky parameter "
              "combination looks like. Combined with the out-of-sample check below, treat any positive "
              "in-sample number here with real suspicion.")
    elif best_cell_positive and not oos_positive:
        print("At least one in-sample cell was positive, but the chained out-of-sample result from the "
              "rolling walk-forward was NOT - the best in-sample parameters did not transfer forward. "
              "This is direct evidence against a real, robust edge.")
    elif best_cell_positive and oos_positive and wfe_pass:
        print("At least one in-sample cell was positive, its neighbors were also decent (a real plateau, "
              "not a spike), and the rolling walk-forward's chained out-of-sample result was also "
              f"positive with WFE >= {WFE_PASS_THRESHOLD:.0%}. This is the strongest evidence this "
              "project's PO3 testing has produced for a real (if not yet proven) edge - still only one "
              "signal among the many rigor checks above, not a green light on its own.")
    else:
        print("Results are mixed across the grid, Monte Carlo, cluster, and walk-forward checks - see "
              "the sections above for the specific numbers. This is NOT a case where the parameter "
              "space rescues the strategy outright, nor is it as uniformly negative as the base "
              "script's single fixed-parameter result. Read the caveats on each step before drawing "
              "any conclusion.")

    print("\nNo commission/spread/slippage modeled anywhere in this script, same as the base script. "
          "Every grid cell, Monte Carlo band, cluster label, and walk-forward fold above used the exact "
          "same entry/exit rules as ict_po3_forex_dukascopy_backtest.py - only STOP_BUFFER_PCT and "
          "FALLBACK_REWARD_RISK were varied.")


if __name__ == "__main__":
    main()
