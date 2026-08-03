# Dow Theory Swing Structure - rigorous 4-step optimization/robustness pass, built on TOP of
# dow_theory_swing_structure_dukascopy_backtest.py (read that file first - this does not replace
# it or re-derive the strategy rules; it reuses the same pivot/trend-state logic and asks a
# narrower question: is there any SWING_LEN (the base script's one structural parameter, deliberately
# NOT grid-searched there) with a real, robust edge, or does whatever the base script's fixed
# SWING_LEN=20 shows hold up across the whole plausible range?
#
# HONEST SCOPE NOTE UP FRONT: unlike ict_po3_forex_dukascopy_optimization.py and day_trading_rauf_
# dukascopy_optimization.py's headers, this one does NOT quote an "already-established base result"
# from a real run - this script was built and verified against SYNTHETIC data only (see
# run_smoke_test() and the unit tests below), the same way this project's OTHER unit tests do,
# because the environment this was written in has no live Dukascopy network access. The real
# verdict - what SWING_LEN actually looks like against real 2016-2025 EURUSD/GBPUSD/USDJPY/XAUUSD
# data - is only known once this is actually run (via main(), or the webapp's "Run Deep
# Optimization" button, both of which need real Dukascopy access this environment doesn't have).
# Whatever it prints when actually run is the real answer - nothing here is pre-decided, but
# nothing here has been pre-checked against real data either, and that's stated plainly rather
# than implied otherwise.
#
# THE 4 STEPS (methodology fixed in advance, not iterated on after seeing results):
#   1. PARAMETER STABILITY GRID: SWING_LEN in SWING_LEN_GRID (one dimension, not two - this
#      strategy has exactly one structural parameter), full FETCH_START..FETCH_END range minus
#      the lockbox window, all 4 instruments per value. Text table always printed.
#   2. MONTE CARLO PER CELL: every cell (not just the best), bootstrap-with-replacement AND
#      shuffle-without-replacement over that cell's actual realized trade R-multiples, via the
#      shared research/optimization_engine.py functions (not re-implemented here - see that
#      module's own "WALK-FORWARD FOLDS, MONTE CARLO, AND N-DIMENSIONAL CLUSTER/PLATEAU CHECK"
#      section header for why these are centralized rather than copy-pasted per script).
#   3. CLUSTER ANALYSIS: sklearn KMeans (k=3, or fewer if the grid is smaller) over
#      (normalized SWING_LEN position, avg R/trade) feature vectors via optimization_engine.
#      run_cluster_analysis, plus optimization_engine.neighbor_plateau_check's always-available
#      (no sklearn) direct 1-D neighbor check (a SWING_LEN grid point has up to 2 neighbors -
#      one smaller, one larger - unlike PO3/Rauf's 2-parameter up/down/left/right).
#   4. ROLLING WALK-FORWARD: WF_IS_YEARS in-sample / WF_OOS_YEARS out-of-sample, rolled forward
#      WF_STEP_YEARS at a time across FETCH_START..search_end (search_end excludes the lockbox
#      window - see LOCKBOX_MONTHS below), via optimization_engine.walk_forward_folds. Re-runs
#      the full grid on each fold's in-sample window only, picks the in-sample winner, scores it
#      out-of-sample, chains all folds' OOS trades together, reports Walk-Forward Efficiency.
#      HONEST CAVEAT SPECIFIC TO THIS STRATEGY: the base script's own header says a 20-day-each-
#      side fractal pivot produces "a modest handful to a few dozen trades" across a SINGLE
#      instrument's full 9-year history. Splitting that into 3yr-IS/1yr-OOS rolling folds means
#      several folds - especially the 1-year OOS windows - will plausibly see zero or one trade
#      per instrument. Walk-Forward Efficiency on a near-empty OOS window is not meaningful
#      evidence either way, and this script's own STEP 4 output says so explicitly rather than
#      treating a 0-or-1-trade fold as a real pass or fail - this is the SAME honesty stance the
#      base script already takes about its own low trade count, carried through to this pass.
#
# WINDOWING CONVENTION: unlike day_trading_rauf_dukascopy_backtest.py/ict_po3_forex_dukascopy_
# backtest.py (whose base backtest_instrument already accepts window_start/window_end), the base
# Dow Theory script's backtest_instrument(label, df) takes no window - it was written with "NO
# PARAMETER GRID SEARCH" as an explicit design choice (see its own header). Rather than modify
# that already-shipped, tested file, this script defines its own windowed rebuild of the exact
# same pivot/trend-state logic below (run_dow_theory_backtest) - the SAME choice ict_po3_forex_
# dukascopy_optimization.py made for its own run_po3_backtest, for the identical reason: never
# risk the base script's existing behavior/tests. A window restricts which DAILY bars get
# PROCESSED (pivot detection is recomputed fresh from only the bars inside [window_start,
# window_end) - not carried over from outside it), matching "each fold sees only its own window"
# walk-forward discipline, not "full-history state, entries gated by date" (which would let a
# fold benefit from pivot context only knowable using data outside that exact fold).
#
# SWING_LEN OVERRIDE CONVENTION: SWING_LEN is applied via setattr(base_module, "SWING_LEN", value)
# before calling the base module's find_confirmed_swing_pivots - the exact same "override a
# module-level constant by attribute" convention webapp/registry.py's ParamSpec/_apply_overrides
# already uses project-wide for every strategy in this catalog (confirmed by reading registry.py
# directly), not a new mechanism invented for this script.

# !pip install --upgrade dukascopy-python scikit-learn -q   # uncomment in Colab

import calendar
import datetime
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optimization_engine as opt_engine

from tqdm.auto import tqdm

import dow_theory_swing_structure_dukascopy_backtest as base


class _SuppressDukascopyInfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING


logging.getLogger("DUKASCRIPT").addFilter(_SuppressDukascopyInfoFilter())

CACHE_DIR = "/content/drive/MyDrive/dukascopy_cache" if os.path.isdir("/content/drive/MyDrive") else "dukascopy_cache"
FETCH_CHUNK_MONTHS = 3

INSTRUMENTS = base.INSTRUMENTS
FETCH_START = datetime.datetime(2016, 1, 1)
FETCH_END = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_MIN_5
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

# STEP 1 grid - one dimension (this strategy's only structural parameter). Centered on the base
# script's own default (20), spanning from a much more selective pivot (50 days each side) down
# to a much looser one (10) - within registry.py's own ParamSpec bounds [5, 60].
SWING_LEN_GRID = [10, 15, 20, 25, 30, 40, 50]

SEARCH_METHOD = "grid"   # 7 combos total - exhaustive grid is the right choice at this size, same
                          # reasoning as PO3/Rauf's own SEARCH_METHOD default (see optimization_
                          # engine.py's header on when Bayesian/genetic search actually earn their
                          # keep - not here, today).
OBJECTIVE = "total_r"

MC_ITERATIONS = 2000

WF_IS_YEARS = 3
WF_OOS_YEARS = 1
WF_STEP_YEARS = 1

# --- lockbox (research/optimization_engine.py: split_lockbox/lockbox_confirm) ---
# The final LOCKBOX_MONTHS of FETCH_START..FETCH_END are carved off BEFORE any search runs and
# are NEVER touched by STEP 1's grid search or any STEP 4 fold (both bounded by `search_end`, not
# FETCH_END - see main()) - structurally stricter than the walk-forward OOS folds above, which are
# re-touched every time this script re-runs. See optimization_engine.py's own lockbox section
# header for the full reasoning.
LOCKBOX_MONTHS = 12
LOCKBOX_LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lockbox_ledger.json")
LOCKBOX_STRATEGY_ID = "dow_theory_swing_structure"

HEATMAP_PNG_PATH = "dow_theory_swing_structure_optimization_heatmap.png"


def _month_chunks(start, end, months_per_chunk):
    chunk_start = start
    while chunk_start < end:
        month_index = chunk_start.month - 1 + months_per_chunk
        target_year = chunk_start.year + month_index // 12
        target_month = month_index % 12 + 1
        target_day = min(chunk_start.day, calendar.monthrange(target_year, target_month)[1])
        chunk_end = chunk_start.replace(year=target_year, month=target_month, day=target_day)
        yield chunk_start, min(chunk_end, end)
        chunk_start = chunk_end


def fetch_instrument_data(label, instrument_const):
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
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df.index = base.to_ny_time(df.index)

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    return df


def run_dow_theory_backtest(df, swing_len, window_start=None, window_end=None):
    """Windowed rebuild of base.backtest_instrument, parameterized by swing_len - see module
    header's WINDOWING CONVENTION section for why this exists as a separate function rather than
    modifying the base script. `df` is the FULL, un-truncated 5-min bar dataframe for one
    instrument; window_start/window_end (datetime.date or None) restrict which DAILY bars get
    resampled and processed - both signal computation (pivot/trend state, recomputed fresh from
    only the in-window daily bars) and trading. Returns a list of trade dicts, same shape as
    base.backtest_instrument's."""
    daily = base.resample_daily(df)
    if window_start is not None:
        daily = daily[daily.index.date >= window_start]
    if window_end is not None:
        daily = daily[daily.index.date < window_end]
    if daily.empty:
        return []

    original_swing_len = base.SWING_LEN
    base.SWING_LEN = swing_len   # see module header's SWING_LEN OVERRIDE CONVENTION
    try:
        daily = base.compute_daily_swing_signals(daily)
    finally:
        base.SWING_LEN = original_swing_len

    day_map = {
        ts.date(): {
            "fresh_up_act": bool(row["fresh_up_act"]),
            "fresh_down_act": bool(row["fresh_down_act"]),
            "stop_long_effective": row["trail_stop_long_effective"],
            "stop_short_effective": row["trail_stop_short_effective"],
        }
        for ts, row in daily.iterrows()
    }
    if window_start is not None:
        df = df[df.index.date >= window_start]
    if window_end is not None:
        df = df[df.index.date < window_end]
    if df.empty:
        return []

    highs = df["High"].tolist()
    lows = df["Low"].tolist()
    closes = df["Close"].tolist()
    times = df.index
    n = len(closes)

    trades = []
    position = None

    for i in range(n):
        day = times[i].date()
        sig = day_map.get(day)
        if sig is None:
            continue

        if position is not None:
            side = position["side"]
            hi, lo = highs[i], lows[i]
            if side == "LONG":
                eff_stop = sig["stop_long_effective"]
                if eff_stop is not None and not pd.isna(eff_stop):
                    position["stop"] = max(position["stop"], eff_stop)
                if lo <= position["stop"]:
                    r = (position["stop"] - position["entry"]) / position["sl_distance"]
                    trades.append({"side": side, "outcome": "STOP", "r": r, "date": position["entry_date"],
                                   "stop_pct": position["sl_distance"] / position["entry"]})
                    position = None
            else:
                eff_stop = sig["stop_short_effective"]
                if eff_stop is not None and not pd.isna(eff_stop):
                    position["stop"] = min(position["stop"], eff_stop)
                if hi >= position["stop"]:
                    r = (position["entry"] - position["stop"]) / position["sl_distance"]
                    trades.append({"side": side, "outcome": "STOP", "r": r, "date": position["entry_date"],
                                   "stop_pct": position["sl_distance"] / position["entry"]})
                    position = None
            continue

        is_first_bar_of_day = (i == 0) or (times[i - 1].date() != day)
        if not is_first_bar_of_day:
            continue

        if sig["fresh_up_act"]:
            entry = closes[i]
            stop = sig["stop_long_effective"]
            if stop is not None and not pd.isna(stop) and stop < entry:
                sl_distance = entry - stop
                position = {"side": "LONG", "entry": entry, "stop": stop,
                            "sl_distance": sl_distance, "entry_date": day}
        elif sig["fresh_down_act"]:
            entry = closes[i]
            stop = sig["stop_short_effective"]
            if stop is not None and not pd.isna(stop) and stop > entry:
                sl_distance = stop - entry
                position = {"side": "SHORT", "entry": entry, "stop": stop,
                            "sl_distance": sl_distance, "entry_date": day}

    if position is not None:
        side = position["side"]
        last_close = closes[-1]
        pnl = (last_close - position["entry"]) if side == "LONG" else (position["entry"] - last_close)
        trades.append({"side": side, "outcome": "FLAT", "r": pnl / position["sl_distance"],
                        "date": position["entry_date"], "stop_pct": position["sl_distance"] / position["entry"]})

    return trades


def _make_eval_fn(all_dfs, window_start=None, window_end=None):
    """optimization_engine-compatible eval_fn(params) -> list_of_trade_dicts: runs
    run_dow_theory_backtest across every instrument in `all_dfs`, restricted to
    [window_start, window_end), tagging each trade with its instrument label."""
    def eval_fn(params):
        trades = []
        for label, df in all_dfs.items():
            instrument_trades = run_dow_theory_backtest(df, params["SWING_LEN"], window_start, window_end)
            for t in instrument_trades:
                t = dict(t)
                t["instrument"] = label
                trades.append(t)
        return trades
    return eval_fn


def run_param_search(all_dfs, swing_len_grid, window_start=None, window_end=None, method=None,
                      objective=None, show_progress=False, desc="search"):
    method = method or SEARCH_METHOD
    objective = objective or OBJECTIVE
    eval_fn = _make_eval_fn(all_dfs, window_start, window_end)
    objective_fn = opt_engine.get_objective(objective)
    years = None
    if window_start is not None and window_end is not None:
        years = (window_end - window_start).days / 365.0
    param_grid = {"SWING_LEN": list(swing_len_grid)}
    return opt_engine.run_search(method, param_grid, eval_fn, objective_fn, years=years,
                                  show_progress=show_progress, desc=desc)


def print_grid_text_table(search_result, title):
    print(f"\n{title}")
    print(f"{'SWING_LEN':>10s}  {'trades':>7s}  {'total_r':>10s}  {'avg_r':>9s}")
    for entry in sorted(search_result["all"], key=lambda e: e["params"]["SWING_LEN"]):
        trades = entry["trades"]
        n = len(trades)
        tr = opt_engine.total_r(trades)
        ar = opt_engine.avg_r(trades)
        print(f"{entry['params']['SWING_LEN']:>10d}  {n:>7d}  {tr:>+10.2f}  {ar:>+9.4f}")


def make_lockbox_backtest_fn(all_dfs, swing_len):
    """Builds an optimization_engine.lockbox_confirm-compatible backtest_fn(lockbox_start,
    lockbox_end) -> trades, running run_dow_theory_backtest across every instrument in `all_dfs`
    with FIXED, already-selected swing_len - restricted to [lockbox_start, lockbox_end), the one
    and only time those exact bars are ever touched by this script."""
    def backtest_fn(lockbox_start, lockbox_end):
        ls = lockbox_start.date() if hasattr(lockbox_start, "date") else lockbox_start
        le = lockbox_end.date() if hasattr(lockbox_end, "date") else lockbox_end
        trades = []
        for label, df in all_dfs.items():
            for t in run_dow_theory_backtest(df, swing_len, ls, le):
                t = dict(t)
                t["instrument"] = label
                trades.append(t)
        return trades
    return backtest_fn


def run_full_pipeline(all_dfs, swing_len_grid, fetch_start, fetch_end, mc_iterations=MC_ITERATIONS,
                       show_progress=True, verbose=True):
    """Runs STEPs 1-4 against whatever data/grid/date-range it's given - used both by main() (real
    data, real range) and the smoke test (tiny synthetic data, tiny range). fetch_start/fetch_end
    here are this call's OWN search bounds (already excluding the lockbox window when called from
    main() - see main()'s own lockbox split), not necessarily this module's FETCH_START/FETCH_END
    constants."""
    results = {}

    # ---- STEP 1 ----
    if verbose:
        print(f"STEP 1: {len(swing_len_grid)}-value SWING_LEN search ({fetch_start} to {fetch_end})")
    search_result = run_param_search(all_dfs, swing_len_grid, window_start=fetch_start, window_end=fetch_end,
                                       show_progress=show_progress, desc="STEP 1 search")
    results["step1_search"] = search_result
    if search_result["best"] is None or not search_result["best"]["trades"]:
        if verbose:
            print("STEP 1: no trades produced by any SWING_LEN value in this window - nothing further to do.")
        results["verdict"] = "NO TRADES"
        return results
    if verbose:
        print_grid_text_table(search_result, "STEP 1 grid")
        best = search_result["best"]
        print(f"Best by {OBJECTIVE}: SWING_LEN={best['params']['SWING_LEN']}, "
              f"{len(best['trades'])} trades, total_r={opt_engine.total_r(best['trades']):+.2f}")

    # ---- STEP 2 ----
    if verbose:
        print(f"\nSTEP 2: Monte Carlo per grid cell ({mc_iterations} iterations each, bootstrap + shuffle)")
    mc_by_swing_len = {}
    for entry in search_result["all"]:
        r_values = [t["r"] for t in entry["trades"]]
        mc_by_swing_len[entry["params"]["SWING_LEN"]] = {
            "bootstrap": opt_engine.monte_carlo_bootstrap(r_values, n_iter=mc_iterations),
            "shuffle": opt_engine.monte_carlo_shuffle(r_values, n_iter=mc_iterations),
            "n_trades": len(r_values),
        }
    results["step2_monte_carlo"] = mc_by_swing_len
    if verbose:
        for swing_len in sorted(mc_by_swing_len):
            mc = mc_by_swing_len[swing_len]
            if mc["n_trades"] == 0:
                print(f"  SWING_LEN={swing_len:>3d}: 0 trades - Monte Carlo not meaningful")
                continue
            b = mc["bootstrap"]
            print(f"  SWING_LEN={swing_len:>3d} ({mc['n_trades']:>3d} trades): bootstrap total_r "
                  f"P05={b['total_r_p05']:+.2f} P50={b['total_r_p50']:+.2f} P95={b['total_r_p95']:+.2f}, "
                  f"P(total_r<=0)={b['p_total_r_leq_0']:.2f}")

    # ---- STEP 3 ----
    if verbose:
        print("\nSTEP 3: cluster analysis + neighbor-plateau check")
    param_grid = {"SWING_LEN": list(swing_len_grid)}
    cluster_result = opt_engine.run_cluster_analysis(search_result["all"], param_grid, k=min(3, len(swing_len_grid)))
    plateau_result = opt_engine.neighbor_plateau_check(search_result["all"], param_grid)
    results["step3_cluster"] = cluster_result
    results["step3_plateau"] = plateau_result
    if verbose:
        if cluster_result is not None:
            print(f"  Best cell's cluster: {cluster_result['best_cluster_size']} members, "
                  f"mean avg_r={cluster_result['best_cluster_mean_avg_r']:+.4f}, "
                  f"min avg_r in cluster={cluster_result['best_cluster_min_avg_r']:+.4f}")
        print(f"  Neighbor-plateau verdict: {plateau_result['verdict']} "
              f"(best SWING_LEN={plateau_result['best_params']['SWING_LEN']}, "
              f"{len(plateau_result['neighbors'])} in-grid neighbor(s) checked)")

    # ---- STEP 4 ----
    if verbose:
        print(f"\nSTEP 4: rolling walk-forward ({WF_IS_YEARS}yr IS / {WF_OOS_YEARS}yr OOS, step {WF_STEP_YEARS}yr)")
    folds = list(opt_engine.walk_forward_folds(fetch_start, fetch_end, WF_IS_YEARS, WF_OOS_YEARS, WF_STEP_YEARS))
    fold_results = []
    combined_oos_trades = []
    for is_start, is_end, oos_start, oos_end in folds:
        is_search = run_param_search(all_dfs, swing_len_grid, window_start=is_start, window_end=is_end,
                                       desc=f"fold {is_start}-{is_end} IS search")
        if is_search["best"] is None or not is_search["best"]["trades"]:
            fold_results.append({"is_start": is_start, "is_end": is_end, "oos_start": oos_start,
                                  "oos_end": oos_end, "chosen_swing_len": None, "is_avg_r": None,
                                  "oos_trades": [], "note": "no in-sample trades - fold skipped"})
            continue
        chosen_swing_len = is_search["best"]["params"]["SWING_LEN"]
        is_avg_r = opt_engine.avg_r(is_search["best"]["trades"])
        oos_trades = []
        for label, df in all_dfs.items():
            for t in run_dow_theory_backtest(df, chosen_swing_len, oos_start, oos_end):
                t = dict(t)
                t["instrument"] = label
                oos_trades.append(t)
        fold_results.append({"is_start": is_start, "is_end": is_end, "oos_start": oos_start,
                              "oos_end": oos_end, "chosen_swing_len": chosen_swing_len,
                              "is_avg_r": is_avg_r, "oos_trades": oos_trades})
        combined_oos_trades.extend(oos_trades)
    results["step4_folds"] = fold_results
    results["step4_combined_oos_trades"] = combined_oos_trades

    scored_folds = [f for f in fold_results if f["chosen_swing_len"] is not None]
    thin_folds = [f for f in scored_folds if len(f["oos_trades"]) < 5]
    if scored_folds and combined_oos_trades:
        combined_oos_avg_r = opt_engine.avg_r(combined_oos_trades)
        mean_is_avg_r = float(np.mean([f["is_avg_r"] for f in scored_folds]))
        wfe = (combined_oos_avg_r / mean_is_avg_r) if mean_is_avg_r not in (0, None) else float("nan")
        results["step4_wfe"] = wfe
        results["step4_verdict"] = "PASS" if (not np.isnan(wfe) and wfe >= 0.5) else "FAIL"
    else:
        results["step4_wfe"] = None
        results["step4_verdict"] = "NO SCORABLE FOLDS"

    if verbose:
        for f in fold_results:
            if f["chosen_swing_len"] is None:
                print(f"  Fold {f['is_start']}-{f['is_end']} IS / {f['oos_start']}-{f['oos_end']} OOS: "
                      f"{f['note']}")
                continue
            flag = " (THIN - not meaningful evidence)" if len(f["oos_trades"]) < 5 else ""
            print(f"  Fold {f['is_start']}-{f['is_end']} IS / {f['oos_start']}-{f['oos_end']} OOS: "
                  f"chose SWING_LEN={f['chosen_swing_len']}, IS avg_r={f['is_avg_r']:+.4f}, "
                  f"{len(f['oos_trades'])} OOS trades{flag}")
        if thin_folds:
            print(f"  CAVEAT: {len(thin_folds)}/{len(scored_folds)} scored fold(s) had <5 OOS trades - this "
                  f"strategy's own base script warns it's a naturally selective, low-frequency filter (a "
                  f"'modest handful to a few dozen trades' across a SINGLE instrument's full 9-year history), "
                  f"so several folds seeing 0-4 trades is expected, not a bug. Walk-Forward Efficiency computed "
                  f"from mostly-empty OOS windows is not meaningful evidence either way - read step4_wfe/verdict "
                  f"below with that firmly in mind, not as a clean pass/fail the way a higher-frequency "
                  f"strategy's walk-forward would be.")
        if results["step4_wfe"] is not None and not np.isnan(results["step4_wfe"]):
            print(f"  Walk-Forward Efficiency: {results['step4_wfe']:.2f} -> {results['step4_verdict']} "
                  f"(>=0.5 rule-of-thumb threshold, not proof either way)")
        else:
            print(f"  Walk-Forward Efficiency: {results['step4_verdict']}")

    results["verdict"] = "SCORED"
    return results


def main():
    print(f"Downloading {len(INSTRUMENTS)} instruments from Dukascopy over ~"
          f"{(FETCH_END - FETCH_START).days / 365:.0f} years ({FETCH_START.date()} to {FETCH_END.date()}) - "
          f"cached to disk after the first run.\n")

    all_dfs = {}
    for label, instrument_const in tqdm(INSTRUMENTS, desc="Instruments", unit="instrument"):
        try:
            df = fetch_instrument_data(label, instrument_const)
        except Exception as exc:
            print(f"{label}: failed ({exc})")
            continue
        if df is None:
            print(f"{label}: no data")
            continue
        all_dfs[label] = df
        print(f"{label}: {len(df)} bars")

    if not all_dfs:
        print("No data downloaded - check output above.")
        return

    search_start, search_end, lockbox_start, lockbox_end = opt_engine.split_lockbox(
        FETCH_START, FETCH_END, LOCKBOX_MONTHS)
    print(f"\nLockbox window {lockbox_start} to {lockbox_end} carved off BEFORE any search - held out "
          f"from STEP 1's search window and every STEP 4 fold below (both now bounded by {search_end}, "
          f"not {FETCH_END.date()}).")

    results = run_full_pipeline(all_dfs, SWING_LEN_GRID, search_start.date(), search_end.date())

    if results["verdict"] == "SCORED":
        best_swing_len = results["step1_search"]["best"]["params"]["SWING_LEN"]
        print(f"\nConfirming STEP 1's winning SWING_LEN={best_swing_len} against the lockbox window "
              f"{lockbox_start.date()} to {lockbox_end.date()} - never touched by STEP 1-4 above, "
              f"and scoreable AT MOST ONCE EVER for this strategy_id (see optimization_engine.py's "
              f"lockbox section header).")
        backtest_fn = make_lockbox_backtest_fn(all_dfs, best_swing_len)
        try:
            lockbox_result = opt_engine.lockbox_confirm(
                LOCKBOX_STRATEGY_ID, {"SWING_LEN": best_swing_len}, backtest_fn,
                lockbox_start, lockbox_end, ledger_path=LOCKBOX_LEDGER_PATH)
            verdict = "PASSED" if lockbox_result["passed"] else "DID NOT PASS"
            print(f"Lockbox confirmation complete - {verdict}. {lockbox_result['n_trades']} trades, "
                  f"total_r={lockbox_result['total_r']:+.2f}, avg_r={lockbox_result['avg_r']:+.4f}.")
        except opt_engine.LockboxAlreadyUsedError as exc:
            print(f"Lockbox already used: {exc}")


# =============================================================================================
# SMOKE TEST - tiny synthetic data, tiny grid/range, proves the pipeline runs end to end without
# a real Dukascopy connection. This is NOT a substitute for a real run (see module header's
# HONEST SCOPE NOTE) - it only proves the plumbing is correct, the same role run_smoke_test()
# plays in day_trading_rauf_dukascopy_optimization.py.
# =============================================================================================

_FLAT_DAY = (1.1000, 1.1002, 1.0998, 1.1000)


def _flat_days(n, start="2016-01-04"):
    days = pd.bdate_range(start=start, periods=n)
    return [(d, _FLAT_DAY) for d in days]


def _override_day(days_list, index, bars):
    new_list = list(days_list)
    d, _ = new_list[index]
    new_list[index] = (d, bars)
    return new_list


def _build_5min_df(day_specs):
    """One 5-min bar per day, at 09:30 NY time - matches the base script's own test file
    convention (_build_5min_df) closely enough to guarantee a clean resample_daily() round trip
    without needing to import that test file's private helpers directly."""
    rows, idx = [], []
    for date_, (o, h, l, c) in day_specs:
        ts = pd.Timestamp(date_.year, date_.month, date_.day, 9, 30, tz="America/New_York")
        idx.append(ts)
        rows.append((o, h, l, c))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=pd.DatetimeIndex(idx))


def _build_uptrend_df(n_total=150, swing_len=20):
    """Deliberately engineered synthetic price series (same technique as the base script's own
    test file - flat days with a handful of overridden pivot days, spaced swing_len apart) that
    reliably produces a fresh uptrend transition (HH+HL) at SWING_LEN=swing_len, so the smoke test
    proves this script's windowing/override/eval_fn wiring actually generates a real trade rather
    than merely "runs without crashing on data that happens to produce nothing"."""
    days = _flat_days(n_total)
    pivot_low1_idx, low1 = 25, 1.0900
    pivot_high1_idx, high1 = 30, 1.1100
    pivot_low2_idx, low2 = 60, 1.0950     # a HIGHER low than low1
    pivot_high2_idx, high2 = 65, 1.1150   # a HIGHER high than high1
    days = _override_day(days, pivot_low1_idx, (1.1000, 1.1002, low1, 1.0950))
    days = _override_day(days, pivot_high1_idx, (1.1000, high1, 1.0998, 1.1050))
    days = _override_day(days, pivot_low2_idx, (1.1000, 1.1002, low2, 1.0980))
    days = _override_day(days, pivot_high2_idx, (1.1000, high2, 1.0998, 1.1100))
    # continued upward drift after the fresh-uptrend trigger (action day = pivot_high2_idx +
    # swing_len + 1) so the eventual forced-close-at-data-end trade reflects genuine unrealized
    # profit, not a degenerate r=0.0 - a more informative (not just non-crashing) smoke test.
    days = _override_day(days, n_total - 1, (1.1180, 1.1220, 1.1170, 1.1200))
    return _build_5min_df(days)


def run_smoke_test(verbose=True):
    df = _build_uptrend_df(n_total=150, swing_len=20)
    all_dfs = {"SYN": df}
    results = run_full_pipeline(all_dfs, [10, 15, 20, 25], datetime.date(2016, 1, 1), datetime.date(2016, 12, 1),
                                  mc_iterations=200, show_progress=False, verbose=verbose)
    assert results["verdict"] == "SCORED", "smoke test's engineered uptrend fixture should always produce trades"
    if verbose:
        print("\nSmoke test completed without error and produced real trades, as expected.")

    # A second pass over a range wide enough to fit at least one real walk-forward fold (the
    # first pass's range is too short for STEP 4 to have anything to score - "NO SCORABLE FOLDS"
    # above is the correct, honest response to that, not a bug, but it means STEP 4's actual
    # fold-scoring code path - including the "thin fold" caveat this strategy's own low trade
    # count makes routine - is only exercised here, not above).
    long_df = _build_uptrend_df(n_total=1050, swing_len=20)
    long_results = run_full_pipeline({"SYN": long_df}, [10, 20, 30], datetime.date(2016, 1, 1),
                                       datetime.date(2020, 1, 1), mc_iterations=200, show_progress=False,
                                       verbose=verbose)
    assert long_results["verdict"] == "SCORED"
    assert len(long_results["step4_folds"]) >= 1, "a 4-year range should fit at least one 3yr-IS/1yr-OOS fold"
    if verbose:
        print("\nWide-range smoke test completed without error and exercised STEP 4's fold scoring.")
    return results, long_results


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        run_smoke_test()
    else:
        main()
