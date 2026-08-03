# RANDOM ENTRY CONTROL - a deliberately edgeless strategy, wired into the webapp catalog so it can
# be run through the EXACT same pipeline as every real strategy: same instruments, same date range,
# same cost model, same holdout split, same leaderboard.
#
# WHY THIS IS THE MOST IMPORTANT SCRIPT IN THIS DIRECTORY
#
# A real Compare-All run of this catalog returned negative out-of-sample results for essentially
# every strategy - several at -98% or worse. The natural reading is "none of these strategies
# work". But there is a second reading that produces an identical-looking table: the post-hoc cost
# deduction is too harsh. Cost in R-terms is (cost_pct / 100) / stop_pct, so for a strategy with
# tight stops it is LARGE - on that run, several strategies' implied loss per trade (-0.02R to
# -0.18R) sat inside the same range as their own modelled cost per trade (0.011R to 0.22R). When
# the toll is the same size as the measured result, "no edge" and "toll too high" are not
# distinguishable from the result alone.
#
# A random-entry control separates them, because it has NO edge BY CONSTRUCTION:
#
#   - Random scores about the SAME as the real strategies  -> the leaderboard is mostly measuring
#     cost drag. The strategies aren't being shown to be bad; nothing is being shown at all, and
#     the cost model needs validating against real fills before any of it means anything.
#   - Random scores clearly WORSE than the real strategies  -> the strategies do carry real signal,
#     they're just not carrying enough of it to clear costs. Genuinely useful, and a completely
#     different conclusion.
#   - A strategy can't beat random                          -> that strategy specifically is dead,
#     independently of what the cost model says.
#
# None of those readings are available without this row in the table. That is the entire point:
# this is not a 17th strategy, it is the yardstick the other 16 are measured against.
#
# WHAT IT DOES: on each eligible bar, flip a coin, enter at that bar's close, place a symmetric
# 1:1 stop and target at ATR x ATR_MULT, then manage it forward with this project's standard
# conventions - stop checked before target on the same bar (pessimistic tie-break), timeout close
# at MAX_HOLD_BARS. Rules ported unchanged from random_baseline_forex_dukascopy_backtest.py, which
# has been in this repo since early on and was never wired into the app; the only changes here are
# the ones needed to satisfy the webapp's contract (the standard INSTRUMENTS/fetch_instrument_data/
# backtest_instrument shape, plus "stop_pct" and "date" on every trade so cost adjustment and the
# out-of-sample holdout split both actually apply - see webapp/test_strategy_contract.py).
#
# THE SEED IS FIXED (RANDOM_SEED below) and derived per-instrument, so the control is reproducible:
# re-running it on the same range gives the same trades, and it can't be re-rolled until it happens
# to look flattering or damning. That matters more for a control than for a strategy - a control
# you can reroll is not a control.
#
# HONEST NOTE ON WHAT RANDOM ENTRY IS AND ISN'T: a symmetric 1:1 coin flip has zero expectancy
# BEFORE costs, but it is not a perfect null - it still inherits the instrument's own drift and
# volatility clustering, and a timeout exit is not symmetric the way the stop/target pair is. It is
# a strong practical baseline, not a theorem. Read a small gap between it and a real strategy as
# "no clear evidence of edge", not as proof of equivalence.

# !pip install --upgrade dukascopy-python -q   # uncomment in Colab

import calendar
import datetime
import logging
import os
import pickle
import random as _random

import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

from tqdm.auto import tqdm


class _SuppressDukascopyInfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING


logging.getLogger("DUKASCRIPT").addFilter(_SuppressDukascopyInfoFilter())

CACHE_DIR = "/content/drive/MyDrive/dukascopy_cache" if os.path.isdir("/content/drive/MyDrive") else "dukascopy_cache"
FETCH_CHUNK_MONTHS = 3

# The same four instruments 13-14 of the 16 catalog strategies trade, so the control sits on
# exactly the same underlying price series as the things it is a control FOR.
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

ATR_LEN = 14
ATR_MULT = 2.0          # stop and target are both ATR x this -> a symmetric 1:1, zero-expectancy bet
MAX_HOLD_BARS = 500     # timeout, so a trade can't sit open forever in a quiet range
BARS_BETWEEN_ENTRIES = 12   # ~1 hour on 5-min bars. Keeps the trade count in the same order of
                             # magnitude as the busier real strategies without flooding the sample.
RANDOM_SEED = 20260803

COST_PCT_SCENARIOS = [0.0, 0.01, 0.03, 0.05]


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
    for chunk_start, chunk_end in tqdm(list(_month_chunks(FETCH_START, FETCH_END, FETCH_CHUNK_MONTHS)),
                                        desc=f"{label}: downloading {DUKASCOPY_INTERVAL} bars", unit="chunk"):
        chunk = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE,
                                        chunk_start, chunk_end)
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return None
    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    return df


def compute_atr_series(highs, lows, closes, length):
    """Wilder's ATR - identical to random_baseline_forex_dukascopy_backtest.py's and to the other
    scripts in this project that use ATR-scaled stops."""
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


def backtest_instrument(label, df):
    """Coin-flip entries with symmetric 1:1 ATR stops/targets. `label` seeds the RNG alongside
    RANDOM_SEED so each instrument gets its own reproducible stream (and so the control can't be
    quietly re-rolled until it tells a convenient story)."""
    highs, lows, closes = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
    times = df.index
    n = len(closes)
    atr = compute_atr_series(highs, lows, closes, ATR_LEN)
    rng = _random.Random(f"{RANDOM_SEED}-{label}")

    trades = []
    i = ATR_LEN + 1
    while i < n:
        current_atr = atr[i]
        if not current_atr or current_atr <= 0:
            i += 1
            continue

        side = "LONG" if rng.random() < 0.5 else "SHORT"
        entry = closes[i]
        risk = current_atr * ATR_MULT
        if risk <= 0 or entry <= 0:
            i += 1
            continue
        stop = entry - risk if side == "LONG" else entry + risk
        target = entry + risk if side == "LONG" else entry - risk
        entry_date = times[i].date()

        outcome, exit_r = None, None
        j = i + 1
        while j < n and j < i + MAX_HOLD_BARS:
            hi, lo = highs[j], lows[j]
            hit_stop = lo <= stop if side == "LONG" else hi >= stop
            hit_target = hi >= target if side == "LONG" else lo <= target
            # stop checked FIRST - this project's standard pessimistic same-bar tie-break, applied
            # here too so the control is penalised on ambiguous bars exactly like the real strategies
            if hit_stop:
                outcome, exit_r = "SL", -1.0
                break
            if hit_target:
                outcome, exit_r = "TP", 1.0
                break
            j += 1
        else:
            j = min(j, n - 1)

        if outcome is None:
            last_close = closes[j]
            pnl = (last_close - entry) if side == "LONG" else (entry - last_close)
            outcome, exit_r = "FLAT", pnl / risk

        trades.append({"side": side, "outcome": outcome, "r": exit_r, "date": entry_date,
                       "stop_pct": risk / entry})
        i = max(j + 1, i + BARS_BETWEEN_ENTRIES)

    return trades


def main():
    print(f"RANDOM ENTRY CONTROL - {FETCH_START.date()} to {FETCH_END.date()}\n"
          f"This is a yardstick, not a strategy: coin-flip entries with symmetric 1:1 ATR stops.\n"
          f"Compare its cost-adjusted result against the real strategies' - if they can't beat it, "
          f"the difference between 'no edge' and 'costs too high' is what you're actually looking at.\n")

    all_trades = []
    for label, const in tqdm(INSTRUMENTS, desc="Instruments", unit="instrument"):
        try:
            df = fetch_instrument_data(label, const)
        except Exception as exc:
            print(f"{label}: failed ({exc})")
            continue
        if df is None:
            print(f"{label}: no data")
            continue
        for t in backtest_instrument(label, df):
            t["instrument"] = label
            all_trades.append(t)

    if not all_trades:
        print("No trades produced - check the output above.")
        return

    n = len(all_trades)
    total_r = sum(t["r"] for t in all_trades)
    print(f"\n{n} trades, {total_r:+.2f}R total, {total_r / n:+.4f}R/trade (BEFORE costs)")
    print("\nCOST SENSITIVITY (the whole point - watch where a zero-edge system lands):")
    for cost_pct in COST_PCT_SCENARIOS:
        adj = sum(t["r"] - (cost_pct / 100.0) / t["stop_pct"] for t in all_trades)
        print(f"  {cost_pct:.2f}% round-trip: {adj:+.2f}R total, {adj / n:+.4f}R/trade")
    print("\nA symmetric 1:1 coin flip has ~zero expectancy before costs. Whatever it loses at a "
          "given cost level is the toll, not a verdict on any strategy - so any real strategy "
          "scoring near this line has not been shown to have an edge, and one scoring far below it "
          "is genuinely worse than chance.")


if __name__ == "__main__":
    main()
