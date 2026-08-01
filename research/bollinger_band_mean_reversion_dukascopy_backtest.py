# Bollinger Band mean-reversion "fade" backtest via Dukascopy, 2016-2025.
#
# Distinct from every other script in this project's ICT-adjacent lineup
# (PO3, Silver Bullet, Day Trading Rauf) - no session windows, no market
# structure, no sweep-then-confirmation chain. This is the classic,
# textbook mean-reversion idea: price stretches outside a volatility band,
# then trades back toward the average. Intraday, 5-min bars, runs
# continuously all day (not tied to a session), unlike the swing/daily-bar
# systems built alongside this one.
#
# OPERATIONALIZING "Bollinger Band fade" precisely:
#   1. BANDS: SMA(20) of Close as the middle band, +/- 2x rolling stddev(20)
#      of Close (pandas default sample stddev, ddof=1 - noted here since
#      some charting platforms default to population stddev instead; the
#      difference is small at length 20 and doesn't change the qualitative
#      result) for the upper/lower bands. Computed directly on the raw
#      5-min series - no daily resample, no session gating.
#   2. ENTRY - a CONFIRMED RE-ENTRY, not a raw band touch. This is the
#      single most important design choice here, and it exists specifically
#      to avoid the false-positive trap this project's other band/level-
#      touch strategies (PO3, S/R zone bounce) have already had to guard
#      against: a bar merely POKING outside the band and immediately
#      reversing on the very next tick is exactly the kind of noise a
#      single-step synthetic-bar model would over-reward (see the
#      granularity-convergence check in this script's unit tests). Requiring
#      bar i-1's CLOSE below the lower band, then bar i's CLOSE back >= the
#      lower band, means the "outside" excursion is a genuine two-bar
#      structure, not a single wick. Mirror for the upper band -> SHORT.
#   3. STOP - the LOW of bar i-1 (the last bar that closed outside the
#      band; the low, not the close, since that's the actual excursion
#      extreme reached) minus a small buffer (STOP_BUFFER_PCT, same tick-
#      to-percent substitution used elsewhere in this project). Mirror
#      (high + buffer) for shorts.
#   4. TARGET - the middle band (SMA(20)) value AT THE MOMENT OF ENTRY (bar
#      i), fixed at entry and never recomputed as the trade progresses -
#      matches this project's fixed-target convention (e.g. PO3's
#      accumulation-range-edge target, S/R zone bounce's nearest-zone
#      target), not a trailing mean that would let the rule quietly change
#      mid-trade.
#   5. DEGENERATE-GEOMETRY GUARD: skip the trade (don't force it through) if
#      the target isn't on the correct side of entry (e.g. a long needs
#      target > entry - can happen if the middle band has itself drifted
#      below entry between i-1 and i in a fast-moving market) or if
#      sl_distance is below a tiny floor (MIN_SL_PCT - guards against
#      near-zero-stddev periods producing a stop right on top of entry).
#   6. MAX HOLD - force-closed (FLAT, r = pnl / sl_distance) after 48 bars
#      (4 hours) if neither stop nor target is hit. This strategy isn't
#      tied to a session window the way most of this project's other
#      scripts are, so an explicit bounded hold is what keeps the FLAT
#      bucket meaningful instead of letting trades run indefinitely.
#   7. ONE TRADE AT A TIME - a new signal is ignored while a position is
#      already open, same as every other script here.
#
# No parameter grid search - one fixed rule (20-period bands, 2 stddev,
# 48-bar max hold), tested for whether it holds up via a split-period
# honesty check, not tuned to this data.

# !pip install --upgrade dukascopy-python -q   # uncomment this line in Colab

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
# download chunk. A plain logger.setLevel(WARNING) does NOT survive this: the library's own
# fetch() call resets the "DUKASCRIPT" logger's level back to INFO internally on every single
# call. A logging Filter survives that reset (the library never touches .filters), so that's
# what's used instead - see day_trading_rauf_dukascopy_backtest.py for the original writeup.
class _SuppressDukascopyInfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING


logging.getLogger("DUKASCRIPT").addFilter(_SuppressDukascopyInfoFilter())

# Fetched data is cached to disk per (instrument, interval, date range) - see
# day_trading_rauf_dukascopy_backtest.py for the full writeup. Auto-detects a mounted Google
# Drive and uses that instead of the ephemeral local disk if present, so the cache survives
# runtime resets and is shared across every script in this project using the same range.
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

BB_LENGTH = 20             # SMA/stddev lookback for the bands
BB_NUM_STD = 2.0           # band width, in stddevs
STOP_BUFFER_PCT = 0.02     # % of price beyond the excursion bar's low/high - same scale as other scripts
MIN_SL_PCT = 0.02          # floor on stop distance, as a % of entry - guards against near-zero stddev periods

# Illustrative round-trip cost scenarios, as a percentage of entry price - NOT measured real spread
# data, just a few bracketing assumptions to see how much cost this edge can absorb before it
# disappears, same convention introduced in support_resistance_zone_bounce_dukascopy_backtest.py.
COST_PCT_SCENARIOS = [0.0, 0.01, 0.03, 0.05]
MAX_HOLD_BARS = 48         # 4 hours of 5-min bars


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
    """Downloads in FETCH_CHUNK_MONTHS-sized pieces and caches the combined result to disk -
    see day_trading_rauf_dukascopy_backtest.py for the full writeup on why."""
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


def compute_bollinger_bands(closes, length=BB_LENGTH, num_std=BB_NUM_STD):
    """Returns (middle, upper, lower) lists, same length as `closes`, with None for the first
    `length - 1` bars where the rolling window isn't yet full. Pure pandas rolling SMA/stddev -
    no lookahead (each bar's band values only use that bar and the length-1 bars before it)."""
    s = pd.Series(closes, dtype=float)
    middle = s.rolling(length).mean()
    std = s.rolling(length).std()   # pandas default ddof=1 (sample stddev) - see header note
    upper = middle + num_std * std
    lower = middle - num_std * std
    # NaN, not None, is what a float Series actually holds even after .where(...) - converting
    # explicitly to a plain Python list with real None makes the "not enough history yet" checks
    # downstream an explicit, honest `is None` rather than an accidental NaN-comparisons-are-
    # always-False side effect.
    to_list = lambda series: [None if pd.isna(x) else float(x) for x in series]
    return to_list(middle), to_list(upper), to_list(lower)


def backtest_instrument(label, df):
    highs = df["High"].tolist()
    lows = df["Low"].tolist()
    closes = df["Close"].tolist()
    times = df.index
    n = len(closes)

    middle, upper, lower = compute_bollinger_bands(closes)

    trades = []
    open_trade = None

    for i in range(1, n):
        # --- manage an already-open trade: stop / target / max-hold ---
        if open_trade is not None:
            side = open_trade["side"]
            stop, target = open_trade["stop"], open_trade["target"]
            hi, lo = highs[i], lows[i]
            hit_stop = lo <= stop if side == "LONG" else hi >= stop
            hit_target = hi >= target if side == "LONG" else lo <= target
            bars_held = i - open_trade["entry_index"]
            outcome = exit_r = None
            if hit_stop:
                outcome, exit_r = "SL", -1.0
            elif hit_target:
                outcome, exit_r = "TP", open_trade["reward_risk"]
            elif bars_held >= MAX_HOLD_BARS:
                pnl = (closes[i] - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - closes[i])
                outcome, exit_r = "FLAT", pnl / open_trade["sl_distance"]
            if outcome is not None:
                trades.append({"side": side, "outcome": outcome, "r": exit_r, "date": times[i].date(),
                               "stop_pct": open_trade["sl_distance"] / open_trade["entry"]})
                open_trade = None
            continue   # one trade at a time - don't look for a new signal on a bar we just managed

        if middle[i - 1] is None or middle[i] is None:
            continue   # not enough history yet for a full rolling window

        entry = closes[i]
        side = None
        if closes[i - 1] < lower[i - 1] and closes[i] >= lower[i]:
            side = "LONG"
            stop = lows[i - 1] - (STOP_BUFFER_PCT / 100.0) * entry
            sl_distance = entry - stop
            target = middle[i]
            valid_geometry = target > entry
        elif closes[i - 1] > upper[i - 1] and closes[i] <= upper[i]:
            side = "SHORT"
            stop = highs[i - 1] + (STOP_BUFFER_PCT / 100.0) * entry
            sl_distance = stop - entry
            target = middle[i]
            valid_geometry = target < entry
        else:
            continue

        min_sl = (MIN_SL_PCT / 100.0) * entry
        if sl_distance < min_sl or not valid_geometry:
            continue   # degenerate geometry - skip rather than force a bad trade through

        reward_risk = abs(target - entry) / sl_distance
        open_trade = {"side": side, "entry": entry, "stop": stop, "target": target,
                      "sl_distance": sl_distance, "reward_risk": reward_risk, "entry_index": i}

    return trades


def main():
    years = (FETCH_END - FETCH_START).days / 365
    print(f"Downloading {len(INSTRUMENTS)} instruments from Dukascopy over ~{years:.0f} years "
          f"({FETCH_START.date()} to {FETCH_END.date()}) - cached to disk after the first run, "
          f"so this is only slow once.\n")

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
        data[label] = df
        print(f"{label}: {len(df)} bars")

    if not data:
        print("No data downloaded - check output above.")
        return

    all_trades = []
    for label, df in data.items():
        trades = backtest_instrument(label, df)
        for t in trades:
            t["instrument"] = label
        all_trades.extend(trades)

    if not all_trades:
        print("\nNo trades at all - a confirmed re-entry across either band never occurred in this data.")
        return

    total_r = sum(t["r"] for t in all_trades)
    n_trades = len(all_trades)
    print("\n" + "=" * 70)
    print(f"BOLLINGER BAND MEAN-REVERSION FADE - {n_trades} trades, {FETCH_START.date()} to {FETCH_END.date()}")
    print("=" * 70)
    print(f"Total R: {total_r:+.2f}   Avg R/trade: {total_r/n_trades:+.4f}")

    tp = sum(1 for t in all_trades if t["outcome"] == "TP")
    sl = sum(1 for t in all_trades if t["outcome"] == "SL")
    flat = sum(1 for t in all_trades if t["outcome"] == "FLAT")
    print(f"Outcome breakdown: TP {tp} ({tp/n_trades*100:.1f}%)  SL {sl} ({sl/n_trades*100:.1f}%)  "
          f"FLAT {flat} ({flat/n_trades*100:.1f}%)")

    per_instrument = {}
    for t in all_trades:
        per_instrument.setdefault(t["instrument"], []).append(t["r"])
    print("\nPer instrument:")
    for label, rs in sorted(per_instrument.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {label:10s}: {len(rs):4d} trades, {sum(rs):+8.2f}R")

    all_r = [t["r"] for t in all_trades]
    if n_trades >= 2:
        std_r = np.std(all_r, ddof=1)
        z = (total_r / n_trades) / (std_r / (n_trades ** 0.5)) if std_r > 0 else 0.0
    else:
        z = 0.0
    print(f"\nApprox z-score: {z:.2f} (rule of thumb: |z| > 1.96 for ~95% confidence this isn't chance)")
    if n_trades < 100:
        print(f"CAVEAT: only {n_trades} trades - too few to trust regardless of the z-score.")

    all_trades_sorted = sorted(all_trades, key=lambda t: t["date"])
    midpoint_date = all_trades_sorted[len(all_trades_sorted) // 2]["date"]
    first_half = [t for t in all_trades_sorted if t["date"] < midpoint_date]
    second_half = [t for t in all_trades_sorted if t["date"] >= midpoint_date]
    print(f"\nSPLIT-PERIOD CHECK (same fixed rule, not tuned to this data - does it hold up in both halves?):")
    for lbl, half in [("First half", first_half), ("Second half", second_half)]:
        if not half:
            continue
        r = sum(t["r"] for t in half)
        n = len(half)
        half_r = [t["r"] for t in half]
        if n >= 2:
            std_half = np.std(half_r, ddof=1)
            z_half = (r / n) / (std_half / (n ** 0.5)) if std_half > 0 else 0.0
        else:
            z_half = 0.0
        print(f"  {lbl} ({half[0]['date']} to {half[-1]['date']}): {n} trades, {r:+.2f}R, "
              f"{r/n:+.4f}R/trade, z={z_half:.2f}")

    print(f"\nNOTE ON MULTIPLE COMPARISONS: this project has shipped many strategies, several with "
          f"their own parameter grid searches - a single script's z-score in isolation isn't strong "
          f"evidence, since data-snooping risk compounds across every strategy and parameter "
          f"combination tried project-wide, not just this one.")

    print(f"\nCOST SENSITIVITY (illustrative round-trip spread scenarios, NOT measured real spread data):")
    for cost_pct in COST_PCT_SCENARIOS:
        cost_adjusted_total = sum(t["r"] - (cost_pct / 100.0) / t["stop_pct"] for t in all_trades)
        print(f"  {cost_pct:.2f}% round-trip cost: {cost_adjusted_total:+.2f}R total, "
              f"{cost_adjusted_total/n_trades:+.4f}R/trade")
    print(f"  If the total goes negative well before 0.05%, this edge is too thin to survive real "
          f"execution costs - check your actual broker's spread on each instrument against these numbers.")

    print("\nNo commission/spread/slippage modeled. Entry is a simulated market order at the close of the "
          "bar that confirms re-entry into the band - real fills would be worse (that close is the exact "
          "trigger price, not a level you'd realistically get filled right at).")


if __name__ == "__main__":
    main()
