# RSI overbought/oversold mean-reversion backtest via Dukascopy, 2016-2025.
#
# Deliberately kept MECHANICALLY DISTINCT from
# bollinger_band_mean_reversion_dukascopy_backtest.py, not "Bollinger with
# extra steps" - both are oscillator/mean-reversion in spirit, but this one
# tests a different structural claim: a fixed reward:risk target instead of
# a moving-average target, and a swing-low/high stop over a fixed lookback
# window instead of a single prior bar's low/high. Intraday, 5-min bars,
# runs continuously (no session window), same as the Bollinger script.
#
# OPERATIONALIZING "RSI mean-reversion" precisely:
#   1. RSI(14) on 5-min Close, standard Wilder smoothing: average gain and
#      average loss are each seeded as a simple mean of the first 14
#      gains/losses, then updated recursively as
#      avg = (avg * (length - 1) + new_value) / length - the exact same
#      recursive pattern compute_atr_series() in
#      orb_indices_optimization_and_ml.py uses for ATR, applied here to
#      gains/losses instead of true range (see compute_rsi_series() below).
#   2. ENTRY - a CONFIRMED RE-ENTRY into normal range, not a raw threshold
#      touch, for the same false-positive reason as the Bollinger script:
#      RSI dipping to 29 for one bar and bouncing immediately is noise a
#      single-step synthetic model would over-reward. RSI[i-1] < 30 (was
#      oversold) and RSI[i] >= 30 (back to normal) -> LONG. Mirror:
#      RSI[i-1] > 70 and RSI[i] <= 70 -> SHORT.
#   3. STOP - the lowest Low (longs) / highest High (shorts) over a FIXED
#      lookback window of STOP_LOOKBACK_BARS (12 bars = 1 hour) ending at
#      bar i-1, minus/plus a small buffer (STOP_BUFFER_PCT). The task this
#      script was built from offered two equally-defensible options here -
#      a variable-length scan back to where the RSI excursion actually
#      started, or a fixed lookback window - and explicitly said to pick
#      whichever is cleaner to implement correctly and document the
#      choice: this uses the FIXED WINDOW, since a variable-length scan
#      adds real edge-case complexity (what if the excursion is 1 bar? 40
#      bars? spans a data gap?) for a stop level that in practice usually
#      lands in the same place either way once the excursion is more than
#      a few bars deep.
#   4. TARGET - fixed reward:risk of REWARD_RISK x the stop distance. This
#      is the deliberate mechanical difference from the Bollinger script's
#      moving-average target: this rule never looks at where the "average"
#      currently sits, it only cares about the stop distance it already
#      committed to.
#   5. DEGENERATE-GEOMETRY GUARD - skip the trade if sl_distance is below a
#      tiny floor (MIN_SL_PCT), same guard used in the Bollinger script.
#   6. MAX HOLD - force-closed (FLAT, r = pnl / sl_distance) after 48 bars
#      (4 hours), same as the Bollinger script.
#   7. ONE TRADE AT A TIME - a new signal is ignored while a position is
#      already open.
#
# No parameter grid search - one fixed rule (RSI(14), 30/70 thresholds, 12-
# bar stop lookback, 1.5R target, 48-bar max hold), tested for whether it
# holds up via a split-period honesty check, not tuned to this data.

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

RSI_LENGTH = 14
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
STOP_LOOKBACK_BARS = 12    # 1 hour of 5-min bars - see header note on fixed-window vs variable-scan choice
STOP_BUFFER_PCT = 0.02     # % of price beyond the lookback window's extreme - same scale as other scripts
MIN_SL_PCT = 0.02          # floor on stop distance, as a % of entry - guards against degenerate near-zero stops
REWARD_RISK = 1.5          # fixed reward:risk target multiple - the deliberate mechanical difference vs Bollinger
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


def compute_rsi_series(closes, length=RSI_LENGTH):
    """Wilder's RSI: average gain/loss seeded as a simple mean of the first `length` gains/
    losses, then updated recursively - the exact same recursive-smoothing pattern
    compute_atr_series() in orb_indices_optimization_and_ml.py uses for ATR, applied to
    gains/losses instead of true range. Returns a list the same length as `closes`, with None
    until the seed window is full (no lookahead - each value only uses bars up to and
    including that index)."""
    n = len(closes)
    rsi = [None] * n
    avg_gain = avg_loss = None
    gain_seed, loss_seed = [], []
    prev_close = None
    for i in range(n):
        if prev_close is not None:
            diff = closes[i] - prev_close
            gain = max(diff, 0.0)
            loss = max(-diff, 0.0)
            if avg_gain is None:
                gain_seed.append(gain)
                loss_seed.append(loss)
                if len(gain_seed) >= length:
                    avg_gain = sum(gain_seed) / length
                    avg_loss = sum(loss_seed) / length
            else:
                avg_gain = (avg_gain * (length - 1) + gain) / length
                avg_loss = (avg_loss * (length - 1) + loss) / length
        if avg_gain is not None:
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        prev_close = closes[i]
    return rsi


def backtest_instrument(label, df):
    highs = df["High"].tolist()
    lows = df["Low"].tolist()
    closes = df["Close"].tolist()
    times = df.index
    n = len(closes)

    rsi = compute_rsi_series(closes)

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
                outcome, exit_r = "TP", REWARD_RISK
            elif bars_held >= MAX_HOLD_BARS:
                pnl = (closes[i] - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - closes[i])
                outcome, exit_r = "FLAT", pnl / open_trade["sl_distance"]
            if outcome is not None:
                trades.append({"side": side, "outcome": outcome, "r": exit_r, "date": times[i].date()})
                open_trade = None
            continue   # one trade at a time - don't look for a new signal on a bar we just managed

        if rsi[i - 1] is None or rsi[i] is None or i < STOP_LOOKBACK_BARS:
            continue   # not enough history yet for RSI or the stop lookback window

        entry = closes[i]
        side = None
        if rsi[i - 1] < RSI_OVERSOLD and rsi[i] >= RSI_OVERSOLD:
            side = "LONG"
            window_low = min(lows[i - STOP_LOOKBACK_BARS:i])
            stop = window_low - (STOP_BUFFER_PCT / 100.0) * entry
            sl_distance = entry - stop
        elif rsi[i - 1] > RSI_OVERBOUGHT and rsi[i] <= RSI_OVERBOUGHT:
            side = "SHORT"
            window_high = max(highs[i - STOP_LOOKBACK_BARS:i])
            stop = window_high + (STOP_BUFFER_PCT / 100.0) * entry
            sl_distance = stop - entry
        else:
            continue

        min_sl = (MIN_SL_PCT / 100.0) * entry
        if sl_distance < min_sl:
            continue   # degenerate geometry - skip rather than force a bad trade through

        target = entry + REWARD_RISK * sl_distance if side == "LONG" else entry - REWARD_RISK * sl_distance
        open_trade = {"side": side, "entry": entry, "stop": stop, "target": target,
                      "sl_distance": sl_distance, "entry_index": i}

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
        print("\nNo trades at all - a confirmed re-entry across either RSI threshold never occurred in this data.")
        return

    total_r = sum(t["r"] for t in all_trades)
    n_trades = len(all_trades)
    print("\n" + "=" * 70)
    print(f"RSI OVERBOUGHT/OVERSOLD MEAN-REVERSION - {n_trades} trades, {FETCH_START.date()} to {FETCH_END.date()}")
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

    print("\nNo commission/spread/slippage modeled. Entry is a simulated market order at the close of the "
          "bar that confirms re-entry past the RSI threshold - real fills would be worse (that close is "
          "the exact trigger price, not a level you'd realistically get filled right at).")


if __name__ == "__main__":
    main()
