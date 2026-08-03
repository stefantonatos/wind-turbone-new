# Bollinger Band squeeze breakout backtest via Dukascopy, 2016-2025.
#
# NOT THE SAME STRATEGY AS bollinger_band_mean_reversion_dukascopy_backtest.py
# - please don't conflate the two when reading results side by side. That
# script FADES a band touch (a confirmed re-entry back inside the band,
# betting on reversion to the mean). This script does the mechanical
# OPPOSITE: it waits for the bands to visibly CONTRACT (a "squeeze" -
# volatility compression) and then trades WITH the breakout once price
# closes outside a band, betting the contraction was the calm before an
# expansion, not a range to fade. Same underlying band math (reused
# verbatim below, duplicated rather than imported so this script has no
# runtime dependency on the mean-reversion file - see donchian_turtle_
# breakout_dukascopy_backtest.py for the same "duplicate, don't import,
# cite the source" convention already used in this project), so the two
# scripts are directly comparable on a like-for-like SMA(20)/2-stddev
# basis, but the entry logic, stop, and target are all different.
#
# OPERATIONALIZING "Bollinger squeeze breakout" precisely:
#   1. BANDS: SMA(20) of Close (BB_LENGTH) +/- 2 stddev(20) (BB_NUM_STD) -
#      identical construction to bollinger_band_mean_reversion_dukascopy_
#      backtest.py's compute_bollinger_bands(), same ddof=1 sample stddev
#      note applies here too.
#   2. BAND WIDTH: (upper - lower) / middle - a normalized width (a
#      fraction of price), not a raw price-unit width. This matters because
#      raw upper-minus-lower is on a totally different absolute scale for
#      USDJPY (~150) than EURUSD (~1.1), so only the normalized form is
#      comparable across this project's 4 instruments, and it's what makes
#      a single SQUEEZE_PERCENTILE threshold meaningful for all of them.
#   3. SQUEEZE DETECTION: band width at bar i is "squeezed" if it's at or
#      below its OWN rolling SQUEEZE_PERCENTILE-th percentile over the
#      trailing SQUEEZE_LOOKBACK bars (120 bars = 10 hours of 5-min bars),
#      via pandas' rolling().quantile(). This is a deliberately RELATIVE,
#      windowed definition, not an absolute "width is currently small"
#      threshold - an absolute threshold would be meaningless without
#      knowing what "small" means for that instrument's recent volatility
#      regime (a quiet week in USDJPY and a quiet week in XAUUSD look
#      nothing alike in raw terms), and it's also meaningless across time
#      as an instrument's baseline volatility drifts year to year. "In the
#      bottom 10% of the last 10 hours" is a real, checkable, self-
#      relative condition.
#   4. BREAKOUT ENTRY - trading WITH the break, the opposite of the mean-
#      reversion script's fade: once a squeeze bar has occurred, the FIRST
#      close beyond the upper band -> LONG, or beyond the lower band ->
#      SHORT. "Once squeezed" is operationalized as: find the most recent
#      bar at or before i that was itself squeezed and is within
#      SQUEEZE_RECENCY_BARS (5 bars) of i (see most_recent_squeeze_anchor()
#      below) - this is what stops the rule from firing on a stale squeeze
#      from hours ago where volatility has already long since normalized;
#      it must be catching an actual squeeze-then-expand event, not any old
#      band close that happens to follow a squeeze at some arbitrary
#      distance in the past.
#   5. "FIRST close" is enforced per squeeze episode via a one-shot
#      consumption flag keyed on that most-recent-squeeze-bar index (the
#      "anchor"): once a breakout is attempted off a given anchor bar
#      (whether or not a trade actually results - see the degenerate-
#      geometry point below), that anchor is marked used and won't fire
#      again even if price keeps closing outside the band on subsequent
#      bars while flat. A FRESH squeeze bar appearing later (advancing the
#      anchor) re-arms the rule - matching this project's established "one-
#      shot signal, regardless of outcome" convention (see PO3's
#      `manipulated` flag and Rauf's `setup = None` consumption, both set
#      immediately on detection independent of whether a trade actually
#      opens).
#   6. STOP - the OPPOSITE band's value at the breakout bar (upper band for
#      a short, lower band for a long). The squeeze's own tight width gives
#      a naturally close, volatility-scaled stop - guarded by the same
#      MIN_SL_PCT floor convention used in the mean-reversion script (skip
#      the trade rather than force a near-zero-width stop through).
#   7. TARGET - a FIXED reward:risk of REWARD_RISK (2.0 by default), not a
#      measured-move or range-derived level. Unlike PO3 (opposite side of
#      an accumulation range) or the mean-reversion script (the middle
#      band), a squeeze breakout doesn't have a natural "opposite edge" to
#      target - the whole premise is that price is leaving a tight range,
#      not returning to one - so a fixed R:R is the honest choice here
#      rather than inventing a measured-move target this source commentary
#      never specified.
#   8. DEGENERATE-GEOMETRY GUARD: skip (don't force) if sl_distance falls
#      below MIN_SL_PCT of entry. By construction sl_distance = entry -
#      lower[i] (long) or upper[i] - entry (short), and since closes[i] is
#      strictly beyond the band being entered on, this is virtually always
#      positive and non-trivial - but it's still checked explicitly (a
#      near-zero-stddev band could theoretically make it tiny) rather than
#      assumed safe.
#   9. MAX HOLD - force-closed (FLAT, r = pnl / sl_distance) after
#      MAX_HOLD_BARS (48 bars = 4 hours), same convention and same value as
#      the mean-reversion script. Continuous, no session window.
#  10. ONE TRADE AT A TIME - a new signal is ignored while a position is
#      already open, same as every other script here.
#
# No parameter grid search - one fixed rule (20-period bands, 2 stddev,
# 120-bar/10th-percentile squeeze, 5-bar recency, 2:1 R:R, 48-bar max
# hold), tested for whether it holds up via a split-period honesty check,
# not tuned to this data.
#
# GRANULARITY / FALSE-POSITIVE SANITY CHECK (this project's established
# practice - see ict_po3_forex_dukascopy_backtest.py's header for why a
# single-step synthetic-bar OHLC construction can manufacture a fake edge):
# checked here too, via a synthetic zero-drift multi-step random walk run at
# increasing intrabar resolution (test_bollinger_squeeze_breakout_dukascopy_
# backtest.py's TestGranularityConvergence). RESULT: unlike PO3's
# manipulation-bar stop or the mean-reversion script's excursion-bar-low/
# high stop - both pinned to that SAME bar's own wick, which is exactly what
# a wickless coarse-resolution construction distorts - this strategy's
# entry (a plain close), stop (the opposite band value, itself a rolling
# SMA/stddev of CLOSES) and target (a fixed R:R multiple) are ALL close-
# based and never reference the breakout bar's own high/low. Empirically,
# the measured z-score stays small (|z| < 3) and non-monotonic in substep
# count at every resolution tested - no large coarse-resolution artifact
# collapsing as resolution increases, because there's no direct channel for
# one to exist in this particular geometry. That's the actual, checked
# result, not an assumption - see the test file for the full writeup and
# numbers.

# !pip install --upgrade dukascopy-python -q   # uncomment this line in Colab

import calendar
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

BB_LENGTH = 20                 # SMA/stddev lookback for the bands - same as the mean-reversion script
BB_NUM_STD = 2.0                # band width, in stddevs - same as the mean-reversion script
SQUEEZE_LOOKBACK = 120          # bars (10 hours of 5-min bars) - the rolling window a squeeze is measured against
SQUEEZE_PERCENTILE = 10         # "squeezed" = band width in the bottom 10% of its own last SQUEEZE_LOOKBACK bars
SQUEEZE_RECENCY_BARS = 5        # a squeeze must have occurred within this many bars of the breakout to count
MIN_SL_PCT = 0.02               # floor on stop distance, as a % of entry - guards against near-zero-width bands
REWARD_RISK = 2.0               # fixed reward:risk target - see header point 7 on why not a measured move

# Illustrative round-trip cost scenarios, as a percentage of entry price - NOT measured real spread
# data, just a few bracketing assumptions to see how much cost this edge can absorb before it
# disappears, same convention introduced in support_resistance_zone_bounce_dukascopy_backtest.py.
COST_PCT_SCENARIOS = [0.0, 0.01, 0.03, 0.05]
MAX_HOLD_BARS = 48              # 4 hours of 5-min bars


def to_ny_time(index):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert("America/New_York")


def _month_chunks(start, end, months_per_chunk):
    """Splits [start, end) into months_per_chunk-sized pieces, advancing by calendar month
    while keeping the same day-of-month as `start` where possible. BUG FIX: a plain
    `chunk_start.replace(month=...)` raises ValueError("day is out of range for month")
    whenever start falls on the 29th-31st and the target month is shorter (e.g. start on
    Jan 31 + 3 months -> April 31, which doesn't exist) - this silently broke ANY date range
    whose start date landed on one of those days, which is a completely ordinary thing for a
    user-picked "last N days" range to do. Clamps to the target month's actual last day
    instead, the same convention date-arithmetic libraries like dateutil use for
    add-N-months."""
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
    """Identical to compute_bollinger_bands() in bollinger_band_mean_reversion_dukascopy_
    backtest.py - duplicated here rather than imported (this project's convention: see
    donchian_turtle_breakout_dukascopy_backtest.py duplicating compute_atr_series from
    orb_indices_optimization_and_ml.py) so this script has no runtime dependency on that file.
    Returns (middle, upper, lower) lists, same length as `closes`, with None for the first
    `length - 1` bars where the rolling window isn't yet full. Pure pandas rolling SMA/stddev -
    no lookahead (each bar's band values only use that bar and the length-1 bars before it)."""
    s = pd.Series(closes, dtype=float)
    middle = s.rolling(length).mean()
    std = s.rolling(length).std()   # pandas default ddof=1 (sample stddev)
    upper = middle + num_std * std
    lower = middle - num_std * std
    to_list = lambda series: [None if pd.isna(x) else float(x) for x in series]
    return to_list(middle), to_list(upper), to_list(lower)


def compute_band_width(middle, upper, lower):
    """Normalized band width per bar: (upper - lower) / middle - see header point 2 on why
    normalized, not raw price units. None wherever any input is None, or middle is exactly 0
    (guards a division that shouldn't realistically happen for these instruments' prices, but
    is checked explicitly rather than assumed safe)."""
    n = len(middle)
    out = [None] * n
    for i in range(n):
        m, u, l = middle[i], upper[i], lower[i]
        if m is None or u is None or l is None or m == 0:
            continue
        out[i] = (u - l) / m
    return out


def compute_squeeze_flags(band_width, lookback=SQUEEZE_LOOKBACK, percentile=SQUEEZE_PERCENTILE):
    """True at bar i if band_width[i] is at or below the `percentile`-th percentile of band
    width over the trailing `lookback`-bar window ENDING AT bar i (inclusive) - see header point
    3 for why this relative, rolling-window definition (not an absolute "width is small"
    threshold) is the honest way to operationalize "squeezed". Needs `lookback` consecutive
    valid (non-None) width values ending at i before it can say anything; bars without enough
    history are False (not a separate None/tri-state) since "not enough history to call it a
    squeeze" and "not squeezed" have the same downstream effect on the entry rule."""
    n = len(band_width)
    s = pd.Series(band_width, dtype=float)   # None -> NaN
    q = s.rolling(lookback).quantile(percentile / 100.0)
    out = [False] * n
    for i in range(n):
        wi, qi = s.iloc[i], q.iloc[i]
        if pd.isna(wi) or pd.isna(qi):
            continue
        out[i] = bool(wi <= qi)
    return out


def most_recent_squeeze_anchor(squeeze_flags, i, recency_bars=SQUEEZE_RECENCY_BARS):
    """The most recent bar index j <= i where squeeze_flags[j] is True and i - j <
    recency_bars (i.e. within the last `recency_bars` bars, inclusive of bar i itself) - or None
    if no such bar exists. This is the "squeeze episode" identity used by the one-shot breakout
    consumption logic in backtest_instrument(): as long as this anchor doesn't change, at most
    one breakout attempt is taken from it (header point 5)."""
    lo = max(0, i - recency_bars + 1)
    for j in range(i, lo - 1, -1):
        if squeeze_flags[j]:
            return j
    return None


def backtest_instrument(label, df):
    highs = df["High"].tolist()
    lows = df["Low"].tolist()
    closes = df["Close"].tolist()
    times = df.index
    n = len(closes)

    middle, upper, lower = compute_bollinger_bands(closes)
    band_width = compute_band_width(middle, upper, lower)
    squeeze_flags = compute_squeeze_flags(band_width)

    trades = []
    open_trade = None
    consumed_anchor = None   # the most-recent-squeeze-bar index already used for a breakout attempt

    for i in range(n):
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
                trades.append({"side": side, "outcome": outcome, "r": exit_r, "date": times[i].date(),
                               "stop_pct": open_trade["sl_distance"] / open_trade["entry"]})
                open_trade = None
            continue   # one trade at a time - don't look for a new signal on a bar we just managed

        if middle[i] is None or upper[i] is None or lower[i] is None:
            continue   # not enough history yet for a full rolling band window

        anchor = most_recent_squeeze_anchor(squeeze_flags, i)
        if anchor is None:
            continue   # no squeeze within the recency window - nothing to trade with

        if anchor == consumed_anchor:
            continue   # this squeeze episode already had its one breakout attempt (header point 5)

        entry = closes[i]
        side = None
        if entry > upper[i]:
            side = "LONG"
        elif entry < lower[i]:
            side = "SHORT"
        else:
            continue   # squeezed and recent, but no breakout yet this bar - anchor stays available

        consumed_anchor = anchor   # FIRST close beyond the band for this episode - consume regardless of outcome

        if side == "LONG":
            stop = lower[i]
            sl_distance = entry - stop
        else:
            stop = upper[i]
            sl_distance = stop - entry

        min_sl = (MIN_SL_PCT / 100.0) * entry
        if sl_distance < min_sl:
            continue   # degenerate geometry - skip rather than force a near-zero stop through

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
        print("\nNo trades at all - a squeeze followed by a recency-window breakout never occurred in this data.")
        return

    total_r = sum(t["r"] for t in all_trades)
    n_trades = len(all_trades)
    print("\n" + "=" * 70)
    print(f"BOLLINGER BAND SQUEEZE BREAKOUT - {n_trades} trades, {FETCH_START.date()} to {FETCH_END.date()}")
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

    # CORRECT z-score: (mean R) / (standard error of the mean), i.e.
    # z = (total_r / n) / (std(r, ddof=1) / sqrt(n)) - NOT the old buggy avg_r * sqrt(n) shortcut
    # this project caught and fixed project-wide (see ict_po3_forex_dukascopy_backtest.py).
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
          "breakout bar itself - real fills would be worse (that close is the exact trigger price, not a "
          "level you'd realistically get filled right at, especially on a volatility-expansion breakout "
          "where spreads often widen).")


if __name__ == "__main__":
    main()
