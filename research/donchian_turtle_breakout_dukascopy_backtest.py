# Donchian Channel / Turtle-style breakout backtest via Dukascopy, 2016-2025.
#
# HONESTY NOTE UP FRONT: this is a genuinely different animal from every
# other intraday, session-bound script in this project (ORB, PO3, Silver
# Bullet, Rauf's sweep-and-reversal, the S/R zone bounce, etc.). Those all
# open and close a trade within the same trading day (or a fixed few-hour
# window). This is a multi-day SWING/POSITION system in the original
# 1980s Turtle Trading mold - a trade can stay open for days or weeks,
# riding a trend until a trailing channel finally reverses far enough to
# stop it out. It is evaluated on DAILY bars for its channel/ATR math, not
# forced into the intraday mold the rest of this project uses. Don't read
# "5-min bars" below as "another intraday scalp" - the 5-min granularity
# is used ONLY to check, tick by tick, whether a level computed from
# COMPLETE PRIOR DAYS gets touched, so a trade can be entered/exited
# intrabar without needing to wait for a daily close (which is also the
# only way to avoid lookahead here - see the no-lookahead section below).
#
# THE RULES (operationalized exactly as specified - no invented variants):
#   1. Resample 5-min bars to DAILY OHLC per instrument (standard: open =
#      first 5-min open of the day, high = max, low = min, close = last
#      5-min close).
#   2. Rolling 20-day Donchian entry channel: channel_high[d] = max daily
#      High over the PRIOR 20 COMPLETED days (i.e. days d-20..d-1, never
#      including day d itself); channel_low[d] = min daily Low over the
#      same prior-20-day window. Implemented as
#      `High.rolling(20).max().shift(1)` - the shift is what pushes the
#      window strictly into the past relative to day d (see the unit
#      tests for a direct proof this doesn't leak day d's own range).
#   3. Rolling 10-day EXIT channel, same construction: exit_high[d] (prior
#      10 days' max High), exit_low[d] (prior 10 days' min Low).
#   4. Daily ATR(14), Wilder's smoothing - same formula as
#      `compute_atr_series` in research/orb_indices_optimization_and_ml.py
#      (first 14 true ranges seeded as a simple average, then
#      `atr = (prev_atr * 13 + tr) / 14` from there on). The ATR value
#      used for an entry on day d is ATR as of the END of day d-1 (again
#      shifted by one full day) - a trade entered intraday on day d cannot
#      know day d's own high/low/true-range yet.
#   5. ENTRY, checked at 5-min granularity (not waiting for the daily bar
#      to close - that's what lets a trade be taken the moment a level is
#      crossed rather than a day late): if flat, and a 5-min bar's Close >
#      that day's channel_high (built entirely from days before today) ->
#      LONG. If Close < that day's channel_low -> SHORT. Only one entry
#      while flat - once in a trade, further breakouts of the same channel
#      are simply not checked (see below), which naturally also enforces
#      "only enter once per breakout".
#   6. INITIAL STOP: entry +/- 2*ATR(14) (that prior-day ATR value).
#      `sl_distance = abs(entry - stop)` is recorded ONCE at entry and
#      never changes - every R-multiple below is entry P&L divided by this
#      fixed number, matching this project's convention everywhere else.
#   7. TRAILING EXIT: while in a trade, exit the instant price touches the
#      TRAILING channel - exit_low for a long (the 10-day rolling low,
#      recomputed fresh each day from days strictly before today),
#      exit_high for a short. Checked against every 5-min bar's high/low.
#      Filled at the exact channel level (a resting stop order sitting at
#      that price), same convention as PO3/Silver Bullet filling exactly
#      at a precomputed target/stop level rather than modeling exact
#      intrabar slippage past it.
#   8. STOP-VS-TRAILING PRECEDENCE: if a single bar's range would touch
#      BOTH the initial ATR stop and the trailing channel, the ATR stop
#      wins (it's the tighter, closer level and would have been hit first
#      moving through the bar's adverse range) - same "tighter level takes
#      precedence" convention already used elsewhere in this project.
#   9. NO PROFIT TARGET. This is pure trend-following - it rides until the
#      trailing exit fires, or, if that never happens before FETCH_END,
#      the trade is force-closed at the last available 5-min bar's close
#      (outcome "FLAT", r = pnl / sl_distance, same timeout-accounting
#      convention used everywhere else in this project).
#  10. ONE TRADE AT A TIME per instrument - no pyramiding on top of a
#      running position, no reversing straight into the opposite side on
#      an opposite signal while already in a trade. Simple single-position
#      state machine, matching this project's stated preference for one
#      fixed rule over compounding complexity.
#
# NO LOOKAHEAD - the two places this could go wrong, and how they're
# avoided:
#   - Channel/ATR values for day d must never see day d's own bar. Solved
#     by computing the rolling window on the daily frame and then
#     `.shift(1)`, so the value attached to day d's calendar date is
#     always built purely from days before it. Proven directly in
#     `TestNoLookahead` below with a synthetic day whose own range would
#     produce a different (wrong) channel if a naive off-by-one bug used
#     `.rolling(20).max()` WITHOUT the shift.
#   - Because entries/exits are checked at 5-min granularity against a
#     channel that is already finalized as of the START of day d (it only
#     depends on days before d), there's no need to wait for day d's own
#     daily close to react to a breakout mid-day - reacting immediately is
#     the honest behavior here, not a shortcut.
#
# GRANULARITY / FALSE-POSITIVE SANITY CHECK (this project's established
# practice - see ict_po3_forex_dukascopy_backtest.py's header for why
# single-step synthetic OHLC construction can create fake edges): this is
# a swing system reacting to multi-day channel levels, not an
# entry-timing-within-a-bar system like PO3, so the risk of a granularity
# artifact producing a fake edge is smaller here - but it's checked, not
# assumed away. See `TestRandomWalkNullResult` in the test file: a
# multi-year, no-drift, intrabar-noisy random walk run through the exact
# same backtest function shows no economically large average R/trade,
# which is what "no structural artifact" should look like.
#
# NO PARAMETER GRID SEARCH - one fixed rule (20/10/14/2.0, the traditional
# Turtle System 1-ish parameters), tested via a split-period (first half
# vs second half of trades by date) honesty check plus a per-instrument
# breakdown and an approximate z-score, not tuned to this data.
#
# NO COMMISSION, SPREAD, OR SLIPPAGE MODELED. Entries and channel-exit
# fills are simulated at exact bar-close / exact-channel-level prices;
# real fills, especially the ATR stop in a fast-moving break, would
# usually be worse than this backtest assumes.

# !pip install --upgrade dukascopy-python -q   # uncomment in Colab

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
# not just repeat runs of this one file. Falls back to a local (session-only) cache if Drive
# isn't mounted, so this still works without any setup, just without the persistence.
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

DONCHIAN_PERIOD = 20   # entry channel, in complete prior trading days
EXIT_PERIOD = 10       # trailing exit channel, in complete prior trading days
ATR_PERIOD = 14        # Wilder ATR length
ATR_STOP_MULT = 2.0    # initial stop = entry +/- ATR_STOP_MULT * ATR(14)

# Illustrative round-trip cost scenarios, as a percentage of entry price - NOT measured real spread
# data, just a few bracketing assumptions to see how much cost this edge can absorb before it
# disappears, same convention introduced in support_resistance_zone_bounce_dukascopy_backtest.py.
COST_PCT_SCENARIOS = [0.0, 0.01, 0.03, 0.05]


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
    """Downloads in FETCH_CHUNK_MONTHS-sized pieces (shows real progress instead of one long
    silent call) and caches the combined result to disk, so re-running this script after
    changing a strategy parameter below loads instantly instead of re-downloading everything.
    Uses the exact same cache key convention as the other Dukascopy scripts in this project, so
    a cache already populated by e.g. day_trading_rauf_dukascopy_backtest.py is reused here too
    (same instrument, interval, and date range)."""
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


def resample_daily(df):
    """5-min OHLC -> daily OHLC (standard resample: first open, max high, min low, last close).
    Days with no bars (weekends, holidays) simply produce no row - this keeps the rolling
    windows below counting real TRADING days, not calendar days."""
    daily = df.resample("1D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    daily = daily.dropna(subset=["Open", "High", "Low", "Close"])
    return daily


def compute_atr_series(highs, lows, closes, length):
    """Wilder's ATR - identical formula to compute_atr_series in
    research/orb_indices_optimization_and_ml.py (first `length` true ranges seeded as a plain
    average, then smoothed one bar at a time: atr = (prev_atr*(length-1) + tr) / length)."""
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


def build_daily_signals(daily):
    """Attaches channel_high/channel_low (DONCHIAN_PERIOD), exit_high/exit_low (EXIT_PERIOD),
    and atr_entry (ATR_PERIOD) to each daily row - every one of these is built ONLY from days
    strictly before that row's own date (the `.shift(1)` after each rolling window is what
    enforces that; see TestNoLookahead in the test file for a direct proof)."""
    daily = daily.copy()
    highs = daily["High"].tolist()
    lows = daily["Low"].tolist()
    closes = daily["Close"].tolist()

    atr = compute_atr_series(highs, lows, closes, ATR_PERIOD)
    daily["atr_entry"] = pd.Series(atr, index=daily.index).shift(1)

    daily["channel_high"] = daily["High"].rolling(DONCHIAN_PERIOD).max().shift(1)
    daily["channel_low"] = daily["Low"].rolling(DONCHIAN_PERIOD).min().shift(1)
    daily["exit_high"] = daily["High"].rolling(EXIT_PERIOD).max().shift(1)
    daily["exit_low"] = daily["Low"].rolling(EXIT_PERIOD).min().shift(1)
    return daily


def backtest_instrument(label, df):
    """Runs the full Donchian/Turtle rule set over one instrument's 5-min bars. `df` must have
    Open/High/Low/Close columns and a tz-aware DatetimeIndex (as produced by
    fetch_instrument_data). Returns a list of trade dicts."""
    daily = build_daily_signals(resample_daily(df))
    day_map = {
        ts.date(): {
            "channel_high": row["channel_high"],
            "channel_low": row["channel_low"],
            "exit_high": row["exit_high"],
            "exit_low": row["exit_low"],
            "atr": row["atr_entry"],
        }
        for ts, row in daily.iterrows()
    }

    highs = df["High"].tolist()
    lows = df["Low"].tolist()
    closes = df["Close"].tolist()
    times = df.index
    n = len(closes)

    trades = []
    position = None   # dict: side, entry, stop, sl_distance, entry_date

    for i in range(n):
        day = times[i].date()
        sig = day_map.get(day)
        if sig is None:
            continue

        if position is not None:
            side = position["side"]
            stop = position["stop"]
            hi, lo = highs[i], lows[i]
            ex_low, ex_high = sig["exit_low"], sig["exit_high"]

            stop_pct = position["sl_distance"] / position["entry"]
            common = {"entry_price": position["entry"], "stop_price": stop, "target_price": None,
                      "entry_time": position["entry_time"], "exit_time": times[i]}
            if side == "LONG":
                hit_stop = lo <= stop
                hit_trail = (ex_low is not None) and not pd.isna(ex_low) and lo <= ex_low
                if hit_stop:
                    trades.append({"side": side, "outcome": "SL", "r": -1.0,
                                   "date": position["entry_date"], "stop_pct": stop_pct,
                                   "exit_price": stop, **common})
                    position = None
                elif hit_trail:
                    r = (ex_low - position["entry"]) / position["sl_distance"]
                    trades.append({"side": side, "outcome": "TRAIL", "r": r,
                                   "date": position["entry_date"], "stop_pct": stop_pct,
                                   "exit_price": ex_low, **common})
                    position = None
            else:   # SHORT
                hit_stop = hi >= stop
                hit_trail = (ex_high is not None) and not pd.isna(ex_high) and hi >= ex_high
                if hit_stop:
                    trades.append({"side": side, "outcome": "SL", "r": -1.0,
                                   "date": position["entry_date"], "stop_pct": stop_pct,
                                   "exit_price": stop, **common})
                    position = None
                elif hit_trail:
                    r = (position["entry"] - ex_high) / position["sl_distance"]
                    trades.append({"side": side, "outcome": "TRAIL", "r": r,
                                   "date": position["entry_date"], "stop_pct": stop_pct,
                                   "exit_price": ex_high, **common})
                    position = None
            # one trade at a time: whether the position just closed or is still open, don't
            # also evaluate a fresh entry on this same bar (matches this project's preference
            # for a single simple state machine over compounding same-bar re-entries)
            continue

        ch_high, ch_low, atr = sig["channel_high"], sig["channel_low"], sig["atr"]
        if ch_high is None or ch_low is None or atr is None or pd.isna(ch_high) or pd.isna(ch_low) or pd.isna(atr):
            continue   # warmup period - not enough prior days yet for a valid channel/ATR

        close = closes[i]
        if close > ch_high:
            entry = close
            stop = entry - ATR_STOP_MULT * atr
            sl_distance = entry - stop
            if sl_distance > 0:
                position = {"side": "LONG", "entry": entry, "stop": stop,
                            "sl_distance": sl_distance, "entry_date": day, "entry_time": times[i]}
        elif close < ch_low:
            entry = close
            stop = entry + ATR_STOP_MULT * atr
            sl_distance = stop - entry
            if sl_distance > 0:
                position = {"side": "SHORT", "entry": entry, "stop": stop,
                            "sl_distance": sl_distance, "entry_date": day, "entry_time": times[i]}

    if position is not None:
        side = position["side"]
        last_close = closes[-1]
        pnl = (last_close - position["entry"]) if side == "LONG" else (position["entry"] - last_close)
        trades.append({"side": side, "outcome": "FLAT", "r": pnl / position["sl_distance"],
                        "date": position["entry_date"], "stop_pct": position["sl_distance"] / position["entry"],
                        "entry_price": position["entry"], "stop_price": position["stop"], "target_price": None,
                        "exit_price": last_close, "entry_time": position["entry_time"], "exit_time": times[-1]})

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
        print("\nNo trades at all - the 20-day channel was never broken cleanly in this data.")
        return

    total_r = sum(t["r"] for t in all_trades)
    n_trades = len(all_trades)
    print("\n" + "=" * 70)
    print(f"DONCHIAN / TURTLE-STYLE BREAKOUT (swing, daily channels) - {n_trades} trades, "
          f"{FETCH_START.date()} to {FETCH_END.date()}")
    print("=" * 70)
    print(f"Total R: {total_r:+.2f}   Avg R/trade: {total_r/n_trades:+.4f}")

    sl = sum(1 for t in all_trades if t["outcome"] == "SL")
    trail = sum(1 for t in all_trades if t["outcome"] == "TRAIL")
    flat = sum(1 for t in all_trades if t["outcome"] == "FLAT")
    print(f"Outcome breakdown: SL {sl} ({sl/n_trades*100:.1f}%)  TRAIL {trail} ({trail/n_trades*100:.1f}%)  "
          f"FLAT(timeout) {flat} ({flat/n_trades*100:.1f}%)")

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
    for label, half in [("First half", first_half), ("Second half", second_half)]:
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
        print(f"  {label} ({half[0]['date']} to {half[-1]['date']}): {n} trades, {r:+.2f}R, "
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

    print("\nNo commission/spread/slippage modeled. Entries and trailing-channel exits are filled at exact "
          "simulated prices (bar close / exact channel level) - real fills, especially the ATR stop during a "
          "fast break, would usually be worse than shown here. This is a multi-day swing system: expect far "
          "fewer, much longer-held trades than this project's intraday scripts, each with a much wider R "
          "distribution (a handful of big trend trades can dominate the total).")


if __name__ == "__main__":
    main()
