# Moving Average Golden Cross / Death Cross backtest via Dukascopy, 2016-2025.
#
# HONESTY NOTE UP FRONT (same as donchian_turtle_breakout_dukascopy_backtest.py):
# this is a SWING/POSITION system evaluated on DAILY bars, not one of this
# project's intraday, session-bound scripts - a position can stay open for
# months. 5-min bars are used only to (a) check every day's evolving high/
# low against a hard ATR stop, and (b) get a realistic fill at the START of
# the trading day that follows a signal, rather than assuming a fill at a
# daily close that hasn't actually happened yet.
#
# ALSO HONEST UP FRONT: a 50-day/200-day SMA cross is, BY DESIGN, a rare
# event - across a single instrument's 9-year history here, expect
# something on the order of low single digits to maybe a dozen crosses
# total, not hundreds. That is not a bug or a sign this port is broken; a
# 50/200 cross firing constantly would mean the smoothing periods are far
# too short for what this strategy actually is (a multi-month/multi-year
# trend filter). Nothing here apologizes for a small sample - it's simply
# what this specific, extremely well-known strategy produces, and the
# split-period/z-score section below says so plainly rather than treating
# a thin sample as strong evidence either way.
#
# THE RULES (operationalized exactly as specified):
#   1. Resample 5-min bars to DAILY OHLC per instrument (same construction
#      as donchian_turtle_breakout_dukascopy_backtest.py: first open, max
#      high, min low, last close).
#   2. Daily SMA(50) and SMA(200) of Close.
#   3. GOLDEN CROSS on day d: sma50[d-1] <= sma200[d-1] AND sma50[d] >
#      sma200[d] (fast crosses from at-or-below to strictly above). DEATH
#      CROSS is the mirror: sma50[d-1] >= sma200[d-1] AND sma50[d] <
#      sma200[d].
#   4. NO-LOOKAHEAD TIMING: a cross on day d is only KNOWN once day d's
#      daily bar is actually complete (i.e. at the end of day d, not
#      before). This port acts on it starting the NEXT trading day's FIRST
#      5-min bar, not day d's own close - a trader could not have used day
#      d's own close to enter before the day was even over. Concretely:
#      `act_golden`/`act_death` for calendar day D = whether a cross fired
#      on daily row D's immediately preceding trading day (`.shift(1)` on
#      the daily golden/death boolean series) - see TestNoLookahead in the
#      test file for a synthetic proof that a naive "act same-day" bug
#      would trade a day earlier than is actually possible.
#   5. ENTRY: if flat and today is a golden-cross action day -> LONG,
#      filled at this (the action day's) FIRST 5-min bar's Close - a
#      simulated market order, same "fill at bar close" convention used
#      throughout this project. Death-cross action day -> SHORT, mirrored.
#   6. EXIT ON OPPOSITE CROSS: if already LONG and a death-cross action
#      fires, close the long right there (filled at that bar's Close) -
#      the opposite cross IS this system's natural exit, no separate stop
#      needed to trigger it. Mirrored for a SHORT closing on a golden
#      cross.
#   7. WHAT HAPPENS RIGHT AFTER THAT CLOSE - TWO HONEST OPTIONS EXISTED,
#      ONE WAS PICKED: the classic "always in market" crossover system
#      would immediately flip into the opposite position using that same
#      signal (close long, open short, same bar). This port instead picks
#      the SIMPLER alternative: go FLAT and WAIT for the next fresh cross
#      to open a new position, rather than reversing straight into it. Why:
#      matches this project's stated preference for one simple, explicit
#      rule over compounding behaviors, and keeps "why did this trade
#      open" answerable by pointing at exactly one cross event rather than
#      two (the close AND the reversal) sharing a single signal. This is
#      enforced in code by tracking whether the position was ALREADY flat
#      BEFORE today's opposite-cross close is processed - only a
#      position that was flat coming into the day can open a fresh trade
#      that same day (see backtest_instrument's `flat_at_start_of_day`).
#   8. R-MULTIPLE BASIS: sl_distance = 3 * ATR(14, daily) at entry (the
#      prior-day ATR value, same no-lookahead shift used in the Donchian
#      script) - this is the sole denominator every R-multiple below is
#      measured against, exactly like the Donchian script's 2*ATR stop.
#   9. IS THE ATR LEVEL ALSO A REAL STOP, OR JUST an R-multiple ruler? -
#      TREATED AS A REAL HARD STOP TOO (the recommended choice): checked
#      against every 5-min bar's high/low while in the trade, same as
#      every other script in this project always carrying an actual
#      risk-defining stop-loss level. It is NOT the system's primary exit
#      mechanism (the opposite cross is, and will almost always fire long
#      before a 3xATR adverse move does, since 3xATR is a wide buffer) -
#      but if price does move 3xATR against the entry before the next
#      opposite cross, this closes the trade there rather than silently
#      letting an unstopped position ride. If both the ATR stop and an
#      opposite-cross exit would apply on the same bar, the ATR stop wins
#      (checked first) - same "tighter/first-hit level takes precedence"
#      convention as the Donchian script.
#  10. FORCE-CLOSE (FLAT/timeout) at FETCH_END if still in a position when
#      the data ends, r = pnl / sl_distance, same convention as every
#      other script here.
#
# GRANULARITY / FALSE-POSITIVE SANITY CHECK (see donchian_turtle_breakout_
# dukascopy_backtest.py's header and ict_po3_forex_dukascopy_backtest.py's
# for why this project checks this rather than assuming it away): a swing
# system reacting to a 50/200-day cross is an even smaller target for a
# granularity artifact than the Donchian script (crosses are rarer and the
# signal itself is already computed on closed daily bars) - but it's still
# checked, not assumed. See TestRandomWalkNullResult in the test file: a
# multi-year, no-drift random walk run through this exact backtest
# function shows no economically large average R/trade.
#
# NO PARAMETER GRID SEARCH - one fixed, extremely well-known rule
# (50/200/14/3.0), tested via a split-period honesty check, per-instrument
# breakdown, and an approximate z-score, not tuned to this data.
#
# NO COMMISSION, SPREAD, OR SLIPPAGE MODELED. Entries/exits are simulated
# market orders at the relevant bar's close; real fills would be worse.

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

SMA_FAST = 50
SMA_SLOW = 200
ATR_PERIOD = 14
ATR_STOP_MULT = 3.0   # sl_distance = ATR_STOP_MULT * ATR(14) - both the R basis AND a real hard stop, see header

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
    Days with no bars (weekends, holidays) simply produce no row."""
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
    """Attaches sma_fast/sma_slow, the raw golden_cross/death_cross booleans (true on the day
    the cross itself completes), the shifted act_golden/act_death flags (true on the NEXT
    trading day, when it's actually legal to act on it), and atr_entry (prior-day ATR(14))."""
    daily = daily.copy()
    highs = daily["High"].tolist()
    lows = daily["Low"].tolist()
    closes = daily["Close"].tolist()

    atr = compute_atr_series(highs, lows, closes, ATR_PERIOD)
    daily["atr_entry"] = pd.Series(atr, index=daily.index).shift(1)

    fast = daily["Close"].rolling(SMA_FAST).mean()
    slow = daily["Close"].rolling(SMA_SLOW).mean()
    daily["sma_fast"] = fast
    daily["sma_slow"] = slow

    prev_fast = fast.shift(1)
    prev_slow = slow.shift(1)
    valid = fast.notna() & slow.notna() & prev_fast.notna() & prev_slow.notna()
    golden = valid & (prev_fast <= prev_slow) & (fast > slow)
    death = valid & (prev_fast >= prev_slow) & (fast < slow)
    daily["golden_cross"] = golden
    daily["death_cross"] = death

    # a cross detected ON day d is only actionable starting day d's NEXT trading day - see
    # header point 4. This shift is the entire no-lookahead mechanism for this strategy.
    daily["act_golden"] = golden.shift(1, fill_value=False)
    daily["act_death"] = death.shift(1, fill_value=False)
    return daily


def backtest_instrument(label, df):
    """Runs the full Golden/Death Cross rule set over one instrument's 5-min bars. `df` must
    have Open/High/Low/Close columns and a tz-aware DatetimeIndex (as produced by
    fetch_instrument_data). Returns a list of trade dicts."""
    daily = build_daily_signals(resample_daily(df))
    day_map = {
        ts.date(): {
            "act_golden": bool(row["act_golden"]),
            "act_death": bool(row["act_death"]),
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
    current_day = None

    for i in range(n):
        day = times[i].date()
        sig = day_map.get(day)
        if sig is None:
            continue
        is_first_bar_of_day = day != current_day
        if is_first_bar_of_day:
            current_day = day
            # captured BEFORE any of today's actions (both the ATR-stop check right below and
            # the opposite-cross check further down) - this is what "already flat before today"
            # actually means. Capturing this AFTER the stop check would be a bug: it would let a
            # same-day stop-out make the day look like it "started flat", which would then let
            # that same day's already-consumed opposite-cross signal open a brand-new reversal
            # position on the very bar the old one was stopped out - exactly the always-in-market
            # reversal behavior header point 7 deliberately opted out of. Fixed by fixing this
            # flag once per day, before anything else happens that day.
            flat_at_start_of_day = position is None

        # --- ATR hard stop: checked on every bar while in a trade, takes precedence over an
        # opposite-cross close if both would apply on the same bar (checked first, below) ---
        if position is not None:
            side = position["side"]
            stop = position["stop"]
            hi, lo = highs[i], lows[i]
            hit_stop = (lo <= stop) if side == "LONG" else (hi >= stop)
            if hit_stop:
                trades.append({"side": side, "outcome": "SL", "r": -1.0, "date": position["entry_date"],
                               "stop_pct": position["sl_distance"] / position["entry"],
                               "entry_price": position["entry"], "stop_price": stop, "target_price": None,
                               "exit_price": stop, "entry_time": position["entry_time"], "exit_time": times[i]})
                position = None

        # --- cross-triggered actions: evaluated ONLY on the action day's first 5-min bar - a
        # cross is a once-per-day daily event, not something to re-check bar by bar (that would
        # let a same-day ATR stop-out get immediately re-opened by the day's already-consumed
        # cross signal, or let a same-day exit reverse straight into the opposite side - see
        # header point 7 on why that reversal path was deliberately not built) ---
        if is_first_bar_of_day:
            close = closes[i]

            if position is not None:
                side = position["side"]
                opposite_fired = sig["act_death"] if side == "LONG" else sig["act_golden"]
                if opposite_fired:
                    pnl = (close - position["entry"]) if side == "LONG" else (position["entry"] - close)
                    trades.append({"side": side, "outcome": "CROSS", "r": pnl / position["sl_distance"],
                                   "date": position["entry_date"],
                                   "stop_pct": position["sl_distance"] / position["entry"],
                                   "entry_price": position["entry"], "stop_price": position["stop"],
                                   "target_price": None, "exit_price": close,
                                   "entry_time": position["entry_time"], "exit_time": times[i]})
                    position = None

            atr = sig["atr"]
            can_enter = flat_at_start_of_day and position is None and atr is not None and not pd.isna(atr)
            if can_enter:
                if sig["act_golden"]:
                    entry = close
                    stop = entry - ATR_STOP_MULT * atr
                    sl_distance = entry - stop
                    if sl_distance > 0:
                        position = {"side": "LONG", "entry": entry, "stop": stop,
                                    "sl_distance": sl_distance, "entry_date": day, "entry_time": times[i]}
                elif sig["act_death"]:
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
        print("\nNo trades at all - no 50/200-day cross fired for any instrument in this window "
              "(expected to be rare - see the header note on sample size).")
        return

    total_r = sum(t["r"] for t in all_trades)
    n_trades = len(all_trades)
    print("\n" + "=" * 70)
    print(f"MOVING AVERAGE GOLDEN/DEATH CROSS (50/200 SMA, swing, daily) - {n_trades} trades, "
          f"{FETCH_START.date()} to {FETCH_END.date()}")
    print("=" * 70)
    print(f"Total R: {total_r:+.2f}   Avg R/trade: {total_r/n_trades:+.4f}")
    if n_trades < 30:
        print(f"CAVEAT: a 50/200-day cross is inherently rare - {n_trades} trades total across "
              f"{len(INSTRUMENTS)} instruments over ~{years:.0f} years is the EXPECTED order of "
              f"magnitude for this strategy, not a data problem. Treat every statistic below "
              f"accordingly.")

    sl = sum(1 for t in all_trades if t["outcome"] == "SL")
    cross = sum(1 for t in all_trades if t["outcome"] == "CROSS")
    flat = sum(1 for t in all_trades if t["outcome"] == "FLAT")
    print(f"Outcome breakdown: SL {sl} ({sl/n_trades*100:.1f}%)  CROSS(opposite signal) {cross} "
          f"({cross/n_trades*100:.1f}%)  FLAT(timeout) {flat} ({flat/n_trades*100:.1f}%)")

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
    print(f"\nApprox z-score: {z:.2f} (rule of thumb: |z| > 1.96 for ~95% confidence this isn't chance - "
          f"with this few trades, treat this as a very rough indicator, not a real test)")
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

    print("\nNo commission/spread/slippage modeled. Entries/exits are simulated market orders at the "
          "relevant bar's close; real fills would be worse. Flat-then-wait was chosen over always-in-market "
          "reversal on the same signal (see header point 7) - this understates trade count relative to a "
          "classic always-in-market crossover system, which is a deliberate simplicity trade-off, not an "
          "oversight.")


if __name__ == "__main__":
    main()
