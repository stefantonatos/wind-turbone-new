# Dow Theory Swing Structure (trend-following via confirmed higher-highs/
# higher-lows) backtest via Dukascopy, 2016-2025.
#
# HONESTY NOTE UP FRONT (same as donchian_turtle_breakout_dukascopy_
# backtest.py and ma_golden_death_cross_dukascopy_backtest.py): this is a
# SWING/POSITION system evaluated on DAILY bars, not one of this project's
# intraday, session-bound scripts - a position can stay open for weeks or
# months, trailing behind confirmed swing structure. 5-min bars are used
# only to (a) check every day intrabar against a trailing stop level built
# from daily swing pivots, and (b) get a realistic fill at the actual
# clock time a signal becomes knowable, rather than assuming a fill at a
# daily close that hasn't happened yet.
#
# Source commentary (paraphrased, from the same systematic trader's tier-
# list this project's asian_range_breakout_dukascopy_backtest.py draws
# from): classic Dow Theory trend-following via confirmed swing structure -
# higher-highs/higher-lows mean an uptrend, lower-highs/lower-lows mean a
# downtrend. You will get whipsawed in and out constantly and take larger
# drawdowns than necessary, but when it does trend and run, you can bank
# really nice profits. That description - frequent small whipsaw losses,
# occasional large trend-following winners - is explicitly a trailing-stop
# trend system's profile, not a fixed-target one, so this port reuses the
# SAME trailing-exit mechanic already implemented in donchian_turtle_
# breakout_dukascopy_backtest.py rather than inventing a new one.
#
# OPERATIONALIZING "trade confirmed HH/HL and LH/LL structure":
#   1. Resample 5-min bars to DAILY OHLC per instrument (same construction
#      as the Donchian/Turtle and MA Golden/Death Cross scripts: first
#      open, max high, min low, last close).
#   2. CONFIRMED SWING PIVOTS on the daily High/Low series, using the same
#      fractal-pivot pattern as support_resistance_zone_bounce_dukascopy_
#      backtest.py's find_confirmed_pivots: a daily bar at index i is a
#      swing high only if it's the strict max of the SWING_LEN bars on
#      BOTH sides (mirror for swing low, strict min). A pivot at index i
#      is only usable starting at index i + SWING_LEN - the earliest point
#      its formation could actually be known, since confirming "this was a
#      local extreme" requires SWING_LEN days AFTER it to exist too. Using
#      it any earlier would be lookahead (checked directly in the unit
#      tests). SWING_LEN = 20 trading days each side is this script's one
#      structural parameter - not grid-searched, matching this project's
#      "single fixed rule" style elsewhere (see the "no parameter grid
#      search" note below).
#   3. TREND STRUCTURE, tracked as the running sequence of confirmed swing
#      highs and confirmed swing lows (in the order they get CONFIRMED,
#      not the order they FORMED - a real trader also only learns about
#      them in confirmation order): UPTREND holds when the two most
#      recently confirmed swing highs are strictly increasing AND the two
#      most recently confirmed swing lows are strictly increasing (a
#      genuine HH+HL structure - BOTH conditions, not just one).
#      DOWNTREND mirrors it (LH+LL, both strictly decreasing). Anything
#      else (mixed HH+LL, LH+HL, or fewer than two confirmed pivots of
#      either kind yet) is "no structure" - flat, no signal.
#   4. ENTRY: on the exact day a confirmation event causes the trend state
#      to freshly transition from "not uptrend" to "uptrend" (the pivot
#      that completes the HH+HL pattern), and only while flat -> LONG.
#      Mirror for a fresh downtrend transition -> SHORT. If the uptrend
#      structure was ALREADY true before this day's confirmation (i.e.
#      today's event didn't change the state), that's not a fresh
#      transition and does not re-trigger an entry even if currently
#      flat - "freshly confirmed" means the transition itself, not merely
#      "currently in that state" (see TestFreshTransitionOnly).
#   5. NO-LOOKAHEAD ENTRY TIMING: a confirmation event on day d is only
#      knowable once day d's own daily bar is complete (mirrors MA Golden/
#      Death Cross's exact reasoning for its own cross-timing choice) - a
#      trader could not have used day d's own close to enter before day d
#      was even over. This port acts starting the NEXT trading day's FIRST
#      5-min bar (`.shift(1)` on the daily fresh-transition and stop-level
#      series - see TestNoLookahead for a synthetic proof a naive "act
#      same-day" bug would trade a day earlier than is actually possible),
#      filled at that first bar's Close - same "fill at bar close"
#      convention used throughout this project.
#   6. STOP: the most recent confirmed swing low (longs) / swing high
#      (shorts) as of the entry day - the structure's own invalidation
#      point. `sl_distance` is recorded ONCE at entry and never changes -
#      every R-multiple below is measured against this fixed number, same
#      convention as every other script here.
#   7. TRAILING EXIT: reuses donchian_turtle_breakout_dukascopy_backtest.
#      py's exact trailing mechanic - as new swing pivots confirm in the
#      trend's favor, the stop trails to each new confirmed swing low
#      (longs) / swing high (shorts), checked against every 5-min bar's
#      high/low (a resting stop order, filled at the exact trail level,
#      same convention as Donchian's exit channel). The trail only ever
#      RATCHETS in the favorable direction (`max()` for longs, `min()` for
#      shorts) - a later, less favorable confirmed pivot (which shouldn't
#      occur while the trend state is genuinely still intact, but isn't
#      assumed away) can never loosen an already-tighter stop. Like the
#      entry timing above, a given day's own newly confirmed pivot only
#      becomes usable as a trailing level starting the NEXT day (same
#      shift-by-one-day convention) - it cannot tighten today's own
#      still-in-progress intrabar stop checks.
#   8. NO FIXED TAKE-PROFIT TARGET - pure trend-following, exactly like
#      Donchian/Turtle: ride the trade until the trailing stop is touched
#      (a structure break - price trading back through the last
#      confirmed swing low/high that was backing the trend) or, if that
#      never happens before FETCH_END, force-close at the last available
#      5-min bar's close (outcome "FLAT", r = pnl / sl_distance, same
#      timeout-accounting convention used everywhere else in this
#      project).
#   9. ONE TRADE AT A TIME per instrument - no pyramiding on top of a
#      running position, no reversing straight into an opposite fresh
#      signal while already in a trade. A fresh signal that fires while
#      already in a position (of either direction) is simply ignored,
#      same "only enter when flat" convention as this project's other
#      scripts.
#
# NO PARAMETER GRID SEARCH - one fixed rule (SWING_LEN = 20 trading days
# each side, the only structural parameter), tested via a split-period
# (first half vs second half of trades by date) honesty check plus a
# per-instrument breakdown and a correctly-computed z-score, not tuned to
# this data. A 20-day-each-side fractal pivot is, BY DESIGN, a fairly
# selective filter - across a single instrument's 9-year history here,
# expect a modest handful to a few dozen trades total, not hundreds. That
# is not a bug; it is what a genuinely multi-day/multi-week swing
# structure system produces, and the split-period/z-score section below
# says so plainly rather than treating a thin sample as strong evidence
# either way (same honesty stance as ma_golden_death_cross_dukascopy_
# backtest.py takes about its own naturally rare 50/200 cross signal).
#
# NOTE ON MULTIPLE COMPARISONS: this project has shipped many strategies,
# several with their own parameter grid searches - a single script's
# z-score in isolation isn't strong evidence, since data-snooping risk
# compounds across every strategy and parameter combination tried
# project-wide, not just this one.
#
# NO COMMISSION, SPREAD, OR SLIPPAGE MODELED. Entries and trailing-stop
# exits are simulated at exact prices (bar close / exact trail level) -
# real fills, especially the trailing stop during a fast reversal, would
# usually be worse than shown here. This is a multi-day swing system:
# expect far fewer, much longer-held trades than this project's intraday
# scripts, each with a much wider R distribution (a handful of big trend
# trades can dominate the total) - same caveat Donchian/Turtle's own
# header gives, for the same underlying reason.

# !pip install --upgrade dukascopy-python -q   # uncomment in Colab

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

SWING_LEN = 20   # bars (trading days) required on EACH side to confirm a swing high/low - the one structural parameter

# Illustrative round-trip cost scenarios, as a percentage of entry price - NOT measured real spread
# data, just a few bracketing assumptions to see how much cost this edge can absorb before it
# disappears, same convention introduced in support_resistance_zone_bounce_dukascopy_backtest.py.
COST_PCT_SCENARIOS = [0.0, 0.01, 0.03, 0.05]


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
    Days with no bars (weekends, holidays) simply produce no row - this keeps SWING_LEN counting
    real TRADING days, not calendar days. Same construction as donchian_turtle_breakout_
    dukascopy_backtest.py and ma_golden_death_cross_dukascopy_backtest.py."""
    daily = df.resample("1D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    daily = daily.dropna(subset=["Open", "High", "Low", "Close"])
    return daily


def find_confirmed_swing_pivots(highs, lows, lookback):
    """Same fractal-pivot pattern as support_resistance_zone_bounce_dukascopy_backtest.py's
    find_confirmed_pivots: a bar at index i is a swing point only if it's the strict max/min of
    the `lookback` bars on BOTH sides. Returns two dicts keyed by CONFIRM index (i + lookback) -
    the earliest index this pivot's existence could actually be known - each mapping to
    (formation_index, price)."""
    n = len(highs)
    swing_highs, swing_lows = {}, {}
    for i in range(lookback, n - lookback):
        window_highs = highs[i - lookback:i + lookback + 1]
        if highs[i] == max(window_highs) and window_highs.count(highs[i]) == 1:
            swing_highs[i + lookback] = (i, highs[i])
        window_lows = lows[i - lookback:i + lookback + 1]
        if lows[i] == min(window_lows) and window_lows.count(lows[i]) == 1:
            swing_lows[i + lookback] = (i, lows[i])
    return swing_highs, swing_lows


def compute_daily_swing_signals(daily):
    """Attaches, to every daily row, the running trend state and trailing-stop levels this
    strategy needs - all built ONLY from confirmed pivots as of that row's own date, then shifted
    by one full day (`fresh_up_act`, `fresh_down_act`, `trail_stop_long_effective`,
    `trail_stop_short_effective`) so nothing here is usable earlier than a real trader could have
    known it (see TestNoLookahead in the test file)."""
    highs = daily["High"].tolist()
    lows = daily["Low"].tolist()
    n = len(highs)
    swing_highs_by_confirm, swing_lows_by_confirm = find_confirmed_swing_pivots(highs, lows, SWING_LEN)

    confirmed_highs, confirmed_lows = [], []
    was_uptrend = was_downtrend = False
    trend_state = [None] * n
    fresh_up = [False] * n
    fresh_down = [False] * n
    trail_stop_long_asof = [None] * n     # latest confirmed swing low AS OF (including) day i
    trail_stop_short_asof = [None] * n    # latest confirmed swing high AS OF (including) day i

    for i in range(n):
        if i in swing_highs_by_confirm:
            confirmed_highs.append(swing_highs_by_confirm[i][1])
        if i in swing_lows_by_confirm:
            confirmed_lows.append(swing_lows_by_confirm[i][1])

        trail_stop_long_asof[i] = confirmed_lows[-1] if confirmed_lows else None
        trail_stop_short_asof[i] = confirmed_highs[-1] if confirmed_highs else None

        is_up = (len(confirmed_highs) >= 2 and len(confirmed_lows) >= 2 and
                 confirmed_highs[-1] > confirmed_highs[-2] and confirmed_lows[-1] > confirmed_lows[-2])
        is_down = (len(confirmed_highs) >= 2 and len(confirmed_lows) >= 2 and
                   confirmed_highs[-1] < confirmed_highs[-2] and confirmed_lows[-1] < confirmed_lows[-2])
        if is_up and is_down:
            is_up = is_down = False   # defensive only - shouldn't be reachable given the strict, mirrored comparisons above

        trend_state[i] = "UP" if is_up else ("DOWN" if is_down else None)
        fresh_up[i] = is_up and not was_uptrend
        fresh_down[i] = is_down and not was_downtrend
        was_uptrend, was_downtrend = is_up, is_down

    out = daily.copy()
    out["trend_state"] = trend_state
    out["fresh_up"] = fresh_up
    out["fresh_down"] = fresh_down
    out["trail_stop_long_asof"] = trail_stop_long_asof
    out["trail_stop_short_asof"] = trail_stop_short_asof
    # shift by one day: what's safe to ACT ON at the start of day d is exactly what was known as
    # of the end of day d-1 - a fresh confirmation or updated trailing level from day d itself
    # only becomes knowable once day d's own bar is complete (see header no-lookahead section,
    # and ma_golden_death_cross_dukascopy_backtest.py's identical reasoning for its own cross).
    out["fresh_up_act"] = out["fresh_up"].shift(1).fillna(False).astype(bool)
    out["fresh_down_act"] = out["fresh_down"].shift(1).fillna(False).astype(bool)
    out["trail_stop_long_effective"] = out["trail_stop_long_asof"].shift(1)
    out["trail_stop_short_effective"] = out["trail_stop_short_asof"].shift(1)
    return out


def backtest_instrument(label, df):
    """Runs the full Dow Theory Swing Structure rule set over one instrument's 5-min bars. `df`
    must have Open/High/Low/Close columns and a tz-aware DatetimeIndex. Returns a list of trade
    dicts."""
    daily = compute_daily_swing_signals(resample_daily(df))
    day_map = {
        ts.date(): {
            "fresh_up_act": bool(row["fresh_up_act"]),
            "fresh_down_act": bool(row["fresh_down_act"]),
            "stop_long_effective": row["trail_stop_long_effective"],
            "stop_short_effective": row["trail_stop_short_effective"],
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
            hi, lo = highs[i], lows[i]
            if side == "LONG":
                eff_stop = sig["stop_long_effective"]
                if eff_stop is not None and not pd.isna(eff_stop):
                    position["stop"] = max(position["stop"], eff_stop)   # trail only ever ratchets favorably
                if lo <= position["stop"]:
                    r = (position["stop"] - position["entry"]) / position["sl_distance"]
                    trades.append({"side": side, "outcome": "STOP", "r": r, "date": position["entry_date"],
                                   "stop_pct": position["sl_distance"] / position["entry"]})
                    position = None
            else:   # SHORT
                eff_stop = sig["stop_short_effective"]
                if eff_stop is not None and not pd.isna(eff_stop):
                    position["stop"] = min(position["stop"], eff_stop)
                if hi >= position["stop"]:
                    r = (position["entry"] - position["stop"]) / position["sl_distance"]
                    trades.append({"side": side, "outcome": "STOP", "r": r, "date": position["entry_date"],
                                   "stop_pct": position["sl_distance"] / position["entry"]})
                    position = None
            # one trade at a time: whether the position just closed or is still open, don't also
            # evaluate a fresh entry on this same bar (matches Donchian's identical convention)
            continue

        is_first_bar_of_day = (i == 0) or (times[i - 1].date() != day)
        if not is_first_bar_of_day:
            continue   # entries only fire on the first 5-min bar of the "action day" (see header)

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
        print("\nNo trades at all - a confirmed HH+HL or LH+LL swing structure transition never "
              "occurred in this data (see header note: a 20-day-each-side fractal pivot is a "
              "selective filter by design).")
        return

    total_r = sum(t["r"] for t in all_trades)
    n_trades = len(all_trades)
    print("\n" + "=" * 70)
    print(f"DOW THEORY SWING STRUCTURE (swing, daily pivots) - {n_trades} trades, "
          f"{FETCH_START.date()} to {FETCH_END.date()}")
    print("=" * 70)
    print(f"Total R: {total_r:+.2f}   Avg R/trade: {total_r/n_trades:+.4f}")

    stop = sum(1 for t in all_trades if t["outcome"] == "STOP")
    flat = sum(1 for t in all_trades if t["outcome"] == "FLAT")
    print(f"Outcome breakdown: STOP(trail) {stop} ({stop/n_trades*100:.1f}%)  "
          f"FLAT(timeout) {flat} ({flat/n_trades*100:.1f}%)")

    per_instrument = {}
    for t in all_trades:
        per_instrument.setdefault(t["instrument"], []).append(t["r"])
    print("\nPer instrument:")
    for label, rs in sorted(per_instrument.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {label:10s}: {len(rs):4d} trades, {sum(rs):+8.2f}R")

    per_side = {}
    for t in all_trades:
        per_side.setdefault(t["side"], []).append(t["r"])
    print("\nPer direction:")
    for side, rs in per_side.items():
        print(f"  {side:10s}: {len(rs):4d} trades, {sum(rs):+8.2f}R")

    all_r = [t["r"] for t in all_trades]
    if n_trades >= 2:
        std_r = np.std(all_r, ddof=1)
        z = (total_r / n_trades) / (std_r / (n_trades ** 0.5)) if std_r > 0 else 0.0
    else:
        z = 0.0
    print(f"\nApprox z-score: {z:.2f} (rule of thumb: |z| > 1.96 for ~95% confidence this isn't chance)")
    if n_trades < 100:
        print(f"CAVEAT: only {n_trades} trades - too few to trust regardless of the z-score (see header "
              f"note: this is expected for a selective, multi-week swing-structure filter, not a bug).")

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

    print("\nNo commission/spread/slippage modeled. Entries and trailing-stop exits are simulated at exact "
          "prices (bar close / exact trail level) - real fills, especially the trailing stop during a fast "
          "reversal, would usually be worse than shown here. This is a multi-day swing system: expect far "
          "fewer, much longer-held trades than this project's intraday scripts, each with a much wider R "
          "distribution (a handful of big trend trades can dominate the total).")


if __name__ == "__main__":
    main()
