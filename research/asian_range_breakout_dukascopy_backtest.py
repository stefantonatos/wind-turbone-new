# Asian Range Breakout backtest via Dukascopy, 2016-2025.
#
# Source commentary (paraphrased, from a systematic trader's own strategy
# tier-list): markets move in cycles of consolidation then breakout; the
# Asian session is a consolidated range, and when the higher-volatility
# London session breaks out of that range, it tends to produce strong,
# high-strike-rate moves. This is the OPPOSITE trading philosophy from
# ict_po3_forex_dukascopy_backtest.py in this project: PO3 FADES a range
# break (it bets the break is a manipulation to be reversed). This script
# TRADES WITH the breakout direction - momentum continuation, not mean
# reversion. Two structurally similar-looking session-range scripts, one
# fading and one following, is intentional: they're testing opposite
# theories of what a session-range break means, not the same idea twice.
#
# OPERATIONALIZING "Asian range, then trade the London breakout":
#   1. ASIAN RANGE: high/low built during ASIAN_START-ASIAN_END (19:00-
#      00:00 NY time, 5 hours - the Tokyo/Asian session), same
#      accumulation-range-building pattern as PO3's ACCUMULATION_START/
#      END. This window sits entirely within ONE calendar day (19:00
#      through 23:55, ending exactly at that day's midnight boundary), so
#      the range itself never crosses a calendar-day boundary - but the
#      range it produces is used the FOLLOWING calendar day, since London
#      opens hours after this window closes. That day-to-day carry (this
#      evening's range feeds tomorrow morning's breakout check, not
#      today's) is the one piece of real bookkeeping this script has that
#      PO3's same-day accumulation/manipulation/distribution didn't need -
#      see TestNoLookahead in the test file for a direct proof this carry
#      goes the right direction (yesterday's range, never today's own
#      still-forming one).
#   2. BREAKOUT WINDOW: BREAKOUT_START-BREAKOUT_END (02:00-05:00 NY, the
#      London-open session), on the calendar day AFTER the Asian range
#      that feeds it. The FIRST bar in this window whose CLOSE breaks
#      above the Asian range's high, or below its low, triggers entry IN
#      THAT DIRECTION - long on an upside break, short on a downside
#      break. This is a close-confirmed trigger (not a bare wick touch),
#      but the "which side broke" ambiguity check below still looks at
#      each bar's full high/low, not just its close - see the ambiguity
#      rule immediately below for why both matter.
#   3. SAME-BAR DOUBLE SWEEP: if a single bar's high/low wicks beyond BOTH
#      sides of the Asian range (a wide/volatile bar), that's genuinely
#      ambiguous which direction broke first from OHLC data alone - same
#      "don't guess" handling as PO3's `broke_high and broke_low` case.
#      This is checked independently of the close-confirmation rule above:
#      a bar can, in principle, close-confirm one direction while its wick
#      also touched the opposite side - that still counts as the ambiguous
#      case and the day is skipped, no trade taken, rather than trusting
#      the close to have picked the "real" direction.
#   4. ONE TRADE PER DAY: once a trigger fires (whether it results in an
#      actual trade or is skipped as an ambiguous double-sweep), that
#      day's breakout window is done - no further entries are evaluated
#      even if later bars in the same window would otherwise qualify.
#   5. STOP: the OPPOSITE side of the Asian range from the breakout
#      direction (long entries stop below the Asian range low, shorts stop
#      above the Asian range high), with a small buffer beyond it
#      (STOP_BUFFER_PCT, same tick-to-percent buffer convention used
#      throughout this project). Because the entry itself is already
#      beyond the broken side of the range, this stop distance is always
#      strictly positive by construction - no separate degenerate-stop
#      guard is needed here (checked directly in the unit tests).
#   6. TARGET: a MEASURED MOVE - the Asian range's own height (high - low)
#      projected from the breakout entry point, scaled by
#      TARGET_RANGE_MULT (1.5 by default). If the Asian range is
#      degenerate (its height is below MIN_RANGE_PCT of the entry price -
#      a near-zero range would otherwise project a meaninglessly tiny
#      target), this falls back to a fixed FALLBACK_REWARD_RISK (2.0)
#      applied to the stop distance instead - same guard-and-fallback
#      shape as PO3's MIN_RANGE_PCT/FALLBACK_REWARD_RISK, just guarding
#      the range height here instead of the stop distance (PO3 has no
#      measured-move target to guard; this script does).
#   7. HOLD: through the London morning, force-closed at FORCE_CLOSE_TIME
#      (11:00 NY) regardless of stop/target status if still open - marked
#      to market at that bar's close, outcome "FLAT", r = pnl /
#      sl_distance, same accounting convention as every other script here.
#      No new setups form after FORCE_CLOSE_TIME either.
#
# NO PARAMETER GRID SEARCH - one fixed rule set (19:00-00:00 Asian window,
# 02:00-05:00 breakout window, 1.5x measured-move target, 11:00 force
# close), tested for whether it holds up (split-period check, per-
# instrument breakdown, correct z-score), not tuned to this data.
#
# GRANULARITY / FALSE-POSITIVE SANITY CHECK - see ict_po3_forex_dukascopy_
# backtest.py's header for the original reasoning: PO3's entry price and
# its stop are both derived from the SAME manipulation bar's own wick
# extreme, which single-step synthetic OHLC construction can spuriously
# couple together and produce a fake "edge" on data that has none. This
# script's stop/target checks (during trade management) are also wick-
# based, so the same risk was checked here rather than assumed away - see
# TestGranularityConvergence in the test file for a proper multi-substep,
# zero-drift random walk run through this exact backtest function at
# increasing intrabar resolution (1/4/16/64 substeps per bar). THE HONEST
# RESULT: unlike PO3/the Bollinger Band script (whose wick-touch-driven
# entries or exits show a dramatic coarse-resolution artifact that
# collapses as resolution increases), this script does NOT show that
# pattern - no large average-R edge appears at ANY tested resolution,
# plausibly because its entry is CLOSE-confirmed (not wick-triggered, and
# a driftless Gaussian random walk's close-crossing-a-level distribution
# is exactly resolution-invariant by construction) and its stop/target
# geometry comes from a MULTI-BAR range aggregate rather than one bar's
# own wick. The check does still find one real, modest, bounded effect
# (a narrower coarse-resolution Asian range pulling the measured-move
# target a little closer) - documented and bounded in the test rather
# than ignored. A negative result from actually running the check is
# still the honest thing to report, not a reason to have skipped it.
#
# NOTE ON MULTIPLE COMPARISONS: this project has shipped many strategies,
# several with their own parameter grid searches - a single script's
# z-score in isolation isn't strong evidence, since data-snooping risk
# compounds across every strategy and parameter combination tried
# project-wide, not just this one.
#
# NO COMMISSION, SPREAD, OR SLIPPAGE MODELED. Entry is a simulated market
# order at the close of the bar whose close breaks the Asian range - real
# fills would be worse (that close is the exact trigger price, not a level
# you'd realistically get filled right at during a fast London-open move).

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

# --- session windows, NY time ---
ASIAN_START = pd.Timestamp("19:00").time()       # Asian/Tokyo session range-building window start
ASIAN_END = pd.Timestamp("00:00").time()         # end of window (midnight - see header note: stays within one calendar day)
BREAKOUT_START = pd.Timestamp("02:00").time()    # London-open breakout window start
BREAKOUT_END = pd.Timestamp("05:00").time()
FORCE_CLOSE_TIME = pd.Timestamp("11:00").time()  # room through the London morning, short of the NY session overlap

STOP_BUFFER_PCT = 0.02         # % of price beyond the Asian range's opposite side - same scale as other scripts
TARGET_RANGE_MULT = 1.5        # measured-move target = this * the Asian range's own height
FALLBACK_REWARD_RISK = 2.0     # used only if the Asian range is degenerate (see MIN_RANGE_PCT)
MIN_RANGE_PCT = 0.02           # floor on the Asian range's height, as a % of entry price, before it's "degenerate"


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


def backtest_instrument(label, df):
    """Runs the Asian Range Breakout rule set over one instrument's 5-min bars. `df` must have
    Open/High/Low/Close columns and a tz-aware (NY time) DatetimeIndex. Returns a list of trade
    dicts."""
    highs = df["High"].tolist()
    lows = df["Low"].tolist()
    closes = df["Close"].tolist()
    times = df.index
    n = len(closes)

    trades = []
    current_day = None
    open_trade = None
    building_range = None    # {"high", "low"} accumulating during TODAY's 19:00-00:00 window - becomes TOMORROW's available_range
    available_range = None   # the range carried over from YESTERDAY's Asian window - what today's breakout window trades against
    traded_today = False

    for i in range(n):
        t = times[i]
        today = t.date()
        tod = t.time()

        if today != current_day:
            if open_trade is not None:
                # defensive only: a position should always be resolved by the prior day's
                # FORCE_CLOSE_TIME below, so this path shouldn't fire on real data - guards
                # against a data gap leaving a trade stranded across the day boundary.
                side = open_trade["side"]
                last_close = closes[i - 1]
                pnl = (last_close - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - last_close)
                trades.append({"side": side, "outcome": "FLAT", "r": pnl / open_trade["sl_distance"],
                                "date": times[i - 1].date()})
                open_trade = None
            current_day = today
            # whatever accumulated yesterday evening becomes today's tradeable range; if
            # yesterday's window never saw a bar (a data gap, e.g. a holiday), there's simply
            # nothing to trade against today
            available_range = dict(building_range) if building_range is not None else None
            building_range = None
            traded_today = False

        # --- manage an already-open trade: stop / target / forced close ---
        if open_trade is not None:
            side = open_trade["side"]
            stop, target = open_trade["stop"], open_trade["target"]
            hi, lo = highs[i], lows[i]
            hit_stop = lo <= stop if side == "LONG" else hi >= stop
            hit_target = hi >= target if side == "LONG" else lo <= target
            if hit_stop:
                trades.append({"side": side, "outcome": "SL", "r": -1.0, "date": today})
                open_trade = None
            elif hit_target:
                trades.append({"side": side, "outcome": "TP", "r": open_trade["reward_risk"], "date": today})
                open_trade = None
            elif tod >= FORCE_CLOSE_TIME:
                pnl = (closes[i] - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - closes[i])
                trades.append({"side": side, "outcome": "FLAT", "r": pnl / open_trade["sl_distance"], "date": today})
                open_trade = None

        # --- build today's Asian range (feeds TOMORROW's breakout window) ---
        if tod >= ASIAN_START:
            if building_range is None:
                building_range = {"high": highs[i], "low": lows[i]}
            else:
                building_range["high"] = max(building_range["high"], highs[i])
                building_range["low"] = min(building_range["low"], lows[i])
            continue

        # --- breakout window: trade against YESTERDAY's Asian range ---
        if available_range is None or traded_today or open_trade is not None:
            continue
        if not (BREAKOUT_START <= tod < BREAKOUT_END):
            continue

        range_high, range_low = available_range["high"], available_range["low"]
        broke_high = highs[i] > range_high
        broke_low = lows[i] < range_low

        if broke_high and broke_low:
            # same-bar double sweep - ambiguous which side broke first from OHLC alone, so this
            # is not a valid trigger at all: skip the day, matching PO3's identical handling
            traded_today = True
            continue

        if closes[i] > range_high:
            side = "LONG"
        elif closes[i] < range_low:
            side = "SHORT"
        else:
            continue   # no close-confirmed break yet this bar - keep watching within the window

        entry = closes[i]
        buffer_price = (STOP_BUFFER_PCT / 100.0) * entry
        range_height = range_high - range_low
        if side == "LONG":
            stop = range_low - buffer_price
            sl_distance = entry - stop
        else:
            stop = range_high + buffer_price
            sl_distance = stop - entry

        traded_today = True   # one trade per day: this trigger consumes today's slot either way
        if sl_distance <= 0:
            continue   # degenerate geometry (shouldn't occur given entry is already past the broken side) - skip defensively

        min_range = (MIN_RANGE_PCT / 100.0) * entry
        if range_height >= min_range:
            measured_move = range_height * TARGET_RANGE_MULT
            target = entry + measured_move if side == "LONG" else entry - measured_move
            reward_risk = measured_move / sl_distance
        else:
            reward_risk = FALLBACK_REWARD_RISK
            target = entry + sl_distance * reward_risk if side == "LONG" else entry - sl_distance * reward_risk

        open_trade = {"side": side, "entry": entry, "stop": stop, "target": target,
                      "sl_distance": sl_distance, "reward_risk": reward_risk}

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
        print("\nNo trades at all - the London breakout window never produced a clean close-confirmed "
              "break of the prior evening's Asian range in this data.")
        return

    total_r = sum(t["r"] for t in all_trades)
    n_trades = len(all_trades)
    print("\n" + "=" * 70)
    print(f"ASIAN RANGE BREAKOUT - {n_trades} trades, {FETCH_START.date()} to {FETCH_END.date()}")
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

    print("\nNo commission/spread/slippage modeled. Entry is a simulated market order at the close of the "
          "first bar that closes beyond the Asian range in the breakout window - real fills would be worse "
          "(that close is the exact trigger price, not a level you'd realistically get filled right at "
          "during a fast London-open move).")


if __name__ == "__main__":
    main()
