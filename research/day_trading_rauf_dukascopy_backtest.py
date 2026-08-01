# "Scam or Slam - Day Trading Rauf Strategy" backtest via Dukascopy, 2016-2025.
#
# A TradingView Pine v6 script (open-source, ~161 likes) found while sourcing
# strategies for this project - its own shown evidence is a small live
# strategy-tester result (+$568 / +1.14%), which is not meaningful evidence
# on its own. That's exactly why it's being re-derived and tested honestly
# here, same as every other found strategy in this repo.
#
# Distinct from ict_po3 and ict_silver_bullet: those enter immediately off a
# break (PO3) or off a sweep -> MSS -> FVG chain with a resting limit order
# (Silver Bullet). This strategy is sweep-then-wait-for-a-confirmation-
# candle-pattern - a third, structurally different entry style worth having
# in this project's ICT-adjacent lineup.
#
# OPERATIONALIZING the Pine source's default configuration only (see the
# "not built" list at the bottom of this header - this port intentionally
# does not implement every option the original script exposes):
#   1. TWO time-based ranges per day, NY time - a "London range" (01:12-
#      02:12) and a "New York range" (08:12-09:12), each just the plain
#      high/low of price during that window (an opening-range, computed
#      twice a day). Tracked completely independently of each other.
#   2. SWEEP: once a range's window ends ("ready"), watch for a bar whose
#      high/low breaks beyond the range's high/low. If a single bar breaks
#      BOTH sides, that's genuinely ambiguous which side was swept first
#      from OHLC data alone - the original script's default ("Ignore
#      Same-Bar Double Sweep") just resets and treats it as no sweep at
#      all, which this port preserves exactly rather than guessing a
#      resolution order. A sweep in the OPPOSITE direction of an already-
#      active setup flips it (fresh confirmation state); a sweep in the
#      SAME direction just extends how far the sweep extreme reaches.
#   3. ENTRY CONFIRMATION - "3 Candle Reversal" only (the script's default;
#      see the scope note below on why the other two modes - CHoCH, IFVG -
#      were not built this pass): after a low-sweep, watch for 3
#      consecutive bearish candles (close < open each) to form a local
#      high = max of their three highs; once that level exists, a LONG
#      triggers the instant a later candle's close breaks above it. Mirror
#      for a high-sweep: 3 consecutive bullish candles set a low = min of
#      their three lows, SHORT triggers on a close below it. The 3-candle
#      window doesn't have to start immediately at the sweep bar - it can
#      be any 3 consecutive same-direction candles occurring at least 3
#      bars after the sweep, matching the Pine source's bar_index gate.
#   4. STOP - "Sweep Extreme + Tick Buffer" mode: stop sits just beyond the
#      sweep's actual wick extreme. Since this isn't a tick-based venue,
#      the tick buffer is replaced with a small PERCENTAGE-of-price buffer
#      (STOP_BUFFER_PCT below) - the same tick-to-percent substitution
#      already used in ict_po3 and ict_silver_bullet's STOP_BUFFER_PCT.
#   5. TARGET - "Opposite Range Side": the far boundary of the SAME range
#      that got swept (swept the low -> target is that range's high, and
#      vice versa). A degenerate case the Pine source doesn't explicitly
#      guard against - price already past the opposite side by the time
#      the confirmation closes, which would make the "target" backwards
#      relative to entry - is skipped here rather than taken as a broken
#      trade; noted in the unit tests.
#   6. "1 Trade Per Range" frequency: each range can produce at most one
#      trade per day, independent of the other range; once used (a trade
#      was actually taken from it), it's done for the day regardless of
#      outcome. A confirmation that fires but can't be acted on (see next
#      point) does NOT consume the range's one-trade slot - only an
#      actual entry does, since nothing happened otherwise.
#   7. POSITION HANDLING - simplified to "Only Enter When Flat" (the
#      script's own alternative mode), NOT its default "Take Every Signal -
#      Close Current Trade First" mode, which force-closes an existing
#      trade to open a new one. Every other script in this project manages
#      one position at a time; if a confirmed signal from one range fires
#      while a trade from the OTHER range is still open, it's simply
#      skipped (not queued, not retried) - matching how the rest of this
#      project already works.
#   8. FORCED CLOSE at 16:00 NY regardless of stop/target status - marked
#      to market at that bar's close, recorded as outcome "FLAT" with
#      r = pnl / original_stop_distance, same accounting convention as
#      every other script here. Past that time, no new setups form either
#      (both ranges freeze for the rest of the day).
#
# NOT built this pass (see the task's own scope note - a single, honestly
# verified configuration beats five untested ones): CHoCH confirmation,
# IFVG confirmation, "Any Confirmation" mode, ATR stop, Fixed Points stop/
# target, Fixed R:R target, and the other three trade-frequency options.
# No parameter grid search either - this is one fixed rule set, tested for
# whether it holds up, not tuned to this data.

# !pip install --upgrade dukascopy-python tqdm -q   # uncomment this line in Colab

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

# --- the two time-based ranges, NY time (script defaults) ---
RANGES = [
    ("LONDON", pd.Timestamp("01:12").time(), pd.Timestamp("02:12").time()),
    ("NY", pd.Timestamp("08:12").time(), pd.Timestamp("09:12").time()),
]
FORCE_CLOSE_TIME = pd.Timestamp("16:00").time()

STOP_BUFFER_PCT = 0.02   # % of price beyond the sweep's wick extreme - see header note on the tick-to-% swap

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
    changing a strategy parameter below loads instantly instead of re-downloading everything."""
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
    highs = df["High"].tolist()
    lows = df["Low"].tolist()
    closes = df["Close"].tolist()
    opens = df["Open"].tolist()
    times = df.index
    n = len(closes)

    trades = []
    current_day = None
    open_trade = None
    range_states = {name: {"high": None, "low": None, "ready": False, "traded_today": False, "setup": None}
                     for name, _, _ in RANGES}

    for i in range(n):
        t = times[i]
        today = t.date()
        tod = t.time()

        if today != current_day:
            if open_trade is not None:
                # defensive only: a position should always be resolved by the prior day's 16:00
                # forced close below, so this path shouldn't fire on real data - guards against a
                # data gap leaving a trade stranded across the day boundary.
                side = open_trade["side"]
                last_close = closes[i - 1]
                pnl = (last_close - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - last_close)
                trades.append({"side": side, "outcome": "FLAT", "r": pnl / open_trade["sl_distance"],
                                "date": times[i - 1].date(), "range": open_trade["range"],
                                "stop_pct": open_trade["sl_distance"] / open_trade["entry"]})
                open_trade = None
            current_day = today
            for st in range_states.values():
                st["high"] = st["low"] = None
                st["ready"] = False
                st["traded_today"] = False
                st["setup"] = None

        # --- manage an already-open trade: stop / target / forced close ---
        if open_trade is not None:
            side = open_trade["side"]
            stop, target = open_trade["stop"], open_trade["target"]
            hi, lo = highs[i], lows[i]
            hit_stop = lo <= stop if side == "LONG" else hi >= stop
            hit_target = hi >= target if side == "LONG" else lo <= target
            stop_pct = open_trade["sl_distance"] / open_trade["entry"]
            if hit_stop:
                trades.append({"side": side, "outcome": "SL", "r": -1.0, "date": today,
                               "range": open_trade["range"], "stop_pct": stop_pct})
                open_trade = None
            elif hit_target:
                trades.append({"side": side, "outcome": "TP", "r": open_trade["reward_risk"], "date": today,
                                "range": open_trade["range"], "stop_pct": stop_pct})
                open_trade = None
            elif tod >= FORCE_CLOSE_TIME:
                pnl = (closes[i] - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - closes[i])
                trades.append({"side": side, "outcome": "FLAT", "r": pnl / open_trade["sl_distance"], "date": today,
                                "range": open_trade["range"], "stop_pct": stop_pct})
                open_trade = None

        # --- range building, sweep detection, and entry confirmation, per range ---
        for name, win_start, win_end in RANGES:
            rs = range_states[name]

            if win_start <= tod < win_end:
                if rs["high"] is None:
                    rs["high"], rs["low"] = highs[i], lows[i]
                else:
                    rs["high"] = max(rs["high"], highs[i])
                    rs["low"] = min(rs["low"], lows[i])
                continue

            if not rs["ready"]:
                if rs["high"] is not None:
                    rs["ready"] = True
                else:
                    continue   # this range never saw a bar in its window today (data gap) - nothing to trade

            if rs["traded_today"]:
                continue

            if tod >= FORCE_CLOSE_TIME:
                rs["setup"] = None   # trading day over - no new setups form this late
                continue

            range_high, range_low = rs["high"], rs["low"]
            broke_high = highs[i] > range_high
            broke_low = lows[i] < range_low

            if broke_high and broke_low:
                # same-bar double sweep - ambiguous which side broke first from OHLC alone, so per
                # the script's default ("Ignore Same-Bar Double Sweep") this is just not a valid
                # sweep at all: reset, no setup carried forward
                rs["setup"] = None
            elif broke_low:
                if rs["setup"] is None or rs["setup"]["side"] == -1:
                    rs["setup"] = {"side": 1, "sweep_extreme": lows[i], "setup_bar": i, "three_level": None}
                else:
                    rs["setup"]["sweep_extreme"] = min(rs["setup"]["sweep_extreme"], lows[i])
            elif broke_high:
                if rs["setup"] is None or rs["setup"]["side"] == 1:
                    rs["setup"] = {"side": -1, "sweep_extreme": highs[i], "setup_bar": i, "three_level": None}
                else:
                    rs["setup"]["sweep_extreme"] = max(rs["setup"]["sweep_extreme"], highs[i])

            setup = rs["setup"]
            if setup is None:
                continue
            bars_since = i - setup["setup_bar"]

            if setup["side"] == 1:
                if setup["three_level"] is None and bars_since >= 3:
                    if closes[i] < opens[i] and closes[i - 1] < opens[i - 1] and closes[i - 2] < opens[i - 2]:
                        setup["three_level"] = max(highs[i], highs[i - 1], highs[i - 2])
                if setup["three_level"] is not None and closes[i] > setup["three_level"]:
                    entry = closes[i]
                    buffer_price = (STOP_BUFFER_PCT / 100.0) * entry
                    stop = setup["sweep_extreme"] - buffer_price
                    target = range_high
                    sl_distance = entry - stop
                    rs["setup"] = None   # one-shot: whether or not the trade is actually taken below
                    if sl_distance > 0 and target > entry and open_trade is None:
                        reward_risk = (target - entry) / sl_distance
                        open_trade = {"side": "LONG", "entry": entry, "stop": stop, "target": target,
                                      "sl_distance": sl_distance, "reward_risk": reward_risk, "range": name}
                        rs["traded_today"] = True
            else:
                if setup["three_level"] is None and bars_since >= 3:
                    if closes[i] > opens[i] and closes[i - 1] > opens[i - 1] and closes[i - 2] > opens[i - 2]:
                        setup["three_level"] = min(lows[i], lows[i - 1], lows[i - 2])
                if setup["three_level"] is not None and closes[i] < setup["three_level"]:
                    entry = closes[i]
                    buffer_price = (STOP_BUFFER_PCT / 100.0) * entry
                    stop = setup["sweep_extreme"] + buffer_price
                    target = range_low
                    sl_distance = stop - entry
                    rs["setup"] = None
                    if sl_distance > 0 and target < entry and open_trade is None:
                        reward_risk = (entry - target) / sl_distance
                        open_trade = {"side": "SHORT", "entry": entry, "stop": stop, "target": target,
                                      "sl_distance": sl_distance, "reward_risk": reward_risk, "range": name}
                        rs["traded_today"] = True

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
        print("\nNo trades at all - sweeps followed by a valid 3-candle-reversal confirmation never "
              "occurred in this data.")
        return

    total_r = sum(t["r"] for t in all_trades)
    n_trades = len(all_trades)
    print("\n" + "=" * 70)
    print(f"SCAM OR SLAM (Day Trading Rauf) - {n_trades} trades, {FETCH_START.date()} to {FETCH_END.date()}")
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

    per_range = {}
    for t in all_trades:
        per_range.setdefault(t["range"], []).append(t["r"])
    print("\nPer range:")
    for name, rs in per_range.items():
        print(f"  {name:10s}: {len(rs):4d} trades, {sum(rs):+8.2f}R")

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
          "bar whose close breaks the 3-candle-reversal level - real fills would be worse (that close is "
          "the exact trigger price, not a level you'd realistically get filled right at).")


if __name__ == "__main__":
    main()
