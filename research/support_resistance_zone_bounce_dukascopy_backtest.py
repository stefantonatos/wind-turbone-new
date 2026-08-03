# Support/resistance "zone bounce" backtest - the simple, discretionary
# style described as "draw boxes where there's support and resistance on
# the daily/weekly/4H chart, buy or sell when price gets there."
#
# This is a DIFFERENT, simpler idea from the "hedging" mechanics found
# researching Nick Shawn/MissionFX online (open an opposing position when
# a trade goes wrong, rather than take the stop) - that hedging structure
# is not what's coded here, both because it doesn't reduce to a clean
# R-multiple backtest (the real risk is correlated/tail drawdown, not
# per-trade win rate) and because of the separate, serious fraud
# allegations against that operation flagged earlier. What's tested here
# is just the plain "trade the zone" idea: no hedging, no averaging down -
# a normal stop-loss if the zone breaks.
#
# OPERATIONALIZING "draw boxes and trade them" objectively:
#   1. ZONES come from confirmed swing highs/lows (fractal pivots: a bar
#      is a swing point only if it's the max/min of the ZONE_PIVOT_
#      LOOKBACK bars on BOTH sides) with an ATR-based width, so each zone
#      is a small box around the pivot price, not a single line - this is
#      the direct translation of "drawing a box around a support/
#      resistance level" rather than trading an exact price.
#   2. A swing pivot at bar i is only usable starting at bar i +
#      ZONE_PIVOT_LOOKBACK, once the bars AFTER it exist to confirm it was
#      really a local high/low - using it any earlier would be lookahead
#      (checked explicitly in the unit tests below).
#   3. ENTRY: price dips into a support zone and the bar's CLOSE holds
#      inside/above it (a held test, not a breakdown-in-progress) -> long.
#      Mirror for resistance -> short. Stop beyond the zone. Target is the
#      nearest opposite-type zone in the trade's direction, or a fixed
#      2:1 R:R if none exists within a reasonable distance.
#   4. A zone that gets closed through (beyond its far edge, past a small
#      buffer) is invalidated and removed - broken support isn't support
#      anymore, matching how a real chart trader would erase that box.
#
# Daily bars are used as the primary timeframe (the first one mentioned),
# not weekly/4H - a reasonable middle ground, noted here rather than
# silently assumed. Same instrument universe as the ORB-indices work
# (6 indices) plus the 7 FX majors, 2010-2025 via Dukascopy.

# !pip install --upgrade dukascopy-python -q   # uncomment this line in Colab

import datetime

import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

INSTRUMENTS = [
    ("EURUSD", dki.INSTRUMENT_FX_MAJORS_EUR_USD),
    ("GBPUSD", dki.INSTRUMENT_FX_MAJORS_GBP_USD),
    ("USDJPY", dki.INSTRUMENT_FX_MAJORS_USD_JPY),
    ("USDCHF", dki.INSTRUMENT_FX_MAJORS_USD_CHF),
    ("USDCAD", dki.INSTRUMENT_FX_MAJORS_USD_CAD),
    ("AUDUSD", dki.INSTRUMENT_FX_MAJORS_AUD_USD),
    ("NZDUSD", dki.INSTRUMENT_FX_MAJORS_NZD_USD),
    ("SP500", dki.INSTRUMENT_IDX_AMERICA_E_SANDP_500),
    ("NASDAQ100", dki.INSTRUMENT_IDX_AMERICA_E_NQ_100),
    ("DOWJONES", dki.INSTRUMENT_IDX_AMERICA_E_D_J_IND),
    ("DAX", dki.INSTRUMENT_IDX_EUROPE_E_DAAX),
    ("FTSE100", dki.INSTRUMENT_IDX_EUROPE_E_FUTSEE_100),
    ("NIKKEI225", dki.INSTRUMENT_IDX_ASIA_E_N225JAP),
]

FETCH_START = datetime.datetime(2010, 1, 1)
FETCH_END = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_DAY_1
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

ATR_LEN = 14
ZONE_PIVOT_LOOKBACK = 5          # bars on EACH side required to confirm a swing high/low
ZONE_WIDTH_ATR_MULT = 0.25       # zone half-width, as a fraction of ATR - makes it a box, not a line
ZONE_BREAK_BUFFER_ATR_MULT = 0.1 # how far past a zone's far edge counts as "broken", not just tested
MAX_TARGET_DISTANCE_ATR_MULT = 20.0   # cap on how far away an opposite zone can be and still count as the target
FALLBACK_REWARD_RISK = 2.0       # used only when no opposite zone exists within the cap
MAX_HOLD_BARS = 60               # ~3 months of daily bars - this is meant to be a swing/position style, not a scalp

# Illustrative round-trip cost scenarios, as a percentage of entry price - NOT measured real spread
# data (Dukascopy's OHLC endpoint doesn't expose historical bid/ask spread), just a few bracketing
# assumptions to see how much cost this edge can absorb before it disappears. 0.01% is roughly an
# ECN-style FX major spread; 0.05% is closer to a wider retail/CFD spread on less liquid instruments.
COST_PCT_SCENARIOS = [0.0, 0.01, 0.03, 0.05]


def fetch_daily_ohlc(instrument_const):
    df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, FETCH_START, FETCH_END)
    if df.empty:
        return None
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def compute_atr_series(highs, lows, closes, length):
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


def find_confirmed_pivots(highs, lows, lookback):
    """Returns two dicts: {confirmed_at_index: (pivot_index, price)} for
    swing highs and swing lows. A pivot at index i is only placed at key
    i + lookback - the earliest point its formation could actually be
    known, since confirming it requires the `lookback` bars AFTER it."""
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


def backtest_instrument(label, df):
    highs, lows, closes = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
    times = df.index
    n = len(closes)
    atr = compute_atr_series(highs, lows, closes, ATR_LEN)

    swing_highs_by_confirm_idx, swing_lows_by_confirm_idx = find_confirmed_pivots(highs, lows, ZONE_PIVOT_LOOKBACK)

    # active zones: list of dicts {top, bottom, kind: "support"/"resistance", in_trade: bool}
    active_zones = []
    trades = []
    open_trade = None
    i = 0
    while i < n:
        current_atr = atr[i]

        # newly confirmed pivots become active zones starting this bar
        if i in swing_highs_by_confirm_idx and current_atr:
            _, price = swing_highs_by_confirm_idx[i]
            half_width = ZONE_WIDTH_ATR_MULT * current_atr
            active_zones.append({"top": price + half_width, "bottom": price - half_width,
                                  "kind": "resistance", "in_trade": False})
        if i in swing_lows_by_confirm_idx and current_atr:
            _, price = swing_lows_by_confirm_idx[i]
            half_width = ZONE_WIDTH_ATR_MULT * current_atr
            active_zones.append({"top": price + half_width, "bottom": price - half_width,
                                  "kind": "support", "in_trade": False})

        if current_atr:
            break_buffer = ZONE_BREAK_BUFFER_ATR_MULT * current_atr
            still_active = []
            for z in active_zones:
                broken = (closes[i] < z["bottom"] - break_buffer if z["kind"] == "support"
                          else closes[i] > z["top"] + break_buffer)
                if not broken:
                    still_active.append(z)
            active_zones = still_active

        # manage an open trade forward
        if open_trade is not None:
            hi, lo = highs[i], lows[i]
            side, stop, target = open_trade["side"], open_trade["stop"], open_trade["target"]
            hit_stop = lo <= stop if side == "LONG" else hi >= stop
            hit_target = hi >= target if side == "LONG" else lo <= target
            bars_held = i - open_trade["entry_index"]
            if hit_stop:
                open_trade["outcome"], open_trade["exit_r"] = "SL", -1.0
            elif hit_target:
                open_trade["outcome"], open_trade["exit_r"] = "TP", open_trade["reward_risk"]
            elif bars_held >= MAX_HOLD_BARS:
                pnl = (closes[i] - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - closes[i])
                open_trade["outcome"], open_trade["exit_r"] = "FLAT", pnl / open_trade["sl_distance"]

            if open_trade["outcome"] is not None:
                trades.append({"side": side, "outcome": open_trade["outcome"], "r": open_trade["exit_r"],
                                "date": times[i].date(), "entry": open_trade["entry"],
                                "sl_distance": open_trade["sl_distance"]})
                for z in active_zones:
                    if z is open_trade["zone"]:
                        z["in_trade"] = False
                open_trade = None
                i += 1
                continue
            i += 1
            continue

        if not current_atr:
            i += 1
            continue

        # look for a new entry: a bar touching an untouched support/resistance zone and holding it
        entry_made = False
        for z in active_zones:
            if z["in_trade"]:
                continue
            if z["kind"] == "support" and lows[i] <= z["top"] and closes[i] >= z["bottom"]:
                side = "LONG"
            elif z["kind"] == "resistance" and highs[i] >= z["bottom"] and closes[i] <= z["top"]:
                side = "SHORT"
            else:
                continue

            entry_price = closes[i]
            buffer_price = ZONE_BREAK_BUFFER_ATR_MULT * current_atr
            if side == "LONG":
                stop = z["bottom"] - buffer_price
                sl_distance = entry_price - stop
            else:
                stop = z["top"] + buffer_price
                sl_distance = stop - entry_price
            if sl_distance <= 0:
                continue

            opposite_kind = "resistance" if side == "LONG" else "support"
            candidates = [oz for oz in active_zones if oz["kind"] == opposite_kind and oz is not z and
                          ((oz["bottom"] > entry_price) if side == "LONG" else (oz["top"] < entry_price))]
            max_dist = MAX_TARGET_DISTANCE_ATR_MULT * current_atr
            if side == "LONG":
                candidates = [oz for oz in candidates if oz["bottom"] - entry_price <= max_dist]
            else:
                candidates = [oz for oz in candidates if entry_price - oz["top"] <= max_dist]

            if candidates:
                nearest = (min(candidates, key=lambda oz: oz["bottom"]) if side == "LONG"
                           else max(candidates, key=lambda oz: oz["top"]))
                target = nearest["bottom"] if side == "LONG" else nearest["top"]
                reward_risk = abs(target - entry_price) / sl_distance
            else:
                reward_risk = FALLBACK_REWARD_RISK
                target = entry_price + sl_distance * reward_risk if side == "LONG" else entry_price - sl_distance * reward_risk

            z["in_trade"] = True
            open_trade = {"side": side, "entry": entry_price, "stop": stop, "target": target,
                          "sl_distance": sl_distance, "reward_risk": reward_risk, "entry_index": i,
                          "zone": z, "outcome": None, "exit_r": None}
            entry_made = True
            break

        i += 1

    return trades


def main():
    years = (FETCH_END - FETCH_START).days / 365
    print(f"Downloading {len(INSTRUMENTS)} instruments (daily bars) from Dukascopy over ~{years:.0f} years "
          f"({FETCH_START.date()} to {FETCH_END.date()}) - daily data is small, expect a few minutes.\n")

    all_trades = []
    per_instrument_counts = {}
    for label, instrument_const in INSTRUMENTS:
        print(f"{label}...", end=" ")
        try:
            df = fetch_daily_ohlc(instrument_const)
        except Exception as exc:
            print(f"failed ({exc})")
            continue
        if df is None or len(df) < 100:
            print("no/insufficient data")
            continue
        trades = backtest_instrument(label, df)
        for t in trades:
            t["instrument"] = label
        all_trades.extend(trades)
        per_instrument_counts[label] = trades
        print(f"{len(df)} daily bars, {len(trades)} trades")

    if not all_trades:
        print("No trades at all - check output above.")
        return

    total_r = sum(t["r"] for t in all_trades)
    n_trades = len(all_trades)
    print("\n" + "=" * 70)
    print(f"SUPPORT/RESISTANCE ZONE BOUNCE - {n_trades} trades, {FETCH_START.date()} to {FETCH_END.date()}")
    print("=" * 70)
    print(f"Total R: {total_r:+.2f}   Avg R/trade: {total_r/n_trades:+.4f}")

    tp = sum(1 for t in all_trades if t["outcome"] == "TP")
    sl = sum(1 for t in all_trades if t["outcome"] == "SL")
    flat = sum(1 for t in all_trades if t["outcome"] == "FLAT")
    print(f"Outcome breakdown: TP {tp} ({tp/n_trades*100:.1f}%)  SL {sl} ({sl/n_trades*100:.1f}%)  "
          f"FLAT {flat} ({flat/n_trades*100:.1f}%)")

    print("\nPer instrument:")
    for label, trades in sorted(per_instrument_counts.items(), key=lambda kv: -sum(t["r"] for t in kv[1])):
        r = sum(t["r"] for t in trades)
        print(f"  {label:10s}: {len(trades):4d} trades, {r:+8.2f}R")

    all_r = [t["r"] for t in all_trades]
    if n_trades >= 2:
        std_r = np.std(all_r, ddof=1)
        z = (total_r / n_trades) / (std_r / (n_trades ** 0.5)) if std_r > 0 else 0.0
    else:
        z = 0.0
    print(f"\nApprox z-score: {z:.2f} (rule of thumb: |z| > 1.96 for ~95% confidence this isn't chance - "
          f"but R-multiples here are heavily skewed by variable-target-distance wins, so treat this as "
          f"suggestive, not exact - a normal-distribution z-score is an approximation on skewed data.)")
    if n_trades < 100:
        print(f"CAVEAT: only {n_trades} trades - too few to trust regardless of the z-score.")

    win_rs = sorted(t["r"] for t in all_trades if t["outcome"] == "TP")
    if win_rs:
        median_win_r = win_rs[len(win_rs) // 2]
        mean_win_r = sum(win_rs) / len(win_rs)
        print(f"\nMedian winning trade's R-multiple: {median_win_r:.2f} (vs mean win of {mean_win_r:.2f} - "
              f"a big gap here means a few huge-target wins are doing a lot of the work, not a broadly "
              f"repeatable payout).")

    all_trades_sorted = sorted(all_trades, key=lambda t: t["date"])
    midpoint_date = all_trades_sorted[len(all_trades_sorted) // 2]["date"]
    first_half = [t for t in all_trades_sorted if t["date"] < midpoint_date]
    second_half = [t for t in all_trades_sorted if t["date"] >= midpoint_date]
    print(f"\nSPLIT-PERIOD CHECK (does the SAME rule, same fixed parameters, hold up in both halves "
          f"of the sample, not just overall):")
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
    print(f"  If one half is strongly positive and the other flat or negative, that's the same warning "
          f"sign flagged elsewhere in this project - don't trust the combined number over both halves.")

    print(f"\nNOTE ON MULTIPLE COMPARISONS: this project has shipped many strategies, several with "
          f"their own parameter grid searches - a single script's z-score in isolation isn't strong "
          f"evidence, since data-snooping risk compounds across every strategy and parameter "
          f"combination tried project-wide, not just this one.")

    print(f"\nCOST SENSITIVITY (illustrative round-trip spread scenarios, NOT measured real spread data - "
          f"see COST_PCT_SCENARIOS comment):")
    for cost_pct in COST_PCT_SCENARIOS:
        cost_adjusted_total = sum(t["r"] - (cost_pct / 100.0) * t["entry"] / t["sl_distance"] for t in all_trades)
        print(f"  {cost_pct:.2f}% round-trip cost: {cost_adjusted_total:+.2f}R total, "
              f"{cost_adjusted_total/n_trades:+.4f}R/trade")
    print(f"  If the total goes negative well before 0.05%, this edge is too thin to survive real "
          f"execution costs - check your actual broker's spread on each instrument against these numbers.")

    print("\nEntry is simulated at the daily close of the bar that tests the zone - a real discretionary "
          "trader would place a limit order inside the zone itself, which could fill at a better or worse "
          "price depending on how the bar unfolds intraday.")


if __name__ == "__main__":
    main()
