# ICT "Power of Three" (PO3 / Accumulation-Manipulation-Distribution)
# backtest via Dukascopy, 2016-2025.
#
# Distinct from ict_silver_bullet_forex_dukascopy_backtest.py, not a
# rehash under a new name: Silver Bullet requires a sweep -> market
# structure shift -> fair value gap -> retest chain inside three fixed
# 1-hour kill zones, and trades WITH the structure shift's direction.
# PO3 is simpler and directly opposite in spirit - it trades AGAINST
# whichever direction manipulates first, betting the "real" move
# (distribution) reverses it. No MSS, no FVG requirement.
#
# OPERATIONALIZING "Accumulation -> Manipulation -> Distribution":
#   1. ACCUMULATION: the range (high/low) formed during ACCUMULATION_
#      START-ACCUMULATION_END (NY time), a same-day simplification of
#      ICT's usual "prior evening's Asian session" framing - crossing
#      midnight to use the literal prior session adds real complexity
#      for a small conceptual gain, so this uses an early-hours same-day
#      window instead. Noted here rather than silently assumed.
#   2. MANIPULATION: within MANIPULATION_START-MANIPULATION_END (right
#      after accumulation, London-session hours), the FIRST bar whose
#      high/low breaks the accumulation range in either direction is the
#      manipulation move - no "closes back inside" confirmation required
#      (unlike the Silver Bullet's sweep test), since PO3's entire
#      premise is that this break is a fake-out to be faded, not
#      confirmed.
#   3. DISTRIBUTION: enter AGAINST the manipulation's direction (it broke
#      the high -> go short, betting on reversal). Stop beyond the
#      manipulation's extreme, target the opposite side of the
#      accumulation range, falling back to a fixed 2:1 R:R if that's
#      already surpassed or too close.
#
# One trade per day per instrument. No parameter grid search - like
# trend_following_momentum_dukascopy_backtest.py, this is a single fixed
# rule tested for temporal consistency (split-period check) rather than
# tuned, since there's nothing here worth curve-fitting to begin with.

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
    ("XAUUSD", dki.INSTRUMENT_FX_METALS_XAU_USD),
]

FETCH_START = datetime.datetime(2016, 1, 1)
FETCH_END = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_MIN_5
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

# --- session windows, all NY time (same-day simplification - see header note) ---
ACCUMULATION_START = pd.Timestamp("00:00").time()
ACCUMULATION_END = pd.Timestamp("05:00").time()
MANIPULATION_START = pd.Timestamp("05:00").time()
MANIPULATION_END = pd.Timestamp("08:00").time()
DISTRIBUTION_END = pd.Timestamp("17:00").time()   # hold through NY session close

STOP_BUFFER_PCT = 0.02          # small buffer beyond the manipulation's wick, same scale as other scripts
FALLBACK_REWARD_RISK = 2.0      # used only if the opposite range edge is already passed or too close
MIN_RANGE_PCT = 0.02            # floor on the accumulation range size, guards against a degenerate near-zero range


def to_ny_time(index):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert("America/New_York")


def fetch_instrument_data(instrument_const):
    df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, FETCH_START, FETCH_END)
    if df.empty:
        return None
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df.index = to_ny_time(df.index)
    return df


def backtest_instrument(label, df):
    highs, lows, closes = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
    times = df.index
    n = len(closes)

    trades = []
    current_day = None
    acc_high = acc_low = None
    manipulated = False
    traded_today = False
    i = 0
    while i < n:
        t = times[i]
        today = t.date()
        tod = t.time()

        if today != current_day:
            current_day = today
            acc_high = acc_low = None
            manipulated = False
            traded_today = False

        if traded_today:
            i += 1
            continue

        if ACCUMULATION_START <= tod < ACCUMULATION_END:
            acc_high = highs[i] if acc_high is None else max(acc_high, highs[i])
            acc_low = lows[i] if acc_low is None else min(acc_low, lows[i])
            i += 1
            continue

        if acc_high is None:
            i += 1
            continue

        if tod >= DISTRIBUTION_END:
            traded_today = True
            i += 1
            continue

        if not (MANIPULATION_START <= tod < MANIPULATION_END):
            i += 1
            continue

        if manipulated:
            i += 1
            continue

        acc_range = acc_high - acc_low
        price = closes[i]
        broke_high = highs[i] > acc_high
        broke_low = lows[i] < acc_low

        if not broke_high and not broke_low:
            i += 1
            continue

        manipulated = True

        # if BOTH sides broke on the same bar (a wide/volatile bar), skip - genuinely ambiguous which
        # direction the "manipulation" actually was, not a case PO3's model has a clean answer for
        if broke_high and broke_low:
            traded_today = True
            i += 1
            continue

        side = "SHORT" if broke_high else "LONG"
        entry = price
        manipulation_extreme = highs[i] if side == "SHORT" else lows[i]
        buffer_price = (STOP_BUFFER_PCT / 100.0) * entry

        if side == "SHORT":
            stop = manipulation_extreme + buffer_price
        else:
            stop = manipulation_extreme - buffer_price
        sl_distance = abs(entry - stop)
        min_sl = (MIN_RANGE_PCT / 100.0) * entry
        if sl_distance < min_sl:
            sl_distance = min_sl
            stop = entry - sl_distance if side == "LONG" else entry + sl_distance

        opposite_edge = acc_low if side == "SHORT" else acc_high
        reaches_opposite = (opposite_edge < entry) if side == "SHORT" else (opposite_edge > entry)
        if reaches_opposite and abs(opposite_edge - entry) > sl_distance:
            target = opposite_edge
            reward_risk = abs(target - entry) / sl_distance
        else:
            reward_risk = FALLBACK_REWARD_RISK
            target = entry - sl_distance * reward_risk if side == "SHORT" else entry + sl_distance * reward_risk

        traded_today = True
        outcome, exit_r = None, None
        j = i + 1
        while j < n and times[j].date() == today and times[j].time() < DISTRIBUTION_END:
            hi, lo = highs[j], lows[j]
            hit_stop = hi >= stop if side == "SHORT" else lo <= stop
            hit_target = lo <= target if side == "SHORT" else hi >= target
            if hit_stop:
                outcome, exit_r = "SL", -1.0
                break
            if hit_target:
                outcome, exit_r = "TP", reward_risk
                break
            j += 1
        else:
            j = min(j, n - 1)

        if outcome is None:
            last_close = closes[j]
            pnl = (entry - last_close) if side == "SHORT" else (last_close - entry)
            outcome, exit_r = "FLAT", pnl / sl_distance

        trades.append({"side": side, "outcome": outcome, "r": exit_r, "date": today})
        i = j + 1

    return trades


def main():
    years = (FETCH_END - FETCH_START).days / 365
    print(f"Downloading {len(INSTRUMENTS)} instruments from Dukascopy over ~{years:.0f} years "
          f"({FETCH_START.date()} to {FETCH_END.date()}) - expect roughly 10-20 minutes.\n")

    data = {}
    for label, instrument_const in INSTRUMENTS:
        print(f"{label}...", end=" ")
        try:
            df = fetch_instrument_data(instrument_const)
        except Exception as exc:
            print(f"failed ({exc})")
            continue
        if df is None:
            print("no data")
            continue
        data[label] = df
        print(f"{len(df)} bars")

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
        print("\nNo trades at all - the manipulation window never produced a clean single-direction break.")
        return

    total_r = sum(t["r"] for t in all_trades)
    n_trades = len(all_trades)
    print("\n" + "=" * 70)
    print(f"ICT POWER OF THREE (PO3) - {n_trades} trades, {FETCH_START.date()} to {FETCH_END.date()}")
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
          "first bar that breaks the accumulation range - real fills would be worse (the actual break is "
          "often a fast wick, not a level you'd realistically get filled right at).")


if __name__ == "__main__":
    main()
