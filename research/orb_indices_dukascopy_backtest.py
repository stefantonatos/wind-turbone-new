# ORB (Opening Range Breakout) tested on real stock indices via
# Dukascopy, years of data instead of the 60-day Yahoo Finance cap used
# in orb_multi_stock_backtest.py (which tested individual US stocks, not
# indices). Research (see quantconnect/orb.py's header) says ORB is
# better-documented to work on indices/futures than on individual stocks
# or forex - this is the first test in this project that actually uses
# that intended instrument class.
#
# Same core rules as quantconnect/orb.py (close-confirmed breakout of the
# opening range, SL = opposite side of the range floored as a % of
# price, TP = SL x REWARD_RISK, one trade/day, session-end flatten),
# ported to plain Python/pandas here rather than a QC algorithm. Skips
# the impulse/volume/regime filters quantconnect/orb.py later added, to
# keep this initial index port a manageable, directly-comparable-to-
# research baseline - those could be added the same way if this looks
# promising.
#
# Each index has its own home-market session, unlike round-the-clock
# forex, so this needs a real per-index (timezone, local session-open
# time) mapping - not just one "London session" gate like the forex
# scripts use.

# !pip install --upgrade dukascopy-python -q   # uncomment this line in Colab

import datetime

import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

# (label, instrument constant, IANA timezone, local session-open time)
INDICES = [
    ("SP500", dki.INSTRUMENT_IDX_AMERICA_E_SANDP_500, "America/New_York", pd.Timestamp("09:30").time()),
    ("NASDAQ100", dki.INSTRUMENT_IDX_AMERICA_E_NQ_100, "America/New_York", pd.Timestamp("09:30").time()),
    ("DOWJONES", dki.INSTRUMENT_IDX_AMERICA_E_D_J_IND, "America/New_York", pd.Timestamp("09:30").time()),
    ("DAX", dki.INSTRUMENT_IDX_EUROPE_E_DAAX, "Europe/Berlin", pd.Timestamp("09:00").time()),
    ("FTSE100", dki.INSTRUMENT_IDX_EUROPE_E_FUTSEE_100, "Europe/London", pd.Timestamp("08:00").time()),
    ("NIKKEI225", dki.INSTRUMENT_IDX_ASIA_E_N225JAP, "Asia/Tokyo", pd.Timestamp("09:00").time()),
]

START_DATE = datetime.datetime(2024, 1, 1)
END_DATE = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_MIN_5
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

RANGE_MINUTES = 15
ENTRY_WINDOW_MINUTES = 180
SESSION_HOLD_HOURS = 8         # flatten this many hours after the session opens
ENTRY_BUFFER_PCT = 0.02        # close must clear the range by this % of price (indices span 4,000-40,000+, so % not a flat point value)
REWARD_RISK = 1.0              # classic measured-move target
MIN_RANGE_PCT = 0.05           # SL distance floor as % of price
REVERSE_SIGNALS = False


def to_local_time(index, tz_name):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(tz_name)


def backtest_index(label, instrument_const, tz_name, session_start):
    df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, START_DATE, END_DATE)
    if df.empty:
        return []
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df.index = to_local_time(df.index, tz_name)

    highs, lows, closes = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
    times = df.index
    n = len(closes)

    range_end = (datetime.datetime.combine(datetime.date.min, session_start)
                 + datetime.timedelta(minutes=RANGE_MINUTES)).time()
    entry_end = (datetime.datetime.combine(datetime.date.min, session_start)
                 + datetime.timedelta(minutes=RANGE_MINUTES + ENTRY_WINDOW_MINUTES)).time()
    session_end = (datetime.datetime.combine(datetime.date.min, session_start)
                   + datetime.timedelta(hours=SESSION_HOLD_HOURS)).time()

    trades = []
    current_day = None
    range_high = range_low = None
    traded_today = False
    i = 0
    while i < n:
        t = times[i]
        today = t.date()
        tod = t.time()

        if today != current_day:
            current_day = today
            range_high = range_low = None
            traded_today = False

        if session_end < session_start and tod < session_start:
            pass  # session wraps past midnight - not handled specially, rare for these 6 indices' home sessions

        if traded_today:
            i += 1
            continue

        if session_start <= tod < range_end:
            range_high = highs[i] if range_high is None else max(range_high, highs[i])
            range_low = lows[i] if range_low is None else min(range_low, lows[i])
            i += 1
            continue

        if range_high is None:
            i += 1
            continue

        if tod >= entry_end:
            traded_today = True
            i += 1
            continue

        buffer_price = (ENTRY_BUFFER_PCT / 100.0) * closes[i]
        price = closes[i]
        buy_setup = price > range_high + buffer_price
        sell_setup = price < range_low - buffer_price

        if REVERSE_SIGNALS:
            buy_setup, sell_setup = sell_setup, buy_setup

        if not buy_setup and not sell_setup:
            i += 1
            continue

        range_size = range_high - range_low
        sl_distance = max(range_size, (MIN_RANGE_PCT / 100.0) * price)
        tp_distance = sl_distance * REWARD_RISK
        side = "LONG" if buy_setup else "SHORT"
        entry = price
        stop = entry - sl_distance if side == "LONG" else entry + sl_distance
        target = entry + tp_distance if side == "LONG" else entry - tp_distance

        traded_today = True
        outcome, exit_r = None, None
        j = i + 1
        while j < n and times[j].date() == today and times[j].time() < session_end:
            hi, lo = highs[j], lows[j]
            hit_stop = lo <= stop if side == "LONG" else hi >= stop
            hit_target = hi >= target if side == "LONG" else lo <= target
            if hit_stop:
                outcome, exit_r = "SL", -1.0
                break
            if hit_target:
                outcome, exit_r = "TP", REWARD_RISK
                break
            j += 1
        else:
            j = min(j, n - 1)

        if outcome is None:
            last_close = closes[j]
            pnl = (last_close - entry) if side == "LONG" else (entry - last_close)
            outcome, exit_r = "FLAT", pnl / sl_distance

        trades.append({"index": label, "side": side, "outcome": outcome, "r": exit_r})
        i = j + 1

    return trades


def main():
    all_trades = []
    for label, instrument_const, tz_name, session_start in INDICES:
        print(f"{label}...", end=" ")
        try:
            trades = backtest_index(label, instrument_const, tz_name, session_start)
        except Exception as exc:
            print(f"failed ({exc})")
            continue
        if not trades:
            print("0 trades (or no data)")
            continue
        all_trades.extend(trades)
        total_r = sum(t["r"] for t in trades)
        print(f"{len(trades)} trades, {total_r:+.2f}R")

    print("\n" + "=" * 60)
    print(f"TOTAL: {len(all_trades)} trades across {len(INDICES)} indices, "
          f"{START_DATE.date()} to {END_DATE.date()}, REWARD_RISK={REWARD_RISK}  REVERSE_SIGNALS={REVERSE_SIGNALS}")
    print("=" * 60)

    if not all_trades:
        print("No trades at all - check ticker output above for download failures.")
        return

    df = pd.DataFrame(all_trades)
    total_r = df["r"].sum()
    tp_count = (df["outcome"] == "TP").sum()
    sl_count = (df["outcome"] == "SL").sum()
    flat_df = df[df["outcome"] == "FLAT"]
    flat_pos = (flat_df["r"] > 0).sum()
    flat_neg = len(flat_df) - flat_pos

    print(f"Total: {total_r:+.2f}R   Average: {total_r/len(df):+.3f}R/trade   <- this decides profitability, not win rate")
    print(f"\nOutcome breakdown ({len(df)} trades):")
    print(f"  TP   (+{REWARD_RISK:.1f}R each): {tp_count:5d}  ({tp_count/len(df)*100:.1f}%)")
    print(f"  SL   (-1.0R each):  {sl_count:5d}  ({sl_count/len(df)*100:.1f}%)")
    print(f"  FLAT (session-end): {len(flat_df):5d}  ({len(flat_df)/len(df)*100:.1f}%)  "
          f"[{flat_pos} closed positive, {flat_neg} closed negative]")

    print("\nPer-index:")
    per_index = df.groupby("index")["r"].agg(trades="count", total_r="sum", avg_r="mean")
    print(per_index.sort_values("total_r", ascending=False).round(3))

    print("\nNo commission/spread/slippage modeled above - real results will be worse than this.")


if __name__ == "__main__":
    main()
