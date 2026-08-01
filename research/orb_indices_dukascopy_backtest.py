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
# ported to plain Python/pandas here rather than a QC algorithm - now
# including the full filter set orb.py has (range-vs-ATR, impulsive
# candle, relative volume, volatility regime), not just the bare
# breakout rule.
#
# Each index has its own home-market session, unlike round-the-clock
# forex, so this needs a real per-index (timezone, local session-open
# time) mapping - not just one "London session" gate like the forex
# scripts use.
#
# VOLUME_FILTER caveat: unlike spot forex, index CFD data from Dukascopy
# does return a volume column, but whether it's meaningful (real traded
# volume vs. a tick-count proxy vs. just zero) isn't verified from this
# sandbox (same situation as every other live data question here - can't
# reach Dukascopy's servers to check). This fails OPEN (doesn't block
# trades) if volume looks degenerate, with a one-time diagnostic print
# per index so the first real run settles whether it's usable.

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

# Illustrative round-trip cost scenarios, as a percentage of entry price - NOT measured real spread
# data, just a few bracketing assumptions to see how much cost this edge can absorb before it
# disappears, same convention introduced in support_resistance_zone_bounce_dukascopy_backtest.py.
COST_PCT_SCENARIOS = [0.0, 0.01, 0.03, 0.05]
REVERSE_SIGNALS = False

ATR_LEN = 14
RANGE_ATR_FILTER = True
MIN_RANGE_ATR_MULT = 0.5
MAX_RANGE_ATR_MULT = 3.0

IMPULSE_FILTER = True
RANGE_AVG_LEN = 20
IMPULSE_RANGE_MULT = 1.3

VOLUME_FILTER = True
VOLUME_AVG_LEN = 20
RVOL_MULT = 1.3

VOLATILITY_REGIME_FILTER = True
ATR_BASELINE_LEN = 100
LOW_VOL_MULT = 0.7
HIGH_VOL_MULT = 1.5
ALLOWED_REGIMES = {"normal", "high"}


def to_local_time(index, tz_name):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(tz_name)


def persisted_avg(values, length, start_index=0):
    """Same seed-then-recurse SMMA rule used throughout this project,
    generalized here to average bar-range/volume/ATR series (not just
    price) - reused for IMPULSE_FILTER, VOLUME_FILTER, and
    VOLATILITY_REGIME_FILTER's baseline.

    start_index skips leading values that aren't valid yet (e.g. ATR's
    own warmup period) - substituting a placeholder like 0.0 for those
    instead would corrupt the seed average with fake data; this instead
    only starts averaging once the source series itself has real values."""
    n = len(values)
    out = [None] * n
    usable = values[start_index:]
    if len(usable) < length:
        return out
    seed = sum(usable[:length]) / length
    out[start_index + length - 1] = seed
    prev = seed
    for i in range(start_index + length, n):
        prev = (prev * (length - 1) + values[i]) / length
        out[i] = prev
    return out


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


def volatility_regime(current_atr, baseline):
    if current_atr is None or baseline is None or baseline <= 0:
        return "normal"
    if current_atr > HIGH_VOL_MULT * baseline:
        return "high"
    if current_atr < LOW_VOL_MULT * baseline:
        return "low"
    return "normal"


def backtest_index(label, instrument_const, tz_name, session_start):
    df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, START_DATE, END_DATE)
    if df.empty:
        return []
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df.index = to_local_time(df.index, tz_name)

    highs, lows, closes = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
    volumes = df["volume"].tolist() if "volume" in df.columns else [0] * len(df)
    times = df.index
    n = len(closes)

    atr = compute_atr_series(highs, lows, closes, ATR_LEN)
    bar_ranges = [h - l for h, l in zip(highs, lows)]
    range_avg = persisted_avg(bar_ranges, RANGE_AVG_LEN)
    volume_avg = persisted_avg(volumes, VOLUME_AVG_LEN)
    atr_first_valid = next((idx for idx, a in enumerate(atr) if a is not None), len(atr))
    atr_baseline = persisted_avg([a if a is not None else 0.0 for a in atr], ATR_BASELINE_LEN,
                                  start_index=atr_first_valid)

    volume_diagnostic_shown = False

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

        if not volume_diagnostic_shown and volume_avg[i] is not None:
            print(f"  [{label}] DIAGNOSTIC: {VOLUME_AVG_LEN}-bar avg volume = {volume_avg[i]:.2f} - if this "
                  f"is 0 (or stays 0), volume isn't usable here and VOLUME_FILTER is failing open (not blocking trades)")
            volume_diagnostic_shown = True

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
        current_atr = atr[i]

        if RANGE_ATR_FILTER and current_atr:
            if range_size < MIN_RANGE_ATR_MULT * current_atr or range_size > MAX_RANGE_ATR_MULT * current_atr:
                traded_today = True
                i += 1
                continue

        if VOLATILITY_REGIME_FILTER:
            regime = volatility_regime(current_atr, atr_baseline[i])
            if regime not in ALLOWED_REGIMES:
                traded_today = True
                i += 1
                continue

        if IMPULSE_FILTER and range_avg[i]:
            bar_range = highs[i] - lows[i]
            if bar_range < IMPULSE_RANGE_MULT * range_avg[i]:
                traded_today = True
                i += 1
                continue

        # Fails open: if avg volume is None/0 (warmup not done, or the
        # field is genuinely always zero), this does NOT block the trade.
        if VOLUME_FILTER and volume_avg[i] and volume_avg[i] > 0:
            if volumes[i] < RVOL_MULT * volume_avg[i]:
                traded_today = True
                i += 1
                continue

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

        trades.append({"index": label, "side": side, "outcome": outcome, "r": exit_r,
                       "stop_pct": sl_distance / entry})
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

    print(f"\nCOST SENSITIVITY (illustrative round-trip spread scenarios, NOT measured real spread data):")
    for cost_pct in COST_PCT_SCENARIOS:
        cost_adjusted_total = sum(t["r"] - (cost_pct / 100.0) / t["stop_pct"] for t in all_trades)
        print(f"  {cost_pct:.2f}% round-trip cost: {cost_adjusted_total:+.2f}R total, "
              f"{cost_adjusted_total/len(df):+.4f}R/trade")
    print(f"  If the total goes negative well before 0.05%, this edge is too thin to survive real "
          f"execution costs - check your actual broker's spread on each instrument against these numbers.")

    print("\nNo commission/spread/slippage modeled above - real results will be worse than this.")


if __name__ == "__main__":
    main()
