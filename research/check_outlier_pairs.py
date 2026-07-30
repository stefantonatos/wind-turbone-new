# Diagnostic for CHFJPY (-70.4R) and NZDJPY (-94.9R), the two pairs that
# drove 41% of the total loss in the full-year Dukascopy run of
# moving_average_setup_forex_dukascopy_backtest.py, out of 27 pairs
# tested. Two questions this answers:
#   1. Is the raw price data for these two pairs sane, or does it show
#      signs of bad ticks/gaps that would corrupt the backtest (compared
#      against EURUSD as a normal-pair baseline)?
#   2. Per-pair, WHY are they so much worse - an unusually high stop-out
#      rate, or a few catastrophically bad timeout exits? The main
#      script's per-pair table only shows total_r/avg_r, not the
#      TP/SL/TIMEOUT split per pair, so this fills that gap.
#
# Run this in Google Colab in the same way as the other Dukascopy script
# (pip install dukascopy-python first, in its own cell).

# !pip install --upgrade dukascopy-python -q   # uncomment this line in Colab

import datetime

import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

# Reuse the exact same config as moving_average_setup_forex_dukascopy_backtest.py
# so this is a like-for-like check, not a different test.
PAIRS_TO_CHECK = [
    ("CHFJPY", dki.INSTRUMENT_FX_CROSSES_CHF_JPY),
    ("NZDJPY", dki.INSTRUMENT_FX_CROSSES_NZD_JPY),
    ("EURUSD", dki.INSTRUMENT_FX_MAJORS_EUR_USD),  # baseline - a pair that came back only mildly negative
]

START_DATE = datetime.datetime(2024, 1, 1)
END_DATE = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_MIN_5
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

LONDON_SESSION_START = pd.Timestamp("08:00").time()
LONDON_SESSION_END = pd.Timestamp("16:30").time()
LONDON_SESSION_ONLY = True

MA_FAST, MA_MID, MA_SLOW = 21, 50, 200
RSI_LEN = 14
CONFIRM_BARS = 6
REWARD_RISK = 2.0
MIN_SL_PCT = 0.05
MAX_HOLD_BARS = 500


def to_london_time(index):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert("Europe/London")


def smoothed_ma(values, length):
    n = len(values)
    out = [None] * n
    if n < length:
        return out
    seed = sum(values[:length]) / length
    out[length - 1] = seed
    prev = seed
    for i in range(length, n):
        prev = (prev * (length - 1) + values[i]) / length
        out[i] = prev
    return out


def wilder_rsi(closes, length):
    n = len(closes)
    rsi = [None] * n
    if n < length + 1:
        return rsi
    gain_sum = loss_sum = 0.0
    for i in range(1, length + 1):
        change = closes[i] - closes[i - 1]
        if change >= 0:
            gain_sum += change
        else:
            loss_sum -= change
    avg_gain, avg_loss = gain_sum / length, loss_sum / length
    rsi[length] = 100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(length + 1, n):
        change = closes[i] - closes[i - 1]
        gain, loss = (change, 0) if change > 0 else (0, -change)
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
        rsi[i] = 100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi


def compute_trend_series(closes, ma21, ma50, ma200, confirm_bars):
    n = len(closes)
    trend = ["none"] * n
    for i in range(confirm_bars - 1, n):
        all_up = all_down = True
        for j in range(i - confirm_bars + 1, i + 1):
            if ma21[j] is None or ma50[j] is None or ma200[j] is None:
                all_up = all_down = False
                break
            up = closes[j] > ma200[j] and ma21[j] > ma50[j] and ma50[j] > ma200[j]
            down = closes[j] < ma200[j] and ma21[j] < ma50[j] and ma50[j] < ma200[j]
            all_up = all_up and up
            all_down = all_down and down
            if not all_up and not all_down:
                break
        trend[i] = "up" if all_up else ("down" if all_down else "none")
    return trend


def arrows_at(opens, closes, i):
    if i < 3:
        return False, False
    o0, o1, o2, o3 = opens[i], opens[i - 1], opens[i - 2], opens[i - 3]
    c0, c1, c2, c3 = closes[i], closes[i - 1], closes[i - 2], closes[i - 3]
    strike3_bull = c3 < o3 and c2 < o2 and c1 < o1 and c0 > o1
    strike3_bear = c3 > o3 and c2 > o2 and c1 > o1 and c0 < o1
    engulf_bull = o0 <= c1 and o0 < o1 and c0 > o1
    engulf_bear = o0 >= c1 and o0 > o1 and c0 < o1
    return (strike3_bull or engulf_bull), (strike3_bear or engulf_bear)


def check_data_quality(label, df):
    """Sanity-check raw price data before trusting a backtest built on it."""
    closes = df["Close"].tolist()
    bar_ranges_pct = [(h - l) / c * 100 for h, l, c in zip(df["High"], df["Low"], df["Close"]) if c > 0]
    bar_returns_pct = [abs(closes[i] - closes[i - 1]) / closes[i - 1] * 100 for i in range(1, len(closes))
                        if closes[i - 1] > 0]

    # Expected bar count for ~1 year of 5-min bars, 24/5 (rough - doesn't
    # need to be exact, just a sanity magnitude check)
    weekdays_per_year = 365 * 5 / 7
    expected_bars = int(weekdays_per_year * 24 * 12)

    print(f"\n--- {label} data quality ---")
    print(f"  Bars: {len(df)} (rough expectation for a full year, 24/5: ~{expected_bars})")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Bar range as % of price: mean {np.mean(bar_ranges_pct):.4f}%, max {np.max(bar_ranges_pct):.4f}%")
    print(f"  Bar-to-bar return: mean {np.mean(bar_returns_pct):.4f}%, max {np.max(bar_returns_pct):.4f}%")
    extreme_moves = sum(1 for r in bar_returns_pct if r > 1.0)  # >1% in one 5-min bar is unusual for major/cross FX
    print(f"  Bars with a >1% single-bar move: {extreme_moves} ({extreme_moves/len(bar_returns_pct)*100:.3f}%) "
          f"- a burst of these would suggest bad ticks, not real price action")


def backtest_pair_detailed(label, instrument_const):
    df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, START_DATE, END_DATE)
    if df.empty:
        print(f"{label}: no data")
        return

    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    check_data_quality(label, df)

    if LONDON_SESSION_ONLY:
        df.index = to_london_time(df.index)

    opens, highs, lows, closes = (df["Open"].tolist(), df["High"].tolist(),
                                   df["Low"].tolist(), df["Close"].tolist())
    times = df.index
    n = len(closes)

    ma21 = smoothed_ma(closes, MA_FAST)
    ma50 = smoothed_ma(closes, MA_MID)
    ma200 = smoothed_ma(closes, MA_SLOW)
    rsi = wilder_rsi(closes, RSI_LEN)
    trend = compute_trend_series(closes, ma21, ma50, ma200, CONFIRM_BARS)

    trades = []
    i = max(MA_SLOW, RSI_LEN) + CONFIRM_BARS
    while i < n:
        if LONDON_SESSION_ONLY:
            tod = times[i].time()
            if not (LONDON_SESSION_START <= tod < LONDON_SESSION_END):
                i += 1
                continue
        if rsi[i] is None or trend[i] == "none":
            i += 1
            continue

        bull_arrow, bear_arrow = arrows_at(opens, closes, i)
        buy_setup = trend[i] == "up" and bull_arrow and rsi[i] > 50
        sell_setup = trend[i] == "down" and bear_arrow and rsi[i] < 50
        if not buy_setup and not sell_setup:
            i += 1
            continue

        current_range = highs[i] - lows[i]
        if current_range <= 0:
            i += 1
            continue

        entry = closes[i]
        side = "LONG" if buy_setup else "SHORT"
        risk = max(current_range * 2, (MIN_SL_PCT / 100.0) * entry)
        reward = risk * REWARD_RISK
        stop = entry - risk if side == "LONG" else entry + risk
        target = entry + reward if side == "LONG" else entry - reward

        outcome, exit_r = None, None
        j = i + 1
        while j < n and j < i + MAX_HOLD_BARS:
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
            outcome, exit_r = "TIMEOUT", pnl / risk

        trades.append({"side": side, "outcome": outcome, "r": exit_r, "risk_price_units": risk, "entry": entry})
        i = j + 1

    if not trades:
        print(f"{label}: 0 trades")
        return

    tdf = pd.DataFrame(trades)
    print(f"\n--- {label} strategy result ---")
    print(f"  Trades: {len(tdf)}   Total R: {tdf['r'].sum():+.2f}   Avg R/trade: {tdf['r'].mean():+.4f}")
    for outcome in ("TP", "SL", "TIMEOUT"):
        sub = tdf[tdf["outcome"] == outcome]
        if len(sub) == 0:
            continue
        print(f"  {outcome:8s}: {len(sub):4d} trades ({len(sub)/len(tdf)*100:5.1f}%), "
              f"total {sub['r'].sum():+8.2f}R, avg {sub['r'].mean():+.4f}R, "
              f"worst single trade {sub['r'].min():+.3f}R")
    print(f"  Avg SL distance as % of entry price: {(tdf['risk_price_units']/tdf['entry']*100).mean():.4f}%")


def main():
    for label, instrument_const in PAIRS_TO_CHECK:
        try:
            backtest_pair_detailed(label, instrument_const)
        except Exception as exc:
            print(f"{label}: failed ({exc})")


if __name__ == "__main__":
    main()
