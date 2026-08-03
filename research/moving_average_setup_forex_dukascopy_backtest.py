# Same strategy as moving_average_setup_forex_london_backtest.py (the
# original YouTube-taught trend + arrow + RSI setup), but pulling YEARS of
# free historical forex data from Dukascopy instead of Yahoo Finance's
# 60-day intraday cap. That cap was the real bottleneck in every forex
# test in this project so far - not the backtest logic itself, which has
# checked out clean every time once tested. More history means the
# thin-sample "-9.09R over 2264 trades" result from the 60-day test can
# actually be checked against a much bigger sample.
#
# pip install dukascopy-python   # not on PyPI's index of things Colab
# preinstalls - run this in its own cell first, then restart if Colab
# complains about the import not being found immediately after install.
#
# Indicator math is identical to moving_average_setup_forex_london_backtest.py
# (itself checked numerically against quantconnect/main.py's proven
# incremental version) - only the data-loading function changed.

# !pip install --upgrade dukascopy-python -q   # uncomment this line in Colab

import datetime

import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

# (constant, human label) pairs - majors + common crosses, matching this
# project's other forex scripts' coverage.
TICKERS = [
    ("EURUSD", dki.INSTRUMENT_FX_MAJORS_EUR_USD), ("GBPUSD", dki.INSTRUMENT_FX_MAJORS_GBP_USD),
    ("USDJPY", dki.INSTRUMENT_FX_MAJORS_USD_JPY), ("USDCHF", dki.INSTRUMENT_FX_MAJORS_USD_CHF),
    ("USDCAD", dki.INSTRUMENT_FX_MAJORS_USD_CAD), ("AUDUSD", dki.INSTRUMENT_FX_MAJORS_AUD_USD),
    ("NZDUSD", dki.INSTRUMENT_FX_MAJORS_NZD_USD),
    ("EURGBP", dki.INSTRUMENT_FX_CROSSES_EUR_GBP), ("EURJPY", dki.INSTRUMENT_FX_CROSSES_EUR_JPY),
    ("GBPJPY", dki.INSTRUMENT_FX_CROSSES_GBP_JPY), ("EURCHF", dki.INSTRUMENT_FX_CROSSES_EUR_CHF),
    ("EURAUD", dki.INSTRUMENT_FX_CROSSES_EUR_AUD), ("EURCAD", dki.INSTRUMENT_FX_CROSSES_EUR_CAD),
    ("EURNZD", dki.INSTRUMENT_FX_CROSSES_EUR_NZD), ("GBPCHF", dki.INSTRUMENT_FX_CROSSES_GBP_CHF),
    ("GBPAUD", dki.INSTRUMENT_FX_CROSSES_GBP_AUD), ("GBPCAD", dki.INSTRUMENT_FX_CROSSES_GBP_CAD),
    ("GBPNZD", dki.INSTRUMENT_FX_CROSSES_GBP_NZD), ("AUDJPY", dki.INSTRUMENT_FX_CROSSES_AUD_JPY),
    ("AUDNZD", dki.INSTRUMENT_FX_CROSSES_AUD_NZD), ("AUDCAD", dki.INSTRUMENT_FX_CROSSES_AUD_CAD),
    ("AUDCHF", dki.INSTRUMENT_FX_CROSSES_AUD_CHF), ("CADJPY", dki.INSTRUMENT_FX_CROSSES_CAD_JPY),
    ("CHFJPY", dki.INSTRUMENT_FX_CROSSES_CHF_JPY), ("NZDJPY", dki.INSTRUMENT_FX_CROSSES_NZD_JPY),
    ("NZDCAD", dki.INSTRUMENT_FX_CROSSES_NZD_CAD), ("NZDCHF", dki.INSTRUMENT_FX_CROSSES_NZD_CHF),
]

# 1 year x 27 pairs of 5-min bars is already ~40x more data than the
# 60-day yfinance cap gave per pair. Widen this once you've confirmed a
# run completes in reasonable time - Dukascopy free data goes back many
# years, this isn't a hard ceiling like Yahoo's.
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

# Illustrative round-trip cost scenarios, as a percentage of entry price - NOT measured real spread
# data, just a few bracketing assumptions to see how much cost this edge can absorb before it
# disappears, same convention introduced in support_resistance_zone_bounce_dukascopy_backtest.py.
COST_PCT_SCENARIOS = [0.0, 0.01, 0.03, 0.05]
REVERSE_SIGNALS = False
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


def load_pair_data(instrument_const):
    df = dukascopy_python.fetch(
        instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, START_DATE, END_DATE,
    )
    if df.empty:
        return None
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    if LONDON_SESSION_ONLY:
        df.index = to_london_time(df.index)
    return df


def backtest_pair(label, instrument_const):
    df = load_pair_data(instrument_const)
    if df is None:
        return []

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

        if REVERSE_SIGNALS:
            buy_setup, sell_setup = sell_setup, buy_setup

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

        trades.append({"ticker": label, "side": side, "outcome": outcome, "r": exit_r,
                       "stop_pct": risk / entry})
        i = j + 1

    return trades


def main():
    all_trades = []
    for label, instrument_const in TICKERS:
        print(f"{label}...", end=" ")
        try:
            trades = backtest_pair(label, instrument_const)
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
    print(f"TOTAL: {len(all_trades)} trades across {len(TICKERS)} forex pairs, "
          f"{START_DATE.date()} to {END_DATE.date()}, London-only={LONDON_SESSION_ONLY}")
    print(f"REWARD_RISK={REWARD_RISK}  REVERSE_SIGNALS={REVERSE_SIGNALS}")
    print("=" * 60)

    if not all_trades:
        print("No trades at all - check ticker output above for download failures.")
        return

    df = pd.DataFrame(all_trades)
    total_r = df["r"].sum()
    tp_count = (df["outcome"] == "TP").sum()
    sl_count = (df["outcome"] == "SL").sum()
    timeout_df = df[df["outcome"] == "TIMEOUT"]
    timeout_pos = (timeout_df["r"] > 0).sum()
    timeout_neg = len(timeout_df) - timeout_pos

    print(f"Total: {total_r:+.2f}R   Average: {total_r/len(df):+.3f}R/trade   <- this decides profitability, not win rate")
    print(f"\nOutcome breakdown ({len(df)} trades):")
    print(f"  TP  (+{REWARD_RISK:.1f}R each): {tp_count:5d}  ({tp_count/len(df)*100:.1f}%)")
    print(f"  SL  (-1.0R each):  {sl_count:5d}  ({sl_count/len(df)*100:.1f}%)")
    print(f"  Timed out:         {len(timeout_df):5d}  ({len(timeout_df)/len(df)*100:.1f}%)  "
          f"[{timeout_pos} closed positive, {timeout_neg} closed negative]")

    breakeven_wr = 1 / (1 + REWARD_RISK) * 100
    naive_win_rate = (df["r"] > 0).sum() / len(df) * 100
    print(f"\n'Win rate' (any trade that closed r>0): {naive_win_rate:.1f}% - not directly comparable to the "
          f"{breakeven_wr:.1f}% clean-payout breakeven line once TIMEOUT trades are in the mix. Total R above is "
          f"the number that actually matters.")

    print("\nPer-pair:")
    per_ticker = df.groupby("ticker")["r"].agg(trades="count", total_r="sum", avg_r="mean")
    print(per_ticker.sort_values("total_r", ascending=False).round(3))

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
