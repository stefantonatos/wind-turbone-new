# Backtest of the original YouTube-taught trend + arrow + RSI setup across
# many liquid US stocks, using free Yahoo Finance intraday data. Run this
# in Google Colab - paste the whole file into one cell, run it, read the
# summary at the bottom.
#
# This is the SAME strategy as telegram-relay/src/strategy.js (the live
# Telegram bot) and quantconnect/main.py (the QC/forex backtest) - a
# 21/50/200 Wilder-smoothed MA stack for trend, held for CONFIRM_BARS
# consecutive bars, gated by a 3 Line Strike or Engulfing Candle arrow
# pattern and RSI vs 50. REWARD_RISK defaults to 2.0 here to match what
# the video actually taught (2:1) - main.py later also tested 1:1, which
# you can get here too by changing the constant.
#
# The indicator math (smoothed_ma, wilder_rsi, arrow detection) is a
# direct line-for-line port of the already-verified logic in
# quantconnect/main.py (which was itself cross-checked against
# strategy.js on real data before shipping) - not re-derived from
# scratch, to avoid quietly drifting from the rules actually being run
# live and in QuantConnect.
#
# Same rigor as research/orb_multi_stock_backtest.py: only one trade open
# at a time, every trade tracked as a real R-multiple (not just win/loss),
# a trade that doesn't resolve within MAX_HOLD_BARS gets closed out and
# scored at whatever price it's at rather than silently discarded, and no
# commission/spread/slippage is modeled - real fills will be worse than
# this.

# !pip install --upgrade yfinance -q   # uncomment this line in Colab

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "JPM", "BAC", "V", "MA", "HD", "WMT", "UNH", "XOM",
]

INTRADAY_INTERVAL = "5m"
INTRADAY_PERIOD = "60d"   # Yahoo's hard cap for 5m data

MA_FAST, MA_MID, MA_SLOW = 21, 50, 200
RSI_LEN = 14
CONFIRM_BARS = 6          # trend stack must hold for this many consecutive bars
REWARD_RISK = 2.0         # TP distance = SL distance x this. 2.0 = the original taught 2:1 rule
MIN_SL_PCT = 0.05         # SL distance floor as % of price (stocks span $10s-$1000s, so % not $ - see ORB script's same fix)
REVERSE_SIGNALS = False   # flip to True to test the reversed direction
MAX_HOLD_BARS = 500       # ~a week of 5-min bars - a trade open longer than this gets closed out and scored, not left dangling


def smoothed_ma(values, length):
    """Wilder-style smoothing: seed with the simple average of the first
    `length` values, then recursively smooth forever after. Same
    seed-then-recurse rule as strategy.js's smoothedMA() / main.py's
    UpdateSMMA - deliberately NOT pandas' .ewm(), which seeds from the
    very first value instead and would silently compute a different
    (if similar-looking) series for the early bars of each warmup."""
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
    """Direct port of quantconnect/main.py's WilderRSI."""
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
    """'up'/'down'/'none' per bar - the MA stack must hold for the last
    `confirm_bars` consecutive bars, not just the current one (a single-bar
    snapshot can flip on a brief spike - this project already found that
    the hard way on the live bot)."""
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
    """3 Line Strike or Engulfing Candle, evaluated at bar i. Direct port
    of quantconnect/main.py's ComputeArrows."""
    if i < 3:
        return False, False
    o0, o1, o2, o3 = opens[i], opens[i - 1], opens[i - 2], opens[i - 3]
    c0, c1, c2, c3 = closes[i], closes[i - 1], closes[i - 2], closes[i - 3]
    strike3_bull = c3 < o3 and c2 < o2 and c1 < o1 and c0 > o1
    strike3_bear = c3 > o3 and c2 > o2 and c1 > o1 and c0 < o1
    engulf_bull = o0 <= c1 and o0 < o1 and c0 > o1
    engulf_bear = o0 >= c1 and o0 > o1 and c0 < o1
    return (strike3_bull or engulf_bull), (strike3_bear or engulf_bear)


def backtest_ticker(ticker):
    df = yf.download(ticker, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL,
                      progress=False, auto_adjust=False)
    if df.empty:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    opens, highs, lows, closes = (df["Open"].tolist(), df["High"].tolist(),
                                   df["Low"].tolist(), df["Close"].tolist())
    n = len(closes)

    ma21 = smoothed_ma(closes, MA_FAST)
    ma50 = smoothed_ma(closes, MA_MID)
    ma200 = smoothed_ma(closes, MA_SLOW)
    rsi = wilder_rsi(closes, RSI_LEN)
    trend = compute_trend_series(closes, ma21, ma50, ma200, CONFIRM_BARS)

    trades = []
    i = max(MA_SLOW, RSI_LEN) + CONFIRM_BARS
    while i < n:
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

        outcome, exit_r, hold_bars = None, None, 0
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

        trades.append({"ticker": ticker, "side": side, "outcome": outcome, "r": exit_r})
        i = j + 1  # only one trade open at a time - resume scanning after this one closes

    return trades


def main():
    all_trades = []
    for ticker in TICKERS:
        print(f"{ticker}...", end=" ")
        try:
            trades = backtest_ticker(ticker)
        except Exception as exc:
            print(f"failed ({exc})")
            continue
        all_trades.extend(trades)
        if trades:
            wins = sum(1 for t in trades if t["r"] > 0)
            total_r = sum(t["r"] for t in trades)
            print(f"{len(trades)} trades, {wins}/{len(trades)} win, {total_r:+.2f}R")
        else:
            print("0 trades")

    print("\n" + "=" * 60)
    print(f"TOTAL: {len(all_trades)} trades across {len(TICKERS)} tickers, {INTRADAY_PERIOD} of {INTRADAY_INTERVAL} data")
    print(f"REWARD_RISK={REWARD_RISK}  REVERSE_SIGNALS={REVERSE_SIGNALS}")
    print("=" * 60)

    if not all_trades:
        print("No trades at all - check ticker output above for download failures.")
        return

    df = pd.DataFrame(all_trades)
    wins = (df["r"] > 0).sum()
    total_r = df["r"].sum()

    print(f"Win rate: {wins/len(df)*100:.1f}%  ({wins}/{len(df)})")
    print(f"Total: {total_r:+.2f}R   Average: {total_r/len(df):+.3f}R/trade")
    print(f"Outcome breakdown: {df['outcome'].value_counts().to_dict()}")
    print("\nNo commission/spread/slippage modeled above - real results will be worse than this.")

    print("\nPer-ticker:")
    per_ticker = df.groupby("ticker")["r"].agg(trades="count", total_r="sum", avg_r="mean")
    print(per_ticker.sort_values("total_r", ascending=False).round(3))

    if len(df) < 30:
        print(f"\n{len(df)} trades is a thin sample - treat this as a first look, not a conclusion.")


if __name__ == "__main__":
    main()
