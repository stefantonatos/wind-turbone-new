# Backtest of the original YouTube-taught trend + arrow + RSI setup across
# as many forex pairs as Yahoo Finance offers free intraday data for,
# restricted to the London session only. Run this in Google Colab - paste
# the whole file into one cell, run it, read the summary at the bottom.
#
# Same strategy as telegram-relay/src/strategy.js / quantconnect/main.py /
# research/moving_average_setup_backtest.py (the stock version of this same
# test) - a 21/50/200 Wilder-smoothed MA stack for trend, held for
# CONFIRM_BARS consecutive bars, gated by a 3 Line Strike or Engulfing
# Candle arrow pattern and RSI vs 50. Indicator math is a direct port of
# the already-verified logic in quantconnect/main.py, not re-derived.
#
# "London session only" means: NEW entries are only considered while the
# current bar falls within LONDON_SESSION_START-LONDON_SESSION_END London
# LOCAL time (DST-aware - see to_london_time() below). An already-open
# trade is still managed bar-by-bar after the session ends, same as the
# live bot and quantconnect/main.py - there's no forced session-end
# flatten in the original strategy's rules, so this doesn't invent one.
#
# Forex trades ~24/5, unlike stocks - so unlike the ORB stock script,
# there's no single trading-day boundary to work around here, and the MA/
# RSI/trend state simply flows continuously through the whole series.

# !pip install --upgrade yfinance -q   # uncomment this line in Colab

import numpy as np
import pandas as pd
import yfinance as yf

# As many liquid pairs as Yahoo Finance's FX tickers cover - 7 majors +
# the common crosses across them.
TICKERS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X", "AUDUSD=X", "NZDUSD=X",
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "EURCHF=X", "EURAUD=X", "EURCAD=X", "EURNZD=X",
    "GBPCHF=X", "GBPAUD=X", "GBPCAD=X", "GBPNZD=X",
    "AUDJPY=X", "AUDNZD=X", "AUDCAD=X", "AUDCHF=X",
    "CADJPY=X", "CHFJPY=X", "NZDJPY=X", "NZDCAD=X", "NZDCHF=X",
]

INTRADAY_INTERVAL = "5m"
INTRADAY_PERIOD = "60d"   # Yahoo's hard cap for 5m data

LONDON_SESSION_START = pd.Timestamp("08:00").time()
LONDON_SESSION_END = pd.Timestamp("16:30").time()

MA_FAST, MA_MID, MA_SLOW = 21, 50, 200
RSI_LEN = 14
CONFIRM_BARS = 6
REWARD_RISK = 2.0         # TP distance = SL distance x this. 2.0 = the original taught 2:1 rule
MIN_SL_PCT = 0.05         # SL distance floor as % of price
REVERSE_SIGNALS = True    # currently testing the fade - flip back to False to test the straight setup
MAX_HOLD_BARS = 500       # ~a week of 5-min bars - a trade open longer than this gets closed out and scored


def to_london_time(index):
    """yfinance intraday timestamps aren't always tz-aware depending on
    ticker/version - if naive, assume UTC (Yahoo's common default for FX)
    rather than silently treating it as already-London local time."""
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


_diagnostic_shown = False


def backtest_ticker(ticker):
    global _diagnostic_shown
    df = yf.download(ticker, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL,
                      progress=False, auto_adjust=False)
    if df.empty:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    original_tz = df.index.tz
    df.index = to_london_time(df.index)

    if not _diagnostic_shown:
        print(f"\n  DIAGNOSTIC: {ticker} raw tz={original_tz} -> first bar in London time: {df.index[0]}\n"
              f"  (if that clock time looks implausible for a market-data timestamp, the UTC assumption above is wrong)")
        _diagnostic_shown = True

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

        trades.append({"ticker": ticker, "side": side, "outcome": outcome, "r": exit_r})
        i = j + 1

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
    print(f"TOTAL: {len(all_trades)} trades across {len(TICKERS)} forex pairs, London session only, "
          f"{INTRADAY_PERIOD} of {INTRADAY_INTERVAL} data")
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

    print("\nPer-pair:")
    per_ticker = df.groupby("ticker")["r"].agg(trades="count", total_r="sum", avg_r="mean")
    print(per_ticker.sort_values("total_r", ascending=False).round(3))

    if len(df) < 30:
        print(f"\n{len(df)} trades is a thin sample - treat this as a first look, not a conclusion.")


if __name__ == "__main__":
    main()
