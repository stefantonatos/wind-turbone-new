# ORB (Opening Range Breakout) backtest across many liquid US stocks/ETFs,
# using free Yahoo Finance intraday data. Run this in Google Colab - paste
# the whole file into one cell, run it, read the summary at the bottom.
#
# Why multi-stock: one forex pair over a short window (this project's other
# tests) doesn't produce enough trades to say anything statistically
# meaningful. Testing the same rules across ~20 liquid names multiplies the
# sample size for free, and ORB is documented to work better on US
# stocks/indices than forex in the first place.
#
# Two things this deliberately does differently from a naive version of
# this idea, both because getting them wrong silently inflates the
# apparent edge:
#   1. ATR (volatility baseline) is computed from DAILY bars, 14-day
#      average true range - the standard definition - not from a rolling
#      window of 5-min bars that straddles across day boundaries (which
#      mixes yesterday afternoon's volatility into this morning's filter
#      decision).
#   2. Relative volume is computed against the average volume in THAT SAME
#      5-minute-of-day slot over the trailing 10 days (proper RVOL - volume
#      is naturally U-shaped through the trading day, so comparing 9:35am
#      volume to a whole-day average is comparing apples to oranges), not
#      a single static per-day snapshot value.
#
# Every signal is tracked as an actual R-multiple (not just win/loss), and
# a trade that doesn't hit its stop or target by end of day is flattened at
# the close and scored on realized R, not silently discarded - discarding
# incomplete trades biases the sample toward whichever side resolves
# faster. No commission/spread/slippage is modeled - real fills will be
# worse than this, so treat any edge shown here as an upper bound.

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
INTRADAY_PERIOD = "60d"     # Yahoo's hard cap for 5m data - can't get more history this way
ORB_BARS = 3                # 3 x 5min = first 15 minutes of the session
REWARD_RISK = 1.0           # target distance = risk distance x this (measured-move target)
REVERSE_SIGNALS = False     # flip to True to fade the breakout instead of taking it
MIN_RANGE_PCT = 0.05        # opening range must be at least this % of price (scales across tickers, unlike a flat $ floor)
RANGE_ATR_MIN_MULT = 0.3
RANGE_ATR_MAX_MULT = 2.5
RVOL_LOOKBACK_DAYS = 10
RVOL_MIN_PERIODS = 3
RVOL_THRESHOLD = 1.3


def daily_atr(ticker, length=14):
    daily = yf.download(ticker, period="4mo", interval="1d", progress=False, auto_adjust=False)
    if daily.empty:
        return None
    high, low, close = daily["High"], daily["Low"], daily["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(length).mean()
    return atr_series  # indexed by date; use .loc[some_date] for that day's ATR-as-of-yesterday


def relative_volume(intraday_df):
    """RVOL per bar: this bar's volume vs the average volume in the SAME
    time-of-day slot over the trailing RVOL_LOOKBACK_DAYS sessions. Shifted
    by one session so a bar is never compared against a baseline that
    includes itself."""
    tod = intraday_df.index.time
    vol_by_slot = intraday_df.groupby(tod)["Volume"]
    baseline = vol_by_slot.transform(
        lambda s: s.shift(1).rolling(RVOL_LOOKBACK_DAYS, min_periods=RVOL_MIN_PERIODS).mean()
    )
    return intraday_df["Volume"] / baseline


def simulate_day(day_df, atr_asof_yesterday, ticker_price_ref):
    if len(day_df) <= ORB_BARS + 1 or atr_asof_yesterday is None or pd.isna(atr_asof_yesterday):
        return None

    orb = day_df.iloc[:ORB_BARS]
    orb_high = float(orb["High"].max())
    orb_low = float(orb["Low"].min())
    orb_width = orb_high - orb_low

    if orb_width < (MIN_RANGE_PCT / 100.0) * ticker_price_ref:
        return None
    if not (RANGE_ATR_MIN_MULT * atr_asof_yesterday <= orb_width <= RANGE_ATR_MAX_MULT * atr_asof_yesterday):
        return None

    post_orb = day_df.iloc[ORB_BARS:]

    for i, (ts, row) in enumerate(post_orb.iterrows()):
        rvol = row["rvol"]
        if pd.isna(rvol) or rvol < RVOL_THRESHOLD:
            continue

        close = float(row["Close"])
        raw_side = None
        if close > orb_high:
            raw_side = "LONG"
        elif close < orb_low:
            raw_side = "SHORT"
        if raw_side is None:
            continue

        entry = close
        # Risk distance comes from the RAW breakout geometry (distance to
        # the opposite side of the range from wherever price actually broke
        # out) - computed before any REVERSE_SIGNALS flip, and always
        # positive by construction. Computing it after flipping direction
        # would reuse the wrong boundary (e.g. a raw upside breakout, faded
        # short, would wrongly measure "distance to orb_high" from a price
        # that's already above orb_high - negative, nonsensical).
        risk = (entry - orb_low) if raw_side == "LONG" else (orb_high - entry)
        if risk <= 0:
            continue

        side = raw_side
        if REVERSE_SIGNALS:
            side = "SHORT" if raw_side == "LONG" else "LONG"

        if side == "LONG":
            stop, target = entry - risk, entry + risk * REWARD_RISK
        else:
            stop, target = entry + risk, entry - risk * REWARD_RISK

        future = post_orb.iloc[i + 1:]
        for _, bar in future.iterrows():
            hi, lo = float(bar["High"]), float(bar["Low"])
            hit_stop = lo <= stop if side == "LONG" else hi >= stop
            hit_target = hi >= target if side == "LONG" else lo <= target
            if hit_stop:
                # if a single bar's range spans both levels, assume stop hit
                # first - can't tell from OHLC alone, and this is the
                # conservative assumption
                return {"side": side, "outcome": "SL", "r": -1.0}
            if hit_target:
                return {"side": side, "outcome": "TP", "r": REWARD_RISK}

        # Neither hit by end of day - flatten at the last close, score the realized R.
        last_close = float(post_orb["Close"].iloc[-1])
        pnl = (last_close - entry) if side == "LONG" else (entry - last_close)
        return {"side": side, "outcome": "FLAT", "r": pnl / risk}

    return None  # no breakout with sufficient RVOL confirmed today


def backtest_ticker(ticker):
    intraday = yf.download(ticker, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL,
                            progress=False, auto_adjust=False)
    if intraday.empty:
        return []
    if isinstance(intraday.columns, pd.MultiIndex):
        intraday.columns = intraday.columns.get_level_values(0)

    intraday["rvol"] = relative_volume(intraday)
    atr = daily_atr(ticker)

    trades = []
    for session_date, day_df in intraday.groupby(intraday.index.date):
        if atr is None:
            continue
        prior_days = atr.index[atr.index.date < session_date]
        if len(prior_days) == 0:
            continue
        atr_asof_yesterday = atr.loc[prior_days[-1]]
        if hasattr(atr_asof_yesterday, "item"):
            atr_asof_yesterday = atr_asof_yesterday.item()

        price_ref = float(day_df["Close"].iloc[0])
        result = simulate_day(day_df, atr_asof_yesterday, price_ref)
        if result is not None:
            result["ticker"] = ticker
            result["date"] = session_date
            trades.append(result)

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
    print("=" * 60)

    if not all_trades:
        print("No trades at all - filters may be too strict, or data didn't download. Check ticker output above.")
        return

    df = pd.DataFrame(all_trades)
    wins = (df["r"] > 0).sum()
    total_r = df["r"].sum()
    win_rate = wins / len(df) * 100

    print(f"Win rate: {win_rate:.1f}%  ({wins}/{len(df)})")
    print(f"Total: {total_r:+.2f}R   Average: {total_r/len(df):+.3f}R/trade")
    print(f"Outcome breakdown: {df['outcome'].value_counts().to_dict()}")
    print(f"\nNo commission/spread/slippage modeled above - real results will be worse than this.")

    print("\nPer-ticker:")
    per_ticker = df.groupby("ticker")["r"].agg(trades="count", total_r="sum", avg_r="mean")
    print(per_ticker.sort_values("total_r", ascending=False).round(3))

    if len(df) < 30:
        print(f"\n{len(df)} trades is a thin sample - treat this as a first look, not a conclusion. "
              f"Widen RANGE_ATR_MIN_MULT/MAX_MULT or lower RVOL_THRESHOLD to get more, or just extend the ticker list.")


if __name__ == "__main__":
    main()
