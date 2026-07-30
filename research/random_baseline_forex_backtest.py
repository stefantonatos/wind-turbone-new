# Random-entry baseline for forex, 1:1 reward:risk - the control group for
# every other strategy in this project. Run this in Google Colab - paste
# the whole file into one cell, run it, read the summary at the bottom.
#
# The idea: if a real strategy (the MA/arrow/RSI setup, ORB, etc.) can't
# beat a coin flip with the same risk management, its entry signal isn't
# adding anything - the "edge" some backtest showed is indistinguishable
# from what randomness alone produces on real price data. This is the
# benchmark other results in this project should be checked against, not
# just a novelty.
#
# A SINGLE random run isn't a meaningful benchmark by itself (it could
# look good or bad purely by luck), so this runs the random-entry
# strategy N_SIMULATIONS times, each with a different seed, over the SAME
# real downloaded price data, and reports the distribution of outcomes
# (mean, std dev, percentiles) - that distribution is what "pure chance"
# looks like here, and it's what a real strategy's result should be
# compared against, not zero.
#
# Stop/target distance is sized off ATR (not a fixed pip amount), same
# reasoning as every other script here: a flat distance doesn't scale
# across pairs with very different typical volatility (GBPJPY vs EURCHF).
# Reward = risk exactly (1:1) as requested - no measured-move or
# multiple-R target here.

# !pip install --upgrade yfinance -q   # uncomment this line in Colab

import random as _random

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X", "AUDUSD=X", "NZDUSD=X",
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "EURCHF=X", "EURAUD=X", "EURCAD=X", "EURNZD=X",
    "GBPCHF=X", "GBPAUD=X", "GBPCAD=X", "GBPNZD=X",
    "AUDJPY=X", "AUDNZD=X", "AUDCAD=X", "AUDCHF=X",
    "CADJPY=X", "CHFJPY=X", "NZDJPY=X", "NZDCAD=X", "NZDCHF=X",
]

INTRADAY_INTERVAL = "5m"
INTRADAY_PERIOD = "60d"     # Yahoo's hard cap for 5m data

ATR_LEN = 14
ATR_MULT = 2.0              # SL distance = ATR x this. Reward = same distance (1:1), no measured-move multiplier.
MAX_HOLD_BARS = 500         # a trade neither hitting SL nor TP within this many bars gets closed out and scored
LONDON_SESSION_ONLY = False  # True to restrict entries to 08:00-16:30 London time, for comparison against the MA-setup forex/London test
LONDON_SESSION_START = pd.Timestamp("08:00").time()
LONDON_SESSION_END = pd.Timestamp("16:30").time()

N_SIMULATIONS = 50           # independent random-seed replications over the same real price data


def to_london_time(index):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert("Europe/London")


def compute_atr_series(highs, lows, closes, length):
    """Standard batch Wilder ATR - same seed-then-recurse rule as
    quantconnect/donchian.py's UpdateATR, computed over the full series at
    once since there's no live-streaming memory constraint here."""
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


def load_ticker_data(ticker):
    df = yf.download(ticker, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL,
                      progress=False, auto_adjust=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if LONDON_SESSION_ONLY:
        df.index = to_london_time(df.index)
    return df


def simulate_random_run(df, atr, seed):
    """One full pass over the price series, flipping a coin for direction
    every time no trade is open. Returns the list of trade dicts."""
    rng = _random.Random(seed)
    highs, lows, closes = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
    times = df.index
    n = len(closes)

    trades = []
    i = ATR_LEN + 1
    while i < n:
        if atr[i] is None or atr[i] <= 0:
            i += 1
            continue

        if LONDON_SESSION_ONLY:
            tod = times[i].time()
            if not (LONDON_SESSION_START <= tod < LONDON_SESSION_END):
                i += 1
                continue

        side = "LONG" if rng.random() < 0.5 else "SHORT"
        entry = closes[i]
        risk = atr[i] * ATR_MULT
        stop = entry - risk if side == "LONG" else entry + risk
        target = entry + risk if side == "LONG" else entry - risk  # 1:1

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
                outcome, exit_r = "TP", 1.0
                break
            j += 1
        else:
            j = min(j, n - 1)

        if outcome is None:
            last_close = closes[j]
            pnl = (last_close - entry) if side == "LONG" else (entry - last_close)
            outcome, exit_r = "TIMEOUT", pnl / risk

        trades.append({"side": side, "outcome": outcome, "r": exit_r})
        i = j + 1

    return trades


def main():
    print(f"Downloading {len(TICKERS)} pairs...")
    ticker_data = {}
    for ticker in TICKERS:
        df = load_ticker_data(ticker)
        if df is None:
            print(f"  {ticker}: no data")
            continue
        highs, lows, closes = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
        atr = compute_atr_series(highs, lows, closes, ATR_LEN)
        ticker_data[ticker] = (df, atr)
        print(f"  {ticker}: {len(df)} bars loaded")

    if not ticker_data:
        print("No data downloaded at all - check ticker output above.")
        return

    print(f"\nRunning {N_SIMULATIONS} random-seed replications across {len(ticker_data)} pairs...")
    replication_totals = []
    all_trades_last_run = None  # keep one replication's trades for the outcome-breakdown print

    for sim in range(N_SIMULATIONS):
        sim_trades = []
        for ticker, (df, atr) in ticker_data.items():
            sim_trades.extend(simulate_random_run(df, atr, seed=sim * 1000 + hash(ticker) % 1000))
        total_r = sum(t["r"] for t in sim_trades)
        replication_totals.append({"seed": sim, "trades": len(sim_trades), "total_r": total_r})
        if sim == N_SIMULATIONS - 1:
            all_trades_last_run = sim_trades

    rep_df = pd.DataFrame(replication_totals)

    print("\n" + "=" * 60)
    print(f"RANDOM BASELINE: {N_SIMULATIONS} replications, {len(ticker_data)} forex pairs, "
          f"1:1 R:R, ATR x{ATR_MULT} stop, London-only={LONDON_SESSION_ONLY}")
    print("=" * 60)
    print(f"Trades per replication: mean {rep_df['trades'].mean():.0f}  "
          f"(min {rep_df['trades'].min()}, max {rep_df['trades'].max()})")
    print(f"\nTotal R per replication across {N_SIMULATIONS} random seeds:")
    print(f"  mean:   {rep_df['total_r'].mean():+.2f}R")
    print(f"  std:    {rep_df['total_r'].std():.2f}R")
    print(f"  min:    {rep_df['total_r'].min():+.2f}R")
    print(f"  25th percentile: {rep_df['total_r'].quantile(0.25):+.2f}R")
    print(f"  median: {rep_df['total_r'].median():+.2f}R")
    print(f"  75th percentile: {rep_df['total_r'].quantile(0.75):+.2f}R")
    print(f"  max:    {rep_df['total_r'].max():+.2f}R")

    print(f"\nThis is the null distribution - what pure chance produces on this real price data with "
          f"this exact risk setup. A real strategy's total R should be compared against THIS spread, "
          f"not against zero: if it lands inside this range, its 'edge' isn't distinguishable from luck.")

    if all_trades_last_run:
        df_last = pd.DataFrame(all_trades_last_run)
        tp = (df_last["outcome"] == "TP").sum()
        sl = (df_last["outcome"] == "SL").sum()
        timeout = (df_last["outcome"] == "TIMEOUT").sum()
        print(f"\nOutcome breakdown for one representative replication ({len(df_last)} trades):")
        print(f"  TP: {tp} ({tp/len(df_last)*100:.1f}%)   SL: {sl} ({sl/len(df_last)*100:.1f}%)   "
              f"Timed out: {timeout} ({timeout/len(df_last)*100:.1f}%)")

    print("\nNo commission/spread/slippage modeled above - real results will be worse than this.")


if __name__ == "__main__":
    main()
