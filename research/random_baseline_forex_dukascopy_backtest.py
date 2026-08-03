# Same random coin-flip 1:1 baseline as random_baseline_forex_backtest.py,
# but pulling years of real data from Dukascopy instead of Yahoo
# Finance's 60-day cap - so the "what does pure chance produce" null
# distribution is measured against the same real multi-year sample the
# actual strategies (moving_average_setup_forex_dukascopy_backtest.py)
# are being judged against, not a much shorter window.
#
# ATR computation is identical to random_baseline_forex_backtest.py -
# only the data source changed.

# !pip install --upgrade dukascopy-python -q   # uncomment this line in Colab

import datetime
import random as _random

import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

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

START_DATE = datetime.datetime(2024, 1, 1)
END_DATE = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_MIN_5
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

ATR_LEN = 14
ATR_MULT = 2.0
MAX_HOLD_BARS = 500
LONDON_SESSION_ONLY = True   # matches moving_average_setup_forex_dukascopy_backtest.py's default, for a fair comparison
LONDON_SESSION_START = pd.Timestamp("08:00").time()
LONDON_SESSION_END = pd.Timestamp("16:30").time()

N_SIMULATIONS = 50


def to_london_time(index):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert("Europe/London")


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


def load_pair_data(instrument_const):
    df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, START_DATE, END_DATE)
    if df.empty:
        return None
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    if LONDON_SESSION_ONLY:
        df.index = to_london_time(df.index)
    return df


def simulate_random_run(df, atr, seed):
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
        target = entry + risk if side == "LONG" else entry - risk

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
    print(f"Downloading {len(TICKERS)} pairs from Dukascopy ({START_DATE.date()} to {END_DATE.date()})...")
    ticker_data = {}
    for label, instrument_const in TICKERS:
        try:
            df = load_pair_data(instrument_const)
        except Exception as exc:
            print(f"  {label}: failed ({exc})")
            continue
        if df is None:
            print(f"  {label}: no data")
            continue
        highs, lows, closes = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
        atr = compute_atr_series(highs, lows, closes, ATR_LEN)
        ticker_data[label] = (df, atr)
        print(f"  {label}: {len(df)} bars loaded")

    if not ticker_data:
        print("No data downloaded at all - check ticker output above.")
        return

    print(f"\nRunning {N_SIMULATIONS} random-seed replications across {len(ticker_data)} pairs...")
    replication_totals = []
    all_trades_last_run = None

    for sim in range(N_SIMULATIONS):
        sim_trades = []
        for label, (df, atr) in ticker_data.items():
            sim_trades.extend(simulate_random_run(df, atr, seed=sim * 1000 + hash(label) % 1000))
        total_r = sum(t["r"] for t in sim_trades)
        replication_totals.append({"seed": sim, "trades": len(sim_trades), "total_r": total_r})
        if sim == N_SIMULATIONS - 1:
            all_trades_last_run = sim_trades

    rep_df = pd.DataFrame(replication_totals)

    print("\n" + "=" * 60)
    print(f"RANDOM BASELINE (Dukascopy): {N_SIMULATIONS} replications, {len(ticker_data)} forex pairs, "
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

    print(f"\nCompare the real strategy's -403R (moving_average_setup_forex_dukascopy_backtest.py's straight run) "
          f"against THIS spread - if -403R falls inside this random range, the strategy isn't distinguishable "
          f"from chance even at this trade volume.")

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
