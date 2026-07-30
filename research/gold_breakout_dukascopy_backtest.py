# Same strategy as gold_breakout_backtest.py (the fixed version of the
# found "Gold 15m High-RR" strategy: buy on a close above 20-SMA +
# 0.5xATR, target 5xATR, stop 1xATR, exit after 5 bars), but pulling
# years of real Gold data from Dukascopy instead of Yahoo Finance's
# 60-day intraday cap. Dukascopy classifies Gold under FX metals
# (XAU/USD), not commodities - confirmed by inspecting the installed
# dukascopy_python.instruments module directly.
#
# Indicator math (compute_sma, compute_atr_series) is identical to
# gold_breakout_backtest.py - only the data source changed.

# !pip install --upgrade dukascopy-python -q   # uncomment this line in Colab

import datetime

import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

TICKERS = [("XAUUSD", dki.INSTRUMENT_FX_METALS_XAU_USD), ("XAGUSD", dki.INSTRUMENT_FX_METALS_XAG_USD)]

START_DATE = datetime.datetime(2024, 1, 1)
END_DATE = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_MIN_15
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

SMA_LEN = 20
ATR_LEN = 14
BREAKOUT_ATR_OFFSET = 0.5
REWARD_ATR_MULT = 5.0
RISK_ATR_MULT = 1.0
MAX_HOLD_BARS = 5
POSITION_PCT = 0.20
STARTING_EQUITY = 10000.0


def compute_sma(values, length):
    n = len(values)
    out = [None] * n
    for i in range(length - 1, n):
        out[i] = sum(values[i - length + 1:i + 1]) / length
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


def backtest_ticker(label, instrument_const):
    df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, START_DATE, END_DATE)
    if df.empty:
        return None
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})

    highs, lows, closes = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
    n = len(closes)

    sma20 = compute_sma(closes, SMA_LEN)
    atr14 = compute_atr_series(highs, lows, closes, ATR_LEN)

    trades = []
    equity = STARTING_EQUITY
    i = max(SMA_LEN, ATR_LEN) + 1
    while i < n:
        if sma20[i] is None or atr14[i] is None or atr14[i] <= 0:
            i += 1
            continue

        breakout_level = sma20[i] + BREAKOUT_ATR_OFFSET * atr14[i]
        if not (closes[i] > breakout_level):
            i += 1
            continue

        entry = closes[i]
        entry_atr = atr14[i]
        risk = RISK_ATR_MULT * entry_atr
        reward = REWARD_ATR_MULT * entry_atr
        stop = entry - risk
        target = entry + reward
        units = (equity * POSITION_PCT) / entry

        outcome, exit_price = None, None
        j_final = min(i + MAX_HOLD_BARS, n - 1)
        for j in range(i + 1, j_final + 1):
            if lows[j] <= stop:
                outcome, exit_price = "SL", stop
                break
            if highs[j] >= target:
                outcome, exit_price = "TP", target
                break
        else:
            outcome, exit_price = "TIMEOUT", closes[j_final]

        pnl_dollars = units * (exit_price - entry)
        equity += pnl_dollars
        r_multiple = (exit_price - entry) / risk

        trades.append({
            "ticker": label, "outcome": outcome, "r": r_multiple,
            "pnl_dollars": pnl_dollars, "equity_after": equity,
        })
        i = j_final + 1

    return trades


def main():
    all_trades = []
    for label, instrument_const in TICKERS:
        print(f"{label}...", end=" ")
        try:
            trades = backtest_ticker(label, instrument_const)
        except Exception as exc:
            print(f"failed ({exc})")
            continue
        if trades is None:
            print("no data")
            continue
        all_trades.extend(trades)
        total_r = sum(t["r"] for t in trades)
        final_equity = trades[-1]["equity_after"] if trades else STARTING_EQUITY
        print(f"{len(trades)} trades, {total_r:+.2f}R, equity ${STARTING_EQUITY:,.0f} -> ${final_equity:,.0f}")

    print("\n" + "=" * 60)
    print(f"TOTAL across {len(TICKERS)} tickers, {START_DATE.date()} to {END_DATE.date()}, "
          f"REWARD:RISK={REWARD_ATR_MULT:.0f}:{RISK_ATR_MULT:.0f}")
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

    breakeven_wr = 1 / (1 + REWARD_ATR_MULT / RISK_ATR_MULT) * 100
    print(f"Total: {total_r:+.2f}R   Average: {total_r/len(df):+.3f}R/trade   <- this decides profitability, not win rate")
    print(f"\nOutcome breakdown ({len(df)} trades):")
    print(f"  TP  (+{REWARD_ATR_MULT/RISK_ATR_MULT:.1f}R each): {tp_count:5d}  ({tp_count/len(df)*100:.1f}%)")
    print(f"  SL  (-1.0R each):  {sl_count:5d}  ({sl_count/len(df)*100:.1f}%)")
    print(f"  Timed out:         {len(timeout_df):5d}  ({len(timeout_df)/len(df)*100:.1f}%)  "
          f"[{timeout_pos} closed positive, {timeout_neg} closed negative]")
    print(f"\nBreakeven win rate for this {REWARD_ATR_MULT:.0f}:{RISK_ATR_MULT:.0f} payout is {breakeven_wr:.1f}%.")

    print(f"\nEquity (flat {POSITION_PCT*100:.0f}% notional sizing):")
    for label, _ in TICKERS:
        ticker_trades = [t for t in all_trades if t["ticker"] == label]
        if ticker_trades:
            print(f"  {label}: ${STARTING_EQUITY:,.0f} -> ${ticker_trades[-1]['equity_after']:,.0f}")

    print("\nNo commission/spread/slippage modeled above - real results will be worse than this.")


if __name__ == "__main__":
    main()
