# Backtest of the "Gold 15m High-RR vs 1:5 RR" strategy someone posted:
# buy Gold on a 15-min close above (20-SMA + 0.5xATR), target 5xATR,
# stop 1xATR, exit after 5 bars if neither hits. Long-only, as the
# original code only ever defined a buy condition.
#
# Two real fixes made to the pseudocode before this would be worth
# running at all:
#   1. The original had `if (ENTRY_RISKY):` / `if (EXIT_AFTER_5):` -
#      missing the () to actually call those functions. In real Python
#      that checks whether the function OBJECT is truthy (always yes),
#      so it would trade on literally every bar regardless of price.
#      Fixed here by actually evaluating the conditions.
#   2. "SELL_GOLD()" was ambiguous - closing the long, or opening a new
#      short? Implemented here as closing the existing long (which is
#      what "exit after 5 bars" clearly means), not a fresh short entry.
#
# Reward:risk here is a genuine 5:1 (not the 1:1 used elsewhere in this
# project's other scripts) - worth knowing going in: for a target 5x
# farther from entry than the stop, plain random price movement hits the
# CLOSER barrier (the stop) far more often, so the "natural" win rate
# baseline for this exact payout is close to 1/(1+5) = 16.7% - roughly
# breakeven by construction, same mechanism as the fake-80%-win-rate bug
# found earlier in a different script, just running in the other
# direction. The real question is whether the entry signal clears that
# ~17% bar with real margin, not whether win rate is merely positive.
#
# Position sizing is left as the original specified it - a flat 20% of
# current equity notional, NOT the risk-based (% of equity risked to the
# stop) sizing used in this project's other scripts - so this backtest
# also tracks an actual equity curve in dollars, not just R-multiples,
# since R-multiples alone would hide how sizing behaves as ATR (and
# therefore position risk) changes over time.

# !pip install --upgrade yfinance -q   # uncomment this line in Colab

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["GC=F", "XAUUSD=X"]   # Gold futures and spot gold - whichever Yahoo actually has good 15m data for

INTRADAY_INTERVAL = "15m"
INTRADAY_PERIOD = "60d"    # Yahoo's hard cap for 15m data

SMA_LEN = 20
ATR_LEN = 14
BREAKOUT_ATR_OFFSET = 0.5   # entry level = SMA + this many ATRs above it
REWARD_ATR_MULT = 5.0       # TP distance = this many ATRs
RISK_ATR_MULT = 1.0         # SL distance = this many ATRs (so REWARD:RISK = 5:1 here, not 1:1)
MAX_HOLD_BARS = 5           # exit after this many bars if neither TP nor SL hit
POSITION_PCT = 0.20         # flat % of equity notional per trade, as the original specified
STARTING_EQUITY = 10000.0


def compute_sma(values, length):
    n = len(values)
    out = [None] * n
    for i in range(length - 1, n):
        out[i] = sum(values[i - length + 1:i + 1]) / length
    return out


def compute_atr_series(highs, lows, closes, length):
    """Wilder ATR - same seed-then-recurse rule used throughout this
    project (donchian.py, random_baseline_forex_backtest.py, etc.)."""
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


def backtest_ticker(ticker):
    df = yf.download(ticker, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL,
                      progress=False, auto_adjust=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

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
        entry_condition = closes[i] > breakout_level   # the actual call the original code never made
        if not entry_condition:
            i += 1
            continue

        entry = closes[i]
        entry_atr = atr14[i]     # locked in at entry - not re-evaluated bar-to-bar during the hold
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
            "ticker": ticker, "outcome": outcome, "r": r_multiple,
            "pnl_dollars": pnl_dollars, "equity_after": equity,
        })
        i = j_final + 1  # one trade at a time - resume scanning after this one closes

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
        if trades is None:
            print("no data")
            continue
        all_trades.extend(trades)
        total_r = sum(t["r"] for t in trades)
        final_equity = trades[-1]["equity_after"] if trades else STARTING_EQUITY
        print(f"{len(trades)} trades, {total_r:+.2f}R, equity ${STARTING_EQUITY:,.0f} -> ${final_equity:,.0f}")

    print("\n" + "=" * 60)
    print(f"TOTAL across {len(TICKERS)} Gold tickers, {INTRADAY_PERIOD} of {INTRADAY_INTERVAL} data, "
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
    print(f"\nBreakeven win rate for this {REWARD_ATR_MULT:.0f}:{RISK_ATR_MULT:.0f} payout is {breakeven_wr:.1f}% - "
          f"and a target this far from the stop tends to get hit less often than the stop under plain random "
          f"movement, so treat any win rate near there as 'no real edge shown', not 'almost profitable'.")

    print(f"\nEquity (flat {POSITION_PCT*100:.0f}% notional sizing, as the original strategy specified):")
    for ticker in TICKERS:
        ticker_trades = [t for t in all_trades if t["ticker"] == ticker]
        if ticker_trades:
            print(f"  {ticker}: ${STARTING_EQUITY:,.0f} -> ${ticker_trades[-1]['equity_after']:,.0f}")

    print("\nNo commission/spread/slippage modeled above - real results will be worse than this.")


if __name__ == "__main__":
    main()
