# How long would it actually take to pass a prop firm challenge with the
# ONE validated result from this project - ORB on indices, 10-minute
# opening range, 2:1 reward:risk, filters on (range-vs-ATR, impulsive
# candle, relative volume, volatility regime). This is the only strategy
# tested here that survived an honest out-of-sample check
# (orb_indices_optimization_and_ml.py: +0.0625R/trade over 721 trades in
# 2024, data the parameter search never saw).
#
# Strategy logic is identical to orb_indices_optimization_and_ml.py's
# run_backtest with RANGE_MINUTES=10, REWARD_RISK=2.0, apply_filters=True
# (its grid-search winner) - just with a date field added per trade so
# the prop-firm challenge simulator (same logic as
# prop_firm_challenge_simulator.py) can group by day for the daily
# drawdown check.
#
# Runs over the FULL available history (2016-2025, not just the 2024
# out-of-sample slice) so the Monte Carlo pass-rate/days-to-pass
# estimate has a real trade sample to draw from, not just 721 trades.

# !pip install --upgrade dukascopy-python -q   # uncomment this line in Colab

import datetime
import random as _random

import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

INDICES = [
    ("SP500", dki.INSTRUMENT_IDX_AMERICA_E_SANDP_500, "America/New_York", pd.Timestamp("09:30").time()),
    ("NASDAQ100", dki.INSTRUMENT_IDX_AMERICA_E_NQ_100, "America/New_York", pd.Timestamp("09:30").time()),
    ("DOWJONES", dki.INSTRUMENT_IDX_AMERICA_E_D_J_IND, "America/New_York", pd.Timestamp("09:30").time()),
    ("DAX", dki.INSTRUMENT_IDX_EUROPE_E_DAAX, "Europe/Berlin", pd.Timestamp("09:00").time()),
    ("FTSE100", dki.INSTRUMENT_IDX_EUROPE_E_FUTSEE_100, "Europe/London", pd.Timestamp("08:00").time()),
    ("NIKKEI225", dki.INSTRUMENT_IDX_ASIA_E_N225JAP, "Asia/Tokyo", pd.Timestamp("09:00").time()),
]

FETCH_START = datetime.datetime(2016, 1, 1)
FETCH_END = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_MIN_5
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

# --- the validated winning combination from orb_indices_optimization_and_ml.py ---
RANGE_MINUTES = 10
REWARD_RISK = 2.0
ENTRY_WINDOW_MINUTES = 180
SESSION_HOLD_HOURS = 8
ENTRY_BUFFER_PCT = 0.02
MIN_RANGE_PCT = 0.05
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

# --- prop firm challenge rules (same defaults as prop_firm_challenge_simulator.py) ---
INITIAL_BALANCE = 10000.0
RISK_PCT_PER_TRADE = 1.0
PROFIT_TARGET_PCT = 8.0
MAX_DAILY_LOSS_PCT = 5.0
MAX_OVERALL_LOSS_PCT = 10.0
MIN_TRADING_DAYS = 4
DRAWDOWN_MODE = "static"
N_MONTE_CARLO_RUNS = 300


def to_local_time(index, tz_name):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(tz_name)


def persisted_avg(values, length, start_index=0):
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
    df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, FETCH_START, FETCH_END)
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
            if (highs[i] - lows[i]) < IMPULSE_RANGE_MULT * range_avg[i]:
                traded_today = True
                i += 1
                continue
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

        # date drives the prop-firm challenge's daily-drawdown grouping below
        trades.append({"index": label, "side": side, "outcome": outcome, "r": exit_r, "date": today})
        i = j + 1

    return trades


# --- prop firm challenge simulation (identical logic to prop_firm_challenge_simulator.py) ---

def simulate_challenge(trades, start_idx,
                        initial_balance=INITIAL_BALANCE, risk_pct_per_trade=RISK_PCT_PER_TRADE,
                        profit_target_pct=PROFIT_TARGET_PCT, max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
                        max_overall_loss_pct=MAX_OVERALL_LOSS_PCT, min_trading_days=MIN_TRADING_DAYS,
                        drawdown_mode=DRAWDOWN_MODE):
    equity = initial_balance
    peak_equity = initial_balance
    profit_target_level = initial_balance * (1 + profit_target_pct / 100)
    static_floor = initial_balance * (1 - max_overall_loss_pct / 100)
    risk_dollars = initial_balance * (risk_pct_per_trade / 100)

    current_day = None
    day_start_equity = equity
    trading_days_seen = set()

    for idx in range(start_idx, len(trades)):
        trade = trades[idx]
        day = trade["date"]
        if day != current_day:
            current_day = day
            day_start_equity = equity
            trading_days_seen.add(day)

        equity += trade["r"] * risk_dollars
        peak_equity = max(peak_equity, equity)

        daily_loss_pct = (day_start_equity - equity) / initial_balance * 100
        if daily_loss_pct >= max_daily_loss_pct:
            return {"outcome": "FAIL", "reason": "daily_drawdown", "trades_taken": idx - start_idx + 1,
                    "days_taken": len(trading_days_seen), "final_equity": equity}

        overall_floor = static_floor if drawdown_mode == "static" else peak_equity * (1 - max_overall_loss_pct / 100)
        if equity <= overall_floor:
            return {"outcome": "FAIL", "reason": f"overall_drawdown_{drawdown_mode}",
                    "trades_taken": idx - start_idx + 1, "days_taken": len(trading_days_seen), "final_equity": equity}

        if equity >= profit_target_level and len(trading_days_seen) >= min_trading_days:
            return {"outcome": "PASS", "trades_taken": idx - start_idx + 1,
                    "days_taken": len(trading_days_seen), "final_equity": equity}

    return {"outcome": "INCONCLUSIVE", "reason": "ran_out_of_data", "trades_taken": len(trades) - start_idx,
            "days_taken": len(trading_days_seen), "final_equity": equity}


def monte_carlo_pass_rate(trades, n_simulations=N_MONTE_CARLO_RUNS, **challenge_kwargs):
    if len(trades) < 10:
        return []
    rng = _random.Random(20240101)
    max_start = max(1, len(trades) - 5)
    results = []
    for _ in range(n_simulations):
        start_idx = rng.randrange(0, max_start)
        results.append(simulate_challenge(trades, start_idx, **challenge_kwargs))
    return results


def main():
    print(f"Using the validated combo: RANGE_MINUTES={RANGE_MINUTES}, REWARD_RISK={REWARD_RISK}, filters on.")
    print(f"This pulls {len(INDICES)} indices over ~9 years - expect roughly 5-10 minutes.\n")

    all_trades = []
    for label, instrument_const, tz_name, session_start in INDICES:
        print(f"{label}...", end=" ")
        try:
            trades = backtest_index(label, instrument_const, tz_name, session_start)
        except Exception as exc:
            print(f"failed ({exc})")
            continue
        all_trades.extend(trades)
        print(f"{len(trades)} trades")

    if not all_trades:
        print("No trades at all - check ticker output above for download failures.")
        return

    all_trades.sort(key=lambda t: t["date"])

    print("\n" + "=" * 60)
    print(f"HOW LONG TO PASS A PROP FIRM CHALLENGE - {len(all_trades)} real trades, "
          f"{FETCH_START.date()} to {FETCH_END.date()}")
    print(f"${INITIAL_BALANCE:,.0f} account, {RISK_PCT_PER_TRADE:.1f}% risk/trade")
    print(f"Target: +{PROFIT_TARGET_PCT:.0f}%   Max daily loss: {MAX_DAILY_LOSS_PCT:.0f}%   "
          f"Max overall loss: {MAX_OVERALL_LOSS_PCT:.0f}%   Min days: {MIN_TRADING_DAYS}")
    print("=" * 60)

    mc_results = monte_carlo_pass_rate(all_trades)
    if not mc_results:
        print("Not enough trades for a Monte Carlo estimate.")
        return

    mc_df = pd.DataFrame(mc_results)
    n = len(mc_df)
    pass_rate = (mc_df["outcome"] == "PASS").mean() * 100
    fail_rate = (mc_df["outcome"] == "FAIL").mean() * 100
    inconclusive_rate = (mc_df["outcome"] == "INCONCLUSIVE").mean() * 100

    print(f"\n{n} simulated attempts, each starting from a different real point in the trade history:")
    print(f"  PASS:         {pass_rate:5.1f}%")
    print(f"  FAIL:         {fail_rate:5.1f}%")
    print(f"  INCONCLUSIVE: {inconclusive_rate:5.1f}%")

    passes = mc_df[mc_df["outcome"] == "PASS"]
    if len(passes) > 0:
        print(f"\nOf the {len(passes)} passing attempts - THIS is the answer to 'how long':")
        print(f"  Median:  {passes['days_taken'].median():.0f} trading days ({passes['trades_taken'].median():.0f} trades)")
        print(f"  Fastest: {passes['days_taken'].min():.0f} trading days")
        print(f"  Slowest: {passes['days_taken'].max():.0f} trading days")
        print(f"  25th/75th percentile: {passes['days_taken'].quantile(0.25):.0f} / "
              f"{passes['days_taken'].quantile(0.75):.0f} trading days")
    else:
        print("\nNo passing attempts in this Monte Carlo run.")

    fails = mc_df[mc_df["outcome"] == "FAIL"]
    if len(fails) > 0:
        print(f"\nOf the {len(fails)} failures, reason breakdown:")
        print(fails["reason"].value_counts())
        print(f"  Median days before failing: {fails['days_taken'].median():.0f}")

    print(f"\nCAVEAT: daily drawdown checked at trade-close granularity, not tick-by-tick floating equity. "
          f"No commission/spread/slippage modeled. Treat this as a rough real-world estimate, not exact.")


if __name__ == "__main__":
    main()
