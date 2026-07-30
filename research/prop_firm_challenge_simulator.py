# Simulates a prop firm evaluation challenge on top of the MA-setup
# forex strategy's real trade history (moving_average_setup_forex_
# dukascopy_backtest.py's trades, with a date attached to each one) - a
# $10k account, a profit target, and hard drawdown limits that end the
# challenge in FAILURE the instant they're breached, exactly like a real
# funded-account evaluation.
#
# Defaults match a common style of challenge (FTMO's public rules are the
# reference point): 8% profit target, 5% max daily loss, 10% max overall
# loss, evaluated from a $10k account risking 1% of the STARTING balance
# per trade (fixed dollar risk, not re-compounded on current equity -
# more conservative and more typical of how challenge accounts are
# actually risk-managed than letting position size grow with equity).
# All of these are configurable since real firms vary.
#
# Two things this reports:
#   1. A single full-history walkthrough: total return, max drawdown, max
#      daily drawdown actually experienced over the whole real backtest.
#   2. A Monte Carlo pass-rate: since a real trader could start their
#      challenge on any date, this re-runs the challenge simulation
#      starting from many different points in the SAME real trade
#      sequence (preserving chronological order forward from each start -
#      not a fully shuffled bootstrap, since shuffling would destroy any
#      real time-varying performance the strategy has across the year)
#      and reports what fraction of those attempts would have passed,
#      failed, or run out of data before resolving either way.
#
# CAVEAT: daily drawdown is checked at trade-CLOSE granularity only, not
# tick-by-tick floating equity - a real prop firm's daily-loss monitor
# often watches floating (open-position) equity too, which isn't
# reproducible from R-multiple trade outcomes alone. This will
# undercount daily drawdown breaches that would only show up mid-trade
# before recovering by the close. Treat pass rates here as an upper
# bound on what you'd actually experience, not an exact match.

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

LONDON_SESSION_START = pd.Timestamp("08:00").time()
LONDON_SESSION_END = pd.Timestamp("16:30").time()
LONDON_SESSION_ONLY = True

MA_FAST, MA_MID, MA_SLOW = 21, 50, 200
RSI_LEN = 14
CONFIRM_BARS = 6
REWARD_RISK = 2.0
MIN_SL_PCT = 0.05
REVERSE_SIGNALS = False
MAX_HOLD_BARS = 500

# --- prop firm challenge rules (defaults resemble a common FTMO-style challenge) ---
INITIAL_BALANCE = 10000.0
RISK_PCT_PER_TRADE = 1.0        # % of INITIAL_BALANCE risked per trade - fixed dollar risk, not compounded on current equity
PROFIT_TARGET_PCT = 8.0
MAX_DAILY_LOSS_PCT = 5.0
MAX_OVERALL_LOSS_PCT = 10.0
MIN_TRADING_DAYS = 4
DRAWDOWN_MODE = "static"        # "static" = measured from INITIAL_BALANCE; "trailing" = measured from the equity peak so far
N_MONTE_CARLO_RUNS = 300


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
    df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, START_DATE, END_DATE)
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
        entry_date = times[i].date()
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

        # entry_date drives daily-drawdown grouping in the challenge simulator below -
        # not tracked by the plain research/moving_average_setup_forex_dukascopy_backtest.py
        trades.append({"ticker": label, "side": side, "outcome": outcome, "r": exit_r, "date": entry_date})
        i = j + 1

    return trades


# --- prop firm challenge simulation ---

def simulate_challenge(trades, start_idx,
                        initial_balance=INITIAL_BALANCE, risk_pct_per_trade=RISK_PCT_PER_TRADE,
                        profit_target_pct=PROFIT_TARGET_PCT, max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
                        max_overall_loss_pct=MAX_OVERALL_LOSS_PCT, min_trading_days=MIN_TRADING_DAYS,
                        drawdown_mode=DRAWDOWN_MODE):
    """Walks trades[start_idx:] forward in chronological order, as if a
    trader started their challenge right at that trade. Stops the moment
    a drawdown limit is breached (FAIL), the moment the profit target is
    reached AND the minimum trading days requirement is satisfied (PASS),
    or the trade list runs out first (INCONCLUSIVE - not enough history
    to know)."""
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


def full_history_equity_curve(trades, initial_balance=INITIAL_BALANCE, risk_pct_per_trade=RISK_PCT_PER_TRADE):
    """Single full-history walkthrough - not gated by any challenge rule,
    just the raw equity path, to report actual total return / max
    drawdown / max daily drawdown experienced over the whole real
    backtest."""
    equity = initial_balance
    peak_equity = initial_balance
    max_drawdown_pct = 0.0
    risk_dollars = initial_balance * (risk_pct_per_trade / 100)

    current_day = None
    day_start_equity = equity
    max_daily_drawdown_pct = 0.0
    wiped_out_at_trade = None

    for idx, trade in enumerate(trades):
        day = trade["date"]
        if day != current_day:
            current_day = day
            day_start_equity = equity

        equity += trade["r"] * risk_dollars
        peak_equity = max(peak_equity, equity)
        drawdown_pct = (peak_equity - equity) / initial_balance * 100
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

        daily_loss_pct = (day_start_equity - equity) / initial_balance * 100
        max_daily_drawdown_pct = max(max_daily_drawdown_pct, daily_loss_pct)

        # A real account can't go below $0 - it would have been margin-called
        # and stopped out long before this, not kept trading with the same
        # fixed dollar risk into negative territory. Stop the walkthrough
        # here rather than let equity run further into nonsense negative
        # numbers - this function deliberately ignores challenge drawdown
        # rules to show the raw path, but going broke is a hard floor no
        # matter what.
        if equity <= 0:
            wiped_out_at_trade = idx + 1
            equity = 0.0
            break

    total_return_pct = (equity - initial_balance) / initial_balance * 100
    return {
        "final_equity": equity, "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct, "max_daily_drawdown_pct": max_daily_drawdown_pct,
        "wiped_out_at_trade": wiped_out_at_trade,
    }


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
    all_trades = []
    for label, instrument_const in TICKERS:
        print(f"{label}...", end=" ")
        try:
            trades = backtest_pair(label, instrument_const)
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
    print(f"PROP FIRM CHALLENGE SIMULATION - {len(all_trades)} real trades, "
          f"{START_DATE.date()} to {END_DATE.date()}")
    print(f"${INITIAL_BALANCE:,.0f} account, {RISK_PCT_PER_TRADE:.1f}% risk/trade (fixed $ from starting balance)")
    print(f"Target: +{PROFIT_TARGET_PCT:.0f}%   Max daily loss: {MAX_DAILY_LOSS_PCT:.0f}%   "
          f"Max overall loss: {MAX_OVERALL_LOSS_PCT:.0f}% ({DRAWDOWN_MODE})   Min days: {MIN_TRADING_DAYS}")
    print("=" * 60)

    curve = full_history_equity_curve(all_trades, initial_balance=INITIAL_BALANCE,
                                       risk_pct_per_trade=RISK_PCT_PER_TRADE)
    print(f"\nFull-history walkthrough (no challenge rules applied, just the raw path):")
    if curve["wiped_out_at_trade"] is not None:
        print(f"  ACCOUNT WIPED OUT on trade {curve['wiped_out_at_trade']} of {len(all_trades)} - "
              f"equity hit $0 at fixed {RISK_PCT_PER_TRADE:.1f}% risk/trade well before the full "
              f"history played out. Nothing past that point is real - a real account would have been "
              f"margin-called and stopped, not kept trading.")
    print(f"  Total return: {curve['total_return_pct']:+.2f}%   Final equity: ${curve['final_equity']:,.2f}")
    print(f"  Max drawdown (peak to trough): {curve['max_drawdown_pct']:.2f}%")
    print(f"  Max single-day drawdown seen: {curve['max_daily_drawdown_pct']:.2f}%")
    print(f"  (For reference: your daily/overall limits are {MAX_DAILY_LOSS_PCT:.0f}% / {MAX_OVERALL_LOSS_PCT:.0f}%)")

    mc_results = monte_carlo_pass_rate(
        all_trades, n_simulations=N_MONTE_CARLO_RUNS,
        initial_balance=INITIAL_BALANCE, risk_pct_per_trade=RISK_PCT_PER_TRADE,
        profit_target_pct=PROFIT_TARGET_PCT, max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
        max_overall_loss_pct=MAX_OVERALL_LOSS_PCT, min_trading_days=MIN_TRADING_DAYS,
        drawdown_mode=DRAWDOWN_MODE,
    )
    if not mc_results:
        print("\nNot enough trades for a Monte Carlo pass-rate estimate.")
        return

    mc_df = pd.DataFrame(mc_results)
    n = len(mc_df)
    pass_rate = (mc_df["outcome"] == "PASS").mean() * 100
    fail_rate = (mc_df["outcome"] == "FAIL").mean() * 100
    inconclusive_rate = (mc_df["outcome"] == "INCONCLUSIVE").mean() * 100

    print(f"\nMonte Carlo: {n} simulated challenge attempts, each starting from a different real point in "
          f"the trade history (chronological order preserved forward from each start):")
    print(f"  PASS:         {pass_rate:5.1f}%")
    print(f"  FAIL:         {fail_rate:5.1f}%")
    print(f"  INCONCLUSIVE: {inconclusive_rate:5.1f}%  (ran out of trade history before resolving either way)")

    fails = mc_df[mc_df["outcome"] == "FAIL"]
    if len(fails) > 0:
        print(f"\nOf the failures, reason breakdown:")
        print(fails["reason"].value_counts())

    passes = mc_df[mc_df["outcome"] == "PASS"]
    if len(passes) > 0:
        print(f"\nOf the passes: median {passes['trades_taken'].median():.0f} trades, "
              f"{passes['days_taken'].median():.0f} trading days to reach target")

    print(f"\nCAVEAT: daily drawdown is checked at trade-close granularity, not tick-by-tick floating "
          f"equity - real firms often also watch floating equity intra-trade, which isn't reproducible "
          f"from R-multiple outcomes alone. Treat this pass rate as an upper bound, not an exact figure. "
          f"Also: no commission/spread/slippage modeled in the underlying trades themselves.")


if __name__ == "__main__":
    main()
