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
import math
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

# --- risk-per-trade sweep (see the big comment block above simulate_challenge_path) ---
# The set below deliberately spans from "obviously too small to finish in a reasonable
# number of trades" (0.1%) to "obviously too large to survive even one bad day" (5%,
# which at these default account rules breaches the daily-loss limit on the very FIRST
# losing trade - see consecutive_losses_to_breach) - both ends are kept ON PURPOSE
# because showing that collapse at each extreme IS the point of the sweep, not something
# to filter out before showing the reader.
RISK_SWEEP_LEVELS_PCT = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
RISK_SWEEP_N_ITER = 2000               # matches this project's other Monte Carlo work, e.g.
                                        # ict_po3_forex_dukascopy_optimization.py's MC_ITERATIONS
RISK_SWEEP_MAX_TRADES_PER_PATH = 2000  # hard cap per simulated path so a too-small risk level
                                        # can't spin forever - if it doesn't resolve by then the
                                        # attempt is marked INCONCLUSIVE, which is itself signal
                                        # (that risk level is impractically slow for this edge)

# --- losing-streak probability (standalone, sizing-independent diagnostic) ---
LOSING_STREAK_KS = [2, 3, 4, 5, 6, 8]
LOSING_STREAK_REFERENCE_N = 250   # ~1 trade/trading-day/year - a standardized sample size so
                                   # results are comparable across strategies with very
                                   # different real historical trade counts


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


# =============================================================================
# RISK-PER-TRADE SWEEP
# =============================================================================
#
# The point this section demonstrates: with a FIXED edge (this strategy's real,
# already-backtested win rate and R-multiple distribution - not a hypothetical
# toy 60%/1:1 example), the probability of actually reaching the challenge's
# profit target BEFORE violating a loss rule depends heavily on risk-per-trade
# sizing. Not because the edge changes - it's the exact same trade sample at
# every risk level below - but because a bigger risk-per-trade means the account
# can survive fewer consecutive losing trades before hitting the daily-loss or
# max-drawdown limit, and a real edge needs enough "runway" (enough trades) to
# actually converge toward its expected outcome. Smaller risk isn't
# unconditionally better either: shrink it enough and the average number of
# trades needed to reach the profit target becomes impractically large. There's
# a real tradeoff curve here - see print_risk_sweep_table's output.
#
# Bootstrap resampling (WITH replacement) matches this project's established
# Monte Carlo convention from the optimization scripts (see
# ict_po3_forex_dukascopy_optimization.py's monte_carlo_bootstrap): one
# (n_iter, n_per_path) integer draw + fancy indexing, not a Python-level loop.
# The per-path account-rule walkthrough (running balance, early-stop on
# breach/pass) still needs a per-path loop - kept as small and cheap per
# iteration as reasonably possible.


def consecutive_losses_to_breach(risk_pct_per_trade, max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
                                  max_overall_loss_pct=MAX_OVERALL_LOSS_PCT):
    """Deterministic (NOT a simulation): starting from a flat, fresh account (equity ==
    peak == initial balance, so static and trailing overall-drawdown modes agree), how
    many full -1R losing trades IN A ROW can the account absorb before the NEXT one
    breaches a loss rule? Assumes the worst realistic case for the daily-loss check -
    all the losses landing inside the same trading day, since the daily limit resets
    every day and that's the scenario a real trader could actually face.

    Each -1R loss at `risk_pct_per_trade`% of the (fixed, non-compounded) starting
    balance moves daily/overall drawdown by exactly `risk_pct_per_trade` percentage
    points, so the k-th consecutive loss breaches whichever limit is hit first:
    daily at k = ceil(max_daily_loss_pct / risk_pct_per_trade), overall at
    k = ceil(max_overall_loss_pct / risk_pct_per_trade). The account survives
    (breach_at - 1) losses; the breach_at-th one ends the challenge.

    Returns (survivable_losses, breach_reason, breach_at_loss_count).
    """
    if risk_pct_per_trade <= 0:
        return float("inf"), "n/a", float("inf")
    eps = 1e-9
    daily_breach_at = math.ceil(max_daily_loss_pct / risk_pct_per_trade - eps)
    overall_breach_at = math.ceil(max_overall_loss_pct / risk_pct_per_trade - eps)
    if daily_breach_at <= overall_breach_at:
        return daily_breach_at - 1, "daily_drawdown", daily_breach_at
    return overall_breach_at - 1, "overall_drawdown", overall_breach_at


def estimate_trades_per_day(trades):
    """Average trades/day empirically observed in the real trade list - used to bucket
    a bootstrap-resampled trade sequence into SYNTHETIC trading days for the
    daily-loss-limit check. This is a modeling simplification (real trading days don't
    all have the same trade count) made necessary because resampling individual
    R-multiples with replacement destroys the real calendar dates that
    simulate_challenge() above groups by - called out explicitly in the sweep's
    printed caveats, not hidden."""
    if not trades:
        return 1
    n_days = len(set(t["date"] for t in trades))
    if n_days == 0:
        return 1
    return max(1, round(len(trades) / n_days))


def bootstrap_resample_r_matrix(r_values, n_iter, n_per_path, rng):
    """Bootstrap resample WITH replacement, vectorized exactly like this project's other
    Monte Carlo work: one (n_iter, n_per_path) integer draw + fancy indexing, not a
    Python-level loop. Shared by the risk sweep and the losing-streak diagnostic below."""
    r = np.asarray(r_values, dtype=float)
    n = len(r)
    idx = rng.integers(0, n, size=(n_iter, n_per_path))
    return r[idx]


def simulate_challenge_path(r_values_path, initial_balance, risk_pct_per_trade, profit_target_pct,
                             max_daily_loss_pct, max_overall_loss_pct, min_trading_days,
                             drawdown_mode, trades_per_day):
    """One bootstrap-resampled trade sequence run through the SAME account-rule
    mechanics as simulate_challenge() above (early-stop on the first PASS/FAIL) - just
    fed synthetic R-multiples instead of the real chronological trade list.
    trades_per_day buckets the sequence into synthetic trading days for the daily-loss
    check (see estimate_trades_per_day's docstring for why)."""
    trades_per_day = max(1, trades_per_day)
    equity = initial_balance
    peak_equity = initial_balance
    profit_target_level = initial_balance * (1 + profit_target_pct / 100)
    static_floor = initial_balance * (1 - max_overall_loss_pct / 100)
    risk_dollars = initial_balance * (risk_pct_per_trade / 100)

    day_start_equity = equity
    trading_days_seen = 0
    consecutive_losses = 0
    max_consecutive_losses = 0

    for idx, r in enumerate(r_values_path):
        if idx % trades_per_day == 0:
            day_start_equity = equity
            trading_days_seen += 1

        equity += r * risk_dollars
        peak_equity = max(peak_equity, equity)
        consecutive_losses = consecutive_losses + 1 if r < 0 else 0
        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

        daily_loss_pct = (day_start_equity - equity) / initial_balance * 100
        if daily_loss_pct >= max_daily_loss_pct:
            return {"outcome": "FAIL", "reason": "daily_drawdown", "trades_taken": idx + 1,
                    "days_taken": trading_days_seen, "final_equity": equity,
                    "max_consecutive_losses": max_consecutive_losses}

        overall_floor = static_floor if drawdown_mode == "static" else peak_equity * (1 - max_overall_loss_pct / 100)
        if equity <= overall_floor:
            return {"outcome": "FAIL", "reason": f"overall_drawdown_{drawdown_mode}", "trades_taken": idx + 1,
                    "days_taken": trading_days_seen, "final_equity": equity,
                    "max_consecutive_losses": max_consecutive_losses}

        if equity >= profit_target_level and trading_days_seen >= min_trading_days:
            return {"outcome": "PASS", "trades_taken": idx + 1, "days_taken": trading_days_seen,
                    "final_equity": equity, "max_consecutive_losses": max_consecutive_losses}

    return {"outcome": "INCONCLUSIVE", "reason": "ran_out_of_bootstrap_path", "trades_taken": len(r_values_path),
            "days_taken": trading_days_seen, "final_equity": equity,
            "max_consecutive_losses": max_consecutive_losses}


def risk_sweep(trades, risk_levels_pct=None, n_iter=RISK_SWEEP_N_ITER,
                max_trades_per_path=RISK_SWEEP_MAX_TRADES_PER_PATH,
                initial_balance=INITIAL_BALANCE, profit_target_pct=PROFIT_TARGET_PCT,
                max_daily_loss_pct=MAX_DAILY_LOSS_PCT, max_overall_loss_pct=MAX_OVERALL_LOSS_PCT,
                min_trading_days=MIN_TRADING_DAYS, drawdown_mode=DRAWDOWN_MODE, seed=2026):
    """THE risk-per-trade sweep. For each risk level, bootstrap-resamples n_iter trade
    sequences WITH replacement from `trades`' real R-multiples (not a synthetic/assumed
    win rate) and runs each one through the exact same account-rule mechanics as
    simulate_challenge() - just repeated at scale, across many risk-per-trade fractions.

    The SAME underlying bootstrap-resampled paths (same random draw, same seed) are
    reused across every risk level - only risk_dollars and the resulting pass/fail
    thresholds change - so differences in outcome ACROSS risk levels are attributable to
    risk sizing, not extra sampling noise (a common-random-numbers variance-reduction
    trick).

    Returns a list of dicts, one per risk level, in the order given."""
    if risk_levels_pct is None:
        risk_levels_pct = RISK_SWEEP_LEVELS_PCT
    if len(trades) < 10:
        return []

    r_values = np.array([t["r"] for t in trades], dtype=float)
    trades_per_day = estimate_trades_per_day(trades)
    rng = np.random.default_rng(seed)
    r_matrix = bootstrap_resample_r_matrix(r_values, n_iter, max_trades_per_path, rng)

    sweep_results = []
    for risk_pct in risk_levels_pct:
        survivable_losses, breach_reason, breach_at = consecutive_losses_to_breach(
            risk_pct, max_daily_loss_pct, max_overall_loss_pct)

        outcomes = [
            simulate_challenge_path(path, initial_balance, risk_pct, profit_target_pct,
                                     max_daily_loss_pct, max_overall_loss_pct, min_trading_days,
                                     drawdown_mode, trades_per_day)
            for path in r_matrix
        ]

        n = len(outcomes)
        passes = [o for o in outcomes if o["outcome"] == "PASS"]
        n_pass, n_fail = len(passes), sum(1 for o in outcomes if o["outcome"] == "FAIL")
        n_inconclusive = n - n_pass - n_fail
        hit_breach_streak = sum(1 for o in outcomes if o["max_consecutive_losses"] >= breach_at)

        sweep_results.append({
            "risk_pct_per_trade": risk_pct,
            "n_iter": n,
            "pass_prob": n_pass / n,
            "fail_prob": n_fail / n,
            "inconclusive_prob": n_inconclusive / n,
            "avg_trades_to_pass": float(np.mean([o["trades_taken"] for o in passes])) if passes else float("nan"),
            "avg_days_to_pass": float(np.mean([o["days_taken"] for o in passes])) if passes else float("nan"),
            "max_survivable_consecutive_losses": survivable_losses,
            "breach_reason": breach_reason,
            "consecutive_losses_that_would_breach": breach_at,
            "frac_paths_hitting_breach_streak": hit_breach_streak / n,
            "trades_per_day_assumed": trades_per_day,
        })

    return sweep_results


def print_risk_sweep_table(sweep_results, header="RISK-PER-TRADE SWEEP"):
    if not sweep_results:
        print(f"\n{header}: not enough trades to run the sweep.")
        return

    print(f"\n{'=' * 78}\n{header} - same real bootstrap-resampled trade sequences, "
          f"{sweep_results[0]['n_iter']} Monte Carlo attempts per risk level (synthetic trading days of "
          f"{sweep_results[0]['trades_per_day_assumed']} trades/day, from this strategy's own trade rate)\n"
          f"{'=' * 78}")
    print(f"{'risk%':>7} {'pass%':>7} {'fail%':>7} {'inconcl%':>9} {'avg trades':>11} {'avg days':>9} "
          f"{'survives':>9} {'breach@':>8} {'streak seen':>12}")
    for row in sweep_results:
        avg_trades = f"{row['avg_trades_to_pass']:.0f}" if not math.isnan(row["avg_trades_to_pass"]) else "n/a"
        avg_days = f"{row['avg_days_to_pass']:.0f}" if not math.isnan(row["avg_days_to_pass"]) else "n/a"
        print(f"{row['risk_pct_per_trade']:>6.2f}% {row['pass_prob'] * 100:>6.1f}% {row['fail_prob'] * 100:>6.1f}% "
              f"{row['inconclusive_prob'] * 100:>8.1f}% {avg_trades:>11} {avg_days:>9} "
              f"{row['max_survivable_consecutive_losses']:>9} {row['consecutive_losses_that_would_breach']:>8} "
              f"{row['frac_paths_hitting_breach_streak'] * 100:>10.1f}%")

    best = max(sweep_results, key=lambda r: r["pass_prob"])
    finite_speed = [r for r in sweep_results if not math.isnan(r["avg_trades_to_pass"])]
    fastest = min(finite_speed, key=lambda r: r["avg_trades_to_pass"]) if finite_speed else None

    print(f"\nHighest payout probability: {best['risk_pct_per_trade']:.2f}% risk/trade -> "
          f"{best['pass_prob'] * 100:.1f}% of attempts reached the target (surviving up to "
          f"{best['max_survivable_consecutive_losses']} losses in a row before the {best['breach_reason']} "
          f"rule would end the challenge).")
    if fastest is not None:
        print(f"Fastest average time-to-payout (among attempts that passed): "
              f"{fastest['risk_pct_per_trade']:.2f}% risk/trade -> {fastest['avg_trades_to_pass']:.0f} trades "
              f"/ {fastest['avg_days_to_pass']:.0f} days on average.")
        if fastest["risk_pct_per_trade"] != best["risk_pct_per_trade"]:
            print("These are DIFFERENT risk levels - that IS the tradeoff: the level that passes most often is "
                  "not necessarily the level that pays out fastest. Smaller risk/trade buys more 'runway' (more "
                  "consecutive losses survivable), which raises pass probability, but each trade also moves the "
                  "account a smaller fraction of the way to the profit target, so reaching it takes more trades "
                  "on average - see the avg-trades column climb as risk shrinks. There is a real sweet spot here, "
                  "not a monotonic 'always shrink it' answer.")
        else:
            print("Here the highest-pass-probability level and the fastest-to-payout level coincide.")

    print(f"\nCAVEAT: this uses bootstrap resampling of a FINITE historical trade sample (with replacement) - "
          f"it approximates true future variance but is not a guarantee of any specific probability, especially "
          f"at the tails. Trading-day boundaries inside each resampled path are SYNTHETIC (grouped by this "
          f"strategy's own average trades/day, since resampling individual R-multiples destroys the real "
          f"calendar dates the single-run simulator groups by) - a real trader's actual daily clustering could "
          f"differ. No commission/spread/slippage modeled in the underlying trades. 'max survivable consecutive "
          f"losses' is a deterministic worst-case calculation from the account rules alone, not itself a Monte "
          f"Carlo output - it assumes the losses land on the same trading day, the true worst case.")


# =============================================================================
# MULTI-PHASE CHALLENGE SIMULATION (real prop firms - see prop_firm_presets.py)
# =============================================================================
#
# Real prop firms almost universally run 2 (sometimes 1) sequential evaluation
# PHASES, each with its own profit target/daily-loss/drawdown/min-days rules -
# a trader must clear every phase to get funded. The single-phase
# simulate_challenge_path() above only models one target, which understates how
# hard a real 2-step challenge actually is (clearing a 10% target is not the
# same as clearing 10% THEN 5% with a fresh drawdown floor). This section chains
# simulate_challenge_path() calls, one per phase, against ONE continuous
# bootstrap-resampled trade sequence - not independent re-draws per phase - so a
# multi-phase attempt consumes the trader's simulated trades in one continuous
# string, exactly like a real trader would move from Phase 1 into Phase 2
# without their trade-taking behavior resetting.

def simulate_multi_phase_challenge_path(r_values_path, initial_balance, risk_pct_per_trade,
                                         phases, trades_per_day):
    """Chains simulate_challenge_path() once per phase in `phases` (a list of dicts with
    profit_target_pct/max_daily_loss_pct/max_overall_loss_pct/min_trading_days/drawdown_mode -
    see prop_firm_presets.py). Each phase starts FRESH from `initial_balance` (matching how
    real prop firms reset the account for the next phase rather than carrying forward the
    prior phase's ending equity/profit), but consumes the NEXT unused portion of
    r_values_path - the trade sequence itself is never restarted between phases.

    Stops at the first phase that doesn't PASS. Returns:
      {"outcome": "PASS"|"FAIL"|"INCONCLUSIVE", "phase_failed": int|None (0-indexed, None if
       outcome is PASS), "trades_taken": int, "days_taken": int, "max_consecutive_losses": int}
    trades_taken/days_taken are SUMMED across every phase actually attempted (a FAIL in phase 2
    still counts phase 1's trades/days, since the trader really did take them)."""
    cursor = 0
    total_trades = 0
    total_days = 0
    overall_max_consec_losses = 0

    for phase_idx, phase in enumerate(phases):
        remaining_path = r_values_path[cursor:]
        if len(remaining_path) == 0:
            return {"outcome": "INCONCLUSIVE", "phase_failed": phase_idx, "trades_taken": total_trades,
                    "days_taken": total_days, "max_consecutive_losses": overall_max_consec_losses}

        result = simulate_challenge_path(
            remaining_path, initial_balance, risk_pct_per_trade,
            phase["profit_target_pct"], phase["max_daily_loss_pct"], phase["max_overall_loss_pct"],
            phase["min_trading_days"], phase["drawdown_mode"], trades_per_day)

        cursor += result["trades_taken"]
        total_trades += result["trades_taken"]
        total_days += result["days_taken"]
        overall_max_consec_losses = max(overall_max_consec_losses, result["max_consecutive_losses"])

        if result["outcome"] != "PASS":
            return {"outcome": result["outcome"], "phase_failed": phase_idx, "trades_taken": total_trades,
                    "days_taken": total_days, "max_consecutive_losses": overall_max_consec_losses}

    return {"outcome": "PASS", "phase_failed": None, "trades_taken": total_trades,
            "days_taken": total_days, "max_consecutive_losses": overall_max_consec_losses}


def multi_phase_risk_sweep(trades, preset, risk_levels_pct=None, n_iter=RISK_SWEEP_N_ITER,
                            max_trades_per_path=None, seed=2026):
    """The multi-phase equivalent of risk_sweep() above, for ONE real prop-firm preset (see
    prop_firm_presets.py) across a range of risk-per-trade levels. `preset` is a dict as
    returned by prop_firm_presets.get_preset() - uses its own initial_balance and phases list.

    max_trades_per_path defaults to 2x RISK_SWEEP_MAX_TRADES_PER_PATH if not given - a
    multi-phase attempt needs enough runway to get through every phase, not just one.

    Returns a list of dicts, one per risk level: pass_prob (cleared EVERY phase),
    fail_prob, inconclusive_prob, avg_trades_to_pass/avg_days_to_pass (among full passes),
    and fail_by_phase - a dict of {phase_index: count} showing which phase most commonly
    ends a failed attempt (a genuinely useful diagnostic a single-phase sweep can't show:
    e.g. "most failures happen in Phase 2's tighter drawdown floor, not Phase 1")."""
    if risk_levels_pct is None:
        risk_levels_pct = RISK_SWEEP_LEVELS_PCT
    if len(trades) < 10:
        return []
    if max_trades_per_path is None:
        max_trades_per_path = RISK_SWEEP_MAX_TRADES_PER_PATH * 2

    r_values = np.array([t["r"] for t in trades], dtype=float)
    trades_per_day = estimate_trades_per_day(trades)
    rng = np.random.default_rng(seed)
    r_matrix = bootstrap_resample_r_matrix(r_values, n_iter, max_trades_per_path, rng)

    initial_balance = preset["initial_balance"]
    phases = preset["phases"]
    n_phases = len(phases)

    sweep_results = []
    for risk_pct in risk_levels_pct:
        outcomes = [
            simulate_multi_phase_challenge_path(path, initial_balance, risk_pct, phases, trades_per_day)
            for path in r_matrix
        ]

        n = len(outcomes)
        passes = [o for o in outcomes if o["outcome"] == "PASS"]
        n_pass = len(passes)
        n_fail = sum(1 for o in outcomes if o["outcome"] == "FAIL")
        n_inconclusive = n - n_pass - n_fail

        fail_by_phase = {i: 0 for i in range(n_phases)}
        for o in outcomes:
            if o["outcome"] == "FAIL" and o["phase_failed"] is not None:
                fail_by_phase[o["phase_failed"]] += 1

        # days-to-pass distribution, not just the mean - a mean gets dragged around by a
        # long right tail (a handful of unlucky-but-still-passing paths that took much
        # longer than typical), so "how long will THIS actually take me" is better answered
        # by the median plus a p25-p75 "typical range" than by a single average number.
        pass_days = [o["days_taken"] for o in passes]
        pass_trades = [o["trades_taken"] for o in passes]
        if passes:
            median_days = float(np.median(pass_days))
            p25_days, p75_days = (float(x) for x in np.percentile(pass_days, [25, 75]))
            median_trades = float(np.median(pass_trades))
        else:
            median_days = p25_days = p75_days = median_trades = float("nan")

        sweep_results.append({
            "risk_pct_per_trade": risk_pct,
            "n_iter": n,
            "pass_prob": n_pass / n,
            "fail_prob": n_fail / n,
            "inconclusive_prob": n_inconclusive / n,
            "avg_trades_to_pass": float(np.mean(pass_trades)) if passes else float("nan"),
            "avg_days_to_pass": float(np.mean(pass_days)) if passes else float("nan"),
            "median_days_to_pass": median_days,
            "p25_days_to_pass": p25_days,
            "p75_days_to_pass": p75_days,
            "median_trades_to_pass": median_trades,
            "fail_by_phase": fail_by_phase,
            "trades_per_day_assumed": trades_per_day,
        })

    return sweep_results


def print_multi_phase_sweep_table(sweep_results, preset, header=None):
    if not sweep_results:
        print(f"\n{header or 'MULTI-PHASE PROP FIRM SWEEP'}: not enough trades to run the sweep.")
        return

    phase_names = [p["name"] for p in preset["phases"]]
    header = header or f"MULTI-PHASE SWEEP - {preset['display_name']}"
    print(f"\n{'=' * 78}\n{header}\n"
          f"({sweep_results[0]['n_iter']} Monte Carlo attempts per risk level, chained across "
          f"{len(phase_names)} phase(s): {', '.join(phase_names)})\n{'=' * 78}")
    print(f"{'risk%':>7} {'pass%':>7} {'fail%':>7} {'inconcl%':>9} {'avg trades':>11} {'avg days':>9}  "
          f"fail-by-phase")
    for row in sweep_results:
        avg_trades = f"{row['avg_trades_to_pass']:.0f}" if not math.isnan(row["avg_trades_to_pass"]) else "n/a"
        avg_days = f"{row['avg_days_to_pass']:.0f}" if not math.isnan(row["avg_days_to_pass"]) else "n/a"
        fail_breakdown = "  ".join(f"{phase_names[i]}={cnt}" for i, cnt in row["fail_by_phase"].items())
        print(f"{row['risk_pct_per_trade']:>6.2f}% {row['pass_prob'] * 100:>6.1f}% {row['fail_prob'] * 100:>6.1f}% "
              f"{row['inconclusive_prob'] * 100:>8.1f}% {avg_trades:>11} {avg_days:>9}  {fail_breakdown}")

    best = max(sweep_results, key=lambda r: r["pass_prob"])
    print(f"\nHighest full-pass probability: {best['risk_pct_per_trade']:.2f}% risk/trade -> "
          f"{best['pass_prob'] * 100:.1f}% of attempts cleared every phase.")
    if not math.isnan(best["median_days_to_pass"]):
        print(f"At that risk level, passing attempts typically took {best['median_days_to_pass']:.0f} days "
              f"(median), usually somewhere between {best['p25_days_to_pass']:.0f} and "
              f"{best['p75_days_to_pass']:.0f} days (25th-75th percentile).")
    print(f"\nSource: {', '.join(preset['source_urls'])}")
    print(f"Sourcing note: {preset['source_note']}")
    print(f"\nCAVEAT: bootstrap resampling of a FINITE historical trade sample - approximates future "
          f"variance, not a guarantee. No commission/spread/slippage modeled in the underlying trades. "
          f"Each phase resets to a fresh account balance (real prop-firm convention), consuming the next "
          f"unused portion of the same continuous simulated trade sequence.")


# =============================================================================
# LOSING-STREAK PROBABILITY (standalone, sizing-independent diagnostic)
# =============================================================================
#
# A related but DIFFERENT idea from the risk sweep above: the probability of
# hitting a losing streak of a given length is a property of the strategy's win
# rate and sample size, NOT of position sizing - it doesn't involve risk-per-trade
# or account rules at all. Even a genuinely good edge will very likely produce a
# scary-looking losing streak somewhere in a typical year of trading, and that's
# normal variance, not proof the edge broke. This section quantifies exactly how
# likely, using this strategy's own real trade sequence.


def prob_run_at_least_k(n, k, p):
    """Exact probability, assuming n iid Bernoulli(p) trials (p = probability a single
    trial is a 'loss'), of at least one run of >= k consecutive losses somewhere in the
    n trials. Computed via a small forward DP over "current consecutive-loss count"
    states 0..k-1 plus an absorbing state k ("a qualifying run has already occurred") -
    exact for the iid-Bernoulli MODEL, not an approximation. Whether real trade outcomes
    actually behave like iid Bernoulli draws is a separate question - that's exactly
    what comparing this to the empirical bootstrap number (below) checks."""
    if k <= 0:
        return 1.0
    if n < k:
        return 0.0
    p = min(max(p, 0.0), 1.0)
    state = [0.0] * (k + 1)
    state[0] = 1.0
    for _ in range(n):
        new_state = [0.0] * (k + 1)
        new_state[k] = state[k]
        for i in range(k):
            si = state[i]
            if si == 0.0:
                continue
            nxt = min(i + 1, k)
            new_state[nxt] += si * p
            new_state[0] += si * (1 - p)
        state = new_state
    return state[k]


def losing_streak_probabilities(r_values, ks=None, n_iter=RISK_SWEEP_N_ITER, sample_sizes=None, seed=7):
    """Standalone diagnostic (no risk-per-trade or account rules involved - purely a
    property of the strategy's win/loss SEQUENCE): for a typical run of N trades, what's
    the probability of a losing streak of length >= k somewhere in it? A trade counts as
    a loss iff its R-multiple is < 0 (a breakeven trade, r == 0, is neither a win nor a
    loss and resets a losing streak same as a win would).

    Reports BOTH:
      - empirical: bootstrap-resample N trades WITH replacement from the real R-multiple
        history (same machinery as the risk-per-trade sweep above), n_iter times, and
        measure how often the longest consecutive-loss run in the simulated sequence
        reaches >= k.
      - theoretical: the exact iid-Bernoulli closed form (prob_run_at_least_k) at the
        strategy's empirical loss rate, for comparison. A big gap between the two means
        real outcomes are NOT well-approximated as independent draws (e.g. correlated
        market regimes) - reported plainly, not smoothed over.
    """
    if ks is None:
        ks = LOSING_STREAK_KS
    r = np.asarray(r_values, dtype=float)
    n_total = len(r)
    if n_total < 10:
        return []

    loss_rate = float((r < 0).mean())
    win_rate = float((r > 0).mean())

    if sample_sizes is None:
        sample_sizes = sorted(set([n_total, LOSING_STREAK_REFERENCE_N]))

    rng = np.random.default_rng(seed)
    results = []
    for n in sample_sizes:
        if n <= 0:
            continue
        r_matrix = bootstrap_resample_r_matrix(r, n_iter, n, rng)
        loss_bool = r_matrix < 0
        consec = np.zeros((n_iter, n), dtype=np.int64)
        consec[:, 0] = loss_bool[:, 0]
        for j in range(1, n):
            consec[:, j] = np.where(loss_bool[:, j], consec[:, j - 1] + 1, 0)
        longest = consec.max(axis=1)

        per_k = []
        for k in ks:
            empirical_p = float((longest >= k).mean())
            theoretical_p = prob_run_at_least_k(n, k, loss_rate)
            per_k.append({"k": k, "empirical_p": empirical_p, "theoretical_p": theoretical_p})

        results.append({"sample_size": n, "n_iter": n_iter, "loss_rate": loss_rate,
                         "win_rate": win_rate, "avg_r_per_trade": float(r.mean()), "per_k": per_k})

    return results


def print_losing_streak_table(streak_results, header="LOSING-STREAK PROBABILITY (sizing-independent)"):
    if not streak_results:
        print(f"\n{header}: not enough trades to run this diagnostic.")
        return

    win_rate = streak_results[0]["win_rate"]
    avg_r = streak_results[0]["avg_r_per_trade"]
    print(f"\n{'=' * 78}\n{header}\n{'=' * 78}")
    print(f"Real empirical win rate: {win_rate * 100:.1f}%   Avg R/trade: {avg_r:+.4f}   "
          f"(does NOT depend on risk-per-trade or account rules - a property of the win/loss sequence alone)")

    for res in streak_results:
        print(f"\n  Sample size N={res['sample_size']} trades ({res['n_iter']} bootstrap iterations):")
        print(f"    {'k (streak len)':>16} {'empirical P(>=k)':>18} {'theoretical P(>=k)':>20} {'gap':>8}")
        for row in res["per_k"]:
            gap = row["empirical_p"] - row["theoretical_p"]
            print(f"    {row['k']:>16} {row['empirical_p'] * 100:>17.1f}% {row['theoretical_p'] * 100:>19.1f}% "
                  f"{gap * 100:>+7.1f}pp")

    ref = next((r for r in streak_results if r["sample_size"] == LOSING_STREAK_REFERENCE_N), streak_results[-1])
    k_focus = 4 if any(row["k"] == 4 for row in ref["per_k"]) else ref["per_k"][0]["k"]
    focus_row = next(row for row in ref["per_k"] if row["k"] == k_focus)
    p_pct = focus_row["empirical_p"] * 100

    if avg_r > 0:
        print(f"\nSummary: even at this strategy's real {win_rate * 100:.1f}% win rate (positive expectancy, "
              f"{avg_r:+.4f}R/trade average), there's a {p_pct:.0f}% chance of a {k_focus}-in-a-row losing "
              f"streak somewhere in a typical {ref['sample_size']}-trade run - expected variance from a real "
              f"edge, not necessarily a sign the edge broke.")
    elif avg_r == 0:
        print(f"\nSummary: this strategy's real average R/trade is exactly breakeven ({avg_r:+.4f}R) - a "
              f"{p_pct:.0f}% chance of a {k_focus}-in-a-row losing streak in {ref['sample_size']} trades here "
              f"isn't reassuring OR damning on its own, since there's no real edge either way to attribute "
              f"the streak to variance around.")
    else:
        print(f"\nSummary: this strategy's real average R/trade is NEGATIVE ({avg_r:+.4f}R/trade) - a "
              f"{p_pct:.0f}% chance of a {k_focus}-in-a-row losing streak in {ref['sample_size']} trades is NOT "
              f"'just variance' to wave off here. With a decisively negative edge, losing streaks are the "
              f"expected long-run OUTCOME, not noise around a real edge.")

    max_gap = max(abs(row["empirical_p"] - row["theoretical_p"]) for res in streak_results for row in res["per_k"])
    if max_gap > 0.05:
        print(f"\nNOTE: empirical and theoretical (iid-Bernoulli) probabilities diverge by up to "
              f"{max_gap * 100:.1f} percentage points across the k values checked - real trade outcomes here "
              f"are NOT well approximated as independent draws (likely correlated market regimes/clustering), "
              f"so trust the empirical bootstrap column over the theoretical one.")
    else:
        print(f"\nEmpirical and theoretical probabilities are close (max gap {max_gap * 100:.1f}pp) - the "
              f"iid-Bernoulli approximation is reasonable for this strategy's loss sequence.")

    print(f"\nCAVEAT: bootstrap resampling of a FINITE historical sample - approximates true future variance, "
          f"not a guarantee. No commission/spread/slippage modeled in the underlying trades.")


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

    sweep_results = risk_sweep(
        all_trades, initial_balance=INITIAL_BALANCE, profit_target_pct=PROFIT_TARGET_PCT,
        max_daily_loss_pct=MAX_DAILY_LOSS_PCT, max_overall_loss_pct=MAX_OVERALL_LOSS_PCT,
        min_trading_days=MIN_TRADING_DAYS, drawdown_mode=DRAWDOWN_MODE,
    )
    print_risk_sweep_table(sweep_results)

    streak_results = losing_streak_probabilities([t["r"] for t in all_trades])
    print_losing_streak_table(streak_results)


if __name__ == "__main__":
    main()
