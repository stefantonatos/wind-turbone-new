# Prop firm challenge simulation on the support/resistance zone-bounce
# strategy - the strongest raw result in this project so far (verified via
# support_resistance_zone_bounce_dukascopy_backtest.py: both split-period
# halves independently significant, edge survives realistic cost scenarios
# up to ~0.03-0.05% round-trip). Same idea as orb_indices_prop_firm_
# challenge.py: combine the validated strategy's real trade history with
# the already-built/unit-tested prop-firm challenge simulator
# (prop_firm_challenge_simulator.py's simulate_challenge/monte_carlo_pass_
# rate, reused here verbatim) to estimate PASS/FAIL rate and, for passing
# attempts, how many trading days it actually takes.
#
# Strategy logic (zone detection, entry/exit, fixed 2:1 fallback R:R) is
# identical to support_resistance_zone_bounce_dukascopy_backtest.py's
# backtest_instrument - verified byte-identical via AST comparison before
# shipping, aside from adding "instrument" to each trade dict (not needed
# for the challenge simulator itself, just useful for debugging).
#
# Runs over the full available history (2010-2025) so the Monte Carlo
# estimate has a large real trade sample to draw from, same reasoning as
# the ORB version.

# !pip install --upgrade dukascopy-python -q   # uncomment this line in Colab

import datetime
import os
import random as _random

import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

INSTRUMENTS = [
    ("EURUSD", dki.INSTRUMENT_FX_MAJORS_EUR_USD),
    ("GBPUSD", dki.INSTRUMENT_FX_MAJORS_GBP_USD),
    ("USDJPY", dki.INSTRUMENT_FX_MAJORS_USD_JPY),
    ("USDCHF", dki.INSTRUMENT_FX_MAJORS_USD_CHF),
    ("USDCAD", dki.INSTRUMENT_FX_MAJORS_USD_CAD),
    ("AUDUSD", dki.INSTRUMENT_FX_MAJORS_AUD_USD),
    ("NZDUSD", dki.INSTRUMENT_FX_MAJORS_NZD_USD),
    ("SP500", dki.INSTRUMENT_IDX_AMERICA_E_SANDP_500),
    ("NASDAQ100", dki.INSTRUMENT_IDX_AMERICA_E_NQ_100),
    ("DOWJONES", dki.INSTRUMENT_IDX_AMERICA_E_D_J_IND),
    ("DAX", dki.INSTRUMENT_IDX_EUROPE_E_DAAX),
    ("FTSE100", dki.INSTRUMENT_IDX_EUROPE_E_FUTSEE_100),
    ("NIKKEI225", dki.INSTRUMENT_IDX_ASIA_E_N225JAP),
]

FETCH_START = datetime.datetime(2010, 1, 1)
FETCH_END = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_DAY_1
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

# --- the validated S/R zone bounce strategy, unchanged from support_resistance_zone_bounce_dukascopy_backtest.py ---
ATR_LEN = 14
ZONE_PIVOT_LOOKBACK = 5
ZONE_WIDTH_ATR_MULT = 0.25
ZONE_BREAK_BUFFER_ATR_MULT = 0.1
MAX_TARGET_DISTANCE_ATR_MULT = 20.0
FALLBACK_REWARD_RISK = 2.0
MAX_HOLD_BARS = 60

# --- prop firm challenge rules (same defaults as prop_firm_challenge_simulator.py / orb_indices_prop_firm_challenge.py) ---
INITIAL_BALANCE = 10000.0
RISK_PCT_PER_TRADE = 1.0
PROFIT_TARGET_PCT = 8.0
MAX_DAILY_LOSS_PCT = 5.0
MAX_OVERALL_LOSS_PCT = 10.0
MIN_TRADING_DAYS = 4
DRAWDOWN_MODE = "static"
N_MONTE_CARLO_RUNS = 300


def fetch_daily_ohlc(instrument_const):
    df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, FETCH_START, FETCH_END)
    if df.empty:
        return None
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


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


def find_confirmed_pivots(highs, lows, lookback):
    n = len(highs)
    swing_highs, swing_lows = {}, {}
    for i in range(lookback, n - lookback):
        window_highs = highs[i - lookback:i + lookback + 1]
        if highs[i] == max(window_highs) and window_highs.count(highs[i]) == 1:
            swing_highs[i + lookback] = (i, highs[i])
        window_lows = lows[i - lookback:i + lookback + 1]
        if lows[i] == min(window_lows) and window_lows.count(lows[i]) == 1:
            swing_lows[i + lookback] = (i, lows[i])
    return swing_highs, swing_lows


def backtest_instrument(label, df):
    highs, lows, closes = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
    times = df.index
    n = len(closes)
    atr = compute_atr_series(highs, lows, closes, ATR_LEN)

    swing_highs_by_confirm_idx, swing_lows_by_confirm_idx = find_confirmed_pivots(highs, lows, ZONE_PIVOT_LOOKBACK)

    active_zones = []
    trades = []
    open_trade = None
    i = 0
    while i < n:
        current_atr = atr[i]

        if i in swing_highs_by_confirm_idx and current_atr:
            _, price = swing_highs_by_confirm_idx[i]
            half_width = ZONE_WIDTH_ATR_MULT * current_atr
            active_zones.append({"top": price + half_width, "bottom": price - half_width,
                                  "kind": "resistance", "in_trade": False})
        if i in swing_lows_by_confirm_idx and current_atr:
            _, price = swing_lows_by_confirm_idx[i]
            half_width = ZONE_WIDTH_ATR_MULT * current_atr
            active_zones.append({"top": price + half_width, "bottom": price - half_width,
                                  "kind": "support", "in_trade": False})

        if current_atr:
            break_buffer = ZONE_BREAK_BUFFER_ATR_MULT * current_atr
            still_active = []
            for z in active_zones:
                broken = (closes[i] < z["bottom"] - break_buffer if z["kind"] == "support"
                          else closes[i] > z["top"] + break_buffer)
                if not broken:
                    still_active.append(z)
            active_zones = still_active

        if open_trade is not None:
            hi, lo = highs[i], lows[i]
            side, stop, target = open_trade["side"], open_trade["stop"], open_trade["target"]
            hit_stop = lo <= stop if side == "LONG" else hi >= stop
            hit_target = hi >= target if side == "LONG" else lo <= target
            bars_held = i - open_trade["entry_index"]
            if hit_stop:
                open_trade["outcome"], open_trade["exit_r"] = "SL", -1.0
            elif hit_target:
                open_trade["outcome"], open_trade["exit_r"] = "TP", open_trade["reward_risk"]
            elif bars_held >= MAX_HOLD_BARS:
                pnl = (closes[i] - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - closes[i])
                open_trade["outcome"], open_trade["exit_r"] = "FLAT", pnl / open_trade["sl_distance"]

            if open_trade["outcome"] is not None:
                trades.append({"index": label, "side": side, "outcome": open_trade["outcome"],
                                "r": open_trade["exit_r"], "date": times[i].date()})
                for z in active_zones:
                    if z is open_trade["zone"]:
                        z["in_trade"] = False
                open_trade = None
                i += 1
                continue
            i += 1
            continue

        if not current_atr:
            i += 1
            continue

        for z in active_zones:
            if z["in_trade"]:
                continue
            if z["kind"] == "support" and lows[i] <= z["top"] and closes[i] >= z["bottom"]:
                side = "LONG"
            elif z["kind"] == "resistance" and highs[i] >= z["bottom"] and closes[i] <= z["top"]:
                side = "SHORT"
            else:
                continue

            entry_price = closes[i]
            buffer_price = ZONE_BREAK_BUFFER_ATR_MULT * current_atr
            if side == "LONG":
                stop = z["bottom"] - buffer_price
                sl_distance = entry_price - stop
            else:
                stop = z["top"] + buffer_price
                sl_distance = stop - entry_price
            if sl_distance <= 0:
                continue

            opposite_kind = "resistance" if side == "LONG" else "support"
            candidates = [oz for oz in active_zones if oz["kind"] == opposite_kind and oz is not z and
                          ((oz["bottom"] > entry_price) if side == "LONG" else (oz["top"] < entry_price))]
            max_dist = MAX_TARGET_DISTANCE_ATR_MULT * current_atr
            if side == "LONG":
                candidates = [oz for oz in candidates if oz["bottom"] - entry_price <= max_dist]
            else:
                candidates = [oz for oz in candidates if entry_price - oz["top"] <= max_dist]

            if candidates:
                nearest = (min(candidates, key=lambda oz: oz["bottom"]) if side == "LONG"
                           else max(candidates, key=lambda oz: oz["top"]))
                target = nearest["bottom"] if side == "LONG" else nearest["top"]
                reward_risk = abs(target - entry_price) / sl_distance
            else:
                reward_risk = FALLBACK_REWARD_RISK
                target = entry_price + sl_distance * reward_risk if side == "LONG" else entry_price - sl_distance * reward_risk

            z["in_trade"] = True
            open_trade = {"side": side, "entry": entry_price, "stop": stop, "target": target,
                          "sl_distance": sl_distance, "reward_risk": reward_risk, "entry_index": i,
                          "zone": z, "outcome": None, "exit_r": None}
            break

        i += 1

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


def _load_prop_firm_simulator():
    """The risk-per-trade sweep and losing-streak-probability diagnostic live in
    prop_firm_challenge_simulator.py (the generic script) since that logic is
    strategy-agnostic - it only needs a list of real R-multiples, not this script's
    zone-bounce-specific backtest machinery. Loaded via spec_from_file_location (same
    pattern already used elsewhere in this project, e.g. day_trading_rauf_dukascopy_
    optimization.py importing day_trading_rauf_dukascopy_backtest.py) rather than a
    package import, since this directory has no __init__.py and these scripts are
    designed to also run standalone in Colab."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prop_firm_challenge_simulator.py")
    spec = importlib.util.spec_from_file_location("prop_firm_challenge_simulator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    years = (FETCH_END - FETCH_START).days / 365
    print(f"Using the validated S/R zone bounce strategy (fixed 2:1 fallback R:R, nearest-zone target).")
    print(f"Pulls {len(INSTRUMENTS)} instruments (daily bars) over ~{years:.0f} years - expect a few minutes.\n")

    all_trades = []
    for label, instrument_const in INSTRUMENTS:
        print(f"{label}...", end=" ")
        try:
            df = fetch_daily_ohlc(instrument_const)
        except Exception as exc:
            print(f"failed ({exc})")
            continue
        if df is None or len(df) < 100:
            print("no/insufficient data")
            continue
        trades = backtest_instrument(label, df)
        all_trades.extend(trades)
        print(f"{len(df)} daily bars, {len(trades)} trades")

    if not all_trades:
        print("No trades at all - check output above.")
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
          f"No commission/spread/slippage modeled (the underlying strategy's edge is real but modest - "
          f"see support_resistance_zone_bounce_dukascopy_backtest.py's cost-sensitivity check - so real "
          f"costs would meaningfully change this pass rate, not just round it). Treat this as a rough "
          f"real-world estimate, not exact.")

    sim = _load_prop_firm_simulator()
    sweep_results = sim.risk_sweep(
        all_trades, initial_balance=INITIAL_BALANCE, profit_target_pct=PROFIT_TARGET_PCT,
        max_daily_loss_pct=MAX_DAILY_LOSS_PCT, max_overall_loss_pct=MAX_OVERALL_LOSS_PCT,
        min_trading_days=MIN_TRADING_DAYS, drawdown_mode=DRAWDOWN_MODE,
    )
    sim.print_risk_sweep_table(sweep_results,
                                header="RISK-PER-TRADE SWEEP - S/R zone bounce (strongest raw result)")

    streak_results = sim.losing_streak_probabilities([t["r"] for t in all_trades])
    sim.print_losing_streak_table(streak_results,
                                   header="LOSING-STREAK PROBABILITY - S/R zone bounce (sizing-independent)")


if __name__ == "__main__":
    main()
