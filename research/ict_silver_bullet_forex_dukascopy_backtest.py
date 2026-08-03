# ICT "Silver Bullet" backtest on the instruments ICT concepts were actually
# designed for (EURUSD, GBPUSD, USDJPY, gold) via Dukascopy, 2016-2025,
# using the same walk-forward in-sample/out-of-sample discipline that
# validated the ORB-on-indices result elsewhere in this project.
#
# Why this and not full Order Blocks / Power of Three: Silver Bullet is the
# one ICT sub-strategy with genuinely mechanical rules (fixed 1-hour NY-time
# windows, not "read higher-timeframe bias" style discretion), so it's the
# only one that can be coded objectively rather than hand-waved.
#
# Every "72% win rate" / "61% win rate, 2.17 profit factor" number found
# researching this traces back to blog posts or short YouTube-length samples
# with no disclosed out-of-sample split - i.e. exactly the overfitting
# pattern this whole project exists to catch. This script does not trust
# any of those numbers; it re-derives everything from real multi-year data.
#
# Rules implemented, staged per the ICT sequence (per day, per one of the
# three Silver Bullet windows, all times NY):
#   1. LIQUIDITY SWEEP: price wicks below/above the recent (LIQUIDITY_
#      LOOKBACK_BARS) swing low/high, then closes back inside it.
#   2. MARKET STRUCTURE SHIFT: within MSS_SEARCH_BARS bars after the sweep,
#      a close breaks back above/below the local swing high/low that formed
#      in the STRUCTURE_LOOKBACK_BARS bars leading into the sweep.
#   3. FAIR VALUE GAP: within FVG_SEARCH_BARS bars after the MSS, a 3-candle
#      imbalance forms in the trade direction.
#   4. ENTRY: a limit order at the FVG midpoint ("consequent encroachment"),
#      filled if price retraces into the zone within ENTRY_GRACE_BARS.
#   5. Stop beyond the sweep's wick extreme, fixed R:R target (grid-searched
#      the same way ORB's RANGE_MINUTES/REWARD_RISK were - kept to ONE
#      searched parameter to keep the search space small, per this
#      project's standing overfitting guard).
#
# One attempt per window per day per instrument - if any stage fails
# (no MSS, no FVG, FVG never retested), that window is skipped, no retry.

# !pip install --upgrade dukascopy-python -q   # uncomment this line in Colab

import datetime

import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

INSTRUMENTS = [
    ("EURUSD", dki.INSTRUMENT_FX_MAJORS_EUR_USD),
    ("GBPUSD", dki.INSTRUMENT_FX_MAJORS_GBP_USD),
    ("USDJPY", dki.INSTRUMENT_FX_MAJORS_USD_JPY),
    ("XAUUSD", dki.INSTRUMENT_FX_METALS_XAU_USD),
]

FETCH_START = datetime.datetime(2016, 1, 1)
FETCH_END = datetime.datetime(2025, 1, 1)
SPLIT_DATE = datetime.date(2024, 1, 1)   # in-sample 2016-2023, out-of-sample 2024 - same convention as orb_indices_optimization_and_ml.py
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_MIN_5
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

# --- the three official ICT Silver Bullet windows, NY time, same for every instrument ---
WINDOWS = [
    ("LONDON", pd.Timestamp("03:00").time(), pd.Timestamp("04:00").time()),
    ("NY_AM", pd.Timestamp("10:00").time(), pd.Timestamp("11:00").time()),
    ("NY_PM", pd.Timestamp("14:00").time(), pd.Timestamp("15:00").time()),
]

# --- fixed structural parameters (NOT grid-searched - keeps the search space small) ---
LIQUIDITY_LOOKBACK_BARS = 24   # 2 hours at 5-min - defines the swing high/low that must be swept
STRUCTURE_LOOKBACK_BARS = 6    # 30 min - local swing used for the market-structure-shift check
MSS_SEARCH_BARS = 12           # 1 hour to find the structure break after a sweep
FVG_SEARCH_BARS = 6            # 30 min to find a fair value gap after the structure shift
ENTRY_GRACE_BARS = 12          # 1 hour to wait for price to retrace into the FVG
MAX_HOLD_BARS = 288            # 24 hours max hold once filled, then FLAT at last close
STOP_BUFFER_PCT = 0.02         # small buffer beyond the sweep's wick, same scale as ORB's ENTRY_BUFFER_PCT

# --- grid search space (kept to one parameter, per this project's overfitting guard) ---
REWARD_RISK_GRID = [1.5, 2.0, 3.0]

# Illustrative round-trip cost scenarios, as a percentage of entry price - NOT measured real spread
# data, just a few bracketing assumptions to see how much cost this edge can absorb before it
# disappears, same convention introduced in support_resistance_zone_bounce_dukascopy_backtest.py.
COST_PCT_SCENARIOS = [0.0, 0.01, 0.03, 0.05]


def to_ny_time(index):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert("America/New_York")


def fetch_instrument_data(instrument_const):
    df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, FETCH_START, FETCH_END)
    if df.empty:
        return None
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df.index = to_ny_time(df.index)
    return df


def find_window_bars(times, day, win_start, win_end):
    """Returns (start_idx, end_idx) of bars within [win_start, win_end) on
    the given day, searching only the small slice of the index around that
    day (not the whole series) - callers pass in date-matching bars already
    sliced by day_indices for performance."""
    start_idx = end_idx = None
    for idx in day:
        tod = times[idx].time()
        if win_start <= tod < win_end:
            if start_idx is None:
                start_idx = idx
            end_idx = idx
    return start_idx, end_idx


def find_silver_bullet_trade(highs, lows, closes, times, window_start_idx, window_end_idx, n, reward_risk):
    """Runs the full sweep -> MSS -> FVG -> entry state machine for one
    window instance. Returns a trade dict or None if any stage fails."""
    if window_start_idx is None or window_start_idx < LIQUIDITY_LOOKBACK_BARS:
        return None

    swing_low_ref = min(lows[window_start_idx - LIQUIDITY_LOOKBACK_BARS:window_start_idx])
    swing_high_ref = max(highs[window_start_idx - LIQUIDITY_LOOKBACK_BARS:window_start_idx])

    # --- stage 1: liquidity sweep ---
    i_sweep, sweep_side = None, None
    for i in range(window_start_idx, window_end_idx + 1):
        if lows[i] < swing_low_ref and closes[i] > swing_low_ref:
            i_sweep, sweep_side = i, "LONG"
            break
        if highs[i] > swing_high_ref and closes[i] < swing_high_ref:
            i_sweep, sweep_side = i, "SHORT"
            break
    if i_sweep is None:
        return None

    sweep_extreme = lows[i_sweep] if sweep_side == "LONG" else highs[i_sweep]

    # --- stage 2: market structure shift ---
    if i_sweep < STRUCTURE_LOOKBACK_BARS:
        return None
    structure_window = range(i_sweep - STRUCTURE_LOOKBACK_BARS, i_sweep)
    if sweep_side == "LONG":
        structure_ref = max(highs[j] for j in structure_window)
    else:
        structure_ref = min(lows[j] for j in structure_window)

    j_mss = None
    for j in range(i_sweep + 1, min(i_sweep + MSS_SEARCH_BARS, n - 1) + 1):
        if sweep_side == "LONG" and closes[j] > structure_ref:
            j_mss = j
            break
        if sweep_side == "SHORT" and closes[j] < structure_ref:
            j_mss = j
            break
    if j_mss is None:
        return None

    # --- stage 3: fair value gap ---
    fvg_top = fvg_bottom = None
    for k in range(max(j_mss, 2), min(j_mss + FVG_SEARCH_BARS, n - 1) + 1):
        if sweep_side == "LONG" and highs[k - 2] < lows[k]:
            fvg_bottom, fvg_top = highs[k - 2], lows[k]
            break
        if sweep_side == "SHORT" and lows[k - 2] > highs[k]:
            fvg_top, fvg_bottom = lows[k - 2], highs[k]
            break
    if fvg_top is None:
        return None

    entry_price = (fvg_top + fvg_bottom) / 2

    # --- stage 4: entry fill (limit order at the FVG midpoint) ---
    m_fill = None
    search_start = max(j_mss, 2) + 1
    for m in range(search_start, min(search_start + ENTRY_GRACE_BARS, n - 1) + 1):
        if lows[m] <= entry_price <= highs[m]:
            m_fill = m
            break
    if m_fill is None:
        return None

    # --- stage 5: manage the trade forward ---
    buffer_price = (STOP_BUFFER_PCT / 100.0) * entry_price
    if sweep_side == "LONG":
        stop = sweep_extreme - buffer_price
        sl_distance = entry_price - stop
        target = entry_price + sl_distance * reward_risk
    else:
        stop = sweep_extreme + buffer_price
        sl_distance = stop - entry_price
        target = entry_price - sl_distance * reward_risk

    if sl_distance <= 0:
        return None

    outcome, exit_r = None, None
    p = m_fill + 1
    end_p = min(m_fill + MAX_HOLD_BARS, n - 1)
    while p <= end_p:
        hi, lo = highs[p], lows[p]
        hit_stop = lo <= stop if sweep_side == "LONG" else hi >= stop
        hit_target = hi >= target if sweep_side == "LONG" else lo <= target
        if hit_stop:
            outcome, exit_r = "SL", -1.0
            break
        if hit_target:
            outcome, exit_r = "TP", reward_risk
            break
        p += 1
    else:
        p = end_p

    if outcome is None:
        last_close = closes[p]
        pnl = (last_close - entry_price) if sweep_side == "LONG" else (entry_price - last_close)
        outcome, exit_r = "FLAT", pnl / sl_distance

    return {"side": sweep_side, "outcome": outcome, "r": exit_r, "date": times[window_start_idx].date(),
            "stop_pct": sl_distance / entry_price}


def precompute_days(df):
    """Groups bar indices by calendar date ONCE per instrument - computing
    times[idx].date() per bar is the expensive step (pandas Timestamp
    overhead), so this must not be redone per grid-search combination.
    Reused across every REWARD_RISK value and both the in-sample/
    out-of-sample splits, same pattern as orb_indices_optimization_and_ml.py's
    precompute_indicators()."""
    highs, lows, closes = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
    times = df.index
    n = len(closes)

    days_index = {}
    for idx in range(n):
        days_index.setdefault(times[idx].date(), []).append(idx)

    return {"highs": highs, "lows": lows, "closes": closes, "times": times, "n": n, "days_index": days_index}


def run_backtest(ind, reward_risk, split_before=None, split_after=None):
    highs, lows, closes, times, n = ind["highs"], ind["lows"], ind["closes"], ind["times"], ind["n"]

    trades = []
    for day, day_bars in ind["days_index"].items():
        if split_before is not None and day >= split_before:
            continue
        if split_after is not None and day < split_after:
            continue
        for window_label, win_start, win_end in WINDOWS:
            start_idx, end_idx = find_window_bars(times, day_bars, win_start, win_end)
            if start_idx is None:
                continue
            trade = find_silver_bullet_trade(highs, lows, closes, times, start_idx, end_idx, n, reward_risk)
            if trade is not None:
                trade["window"] = window_label
                trades.append(trade)

    return trades


def main():
    years = (FETCH_END - FETCH_START).days / 365
    print(f"Downloading {len(INSTRUMENTS)} instruments from Dukascopy over ~{years:.0f} years "
          f"({FETCH_START.date()} to {FETCH_END.date()}) - expect roughly 15-25 minutes "
          f"(most of it the grid search's plain-Python per-bar loop, not the download).\n")

    data = {}
    for label, instrument_const in INSTRUMENTS:
        print(f"{label}...", end=" ")
        try:
            df = fetch_instrument_data(instrument_const)
        except Exception as exc:
            print(f"failed ({exc})")
            continue
        if df is None:
            print("no data")
            continue
        data[label] = precompute_days(df)
        print(f"{len(df)} bars")

    if not data:
        print("No data downloaded - check output above.")
        return

    print("\n" + "=" * 70)
    print(f"GRID SEARCH on IN-SAMPLE ({FETCH_START.date()} to {SPLIT_DATE}), "
          f"validated OUT-OF-SAMPLE ({SPLIT_DATE} to {FETCH_END.date()})")
    print("=" * 70)

    grid_results = []
    for reward_risk in REWARD_RISK_GRID:
        total_r, n_trades = 0.0, 0
        for label, ind in data.items():
            trades = run_backtest(ind, reward_risk, split_before=SPLIT_DATE)
            total_r += sum(t["r"] for t in trades)
            n_trades += len(trades)
        grid_results.append({"reward_risk": reward_risk, "in_sample_total_r": total_r, "in_sample_trades": n_trades,
                              "in_sample_avg_r": total_r / n_trades if n_trades else 0.0})
        print(f"  REWARD_RISK={reward_risk:.1f}  -> {n_trades:4d} trades, {total_r:+8.2f}R in-sample")

    grid_df = pd.DataFrame(grid_results).sort_values("in_sample_total_r", ascending=False)
    best = grid_df.iloc[0]
    best_reward_risk = float(best["reward_risk"])
    print(f"\nBest in-sample: REWARD_RISK={best_reward_risk} -> {best['in_sample_total_r']:+.2f}R "
          f"over {best['in_sample_trades']:.0f} trades")

    all_oos_trades = []
    for label, ind in data.items():
        trades = run_backtest(ind, best_reward_risk, split_after=SPLIT_DATE)
        for t in trades:
            t["instrument"] = label
        all_oos_trades.extend(trades)

    total_r_oos = sum(t["r"] for t in all_oos_trades)
    n_trades_oos = len(all_oos_trades)

    print(f"\nSAME combination, OUT-OF-SAMPLE ({SPLIT_DATE} to {FETCH_END.date()}):")
    if n_trades_oos:
        print(f"  {n_trades_oos} trades, {total_r_oos:+.2f}R, {total_r_oos/n_trades_oos:+.4f}R/trade")
    else:
        print("  0 trades")
        print("\nNo out-of-sample trades at all - the setup basically never completed all 5 stages "
              "(sweep -> MSS -> FVG -> retest) inside these narrow 1-hour windows on this data.")
        return

    per_instrument = {}
    for t in all_oos_trades:
        per_instrument.setdefault(t["instrument"], []).append(t["r"])
    print("\nPer instrument (out-of-sample):")
    for label, rs in per_instrument.items():
        print(f"  {label}: {len(rs)} trades, {sum(rs):+.2f}R")

    per_window = {}
    for t in all_oos_trades:
        per_window.setdefault(t["window"], []).append(t["r"])
    print("\nPer window (out-of-sample):")
    for label, rs in per_window.items():
        print(f"  {label}: {len(rs)} trades, {sum(rs):+.2f}R")

    tp = sum(1 for t in all_oos_trades if t["outcome"] == "TP")
    sl = sum(1 for t in all_oos_trades if t["outcome"] == "SL")
    flat = sum(1 for t in all_oos_trades if t["outcome"] == "FLAT")
    print(f"\nOutcome breakdown: TP {tp} ({tp/n_trades_oos*100:.1f}%)  SL {sl} ({sl/n_trades_oos*100:.1f}%)  "
          f"FLAT {flat} ({flat/n_trades_oos*100:.1f}%)")

    in_sample_avg = best["in_sample_avg_r"]
    oos_avg = total_r_oos / n_trades_oos
    print(f"\nIn-sample avg R/trade: {in_sample_avg:+.4f}   Out-of-sample avg R/trade: {oos_avg:+.4f}")
    if oos_avg < in_sample_avg * 0.5:
        print("Out-of-sample is much weaker than in-sample - this is what overfitting looks like, "
              "not a strategy worth trading on this evidence.")

    if n_trades_oos < 100:
        print(f"\nCAVEAT: only {n_trades_oos} out-of-sample trades - too few to distinguish real edge "
              f"from noise with confidence (rule of thumb elsewhere in this project has been 100-200+ "
              f"out-of-sample trades before trusting a result). Treat this as a first look, not a verdict.")

    print(f"\nCOST SENSITIVITY (out-of-sample trades only, illustrative round-trip spread scenarios, "
          f"NOT measured real spread data):")
    for cost_pct in COST_PCT_SCENARIOS:
        cost_adjusted_total = sum(t["r"] - (cost_pct / 100.0) / t["stop_pct"] for t in all_oos_trades)
        print(f"  {cost_pct:.2f}% round-trip cost: {cost_adjusted_total:+.2f}R total, "
              f"{cost_adjusted_total/n_trades_oos:+.4f}R/trade")
    print(f"  If the total goes negative well before 0.05%, this edge is too thin to survive real "
          f"execution costs - check your actual broker's spread on each instrument against these numbers.")

    print("\nNo commission/spread/slippage modeled. Entry is a simulated resting limit order at the FVG "
          "midpoint - real fills would be worse (requoting, partial fills, the level not being reached "
          "before the window/grace period expires in live conditions).")


if __name__ == "__main__":
    main()
