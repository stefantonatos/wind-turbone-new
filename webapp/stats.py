# Pure functions over an already-computed trade list - no fetching, no backtesting.
# Used both right after a fresh run and when filters are toggled on the results page
# (recomputed instantly client-side, never re-triggers a fetch) and when re-viewing an
# older run from the History page.

import datetime
import math
import statistics
from collections import defaultdict

# Below this many trades, a strategy's return figure is mostly noise - not enough signal to
# rank against other strategies or crown as "best" of anything. Used to GATE ranking/highlight
# UI (Compare All leaderboard, Gallery sort-by-return), not just caption around - a strategy
# under this floor should never win a head-to-head comparison on point estimate alone, however
# good that estimate looks. 100 trades is a common rule-of-thumb floor for a binomial-ish
# win/loss series to start being distinguishable from noise; still thin, but a real floor is
# better than none.
MIN_TRADES_FOR_RANKING = 100


# --- typical retail trading cost, applied by DEFAULT everywhere this app shows results -----
# Every backtest in this project runs with NO commission/spread/slippage modeled (see each
# research/*.py script's own header caveat) - each script's own "COST SENSITIVITY" section only
# ever showed a few illustrative what-if scenarios, never actually applied to the headline
# numbers. This is a webapp-layer, post-hoc deduction using that exact same formula
# (cost_adjusted_r = r - (cost_pct / 100) / stop_pct, where stop_pct = sl_distance / entry,
# already recorded on most trades project-wide since the cost-sensitivity rollout) - applied to
# every metric, chart, and leaderboard by default, not just an optional report line.
#
# SOURCING (same discipline as prop_firm_presets.py - real, checkable, gaps flagged honestly,
# not guessed): figures below are TYPICAL RETAIL STANDARD-ACCOUNT round-trip spreads, order-of-
# magnitude from public broker-comparison sources (checked August 2026: EURUSD ~0.6-1.0 pip,
# GBPUSD ~0.6-1.5 pip, USDJPY ~0.1-0.7 pip typical across tested standard accounts; XAUUSD
# ~20-35 "pip"/$0.20-0.35 typical standard-account spread), deliberately rounded toward the
# WIDER/more conservative end of each range - those sources mostly test best-in-class/ECN
# conditions, and understating cost is the more dangerous error for a tool people might trade
# real money on. These are NOT live, NOT broker-specific, and exclude commission (many ECN
# accounts charge a separate per-lot fee on top of a tighter spread) and slippage entirely.
# Re-verify against your actual broker before relying on this for a real decision.
DEFAULT_COST_PCT = 0.03   # instruments not in the table below - matches the middle scenario
                           # every research script's own COST_PCT_SCENARIOS already prints
TYPICAL_COST_PCT_BY_INSTRUMENT = {
    "EURUSD": 0.01,    # ~0.8-1.0 pip typical retail standard-account spread at ~1.08
    "GBPUSD": 0.015,   # ~1.2-1.5 pip at ~1.27
    "USDJPY": 0.01,    # ~0.5-0.7 pip at ~150 (some brokers tighter; kept conservative)
    "XAUUSD": 0.02,    # ~20-30 "pip" ($0.20-0.30) standard-account gold spread at ~$2,600
}


def cost_pct_for_instrument(instrument):
    return TYPICAL_COST_PCT_BY_INSTRUMENT.get(instrument, DEFAULT_COST_PCT)


def apply_cost_adjustment(trades):
    """Returns (adjusted_trades, n_unadjusted). Each trade's 'r' is reduced by its instrument's
    typical round-trip cost (see TYPICAL_COST_PCT_BY_INSTRUMENT/DEFAULT_COST_PCT above),
    converted into R-terms via stop_pct - the exact formula every research script's own COST
    SENSITIVITY section already uses. A trade with no stop_pct (a handful of scripts don't
    record it) is left unadjusted rather than guessed at - n_unadjusted lets the caller be
    honest about partial coverage instead of silently mixing adjusted and unadjusted trades."""
    out = []
    n_unadjusted = 0
    for t in trades:
        t = dict(t)
        stop_pct = t.get("stop_pct")
        r = t.get("r")
        if r is not None and stop_pct:
            cost_pct = cost_pct_for_instrument(t.get("instrument"))
            t["r"] = r - (cost_pct / 100.0) / stop_pct
        else:
            n_unadjusted += 1
        out.append(t)
    return out, n_unadjusted


def scale_trades_r(trades, factor):
    """Returns a shallow-copied trade list with every 'r' multiplied by `factor` - used to
    convert R-multiples into a %-of-account view (factor = the chosen risk-per-trade %,
    e.g. r=+2.0 at 1% risk/trade = +2.0% account return) without duplicating compute_stats/
    equity_curve/per_instrument_breakdown logic for a second unit. z-score is scale-invariant
    (mean and std both scale by the same factor, so their ratio doesn't change) - no special
    handling needed there regardless of which unit is displayed."""
    out = []
    for t in trades:
        t = dict(t)
        if "r" in t and t["r"] is not None:
            t["r"] = t["r"] * factor
        out.append(t)
    return out


def max_drawdown(trades):
    """Peak-to-trough max decline in cumulative R, ordered the SAME way equity_curve() orders
    trades (chronological when every trade has a date, else the order they were produced in) -
    so this number always matches what the equity curve chart actually shows. Returns a
    non-negative R value (0.0 for an equity curve that never dips below its own running peak)."""
    has_dates = all(t.get("date") for t in trades)
    ordered = sorted(trades, key=lambda t: str(t.get("date"))) if has_dates else list(trades)
    cum = 0.0
    peak = 0.0
    worst = 0.0
    for t in ordered:
        cum += t.get("r", 0.0) or 0.0
        peak = max(peak, cum)
        worst = max(worst, peak - cum)
    return worst


def compute_stats(trades):
    n = len(trades)
    if n == 0:
        return None
    r_values = [t.get("r", 0.0) or 0.0 for t in trades]
    total_r = sum(r_values)
    avg_r = total_r / n
    tp = sum(1 for t in trades if t.get("outcome") == "TP")
    sl = sum(1 for t in trades if t.get("outcome") == "SL")
    flat = sum(1 for t in trades if t.get("outcome") == "FLAT")
    # Corrected formula, matching the project-wide fix in commit 85aca19: the old
    # z = avg_r / (1/sqrt(n)) implicitly assumed the R-multiple distribution has
    # stddev exactly 1, which is false and inflates every significance claim. Uses
    # the real sample stddev instead, same as every research/*.py script now does.
    if n >= 2:
        std_r = statistics.stdev(r_values)
        z = (avg_r / (std_r / math.sqrt(n))) if std_r > 0 else 0.0
    else:
        z = 0.0
    # 95% confidence interval on avg/total R - normal approximation (z=1.96 * standard error),
    # same construction as the z-score above, not a bootstrap (more correct for a skewed
    # R-multiple distribution but heavier to recompute on every rerun/filter change - this is
    # explicitly an approximation, same honesty standard as the z-score already carries). A
    # point estimate alone overstates certainty, especially on the small samples common here -
    # this interval is what actually widens or narrows with sample size, and callers should
    # show it alongside the point estimate, not bury it.
    if n >= 2 and std_r > 0:
        se_avg = std_r / math.sqrt(n)
        avg_r_ci_low, avg_r_ci_high = avg_r - 1.96 * se_avg, avg_r + 1.96 * se_avg
    else:
        avg_r_ci_low = avg_r_ci_high = avg_r
    return {
        "n_trades": n,
        "total_r": total_r,
        "avg_r": avg_r,
        "tp": tp, "sl": sl, "flat": flat,
        "tp_pct": tp / n * 100, "sl_pct": sl / n * 100, "flat_pct": flat / n * 100,
        "z_score": z,
        "max_drawdown_r": max_drawdown(trades),
        "avg_r_ci_low": avg_r_ci_low, "avg_r_ci_high": avg_r_ci_high,
        "total_r_ci_low": avg_r_ci_low * n, "total_r_ci_high": avg_r_ci_high * n,
    }


def per_instrument_breakdown(trades):
    buckets = defaultdict(list)
    for t in trades:
        buckets[t.get("instrument", "?")].append(t)
    rows = []
    for instrument, ts in buckets.items():
        s = compute_stats(ts)
        if s is None:
            continue
        rows.append({
            "instrument": instrument,
            "trades": s["n_trades"],
            "total_r": round(s["total_r"], 3),
            "avg_r": round(s["avg_r"], 4),
            "win_pct": round(s["tp_pct"], 1),
        })
    rows.sort(key=lambda r: -r["total_r"])
    return rows


def equity_curve(trades):
    """Returns (x_labels, cumulative_r, chronological). Sorts by date when every
    trade has one; otherwise falls back to the order the trades were produced in
    (per-instrument backtest sequence, then concatenated) and reports
    chronological=False so the caller can caption that distinction."""
    has_dates = all(t.get("date") for t in trades)
    ordered = sorted(trades, key=lambda t: str(t.get("date"))) if has_dates else list(trades)
    cum = 0.0
    xs, ys = [], []
    for i, t in enumerate(ordered):
        cum += t.get("r", 0.0) or 0.0
        xs.append(i + 1)
        ys.append(cum)
    return xs, ys, has_dates


def dollar_equity_curve(trades, risk_pct, starting_balance=10000.0):
    """Same ordering/x-axis convention as equity_curve(), but in account DOLLARS starting from
    `starting_balance`. Takes RAW (unscaled) R-multiple trades - each trade's % account return
    is r * risk_pct (same convention as scale_trades_r), so cumulative % return after the first
    N trades is risk_pct * (cumulative R), added additively onto the starting balance. Returns
    (x_labels, equity_dollars, chronological)."""
    xs, cum_r, chronological = equity_curve(trades)
    equity = [starting_balance * (1 + (r * risk_pct) / 100.0) for r in cum_r]
    return xs, equity, chronological


def daily_pnl(trades):
    """Groups trades by calendar date, summing 'r' per day. Returns {date: total_r} for every
    date that had at least one trade (dates with zero trades are simply absent - the caller
    decides how to render the gap, e.g. a blank calendar cell vs. an explicit 0). Trades with
    no date are skipped entirely (nothing calendar-shaped to plot for sequence-only trades,
    same has_dates convention as equity_curve())."""
    out = {}
    for t in trades:
        d = t.get("date")
        if d is None or not isinstance(d, datetime.date):
            continue
        out[d] = out.get(d, 0.0) + (t.get("r", 0.0) or 0.0)
    return out


def normalize_trade_dates(trades):
    """Makes every trade dict's 'date' a real datetime.date object, whether it just
    came straight out of a live backtest run (already a date) or was reloaded from
    run_history's JSON storage (an ISO string, since date objects aren't JSON-safe)."""
    out = []
    for t in trades:
        t = dict(t)
        d = t.get("date")
        if d is not None and not isinstance(d, datetime.date):
            try:
                t["date"] = datetime.date.fromisoformat(str(d)[:10])
            except ValueError:
                t["date"] = None
        out.append(t)
    return out


def trades_to_jsonable(trades):
    """Inverse direction - date/datetime values -> ISO strings, for run_history's
    JSON-lines/JSON storage."""
    out = []
    for t in trades:
        row = {}
        for k, v in t.items():
            if isinstance(v, (datetime.date, datetime.datetime)):
                row[k] = v.isoformat()
            else:
                row[k] = v
        out.append(row)
    return out
