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

# Fraction of a Compare-All comparison period held out for out-of-sample ranking (see
# split_trades_for_holdout). Compare All doesn't fit any parameters - it just runs each
# strategy's own already-fixed rules - so this isn't guarding against classic parameter
# overfitting. What it IS guarding against: comparing 16 strategies on the exact same window and
# crowning whichever one happens to look best is itself a form of data snooping (strategy-
# selection bias, not parameter-selection bias) - the "winner" might just be the luckiest
# strategy on that specific window, not the best one. Ranking by a HELD-OUT slice the "winner"
# was never chosen using is the same discipline every research/*.py script's own SPLIT_DATE
# in-sample/out-of-sample convention already uses, applied one level up. A fixed, undebatable
# constant, not a user-adjustable slider - making it tunable would reopen exactly the p-hacking
# risk this feature exists to close.
HOLDOUT_FRACTION = 0.25


def split_trades_for_holdout(trades, holdout_fraction=HOLDOUT_FRACTION):
    """Splits trades into (fit_trades, holdout_trades, split_is_date_based) - the LAST
    `holdout_fraction` of the period is held out. Splits by calendar DATE when every trade has
    one (the threshold is computed from the trades' own min/max date span, not the caller's
    original fetch window, so it's exact regardless of how much of a warmup period actually
    produced trades) - falls back to a POSITIONAL split (last fraction of the list, in whatever
    order the caller passed them - typically a runner's own per-instrument-then-concatenated
    sequence) for strategies whose trades don't carry dates at all (e.g. ORB indices). Neither
    half is cost-adjusted here - that's the caller's job, same as every other trade list this
    module hands back."""
    if not trades:
        return [], [], True
    has_dates = all(t.get("date") for t in trades)
    if has_dates:
        dates = sorted(t["date"] for t in trades)
        min_d, max_d = dates[0], dates[-1]
        span_days = (max_d - min_d).days
        threshold = min_d + datetime.timedelta(days=round(span_days * (1 - holdout_fraction)))
        fit = [t for t in trades if t["date"] < threshold]
        holdout = [t for t in trades if t["date"] >= threshold]
        return fit, holdout, True
    n = len(trades)
    split_idx = round(n * (1 - holdout_fraction))
    return trades[:split_idx], trades[split_idx:], False


# --- typical PROP FIRM trading cost, applied by DEFAULT everywhere this app shows results ---
# Every backtest in this project runs with NO commission/spread/slippage modeled (see each
# research/*.py script's own header caveat) - each script's own "COST SENSITIVITY" section only
# ever showed a few illustrative what-if scenarios, never actually applied to the headline
# numbers. This is a webapp-layer, post-hoc deduction using that exact same formula
# (cost_adjusted_r = r - (cost_pct / 100) / stop_pct, where stop_pct = sl_distance / entry,
# already recorded on most trades project-wide since the cost-sensitivity rollout) - applied to
# every metric, chart, and leaderboard by default, not just an optional report line.
#
# WHY PROP FIRM COSTS, NOT RETAIL: this app's own Prop Firm Fit section (render_prop_firm_fit_
# section) and prop_firm_presets.py already frame the whole tool around "would this pass a real
# funded-account evaluation" - the cost model below now matches that framing instead of a generic
# retail account nobody using this tool for that purpose would actually be trading on.
#
# SOURCING (same discipline as prop_firm_presets.py - real, checkable, gaps flagged honestly, not
# guessed - researched via web search, August 2026): the three firms prop_firm_presets.py already
# models (FTMO, FundedNext, The5ers) all run RAW/ECN-style pricing - a near-zero base spread plus
# a separate per-lot $ commission - genuinely different from a retail "standard account"'s single
# wider all-in spread with no separate commission. Figures below are each firm's own round-trip
# cost (spread + commission, converted to $ per 1.0 standard lot - 100,000 units FX, 100 oz gold -
# then to %-of-reference-price, same reference prices the old retail table used: EURUSD ~1.08,
# GBPUSD ~1.27, USDJPY ~150, XAUUSD ~2,600), AVERAGED across the three firms, not any one firm's
# exact number - a real trader would be on one specific firm/account type, not an average of three.
#   EURUSD: FTMO ~1-3 pip spread via its approved liquidity providers (own commission unclear from
#     sources checked) ~$20-24/lot; FundedNext ~0.0-0.2 pip + $5/lot ~$6-7/lot; The5ers ~0.0 pip +
#     $4/lot ~$4/lot. Average ~$11.5/lot -> ~0.011%.
#   GBPUSD: FTMO 0.5 pip + $3/lot ~$8/lot (fxverify.com's FTMO-specific spread comparison);
#     FundedNext/The5ers assumed similar to their own EURUSD structure (no GBP-specific figures
#     found) ~$6.5/$4/lot. Average ~$6.2/lot -> ~0.005%.
#   USDJPY: FTMO 0.4 pip + $3/lot ~$5.7/lot; FundedNext/The5ers not specifically found, assumed
#     similar to their EURUSD structure ~$5/$4/lot. Average ~$4.9/lot -> ~0.005%.
#   XAUUSD: FTMO ~$0.15-0.30/oz spread (own commission not found) ~$22-28/lot; FundedNext ~$0.10-
#     0.25/oz raw spread + $7/lot ~$24.5/lot; The5ers ~$0.10/oz + $4/lot (its own generic "Forex
#     and Gold carry a $4 commission" note) ~$14/lot. Average ~$22/lot -> ~0.009%.
# GENUINE GAPS, flagged rather than papered over: FTMO's own per-instrument commission (Normal vs.
# Swing account types charge differently) and FundedNext/The5ers' GBPUSD/USDJPY-specific figures
# weren't confirmed from the sources checked - those cells above lean on the same firm's EURUSD
# structure as the closest available estimate, not a confirmed number for that exact pair. Prop
# firm pricing also varies by account type/promotion and changes over time - re-verify against
# your actual firm's current conditions page before relying on this for a real decision, same
# caveat this project's cost figures have always carried.
DEFAULT_COST_PCT = 0.008   # instruments not in the table below - roughly the average of the four
                            # researched instruments' own prop-firm cost below, not a fifth
                            # independently-sourced number
TYPICAL_COST_PCT_BY_INSTRUMENT = {
    "EURUSD": 0.011,   # ~$11.5/lot average (FTMO/FundedNext/The5ers) at ~1.08
    "GBPUSD": 0.005,   # ~$6.2/lot average at ~1.27
    "USDJPY": 0.005,   # ~$4.9/lot average at ~150
    "XAUUSD": 0.009,   # ~$22/lot average at ~$2,600
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


def compounded_return_pct(trades, risk_pct):
    """The account's TRUE final return, compounding trade by trade (growth *= 1 + r*risk_pct/100),
    not the naive sum-then-scale ("total_r * risk_pct") used elsewhere for the raw R-multiple
    metric. That naive version implicitly assumes every trade risks a fixed DOLLAR amount off the
    STARTING balance forever, which silently produces impossible numbers once losses accumulate
    (a -10,000% "return", or a literally negative account balance) - risking risk_pct% of the
    CURRENT balance (the whole point of quoting risk as a percentage) means a loss can approach
    but never cross -100%, exactly what compounding naturally enforces here without an explicit
    cap. Order matters for a REAL equity curve (see dollar_equity_curve) but not for this single
    final number - multiplication commutes, so trades are compounded in whatever order they're
    given. Returns a % (e.g. -97.3 for a 97.3% loss, always > -100)."""
    growth = 1.0
    for t in trades:
        r = t.get("r", 0.0) or 0.0
        growth *= max(1.0 + (r * risk_pct) / 100.0, 0.0)
    return (growth - 1.0) * 100.0


def compounded_return_ci(avg_r_ci_low, avg_r_ci_high, risk_pct, n):
    """95% CI on the compounded total return, extending compute_stats' existing avg_r_ci_low/high
    (a per-trade bound) out to n trades the same way total_r_ci_low/high already does for the
    additive metric (treating every trade as if it earned exactly the bound's average R) - just
    compounded instead of multiplied by n, for the same reason compounded_return_pct exists."""
    def compound(avg_r_bound):
        return (max(1.0 + (avg_r_bound * risk_pct) / 100.0, 0.0) ** n - 1.0) * 100.0
    return compound(avg_r_ci_low), compound(avg_r_ci_high)


def compounded_max_drawdown_pct(trades, risk_pct):
    """Max peak-to-trough decline on the COMPOUNDED equity curve (see compounded_return_pct's
    own docstring for why compounding, not addition, is used here) - a real account's drawdown
    is bounded to [0, 100]% by construction (compounded equity can approach but never cross
    zero), unlike max_drawdown()'s additive R-based version, which has no such bound once scaled
    into a % context (confirmed showing e.g. "-3157%" right next to a correctly-capped
    "-100.00%" total return on the same run - the same underlying bug, just a second call site).
    Ordered the same way dollar_equity_curve is. Returns a % in [0, 100]."""
    has_dates = all(t.get("date") for t in trades)
    ordered = sorted(trades, key=lambda t: str(t.get("date"))) if has_dates else list(trades)
    equity = 1.0
    peak = 1.0
    worst_dd_pct = 0.0
    for t in ordered:
        r = t.get("r", 0.0) or 0.0
        equity = max(equity * (1 + (r * risk_pct) / 100.0), 0.0)
        peak = max(peak, equity)
        if peak > 0:
            worst_dd_pct = max(worst_dd_pct, (peak - equity) / peak * 100.0)
    return worst_dd_pct


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
    """Same ordering convention as equity_curve() (chronological when every trade has a date,
    else backtest-sequence order), but in account DOLLARS starting from `starting_balance`.
    Takes RAW (unscaled) R-multiple trades - each trade's own % account return is r * risk_pct
    (same convention as scale_trades_r), COMPOUNDED trade by trade (balance *= 1 + r*risk_pct/100)
    rather than added additively onto the starting balance. The additive version this replaced
    could show a literally negative dollar balance once losses accumulated past -100% of the
    starting amount (confirmed happening on a real, cost-adjusted, large-sample run) - risking a
    % of the CURRENT balance each trade (what "risk_pct% per trade" is supposed to mean) means
    the curve can approach but never cross zero, which is what actually happens to a real account.
    Returns (x_labels, equity_dollars, chronological)."""
    has_dates = all(t.get("date") for t in trades)
    ordered = sorted(trades, key=lambda t: str(t.get("date"))) if has_dates else list(trades)
    xs, equity = [], []
    balance = starting_balance
    for i, t in enumerate(ordered):
        r = t.get("r", 0.0) or 0.0
        balance = max(balance * (1 + (r * risk_pct) / 100.0), 0.0)
        xs.append(i + 1)
        equity.append(balance)
    return xs, equity, has_dates


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
