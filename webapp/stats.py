# Pure functions over an already-computed trade list - no fetching, no backtesting.
# Used both right after a fresh run and when filters are toggled on the results page
# (recomputed instantly client-side, never re-triggers a fetch) and when re-viewing an
# older run from the History page.

import datetime
import math
import statistics
from collections import defaultdict


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
    return {
        "n_trades": n,
        "total_r": total_r,
        "avg_r": avg_r,
        "tp": tp, "sl": sl, "flat": flat,
        "tp_pct": tp / n * 100, "sl_pct": sl / n * 100, "flat_pct": flat / n * 100,
        "z_score": z,
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
