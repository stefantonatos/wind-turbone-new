# Shared, strategy-agnostic parameter-search + objective-function engine.
#
# WHY THIS EXISTS: research/ict_po3_forex_dukascopy_optimization.py and
# research/day_trading_rauf_dukascopy_optimization.py each independently
# implement the same 4-step methodology (grid search -> Monte Carlo -> cluster
# analysis -> rolling walk-forward), duplicating the grid-search loop and
# always picking the winner by raw total R. This module factors two of the
# ideas from a third-party backtesting video referenced in this project's task
# history out of that duplicated code into one reusable place:
#
#   1. more than one SEARCH STRATEGY over a parameter grid - grid (exhaustive),
#      Bayesian (Optuna/TPE), genetic (hand-rolled) - selectable per script via
#      a SEARCH_METHOD config value, instead of only ever doing a full grid
#      sweep.
#   2. a selectable OBJECTIVE FUNCTION to rank parameter combos by - total R
#      (today's default in both companion scripts, preserved as this module's
#      default too), average R/trade, a trade-based Sharpe-like ratio, Calmar
#      (return over max drawdown), or win rate - instead of always picking the
#      "best" combo by total R alone.
#
# HONEST SCOPE CAVEAT (read this before reaching for bayesian_search or
# genetic_search): both companion scripts' parameter spaces are TINY - 2
# parameters, 16-20 total combos (4x4 or 4x5). At that size, exhaustive grid
# search is not just adequate, it is the BETTER choice: it is deterministic,
# leaves no cell unexplored, and introduces no sampling noise to explain away.
# Bayesian and genetic search exist here because this project's parameter
# spaces will plausibly grow (3+ parameters, or wide continuous-like discrete
# ranges) as more strategies get their own optimization companion scripts, and
# at THAT size grid search's combo count grows multiplicatively while Bayesian
# and genetic search's evaluation budget does not have to. Using either one on
# today's 2-parameter grids is a demonstration of the mechanism, not a genuine
# improvement over grid search there - SEARCH_METHOD defaults to "grid" in
# both companion scripts for exactly this reason, and that default should stay
# "grid" until a script's parameter space actually grows past what exhaustive
# search can comfortably cover.
#
# STRATEGY-AGNOSTIC BY DESIGN: this module never imports from, or otherwise
# depends on, either companion script, Dukascopy, or any strategy's backtest
# logic. Every search function takes a generic `param_grid` (dict of
# param_name -> list of discrete candidate values) and an `eval_fn(params) ->
# list_of_trade_dicts` callback supplied by the CALLER - this module only ever
# calls that callback and scores what comes back. The companion scripts import
# FROM this module; this module imports nothing from them.
#
# THREE ADDITIONS ON TOP OF THE ABOVE (dedicated follow-up pass, grounded in the real quant
# literature rather than taking a trading-education video's claims at face value):
#
#   a. consistency_ratio() - a period-consistency objective/diagnostic INSPIRED BY (not a
#      reimplementation of) the mean/std-of-a-return-stream structure behind Grinold & Kahn's
#      Information Coefficient / ICIR. Real ICIR is a cross-sectional FACTOR-SCORING metric (it
#      measures how consistently a factor's cross-sectional score correlates with forward returns
#      across many assets, period over period) - this project has one strategy's discrete trade
#      list per run, not a cross-section of scored assets, so there is no faithful ICIR to compute
#      here. consistency_ratio() is honestly named for what it actually is: mean(R_p)/std(R_p)
#      across fixed calendar periods (monthly or weekly), INCLUDING zero-trade periods. See its
#      own docstring for the full distinction - read that before assuming this project does
#      cross-sectional factor scoring, because it does not.
#
#   b. estimate_decay() - a signal-decay / half-life diagnostic built directly on this project's
#      EXISTING rolling walk-forward fold structure (both companion scripts' STEP 4), pooling every
#      fold's out-of-sample trades by months-since-that-fold's-own-fit rather than treating each
#      fold's calendar months as independent data points. See its docstring for the exponential-
#      decay fit and the guardrails on when a half-life is actually reportable.
#
#   c. split_lockbox() / lockbox_confirm() - a genuinely one-shot, ledger-enforced final holdout,
#      structurally DIFFERENT FROM (and stricter than) the existing walk-forward OOS folds above.
#      The walk-forward folds are re-touched every time a grid/Bayesian/genetic search re-runs
#      (STEP 1's full-range search and every STEP 4 fold's in-sample search all see data up to
#      fetch_end) - useful for what they're for, but they do NOT satisfy "never touched by any
#      search iteration". The lockbox is carved off BEFORE any search runs and is enforced, via a
#      persistent append-only ledger (not just a comment/convention), to be scored AT MOST ONCE per
#      strategy_id. See split_lockbox()/lockbox_confirm()'s docstrings for why this needs to be
#      ledger-enforced rather than trusted to programmer discipline, and why it deliberately does
#      NOT apply a Bonferroni correction the way the existing zscore()/bonferroni_adjusted_z_
#      threshold() section above does (a one-shot-by-construction test has no multiple-comparisons
#      problem to correct for).

# !pip install --upgrade optuna -q   # uncomment in Colab - only needed for bayesian_search;
# grid_search and genetic_search need nothing beyond numpy (already a dependency of both
# companion scripts). Missing optuna never crashes anything here - see bayesian_search below.

import calendar
import datetime
import itertools
import json
import math
import os
import random
import statistics

import numpy as np


# ============================================================================
# OBJECTIVE FUNCTIONS
# ============================================================================
#
# Every objective function has the signature (trades, years) -> float, where:
#   trades: a list of trade dicts, each with at least an "r" key (a float
#           R-multiple), in chronological order (the order a normal backtest
#           loop naturally produces them in - none of these functions re-sort).
#   years:  the span, in years, of the window the trades were drawn from. Only
#           `sharpe` actually uses this; the others accept and ignore it so
#           every objective can be called through the exact same signature.
#
# All of them return a plain float score that a search strategy MAXIMIZES -
# there is no "lower is better" objective in this registry; if one is ever
# added, either negate its natural sense internally or make that convention
# explicit in its own docstring.

def _r_values(trades):
    return np.asarray([t["r"] for t in trades], dtype=float)


def max_drawdown_r(r_values):
    """Largest peak-to-trough drop in the cumulative R sum along `r_values` in
    the order given (chronological, by convention). This is the SAME
    definition already used by both companion scripts' Monte Carlo code
    (monte_carlo_bootstrap/monte_carlo_shuffle in each): cumsum -> running max
    via np.maximum.accumulate -> (running_max - cumsum).max(). Deliberately
    re-implemented here rather than imported from either script, since this
    module must not depend on them - but the arithmetic is copied exactly, not
    redefined, precisely so `calmar` below means the same thing here as
    "max drawdown" already means in this project's Monte Carlo output.
    Returns 0.0 for an empty or single-trade list (see calmar's docstring for
    why a single trade is a special case, not just "small n")."""
    r = np.asarray(r_values, dtype=float)
    if len(r) == 0:
        return 0.0
    cum = np.cumsum(r)
    running_max = np.maximum.accumulate(cum)
    return float((running_max - cum).max())


def total_r(trades, years=None):
    """Sum of every trade's R-multiple. This is today's existing, already-
    verified default behavior in both companion scripts (the grid winner has
    always been picked by raw total R) - kept as this registry's default
    (OBJECTIVES["total_r"]) specifically so that behavior does not regress."""
    return float(_r_values(trades).sum()) if trades else 0.0


def avg_r(trades, years=None):
    """Mean R-multiple per trade. 0.0 for an empty trade list (no crash, no
    NaN) - an empty cell has no evidence either way, and 0.0 keeps it out of
    an optimizer's way without pretending it's the worst possible outcome."""
    if not trades:
        return 0.0
    return float(_r_values(trades).mean())


# Sentinel for sharpe's degenerate cases (fewer than 2 trades, or zero
# variance in R across every trade). A trade-based Sharpe-like ratio is
# mathematically undefined in both cases (you cannot estimate a standard
# deviation from 0 or 1 sample, and dividing by a std of exactly 0 is a
# division by zero) - rather than let either case raise, or silently produce
# NaN/inf that would then corrupt a `max(..., key=...)` comparison in a search
# strategy, both degenerate cases return this large negative sentinel. That
# specific choice (negative, not 0, not a large positive number) is
# deliberate: a search strategy MAXIMIZES this objective, so a degenerate cell
# - one with too little evidence to compute a meaningful risk-adjusted number
# at all - must never look attractive next to a real, well-evidenced cell,
# including a real cell with a merely mediocre or negative Sharpe. It is
# "we don't trust this number" pushed as far from "pick me" as reasonably
# encodable in a float, not a value that pretends to mean something risk-wise.
SHARPE_DEGENERATE_SENTINEL = -1.0e9


def sharpe(trades, years):
    """A TRADE-BASED Sharpe-like ratio - NOT a classic time-series Sharpe
    ratio. There is no equity curve sampled at fixed time intervals anywhere
    in this project's backtests, only a list of discrete trade R-multiples
    with irregular real-world spacing between them, so the classic
    (mean-return / std-return) * sqrt(periods-per-year) formula is adapted
    here to trade COUNT annualized by the window's span instead of a fixed
    sampling period:

        (mean(r) / std(r)) * sqrt(n_trades / years)

    where std(r) is the SAMPLE standard deviation (ddof=1, i.e. divided by
    n-1, the conventional choice when std is being estimated from a sample
    rather than treated as a known population parameter - consistent with how
    a real Sharpe ratio is normally computed from a finite return sample).

    Degenerate cases (see SHARPE_DEGENERATE_SENTINEL above for why this
    specific value): fewer than 2 trades (std is undefined with 0 or 1
    sample), OR std(r) == 0 (every trade had an identical R-multiple - the
    ratio's denominator is exactly zero, whether that constant R was a win or
    a loss) OR years is None/<=0 (nothing to annualize against). All three
    return SHARPE_DEGENERATE_SENTINEL rather than raising or producing NaN/inf
    - this function never crashes and never returns NaN into a comparison."""
    r = _r_values(trades)
    n = len(r)
    if n < 2 or years is None or years <= 0:
        return SHARPE_DEGENERATE_SENTINEL
    std = float(r.std(ddof=1))
    if std <= 0.0:
        return SHARPE_DEGENERATE_SENTINEL
    return float((r.mean() / std) * np.sqrt(n / years))


# Sentinels for calmar's no-drawdown edge case - see calmar's docstring.
CALMAR_NO_DRAWDOWN_POSITIVE_SENTINEL = 1.0e9
CALMAR_NO_DRAWDOWN_NEGATIVE_SENTINEL = -1.0e9


def calmar(trades, years=None):
    """total_r / max_drawdown_r, where max_drawdown_r is defined exactly as in
    max_drawdown_r() above (matching both companion scripts' existing Monte
    Carlo drawdown definition, not a new one).

    Guarded divide-by-zero: max_drawdown_r() returns exactly 0.0 whenever the
    cumulative R curve never has a single down-tick from its own running peak
    - either because every trade was a win (or breakeven) in sequence, giving
    a genuinely monotonic curve with no real drawdown to divide by, OR because
    there are 0 or 1 trades, where a "peak-to-trough drop" is trivially
    undefined/zero regardless of that one trade's sign (a single trade is
    always exactly at its own running peak by construction - this is a
    distinct edge case from "many trades with zero drawdown", not just a
    smaller version of it). Both trigger this same zero-drawdown branch, so it
    is split by the SIGN of total_r rather than by trade count, which is what
    actually distinguishes "a real, if short, winning run" from "a single (or
    zero) trade for which this ratio is simply not computable":
      - total_r > 0  -> CALMAR_NO_DRAWDOWN_POSITIVE_SENTINEL (as good as this
        ratio can express - real gains, no observed drawdown at all, whether
        that's because of genuine streak-strength or too little data to have
        dipped yet; both cases the search strategy should be free to prefer).
      - total_r < 0  -> CALMAR_NO_DRAWDOWN_NEGATIVE_SENTINEL (this only
        happens for a single losing trade, or an empty/degenerate window where
        max_drawdown_r's own single-point definition doesn't register a
        drawdown even though the outcome was a loss - never let that look like
        a "good, no-drawdown" cell).
      - total_r == 0 (includes the empty-trade-list case) -> 0.0, neutral."""
    r = _r_values(trades)
    total = float(r.sum()) if len(r) else 0.0
    dd = max_drawdown_r(r)
    if dd <= 0.0:
        if total > 0.0:
            return CALMAR_NO_DRAWDOWN_POSITIVE_SENTINEL
        if total < 0.0:
            return CALMAR_NO_DRAWDOWN_NEGATIVE_SENTINEL
        return 0.0
    return total / dd


def win_rate(trades, years=None):
    """Fraction of trades with r > 0 (breakeven, r == 0 exactly, counts as a
    non-win, matching this project's Monte Carlo convention elsewhere of
    treating "<= 0" as the not-a-win side of a threshold). 0.0 for an empty
    trade list.

    *** KNOWN TRAP - READ BEFORE SELECTING THIS OBJECTIVE ***
    Optimizing for win rate ALONE ignores payout size entirely, and this is
    not a hypothetical risk in this project - it is exactly what already
    happened here: the Day Trading Rauf base backtest
    (research/day_trading_rauf_dukascopy_backtest.py) has a real, measured
    51.6% win rate and is STILL net-negative overall, because its losers
    outsize its winners on average (see that script's and its optimization
    companion's headers for the full numbers). A parameter combo that
    maximizes win_rate can trivially do so by taking tiny, high-probability
    wins against large, rare losses and come out net-negative in total R -
    this objective exists in this registry to be SELECTABLE for exactly that
    kind of inspection (e.g. "does the highest-win-rate corner of this grid
    also happen to be profitable, or is it a payout-size trap like Rauf's
    fixed-parameter baseline already was?"), not because it is being quietly
    recommended as a good thing to optimize for on its own. Any script
    offering "win_rate" as an OBJECTIVE option must surface this same caveat
    in its own printed output when that objective is selected - the registry
    docstring alone is not enough if a user never reads this file."""
    if not trades:
        return 0.0
    r = _r_values(trades)
    return float(np.mean(r > 0.0))


# ----------------------------------------------------------------------------
# consistency_ratio - a PERIOD-CONSISTENCY objective, INSPIRED BY (not a reimplementation of)
# Grinold & Kahn's Information Coefficient / ICIR.
#
# *** READ THIS BEFORE ASSUMING THIS PROJECT DOES CROSS-SECTIONAL FACTOR SCORING - IT DOES NOT ***
# The real Information Coefficient (IC) is the cross-sectional rank correlation, in a given period,
# between a factor's SCORES across many assets and those same assets' forward returns; ICIR is
# mean(IC)/std(IC) across periods. That is a statement about how consistently a factor ranks many
# assets relative to each other, period over period. This project has exactly ONE strategy's
# chronological list of discrete closed trades per run - there is no cross-section of scored assets
# anywhere in this codebase, so there is no faithful IC/ICIR to compute here, full stop. Calling
# this function "ICIR" would misrepresent what it does.
#
# What consistency_ratio() ACTUALLY is: mean(R_p) / std(R_p, ddof=1), where R_p is the SUM of a
# strategy's realized trade R-multiples falling in calendar period p (monthly by default, or
# weekly), across EVERY period spanning the trades' own date range - INCLUDING periods with zero
# trades (R_p = 0 for those, not skipped). That last part matters: a strategy that trades in every
# period and nets modestly each time should score HIGHER than one with the identical total R
# concentrated into a few active periods and many silent ones - silence is not neutral here, it
# lowers the period count denominator's effective evidence and (via the zero R_p values it
# contributes) can raise or lower the ratio depending on whether those zeros are more or less
# extreme than the active periods' own spread. This is the one piece of ICIR's STRUCTURE (a
# mean/std ratio of a return stream across periods) this function borrows - not its cross-sectional
# meaning.
def _to_date(d):
    """Normalizes a trade's "date" field (datetime.date OR datetime.datetime, both appear across
    this project's various backtest scripts) down to a plain datetime.date for period bucketing."""
    if isinstance(d, datetime.datetime):
        return d.date()
    return d


def _period_key(d, period):
    """Monthly key: (year, month). Weekly key: the ISO (iso_year, iso_week) of `d`'s Monday - ISO
    week keys are used (not a naive day//7 bucket) specifically so a trade near a year boundary is
    grouped with the correct week even when ISO week 1 of a year starts in the tail end of the
    previous calendar year (or vice versa) - datetime.date.isocalendar() already implements this
    correctly, so it's used directly rather than hand-rolled."""
    if period == "M":
        return (d.year, d.month)
    if period == "W":
        iso = d.isocalendar()
        return (iso[0], iso[1])
    raise ValueError(f"consistency_ratio: unknown period {period!r} - choose 'M' (monthly) or 'W' (weekly)")


def _all_period_keys(min_d, max_d, period):
    """Every period key from `min_d`'s period through `max_d`'s period, INCLUSIVE, in order - the
    full span a set of trades could have occurred across, independent of which periods actually
    have a trade in them. This is what makes zero-trade periods show up in consistency_ratio's R_p
    list at all: this function enumerates the calendar, not the trades."""
    if period == "M":
        keys = []
        y, m = min_d.year, min_d.month
        ey, em = max_d.year, max_d.month
        while (y, m) <= (ey, em):
            keys.append((y, m))
            m += 1
            if m == 13:
                m = 1
                y += 1
        return keys
    if period == "W":
        # Step Monday-to-Monday by exactly 7 days (never hand-rolled ISO week arithmetic, which is
        # easy to get subtly wrong around 52/53-week year boundaries) and read each Monday's own
        # ISO (year, week) via isocalendar() - monotonic and gap-free by construction since every
        # step is exactly one week.
        keys = []
        cur = min_d - datetime.timedelta(days=min_d.weekday())
        end = max_d - datetime.timedelta(days=max_d.weekday())
        while cur <= end:
            iso = cur.isocalendar()
            keys.append((iso[0], iso[1]))
            cur += datetime.timedelta(days=7)
        return keys
    raise ValueError(f"consistency_ratio: unknown period {period!r} - choose 'M' (monthly) or 'W' (weekly)")


def bucket_trades_by_period(trades, period="M"):
    """Sums each trade's "r" into its calendar period bucket (see _period_key) - does NOT fill in
    zero-trade periods (that is consistency_ratio's job, via _all_period_keys, since only
    consistency_ratio knows it needs the full calendar span, not just the occupied buckets).
    Exposed as its own function (not inlined into consistency_ratio) so the period-bucketing logic
    itself is directly unit-testable against hand-crafted trade lists with known R_p values."""
    buckets = {}
    for t in trades:
        key = _period_key(_to_date(t["date"]), period)
        buckets[key] = buckets.get(key, 0.0) + float(t["r"])
    return buckets


def consistency_ratio(trades, years=None, period="M"):
    """mean(R_p) / std(R_p, ddof=1) across every calendar period (monthly by default; period="W"
    for weekly) spanning `trades`' own date range, R_p = sum of that period's trade R-multiples,
    INCLUDING zero-trade periods (R_p = 0, not skipped - see the registry-level docstring above for
    why this matters). `years` is accepted and ignored, purely so this function can be plugged into
    OBJECTIVES and called through search strategies' identical (trades, years) -> ... signature
    alongside total_r/avg_r/sharpe/calmar/win_rate.

    UNLIKE every other function in OBJECTIVES, this returns a DICT, not a bare float - the reason a
    score is missing or low-confidence matters to a caller deciding whether to trust it:
        {"value": float | nan, "n_periods": int, "n_trades": int,
         "confidence": "ok" | "low_confidence" | "insufficient"}

    Guard rails:
      - 0 trades -> value=nan, n_periods=0, confidence="insufficient" (nothing to bucket at all).
      - std(R_p) == 0 (or fewer than 2 periods, where a sample std is undefined) -> value=nan,
        NEVER a silent inf or 0 - a constant (or single-period) R_p stream has no meaningful
        consistency ratio, not a "perfectly consistent" one.
      - n_periods < 12 -> confidence="insufficient" AND value is forced to nan regardless of what
        the raw ratio would have been - the whole point of "insufficient" is "don't even report a
        number as comparable", not "report a number with a footnote".
      - 12 <= n_periods < 24 -> confidence="low_confidence" (a real number, but on a short span -
        use with caution).
      - n_periods >= 24 -> confidence="ok".

    See grid_search/bayesian_search/genetic_search's "best" selection (via this module's internal
    _normalize_score helper) for how a dict-valued score - including a nan value - is handled
    without ever letting a nan/insufficient cell win a max() comparison by accident."""
    n_trades = len(trades)
    if n_trades == 0:
        return {"value": float("nan"), "n_periods": 0, "n_trades": 0, "confidence": "insufficient"}

    dates = [_to_date(t["date"]) for t in trades]
    min_d, max_d = min(dates), max(dates)
    all_keys = _all_period_keys(min_d, max_d, period)
    r_by_period = bucket_trades_by_period(trades, period=period)
    r_values = [r_by_period.get(k, 0.0) for k in all_keys]
    n_periods = len(r_values)

    if n_periods >= 2:
        std_r = statistics.stdev(r_values)   # ddof=1, the same convention used by sharpe()/zscore() above
        mean_r = statistics.mean(r_values)
    else:
        std_r = float("nan")
        mean_r = float(r_values[0]) if r_values else float("nan")

    if math.isnan(std_r) or std_r == 0.0:
        value = float("nan")
    else:
        value = mean_r / std_r

    if n_periods < 12:
        confidence = "insufficient"
        value = float("nan")   # don't report a number as comparable - see docstring above
    elif n_periods < 24:
        confidence = "low_confidence"
    else:
        confidence = "ok"

    return {"value": value, "n_periods": n_periods, "n_trades": n_trades, "confidence": confidence}


OBJECTIVES = {
    "total_r": total_r,     # DEFAULT - matches both companion scripts' existing, already-verified behavior
    "avg_r": avg_r,
    "sharpe": sharpe,
    "calmar": calmar,
    "win_rate": win_rate,   # see the loud trap warning in win_rate()'s own docstring above
    "consistency_ratio": consistency_ratio,   # returns a DICT, not a float - see its own docstring
                                               # and _normalize_score below for how search strategies
                                               # handle that without special-casing it themselves
}

DEFAULT_OBJECTIVE = "total_r"


def get_objective(name):
    """Looks up `name` in OBJECTIVES with a clear error listing valid options
    on a miss, instead of a bare KeyError - both companion scripts resolve
    their OBJECTIVE config string through this, so a typo'd config value fails
    loudly and immediately rather than deep inside a search loop."""
    try:
        return OBJECTIVES[name]
    except KeyError:
        raise KeyError(f"Unknown objective {name!r} - choose one of {sorted(OBJECTIVES)}") from None


def _normalize_score(raw_score):
    """Every OBJECTIVES function except consistency_ratio returns a plain float; consistency_ratio
    returns a dict ({"value", "n_periods", "n_trades", "confidence"}, see its own docstring). All
    three search strategies (grid_search/bayesian_search/genetic_search) need a single float to
    rank cells by max(..., key=...) regardless of which shape the configured objective_fn produced
    - this is the ONE place that difference gets reconciled, so grid_search/bayesian_search/
    genetic_search's own bodies never have to special-case "is this objective's score a dict".

    Returns (sort_value, detail): `sort_value` is always a plain float, safe to store in a result
    entry's "score" key and hand straight to max(key=...) - a dict's "value" is extracted for this
    purpose, and nan (explicitly, e.g. consistency_ratio's "insufficient"/std==0 cases) is mapped to
    -inf so a numerically-undefined or too-little-evidence cell can NEVER accidentally win a
    max() comparison against a real, defined score merely because NaN comparisons are unreliable in
    Python/numpy (nan is neither > nor < anything, including other nans) - it must always lose.
    `detail` is the original dict when raw_score was one (None otherwise) - callers that want the
    full picture (n_periods, confidence, ...) for REPORTING, not ranking, get it back via the
    result entry's "score_detail" key (see grid_search/bayesian_search/genetic_search below), so no
    information from a dict-valued objective is thrown away, only reordered for comparison
    purposes."""
    if isinstance(raw_score, dict):
        value = raw_score.get("value")
        detail = raw_score
    else:
        value = raw_score
        detail = None

    if value is None:
        return float("-inf"), detail
    try:
        is_nan = math.isnan(value)
    except TypeError:
        return float("-inf"), detail
    if is_nan:
        return float("-inf"), detail
    return float(value), detail


# ============================================================================
# SIGNIFICANCE / MULTIPLE-TESTING HELPERS
# ============================================================================
#
# These are REPORTING statistics for a verdict section, not OBJECTIVES a
# search strategy is ever handed as its objective_fn (there is no "zscore" key
# in the OBJECTIVES registry above, and there should not be one - actively
# searching a parameter grid FOR the highest z-score is exactly the kind of
# multiple-testing-blind mining bonferroni_adjusted_z_threshold below exists
# to guard against; z-score belongs downstream of a search, describing
# whatever cell was already selected by total_r/avg_r/sharpe/calmar/win_rate,
# not driving the selection itself).

def zscore(trades):
    """Sample z-score of a list of trade R-multiples: mean(r) / (std(r, ddof=1)
    / sqrt(n)), equivalently mean(r) * sqrt(n) / std(r, ddof=1).

    THIS IS A CORRECTED REPLACEMENT for a shortcut that shipped elsewhere in
    this project (research/ict_po3_forex_dukascopy_backtest.py and
    research/day_trading_rauf_dukascopy_backtest.py both compute
    `z = (total_r / n_trades) / (1 / sqrt(n_trades))`, i.e. `avg_r *
    sqrt(n_trades)` with NO division by the trades' actual sample standard
    deviation at all - equivalent to implicitly assuming std(r) == 1 exactly).
    For a stop/target payout structure clustered around -1R and some positive
    R, the real sample std is typically well above 1 (a simple 50/50 Bernoulli
    at (-1, +2) already has std ~= 1.5), so that shortcut systematically
    INFLATES every z-score computed with it by roughly that same factor. This
    function is the corrected version, used by both optimization companion
    scripts' verdict sections below - it is a straight division by the actual
    std(r, ddof=1), not a stand-in constant.

    Degenerate cases: fewer than 2 trades (std is undefined with 0 or 1
    sample) or std(r, ddof=1) == 0 (every trade had an identical R-multiple -
    the standard error is exactly 0) both return 0.0. Unlike sharpe()'s large
    negative sentinel above, 0.0 (neutral, not "actively bad") is the right
    degenerate value HERE specifically because zscore is never handed to a
    search strategy as an objective_fn to maximize - nothing is ever choosing
    "the highest z-score cell", so there is no risk of a degenerate 0.0
    looking artificially attractive next to a real result the way there would
    be if this were used as a search objective."""
    r = _r_values(trades)
    n = len(r)
    if n < 2:
        return 0.0
    std = float(r.std(ddof=1))
    if std <= 0.0:
        return 0.0
    return float(r.mean() / (std / np.sqrt(n)))


def bonferroni_adjusted_z_threshold(n_trials, family_wise_alpha=0.05):
    """The per-test two-sided |z| bar a single result needs to clear for a
    family of `n_trials` roughly-independent tests to hold an overall
    (family-wise) false-positive rate of `family_wise_alpha` (default 5%) -
    the standard Bonferroni correction, applied to a normal z-score.

    WHY THIS EXISTS: neither optimization companion script's grid search (nor
    any other fixed-rule strategy script in this project) currently corrects
    for the fact that MANY combos are being tested against the same data in
    one run (16-20 grid cells per script here alone, before counting every
    OTHER strategy script's own single-point test) - interpreting each cell's
    z-score against the naive |z| > 1.96 rule of thumb as if it were the only
    test ever run against that data is exactly the setup that produces false
    "this looks real" verdicts by chance alone (see Bailey/Borwein/Zhu/López
    de Prado's deflated Sharpe ratio / probability-of-backtest-overfitting
    literature). This helper gives each script's verdict section an honest,
    adjusted bar to compare against, using THAT RUN's own evaluated-combo
    count as `n_trials` - a defensible per-run lower bound, not a claim to
    have corrected for every strategy this project has ever tested (a fully
    project-wide correction would need a running total across every script's
    every run, which this helper deliberately leaves to the caller to supply
    if they want a stricter, cross-script `n_trials` instead).

    Formula: per_test_alpha = 1 - (1 - family_wise_alpha) ** (1 / n_trials),
    then the two-sided z bar is the standard normal's (1 - per_test_alpha / 2)
    quantile. Uses the Python standard library's statistics.NormalDist (exact,
    no extra dependency - available since Python 3.8) rather than scipy, since
    the standard library already provides this at full precision; scipy's
    scipy.stats.norm.ppf would give the identical value if a caller already
    has scipy loaded, but nothing here requires installing it.

    n_trials=1 must reduce to the ordinary single-test 1.96 bar (verified by
    test_optimization_engine.py) - the whole point of this function is that it
    generalizes, not replaces, the familiar single-test threshold."""
    if n_trials < 1:
        raise ValueError("bonferroni_adjusted_z_threshold: n_trials must be >= 1")
    if not (0.0 < family_wise_alpha < 1.0):
        raise ValueError("bonferroni_adjusted_z_threshold: family_wise_alpha must be in (0, 1)")
    per_test_alpha = 1.0 - (1.0 - family_wise_alpha) ** (1.0 / n_trials)
    return float(statistics.NormalDist().inv_cdf(1.0 - per_test_alpha / 2.0))


# ============================================================================
# SEARCH STRATEGIES
# ============================================================================
#
# All three share the same inputs and the same output shape:
#
#   param_grid: dict of param_name -> list of discrete candidate values,
#               exactly how STOP_BUFFER_PCT_GRID/FALLBACK_REWARD_RISK_GRID
#               etc. are already defined in both companion scripts - just
#               collected into one dict, e.g.
#               {"stop_buffer_pct": [...], "fallback_reward_risk": [...]}.
#   eval_fn:    params_dict -> list_of_trade_dicts. Supplied by the calling
#               script. This is the ONLY point of contact this module has with
#               a strategy's actual backtest logic - it never touches
#               Dukascopy data or backtest rules itself.
#   objective_fn: one of OBJECTIVES[...] (or any callable with the same
#               (trades, years) -> float signature).
#   years:      passed straight through to objective_fn on every evaluation.
#
# Return value (all three): a dict with
#   "method":   "grid" | "bayesian" | "genetic"
#   "all":      list of {"params": dict, "trades": [...], "score": float, "score_detail": dict?} -
#               every DISTINCT combo actually evaluated (grid_search: every
#               combo in param_grid, in exhaustive order; bayesian_search:
#               every trial Optuna ran; genetic_search: every distinct
#               chromosome evaluated across the whole run, deduplicated - see
#               genetic_search's own docstring). "score" is ALWAYS a plain
#               float (via _normalize_score above), even when objective_fn is
#               consistency_ratio (dict-returning) - nan/insufficient-data
#               cells are normalized to -inf so they can never accidentally
#               win a "best" comparison. "score_detail" is present (the
#               objective's original dict) only when objective_fn returned
#               one; omitted entirely for plain-float objectives (total_r,
#               avg_r, sharpe, calmar, win_rate) so their entries' shape is
#               completely unchanged from before this key existed.
#   "best":     the single entry of "all" with the highest score (None if
#               "all" is empty)
#   "n_evals":  len(results["all"]) - how many DISTINCT eval_fn calls this
#               search actually made; this is the number to compare across
#               methods for the efficiency claim (see print_search_comparison)
#
# genetic_search additionally returns "history": the best score seen in each
# generation, in order - a convergence trace, for a print or plot.


def grid_search(param_grid, eval_fn, objective_fn, years=None, show_progress=False, desc="grid search"):
    """Exhaustive itertools.product over every combo in param_grid, evaluated
    in the exact order a hand-written nested for-loop over the same
    param_grid (outer loop = first key, inner loop = last key) would produce
    - this generalizes both companion scripts' hardcoded 2-parameter double
    loops (`[(sb, frr) for sb in SB_GRID for frr in FRR_GRID]`) to any number
    of parameters without changing that iteration order, which is what lets
    grid_search be a drop-in replacement for those loops (see
    test_optimization_engine.py's exact-equivalence test)."""
    names = list(param_grid.keys())
    value_lists = [param_grid[name] for name in names]
    combos = list(itertools.product(*value_lists))

    iterator = combos
    if show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(combos, desc=desc, unit="combo", leave=False)
        except ImportError:
            pass

    all_results = []
    for combo in iterator:
        params = dict(zip(names, combo))
        trades = eval_fn(params)
        raw_score = objective_fn(trades, years)
        score, score_detail = _normalize_score(raw_score)
        entry = {"params": params, "trades": trades, "score": score}
        if score_detail is not None:
            entry["score_detail"] = score_detail
        all_results.append(entry)

    best = max(all_results, key=lambda r: r["score"]) if all_results else None
    return {"method": "grid", "all": all_results, "best": best, "n_evals": len(all_results)}


def bayesian_search(param_grid, eval_fn, objective_fn, years=None, n_trials=20, seed=42, show_progress=False):
    """Optuna (TPE sampler) over the SAME discrete candidate lists param_grid
    already defines - each trial samples every parameter via
    trial.suggest_categorical(name, candidate_values), never a wider
    continuous range. This is a deliberate choice, not a missed opportunity to
    "do real Bayesian optimization": the point of offering this as an
    alternative SEARCH STRATEGY is a fair, apples-to-apples comparison against
    grid_search over the identical space it exhaustively covers (fewer
    evaluations to explore the same candidate set), not a different, larger
    problem than grid_search is solving.

    Wrapped in try/except ImportError - if optuna isn't installed, prints a
    clear skip message and returns None, matching this project's existing
    sklearn/matplotlib-missing handling pattern elsewhere (see e.g.
    ict_po3_forex_dukascopy_optimization.py's sklearn_cluster_analysis). This
    function never crashes a calling script over a missing optional
    dependency - callers that get None back should fall back to grid_search
    (both companion scripts' dispatch helpers do exactly this)."""
    try:
        import optuna
    except ImportError:
        print("optuna not installed - skipping Bayesian search ('!pip install optuna -q' and "
              "re-run for this option). Falling back to grid search is the caller's responsibility "
              "(both companion scripts' SEARCH_METHOD dispatch does this automatically).")
        return None

    optuna.logging.set_verbosity(optuna.logging.WARNING)   # this project's convention is a quiet default,
                                                             # matching the DUKASCRIPT logger suppression elsewhere

    names = list(param_grid.keys())
    all_results = []
    seen = {}   # combo tuple -> index into all_results, so a trial Optuna happens to repeat
                # doesn't get scored/evaluated twice or double-counted in n_evals

    def objective(trial):
        params = {name: trial.suggest_categorical(name, param_grid[name]) for name in names}
        key = tuple(params[name] for name in names)
        if key in seen:
            return all_results[seen[key]]["score"]
        trades = eval_fn(params)
        raw_score = objective_fn(trades, years)
        score, score_detail = _normalize_score(raw_score)   # Optuna needs a plain float back - a
                                                              # dict-valued objective (consistency_ratio)
                                                              # would otherwise break study.optimize()
        entry = {"params": params, "trades": trades, "score": score}
        if score_detail is not None:
            entry["score_detail"] = score_detail
        seen[key] = len(all_results)
        all_results.append(entry)
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)

    best = max(all_results, key=lambda r: r["score"]) if all_results else None
    return {"method": "bayesian", "all": all_results, "best": best, "n_evals": len(all_results)}


def genetic_search(param_grid, eval_fn, objective_fn, years=None, population_size=12, generations=8,
                    mutation_rate=0.15, tournament_size=3, elitism=True, seed=42, show_progress=False):
    """Hand-rolled genetic algorithm - no DEAP or other heavy GA dependency;
    this project prefers lean, well-understood custom code, and this search
    space (a handful of discrete parameters) is small enough that a simple GA
    is easy to implement correctly and easy to unit test directly.

    Chromosome = one integer gene per parameter, each gene an INDEX into that
    parameter's own candidate list (not the value itself) - e.g. for
    param_grid={"stop_buffer_pct": [0.01, 0.02, 0.05, 0.1], "fallback_reward_risk":
    [1.0, 1.5, 2.0, 2.5, 3.0]}, chromosome [2, 4] decodes to
    {"stop_buffer_pct": 0.05, "fallback_reward_risk": 3.0}.

    Per generation: tournament selection (tournament_size random contenders,
    the fittest wins) picks two parents, single-point crossover splices them
    at one random gene boundary, then random-reset mutation independently
    gives each gene a `mutation_rate` chance of being replaced with a fresh
    random index (not nudged - a full reset, so mutation can jump anywhere in
    that parameter's candidate list, not just to a neighboring value).
    Elitism: the current generation's single best chromosome is copied
    UNCHANGED into the next generation (no crossover, no mutation applied to
    it), so the best score seen so far can never regress from one generation
    to the next.

    Evaluation caching: a chromosome (by its exact index tuple) is only ever
    passed to eval_fn once per run - if tournament selection, crossover, or
    elitism produces the same chromosome again later, its score is looked up
    instead of re-evaluated. This is both a real efficiency win (fewer actual
    eval_fn calls for the same genetic search) and the reason "all"/"n_evals"
    in the return value reflect DISTINCT chromosomes evaluated, not
    population_size * generations raw slots.

    Returns the standard {"method", "all", "best", "n_evals"} shape plus
    "history": a list of length `generations`, the best score seen in that
    generation (a convergence trace - print it, or plot it, to show the
    search actually improving over generations rather than just asserting
    that it does)."""
    names = list(param_grid.keys())
    value_lists = [param_grid[name] for name in names]
    sizes = [len(v) for v in value_lists]
    if any(s == 0 for s in sizes):
        raise ValueError("genetic_search: every parameter in param_grid needs at least one candidate value")

    rng = random.Random(seed)
    all_results = []
    cache = {}   # chromosome tuple -> index into all_results

    def chromosome_to_params(chrom):
        return {name: value_lists[i][chrom[i]] for i, name in enumerate(names)}

    def score_chromosome(chrom):
        key = tuple(chrom)
        if key in cache:
            return all_results[cache[key]]["score"]
        params = chromosome_to_params(chrom)
        trades = eval_fn(params)
        raw_score = objective_fn(trades, years)
        score, score_detail = _normalize_score(raw_score)
        entry = {"params": params, "trades": trades, "score": score}
        if score_detail is not None:
            entry["score_detail"] = score_detail
        cache[key] = len(all_results)
        all_results.append(entry)
        return score

    def random_chromosome():
        return [rng.randrange(s) for s in sizes]

    def tournament_select(population, scored):
        contenders = rng.sample(range(len(population)), min(tournament_size, len(population)))
        best_i = max(contenders, key=lambda i: scored[i])
        return population[best_i]

    population = [random_chromosome() for _ in range(population_size)]
    history = []

    gen_iter = range(generations)
    if show_progress:
        try:
            from tqdm.auto import tqdm
            gen_iter = tqdm(gen_iter, desc="genetic search", unit="generation")
        except ImportError:
            pass

    for _ in gen_iter:
        scored = [score_chromosome(c) for c in population]
        best_idx = max(range(len(population)), key=lambda i: scored[i])
        best_chrom = population[best_idx]
        history.append(scored[best_idx])

        next_population = [list(best_chrom)] if elitism else []
        while len(next_population) < population_size:
            parent_a = tournament_select(population, scored)
            parent_b = tournament_select(population, scored)
            if len(names) > 1:
                cut = rng.randrange(1, len(names))
                child = parent_a[:cut] + parent_b[cut:]
            else:
                child = list(parent_a)
            child = [rng.randrange(sizes[i]) if rng.random() < mutation_rate else gene
                     for i, gene in enumerate(child)]
            next_population.append(child)

        population = next_population

    best = max(all_results, key=lambda r: r["score"]) if all_results else None
    return {"method": "genetic", "all": all_results, "best": best, "n_evals": len(all_results), "history": history}


# ============================================================================
# dispatch + comparison helpers
# ============================================================================

SEARCH_METHODS = ("grid", "bayesian", "genetic")


def run_search(method, param_grid, eval_fn, objective_fn, years=None, **kwargs):
    """Dispatches to grid_search/bayesian_search/genetic_search by name -
    shared by both companion scripts' SEARCH_METHOD config so the dispatch
    logic (including optuna's graceful fallback) lives in exactly one place.
    Unknown `method` raises ValueError immediately (loudly, at dispatch time)
    rather than silently doing nothing."""
    if method == "grid":
        return grid_search(param_grid, eval_fn, objective_fn, years=years, **kwargs)
    if method == "bayesian":
        result = bayesian_search(param_grid, eval_fn, objective_fn, years=years, **kwargs)
        if result is None:
            print("run_search: bayesian search unavailable (see message above) - falling back to grid_search "
                  "over the same param_grid so the calling script still gets a usable result.")
            return grid_search(param_grid, eval_fn, objective_fn, years=years)
        return result
    if method == "genetic":
        return genetic_search(param_grid, eval_fn, objective_fn, years=years, **kwargs)
    raise ValueError(f"Unknown SEARCH_METHOD {method!r} - choose one of {SEARCH_METHODS}")


def print_search_comparison(results_by_method, param_names=None):
    """Takes {"grid": grid_search_result, "genetic": genetic_search_result, ...}
    (any subset/superset of SEARCH_METHODS, any dict of labels -> result dicts
    from grid_search/bayesian_search/genetic_search) and prints how many
    evaluations each took and what score/params each found - the same
    "was this the same or did it drift" honesty this project's walk-forward
    step already applies to in-sample vs out-of-sample, applied here to
    search-method-vs-search-method instead."""
    print("\nSearch method comparison:")
    print(f"  {'method':<12}{'n_evals':>10}   {'best score':>12}   best params")
    for label, result in results_by_method.items():
        if result is None or result.get("best") is None:
            print(f"  {label:<12}{'--':>10}   {'no result':>12}")
            continue
        best = result["best"]
        print(f"  {label:<12}{result['n_evals']:>10}   {best['score']:>+12.4f}   {best['params']}")


# ============================================================================
# SIGNAL-DECAY / HALF-LIFE DIAGNOSTIC
# ============================================================================
#
# Built directly on top of the EXISTING rolling walk-forward fold structure both companion scripts
# already have (ict_po3_forex_dukascopy_optimization.py's run_walk_forward/generate_walk_forward_
# folds, day_trading_rauf_dukascopy_optimization.py's run_walk_forward/walk_forward_folds) - NOT a
# generic textbook decay formula bolted on separately from real fold data.
#
# Both companion scripts currently chain every fold's out-of-sample trades into one flat list
# (combined_oos_r / combined_oos_trades), which is exactly right for computing Walk-Forward
# Efficiency but throws away WHERE in each fold's own OOS window a trade fell - i.e. it loses fold
# identity entirely. This diagnostic needs that back: for each fold, how many calendar months past
# that fold's own oos_start did a given OOS trade happen (its "months_since_fit"), pooled ACROSS
# folds so month 1 across all 6 PO3/Rauf folds becomes one bucket, month 2 across all 6 folds
# becomes another, etc. - turning e.g. 6 folds x 12 OOS months into up to 72 pooled data points
# instead of 6 separate 12-point series. A negative trend in the chosen metric as months_since_fit
# increases is evidence the fitted parameters' edge (if any) decays the further out-of-sample they
# get used - exactly the kind of thing "re-fit every year" walk-forward practice already assumes is
# true without directly measuring it.

def _months_since_fit(trade_date, oos_start):
    """1-indexed calendar-month offset of `trade_date` from `oos_start` (both date-like) - a trade
    falling within oos_start's own calendar month is month 1 (not month 0), matching the natural
    "how many months into this fold's OOS window" reading and this project's month 1..12 for a
    1-year (WALK_FORWARD_OOS_YEARS=1 / WF_OOS_YEARS=1) OOS window in both companion scripts."""
    trade_date = _to_date(trade_date)
    oos_start = _to_date(oos_start)
    return (trade_date.year - oos_start.year) * 12 + (trade_date.month - oos_start.month) + 1


def pool_oos_trades_by_month(fold_data):
    """Pools every fold's OOS trades by months-since-THAT-FOLD's-own-oos_start (see
    _months_since_fit), collapsing fold identity in favor of relative position within each fold's
    OOS window - e.g. 6 folds each with a 12-month OOS window pool into at most 12 buckets (month
    1..12), each bucket drawing trades from as many as 6 different folds, rather than 6 separate
    12-point series.

    `fold_data`: a list of per-fold dicts, each needing at least "oos_start" (date-like) and
    "oos_trades" (list of trade dicts with a "date" key) - exactly the shape both companion
    scripts' fold_results/fold_rows entries have once their STEP 4 sections tag each fold with its
    own un-merged OOS trades (see this module's header and both companion scripts' walk-forward
    sections) - folds with no "oos_trades" key or an empty one are simply skipped, not an error (a
    fold that produced zero OOS trades has nothing to pool from it, but doesn't invalidate the
    others).

    Returns (pooled, n_folds_used): `pooled` is {month_index: [trade, ...]}; `n_folds_used` is the
    count of DISTINCT folds that contributed at least one trade to any bucket - this is what
    estimate_decay's `min_folds` gate below checks against, not a total trade count or bucket
    count."""
    pooled = {}
    n_folds_used = 0
    for fold in fold_data:
        trades = fold.get("oos_trades") or []
        if not trades:
            continue
        oos_start = fold["oos_start"]
        n_folds_used += 1
        for t in trades:
            month_idx = _months_since_fit(t["date"], oos_start)
            pooled.setdefault(month_idx, []).append(t)
    return pooled, n_folds_used


def estimate_decay(fold_data, metric="avg_r", min_folds=4):
    """Pools `fold_data`'s OOS trades by months-since-fit (see pool_oos_trades_by_month), computes
    `metric` (an OBJECTIVES key, default "avg_r"; "consistency_ratio" is also accepted - its own
    "value" is used, per-bucket, skipping any bucket where consistency_ratio itself reports
    confidence="insufficient") per pooled bucket, and:

      1. Fits an ordinary least-squares line (metric vs. months_since_fit) via np.polyfit - "slope"
         below. This ALWAYS gets reported (whenever there are >= 3 usable pooled buckets): a
         negative slope is decay evidence, a flat-or-positive slope is not, regardless of whether
         the metric ever goes negative (avg_r realistically can, for a losing strategy/fold).

      2. SEPARATELY attempts a genuine exponential-decay fit metric_t = metric_0 * phi**t (i.e.
         log(metric_t) = log(metric_0) + t*log(phi), fit by ordinary least squares in log-space) -
         "phi" and "half_life_months" = ln(0.5)/ln(phi) below. This is ONLY attempted when every
         pooled bucket's metric value is strictly positive (log is undefined/meaningless
         otherwise) AND the fitted phi lands in the OPEN interval (0, 1) - phi <= 0 or phi >= 1
         means "not genuine decay" (no decay, growth, or a nonsensical fit), so half_life_months
         stays None rather than reporting a number that doesn't mean what "half-life" is supposed
         to mean. A negative/flat "slope" (point 1) with no reportable "phi"/"half_life_months"
         (point 2) is a perfectly normal, self-consistent outcome - not every real decay pattern
         is a clean positive-throughout exponential curve, and this function does not pretend one
         is when it isn't.

    `min_folds` (default 4): pool_oos_trades_by_month's `n_folds_used` must be at least this many
    distinct folds for ANYTHING here to be reported - both companion scripts' 6-fold rolling walk-
    forward (WALK_FORWARD_OOS_YEARS=1 / WF_OOS_YEARS=1, so months_since_fit runs 1..12) comfortably
    clears this; a strategy script with fewer than 4 walk-forward folds does NOT have enough
    independent folds for a pooled-by-month decay estimate to mean anything yet, and this function
    says so explicitly via confidence="insufficient_folds" rather than fitting a line through too
    little independent evidence and reporting it as if it were meaningful.

    Returns: {"slope": float, "half_life_months": float | None, "phi": float | None,
              "n_folds_used": int, "confidence": "ok" | "insufficient_folds"}"""
    pooled, n_folds_used = pool_oos_trades_by_month(fold_data)

    def _insufficient():
        return {"slope": float("nan"), "half_life_months": None, "phi": None,
                "n_folds_used": n_folds_used, "confidence": "insufficient_folds"}

    if n_folds_used < min_folds:
        return _insufficient()

    metric_fn = None
    use_consistency = (metric == "consistency_ratio")
    if not use_consistency:
        metric_fn = OBJECTIVES.get(metric)
        if metric_fn is None:
            raise KeyError(f"estimate_decay: unknown metric {metric!r} - choose one of "
                            f"{sorted(OBJECTIVES)} (or 'consistency_ratio')")

    months = sorted(pooled.keys())
    xs, ys = [], []
    for m in months:
        bucket_trades = pooled[m]
        if use_consistency:
            cr = consistency_ratio(bucket_trades)
            if cr["confidence"] == "insufficient" or math.isnan(cr["value"]):
                continue
            value = cr["value"]
        else:
            value = metric_fn(bucket_trades, None)
        if value is None:
            continue
        try:
            if math.isnan(value):
                continue
        except TypeError:
            continue
        xs.append(float(m))
        ys.append(float(value))

    if len(xs) < 3 or len(set(xs)) < 2:
        # too few usable pooled buckets (or all the same month, degenerate for a line fit) to fit
        # anything meaningful, even though enough distinct FOLDS contributed - e.g. every fold's
        # OOS trades happened to land in the very first calendar month of its own window.
        return _insufficient()

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    slope, _intercept = np.polyfit(xs_arr, ys_arr, 1)

    phi = None
    half_life = None
    if np.all(ys_arr > 0.0):
        log_slope, log_intercept = np.polyfit(xs_arr, np.log(ys_arr), 1)
        candidate_phi = float(np.exp(log_slope))
        if 0.0 < candidate_phi < 1.0:
            phi = candidate_phi
            half_life = float(np.log(0.5) / np.log(phi))

    return {
        "slope": float(slope),
        "half_life_months": half_life,
        "phi": phi,
        "n_folds_used": n_folds_used,
        "confidence": "ok",
    }


# ============================================================================
# LOCKBOX / EMBARGOED FINAL HOLDOUT
# ============================================================================
#
# WHY THIS IS STRUCTURALLY DIFFERENT FROM (STRICTER THAN) THE WALK-FORWARD OOS FOLDS ABOVE:
# every one of the existing walk-forward folds' out-of-sample windows gets RE-TOUCHED on every
# single re-run of a grid/Bayesian/genetic search - both companion scripts' STEP 1 (full-range
# search) already sees data all the way to fetch_end, and every STEP 4 fold's in-sample search sees
# data up to that fold's own is_end, which (across all folds) eventually covers the same range the
# OOS folds draw from too. Re-running the whole pipeline with a tweaked grid, a different
# SEARCH_METHOD, or a different OBJECTIVE touches that same data again. That is completely fine for
# what walk-forward folds are FOR (an honest apples-to-apples check of whether an in-sample winner
# transfers forward) - but it means the OOS folds do NOT satisfy a genuine "never touched by any
# search iteration, ever" property, because they get re-touched every time the search itself reruns.
#
# The lockbox is different by construction: split_lockbox() carves the FINAL `lockbox_months` off
# the fetched range, BEFORE any search (grid, Bayesian, genetic, or any walk-forward fold) ever
# runs, and every walk-forward-fold-generating call site in both companion scripts is updated to
# take `search_end` (== the lockbox's own start) as its upper bound instead of `fetch_end` - so the
# lockbox window is excluded from every fold BY CONSTRUCTION, not by a comment saying "don't touch
# this". lockbox_confirm() then enforces, via a persistent append-only ledger on disk (not just a
# docstring convention a future run could forget), that a given strategy_id's lockbox window is
# scored AT MOST ONCE, ever. That one-shot property is the entire point of a lockbox: a holdout
# that gets checked twice (even by the same "final" set of parameters, "just to be sure") stops
# being a true holdout the second time, because now the parameters have effectively been chosen
# with knowledge of how they perform there - the classic multiple-comparisons trap this project's
# own bonferroni_adjusted_z_threshold() exists to guard against elsewhere, reintroduced through the
# back door if the lockbox itself were re-run. A ledger enforces this mechanically because
# programmer discipline/memory ("I'll only run this once, I promise") is exactly the kind of thing
# that quietly fails across re-runs, different notebook sessions, or a different person picking up
# the same script later - the ledger doesn't need to be trusted to remember, it just checks a file.
#
# NOT ANOTHER PLACE TO APPLY BONFERRONI: bonferroni_adjusted_z_threshold() above exists because
# MANY grid cells get tested against the same data in one run. The lockbox is the opposite
# situation by construction - it runs exactly once per strategy_id, ever, enforced by the ledger -
# so there is no multiple-comparisons problem here to correct for. lockbox_confirm()'s pass/fail
# rule is deliberately simple and pre-registered (positive OOS avg R/trade, plus a non-negative
# consistency_ratio value when there's enough data for that to mean anything) rather than yet
# another significance test.

class LockboxAlreadyUsedError(Exception):
    """Raised by lockbox_confirm() when `strategy_id` already has ANY prior recorded attempt in the
    ledger - see this section's header above for why this refuses outright rather than silently
    re-confirming. Catching this and "just running it again anyway" defeats the entire purpose of
    having a lockbox in the first place."""


DEFAULT_LOCKBOX_LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lockbox_ledger.json")


def _shift_months(dt, months):
    """Shifts a date/datetime `dt` by a whole number of calendar `months` (positive = forward,
    negative = back), clamping the day-of-month to the target month's own last valid day when the
    original day doesn't exist there (e.g. Jan 31 - 1 month -> Dec 31, but Mar 31 - 1 month ->
    Feb 28 or 29, since February never has a 31st) - the same defensive rollover
    calendar.monthrange()-based approach as this project's other month-arithmetic helpers (see
    both companion scripts' _month_chunks). Works for both datetime.date and datetime.datetime
    (only .year/.month/.day and .replace(...) are used, both of which either type supports)."""
    total_months = dt.year * 12 + (dt.month - 1) + months
    year, month0 = divmod(total_months, 12)
    month = month0 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def split_lockbox(fetch_start, fetch_end, lockbox_months=12):
    """Carves the final `lockbox_months` off the end of [fetch_start, fetch_end) as a never-
    touched-until-the-very-end final holdout, returning (search_start, search_end, lockbox_start,
    lockbox_end) with search_end == lockbox_start EXACTLY (the two windows are adjacent and non-
    overlapping by construction: [search_start, search_end) + [lockbox_start, lockbox_end) ==
    [fetch_start, fetch_end), split at one point).

    `fetch_start`/`fetch_end` may be datetime.date OR datetime.datetime (whichever a caller's own
    FETCH_START/FETCH_END already are) - the return values are the same type as `fetch_end` (via
    _shift_months's .replace(...)-based arithmetic), so a caller doesn't need to convert types to
    keep using its existing FETCH_START/FETCH_END-shaped code.

    search_start is always exactly fetch_start - the lockbox only ever comes off the END of the
    range, never the beginning (this project's walk-forward folds are already rolling FORWARD in
    time, so the most recent data is both the most realistic OOS test AND the correct place to
    carve a final holdout from - carving from the start would leave the lockbox as the OLDEST data,
    which is backwards for a "how does this perform on data that came after everything the search
    ever saw" check).

    Raises ValueError if `lockbox_months` would consume the entire range (or more) - a lockbox
    that eats the whole fetch window leaves nothing for the search itself to use, which is not a
    valid split, not a valid (if degenerate) lockbox."""
    if lockbox_months <= 0:
        raise ValueError(f"split_lockbox: lockbox_months must be positive, got {lockbox_months!r}")
    lockbox_start = _shift_months(fetch_end, -lockbox_months)
    if lockbox_start <= fetch_start:
        raise ValueError(
            f"split_lockbox: lockbox_months={lockbox_months} consumes the entire "
            f"[{fetch_start}, {fetch_end}) range (or more) - shrink lockbox_months or widen the "
            "fetch range so the search side has something left to search over.")
    return fetch_start, lockbox_start, lockbox_start, fetch_end


def _read_ledger(ledger_path):
    """Loads the lockbox ledger (a JSON list of attempt records) from disk - an empty list if the
    file doesn't exist yet (first-ever lockbox attempt anywhere) or is present but empty."""
    if not os.path.exists(ledger_path):
        return []
    with open(ledger_path, "r") as f:
        content = f.read().strip()
    if not content:
        return []
    return json.loads(content)


def _write_ledger(ledger_path, records):
    """Overwrites the ledger file with the full `records` list (append-only from the CALLER's
    perspective - lockbox_confirm always reads the existing records first and appends to them,
    never truncates or edits a prior record - but the actual disk write is a plain overwrite of
    the whole file, which is simpler and less failure-prone than a true append-only file format for
    a ledger this small). default=str handles date/datetime values in a record without requiring
    every caller to pre-serialize them."""
    parent = os.path.dirname(os.path.abspath(ledger_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(ledger_path, "w") as f:
        json.dump(records, f, indent=2, default=str)


def lockbox_confirm(strategy_id, final_params, backtest_fn, lockbox_start, lockbox_end,
                     ledger_path=DEFAULT_LOCKBOX_LEDGER_PATH):
    """Runs `backtest_fn(lockbox_start, lockbox_end)` (expected to return a list of trade dicts,
    the same shape every eval_fn in this module already produces) on the lockbox window EXACTLY
    ONCE for a given `strategy_id`, ever - see this section's header above for why that one-shot
    property is the entire point of a lockbox and why it's enforced via a persistent ledger rather
    than trusted to convention.

    Before running anything, checks `ledger_path` (a JSON list of {"strategy_id", "params",
    "timestamp", "passed", ...} records, append-only from the caller's perspective - see
    _read_ledger/_write_ledger) for ANY prior record with this exact `strategy_id`. If one exists,
    raises LockboxAlreadyUsedError immediately - `backtest_fn` is never even called - rather than
    silently re-confirming. This is a hard refusal, not a warning: catching the exception and
    calling lockbox_confirm again anyway defeats the entire purpose (see header).

    After a successful run (pass OR fail), appends a new record to the ledger regardless of outcome
    - a FAILED lockbox attempt is just as much a "this strategy_id's lockbox is now used up" event
    as a passed one; the ledger tracks ATTEMPTS, not just passes.

    Pass/fail rule (deliberately simple and pre-registered - see header for why this is NOT another
    place to apply a Bonferroni-style correction): PASS requires BOTH
      1. positive OOS avg R/trade on the lockbox window, AND
      2. EITHER there weren't enough lockbox trades for consistency_ratio to report anything
         comparable (confidence == "insufficient" - in that case this criterion is simply not
         applied, rather than letting an under-powered consistency check veto an otherwise-positive
         result it has no real evidence against), OR consistency_ratio's value is >= 0 when it DOES
         have enough data to report one.

    Returns {"passed": bool, "total_r": float, "avg_r": float, "n_trades": int,
             "consistency": dict} - the same summary shape recorded in the ledger's `params`-
    adjacent fields (see the ledger record itself, on disk, for the full attempt history)."""
    records = _read_ledger(ledger_path)
    prior = [r for r in records if r.get("strategy_id") == strategy_id]
    if prior:
        raise LockboxAlreadyUsedError(
            f"lockbox_confirm: strategy_id={strategy_id!r} already has {len(prior)} recorded "
            f"lockbox attempt(s) in {ledger_path!r} (first attempt at "
            f"{prior[0].get('timestamp', '?')}, passed={prior[0].get('passed', '?')}) - refusing "
            "to run the lockbox a second time for this strategy_id. A lockbox that gets checked "
            "more than once stops being a genuine holdout the second time - see this module's "
            "LOCKBOX / EMBARGOED FINAL HOLDOUT section header for why this is a hard refusal, not "
            "a warning.")

    trades = backtest_fn(lockbox_start, lockbox_end)
    n_trades = len(trades)
    total_r_value = float(sum(t["r"] for t in trades)) if trades else 0.0
    avg_r_value = (total_r_value / n_trades) if n_trades else 0.0
    consistency = consistency_ratio(trades)

    positive_avg = avg_r_value > 0.0
    if consistency["confidence"] == "insufficient":
        consistency_ok = True   # not enough lockbox trades to judge consistency either way - don't
                                 # let an under-powered check veto a result it has no real evidence
                                 # against (see docstring above)
    else:
        consistency_value = consistency["value"]
        consistency_ok = (not math.isnan(consistency_value)) and consistency_value >= 0.0
    passed = bool(positive_avg and consistency_ok)

    record = {
        "strategy_id": strategy_id,
        "params": final_params,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "passed": passed,
        "total_r": total_r_value,
        "avg_r": avg_r_value,
        "n_trades": n_trades,
        "consistency": consistency,
        "lockbox_start": str(lockbox_start),
        "lockbox_end": str(lockbox_end),
    }
    records.append(record)
    _write_ledger(ledger_path, records)

    return {"passed": passed, "total_r": total_r_value, "avg_r": avg_r_value, "n_trades": n_trades,
            "consistency": consistency}
