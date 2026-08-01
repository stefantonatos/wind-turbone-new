# Regime-switching (Hidden Markov Model) Monte Carlo stress test - strategy-
# agnostic, reusable across every backtest script in this project.
#
# !pip install --upgrade dukascopy-python hmmlearn -q   # uncomment in Colab
#
# DISTINCT FROM, NOT A REPLACEMENT FOR, THIS PROJECT'S EXISTING TRADE-LEVEL
# MONTE CARLO (research/optimization_engine.py's monte_carlo_bootstrap /
# monte_carlo_shuffle, research/prop_firm_challenge_simulator.py's
# bootstrap_resample_r_matrix): those resample the ALREADY-REALIZED trade
# R-multiples a real backtest produced - they answer "given this fixed set of
# trades that already happened, how much does trade order/selection affect
# drawdown and total return". That is a question about ONE fixed history.
#
# This module answers a different, more rigorous question: "if the
# underlying market's regime dynamics (volatility clustering, trending vs.
# ranging periods, crisis vs. calm) were resampled from their estimated
# statistical properties rather than replayed exactly as history happened,
# how would this strategy's actual entry/exit LOGIC perform across many
# plausible alternate market histories?" It fits a small Hidden Markov Model
# (Hamilton-style regime-switching - see hmmlearn's GaussianHMM) to a real
# return series, simulates many synthetic alternate price paths from that
# fitted model, and RE-RUNS THE REAL STRATEGY'S BACKTEST LOGIC on every one of
# them - not just a report of regime probabilities in isolation.
#
# COST WARNING - READ BEFORE PICKING n_paths: this is MUCH more expensive
# than the project's existing trade-level Monte Carlo. monte_carlo_bootstrap
# resamples an already-computed array of trade R-multiples - it is a cheap
# vectorized reshuffle, so n_iter=2000 there costs almost nothing. Here, EVERY
# one of n_paths synthetic paths requires re-running the calling script's full
# backtest_fn (indicator computation, signal detection, position management)
# over an entire synthetic OHLC series the same length as the real data. That
# is why run_regime_stress_test defaults to n_paths=200, not 2000 - budget
# accordingly; 2000 full backtest re-runs over a multi-year 5-minute-bar
# series would likely take hours, not minutes.
#
# SYNTHETIC-DATA CONSTRUCTION - the same trap this project has already hit
# once (see research/ict_po3_forex_dukascopy_backtest.py's header and the
# granularity-convergence tests in research/bollinger_band_mean_reversion_
# dukascopy_backtest.py / research/rsi_mean_reversion_dukascopy_backtest.py's
# own test suites): a single-step "close plus small independent wick noise"
# bar model crushes real intrabar volatility and manufactures a fake edge for
# any stop/target strategy tested against it, purely as a construction
# artifact. generate_synthetic_ohlc_path below uses the SAME discipline those
# test suites already validated - each bar's open anchored to the PREVIOUS
# bar's actual close, dozens of independent sub-steps per bar building a
# genuine intrabar random walk - just with the PER-BAR drift/volatility of
# that walk driven by the HMM's fitted regime at that point in the simulated
# sequence, instead of one fixed drift/vol for the whole synthetic series.
# generate_synthetic_ohlc_path's own test suite (test_regime_stress_test.py)
# runs the exact same granularity-convergence diagnostic those test suites
# use, confirmed on this module's own construction.
#
# ARCHITECTURE - mirrors research/optimization_engine.py's spirit: this
# module never imports from, or depends on, any specific strategy script or
# Dukascopy. run_regime_stress_test takes a generic `backtest_fn(label, df)
# -> list_of_trade_dicts` callback supplied by the CALLER (the same shape as
# ict_po3_forex_dukascopy_backtest.py's / rsi_mean_reversion_dukascopy_
# backtest.py's backtest_instrument(label, df)) - this module only ever calls
# that callback against synthetic data it generates and scores what comes
# back. Every strategy script in this project can reuse it unmodified.

import logging

import numpy as np
import pandas as pd


# ============================================================================
# STEP 1: fit a Hidden Markov regime-switching model to a real return series
# ============================================================================

def _n_free_params(n_states):
    """Free parameters in a 1-D GaussianHMM(n_components=n_states, covariance_type='diag'):
    transmat (n_states rows, each summing to 1 -> n_states*(n_states-1) free) + startprob
    (sums to 1 -> n_states-1 free) + n_states means + n_states variances (diag, 1-D).
    Used for BIC = -2*logL + n_free_params*log(n_obs), the standard model-selection criterion
    for picking how many regimes a return series actually supports (see e.g. Hamilton 1989's
    regime-switching model literature - BIC/AIC is the standard tool for choosing the number
    of regimes, not a fixed convention picked here in isolation)."""
    return n_states * n_states + 2 * n_states - 1


def _label_regimes(means, stds):
    """Ranks states by volatility (ascending) and assigns human-readable labels, so a caller
    (and this module's own print output) never has to reason about opaque hmmlearn state
    indices directly. 2 states -> low-vol/high-vol; 3 -> low/mid/high-vol; more than 3 ->
    low-vol, mid-vol-1..mid-vol-(k-2), high-vol. Returns (labels_by_state_index, vol_rank_order)
    where vol_rank_order is state indices sorted ascending by std (lowest-vol state first)."""
    k = len(stds)
    vol_rank_order = sorted(range(k), key=lambda i: stds[i])
    if k == 1:
        ordinal_labels = ["single regime"]
    elif k == 2:
        ordinal_labels = ["low-vol", "high-vol"]
    elif k == 3:
        ordinal_labels = ["low-vol", "mid-vol", "high-vol"]
    else:
        ordinal_labels = ["low-vol"] + [f"mid-vol-{i}" for i in range(1, k - 1)] + ["high-vol"]
    labels_by_state = {state_idx: ordinal_labels[rank] for rank, state_idx in enumerate(vol_rank_order)}
    return labels_by_state, vol_rank_order


def _fit_one(X, n_states, seed, n_init, n_iter):
    """Fits GaussianHMM(n_components=n_states) with n_init random restarts (different
    random_state per restart) and keeps the highest-log-likelihood fit. EM-based HMM fitting
    can converge to a local optimum from a single random initialization - random restarts are
    the standard mitigation, not a novel trick invented here.

    A restart occasionally diverges (a state collapses to zero occupancy, producing NaN
    parameters) - especially likely when n_states is fitted against data that doesn't actually
    support that many regimes (exactly the case BIC scanning across k is meant to catch). Such
    restarts are silently skipped rather than allowed to raise/propagate NaN; if every restart
    for this n_states diverges, (None, -inf) is returned so the caller can exclude this
    n_states from BIC comparison instead of crashing."""
    from hmmlearn.hmm import GaussianHMM

    best_model, best_ll = None, -np.inf
    for i in range(n_init):
        model = GaussianHMM(n_components=n_states, covariance_type="diag",
                             n_iter=n_iter, tol=1e-6, random_state=seed + i * 1009)
        try:
            model.fit(X)
            ll = model.score(X)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.isfinite(ll):
            continue
        if ll > best_ll:
            best_model, best_ll = model, ll
    return best_model, best_ll


def fit_regime_model(returns, n_regimes=None, max_regimes=4, seed=42, n_init=4, n_iter=200):
    """Fits a Hamilton-style Gaussian HMM regime-switching model to `returns` (a 1-D array-like
    of close-to-close LOG returns - the caller computes these from real historical Dukascopy
    data before calling this, e.g. np.diff(np.log(df['Close']))).

    If n_regimes is given, fits exactly that many states. If n_regimes is None (the default),
    fits every candidate state count from 2 through max_regimes and picks the one with the
    lowest BIC (standard model-selection criterion for HMM regime count - see _n_free_params's
    docstring) - the chosen count and the competing BIC values are both returned and printed,
    not just silently applied.

    Wrapped in try/except ImportError: if hmmlearn isn't installed, prints a clear skip message
    and returns None - matching this project's existing sklearn/matplotlib/optuna-missing
    handling pattern elsewhere (see research/optimization_engine.py's bayesian_search). Never
    crashes a calling script over a missing optional dependency.

    Returns a dict (or None if hmmlearn is missing):
      "n_states":      chosen number of regimes
      "model":         the fitted hmmlearn GaussianHMM object
      "means":         np.ndarray (n_states,) - each regime's estimated mean per-bar return
      "stds":          np.ndarray (n_states,) - each regime's estimated per-bar return std
      "transmat":      np.ndarray (n_states, n_states) - fitted transition matrix
      "startprob":     np.ndarray (n_states,) - fitted initial-state distribution
      "state_labels":  dict state_idx -> human-readable label (e.g. "low-vol"/"high-vol")
      "vol_rank_order": state indices sorted ascending by std (lowest-vol state first)
      "bic_scan":      dict n_states -> bic (only when n_regimes was None; else None)
      "summary":       multi-line human-readable string describing every regime + why it was
                        chosen - print this, don't just consume the opaque numeric fields."""
    try:
        import hmmlearn  # noqa: F401
    except ImportError:
        print("hmmlearn not installed - skipping regime-switching stress test "
              "('!pip install hmmlearn -q' and re-run for this feature). "
              "The rest of this project's trade-level Monte Carlo is unaffected.")
        return None

    logging.getLogger("hmmlearn").setLevel(logging.ERROR)   # quiet EM convergence chatter,
                                                              # matching this project's existing
                                                              # optuna quiet-logging convention

    X = np.asarray(returns, dtype=float).reshape(-1, 1)
    n_obs = len(X)
    min_needed = 20 * max(2, max_regimes if n_regimes is None else n_regimes)
    if n_obs < min_needed:
        raise ValueError(f"fit_regime_model: only {n_obs} return observations - need at least "
                          f"~{min_needed} for a stable fit at this regime count")

    if n_regimes is not None:
        model, ll = _fit_one(X, n_regimes, seed, n_init, n_iter)
        if model is None:
            raise ValueError(f"fit_regime_model: every random restart diverged fitting "
                              f"n_regimes={n_regimes} to this return series - try fewer regimes "
                              f"or more data")
        chosen_k = n_regimes
        bic_scan = None
        reason = f"n_regimes={n_regimes} was fixed explicitly - no BIC scan performed."
    else:
        bic_scan = {}
        fitted = {}
        for k in range(2, max_regimes + 1):
            model_k, ll_k = _fit_one(X, k, seed, n_init, n_iter)
            if model_k is None:
                continue   # every restart at this k diverged (state count not supported by the
                           # data) - exclude it from BIC comparison rather than crash
            bic_scan[k] = float(-2.0 * ll_k + _n_free_params(k) * np.log(n_obs))
            fitted[k] = model_k
        if not bic_scan:
            raise ValueError(f"fit_regime_model: every random restart diverged for every "
                              f"n_regimes in 2..{max_regimes} - try more data")
        chosen_k = min(bic_scan, key=bic_scan.get)
        model = fitted[chosen_k]
        others = ", ".join(f"k={k}: BIC={bic_scan[k]:.1f}" for k in sorted(bic_scan) if k != chosen_k)
        reason = (f"BIC model selection over k=2..{max_regimes}: chose n_regimes={chosen_k} "
                  f"(BIC={bic_scan[chosen_k]:.1f}, lowest) over {others}.")

    means = model.means_.flatten()
    stds = np.sqrt(model.covars_[:, 0, 0])
    state_labels, vol_rank_order = _label_regimes(means, stds)

    lines = [f"Regime-switching model: {chosen_k} regimes fitted to {n_obs} return observations.",
             reason, "Per-regime estimates (ranked low-vol -> high-vol):"]
    for state_idx in vol_rank_order:
        persistence = model.transmat_[state_idx, state_idx]
        lines.append(f"  {state_labels[state_idx]:>10s} (state {state_idx}): "
                      f"mean={means[state_idx]:+.6f}/bar  std={stds[state_idx]:.6f}/bar  "
                      f"persistence(self-transition)={persistence:.3f}")
    summary = "\n".join(lines)

    return {
        "n_states": chosen_k,
        "model": model,
        "means": means,
        "stds": stds,
        "transmat": model.transmat_,
        "startprob": model.startprob_,
        "state_labels": state_labels,
        "vol_rank_order": vol_rank_order,
        "bic_scan": bic_scan,
        "summary": summary,
    }


# ============================================================================
# STEP 2: simulate a synthetic OHLC path from a fitted regime model
# ============================================================================

def generate_synthetic_ohlc_path(regime_model, n_bars, bar_interval_minutes, start_price,
                                  sub_steps_per_bar=40, rng=None, start_time=None):
    """Simulates ONE synthetic OHLC price path of `n_bars` bars from a fitted regime model
    (the dict returned by fit_regime_model): first a regime sequence via the fitted transition
    matrix, then for EACH bar a genuine multi-step intrabar random walk whose drift/volatility
    budget is that bar's regime's mean/std - not a single fixed drift/vol for the whole series,
    and NOT a crude close-plus-independent-wick-noise shortcut (see this module's header for why
    that shortcut is a known trap in this project).

    CONSTRUCTION (same discipline as research/bollinger_band_mean_reversion_dukascopy_backtest.py
    / research/rsi_mean_reversion_dukascopy_backtest.py's own test suites' build_synthetic_ohlc:
    each bar's open anchored to the previous bar's actual close, sub_steps_per_bar independent
    sub-steps per bar building one continuous intrabar path) with one deliberate, documented
    difference: those test helpers compound sub-step noise ADDITIVELY in raw price units (fine
    at their fixed ~1.1 FX-rate scale over a few thousand bars); here, since `returns` going into
    fit_regime_model are LOG returns and this needs to stay well-behaved over arbitrarily long
    synthetic series without ever going negative, sub-steps are compounded in LOG-PRICE space and
    exponentiated back to price - mathematically the correct way to compound returns, and
    equivalent to the additive construction at the return magnitudes this project's bars actually
    have. The genuine-multi-step-per-bar, anchored-to-previous-close discipline itself is
    unchanged and is exactly what this function's own granularity-convergence test (in
    test_regime_stress_test.py) verifies survives at this module's construction, too.

    Returns a DataFrame with Open/High/Low/Close columns and a tz-aware DatetimeIndex spaced
    bar_interval_minutes apart - a drop-in replacement for what this project's
    fetch_instrument_data-style functions already return, so it can be passed straight into any
    strategy's backtest_fn(label, df).

    NOTE ON THE SYNTHETIC CALENDAR: bars are spaced continuously (no weekend/holiday gaps) -
    this is a stress test of the strategy's LOGIC under alternate regime dynamics, not a claim
    to reproduce the real forex trading calendar; strategies that gate on time-of-day still see
    varied, realistic times of day, just across a denser (7-day) calendar than real FX markets
    trade on."""
    rng = rng if rng is not None else np.random.default_rng()

    transmat = np.asarray(regime_model["transmat"], dtype=float)
    startprob = np.asarray(regime_model["startprob"], dtype=float)
    means = np.asarray(regime_model["means"], dtype=float)
    stds = np.asarray(regime_model["stds"], dtype=float)
    k = regime_model["n_states"]

    # --- 1. simulate the regime sequence via the fitted transition matrix ---
    # A first-order Markov chain is inherently sequential (each state depends on the previous
    # one), so this can't be fully vectorized across bars - but each step is a cheap O(k)
    # cumulative-probability lookup against a pre-drawn uniform, not a fresh RNG call per bar.
    cum_transmat = np.cumsum(transmat, axis=1)
    cum_startprob = np.cumsum(startprob)
    u_state = rng.random(n_bars)
    states = np.empty(n_bars, dtype=np.int64)
    states[0] = min(int(np.searchsorted(cum_startprob, rng.random())), k - 1)
    for i in range(1, n_bars):
        states[i] = min(int(np.searchsorted(cum_transmat[states[i - 1]], u_state[i])), k - 1)

    # --- 2. genuine multi-step intrabar random walk, drift/vol driven by that bar's regime ---
    # Variance-preserving decomposition: n independent sub-steps each with std sigma/sqrt(n) sum
    # to a total-bar variance of sigma^2 (n * (sigma/sqrt(n))**2 == sigma**2), same scaling
    # convention already validated in the Bollinger/RSI test suites' build_synthetic_ohlc.
    substep_means = (means[states] / sub_steps_per_bar)[:, None]
    substep_stds = (stds[states] / np.sqrt(sub_steps_per_bar))[:, None]
    increments = rng.normal(loc=substep_means, scale=substep_stds, size=(n_bars, sub_steps_per_bar))
    local_path = np.cumsum(increments, axis=1)              # bar-local cumulative log-return path
    bar_close_local = local_path[:, -1]
    bar_high_local = np.maximum(0.0, local_path.max(axis=1))   # 0.0 = the bar's own open, in log-space
    bar_low_local = np.minimum(0.0, local_path.min(axis=1))

    # each bar's open is anchored to the PREVIOUS bar's actual close - cum_before[i] is the
    # cumulative log-return of every prior bar, i.e. exactly where bar i's path starts from.
    cum_before = np.concatenate(([0.0], np.cumsum(bar_close_local)[:-1]))

    opens = start_price * np.exp(cum_before)
    closes = start_price * np.exp(cum_before + bar_close_local)
    highs = start_price * np.exp(cum_before + bar_high_local)
    lows = start_price * np.exp(cum_before + bar_low_local)

    if start_time is None:
        start_time = pd.Timestamp("2020-01-06", tz="America/New_York")
    idx = pd.date_range(start=start_time, periods=n_bars, freq=f"{bar_interval_minutes}min")

    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


# ============================================================================
# STEP 3: the main entry point - fit, simulate n_paths, re-run the real
# strategy on every one, and report where the real result falls
# ============================================================================

def _max_drawdown_r(r_values):
    """Largest peak-to-trough drop in the cumulative R sum, in the order given. Deliberately
    re-implemented here rather than imported from research/optimization_engine.py (this module
    must not depend on any other strategy-facing script, same discipline optimization_engine.py
    itself documents for its own max_drawdown_r) - but the arithmetic is copied exactly, so
    "max drawdown" means the same thing here as it already does everywhere else in this
    project's Monte Carlo output."""
    r = np.asarray(r_values, dtype=float)
    if len(r) == 0:
        return 0.0
    cum = np.cumsum(r)
    running_max = np.maximum.accumulate(cum)
    return float((running_max - cum).max())


def _trade_stats(trades):
    r = [t["r"] for t in trades]
    n = len(r)
    total = float(sum(r)) if n else 0.0
    return {
        "total_r": total,
        "avg_r": (total / n) if n else 0.0,
        "n_trades": n,
        "max_drawdown": _max_drawdown_r(r),
    }


def _percentile_and_verdict(real_value, sim_values, higher_is_better=True, low_pct=5.0, high_pct=95.0):
    """Core of run_regime_stress_test's interpretation step, factored out so it can be unit
    tested directly against a controlled distribution: what percentile does `real_value` fall at
    within `sim_values` (fraction of simulated paths at or below it, as a percentage), and is
    that an outlier worth flagging in either direction?

    higher_is_better=True (total_r, avg_r): a HIGH percentile (>= high_pct) means the real
    result beat almost every regime-simulated path -> "OUTPERFORMS"; a LOW percentile (<=
    low_pct) means it beat almost none -> "UNDERPERFORMS".
    higher_is_better=False (max_drawdown, where smaller is better): the sense is flipped - a LOW
    percentile (real drawdown smaller than almost every simulated path's) -> "OUTPERFORMS"; a
    HIGH percentile -> "UNDERPERFORMS".
    Otherwise (in the middle): "TYPICAL" - the real result is unremarkable against what the
    estimated regime dynamics alone would typically produce.

    Returns (percentile, verdict) or (None, "no simulated paths to compare against") if
    sim_values is empty."""
    sim = np.asarray(sim_values, dtype=float)
    if len(sim) == 0:
        return None, "no simulated paths to compare against"
    pct = float(np.mean(sim <= real_value) * 100.0)
    if higher_is_better:
        if pct >= high_pct:
            verdict = "OUTPERFORMS"
        elif pct <= low_pct:
            verdict = "UNDERPERFORMS"
        else:
            verdict = "TYPICAL"
    else:
        if pct <= low_pct:
            verdict = "OUTPERFORMS"
        elif pct >= high_pct:
            verdict = "UNDERPERFORMS"
        else:
            verdict = "TYPICAL"
    return pct, verdict


def run_regime_stress_test(backtest_fn, real_df, label, n_paths=200, sub_steps_per_bar=40, seed=42,
                            n_regimes=None, max_regimes=4, real_trades=None, bar_interval_minutes=None,
                            low_pct=5.0, high_pct=95.0, progress_cb=None, show_progress=True):
    """Main entry point. Fits a regime model to real_df's actual historical close-to-close log
    returns, generates n_paths synthetic OHLC price series of the same length via
    generate_synthetic_ohlc_path, runs backtest_fn(label, synthetic_df) on EACH ONE (the
    expensive part - see this module's header cost warning), and compares the REAL backtest's
    own result against that simulated distribution.

    backtest_fn: (label, df) -> list_of_trade_dicts, each dict having at least an "r" key - the
                 exact shape research/ict_po3_forex_dukascopy_backtest.py's and research/
                 rsi_mean_reversion_dukascopy_backtest.py's backtest_instrument(label, df)
                 already have. This module never imports any strategy script itself; the caller
                 supplies backtest_fn.
    real_df:     the real historical OHLC DataFrame (Open/High/Low/Close, DatetimeIndex) the
                 strategy was actually backtested on.
    real_trades: the real backtest's own trades, if already computed (list of trade dicts,
                 same shape backtest_fn returns) - pass this to avoid re-running backtest_fn on
                 real_df a second time. If None, this function calls backtest_fn(label, real_df)
                 itself.

    Returns None if hmmlearn is unavailable (fit_regime_model's ImportError path) - otherwise a
    dict:
      "regime_model":  the dict from fit_regime_model
      "real_stats":    {"total_r", "avg_r", "n_trades", "max_drawdown"} for the real backtest
      "path_stats":    list of n_paths per-path stat dicts (same 4 keys)
      "distributions": {"total_r": np.ndarray, "avg_r": ..., "n_trades": ..., "max_drawdown": ...}
                       across all n_paths synthetic runs
      "percentiles":   {"total_r": pct, "avg_r": pct, "max_drawdown": pct} - where the real
                       result falls within the simulated distribution (see
                       _percentile_and_verdict)
      "verdicts":      {"total_r": "OUTPERFORMS"|"UNDERPERFORMS"|"TYPICAL", ...}
      "n_paths":       n_paths actually run (== len(path_stats))"""
    regime_returns = np.diff(np.log(real_df["Close"].to_numpy(dtype=float)))
    regime_model = fit_regime_model(regime_returns, n_regimes=n_regimes, max_regimes=max_regimes, seed=seed)
    if regime_model is None:
        return None   # hmmlearn missing - fit_regime_model already printed the skip message

    print(regime_model["summary"])

    if bar_interval_minutes is None:
        deltas = real_df.index.to_series().diff().dropna()
        bar_interval_minutes = deltas.mode().iloc[0].total_seconds() / 60.0 if len(deltas) else 5.0

    if real_trades is None:
        real_trades = backtest_fn(label, real_df)
    real_stats = _trade_stats(real_trades)

    n_bars = len(real_df)
    start_price = float(real_df["Close"].iloc[0])
    start_time = real_df.index[0]

    print(f"\nRegime stress test: running backtest_fn on {n_paths} synthetic {n_bars}-bar paths "
          f"({sub_steps_per_bar} intrabar sub-steps/bar) - this re-runs the FULL strategy logic "
          f"per path (much more expensive than this project's trade-level Monte Carlo; see this "
          f"module's header). This may take a while for large n_bars.")

    rng = np.random.default_rng(seed)
    path_stats = []
    iterator = range(n_paths)
    if show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc=f"{label}: regime stress test", unit="path")
        except ImportError:
            pass

    for p in iterator:
        synthetic_df = generate_synthetic_ohlc_path(
            regime_model, n_bars=n_bars, bar_interval_minutes=bar_interval_minutes,
            start_price=start_price, sub_steps_per_bar=sub_steps_per_bar, rng=rng, start_time=start_time)
        trades = backtest_fn(label, synthetic_df)
        path_stats.append(_trade_stats(trades))
        if progress_cb is not None:
            progress_cb(p + 1, n_paths)

    distributions = {key: np.array([s[key] for s in path_stats], dtype=float)
                      for key in ("total_r", "avg_r", "n_trades", "max_drawdown")}

    percentiles, verdicts = {}, {}
    for key, higher_is_better in (("total_r", True), ("avg_r", True), ("max_drawdown", False)):
        pct, verdict = _percentile_and_verdict(real_stats[key], distributions[key],
                                                higher_is_better=higher_is_better,
                                                low_pct=low_pct, high_pct=high_pct)
        percentiles[key], verdicts[key] = pct, verdict

    n_zero_trade_paths = int(np.sum(distributions["n_trades"] == 0))
    print(f"\n{'=' * 70}\nREGIME-SWITCHING STRESS TEST - {label} ({n_paths} synthetic paths)\n{'=' * 70}")
    print(f"Real backtest: total_r={real_stats['total_r']:+.2f}  avg_r={real_stats['avg_r']:+.4f}  "
          f"n_trades={real_stats['n_trades']}  max_drawdown={real_stats['max_drawdown']:.2f}")
    if n_zero_trade_paths:
        print(f"NOTE: {n_zero_trade_paths}/{n_paths} synthetic paths produced zero trades "
              "(counted as total_r=avg_r=max_drawdown=0.0 in the distribution below).")
    for key in ("total_r", "avg_r", "max_drawdown"):
        dist = distributions[key]
        p5, p50, p95 = np.percentile(dist, [5, 50, 95])
        pct, verdict = percentiles[key], verdicts[key]
        print(f"  {key:>13s}: simulated p5={p5:+.3f}  p50={p50:+.3f}  p95={p95:+.3f}   |   "
              f"real={real_stats[key]:+.3f} sits at percentile {pct:.1f} -> {verdict}")

    return {
        "regime_model": regime_model,
        "real_stats": real_stats,
        "path_stats": path_stats,
        "distributions": distributions,
        "percentiles": percentiles,
        "verdicts": verdicts,
        "n_paths": len(path_stats),
    }
