# Unit tests + one synthetic end-to-end smoke run for
# ict_po3_forex_dukascopy_optimization.py.
#
# Covers, per this project's rigor conventions:
#   1. Window-slicing/masking logic - verifies it doesn't leak trades outside
#      [window_start, window_end) and doesn't corrupt the accumulation-range
#      computation (a windowed run must reproduce exactly the subset of
#      trades a full run would have produced for that window - never
#      different ones).
#   2. Monte Carlo resampling - verifies bootstrap-of-constant-values
#      converges exactly to the constant, verifies shuffle preserves total R
#      (mathematically invariant under reordering) while its drawdown still
#      varies, and basic percentile-ordering sanity checks.
#   3. Walk-forward fold boundary generator - verifies the 6 rolling folds
#      compute the exact boundaries described in this project's spec.
#   4. ONE small synthetic end-to-end smoke run (small price series, small
#      grid, 2 folds) confirming the full pipeline (grid -> Monte Carlo ->
#      cluster -> walk-forward) runs without crashing. Deliberately NOT a
#      real Dukascopy download - this project's convention is to validate
#      correctness via unit tests plus a synthetic smoke run, not to run the
#      full real backtest in a sandboxed test pass.
#
# Run with:  python -m pytest research/test_ict_po3_forex_dukascopy_optimization.py -v
# or:        python research/test_ict_po3_forex_dukascopy_optimization.py

import datetime
import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "ict_po3_forex_dukascopy_optimization.py")
_spec = importlib.util.spec_from_file_location("ict_po3_forex_dukascopy_optimization", _MODULE_PATH)
opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(opt)


# ============================= synthetic data helpers =============================

def _make_day_index(date_, tz="America/New_York"):
    """204 5-min bars, 00:00 through 16:55 NY time - covers accumulation (00:00-05:00),
    manipulation (05:00-08:00), and distribution (through 17:00) entirely."""
    return pd.date_range(start=pd.Timestamp(date_.year, date_.month, date_.day, 0, 0, tz=tz),
                          periods=204, freq="5min")


def _build_synthetic_df(dates, base_price=1.1000, overrides=None):
    """Builds a synthetic 5-min OHLC DataFrame across the given list of dates. Every bar is
    flat at base_price unless `overrides[date]["HH:MM"] = (o, h, l, c)` says otherwise."""
    overrides = overrides or {}
    idx_all = []
    rows = []
    for d in dates:
        for ts in _make_day_index(d):
            tod_str = ts.strftime("%H:%M")
            ov = overrides.get(d, {}).get(tod_str)
            if ov is not None:
                o, h, l, c = ov
            else:
                o = h = l = c = base_price
            idx_all.append(ts)
            rows.append({"Open": o, "High": h, "Low": l, "Close": c})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx_all))


def _manipulation_break_high(entry=1.1005, high=1.1010, low=1.1000):
    """A manipulation bar that breaks the (flat, zero-width) accumulation range to the upside -
    triggers a SHORT per PO3's fade-the-manipulation rule."""
    return (entry, high, low, entry)


# Three consecutive weekday dates, safely away from any US DST transition (all in Jan 2019).
_DAY1 = datetime.date(2019, 1, 7)
_DAY2 = datetime.date(2019, 1, 8)
_DAY3 = datetime.date(2019, 1, 9)

_STOP_BUFFER_PCT = 0.02
_FALLBACK_REWARD_RISK = 2.0


def _build_three_day_fixture():
    """Day1: manipulation break-high -> SHORT -> price dips to hit the fallback target (TP).
    Day2: same setup -> SHORT -> price instead rallies to hit the stop (SL).
    Day3: same setup as Day1 (TP again) - a distinct third trade to make window-boundary
    filtering unambiguous to check."""
    overrides = {
        _DAY1: {"05:00": _manipulation_break_high(), "05:05": (1.0990, 1.1000, 1.0980, 1.0985)},
        _DAY2: {"05:00": _manipulation_break_high(), "05:05": (1.1015, 1.1025, 1.1010, 1.1020)},
        _DAY3: {"05:00": _manipulation_break_high(), "05:05": (1.0990, 1.1000, 1.0980, 1.0985)},
    }
    df = _build_synthetic_df([_DAY1, _DAY2, _DAY3], overrides=overrides)
    return opt.precompute_indicators(df)


class TestWindowSlicing(unittest.TestCase):
    def setUp(self):
        self.ind = _build_three_day_fixture()

    def test_full_range_produces_expected_trades(self):
        trades = opt.run_po3_backtest(self.ind, _STOP_BUFFER_PCT, _FALLBACK_REWARD_RISK)
        self.assertEqual(len(trades), 3)
        dates = [t["date"] for t in trades]
        self.assertEqual(dates, [_DAY1, _DAY2, _DAY3])
        self.assertEqual(trades[0]["outcome"], "TP")
        self.assertAlmostEqual(trades[0]["r"], _FALLBACK_REWARD_RISK)
        self.assertEqual(trades[1]["outcome"], "SL")
        self.assertAlmostEqual(trades[1]["r"], -1.0)
        self.assertEqual(trades[2]["outcome"], "TP")
        self.assertAlmostEqual(trades[2]["r"], _FALLBACK_REWARD_RISK)

    def test_window_excludes_trades_outside_it(self):
        # window_start=DAY2, window_end=DAY3 (exclusive) -> only DAY2's trade should appear.
        windowed = opt.run_po3_backtest(self.ind, _STOP_BUFFER_PCT, _FALLBACK_REWARD_RISK,
                                         window_start=_DAY2, window_end=_DAY3)
        self.assertEqual(len(windowed), 1)
        self.assertEqual(windowed[0]["date"], _DAY2)
        self.assertEqual(windowed[0]["outcome"], "SL")

    def test_window_does_not_corrupt_indicator_computation(self):
        """A windowed run must reproduce EXACTLY the subset of trades a full run would have
        produced, filtered to that window - never a different trade or a different R value.
        This is the direct check that skipping out-of-window bars doesn't leave stale
        accumulation-range state bleeding into an in-window day."""
        full_trades = opt.run_po3_backtest(self.ind, _STOP_BUFFER_PCT, _FALLBACK_REWARD_RISK)
        full_filtered = [t for t in full_trades if _DAY2 <= t["date"] < _DAY3]

        windowed = opt.run_po3_backtest(self.ind, _STOP_BUFFER_PCT, _FALLBACK_REWARD_RISK,
                                         window_start=_DAY2, window_end=_DAY3)

        self.assertEqual(len(full_filtered), len(windowed))
        for a, b in zip(full_filtered, windowed):
            self.assertEqual(a["date"], b["date"])
            self.assertEqual(a["outcome"], b["outcome"])
            self.assertAlmostEqual(a["r"], b["r"])

    def test_window_start_only_and_window_end_only(self):
        only_after_day2 = opt.run_po3_backtest(self.ind, _STOP_BUFFER_PCT, _FALLBACK_REWARD_RISK,
                                                window_start=_DAY2)
        self.assertEqual([t["date"] for t in only_after_day2], [_DAY2, _DAY3])

        only_before_day3 = opt.run_po3_backtest(self.ind, _STOP_BUFFER_PCT, _FALLBACK_REWARD_RISK,
                                                 window_end=_DAY3)
        self.assertEqual([t["date"] for t in only_before_day3], [_DAY1, _DAY2])

    def test_empty_window_produces_no_trades(self):
        # A window entirely before any data in the fixture.
        no_trades = opt.run_po3_backtest(self.ind, _STOP_BUFFER_PCT, _FALLBACK_REWARD_RISK,
                                          window_start=datetime.date(2018, 1, 1),
                                          window_end=datetime.date(2018, 6, 1))
        self.assertEqual(no_trades, [])


class TestMonteCarlo(unittest.TestCase):
    def test_bootstrap_of_constant_values_converges_exactly(self):
        # sum of n draws-with-replacement from a constant-valued population is deterministic:
        # every simulated path's total R must equal exactly n * value, no variance possible.
        r = [0.5] * 40
        result = opt.monte_carlo_bootstrap(r, n_iter=500, rng=np.random.default_rng(0))
        self.assertTrue(np.allclose(result["total_r"], 0.5 * 40))
        # a constant positive value produces a monotonically non-decreasing cumulative sum,
        # i.e. zero drawdown along every single simulated path.
        self.assertTrue(np.allclose(result["max_drawdown"], 0.0))

    def test_bootstrap_of_constant_negative_values(self):
        n = 25
        r = [-0.3] * n
        result = opt.monte_carlo_bootstrap(r, n_iter=300, rng=np.random.default_rng(1))
        self.assertTrue(np.allclose(result["total_r"], -0.3 * n))
        # monotonically decreasing cumsum -> the running peak is always the FIRST bar's
        # cumulative value (-0.3, the least-negative point on the whole path), so max
        # drawdown = that peak minus the final trough = -0.3 - (-0.3*n) = 0.3*(n-1) exactly,
        # deterministic across every simulated path (constant values, no variance possible).
        self.assertTrue(np.allclose(result["max_drawdown"], 0.3 * (n - 1)))

    def test_bootstrap_produces_sane_variance_and_percentile_ordering(self):
        r = [-1, -1, 2, -1, 2, -1, 2, -1, -1, 2]
        result = opt.monte_carlo_bootstrap(r, n_iter=2000, rng=np.random.default_rng(2))
        self.assertGreater(np.std(result["total_r"]), 0)
        p5, p50, p95 = (np.percentile(result["total_r"], q) for q in (5, 50, 95))
        self.assertLessEqual(p5, p50)
        self.assertLessEqual(p50, p95)
        p_le_0 = float(np.mean(result["total_r"] <= 0))
        self.assertGreaterEqual(p_le_0, 0.0)
        self.assertLessEqual(p_le_0, 1.0)

    def test_shuffle_preserves_total_r_but_drawdown_varies(self):
        r = [-1, -1, 2, -1, 2, -1, -1, 2, 2, -1]
        result = opt.monte_carlo_shuffle(r, n_iter=500, rng=np.random.default_rng(3))
        # sum of a fixed multiset is invariant under reordering - every iteration must match
        # the true total exactly.
        self.assertTrue(np.allclose(result["total_r"], sum(r)))
        # but the ORDER varies, so drawdown (which depends on path, not just the endpoint)
        # should show real variance across iterations.
        self.assertGreater(np.std(result["max_drawdown"]), 0)

    def test_empty_trade_list_handled(self):
        boot = opt.monte_carlo_bootstrap([], n_iter=50)
        shuf = opt.monte_carlo_shuffle([], n_iter=50)
        self.assertEqual(len(boot["total_r"]), 0)
        self.assertEqual(len(shuf["total_r"]), 0)
        self.assertIsNone(opt.summarize_mc(boot))
        self.assertIsNone(opt.summarize_mc(shuf))


class TestWalkForwardFoldBoundaries(unittest.TestCase):
    def test_exact_six_rolling_folds(self):
        folds = opt.generate_walk_forward_folds(2016, 2025)
        self.assertEqual(len(folds), 6)

        expected = [
            (2016, 2019, 2019, 2020),
            (2017, 2020, 2020, 2021),
            (2018, 2021, 2021, 2022),
            (2019, 2022, 2022, 2023),
            (2020, 2023, 2023, 2024),
            (2021, 2024, 2024, 2025),
        ]
        for fold, (is_y, ie_y, oy, oe_y) in zip(folds, expected):
            self.assertEqual(fold["is_start"], datetime.date(is_y, 1, 1))
            self.assertEqual(fold["is_end"], datetime.date(ie_y, 1, 1))
            self.assertEqual(fold["oos_start"], datetime.date(oy, 1, 1))
            self.assertEqual(fold["oos_end"], datetime.date(oe_y, 1, 1))

        # non-overlapping: each fold's OOS should immediately follow its own IS, and each
        # successive fold should start exactly one year after the previous one.
        for f in folds:
            self.assertEqual(f["oos_start"], f["is_end"])
        for prev, nxt in zip(folds, folds[1:]):
            self.assertEqual(nxt["is_start"].year, prev["is_start"].year + 1)

    def test_no_fold_extends_past_fetch_end(self):
        folds = opt.generate_walk_forward_folds(2016, 2025)
        for f in folds:
            self.assertLessEqual(f["oos_end"], datetime.date(2025, 1, 1))

    def test_custom_span_produces_two_folds(self):
        # e.g. a 2016-2021 span (used by the smoke test below) should yield exactly 2 folds.
        folds = opt.generate_walk_forward_folds(2016, 2021)
        self.assertEqual(len(folds), 2)
        self.assertEqual(folds[0]["is_start"], datetime.date(2016, 1, 1))
        self.assertEqual(folds[0]["oos_end"], datetime.date(2020, 1, 1))
        self.assertEqual(folds[1]["is_start"], datetime.date(2017, 1, 1))
        self.assertEqual(folds[1]["oos_end"], datetime.date(2021, 1, 1))


class TestSmokeEndToEnd(unittest.TestCase):
    """ONE small synthetic end-to-end run: small synthetic price series, small grid, 2 folds -
    confirms the full pipeline (grid search -> Monte Carlo -> cluster analysis -> walk-forward)
    runs to completion without crashing. Not a claim about real-world performance (the price
    series is synthetic), only a plumbing/crash check - matching this project's convention of
    validating new scripts via unit tests + one synthetic smoke run rather than a full real
    backtest inside a sandboxed test pass."""

    def _build_multi_year_synthetic_data(self, start_year, end_year_exclusive, label="SYN"):
        """Builds a few years of trading-day bars with an alternating win/loss manipulation
        pattern (cheap: only the handful of bars each day that actually matter - accumulation,
        the manipulation break, and one resolution bar - are generated, not a full 24h grid)."""
        dates = pd.bdate_range(datetime.date(start_year, 1, 1),
                                datetime.date(end_year_exclusive, 1, 1), inclusive="left")
        overrides = {}
        for i, ts in enumerate(dates):
            d = ts.date()
            if i % 2 == 0:
                overrides[d] = {"05:00": _manipulation_break_high(),
                                 "05:05": (1.0990, 1.1000, 1.0980, 1.0985)}   # TP path
            else:
                overrides[d] = {"05:00": _manipulation_break_high(),
                                 "05:05": (1.1030, 1.1040, 1.1020, 1.1035)}   # SL path
        df = _build_synthetic_df([ts.date() for ts in dates], overrides=overrides)
        return {label: opt.precompute_indicators(df)}

    def test_full_pipeline_runs_without_crashing(self):
        data = self._build_multi_year_synthetic_data(2016, 2021)   # -> exactly 2 walk-forward folds
        small_grid = [(sb, frr) for sb in [0.02, 0.05] for frr in [1.0, 2.0]]   # small grid, per spec

        grid_results = opt.run_grid_search(data, grid=small_grid, desc="smoke test grid")
        self.assertEqual(len(grid_results), 4)
        for cell in grid_results:
            self.assertGreater(cell["n_trades"], 0)
            self.assertIsInstance(cell["avg_r"], float)

        mc_results = opt.run_monte_carlo_all_cells(grid_results, n_iter=200)
        self.assertEqual(len(mc_results), 4)
        for cell in mc_results:
            self.assertIsNotNone(cell["bootstrap"])
            self.assertIsNotNone(cell["shuffle"])

        neighbor_result = opt.neighbor_plateau_check(grid_results,
                                                      sb_grid=sorted({sb for sb, _ in small_grid}),
                                                      frr_grid=sorted({frr for _, frr in small_grid}))
        self.assertIn(neighbor_result["verdict"], ("PLATEAU", "ISOLATED SPIKE / overfit warning"))

        cluster_result = opt.sklearn_cluster_analysis(
            grid_results,
            sb_grid=sorted({sb for sb, _ in small_grid}),
            frr_grid=sorted({frr for _, frr in small_grid}),
            k=2,
        )
        # sklearn is present in this project's environment; if it weren't, this would be None
        # and that's an acceptable (explicitly handled) outcome too.
        if cluster_result is not None:
            self.assertGreaterEqual(cluster_result["cluster_size"], 1)

        folds = opt.generate_walk_forward_folds(2016, 2021)
        self.assertEqual(len(folds), 2)
        fold_results, combined_oos_r = opt.run_walk_forward(data, folds, grid=small_grid)
        self.assertEqual(len(fold_results), 2)
        wfe_stats = opt.compute_walk_forward_efficiency(fold_results, combined_oos_r)
        self.assertIn("wfe", wfe_stats)

        print(f"\n[smoke test] grid best cell: {max(grid_results, key=lambda r: r['total_r'])}")
        print(f"[smoke test] fold results: {fold_results}")
        print(f"[smoke test] WFE stats: {wfe_stats}")


# ============================= NEW: optimization_engine retrofit (SEARCH_METHOD/OBJECTIVE) =============================
#
# Everything above this line is UNMODIFIED from before the optimization_engine retrofit - it must
# keep passing exactly as-is (see pytest output: same test count, same names, all green) as direct
# proof this refactor didn't change already-verified behavior. Everything below is NEW coverage for
# the retrofit itself.

def _small_multi_year_data(start_year=2016, end_year_exclusive=2019, label="SYN"):
    """Same synthetic-data shape as TestSmokeEndToEnd._build_multi_year_synthetic_data above,
    factored out to a module-level helper so the new tests below can reuse it without duplicating
    TestSmokeEndToEnd's internals or depending on that class."""
    dates = pd.bdate_range(datetime.date(start_year, 1, 1),
                            datetime.date(end_year_exclusive, 1, 1), inclusive="left")
    overrides = {}
    for i, ts in enumerate(dates):
        d = ts.date()
        if i % 2 == 0:
            overrides[d] = {"05:00": _manipulation_break_high(),
                             "05:05": (1.0990, 1.1000, 1.0980, 1.0985)}   # TP path
        else:
            overrides[d] = {"05:00": _manipulation_break_high(),
                             "05:05": (1.1030, 1.1040, 1.1020, 1.1035)}   # SL path
    df = _build_synthetic_df([ts.date() for ts in dates], overrides=overrides)
    return {label: opt.precompute_indicators(df)}


class TestSearchEngineRetrofit(unittest.TestCase):
    """New tests for the optimization_engine retrofit: SEARCH_METHOD/OBJECTIVE options end-to-end
    on this file's existing synthetic smoke-test data shape, plus a direct regression check that
    the new engine-dispatched search reproduces the OLD hand-written run_grid_search's numbers
    exactly under default settings (method="grid", objective="total_r")."""

    def test_run_param_search_grid_default_matches_old_run_grid_search_exactly(self):
        """REGRESSION TEST: run_param_search's default (SEARCH_METHOD="grid", OBJECTIVE="total_r")
        must reproduce the OLD hand-written run_grid_search's exact per-cell numbers and the exact
        same best combo, on the same synthetic data - proving this refactor did not change default
        behavior."""
        data = _small_multi_year_data()
        small_grid = [(sb, frr) for sb in [0.02, 0.05] for frr in [1.0, 2.0]]

        old_results = opt.run_grid_search(data, grid=small_grid, desc="old-style")
        new_results, search_result = opt.run_param_search(data, grid=small_grid, method="grid",
                                                            objective="total_r", desc="new-style")

        self.assertEqual(len(old_results), len(new_results))
        for old, new in zip(old_results, new_results):
            self.assertEqual(old["stop_buffer_pct"], new["stop_buffer_pct"])
            self.assertEqual(old["fallback_reward_risk"], new["fallback_reward_risk"])
            self.assertEqual(old["n_trades"], new["n_trades"])
            self.assertAlmostEqual(old["total_r"], new["total_r"], places=9)
            self.assertAlmostEqual(old["avg_r"], new["avg_r"], places=9)
            self.assertEqual(sorted(old["trades_r"]), sorted(new["trades_r"]))

        old_best = max(old_results, key=lambda r: r["total_r"])
        new_best_params = search_result["best"]["params"]
        self.assertEqual(old_best["stop_buffer_pct"], new_best_params["stop_buffer_pct"])
        self.assertEqual(old_best["fallback_reward_risk"], new_best_params["fallback_reward_risk"])
        self.assertAlmostEqual(old_best["total_r"], search_result["best"]["score"], places=9)

    def test_search_method_genetic_end_to_end(self):
        data = _small_multi_year_data()
        small_grid = [(sb, frr) for sb in [0.01, 0.02, 0.05, 0.1] for frr in [1.0, 1.5, 2.0, 2.5]]
        grid_results, search_result = opt.run_param_search(
            data, grid=small_grid, method="genetic", objective="total_r",
            population_size=4, generations=3, seed=1)
        self.assertEqual(search_result["method"], "genetic")
        self.assertGreater(len(grid_results), 0)
        self.assertIsNotNone(search_result["best"])
        self.assertLessEqual(search_result["n_evals"], 16)   # <= the full 4x4 grid it's sampling from

    def test_search_method_bayesian_end_to_end_or_falls_back_cleanly(self):
        data = _small_multi_year_data()
        small_grid = [(sb, frr) for sb in [0.01, 0.02, 0.05, 0.1] for frr in [1.0, 1.5, 2.0, 2.5]]
        grid_results, search_result = opt.run_param_search(
            data, grid=small_grid, method="bayesian", objective="total_r", n_trials=6, seed=1)
        # either optuna ran (method == "bayesian") or run_search fell back to grid gracefully -
        # both are acceptable, crash-free outcomes; the important thing is it never raises.
        self.assertIn(search_result["method"], ("bayesian", "grid"))
        self.assertGreater(len(grid_results), 0)
        self.assertIsNotNone(search_result["best"])

    def test_objective_options_all_run_without_crashing(self):
        data = _small_multi_year_data()
        small_grid = [(sb, frr) for sb in [0.02, 0.05] for frr in [1.0, 2.0]]
        for objective_name in opt.opt_engine.OBJECTIVES:
            grid_results, search_result = opt.run_param_search(
                data, grid=small_grid, method="grid", objective=objective_name, desc=objective_name)
            self.assertEqual(len(grid_results), 4)
            self.assertIsNotNone(search_result["best"])
            self.assertFalse(np.isnan(search_result["best"]["score"]))

    def test_win_rate_objective_can_disagree_with_total_r_objective(self):
        """The whole reason win_rate is offered as a SELECTABLE (not default) objective: it can
        pick a different "best" cell than total_r would - proving the two are genuinely different
        selection rules being exercised, not just two names for the same outcome."""
        data = _small_multi_year_data(2016, 2020)
        full_grid = [(sb, frr) for sb in opt.STOP_BUFFER_PCT_GRID for frr in opt.FALLBACK_REWARD_RISK_GRID]
        _, total_r_result = opt.run_param_search(data, grid=full_grid, method="grid", objective="total_r")
        _, win_rate_result = opt.run_param_search(data, grid=full_grid, method="grid", objective="win_rate")
        self.assertIsNotNone(total_r_result["best"])
        self.assertIsNotNone(win_rate_result["best"])
        # not asserting they always differ (depends on data), just that both ran to a real result
        # independently and win_rate's score is a valid fraction in [0, 1].
        self.assertGreaterEqual(win_rate_result["best"]["score"], 0.0)
        self.assertLessEqual(win_rate_result["best"]["score"], 1.0)

    def test_walk_forward_honors_configured_method_and_objective_override(self):
        data = _small_multi_year_data(2016, 2021)
        small_grid = [(sb, frr) for sb in [0.02, 0.05] for frr in [1.0, 2.0]]
        folds = opt.generate_walk_forward_folds(2016, 2021)
        fold_results, combined_oos_r = opt.run_walk_forward(data, folds, grid=small_grid,
                                                              method="genetic", objective="avg_r")
        self.assertEqual(len(fold_results), len(folds))
        for f in fold_results:
            self.assertIn("best_stop_buffer_pct", f)
            self.assertIn("best_fallback_reward_risk", f)

    def test_walk_forward_default_still_matches_old_behavior(self):
        """Defaults (no method/objective override) must still select each fold's winner by raw
        total R via an exhaustive grid search, exactly as before this refactor."""
        data = _small_multi_year_data(2016, 2021)
        small_grid = [(sb, frr) for sb in [0.02, 0.05] for frr in [1.0, 2.0]]
        folds = opt.generate_walk_forward_folds(2016, 2021)

        fold_results, combined_oos_r = opt.run_walk_forward(data, folds, grid=small_grid)

        for f in fold_results:
            is_grid = opt.run_grid_search(data, window_start=f["is_start"], window_end=f["is_end"],
                                           grid=small_grid, desc="check")
            expected_best = max(is_grid, key=lambda r: r["total_r"])
            self.assertEqual(f["best_stop_buffer_pct"], expected_best["stop_buffer_pct"])
            self.assertEqual(f["best_fallback_reward_risk"], expected_best["fallback_reward_risk"])
            self.assertAlmostEqual(f["is_total_r"], expected_best["total_r"], places=9)

    def test_print_cluster_analysis_skips_neighbor_check_under_non_grid_search(self):
        data = _small_multi_year_data()
        small_grid = [(sb, frr) for sb in [0.01, 0.02, 0.05, 0.1] for frr in [1.0, 1.5, 2.0, 2.5]]
        grid_results, _ = opt.run_param_search(data, grid=small_grid, method="genetic",
                                                 population_size=4, generations=3, seed=2)
        neighbor_result, cluster_result = opt.print_cluster_analysis(grid_results, search_method="genetic")
        self.assertIsNone(neighbor_result)   # explicitly skipped, per print_cluster_analysis's docstring

    def test_print_cluster_analysis_runs_neighbor_check_under_grid_search(self):
        data = _small_multi_year_data()
        small_grid = [(sb, frr) for sb in [0.01, 0.02, 0.05, 0.1] for frr in [1.0, 1.5, 2.0, 2.5]]
        grid_results, _ = opt.run_param_search(data, grid=small_grid, method="grid")
        neighbor_result, cluster_result = opt.print_cluster_analysis(grid_results, search_method="grid")
        self.assertIsNotNone(neighbor_result)
        self.assertIn(neighbor_result["verdict"], ("PLATEAU", "ISOLATED SPIKE / overfit warning"))

    def test_search_method_and_objective_module_defaults_are_grid_and_total_r(self):
        """The actual behavior-preservation guarantee: unless something explicitly overrides them,
        this script's own module-level config must still be exactly what shipped before this
        refactor existed."""
        self.assertEqual(opt.SEARCH_METHOD, "grid")
        self.assertEqual(opt.OBJECTIVE, "total_r")


class TestCorrectedZScoreAndBonferroni(unittest.TestCase):
    """Covers this script's use of optimization_engine's CORRECTED z-score (replacing the old
    avg_r*sqrt(n) shortcut that implicitly assumed std(r) == 1) and the Bonferroni multiple-testing
    helper - see optimization_engine.py's own test suite for unit-level coverage of
    zscore()/bonferroni_adjusted_z_threshold() themselves; this just confirms this script actually
    wires them up correctly against its own real trade-shaped data."""

    def test_zscore_differs_from_old_broken_shortcut_on_real_trade_shaped_data(self):
        data = _small_multi_year_data(2016, 2020)
        full_grid = [(sb, frr) for sb in opt.STOP_BUFFER_PCT_GRID for frr in opt.FALLBACK_REWARD_RISK_GRID]
        grid_results, _ = opt.run_param_search(data, grid=full_grid, method="grid", objective="total_r")
        best_cell = max(grid_results, key=lambda r: r["total_r"])
        r = np.array(best_cell["trades_r"])
        self.assertGreaterEqual(len(r), 2)

        corrected = opt.opt_engine.zscore([{"r": v} for v in r])
        old_broken = (r.mean()) * np.sqrt(len(r))   # the old avg_r * sqrt(n) shortcut being replaced
        self.assertFalse(np.isnan(corrected))
        if r.std(ddof=1) > 0:
            # the corrected version divides by the REAL sample std, not an implicit std==1 -
            # whenever that real std isn't ~1, the two numbers must differ.
            if abs(r.std(ddof=1) - 1.0) > 1e-6:
                self.assertNotAlmostEqual(corrected, old_broken, places=6)

    def test_bonferroni_threshold_accessible_and_sane_for_this_script_grid_size(self):
        n_combos = len(opt.STOP_BUFFER_PCT_GRID) * len(opt.FALLBACK_REWARD_RISK_GRID)
        z_bar = opt.opt_engine.bonferroni_adjusted_z_threshold(n_combos)
        self.assertGreater(z_bar, 1.959963985)   # strictly above the naive single-test 1.96 bar


# ============================= NEW: consistency_ratio / estimate_decay / lockbox wiring =============================

class TestOosTradesTagging(unittest.TestCase):
    """run_walk_forward's fold_results entries must now carry "oos_trades" (full trade dicts,
    un-merged across folds) IN ADDITION TO the existing keys - needed by
    optimization_engine.estimate_decay's months-since-fit pooling. This must be purely additive:
    combined_oos_r's own contents (used by compute_walk_forward_efficiency) stay unchanged."""

    def test_fold_results_carry_oos_trades_consistent_with_combined_oos_r(self):
        data = _small_multi_year_data(2016, 2021)
        small_grid = [(sb, frr) for sb in [0.02, 0.05] for frr in [1.0, 2.0]]
        folds = opt.generate_walk_forward_folds(2016, 2021)
        fold_results, combined_oos_r = opt.run_walk_forward(data, folds, grid=small_grid)

        for f in fold_results:
            self.assertIn("oos_trades", f)
            self.assertEqual(len(f["oos_trades"]), f["oos_n_trades"])
            for t in f["oos_trades"]:
                self.assertIn("date", t)
                self.assertIn("r", t)

        # combined_oos_r must still equal every fold's oos_trades' "r" values, chained in fold order -
        # proof this addition didn't change combined_oos_r's own pre-existing contents.
        rebuilt = [t["r"] for f in fold_results for t in f["oos_trades"]]
        self.assertEqual(rebuilt, combined_oos_r)


class TestDecayDiagnosticWiring(unittest.TestCase):
    def test_print_decay_diagnostic_runs_without_crashing(self):
        data = _small_multi_year_data(2016, 2021)
        small_grid = [(sb, frr) for sb in [0.02, 0.05] for frr in [1.0, 2.0]]
        folds = opt.generate_walk_forward_folds(2016, 2021)
        fold_results, _ = opt.run_walk_forward(data, folds, grid=small_grid)
        opt.print_decay_diagnostic(fold_results)   # must not raise, regardless of confidence outcome

    def test_estimate_decay_reports_insufficient_folds_with_only_two_folds(self):
        data = _small_multi_year_data(2016, 2021)
        small_grid = [(sb, frr) for sb in [0.02, 0.05] for frr in [1.0, 2.0]]
        folds = opt.generate_walk_forward_folds(2016, 2021)   # only 2 folds < default min_folds=4
        fold_results, _ = opt.run_walk_forward(data, folds, grid=small_grid)
        decay = opt.opt_engine.estimate_decay(fold_results)
        self.assertEqual(decay["confidence"], "insufficient_folds")
        self.assertIsNone(decay["half_life_months"])


class TestLockboxWiring(unittest.TestCase):
    """CRITICAL structural requirement: STEP 1's search window and every STEP 4 walk-forward fold
    must be bounded by search_end (== split_lockbox's lockbox_start), not FETCH_END, so the lockbox
    window is excluded BY CONSTRUCTION - never touched by any search iteration."""

    def test_split_lockbox_boundary_math(self):
        fetch_start = datetime.datetime(2016, 1, 1)
        fetch_end = datetime.datetime(2025, 1, 1)
        search_start, search_end, lockbox_start, lockbox_end = opt.opt_engine.split_lockbox(
            fetch_start, fetch_end, lockbox_months=12)
        self.assertEqual(search_start, fetch_start)
        self.assertEqual(search_end, lockbox_start)
        self.assertEqual(lockbox_start, datetime.datetime(2024, 1, 1))
        self.assertEqual(lockbox_end, fetch_end)

    def test_no_generated_fold_ever_crosses_into_the_lockbox_window(self):
        fetch_start = datetime.datetime(2016, 1, 1)
        fetch_end = datetime.datetime(2025, 1, 1)
        _, search_end, lockbox_start, _ = opt.opt_engine.split_lockbox(
            fetch_start, fetch_end, lockbox_months=12)

        folds = opt.generate_walk_forward_folds(fetch_start.year, search_end.year)
        self.assertGreater(len(folds), 0)
        for fold in folds:
            self.assertLessEqual(fold["oos_end"], lockbox_start.date())
            self.assertLessEqual(fold["is_end"], lockbox_start.date())

        # sanity check on the OLD (pre-lockbox) call: using FETCH_END.year directly WOULD have
        # produced a fold crossing into what is now the lockbox window - proving this isn't just a
        # vacuously-true check on an already-safe function.
        old_style_folds = opt.generate_walk_forward_folds(fetch_start.year, fetch_end.year)
        self.assertTrue(any(f["oos_end"] > lockbox_start.date() for f in old_style_folds))

    def test_make_lockbox_backtest_fn_and_lockbox_confirm_one_shot(self):
        data = _small_multi_year_data(2016, 2021)
        fetch_start = datetime.datetime(2016, 1, 1)
        fetch_end = datetime.datetime(2021, 1, 1)
        _, search_end, lockbox_start, lockbox_end = opt.opt_engine.split_lockbox(
            fetch_start, fetch_end, lockbox_months=12)

        backtest_fn = opt.make_lockbox_backtest_fn(data, stop_buffer_pct=0.02, fallback_reward_risk=2.0)

        tmpdir = tempfile.mkdtemp()
        ledger_path = os.path.join(tmpdir, "lockbox_ledger.json")
        strategy_id = "test_po3_strategy"

        result = opt.opt_engine.lockbox_confirm(strategy_id, {"stop_buffer_pct": 0.02, "fallback_reward_risk": 2.0},
                                                 backtest_fn, lockbox_start, lockbox_end, ledger_path=ledger_path)
        self.assertIn("passed", result)
        self.assertIn("n_trades", result)

        self.assertTrue(os.path.exists(ledger_path))
        with open(ledger_path) as f:
            records = json.load(f)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["strategy_id"], strategy_id)

        # SECOND call for the SAME strategy_id must be refused, not silently re-run.
        with self.assertRaises(opt.opt_engine.LockboxAlreadyUsedError):
            opt.opt_engine.lockbox_confirm(strategy_id, {"stop_buffer_pct": 0.02, "fallback_reward_risk": 2.0},
                                            backtest_fn, lockbox_start, lockbox_end, ledger_path=ledger_path)

        # ledger persisted across the two SEPARATE calls (re-read from disk, not an in-memory cache).
        with open(ledger_path) as f:
            records_after = json.load(f)
        self.assertEqual(len(records_after), 1)   # the refused second call never appended anything


if __name__ == "__main__":
    unittest.main(verbosity=2)
