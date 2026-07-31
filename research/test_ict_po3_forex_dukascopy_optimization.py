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
import os
import sys
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
