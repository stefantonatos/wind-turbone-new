# Unit tests + one synthetic end-to-end smoke run for
# orb_indices_optimization_and_ml.py, covering the optimization_engine
# retrofit (STEP 1-4 + optional WIDE_SEARCH) added in this upgrade pass.
#
# Covers, per this project's rigor conventions (see
# test_ict_po3_forex_dukascopy_optimization.py for the reference this file's
# structure follows):
#   1. Core run_backtest mechanics on tiny synthetic single/multi-day
#      fixtures (LONG/SHORT, TP/SL/FLAT), split_before/split_after window
#      slicing, and the NEW min_range_atr_mult/max_range_atr_mult/
#      impulse_range_mult/rvol_mult override kwargs actually changing which
#      trades get filtered (the concrete new behavior WIDE_SEARCH depends
#      on).
#   2. REGRESSION TEST: run_param_search's default path (SEARCH_METHOD=
#      "grid", OBJECTIVE="total_r") reproduces the OLD hand-written
#      run_grid_search's exact per-cell numbers and best-combo choice on
#      the same synthetic data - proving the optimization_engine retrofit
#      did not change default behavior.
#   3. Monte Carlo resampling (bootstrap/shuffle) - same checks as the PO3
#      companion test file.
#   4. Walk-forward fold boundary generator - same 6-fold 2016-2025
#      boundaries as PO3/Rauf (this file shares their FETCH_START/
#      FETCH_END convention).
#   5. Cluster / neighbor-plateau analysis, exercised directly against
#      hand-built cell dicts (no backtest needed for this layer).
#   6. WIDE_SEARCH wiring: run_param_search over a multi-dimensional
#      param_grid (Bayesian and genetic) runs end-to-end without crashing,
#      evaluates fewer than the full cross product, and the
#      _resolve_wide_search_method() guard never allows "grid".
#   7. ONE small synthetic end-to-end smoke run (narrow grid -> Monte
#      Carlo -> cluster -> walk-forward) confirming the full pipeline runs
#      without crashing, plus a WIDE_SEARCH smoke run. Deliberately NOT a
#      real Dukascopy download - this project's convention is to validate
#      correctness via unit tests plus a synthetic smoke run.
#   8. LOCKBOX WIRING: research/optimization_engine.py's split_lockbox()/
#      lockbox_confirm() landed in a parallel commit partway through this
#      upgrade and were wired into the WIDE_SEARCH path per the task spec -
#      TestLockboxWiring checks split_lockbox() produces the exact search/
#      lockbox boundaries this file's own module-level FETCH_START/
#      FETCH_END/LOCKBOX_MONTHS config implies, and exercises
#      lockbox_confirm() end-to-end against ORB's own eval_fn (via
#      _make_orb_eval_fn) on synthetic data with an isolated tempfile
#      ledger - not just re-testing optimization_engine.py's own unit
#      tests, but confirming THIS file's wiring of them.
#
# Run with:  python -m pytest research/test_orb_indices_optimization_and_ml.py -v
# or:        python research/test_orb_indices_optimization_and_ml.py

import datetime
import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "orb_indices_optimization_and_ml.py")
_spec = importlib.util.spec_from_file_location("orb_indices_optimization_and_ml", _MODULE_PATH)
orb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(orb)


# ============================= synthetic data helpers =============================

_SESSION_START = pd.Timestamp("09:30").time()
_TZ = "UTC"


def _make_day_index(date_, periods=96):
    """96 5-min bars starting at 09:30 UTC - covers the opening range, the full
    ENTRY_WINDOW_MINUTES=180 entry window, and the full SESSION_HOLD_HOURS=8 hold (480 minutes =
    96 bars)."""
    return pd.date_range(start=pd.Timestamp(date_.year, date_.month, date_.day, 9, 30, tz=_TZ),
                          periods=periods, freq="5min")


def _build_synthetic_df(dates, base_price=100.0, overrides=None, periods=96):
    """Builds a synthetic 5-min OHLCV DataFrame across the given list of dates. Every bar is flat
    at base_price (volume 0) unless overrides[date]["HH:MM"] = (o, h, l, c[, v]) says otherwise."""
    overrides = overrides or {}
    idx_all, rows = [], []
    for d in dates:
        for ts in _make_day_index(d, periods=periods):
            tod_str = ts.strftime("%H:%M")
            ov = overrides.get(d, {}).get(tod_str)
            if ov is not None:
                o, h, l, c = ov[:4]
                v = ov[4] if len(ov) > 4 else 0
            else:
                o = h = l = c = base_price
                v = 0
            idx_all.append(ts)
            rows.append({"Open": o, "High": h, "Low": l, "Close": c, "volume": v})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx_all))


_DAY1 = datetime.date(2019, 1, 7)   # Monday
_DAY2 = datetime.date(2019, 1, 8)
_DAY3 = datetime.date(2019, 1, 9)

_RANGE_MINUTES = 10
_REWARD_RISK = 2.0


def _long_breakout_tp():
    """Range built at 09:30/09:35 (flat 100.0), breakout LONG at 09:40 (close=101.0, well past
    range_high=100.0 + tiny buffer), then a bar that hits the TP target."""
    return {
        "09:40": (101.0, 101.0, 100.5, 101.0),
        "09:45": (105.0, 106.0, 104.5, 105.0),   # sl_distance = max(range_size=0, 0.05%*101) tiny;
                                                   # target is close, this bar's high easily clears it
    }


def _long_breakout_sl():
    return {
        "09:40": (101.0, 101.0, 100.5, 101.0),
        "09:45": (95.0, 95.5, 90.0, 95.0),   # low=90 well below entry - hits stop
    }


def _build_three_day_fixture():
    overrides = {
        _DAY1: _long_breakout_tp(),
        _DAY2: _long_breakout_sl(),
        _DAY3: _long_breakout_tp(),
    }
    df = _build_synthetic_df([_DAY1, _DAY2, _DAY3], overrides=overrides)
    return df, orb.precompute_indicators(df)


class TestRunBacktestMechanics(unittest.TestCase):
    """These fixtures deliberately give the opening range itself ZERO width (flat O=H=L=C during
    the range-building bars) so the LONG breakout is driven purely by ENTRY_BUFFER_PCT against a
    zero-width range - which makes these tests sensitive to RANGE_ATR_FILTER once real ATR history
    accumulates past day 1 (a zero-width range is always "too small" relative to any positive ATR
    reading). That is real, correct run_backtest behavior, not a bug - these tests use
    apply_filters=False throughout because their purpose is entry/exit/window-slicing mechanics,
    not filter behavior (see TestFilterThresholdWiring below for dedicated, filter-aware
    fixtures)."""

    def setUp(self):
        self.df, self.ind = _build_three_day_fixture()

    def test_full_range_produces_expected_trades(self):
        trades = orb.run_backtest(self.df, self.ind, _TZ, _SESSION_START, _RANGE_MINUTES, _REWARD_RISK,
                                   apply_filters=False)
        self.assertEqual(len(trades), 3)
        dates = [t["date"] for t in trades]
        self.assertEqual(dates, [_DAY1, _DAY2, _DAY3])
        self.assertEqual(trades[0]["outcome"], "TP")
        self.assertAlmostEqual(trades[0]["r"], _REWARD_RISK)
        self.assertEqual(trades[1]["outcome"], "SL")
        self.assertAlmostEqual(trades[1]["r"], -1.0)
        self.assertEqual(trades[2]["outcome"], "TP")
        self.assertAlmostEqual(trades[2]["r"], _REWARD_RISK)

    def test_split_before_after_window_slicing(self):
        # split_after=DAY2, split_before=DAY3 (exclusive) -> only DAY2's trade should appear -
        # exactly the two-sided [split_after, split_before) window convention documented in
        # run_backtest's own docstring.
        windowed = orb.run_backtest(self.df, self.ind, _TZ, _SESSION_START, _RANGE_MINUTES, _REWARD_RISK,
                                     apply_filters=False, split_before=_DAY3, split_after=_DAY2)
        self.assertEqual(len(windowed), 1)
        self.assertEqual(windowed[0]["date"], _DAY2)
        self.assertEqual(windowed[0]["outcome"], "SL")

    def test_window_does_not_corrupt_range_computation(self):
        full_trades = orb.run_backtest(self.df, self.ind, _TZ, _SESSION_START, _RANGE_MINUTES, _REWARD_RISK,
                                        apply_filters=False)
        full_filtered = [t for t in full_trades if _DAY2 <= t["date"] < _DAY3]

        windowed = orb.run_backtest(self.df, self.ind, _TZ, _SESSION_START, _RANGE_MINUTES, _REWARD_RISK,
                                     apply_filters=False, split_before=_DAY3, split_after=_DAY2)

        self.assertEqual(len(full_filtered), len(windowed))
        for a, b in zip(full_filtered, windowed):
            self.assertEqual(a["date"], b["date"])
            self.assertEqual(a["outcome"], b["outcome"])
            self.assertAlmostEqual(a["r"], b["r"])

    def test_empty_window_produces_no_trades(self):
        no_trades = orb.run_backtest(self.df, self.ind, _TZ, _SESSION_START, _RANGE_MINUTES, _REWARD_RISK,
                                      split_after=datetime.date(2018, 1, 1), split_before=datetime.date(2018, 6, 1))
        self.assertEqual(no_trades, [])

    def test_filter_override_kwargs_default_to_module_constants(self):
        """With every override left at its default (None), run_backtest's behavior must be
        byte-for-byte identical to calling it with no overrides at all - the core regression-safety
        claim behind adding these kwargs in this upgrade."""
        baseline = orb.run_backtest(self.df, self.ind, _TZ, _SESSION_START, _RANGE_MINUTES, _REWARD_RISK)
        explicit_none = orb.run_backtest(self.df, self.ind, _TZ, _SESSION_START, _RANGE_MINUTES, _REWARD_RISK,
                                          min_range_atr_mult=None, max_range_atr_mult=None,
                                          impulse_range_mult=None, rvol_mult=None)
        explicit_module_values = orb.run_backtest(
            self.df, self.ind, _TZ, _SESSION_START, _RANGE_MINUTES, _REWARD_RISK,
            min_range_atr_mult=orb.MIN_RANGE_ATR_MULT, max_range_atr_mult=orb.MAX_RANGE_ATR_MULT,
            impulse_range_mult=orb.IMPULSE_RANGE_MULT, rvol_mult=orb.RVOL_MULT)
        self.assertEqual(baseline, explicit_none)
        self.assertEqual(baseline, explicit_module_values)


class TestFilterThresholdWiring(unittest.TestCase):
    """Direct proof that the NEW min_range_atr_mult/max_range_atr_mult/impulse_range_mult/rvol_mult
    kwargs actually change which trades get taken - the concrete behavior WIDE_SEARCH's whole
    premise depends on. Builds a fixture with enough real bar history for ATR/rolling averages to
    seed (unlike the tiny 3-day fixture above, where those stay None/0 and every filter is a
    structural no-op), then hand-picks threshold values that flip a specific trade's fate."""

    def _build_seeded_fixture(self):
        # 30 trading days of small, oscillating (never perfectly flat) bars so ATR_LEN=14/
        # RANGE_AVG_LEN=20/VOLUME_AVG_LEN=20 all have real, non-None/non-zero values by the test
        # day - each day's range itself is small and identical, so ATR/range_avg/volume_avg
        # converge to a known, stable value we can reason about.
        dates = pd.bdate_range(datetime.date(2019, 1, 1), periods=31).date.tolist()
        history_dates, test_date = dates[:-1], dates[-1]
        overrides = {}
        for i, d in enumerate(history_dates):
            wobble = 0.2 if i % 2 == 0 else -0.2
            overrides[d] = {
                "09:30": (100.0, 100.3, 99.7, 100.0 + wobble),
                "09:35": (100.0, 100.3, 99.7, 100.0),
            }
        # test day: small range (range_size = 0.4, same as every history day), so whether it
        # passes the ATR-range filter depends entirely on min_range_atr_mult/max_range_atr_mult.
        overrides[test_date] = {
            "09:30": (100.0, 100.3, 99.7, 100.0),
            "09:35": (100.0, 100.3, 99.7, 100.0),
            "09:40": (100.5, 100.5, 100.3, 100.5),   # breakout above range_high=100.3
            "09:45": (102.0, 103.0, 101.5, 102.0),   # clears a small TP target
        }
        df = _build_synthetic_df(history_dates + [test_date], overrides=overrides)
        ind = orb.precompute_indicators(df)
        return df, ind, test_date

    def test_min_range_atr_mult_override_blocks_a_trade_default_would_take(self):
        df, ind, test_date = self._build_seeded_fixture()
        window_start, window_end = test_date, test_date + datetime.timedelta(days=1)

        loose = orb.run_backtest(df, ind, _TZ, _SESSION_START, _RANGE_MINUTES, _REWARD_RISK,
                                  split_after=window_start, split_before=window_end,
                                  min_range_atr_mult=0.01, max_range_atr_mult=100.0)
        self.assertEqual(len(loose), 1, "a very permissive ATR-range band should let the trade through")

        strict = orb.run_backtest(df, ind, _TZ, _SESSION_START, _RANGE_MINUTES, _REWARD_RISK,
                                   split_after=window_start, split_before=window_end,
                                   min_range_atr_mult=50.0, max_range_atr_mult=100.0)
        self.assertEqual(len(strict), 0, "an unreachable min ATR-range multiple should block the trade")

    def test_impulse_range_mult_override_changes_outcome(self):
        df, ind, test_date = self._build_seeded_fixture()
        window_start, window_end = test_date, test_date + datetime.timedelta(days=1)

        loose = orb.run_backtest(df, ind, _TZ, _SESSION_START, _RANGE_MINUTES, _REWARD_RISK,
                                  split_after=window_start, split_before=window_end,
                                  impulse_range_mult=0.01)
        strict = orb.run_backtest(df, ind, _TZ, _SESSION_START, _RANGE_MINUTES, _REWARD_RISK,
                                   split_after=window_start, split_before=window_end,
                                   impulse_range_mult=1000.0)
        self.assertGreaterEqual(len(loose), len(strict))
        self.assertEqual(len(strict), 0)

    def test_rvol_mult_override_changes_outcome(self):
        # give the test-day breakout bar real volume, and a real (nonzero) volume history, so
        # RVOL_MULT has something to actually threshold against.
        dates = pd.bdate_range(datetime.date(2019, 1, 1), periods=31).date.tolist()
        history_dates, test_date = dates[:-1], dates[-1]
        overrides = {}
        for d in history_dates:
            overrides[d] = {
                "09:30": (100.0, 100.3, 99.7, 100.0, 1000),
                "09:35": (100.0, 100.3, 99.7, 100.0, 1000),
            }
        overrides[test_date] = {
            "09:30": (100.0, 100.3, 99.7, 100.0, 1000),
            "09:35": (100.0, 100.3, 99.7, 100.0, 1000),
            "09:40": (100.5, 100.5, 100.3, 100.5, 1000),
            "09:45": (102.0, 103.0, 101.5, 102.0, 1000),
        }
        df = _build_synthetic_df(history_dates + [test_date], overrides=overrides)
        ind = orb.precompute_indicators(df)
        window_start, window_end = test_date, test_date + datetime.timedelta(days=1)

        # min/max_range_atr_mult wide open here - this test isolates RVOL_MULT specifically, the
        # ATR-range filter has its own dedicated test above.
        loose = orb.run_backtest(df, ind, _TZ, _SESSION_START, _RANGE_MINUTES, _REWARD_RISK,
                                  split_after=window_start, split_before=window_end, rvol_mult=0.01,
                                  min_range_atr_mult=0.001, max_range_atr_mult=1000.0)
        strict = orb.run_backtest(df, ind, _TZ, _SESSION_START, _RANGE_MINUTES, _REWARD_RISK,
                                   split_after=window_start, split_before=window_end, rvol_mult=1000.0,
                                   min_range_atr_mult=0.001, max_range_atr_mult=1000.0)
        self.assertEqual(len(loose), 1)
        self.assertEqual(len(strict), 0)


# ============================= optimization_engine retrofit =============================

def _small_multi_year_data(start_year=2016, end_year_exclusive=2019, label="SYN"):
    """A few years of trading-day bars, alternating TP/SL like the three-day fixture above but
    across many days - enough for the walk-forward fold machinery (multi-year spans) to have real
    trades in every fold without needing real Dukascopy data."""
    dates = pd.bdate_range(datetime.date(start_year, 1, 1),
                            datetime.date(end_year_exclusive, 1, 1), inclusive="left")
    overrides = {}
    for i, ts in enumerate(dates):
        d = ts.date()
        overrides[d] = _long_breakout_tp() if i % 2 == 0 else _long_breakout_sl()
    df = _build_synthetic_df([ts.date() for ts in dates], overrides=overrides)
    return {label: (df, orb.precompute_indicators(df), _TZ, _SESSION_START)}


_SMALL_GRID = {"range_minutes": [10, 15], "reward_risk": [1.0, 2.0]}
_SMALL_GRID_TUPLES = [(rm, rr) for rm in _SMALL_GRID["range_minutes"] for rr in _SMALL_GRID["reward_risk"]]


class TestSearchEngineRetrofit(unittest.TestCase):
    def test_run_param_search_grid_default_matches_old_run_grid_search_exactly(self):
        """REGRESSION TEST: run_param_search's default (SEARCH_METHOD="grid", OBJECTIVE="total_r")
        must reproduce the OLD hand-written run_grid_search's exact per-cell numbers and the exact
        same best combo, on the same synthetic data - proving this refactor did not change default
        behavior."""
        data = _small_multi_year_data()

        old_results = orb.run_grid_search(data, grid=_SMALL_GRID_TUPLES, desc="old-style")
        new_results, search_result = orb.run_param_search(data, param_grid=_SMALL_GRID, method="grid",
                                                            objective="total_r", desc="new-style")

        self.assertEqual(len(old_results), len(new_results))
        old_by_params = {(r["range_minutes"], r["reward_risk"]): r for r in old_results}
        new_by_params = {(r["range_minutes"], r["reward_risk"]): r for r in new_results}
        self.assertEqual(set(old_by_params), set(new_by_params))
        for key, old in old_by_params.items():
            new = new_by_params[key]
            self.assertEqual(old["n_trades"], new["n_trades"])
            self.assertAlmostEqual(old["total_r"], new["total_r"], places=9)
            self.assertAlmostEqual(old["avg_r"], new["avg_r"], places=9)
            self.assertEqual(sorted(old["trades_r"]), sorted(new["trades_r"]))

        old_best = max(old_results, key=lambda r: r["total_r"])
        new_best_params = search_result["best"]["params"]
        self.assertEqual(old_best["range_minutes"], new_best_params["range_minutes"])
        self.assertEqual(old_best["reward_risk"], new_best_params["reward_risk"])
        self.assertAlmostEqual(old_best["total_r"], search_result["best"]["score"], places=9)

    def test_search_method_and_objective_module_defaults_are_grid_and_total_r(self):
        self.assertEqual(orb.SEARCH_METHOD, "grid")
        self.assertEqual(orb.OBJECTIVE, "total_r")

    def test_wide_search_off_by_default(self):
        self.assertFalse(orb.WIDE_SEARCH)

    def test_objective_options_all_run_without_crashing(self):
        data = _small_multi_year_data()
        for objective_name in orb.opt_engine.OBJECTIVES:
            grid_results, search_result = orb.run_param_search(
                data, param_grid=_SMALL_GRID, method="grid", objective=objective_name, desc=objective_name)
            self.assertEqual(len(grid_results), 4)
            self.assertIsNotNone(search_result["best"])
            self.assertFalse(np.isnan(search_result["best"]["score"]))

    def test_walk_forward_default_still_matches_old_grid_search_per_fold(self):
        data = _small_multi_year_data(2016, 2021)
        folds = orb.generate_walk_forward_folds(2016, 2021)
        self.assertEqual(len(folds), 2)

        fold_results, combined_oos_r = orb.run_walk_forward(data, folds, param_grid=_SMALL_GRID)

        for f in fold_results:
            is_grid = orb.run_grid_search(data, split_after=f["is_start"], split_before=f["is_end"],
                                           grid=_SMALL_GRID_TUPLES, desc="check")
            expected_best = max(is_grid, key=lambda r: r["total_r"])
            self.assertEqual(f["best_params"]["range_minutes"], expected_best["range_minutes"])
            self.assertEqual(f["best_params"]["reward_risk"], expected_best["reward_risk"])
            self.assertAlmostEqual(f["is_total_r"], expected_best["total_r"], places=9)


class TestWideSearchWiring(unittest.TestCase):
    _WIDE_SMALL_GRID = {
        "range_minutes": [10, 15],
        "reward_risk": [1.0, 2.0],
        "min_range_atr_mult": [0.3, 0.5],
        "max_range_atr_mult": [2.0, 3.0],
        "impulse_range_mult": [1.0, 1.3],
        "rvol_mult": [1.0, 1.3],
    }

    def test_resolve_wide_search_method_never_returns_grid(self):
        self.assertIn(orb._resolve_wide_search_method(), ("bayesian", "genetic"))
        self.assertEqual(orb.WIDE_SEARCH_METHOD, "bayesian")   # module default, unchanged by this test

    def test_genetic_search_over_wide_grid_end_to_end(self):
        data = _small_multi_year_data()
        grid_results, search_result = orb.run_param_search(
            data, param_grid=self._WIDE_SMALL_GRID, method="genetic", objective="total_r",
            population_size=4, generations=3, seed=1)
        self.assertEqual(search_result["method"], "genetic")
        self.assertGreater(len(grid_results), 0)
        self.assertIsNotNone(search_result["best"])
        full_cross_product = 2 ** 6
        self.assertLessEqual(search_result["n_evals"], full_cross_product)
        for cell in grid_results:
            self.assertIn("min_range_atr_mult", cell["params"])
            self.assertIn("rvol_mult", cell["params"])

    def test_bayesian_search_over_wide_grid_end_to_end_or_falls_back_cleanly(self):
        data = _small_multi_year_data()
        grid_results, search_result = orb.run_param_search(
            data, param_grid=self._WIDE_SMALL_GRID, method="bayesian", objective="total_r",
            n_trials=6, seed=1)
        self.assertIn(search_result["method"], ("bayesian", "grid"))
        self.assertGreater(len(grid_results), 0)
        self.assertIsNotNone(search_result["best"])

    def test_wide_search_evaluates_fewer_than_full_cross_product(self):
        """The whole point of WIDE_SEARCH using Bayesian/genetic instead of grid: a fixed,
        small evaluation budget instead of an exponentially-growing exhaustive sweep."""
        data = _small_multi_year_data()
        full_cross_product = 1
        for v in self._WIDE_SMALL_GRID.values():
            full_cross_product *= len(v)
        self.assertEqual(full_cross_product, 64)

        _, search_result = orb.run_param_search(
            data, param_grid=self._WIDE_SMALL_GRID, method="genetic", objective="total_r",
            population_size=4, generations=3, seed=2)
        self.assertLess(search_result["n_evals"], full_cross_product)


# ============================= lockbox wiring (WIDE_SEARCH path) =============================

class TestLockboxWiring(unittest.TestCase):
    """research/optimization_engine.py's split_lockbox()/lockbox_confirm() landed in a parallel
    commit partway through this upgrade and were wired into the WIDE_SEARCH path per the task
    spec - see the module docstring's "LOCKBOX" section in orb_indices_optimization_and_ml.py.
    These tests confirm THIS file's wiring (config + _make_orb_eval_fn as lockbox_confirm's
    backtest_fn), not optimization_engine.py's own unit-level correctness (see
    test_optimization_engine.py's TestSplitLockbox/TestLockboxConfirm for that)."""

    def test_split_lockbox_matches_this_files_actual_fetch_range_and_config(self):
        """The exact boundaries documented in this file's own WIDE SEARCH / LOCKBOX config
        comment - not assumed, checked directly against orb's real FETCH_START/FETCH_END/
        LOCKBOX_MONTHS."""
        search_start, search_end, lockbox_start, lockbox_end = orb.opt_engine.split_lockbox(
            orb.FETCH_START, orb.FETCH_END, orb.LOCKBOX_MONTHS)
        self.assertEqual(search_start, datetime.datetime(2016, 1, 1))
        self.assertEqual(search_end, datetime.datetime(2024, 1, 1))
        self.assertEqual(lockbox_start, datetime.datetime(2024, 1, 1))
        self.assertEqual(lockbox_end, datetime.datetime(2025, 1, 1))
        self.assertEqual(search_end, lockbox_start)   # adjacency guarantee

    def test_wide_search_strategy_id_is_a_stable_string(self):
        self.assertIsInstance(orb.WIDE_SEARCH_STRATEGY_ID, str)
        self.assertGreater(len(orb.WIDE_SEARCH_STRATEGY_ID), 0)

    def test_lockbox_confirm_runs_end_to_end_against_orb_eval_fn(self):
        """Builds a small multi-year synthetic dataset spanning both a "search" window and a
        "lockbox" window, then calls opt_engine.lockbox_confirm with _make_orb_eval_fn-based
        backtest_fn - exactly the pattern main()'s WIDE_SEARCH block uses - against an isolated
        tempfile ledger (never the project's real, shared lockbox_ledger.json)."""
        data = _small_multi_year_data(2016, 2020)   # search side would be e.g. 2016-2019, lockbox 2019-2020
        lockbox_start = datetime.date(2019, 1, 1)
        lockbox_end = datetime.date(2020, 1, 1)
        # range_minutes=15 (not 10) deliberately: with this fixture's breakout bar at 09:40,
        # RANGE_MINUTES=10 makes 09:40 the ENTRY bar (a real, non-zero-width entry against a
        # flat/zero-width opening range), which is exactly the shape RANGE_ATR_FILTER is designed
        # to reject once real ATR history accumulates - genuinely 0 trades after the first day, not
        # a fixture bug (see TestRunBacktestMechanics' docstring for the same behavior). RANGE_
        # MINUTES=15 instead makes 09:40 part of the RANGE itself, giving every day's opening range
        # real width and avoiding that filter - this test is about lockbox wiring, not about
        # re-deriving that filter behavior.
        final_params = {"range_minutes": 15, "reward_risk": 2.0}

        def backtest_fn(window_start, window_end):
            eval_fn = orb._make_orb_eval_fn(data, window_start=window_start, window_end=window_end)
            return eval_fn(final_params)

        tmpdir = tempfile.mkdtemp()
        ledger_path = os.path.join(tmpdir, "lockbox_ledger.json")

        result = orb.opt_engine.lockbox_confirm(
            "test_orb_lockbox_strategy", final_params, backtest_fn, lockbox_start, lockbox_end,
            ledger_path=ledger_path)
        self.assertIn("passed", result)
        self.assertIn("n_trades", result)
        self.assertGreater(result["n_trades"], 0)   # the alternating TP/SL fixture always has real trades

        # one-shot enforcement: a second call for the SAME strategy_id must refuse, not re-run.
        with self.assertRaises(orb.opt_engine.LockboxAlreadyUsedError):
            orb.opt_engine.lockbox_confirm(
                "test_orb_lockbox_strategy", final_params, backtest_fn, lockbox_start, lockbox_end,
                ledger_path=ledger_path)

    def test_lockbox_backtest_fn_only_sees_its_own_window(self):
        """The backtest_fn main() builds for lockbox_confirm must respect window_start/window_end
        exactly like every other windowed call in this file - a lockbox that accidentally leaked
        search-side data would defeat the entire point."""
        data = _small_multi_year_data(2016, 2020)
        final_params = {"range_minutes": 15, "reward_risk": 2.0}   # see the comment in the lockbox
                                                                     # end-to-end test above for why 15

        def backtest_fn(window_start, window_end):
            eval_fn = orb._make_orb_eval_fn(data, window_start=window_start, window_end=window_end)
            return eval_fn(final_params)

        lockbox_only = backtest_fn(datetime.date(2019, 1, 1), datetime.date(2020, 1, 1))
        full_range = backtest_fn(None, None)
        self.assertGreater(len(lockbox_only), 0)
        self.assertLess(len(lockbox_only), len(full_range))
        for t in lockbox_only:
            self.assertGreaterEqual(t["date"], datetime.date(2019, 1, 1))
            self.assertLess(t["date"], datetime.date(2020, 1, 1))


# ============================= Monte Carlo =============================

class TestMonteCarlo(unittest.TestCase):
    def test_bootstrap_of_constant_values_converges_exactly(self):
        r = [0.5] * 40
        result = orb.monte_carlo_bootstrap(r, n_iter=500, rng=np.random.default_rng(0))
        self.assertTrue(np.allclose(result["total_r"], 0.5 * 40))
        self.assertTrue(np.allclose(result["max_drawdown"], 0.0))

    def test_bootstrap_of_constant_negative_values(self):
        n = 25
        r = [-0.3] * n
        result = orb.monte_carlo_bootstrap(r, n_iter=300, rng=np.random.default_rng(1))
        self.assertTrue(np.allclose(result["total_r"], -0.3 * n))
        self.assertTrue(np.allclose(result["max_drawdown"], 0.3 * (n - 1)))

    def test_bootstrap_produces_sane_variance_and_percentile_ordering(self):
        r = [-1, -1, 2, -1, 2, -1, 2, -1, -1, 2]
        result = orb.monte_carlo_bootstrap(r, n_iter=2000, rng=np.random.default_rng(2))
        self.assertGreater(np.std(result["total_r"]), 0)
        p5, p50, p95 = (np.percentile(result["total_r"], q) for q in (5, 50, 95))
        self.assertLessEqual(p5, p50)
        self.assertLessEqual(p50, p95)

    def test_shuffle_preserves_total_r_but_drawdown_varies(self):
        r = [-1, -1, 2, -1, 2, -1, -1, 2, 2, -1]
        result = orb.monte_carlo_shuffle(r, n_iter=500, rng=np.random.default_rng(3))
        self.assertTrue(np.allclose(result["total_r"], sum(r)))
        self.assertGreater(np.std(result["max_drawdown"]), 0)

    def test_empty_trade_list_handled(self):
        boot = orb.monte_carlo_bootstrap([], n_iter=50)
        shuf = orb.monte_carlo_shuffle([], n_iter=50)
        self.assertEqual(len(boot["total_r"]), 0)
        self.assertEqual(len(shuf["total_r"]), 0)
        self.assertIsNone(orb.summarize_mc(boot))
        self.assertIsNone(orb.summarize_mc(shuf))

    def test_run_monte_carlo_all_cells_covers_every_cell(self):
        data = _small_multi_year_data()
        grid_results, _ = orb.run_param_search(data, param_grid=_SMALL_GRID)
        mc_results = orb.run_monte_carlo_all_cells(grid_results, n_iter=100)
        self.assertEqual(len(mc_results), len(grid_results))
        for cell in mc_results:
            self.assertIsNotNone(cell["bootstrap"])
            self.assertIsNotNone(cell["shuffle"])


# ============================= walk-forward fold boundaries =============================

class TestWalkForwardFoldBoundaries(unittest.TestCase):
    def test_exact_six_rolling_folds(self):
        folds = orb.generate_walk_forward_folds(2016, 2025)
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

        for f in folds:
            self.assertEqual(f["oos_start"], f["is_end"])
        for prev, nxt in zip(folds, folds[1:]):
            self.assertEqual(nxt["is_start"].year, prev["is_start"].year + 1)

    def test_no_fold_extends_past_fetch_end(self):
        folds = orb.generate_walk_forward_folds(2016, 2025)
        for f in folds:
            self.assertLessEqual(f["oos_end"], datetime.date(2025, 1, 1))

    def test_actual_fetch_range_produces_same_six_folds(self):
        """This file's own FETCH_START/FETCH_END (2016/2025) must produce the same 6-fold
        convention documented in this file's header and module docstrings - not assumed, checked
        directly against the module's real config."""
        folds = orb.generate_walk_forward_folds(orb.FETCH_START.year, orb.FETCH_END.year)
        self.assertEqual(len(folds), 6)

    def test_custom_span_produces_two_folds(self):
        folds = orb.generate_walk_forward_folds(2016, 2021)
        self.assertEqual(len(folds), 2)
        self.assertEqual(folds[0]["is_start"], datetime.date(2016, 1, 1))
        self.assertEqual(folds[0]["oos_end"], datetime.date(2020, 1, 1))
        self.assertEqual(folds[1]["is_start"], datetime.date(2017, 1, 1))
        self.assertEqual(folds[1]["oos_end"], datetime.date(2021, 1, 1))


# ============================= cluster / plateau analysis =============================

def _make_cell(range_minutes, reward_risk, avg_r, n_trades=100, **extra_params):
    total_r = avg_r * n_trades
    params = {"range_minutes": range_minutes, "reward_risk": reward_risk, **extra_params}
    return {
        "params": params, "range_minutes": range_minutes, "reward_risk": reward_risk,
        "total_r": total_r, "n_trades": n_trades, "avg_r": avg_r,
        "trades_r": [avg_r] * n_trades,
    }


class TestClusterAnalysis(unittest.TestCase):
    def test_neighbor_check_detects_plateau(self):
        # A 3x3 grid where the whole neighborhood around the peak is decent (a plateau).
        grid_results = []
        rm_grid, rr_grid = [10, 15, 20], [1.0, 1.5, 2.0]
        base_avg_r = {(10, 1.0): 0.05, (10, 1.5): 0.06, (10, 2.0): 0.05,
                       (15, 1.0): 0.07, (15, 1.5): 0.10, (15, 2.0): 0.08,
                       (20, 1.0): 0.05, (20, 1.5): 0.07, (20, 2.0): 0.06}
        for (rm, rr), avg_r in base_avg_r.items():
            grid_results.append(_make_cell(rm, rr, avg_r))
        result = orb.neighbor_plateau_check(grid_results, rm_grid=rm_grid, rr_grid=rr_grid)
        self.assertEqual(result["verdict"], "PLATEAU")
        self.assertEqual(result["best_range_minutes"], 15)
        self.assertEqual(result["best_reward_risk"], 1.5)

    def test_neighbor_check_detects_isolated_spike(self):
        grid_results = []
        rm_grid, rr_grid = [10, 15, 20], [1.0, 1.5, 2.0]
        base_avg_r = {(10, 1.0): -0.05, (10, 1.5): -0.04, (10, 2.0): -0.05,
                       (15, 1.0): -0.04, (15, 1.5): 0.50, (15, 2.0): -0.03,
                       (20, 1.0): -0.05, (20, 1.5): -0.04, (20, 2.0): -0.06}
        for (rm, rr), avg_r in base_avg_r.items():
            grid_results.append(_make_cell(rm, rr, avg_r))
        result = orb.neighbor_plateau_check(grid_results, rm_grid=rm_grid, rr_grid=rr_grid)
        self.assertEqual(result["verdict"], "ISOLATED SPIKE / overfit warning")

    def test_sklearn_cluster_analysis_runs_on_narrow_grid(self):
        grid_results = [_make_cell(rm, rr, 0.01 * (rm + rr)) for rm in [10, 15, 20] for rr in [1.0, 1.5, 2.0]]
        result = orb.sklearn_cluster_analysis(grid_results, param_names=("range_minutes", "reward_risk"))
        if result is not None:   # sklearn is present in this project's environment
            self.assertGreaterEqual(result["cluster_size"], 1)

    def test_sklearn_cluster_analysis_runs_on_wide_grid(self):
        grid_results = [
            _make_cell(rm, rr, 0.01 * (rm + rr), min_range_atr_mult=mn, max_range_atr_mult=mx)
            for rm in [10, 15] for rr in [1.0, 2.0] for mn in [0.3, 0.5] for mx in [2.0, 3.0]
        ]
        result = orb.sklearn_cluster_analysis(
            grid_results, param_names=("range_minutes", "reward_risk", "min_range_atr_mult", "max_range_atr_mult"))
        if result is not None:
            self.assertGreaterEqual(result["cluster_size"], 1)
            for member in result["cluster_members"]:
                self.assertIn("min_range_atr_mult", member)

    def test_print_cluster_analysis_skips_neighbor_check_for_wide_param_names(self):
        grid_results = [_make_cell(rm, rr, 0.01 * (rm + rr), min_range_atr_mult=0.5)
                         for rm in [10, 15] for rr in [1.0, 2.0]]
        neighbor_result, cluster_result = orb.print_cluster_analysis(
            grid_results, search_method="grid", param_names=("range_minutes", "reward_risk", "min_range_atr_mult"))
        self.assertIsNone(neighbor_result)

    def test_print_cluster_analysis_skips_neighbor_check_under_non_grid_search(self):
        grid_results = [_make_cell(rm, rr, 0.01 * (rm + rr)) for rm in [10, 15] for rr in [1.0, 2.0]]
        neighbor_result, cluster_result = orb.print_cluster_analysis(grid_results, search_method="genetic")
        self.assertIsNone(neighbor_result)

    def test_print_cluster_analysis_runs_neighbor_check_under_grid_search(self):
        grid_results = [_make_cell(rm, rr, 0.01 * (rm + rr)) for rm in [10, 15, 20] for rr in [1.0, 1.5, 2.0]]
        neighbor_result, cluster_result = orb.print_cluster_analysis(grid_results, search_method="grid")
        self.assertIsNotNone(neighbor_result)
        self.assertIn(neighbor_result["verdict"], ("PLATEAU", "ISOLATED SPIKE / overfit warning"))


# ============================= end-to-end smoke tests =============================

class TestSmokeEndToEnd(unittest.TestCase):
    """ONE small synthetic end-to-end run: small synthetic price series, small grid, 2 folds -
    confirms the full narrow-search pipeline (grid search -> Monte Carlo -> cluster analysis ->
    walk-forward) runs to completion without crashing. Not a claim about real-world performance."""

    def test_full_narrow_pipeline_runs_without_crashing(self):
        data = _small_multi_year_data(2016, 2021)   # -> exactly 2 walk-forward folds

        grid_results, search_result = orb.run_param_search(data, param_grid=_SMALL_GRID, desc="smoke test grid")
        self.assertEqual(len(grid_results), 4)
        for cell in grid_results:
            self.assertGreater(cell["n_trades"], 0)
            self.assertIsInstance(cell["avg_r"], float)

        mc_results = orb.run_monte_carlo_all_cells(grid_results, n_iter=200)
        self.assertEqual(len(mc_results), 4)

        neighbor_result, cluster_result = orb.print_cluster_analysis(grid_results, search_method="grid")
        self.assertIn(neighbor_result["verdict"], ("PLATEAU", "ISOLATED SPIKE / overfit warning"))

        folds = orb.generate_walk_forward_folds(2016, 2021)
        self.assertEqual(len(folds), 2)
        fold_results, combined_oos_r = orb.run_walk_forward(data, folds, param_grid=_SMALL_GRID)
        self.assertEqual(len(fold_results), 2)
        wfe_stats = orb.compute_walk_forward_efficiency(fold_results, combined_oos_r)
        self.assertIn("wfe", wfe_stats)

        print(f"\n[smoke test] grid best cell: {max(grid_results, key=lambda r: r['total_r'])}")
        print(f"[smoke test] fold results: {fold_results}")
        print(f"[smoke test] WFE stats: {wfe_stats}")

    def test_full_wide_search_pipeline_runs_without_crashing(self):
        """Same smoke-test spirit, but exercising the WIDE_SEARCH path end-to-end: genetic search
        over the wide param space -> Monte Carlo -> cluster analysis (no neighbor check) ->
        wide-search walk-forward, all with a tiny evaluation budget so the test stays fast."""
        data = _small_multi_year_data(2016, 2021)
        wide_grid = {
            "range_minutes": [10, 15], "reward_risk": [1.0, 2.0],
            "min_range_atr_mult": [0.3, 0.5], "max_range_atr_mult": [2.0, 3.0],
            "impulse_range_mult": [1.0, 1.3], "rvol_mult": [1.0, 1.3],
        }

        grid_results, search_result = orb.run_param_search(
            data, param_grid=wide_grid, method="genetic", desc="smoke wide search",
            population_size=4, generations=2, seed=7)
        self.assertGreater(len(grid_results), 0)
        self.assertEqual(search_result["method"], "genetic")

        mc_results = orb.run_monte_carlo_all_cells(grid_results, n_iter=100)
        self.assertEqual(len(mc_results), len(grid_results))

        neighbor_result, cluster_result = orb.print_cluster_analysis(
            grid_results, search_method="genetic", param_names=tuple(wide_grid.keys()))
        self.assertIsNone(neighbor_result)   # not applicable under a non-grid, >2-param search

        folds = orb.generate_walk_forward_folds(2016, 2021)
        fold_results, combined_oos_r = orb.run_walk_forward(
            data, folds, param_grid=wide_grid, method="genetic", population_size=4, generations=2)
        self.assertEqual(len(fold_results), len(folds))
        wfe_stats = orb.compute_walk_forward_efficiency(fold_results, combined_oos_r)
        self.assertIn("wfe", wfe_stats)

        # STEP W5 analog: lockbox confirmation of the search's winning combo, on an isolated
        # tempfile ledger - mirrors main()'s WIDE_SEARCH block end to end, including the
        # split_lockbox() call that determines the search/lockbox boundary.
        search_start, search_end, lockbox_start, lockbox_end = orb.opt_engine.split_lockbox(
            datetime.datetime(2016, 1, 1), datetime.datetime(2021, 1, 1), lockbox_months=12)
        final_params = search_result["best"]["params"]

        def backtest_fn(window_start, window_end):
            eval_fn = orb._make_orb_eval_fn(data, window_start=window_start, window_end=window_end)
            return eval_fn(final_params)

        tmpdir = tempfile.mkdtemp()
        ledger_path = os.path.join(tmpdir, "lockbox_ledger.json")
        lockbox_result = orb.opt_engine.lockbox_confirm(
            "smoke_test_wide_search_lockbox", final_params, backtest_fn,
            lockbox_start.date(), lockbox_end.date(), ledger_path=ledger_path)
        self.assertIn("passed", lockbox_result)
        print(f"\n[smoke test] lockbox result: {lockbox_result}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
