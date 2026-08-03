# Unit tests + a synthetic end-to-end smoke run for dow_theory_swing_structure_dukascopy_
# optimization.py.
#
# Covers, per this project's rigor conventions:
#   1. run_dow_theory_backtest (this script's own windowed rebuild of the base script's
#      backtest_instrument) reproduces the base script EXACTLY when given no window and the
#      base script's own default SWING_LEN - proving the rebuild didn't silently diverge from
#      the strategy it's supposed to be optimizing.
#   2. The SWING_LEN override (setattr on the base module) always restores the base module's
#      original value afterward, even when the backtest raises - a permanent leak here would
#      silently corrupt every other test/run sharing this process.
#   3. Windowing actually restricts what gets PROCESSED (not just counted afterward) - a pivot
#      pair confirmed entirely before window_start must not count toward "2 confirmed pivots"
#      inside a later window that excludes it (no-lookahead-across-fold-boundaries).
#   4. eval_fn/lockbox backtest_fn wiring: instrument tagging, and the lockbox fn actually
#      restricts to [lockbox_start, lockbox_end).
#   5. The module's own run_smoke_test() (small + wide synthetic runs) completes without error
#      and reaches "SCORED" - the same role run_smoke_test() plays in the sibling Rauf/PO3
#      optimization scripts, run here as a real pytest case too, not just via `--smoke`.
#
# Run with:  python -m pytest research/test_dow_theory_swing_structure_dukascopy_optimization.py -v
# or:        python research/test_dow_theory_swing_structure_dukascopy_optimization.py

import datetime
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dow_theory_swing_structure_dukascopy_optimization as opt
import dow_theory_swing_structure_dukascopy_backtest as base
import optimization_engine as engine


# ============================= synthetic data helpers =============================

def _flat_days(n, start="2016-01-04"):
    days = pd.bdate_range(start=start, periods=n)
    return [(d, opt._FLAT_DAY) for d in days]


def _override_day(days_list, index, bars):
    new_list = list(days_list)
    d, _ = new_list[index]
    new_list[index] = (d, bars)
    return new_list


def _build_5min_df(day_specs):
    rows, idx = [], []
    for date_, (o, h, l, c) in day_specs:
        ts = pd.Timestamp(date_.year, date_.month, date_.day, 9, 30, tz="America/New_York")
        idx.append(ts)
        rows.append((o, h, l, c))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=pd.DatetimeIndex(idx))


PIVOT_LOW1_IDX, LOW1 = 25, 1.0900
PIVOT_HIGH1_IDX, HIGH1 = 30, 1.1100
PIVOT_LOW2_IDX, LOW2 = 60, 1.0950
PIVOT_HIGH2_IDX, HIGH2 = 65, 1.1150


def _build_uptrend_days(n_total=150, start="2016-01-04"):
    days = _flat_days(n_total, start=start)
    days = _override_day(days, PIVOT_LOW1_IDX, (1.1000, 1.1002, LOW1, 1.0950))
    days = _override_day(days, PIVOT_HIGH1_IDX, (1.1000, HIGH1, 1.0998, 1.1050))
    days = _override_day(days, PIVOT_LOW2_IDX, (1.1000, 1.1002, LOW2, 1.0980))
    days = _override_day(days, PIVOT_HIGH2_IDX, (1.1000, HIGH2, 1.0998, 1.1100))
    return days


# ============================= equivalence with the base script =============================

class TestRunDowTheoryBacktestMatchesBase(unittest.TestCase):
    def setUp(self):
        self._orig_swing_len = base.SWING_LEN

    def tearDown(self):
        base.SWING_LEN = self._orig_swing_len

    def test_no_window_default_swing_len_matches_base_backtest_instrument(self):
        base.SWING_LEN = 20
        df = _build_5min_df(_build_uptrend_days(150))
        expected = base.backtest_instrument("SYN", df)
        actual = opt.run_dow_theory_backtest(df, swing_len=20)
        self.assertEqual(len(expected), len(actual))
        self.assertGreater(len(expected), 0, "fixture should produce at least one trade")
        for e, a in zip(expected, actual):
            self.assertEqual(e["side"], a["side"])
            self.assertEqual(e["outcome"], a["outcome"])
            self.assertAlmostEqual(e["r"], a["r"], places=9)
            self.assertEqual(e["date"], a["date"])

    def test_different_swing_len_can_change_the_result(self):
        df = _build_5min_df(_build_uptrend_days(150))
        trades_10 = opt.run_dow_theory_backtest(df, swing_len=10)
        trades_50 = opt.run_dow_theory_backtest(df, swing_len=50)
        # SWING_LEN=50 requires far more confirmation distance than this fixture's pivots are
        # spaced for - it should produce no trades, unlike the tighter SWING_LEN=10.
        self.assertGreater(len(trades_10), 0)
        self.assertEqual(len(trades_50), 0)


class TestSwingLenOverrideAlwaysRestored(unittest.TestCase):
    def setUp(self):
        self._orig_swing_len = base.SWING_LEN

    def tearDown(self):
        base.SWING_LEN = self._orig_swing_len

    def test_restored_after_a_normal_call(self):
        df = _build_5min_df(_build_uptrend_days(150))
        base.SWING_LEN = 999   # deliberately not touched by this call's swing_len=20 argument
        opt.run_dow_theory_backtest(df, swing_len=20)
        self.assertEqual(base.SWING_LEN, 999)

    def test_restored_even_if_compute_daily_swing_signals_raises(self):
        df = _build_5min_df(_build_uptrend_days(150))
        base.SWING_LEN = 999
        original_fn = base.compute_daily_swing_signals

        def _raising(*args, **kwargs):
            raise RuntimeError("boom")

        base.compute_daily_swing_signals = _raising
        try:
            with self.assertRaises(RuntimeError):
                opt.run_dow_theory_backtest(df, swing_len=20)
        finally:
            base.compute_daily_swing_signals = original_fn
        self.assertEqual(base.SWING_LEN, 999)


# ============================= windowing =============================

class TestWindowing(unittest.TestCase):
    def test_trades_outside_the_window_are_excluded(self):
        df = _build_5min_df(_build_uptrend_days(150))
        full = opt.run_dow_theory_backtest(df, swing_len=20)
        self.assertGreater(len(full), 0)
        trade_date = full[0]["date"]

        windowed = opt.run_dow_theory_backtest(df, swing_len=20,
                                                 window_start=datetime.date(2016, 1, 1),
                                                 window_end=trade_date)
        self.assertEqual(windowed, [], "a window ending before the trade's date must exclude it")

    def test_pivots_confirmed_before_window_start_do_not_leak_into_a_later_window(self):
        # Build an uptrend fixture whose 2 confirming pivots both land BEFORE a window that starts
        # well after them - a version of this strategy re-fit fresh on ONLY the later window
        # should see NO confirmed pivots yet (they were never re-observed inside the window), so
        # it must not fire a signal purely because the full, unwindowed history already had them.
        df = _build_5min_df(_build_uptrend_days(150))
        full = opt.run_dow_theory_backtest(df, swing_len=20)
        self.assertGreater(len(full), 0)
        trigger_date = full[0]["date"]

        later_window_start = trigger_date + datetime.timedelta(days=5)
        windowed = opt.run_dow_theory_backtest(df, swing_len=20, window_start=later_window_start,
                                                 window_end=datetime.date(2016, 12, 1))
        self.assertEqual(windowed, [],
                          "pivots confirmed entirely before window_start must not carry over into "
                          "a fresh in-window trend-state computation")


# ============================= eval_fn / lockbox wiring =============================

class TestMakeEvalFn(unittest.TestCase):
    def test_tags_each_trade_with_its_instrument_label(self):
        df_a = _build_5min_df(_build_uptrend_days(150))
        df_b = _build_5min_df(_build_uptrend_days(150, start="2018-01-02"))
        eval_fn = opt._make_eval_fn({"A": df_a, "B": df_b})
        trades = eval_fn({"SWING_LEN": 20})
        self.assertTrue(trades)
        labels = {t["instrument"] for t in trades}
        self.assertEqual(labels, {"A", "B"})


class TestMakeLockboxBacktestFn(unittest.TestCase):
    def test_restricts_to_the_lockbox_window_and_tags_instrument(self):
        df = _build_5min_df(_build_uptrend_days(150))
        backtest_fn = opt.make_lockbox_backtest_fn({"SYN": df}, swing_len=20)
        full = opt.run_dow_theory_backtest(df, swing_len=20)
        trigger_date = full[0]["date"]

        before = backtest_fn(datetime.date(2016, 1, 1), trigger_date)
        self.assertEqual(before, [])
        spanning = backtest_fn(datetime.date(2016, 1, 1), datetime.date(2016, 12, 1))
        self.assertTrue(spanning)
        self.assertEqual(spanning[0]["instrument"], "SYN")


# ============================= param search glue =============================

class TestRunParamSearch(unittest.TestCase):
    def test_grid_covers_every_swing_len_value(self):
        df = _build_5min_df(_build_uptrend_days(150))
        result = opt.run_param_search({"SYN": df}, [10, 20, 30],
                                        window_start=datetime.date(2016, 1, 1),
                                        window_end=datetime.date(2016, 12, 1))
        tried = {entry["params"]["SWING_LEN"] for entry in result["all"]}
        self.assertEqual(tried, {10, 20, 30})

    def test_best_cell_matches_a_direct_manual_search(self):
        df = _build_5min_df(_build_uptrend_days(150))
        result = opt.run_param_search({"SYN": df}, [10, 20, 50],
                                        window_start=datetime.date(2016, 1, 1),
                                        window_end=datetime.date(2016, 12, 1))
        manual_totals = {
            swing_len: engine.total_r(opt.run_dow_theory_backtest(df, swing_len,
                                                                     datetime.date(2016, 1, 1),
                                                                     datetime.date(2016, 12, 1)))
            for swing_len in [10, 20, 50]
        }
        best_manual = max(manual_totals, key=manual_totals.get)
        self.assertEqual(result["best"]["params"]["SWING_LEN"], best_manual)


# ============================= smoke test =============================

class TestRunSmokeTest(unittest.TestCase):
    def test_smoke_test_completes_and_scores(self):
        short_results, long_results = opt.run_smoke_test(verbose=False)
        self.assertEqual(short_results["verdict"], "SCORED")
        self.assertEqual(long_results["verdict"], "SCORED")
        self.assertGreaterEqual(len(long_results["step4_folds"]), 1)


# ============================= main() end-to-end (synthetic data, no real network) =============================

class TestMainEndToEnd(unittest.TestCase):
    """Exercises main() itself - not just run_full_pipeline/run_smoke_test - by monkeypatching
    fetch_instrument_data and INSTRUMENTS so it runs against synthetic data with no real
    Dukascopy connection. This is the ONLY test that actually calls opt_engine.split_lockbox the
    way main() does and unpacks its real 4-tuple return - a smoke test that only calls
    run_full_pipeline directly (as this file's other tests and run_smoke_test() do) would never
    have caught main()'s own lockbox-unpacking arity bug (3 names for a 4-value return) that this
    test was written specifically to catch."""

    def test_main_runs_without_error_against_synthetic_data(self):
        # LOCKBOX_LEDGER_PATH MUST be overridden to a throwaway tempfile before calling main() -
        # the real path is a genuinely permanent, one-time-ever ledger (see optimization_engine.
        # py's LOCKBOX/EMBARGOED FINAL HOLDOUT section header), and this project's own convention
        # (test_optimization_engine.py's TestLockboxConfirm, and this exact module's own
        # docstring note in make_lockbox_backtest_fn/main) is that testing must never consume it
        # with synthetic data. An earlier version of this test skipped this override by mistake
        # and wrote a bogus synthetic-data entry to the REAL research/lockbox_ledger.json,
        # permanently consuming "dow_theory_swing_structure"'s one real attempt - caught and
        # reverted (untracked file, safe to delete) before this was ever committed. This test
        # exists specifically to make sure that mistake can't happen again silently.
        with tempfile.TemporaryDirectory() as tmp_dir:
            long_df = _build_5min_df(_build_uptrend_days(1050))
            original_instruments = opt.INSTRUMENTS
            original_fetch = opt.fetch_instrument_data
            original_ledger_path = opt.LOCKBOX_LEDGER_PATH
            opt.INSTRUMENTS = [("SYN", "SYN_CONST")]
            opt.fetch_instrument_data = lambda label, const: long_df
            opt.LOCKBOX_LEDGER_PATH = os.path.join(tmp_dir, "lockbox_ledger.json")
            try:
                opt.main()   # must not raise - in particular, must not hit a tuple-unpacking error
            finally:
                opt.INSTRUMENTS = original_instruments
                opt.fetch_instrument_data = original_fetch
                opt.LOCKBOX_LEDGER_PATH = original_ledger_path
        self.assertFalse(os.path.exists(opt.LOCKBOX_LEDGER_PATH),
                          "the real lockbox ledger must not have been created/modified by this test")

    def test_split_lockbox_unpacks_as_search_start_search_end_lockbox_start_lockbox_end(self):
        # Direct check of the exact unpacking main() relies on, independent of running the whole
        # pipeline - if optimization_engine.split_lockbox's return shape ever changes, this fails
        # fast with a clear message instead of a confusing downstream type error.
        search_start, search_end, lockbox_start, lockbox_end = engine.split_lockbox(
            opt.FETCH_START, opt.FETCH_END, opt.LOCKBOX_MONTHS)
        self.assertEqual(search_start, opt.FETCH_START)
        self.assertEqual(search_end, lockbox_start)
        self.assertEqual(lockbox_end, opt.FETCH_END)
        self.assertLess(search_end, lockbox_end)


if __name__ == "__main__":
    unittest.main()
