# Unit tests for parabolic_sar_dukascopy_backtest.py.
#
# Covers, per this project's rigor conventions:
#   1. compute_parabolic_sar's recursive formula, hand-derived and verified bar-by-bar for a
#      short synthetic uptrend-then-reversal sequence (independently worked through by hand in
#      the commit that added this test, not just "whatever the code happens to output").
#   2. The bootstrap fix: without it, a strategy fetched over a window whose FIRST trend never
#      reverses would silently record zero trades and miss the entire move - this is a real bug
#      that was caught and fixed during this script's own development (a steady-uptrend smoke
#      test produced 0 trades before the fix), not a hypothetical.
#   3. The bootstrap's widen-not-skip MIN_SL_PCT handling vs. a normal mid-series reversal's
#      skip-not-force handling - two different, deliberately different guards on the same floor.
#   4. Stop-and-reverse trade management: a reversal closes the open position and opens the
#      opposite one at the same fill level; a forced FLAT close at series end when no further
#      reversal occurs.
#   5. A LONG/SHORT mirror check (same technique used elsewhere in this project's test suite -
#      mirroring price around a pivot and swapping high/low turns a hand-verified uptrend-first
#      scenario into an equally-verified downtrend-first one).
#   6. A smoke end-to-end run on synthetic multi-instrument daily data.
#
# Run with:  python -m pytest research/test_parabolic_sar_dukascopy_backtest.py -v

import importlib.util
import os
import unittest

import numpy as np
import pandas as pd

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parabolic_sar_dukascopy_backtest.py")
_spec = importlib.util.spec_from_file_location("parabolic_sar_dukascopy_backtest", _MODULE_PATH)
psar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(psar)


# ============================= synthetic data helpers =============================

# Hand-derived 10-bar sequence: an uptrend bootstrap at bar 1 (trend continues +1..+7 through
# bar 7, acceleration factor ratcheting 0.02 -> 0.08 as fresh highs are made), then a reversal
# to a downtrend at bar 8 (verified by hand: projected = 9.5679 + 0.08*(13.5-9.5679) = 9.8825,
# which is < highs[7]/highs[6] so sar[8]=9.8825 unclipped, and 9.8825 > lows[8]=9.0 so the trend
# flips; real_sar[8] resets to ep[7]=13.5 per Wilder's own reversal rule), continuing down at
# bar 9. Independently re-verified against the module's own output before being baked into these
# assertions (see the PR/commit history for the by-hand working).
_HIGHS = [10, 11, 12, 13, 13.5, 13.2, 12.5, 11.5, 10.5, 9.5]
_LOWS = [8, 9, 10, 11, 12.5, 12.0, 11.0, 10.0, 9.0, 8.0]
_CLOSES = [9, 10, 11, 12, 13, 12.5, 11.5, 10.5, 9.5, 8.5]


def _make_df(highs, lows, closes, start="2020-01-06", freq="1D"):
    idx = pd.date_range(start=start, periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame({"Open": closes, "High": highs, "Low": lows, "Close": closes}, index=idx)


def _mirror_ohlc(highs, lows, closes, pivot=20.0):
    """Mirrors a bar sequence around `pivot` (mirrored = pivot - value), swapping high/low so the
    result is still valid OHLC - same technique as
    test_evendyer_vwap_orb_dukascopy_backtest.py's _mirror_ohlc, turns a hand-verified
    uptrend-first scenario into an equally-verified downtrend-first one without re-deriving the
    recursive SAR arithmetic by hand a second time."""
    return ([pivot - l for l in lows], [pivot - h for h in highs], [pivot - c for c in closes])


def build_synthetic_ohlc(n_bars, n_substeps, bar_std_pct, start_price=100.0, seed=0):
    """Proper multi-step random walk - same construction as this project's other test files'
    copy of this helper (e.g. test_evendyer_vwap_orb_dukascopy_backtest.py)."""
    rng = np.random.default_rng(seed)
    substep_std = start_price * (bar_std_pct / 100.0) / np.sqrt(n_substeps)
    increments = rng.normal(0.0, substep_std, size=(n_bars, n_substeps))
    opens = np.empty(n_bars)
    highs = np.empty(n_bars)
    lows = np.empty(n_bars)
    closes = np.empty(n_bars)
    anchor = start_price
    for k in range(n_bars):
        path = anchor + np.cumsum(increments[k])
        opens[k] = anchor
        highs[k] = max(anchor, path.max())
        lows[k] = min(anchor, path.min())
        closes[k] = path[-1]
        anchor = closes[k]
    return opens, highs, lows, closes


# ============================= compute_parabolic_sar recursion =============================

class TestComputeParabolicSar(unittest.TestCase):
    def test_hand_derived_uptrend_then_reversal_sequence(self):
        sar, real_sar, trend, ep, af = psar.compute_parabolic_sar(_HIGHS, _LOWS, _CLOSES)

        # bootstrap (bar 1): seeded purely from a 1-bar close comparison
        self.assertEqual(trend[1], 1)
        self.assertAlmostEqual(sar[1], 10.0, places=6)
        self.assertAlmostEqual(ep[1], 11.0, places=6)
        self.assertAlmostEqual(af[1], 0.02, places=6)

        # uptrend continues, run-length counter increments, af ratchets up to its 0.2 cap pace
        self.assertEqual(trend[2], 2)
        self.assertAlmostEqual(af[2], 0.04, places=6)
        self.assertEqual(trend[7], 7)
        self.assertAlmostEqual(ep[7], 13.5, places=6)
        self.assertAlmostEqual(af[7], 0.08, places=6)

        # reversal at bar 8: projected sar breaches lows[8], flips the trend
        self.assertEqual(trend[8], -1)
        self.assertAlmostEqual(sar[8], 9.882502109184, places=6)
        self.assertAlmostEqual(real_sar[8], 13.5, places=6)   # resets to the just-ended trend's EP
        self.assertAlmostEqual(af[8], 0.02, places=6)          # af resets on a fresh trend

        # downtrend continues at bar 9 - no second reversal
        self.assertEqual(trend[9], -2)

    def test_too_short_series_returns_all_defaults_without_crashing(self):
        sar, real_sar, trend, ep, af = psar.compute_parabolic_sar([10], [9], [9.5])
        self.assertEqual(sar, [0.0])
        self.assertEqual(trend, [0])


# ============================= bootstrap handling =============================

class TestBootstrap(unittest.TestCase):
    def test_a_trend_that_never_reverses_still_produces_a_trade_not_zero(self):
        # regression test for the exact bug caught during development: a steady, unbroken trend
        # across the whole fetched window must still be captured, not silently missed because
        # the strategy was waiting for a reversal that never comes
        n = 300
        price = 100 + np.arange(n) * 0.5 + np.random.default_rng(2).normal(0, 0.3, n)
        highs = (price + 0.3).tolist()
        lows = (price - 0.3).tolist()
        closes = price.tolist()
        df = _make_df(highs, lows, closes)
        trades = psar.backtest_instrument("TEST", df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "LONG")
        self.assertEqual(trades[0]["outcome"], "FLAT")
        self.assertGreater(trades[0]["r"], 0)   # a 150-point rally against a near-zero seed stop

    def test_bootstrap_widens_a_degenerate_seed_stop_to_the_floor_rather_than_skipping(self):
        df = _make_df(_HIGHS, _LOWS, _CLOSES)
        trades = psar.backtest_instrument("TEST", df)
        self.assertGreaterEqual(len(trades), 1)
        boot_trade = trades[0]
        self.assertEqual(boot_trade["side"], "LONG")
        # seed sl_distance (|10.0 - 10.0| = 0) is degenerate - must be widened to MIN_SL_PCT of entry
        self.assertAlmostEqual(boot_trade["stop_pct"], psar.MIN_SL_PCT / 100.0, places=8)


# ============================= stop-and-reverse trade management =============================

class TestStopAndReverse(unittest.TestCase):
    def test_reversal_closes_bootstrap_and_opens_opposite_side_at_the_same_fill(self):
        df = _make_df(_HIGHS, _LOWS, _CLOSES)
        trades = psar.backtest_instrument("TEST", df)
        self.assertEqual(len(trades), 2)

        first, second = trades
        self.assertEqual(first["side"], "LONG")
        self.assertEqual(first["outcome"], "SAR")
        self.assertAlmostEqual(first["exit_price"], 9.882502109184, places=6)

        self.assertEqual(second["side"], "SHORT")
        self.assertAlmostEqual(second["entry_price"], 9.882502109184, places=6)   # same level as the exit above
        self.assertAlmostEqual(second["stop_price"], 13.5, places=6)

    def test_no_further_reversal_forces_a_flat_close_at_series_end(self):
        df = _make_df(_HIGHS, _LOWS, _CLOSES)
        trades = psar.backtest_instrument("TEST", df)
        last = trades[-1]
        self.assertEqual(last["outcome"], "FLAT")
        self.assertAlmostEqual(last["exit_price"], _CLOSES[-1], places=6)
        self.assertAlmostEqual(last["r"], 0.38217081278578097, places=6)

    def test_reward_risk_below_min_sl_pct_skips_opening_a_new_position_but_still_records_the_close(self):
        df = _make_df(_HIGHS, _LOWS, _CLOSES)
        orig = psar.MIN_SL_PCT
        psar.MIN_SL_PCT = 50.0   # far above the ~36.6% sl_pct the bar-8 reversal would naturally have
        try:
            trades = psar.backtest_instrument("TEST", df)
        finally:
            psar.MIN_SL_PCT = orig
        # the bootstrap LONG still gets closed and recorded at the reversal...
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "LONG")
        self.assertEqual(trades[0]["outcome"], "SAR")
        # ...but no new SHORT opens, and no final forced-FLAT trade follows it


# ============================= LONG/SHORT mirror =============================

class TestMirror(unittest.TestCase):
    def test_downtrend_first_scenario_mirrors_the_uptrend_first_one(self):
        m_highs, m_lows, m_closes = _mirror_ohlc(_HIGHS, _LOWS, _CLOSES)
        df = _make_df(m_highs, m_lows, m_closes)
        trades = psar.backtest_instrument("TEST", df)
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0]["side"], "SHORT")
        self.assertEqual(trades[1]["side"], "LONG")
        # r-multiples are sign/direction-invariant under mirroring - same magnitudes as the original
        self.assertAlmostEqual(trades[0]["r"], -58.74894540800035, places=4)
        self.assertAlmostEqual(trades[1]["r"], 0.38217081278578097, places=4)


# ============================= smoke end-to-end =============================

class TestSmokeEndToEnd(unittest.TestCase):
    def test_full_pipeline_runs_without_crashing_on_synthetic_multi_instrument_data(self):
        all_trades = []
        for i, label in enumerate(["SYN1", "SYN2", "SYN3"]):
            n_bars = 1500
            opens, highs, lows, closes = build_synthetic_ohlc(n_bars, n_substeps=8, bar_std_pct=1.5,
                                                                 start_price=100 + i * 20, seed=i)
            idx = pd.date_range(start="2018-01-01", periods=n_bars, freq="1D", tz="UTC")
            df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)
            trades = psar.backtest_instrument(label, df)
            for t in trades:
                t["instrument"] = label
            all_trades.extend(trades)

        self.assertGreater(len(all_trades), 0)
        for t in all_trades:
            self.assertIn(t["outcome"], ("SAR", "FLAT"))
            self.assertTrue(np.isfinite(t["r"]))
            self.assertIn(t["side"], ("LONG", "SHORT"))
            self.assertGreaterEqual(t["stop_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
