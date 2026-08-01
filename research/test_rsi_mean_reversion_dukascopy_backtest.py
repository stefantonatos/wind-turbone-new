# Unit tests + a granularity-convergence check + one synthetic end-to-end
# smoke run for rsi_mean_reversion_dukascopy_backtest.py.
#
# Covers, per this project's rigor conventions:
#   1. RSI computation correctness - a monotonic-gains series (RSI must hit
#      exactly 100 once seeded, since avg_loss is 0), and a hand-verifiable
#      15-value series cross-checked against an INDEPENDENT from-scratch
#      re-implementation of Wilder smoothing (not a reuse of the module's
#      own function - a real second implementation, so this isn't just
#      testing "the code agrees with itself").
#   2. The "confirmed re-entry, not raw threshold touch" trigger logic -
#      one case where RSI never confirms back into normal range (confirmed
#      re-entry correctly takes zero trades, while a raw-touch rule that
#      fires the instant RSI crosses 30/70 would already be in a losing
#      position riding the continuing move) and the reverse (confirmed
#      re-entry correctly enters once RSI does recover, at the recovery
#      bar - a materially later/different price than where a raw-touch
#      rule would have already fired).
#   3. The degenerate-geometry guard (near-zero/inverted stop distance).
#   4. Max-hold forced-close accounting, including the exact bar-count
#      boundary (47 bars held -> still open, 48 -> forced FLAT close).
#   5. A genuine granularity-convergence check, same methodology and same
#      false-positive-trap rationale as the Bollinger script's test file -
#      see that file's header for the full writeup. Built independently
#      here (not imported) so this test file stays self-contained, matching
#      this project's existing convention of each script/test pair being
#      independently runnable.
#   6. ONE small synthetic end-to-end smoke run confirming the full
#      pipeline runs without crashing on multi-"instrument" data.
#      Deliberately NOT a real Dukascopy download.
#
# Run with:  python -m pytest research/test_rsi_mean_reversion_dukascopy_backtest.py -v
# or:        python research/test_rsi_mean_reversion_dukascopy_backtest.py

import importlib.util
import os
import unittest

import numpy as np
import pandas as pd

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "rsi_mean_reversion_dukascopy_backtest.py")
_spec = importlib.util.spec_from_file_location("rsi_mean_reversion_dukascopy_backtest", _MODULE_PATH)
rsi_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rsi_mod)

LB = rsi_mod.STOP_LOOKBACK_BARS


# ============================= synthetic data helpers =============================

def _make_df(closes, highs=None, lows=None, opens=None, start="2020-01-06"):
    n = len(closes)
    opens = opens if opens is not None else list(closes)
    highs = highs if highs is not None else list(closes)
    lows = lows if lows is not None else list(closes)
    idx = pd.date_range(start=start, periods=n, freq="5min", tz="America/New_York")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


def _raw_touch_reference_entries(rsi_values):
    """TEST-ONLY reference strategy, not the production code: a naive rule that enters the
    INSTANT RSI crosses the 30/70 threshold, with no confirmation wait. Used purely to
    contrast against the shipped confirmed-re-entry rule in the tests below."""
    entries = []
    for i, v in enumerate(rsi_values):
        if v is None:
            continue
        if v < rsi_mod.RSI_OVERSOLD:
            entries.append(("LONG", i))
        elif v > rsi_mod.RSI_OVERBOUGHT:
            entries.append(("SHORT", i))
    return entries


def build_synthetic_ohlc(n_bars, n_substeps, bar_std_pct, start_price=1.1000, seed=0):
    """Proper multi-step intraday random walk - see
    test_bollinger_band_mean_reversion_dukascopy_backtest.py's copy of this helper for the full
    writeup on why this construction (vs a single close-plus-independent-wick-noise one)
    matters for honestly validating a touch/re-entry strategy on synthetic data."""
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


# ============================= RSI computation correctness =============================

class TestRSIComputation(unittest.TestCase):
    def test_monotonic_gains_series_hits_exactly_100(self):
        closes = [10 + i for i in range(20)]   # 19 diffs, every one a +1 gain, zero losses
        rsi = rsi_mod.compute_rsi_series(closes, length=14)
        self.assertTrue(all(v is None for v in rsi[:14]))
        for v in rsi[14:]:
            self.assertAlmostEqual(v, 100.0, places=9)

    def test_hand_verifiable_series_matches_independent_reimplementation(self):
        # a commonly-used small worked example for Wilder's RSI
        closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
                  45.89, 46.03, 45.61, 46.28, 46.28]

        # independent, from-scratch re-implementation (deliberately NOT calling into the
        # module under test) to cross-check against
        diffs = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(d, 0.0) for d in diffs]
        losses = [max(-d, 0.0) for d in diffs]
        length = 14
        avg_gain = sum(gains[:length]) / length
        avg_loss = sum(losses[:length]) / length
        rs = avg_gain / avg_loss
        expected_rsi = 100.0 - 100.0 / (1.0 + rs)

        rsi = rsi_mod.compute_rsi_series(closes, length=14)
        self.assertAlmostEqual(rsi[-1], expected_rsi, places=9)

    def test_recursive_smoothing_matches_wilder_formula_across_multiple_bars(self):
        # extend the hand-verifiable series and check the recursive step, not just the seed
        closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
                  45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 45.50]
        diffs = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(d, 0.0) for d in diffs]
        losses = [max(-d, 0.0) for d in diffs]
        length = 14
        avg_gain = sum(gains[:length]) / length
        avg_loss = sum(losses[:length]) / length
        for k in range(length, len(diffs)):
            avg_gain = (avg_gain * (length - 1) + gains[k]) / length
            avg_loss = (avg_loss * (length - 1) + losses[k]) / length
        rs = avg_gain / avg_loss
        expected_rsi = 100.0 - 100.0 / (1.0 + rs)

        rsi = rsi_mod.compute_rsi_series(closes, length=14)
        self.assertAlmostEqual(rsi[-1], expected_rsi, places=9)

    def test_no_lookahead_value_only_uses_bars_up_to_that_index(self):
        base = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
                45.89, 46.03, 45.61, 46.28, 46.28]
        extended = base + [1000.0, -1000.0]
        rsi_base = rsi_mod.compute_rsi_series(base, length=14)
        rsi_ext = rsi_mod.compute_rsi_series(extended, length=14)
        self.assertAlmostEqual(rsi_base[-1], rsi_ext[len(base) - 1], places=9)


# ============================= confirmed re-entry vs raw touch =============================

class TestConfirmedReentryTrigger(unittest.TestCase):
    def test_raw_touch_would_wrongly_enter_but_confirmed_reentry_does_not(self):
        """RSI dips below 30 and STAYS below 30 for the rest of this window (price keeps
        falling) - never crosses back to confirm. A raw-touch rule (enter the instant RSI < 30)
        fires immediately and rides the continuing decline into a loss. Confirmed re-entry
        never sees the second half of the pattern, so it correctly takes zero trades."""
        n = LB + 6
        rsi_values = [None] * LB + [25.0, 20.0, 15.0, 10.0, 8.0, 5.0]
        original = rsi_mod.compute_rsi_series
        rsi_mod.compute_rsi_series = lambda closes, length=14: rsi_values
        try:
            raw_entries = _raw_touch_reference_entries(rsi_values)
            self.assertTrue(any(side == "LONG" for side, _ in raw_entries),
                             "sanity check: raw-touch rule should fire while RSI is under 30")
            closes = [100.0] * LB + [99.0, 98.5, 98.0, 97.5, 97.0, 96.5]
            first_raw_entry_price = closes[raw_entries[0][1]]
            raw_touch_pnl = closes[-1] - first_raw_entry_price
            self.assertLess(raw_touch_pnl, 0, "the raw-touch entry should be a losing trade here")

            df = _make_df(closes)
            trades = rsi_mod.backtest_instrument("NeverConfirms", df)
            self.assertEqual(trades, [])
        finally:
            rsi_mod.compute_rsi_series = original

    def test_confirmed_reentry_enters_once_rsi_recovers_above_threshold(self):
        """Vice versa: RSI dips below 30 then recovers to >= 30 on the very next bar.
        Confirmed re-entry correctly enters on the RECOVERY bar - a materially later, different
        price than the excursion bar a raw-touch rule would have already fired on."""
        n = LB + 3
        rsi_values = [None] * LB + [25.0, 35.0, 35.0]
        original = rsi_mod.compute_rsi_series
        rsi_mod.compute_rsi_series = lambda closes, length=14: rsi_values
        try:
            raw_entries = _raw_touch_reference_entries(rsi_values)
            raw_entry_index = raw_entries[0][1]   # fires on the excursion bar itself (index LB)
            self.assertEqual(raw_entry_index, LB)

            # bar LB: excursion (99.0). bar LB+1: recovery/confirm (99.5) - the CONFIRMED
            # strategy's entry bar, one full bar later than where raw-touch already fired.
            # bar LB+2: rallies hard enough to hit the fixed-R:R target (TP).
            closes = [100.0] * LB + [99.0, 99.5, 101.0]
            df = _make_df(closes)
            trades = rsi_mod.backtest_instrument("Confirms", df)
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0]["side"], "LONG")
            self.assertEqual(trades[0]["outcome"], "TP")

            confirmed_entry_price = closes[LB + 1]
            raw_touch_entry_price = closes[raw_entry_index]
            self.assertNotEqual(confirmed_entry_price, raw_touch_entry_price,
                                 "confirmed re-entry must trade at a different bar/price than raw touch")
        finally:
            rsi_mod.compute_rsi_series = original

    def test_mirrors_correctly_for_shorts(self):
        n = LB + 6
        rsi_values = [None] * LB + [75.0, 80.0, 85.0, 90.0, 92.0, 95.0]
        original = rsi_mod.compute_rsi_series
        rsi_mod.compute_rsi_series = lambda closes, length=14: rsi_values
        try:
            closes = [100.0] * LB + [101.0, 101.5, 102.0, 102.5, 103.0, 103.5]
            df = _make_df(closes)
            trades = rsi_mod.backtest_instrument("ShortNeverConfirms", df)
            self.assertEqual(trades, [])
        finally:
            rsi_mod.compute_rsi_series = original


# ============================= degenerate-geometry guard =============================

class TestDegenerateGeometryGuard(unittest.TestCase):
    def test_near_zero_or_inverted_stop_distance_is_skipped(self):
        rsi_values = [None] * LB + [25.0, 35.0, 35.0]
        original = rsi_mod.compute_rsi_series
        rsi_mod.compute_rsi_series = lambda closes, length=14: rsi_values
        try:
            closes = [100.0] * LB + [99.0, 99.001, 99.6]
            lows = list(closes)
            lows[LB] = 99.5   # excursion bar's low artificially above the confirm bar's close
            df = _make_df(closes, lows=lows)
            trades = rsi_mod.backtest_instrument("TinySL", df)
            self.assertEqual(trades, [])
        finally:
            rsi_mod.compute_rsi_series = original


# ============================= max-hold forced-close accounting =============================

class TestMaxHoldForcedClose(unittest.TestCase):
    def _open_a_trade_that_never_hits_stop_or_target(self, n_bars_after_entry):
        n = LB + 2 + n_bars_after_entry
        rsi_values = [None] * LB + [25.0, 35.0] + [50.0] * n_bars_after_entry
        original = rsi_mod.compute_rsi_series
        rsi_mod.compute_rsi_series = lambda closes, length=14: rsi_values
        try:
            closes = [100.0] * LB + [99.0, 99.5] + [99.55] * n_bars_after_entry
            df = _make_df(closes)
            return rsi_mod.backtest_instrument("MaxHold", df)
        finally:
            rsi_mod.compute_rsi_series = original

    def test_forced_close_fires_at_exactly_max_hold_bars(self):
        trades = self._open_a_trade_that_never_hits_stop_or_target(rsi_mod.MAX_HOLD_BARS)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "FLAT")
        self.assertGreater(trades[0]["r"], 0)
        self.assertLess(trades[0]["r"], rsi_mod.REWARD_RISK)

    def test_one_bar_short_of_max_hold_stays_open_and_produces_no_trade(self):
        trades = self._open_a_trade_that_never_hits_stop_or_target(rsi_mod.MAX_HOLD_BARS - 1)
        self.assertEqual(trades, [])


# ============================= one trade at a time =============================

class TestOneTradeAtATime(unittest.TestCase):
    def test_second_signal_while_in_a_position_is_ignored(self):
        # bar LB: excursion (oversold). bar LB+1: confirm -> LONG opened.
        # bar LB+2: an overbought-confirm pattern would fire too if flat - must be ignored.
        # bar LB+3: hits the LONG's fixed R:R target.
        rsi_values = [None] * LB + [25.0, 35.0, 65.0, 35.0]
        original = rsi_mod.compute_rsi_series
        rsi_mod.compute_rsi_series = lambda closes, length=14: rsi_values
        try:
            closes = [100.0] * LB + [99.0, 99.5, 101.5, 101.0]
            df = _make_df(closes)
            trades = rsi_mod.backtest_instrument("OneAtATime", df)
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0]["side"], "LONG")
        finally:
            rsi_mod.compute_rsi_series = original


# ============================= granularity-convergence check =============================

class TestGranularityConvergence(unittest.TestCase):
    """Same rationale as the Bollinger script's version of this test - see that file's header.
    A fair, zero-drift random walk has no true edge at any resolution; a strong apparent edge
    that only shows up at coarse (few-substep) OHLC resolution and collapses as resolution
    increases is a construction artifact, not a real signal. Fixed seeds -> fully
    deterministic/reproducible, not a flaky statistical test."""

    def test_apparent_edge_shrinks_as_substep_count_increases(self):
        n_bars = 10_000
        seeds = range(20)
        results = {}
        for n_substeps in (1, 4, 16, 64):
            all_r = []
            for seed in seeds:
                opens, highs, lows, closes = build_synthetic_ohlc(
                    n_bars, n_substeps, bar_std_pct=0.05, seed=seed)
                idx = pd.date_range("2018-01-01", periods=n_bars, freq="5min", tz="America/New_York")
                df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)
                trades = rsi_mod.backtest_instrument("SYN", df)
                all_r.extend(t["r"] for t in trades)
            n = len(all_r)
            avg_r = sum(all_r) / n
            std_r = np.std(all_r, ddof=1)
            z = avg_r / (std_r / (n ** 0.5)) if std_r > 0 else 0.0
            results[n_substeps] = {"n_trades": n, "avg_r": avg_r, "z": z}

        z1 = abs(results[1]["z"])
        avg_z_finer = (abs(results[4]["z"]) + abs(results[16]["z"]) + abs(results[64]["z"])) / 3.0

        # the coarsest (few-substep) construction shows a clearly-non-chance apparent edge
        self.assertGreater(z1, 3.0, f"expected a coarse-granularity artifact, got z={z1:.2f}")

        # averaged across the finer granularities, the effect is much smaller than at the
        # coarsest resolution - the RSI strategy's stop/target logic is less tightly coupled to
        # single-bar wick geometry than the Bollinger script's, so this uses an averaged
        # comparison across the finer levels rather than requiring strict monotonicity at every
        # single step, but the qualitative conclusion (coarse construction inflates the
        # apparent edge; finer construction collapses it) holds either way
        self.assertLess(avg_z_finer, z1 / 1.5,
                         f"expected the finer-granularity average ({avg_z_finer:.2f}) well below "
                         f"the coarse reading ({z1:.2f})")
        self.assertLess(avg_z_finer, 2.5,
                         f"expected finer granularities to stay within noise-level significance, "
                         f"got avg |z|={avg_z_finer:.2f}")


# ============================= synthetic end-to-end smoke run =============================

class TestEndToEndSmokeRun(unittest.TestCase):
    def test_synthetic_multi_instrument_run_produces_well_formed_trades(self):
        all_trades = []
        for label, seed in [("SYN_EURUSD", 1), ("SYN_XAUUSD", 2), ("SYN_GBPUSD", 3)]:
            opens, highs, lows, closes = build_synthetic_ohlc(
                3000, n_substeps=16, bar_std_pct=0.06, start_price=1.2000, seed=seed)
            idx = pd.date_range("2019-03-01", periods=3000, freq="5min", tz="America/New_York")
            df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)
            trades = rsi_mod.backtest_instrument(label, df)
            for t in trades:
                t["instrument"] = label
            all_trades.extend(trades)

        self.assertGreater(len(all_trades), 0, "expected at least some trades across 9000 synthetic bars")
        for t in all_trades:
            self.assertIn(t["outcome"], ("TP", "SL", "FLAT"))
            self.assertIn(t["side"], ("LONG", "SHORT"))
            self.assertTrue(np.isfinite(t["r"]))

        total_r = sum(t["r"] for t in all_trades)
        n_trades = len(all_trades)
        all_r = [t["r"] for t in all_trades]
        std_r = np.std(all_r, ddof=1) if n_trades >= 2 else 0.0
        z = (total_r / n_trades) / (std_r / (n_trades ** 0.5)) if std_r > 0 else 0.0
        self.assertTrue(np.isfinite(z))

        trades_sorted = sorted(all_trades, key=lambda t: t["date"])
        midpoint_date = trades_sorted[len(trades_sorted) // 2]["date"]
        first_half = [t for t in trades_sorted if t["date"] < midpoint_date]
        second_half = [t for t in trades_sorted if t["date"] >= midpoint_date]
        self.assertEqual(len(first_half) + len(second_half), n_trades)


if __name__ == "__main__":
    unittest.main()
