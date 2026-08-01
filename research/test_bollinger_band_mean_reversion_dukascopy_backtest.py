# Unit tests + a granularity-convergence check + one synthetic end-to-end
# smoke run for bollinger_band_mean_reversion_dukascopy_backtest.py.
#
# Covers, per this project's rigor conventions:
#   1. Band computation correctness (SMA/stddev) on a small known series,
#      cross-checked against an independent numpy computation.
#   2. The "confirmed re-entry, not raw touch" trigger logic - two
#      deliberately contrasting cases: one where a naive raw-touch rule
#      would have entered and lost money while the confirmed-re-entry rule
#      correctly stays flat, and the reverse - a case the confirmed rule
#      correctly trades that a naive "must touch the band on the trigger
#      bar itself" rule would never even recognize as a signal.
#   3. The degenerate-geometry guard (bad target side, near-zero/inverted
#      stop distance) - both must be skipped, not forced through.
#   4. Max-hold forced-close accounting, including the exact bar-count
#      boundary (47 bars held -> still open, 48 -> forced FLAT close).
#   5. A genuine granularity-convergence check: synthetic OHLC built from a
#      proper multi-step intraday random walk (each bar's open anchored to
#      the previous bar's actual close, dozens of sub-steps per bar - NOT a
#      single close-plus-independent-wick-noise construction), run at
#      increasing sub-bar resolution. This directly tests the false-
#      positive trap this project's other band/level-touch strategies have
#      had to guard against: a coarse, few-substep OHLC approximation lets
#      a stop/target strategy's geometry couple spuriously with the
#      construction's own wick noise, producing an apparent "edge" in a
#      process that has none by construction (a fair, zero-drift random
#      walk). The check confirms that edge shrinks hard as granularity
#      increases, i.e. it's a construction artifact, not a real signal.
#   6. ONE small synthetic end-to-end smoke run confirming the full
#      fetch-independent pipeline (indicator -> entries -> exits ->
#      reporting-style aggregation) runs without crashing on multi-
#      "instrument" data. Deliberately NOT a real Dukascopy download -
#      this project's convention is to validate via unit tests plus a
#      synthetic smoke run, not to run the full real backtest in a
#      sandboxed test pass.
#
# Run with:  python -m pytest research/test_bollinger_band_mean_reversion_dukascopy_backtest.py -v
# or:        python research/test_bollinger_band_mean_reversion_dukascopy_backtest.py

import importlib.util
import os
import unittest

import numpy as np
import pandas as pd

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "bollinger_band_mean_reversion_dukascopy_backtest.py")
_spec = importlib.util.spec_from_file_location("bollinger_band_mean_reversion_dukascopy_backtest", _MODULE_PATH)
bb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bb)


# ============================= synthetic data helpers =============================

def _make_df(closes, highs=None, lows=None, opens=None, start="2020-01-06"):
    """Builds a minimal OHLC DataFrame indexed by 5-min bars. Highs/lows/opens default to
    the close series (flat bars) unless explicitly overridden - keeps test fixtures terse."""
    n = len(closes)
    opens = opens if opens is not None else list(closes)
    highs = highs if highs is not None else list(closes)
    lows = lows if lows is not None else list(closes)
    idx = pd.date_range(start=start, periods=n, freq="5min", tz="America/New_York")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


def _raw_touch_reference_entries(closes, lower, upper):
    """TEST-ONLY reference strategy, not the production code: a naive rule that enters the
    INSTANT a bar's close is outside a band, with no confirmation wait at all. Used purely to
    contrast against the shipped confirmed-re-entry rule in the tests below."""
    entries = []
    for i in range(len(closes)):
        if lower[i] is None:
            continue
        if closes[i] < lower[i]:
            entries.append(("LONG", i))
        elif closes[i] > upper[i]:
            entries.append(("SHORT", i))
    return entries


def build_synthetic_ohlc(n_bars, n_substeps, bar_std_pct, start_price=1.1000, seed=0):
    """Proper multi-step intraday random walk: each bar's OPEN is anchored to the PREVIOUS
    bar's actual close, and the bar's high/low/close emerge from n_substeps independent
    Gaussian sub-steps along one continuous path within the bar - NOT a single close-plus-
    independent-wick-noise construction (see this file's header + the granularity-convergence
    test below for why that distinction matters). Per-substep stddev is scaled by
    1/sqrt(n_substeps) so each bar's total variance stays constant regardless of granularity -
    the only thing changing across n_substeps is how finely the intrabar path is resolved."""
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


# ============================= band computation correctness =============================

class TestBollingerBandsComputation(unittest.TestCase):
    def test_known_series_matches_independent_numpy_calc(self):
        closes = list(range(1, 21))   # 1..20, exactly one full 20-length window
        middle, upper, lower = bb.compute_bollinger_bands(closes, length=20, num_std=2.0)

        expected_sma = float(np.mean(closes))
        expected_std = float(np.std(closes, ddof=1))   # pandas default is sample stddev (ddof=1)
        self.assertAlmostEqual(middle[-1], expected_sma, places=9)
        self.assertAlmostEqual(upper[-1], expected_sma + 2 * expected_std, places=9)
        self.assertAlmostEqual(lower[-1], expected_sma - 2 * expected_std, places=9)

    def test_bands_are_none_until_window_is_full(self):
        closes = list(range(1, 20))   # only 19 values - one short of a full 20-length window
        middle, upper, lower = bb.compute_bollinger_bands(closes, length=20, num_std=2.0)
        self.assertTrue(all(v is None for v in middle))
        self.assertTrue(all(v is None for v in upper))
        self.assertTrue(all(v is None for v in lower))

    def test_no_lookahead_band_value_only_uses_bars_up_to_that_index(self):
        # extending the series with wildly different future values must not change an
        # already-computed earlier band value
        base = list(range(1, 21))
        extended = base + [1000.0, -1000.0, 500.0]
        mid_base, up_base, lo_base = bb.compute_bollinger_bands(base)
        mid_ext, up_ext, lo_ext = bb.compute_bollinger_bands(extended)
        self.assertAlmostEqual(mid_base[19], mid_ext[19], places=9)
        self.assertAlmostEqual(up_base[19], up_ext[19], places=9)
        self.assertAlmostEqual(lo_base[19], lo_ext[19], places=9)


# ============================= confirmed re-entry vs raw touch =============================

class TestConfirmedReentryTrigger(unittest.TestCase):
    def test_raw_touch_would_wrongly_enter_but_confirmed_reentry_does_not(self):
        """Price closes below the lower band and then KEEPS FALLING - never closes back
        inside. A raw-touch rule (enter the instant close < lower band) fires immediately and
        rides a continuing decline into a loss. The confirmed-re-entry rule never gets its
        second half of the pattern (a close back >= the lower band), so it correctly takes
        zero trades in this window."""

        def const_bands(closes):
            n = len(closes)
            return [100.0] * n, [101.0] * n, [99.0] * n

        original = bb.compute_bollinger_bands
        bb.compute_bollinger_bands = const_bands
        try:
            closes = [100.0, 99.0, 98.5, 98.0, 97.5, 97.0, 96.5, 96.0]
            df = _make_df(closes)

            # what a naive raw-touch rule would have done
            _, upper, lower = const_bands(closes)
            raw_entries = _raw_touch_reference_entries(closes, lower, upper)
            self.assertTrue(any(side == "LONG" for side, _ in raw_entries),
                             "sanity check: raw-touch rule should fire on this falling series")
            first_raw_entry_price = closes[raw_entries[0][1]]
            raw_touch_pnl = closes[-1] - first_raw_entry_price
            self.assertLess(raw_touch_pnl, 0, "the raw-touch entry should be a losing trade here")

            # the production confirmed-re-entry rule takes nothing
            trades = bb.backtest_instrument("X", df)
            self.assertEqual(trades, [])
        finally:
            bb.compute_bollinger_bands = original

    def test_confirmed_reentry_enters_even_when_confirm_bar_never_touches_the_band(self):
        """Vice versa: the excursion bar (i-1) closes below the lower band, and the NEXT bar's
        close is back above the lower band by a comfortable margin - its own low never comes
        anywhere near the band line. A raw rule that requires the trigger bar itself to
        physically touch the band would never recognize this as a valid signal at all, but the
        confirmed-re-entry rule (purely close-based) correctly enters here."""

        def const_bands(closes):
            n = len(closes)
            return [100.0] * n, [101.0] * n, [99.0] * n

        original = bb.compute_bollinger_bands
        bb.compute_bollinger_bands = const_bands
        try:
            # bar0: excursion, close=98.0 (< lower=99). bar1: confirm, close=99.7 (>= lower=99,
            # and its low/high - flat OHLC - never dips down to touch 99 at all).
            # bar2: rallies further so the fixed middle-band target (100.0) is reached (TP).
            closes = [98.0, 99.7, 100.5]
            df = _make_df(closes)

            confirm_bar_low = closes[1]
            self.assertGreater(confirm_bar_low, 99.0,
                                "sanity check: confirm bar must not itself touch the lower band")

            trades = bb.backtest_instrument("Y", df)
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0]["side"], "LONG")
            self.assertEqual(trades[0]["outcome"], "TP")
        finally:
            bb.compute_bollinger_bands = original

    def test_mirrors_correctly_for_shorts(self):
        """The same raw-touch-avoided behavior, mirrored for the upper band / SHORT side."""

        def const_bands(closes):
            n = len(closes)
            return [100.0] * n, [101.0] * n, [99.0] * n

        original = bb.compute_bollinger_bands
        bb.compute_bollinger_bands = const_bands
        try:
            closes = [100.0, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0]   # keeps rallying, never confirms back
            df = _make_df(closes)
            trades = bb.backtest_instrument("Z", df)
            self.assertEqual(trades, [])
        finally:
            bb.compute_bollinger_bands = original


# ============================= degenerate-geometry guard =============================

class TestDegenerateGeometryGuard(unittest.TestCase):
    def test_target_on_wrong_side_of_entry_is_skipped(self):
        """The confirm bar's middle band has (in a fast market) drifted below the entry price
        for a long - target < entry is nonsensical geometry and must be skipped, not forced."""

        def bands_bad_target(closes):
            n = len(closes)
            middle = [100.0] * n
            middle[2] = 99.0   # confirm bar's middle deliberately below entry
            return middle, [101.0] * n, [99.5] * n

        original = bb.compute_bollinger_bands
        bb.compute_bollinger_bands = bands_bad_target
        try:
            closes = [100.0, 99.0, 100.0, 100.0]
            df = _make_df(closes)
            trades = bb.backtest_instrument("BadTarget", df)
            self.assertEqual(trades, [])
        finally:
            bb.compute_bollinger_bands = original

    def test_near_zero_or_inverted_stop_distance_is_skipped(self):
        """A near-zero (or, in this constructed edge case, inverted) stop distance must be
        skipped rather than taken with a degenerate risk/reward."""

        def bands_const(closes):
            n = len(closes)
            return [100.0] * n, [101.0] * n, [99.0] * n

        original = bb.compute_bollinger_bands
        bb.compute_bollinger_bands = bands_const
        try:
            closes = [100.0, 98.0, 100.0, 100.5]
            lows = list(closes)
            lows[1] = 100.001   # excursion bar's low artificially above the confirm bar's close
            df = _make_df(closes, lows=lows)
            trades = bb.backtest_instrument("TinySL", df)
            self.assertEqual(trades, [])
        finally:
            bb.compute_bollinger_bands = original


# ============================= max-hold forced-close accounting =============================

class TestMaxHoldForcedClose(unittest.TestCase):
    def _open_a_trade_that_never_hits_stop_or_target(self, n_bars_after_entry):
        def bands_const(closes):
            n = len(closes)
            return [100.0] * n, [200.0] * n, [99.0] * n   # target far out of reach

        original = bb.compute_bollinger_bands
        bb.compute_bollinger_bands = bands_const
        try:
            closes = [98.0, 99.5] + [99.6] * n_bars_after_entry
            df = _make_df(closes)
            return bb.backtest_instrument("MaxHold", df)
        finally:
            bb.compute_bollinger_bands = original

    def test_forced_close_fires_at_exactly_max_hold_bars(self):
        trades = self._open_a_trade_that_never_hits_stop_or_target(bb.MAX_HOLD_BARS)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "FLAT")
        # r = pnl / sl_distance; pnl = 99.6 - 99.5 = 0.1 (small, positive but not a real TP)
        self.assertGreater(trades[0]["r"], 0)
        self.assertLess(trades[0]["r"], 1.0)

    def test_one_bar_short_of_max_hold_stays_open_and_produces_no_trade(self):
        # bars_held only reaches MAX_HOLD_BARS - 1 by the end of the data - the position is
        # still open when the series ends, so nothing is ever appended to `trades` (matching
        # this project's convention elsewhere of not fabricating a close past the data's end)
        trades = self._open_a_trade_that_never_hits_stop_or_target(bb.MAX_HOLD_BARS - 1)
        self.assertEqual(trades, [])


# ============================= one trade at a time =============================

class TestOneTradeAtATime(unittest.TestCase):
    def test_second_signal_while_in_a_position_is_ignored(self):
        def bands_const(closes):
            n = len(closes)
            return [100.0] * n, [101.0] * n, [99.0] * n

        original = bb.compute_bollinger_bands
        bb.compute_bollinger_bands = bands_const
        try:
            # bar0: excursion below lower. bar1: confirm -> LONG opened at 99.5, stop far below.
            # bar2: a SHORT confirm pattern would trigger too (close above upper) if flat, but
            # a trade is already open - must be ignored. bar3: finally hits the LONG's target.
            closes = [98.0, 99.5, 101.5, 100.5]
            df = _make_df(closes)
            trades = bb.backtest_instrument("OneAtATime", df)
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0]["side"], "LONG")
        finally:
            bb.compute_bollinger_bands = original


# ============================= granularity-convergence check =============================

class TestGranularityConvergence(unittest.TestCase):
    """The core honesty check for a touch/re-entry strategy like this one: does the apparent
    edge measured on synthetic data survive as the synthetic OHLC's intrabar path gets more
    finely resolved? A fair, zero-drift random walk has NO true edge at any resolution - if a
    strategy shows a strong "edge" only at coarse resolution and that edge collapses as
    resolution increases, the coarse result was a construction artifact (the wick noise
    coupling spuriously with the stop/target geometry), not a real signal. Fixed seeds make
    this fully deterministic/reproducible, not a flaky statistical test."""

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
                trades = bb.backtest_instrument("SYN", df)
                all_r.extend(t["r"] for t in trades)
            n = len(all_r)
            avg_r = sum(all_r) / n
            z = avg_r / (1 / (n ** 0.5))
            results[n_substeps] = {"n_trades": n, "avg_r": avg_r, "z": z}

        z1 = abs(results[1]["z"])
        z4 = abs(results[4]["z"])
        z16 = abs(results[16]["z"])
        z64 = abs(results[64]["z"])

        # the coarsest (few-substep) construction shows a large, clearly-non-chance apparent
        # edge (this is the false-positive trap this check exists to catch)
        self.assertGreater(z1, 10.0, f"expected a strong coarse-granularity artifact, got z={z1:.2f}")

        # granularity strictly increasing from 1 -> 4 -> 16 substeps shows monotonically
        # shrinking apparent significance
        self.assertGreater(z1, z4)
        self.assertGreater(z4, z16)

        # by 64 substeps the effect has collapsed to a small fraction of the coarse-granularity
        # reading - no longer the dominant, obviously-not-chance signal seen at n_substeps=1
        self.assertLess(z64, z1 / 3.0)


# ============================= synthetic end-to-end smoke run =============================

class TestEndToEndSmokeRun(unittest.TestCase):
    """ONE small synthetic multi-"instrument" run confirming the full pipeline (indicator ->
    entry -> exit -> reporting-style aggregation) runs without crashing and produces
    plausibly-structured output. Deliberately NOT a real Dukascopy download."""

    def test_synthetic_multi_instrument_run_produces_well_formed_trades(self):
        all_trades = []
        for label, seed in [("SYN_EURUSD", 1), ("SYN_XAUUSD", 2), ("SYN_GBPUSD", 3)]:
            opens, highs, lows, closes = build_synthetic_ohlc(
                3000, n_substeps=16, bar_std_pct=0.06, start_price=1.2000, seed=seed)
            idx = pd.date_range("2019-03-01", periods=3000, freq="5min", tz="America/New_York")
            df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)
            trades = bb.backtest_instrument(label, df)
            for t in trades:
                t["instrument"] = label
            all_trades.extend(trades)

        self.assertGreater(len(all_trades), 0, "expected at least some trades across 9000 synthetic bars")
        for t in all_trades:
            self.assertIn(t["outcome"], ("TP", "SL", "FLAT"))
            self.assertIn(t["side"], ("LONG", "SHORT"))
            self.assertTrue(np.isfinite(t["r"]))

        # reporting-style aggregation (mirrors main()'s computations) must not crash
        total_r = sum(t["r"] for t in all_trades)
        n_trades = len(all_trades)
        z = (total_r / n_trades) / (1 / (n_trades ** 0.5))
        self.assertTrue(np.isfinite(z))

        trades_sorted = sorted(all_trades, key=lambda t: t["date"])
        midpoint_date = trades_sorted[len(trades_sorted) // 2]["date"]
        first_half = [t for t in trades_sorted if t["date"] < midpoint_date]
        second_half = [t for t in trades_sorted if t["date"] >= midpoint_date]
        self.assertEqual(len(first_half) + len(second_half), n_trades)


if __name__ == "__main__":
    unittest.main()
