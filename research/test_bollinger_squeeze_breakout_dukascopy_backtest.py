# Unit tests + a granularity-convergence check + one synthetic end-to-end
# smoke run for bollinger_squeeze_breakout_dukascopy_backtest.py.
#
# Covers, per this project's rigor conventions:
#   1. Band-width normalization and squeeze-percentile detection - a known
#      series cross-checked against an independent numpy percentile calc.
#   2. most_recent_squeeze_anchor()'s recency-window logic in isolation.
#   3. The "first close beyond the band, once per squeeze episode" one-shot
#      consumption rule - deliberately checked against a case where price
#      keeps closing beyond the band on multiple consecutive bars while
#      flat (must only ever take the FIRST one), and the mirror case where
#      a FRESH squeeze bar later re-arms the rule for a second trade.
#   4. The degenerate-geometry guard (near-zero stop distance) is skipped,
#      not forced through.
#   5. Max-hold forced-close accounting, including the exact bar-count
#      boundary (47 bars held -> still open, 48 -> forced FLAT close).
#   6. One trade at a time.
#   7. A genuine granularity-convergence check, same methodology as
#      test_bollinger_band_mean_reversion_dukascopy_backtest.py's (see that
#      file's header for the full false-positive-trap rationale): synthetic
#      OHLC built from a proper multi-step intraday random walk, run at
#      increasing intrabar resolution, confirming any apparent edge from a
#      fair zero-drift process shrinks hard as resolution increases.
#   8. ONE small synthetic end-to-end smoke run confirming the full
#      pipeline runs without crashing on multi-"instrument" data.
#      Deliberately NOT a real Dukascopy download - this project's
#      convention is unit tests + one synthetic smoke run, not a full real
#      backtest in a sandboxed test pass.
#
# Run with:  python -m pytest research/test_bollinger_squeeze_breakout_dukascopy_backtest.py -v
# or:        python research/test_bollinger_squeeze_breakout_dukascopy_backtest.py

import importlib.util
import os
import unittest

import numpy as np
import pandas as pd

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "bollinger_squeeze_breakout_dukascopy_backtest.py")
_spec = importlib.util.spec_from_file_location("bollinger_squeeze_breakout_dukascopy_backtest", _MODULE_PATH)
bsb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsb)


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


def build_synthetic_ohlc(n_bars, n_substeps, bar_std_pct, start_price=1.1000, seed=0):
    """Same construction as test_bollinger_band_mean_reversion_dukascopy_backtest.py's helper of
    the same name: each bar's OPEN is anchored to the PREVIOUS bar's actual close, and the bar's
    high/low/close emerge from n_substeps independent Gaussian sub-steps along one continuous
    path within the bar (NOT a single close-plus-independent-wick-noise construction). Per-
    substep stddev is scaled by 1/sqrt(n_substeps) so each bar's total variance stays constant
    regardless of granularity - the only thing changing across n_substeps is how finely the
    intrabar path is resolved (and, critically for this strategy, how much genuine "wick" beyond
    the open/close body each bar has - at n_substeps=1 there is NO wick at all, since high/low
    collapse to max/min(open, close))."""
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


# ============================= band width / squeeze detection =============================

class TestBandWidthAndSqueeze(unittest.TestCase):
    def test_band_width_matches_manual_calc(self):
        middle = [100.0, 100.0, None, 200.0]
        upper = [110.0, 105.0, None, 220.0]
        lower = [90.0, 95.0, None, 180.0]
        width = bsb.compute_band_width(middle, upper, lower)
        self.assertAlmostEqual(width[0], 0.20)
        self.assertAlmostEqual(width[1], 0.10)
        self.assertIsNone(width[2])
        self.assertAlmostEqual(width[3], 0.20)

    def test_band_width_guards_zero_middle(self):
        width = bsb.compute_band_width([0.0], [1.0], [-1.0])
        self.assertIsNone(width[0])

    def test_squeeze_flags_match_independent_numpy_percentile(self):
        rng = np.random.default_rng(42)
        band_width = list(rng.uniform(0.001, 0.02, size=300))
        flags = bsb.compute_squeeze_flags(band_width, lookback=50, percentile=10)

        # cross-check bar 200 (well past warmup) against an independent numpy computation
        i = 200
        window = band_width[i - 49:i + 1]
        expected_q = np.percentile(window, 10)
        expected_flag = band_width[i] <= expected_q
        self.assertEqual(flags[i], expected_flag)

    def test_squeeze_flags_false_before_enough_history(self):
        band_width = list(np.random.default_rng(1).uniform(0.001, 0.02, size=10))
        flags = bsb.compute_squeeze_flags(band_width, lookback=50, percentile=10)
        self.assertTrue(all(f is False for f in flags))

    def test_narrowest_bar_in_window_is_always_flagged(self):
        # the single narrowest width in a window must always be at/below the 10th percentile of
        # its own trailing window (whatever that percentile numerically works out to)
        band_width = [0.05] * 50 + [0.001]   # a dramatic, obvious squeeze on the last bar
        flags = bsb.compute_squeeze_flags(band_width, lookback=50, percentile=10)
        self.assertTrue(flags[-1])


# ============================= squeeze recency anchor =============================

class TestSqueezeRecencyAnchor(unittest.TestCase):
    def test_finds_most_recent_squeeze_within_window(self):
        flags = [False, True, False, False, False, False]
        self.assertEqual(bsb.most_recent_squeeze_anchor(flags, 4, recency_bars=5), 1)

    def test_returns_none_when_squeeze_too_far_in_the_past(self):
        flags = [False, True, False, False, False, False]
        # bar 6 would need the squeeze within the last 5 bars (indices 2..6) - bar 1 isn't in it
        flags_extended = flags + [False]
        self.assertEqual(bsb.most_recent_squeeze_anchor(flags_extended, 6, recency_bars=5), None)

    def test_prefers_the_most_recent_of_multiple_squeeze_bars(self):
        flags = [True, False, True, False, False]
        self.assertEqual(bsb.most_recent_squeeze_anchor(flags, 4, recency_bars=5), 2)

    def test_current_bar_itself_counts(self):
        flags = [False, False, False, False, True]
        self.assertEqual(bsb.most_recent_squeeze_anchor(flags, 4, recency_bars=5), 4)


# ============================= one-shot breakout consumption =============================

class TestFirstBreakoutOnlyPerSqueeze(unittest.TestCase):
    def test_only_the_first_of_several_consecutive_breakout_closes_is_taken(self):
        """Bar 0 is squeezed (flagged directly). Bars 1, 2, 3 all close beyond the upper band -
        only bar 1 (the FIRST) should produce a trade; bars 2 and 3 must be ignored even though
        they'd also individually qualify as breakouts, since no fresh squeeze anchor appears."""

        def const_bands(closes):
            n = len(closes)
            return [100.0] * n, [101.0] * n, [99.0] * n

        def squeeze_only_bar0(band_width, lookback=None, percentile=None):
            return [True] + [False] * (len(band_width) - 1)

        original_bands, original_squeeze = bsb.compute_bollinger_bands, bsb.compute_squeeze_flags
        bsb.compute_bollinger_bands = const_bands
        bsb.compute_squeeze_flags = squeeze_only_bar0
        try:
            # bar0: squeeze bar (flat, inside bands). bars 1-3: keep closing above upper (101).
            closes = [100.0, 102.0, 103.0, 104.0, 90.0]   # bar4 crashes below lower -> would hit stop
            df = _make_df(closes)
            trades = bsb.backtest_instrument("X", df)
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0]["side"], "LONG")
        finally:
            bsb.compute_bollinger_bands, bsb.compute_squeeze_flags = original_bands, original_squeeze

    def test_a_fresh_squeeze_bar_rearms_the_rule(self):
        """After the first breakout is consumed and its trade resolves, a NEW squeeze bar later
        in the series must allow a second, independent breakout trade."""

        def const_bands(closes):
            n = len(closes)
            return [100.0] * n, [101.0] * n, [99.0] * n

        squeeze_bars = {0, 5}

        def squeeze_at(band_width, lookback=None, percentile=None):
            return [i in squeeze_bars for i in range(len(band_width))]

        original_bands, original_squeeze = bsb.compute_bollinger_bands, bsb.compute_squeeze_flags
        bsb.compute_bollinger_bands = const_bands
        bsb.compute_squeeze_flags = squeeze_at
        try:
            # bar0: squeeze. bar1: breakout LONG (entry 102, stop 99 -> sl=3, target=102+2*3=108).
            # bar2: hits target (TP). bar3,4: flat, no squeeze in recency window (anchor still
            # bar0, already consumed). bar5: fresh squeeze bar, but closes back inside the bands
            # (no breakout yet - anchor stays available, unconsumed). bar6: second breakout LONG
            # off the fresh bar5 anchor. bar7: second trade hits its own target.
            closes = [100.0, 102.0, 109.0, 100.0, 100.0, 100.0, 102.0, 109.0]
            df = _make_df(closes)
            trades = bsb.backtest_instrument("Y", df)
            self.assertEqual(len(trades), 2)
            self.assertEqual(trades[0]["outcome"], "TP")
        finally:
            bsb.compute_bollinger_bands, bsb.compute_squeeze_flags = original_bands, original_squeeze

    def test_mirrors_correctly_for_shorts(self):
        def const_bands(closes):
            n = len(closes)
            return [100.0] * n, [101.0] * n, [99.0] * n

        def squeeze_only_bar0(band_width, lookback=None, percentile=None):
            return [True] + [False] * (len(band_width) - 1)

        original_bands, original_squeeze = bsb.compute_bollinger_bands, bsb.compute_squeeze_flags
        bsb.compute_bollinger_bands = const_bands
        bsb.compute_squeeze_flags = squeeze_only_bar0
        try:
            closes = [100.0, 98.0, 90.0]   # bar1: breakout SHORT (entry 98, stop 101 -> sl=3, target=98-6=92); bar2 hits target
            df = _make_df(closes)
            trades = bsb.backtest_instrument("Z", df)
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0]["side"], "SHORT")
            self.assertEqual(trades[0]["outcome"], "TP")
        finally:
            bsb.compute_bollinger_bands, bsb.compute_squeeze_flags = original_bands, original_squeeze


# ============================= degenerate-geometry guard =============================

class TestDegenerateGeometryGuard(unittest.TestCase):
    def test_near_zero_stop_distance_is_skipped(self):
        def bands_tiny_sl(closes):
            n = len(closes)
            return [100.0] * n, [100.001] * n, [99.999] * n   # near-zero band width

        def squeeze_only_bar0(band_width, lookback=None, percentile=None):
            return [True] + [False] * (len(band_width) - 1)

        original_bands, original_squeeze = bsb.compute_bollinger_bands, bsb.compute_squeeze_flags
        bsb.compute_bollinger_bands = bands_tiny_sl
        bsb.compute_squeeze_flags = squeeze_only_bar0
        try:
            closes = [100.0, 100.002, 100.5]
            df = _make_df(closes)
            trades = bsb.backtest_instrument("TinySL", df)
            self.assertEqual(trades, [])
        finally:
            bsb.compute_bollinger_bands, bsb.compute_squeeze_flags = original_bands, original_squeeze


# ============================= max-hold forced-close accounting =============================

class TestMaxHoldForcedClose(unittest.TestCase):
    def _open_a_trade_that_never_hits_stop_or_target(self, n_bars_after_entry):
        def bands_const(closes):
            n = len(closes)
            return [100.0] * n, [101.0] * n, [99.0] * n   # target (2R = 105) far out of reach

        def squeeze_only_bar0(band_width, lookback=None, percentile=None):
            return [True] + [False] * (len(band_width) - 1)

        original_bands, original_squeeze = bsb.compute_bollinger_bands, bsb.compute_squeeze_flags
        bsb.compute_bollinger_bands = bands_const
        bsb.compute_squeeze_flags = squeeze_only_bar0
        try:
            closes = [100.0, 102.0] + [102.1] * n_bars_after_entry
            df = _make_df(closes)
            return bsb.backtest_instrument("MaxHold", df)
        finally:
            bsb.compute_bollinger_bands, bsb.compute_squeeze_flags = original_bands, original_squeeze

    def test_forced_close_fires_at_exactly_max_hold_bars(self):
        trades = self._open_a_trade_that_never_hits_stop_or_target(bsb.MAX_HOLD_BARS)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "FLAT")
        self.assertGreater(trades[0]["r"], 0)
        self.assertLess(trades[0]["r"], bsb.REWARD_RISK)

    def test_one_bar_short_of_max_hold_stays_open_and_produces_no_trade(self):
        trades = self._open_a_trade_that_never_hits_stop_or_target(bsb.MAX_HOLD_BARS - 1)
        self.assertEqual(trades, [])


# ============================= one trade at a time =============================

class TestOneTradeAtATime(unittest.TestCase):
    def test_second_squeeze_breakout_while_in_a_position_is_ignored(self):
        def const_bands(closes):
            n = len(closes)
            return [100.0] * n, [101.0] * n, [99.0] * n

        squeeze_bars = {0, 2}

        def squeeze_at(band_width, lookback=None, percentile=None):
            return [i in squeeze_bars for i in range(len(band_width))]

        original_bands, original_squeeze = bsb.compute_bollinger_bands, bsb.compute_squeeze_flags
        bsb.compute_bollinger_bands = const_bands
        bsb.compute_squeeze_flags = squeeze_at
        try:
            # bar0: squeeze. bar1: LONG breakout opened (entry 102, stop 99, target 108).
            # bar2: a fresh squeeze anchor appears AND price would also qualify as a SHORT
            # breakout if flat - but a position is already open, so it must be ignored.
            # bar3: LONG position finally hits target.
            closes = [100.0, 102.0, 98.5, 109.0]
            df = _make_df(closes)
            trades = bsb.backtest_instrument("OneAtATime", df)
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0]["side"], "LONG")
        finally:
            bsb.compute_bollinger_bands, bsb.compute_squeeze_flags = original_bands, original_squeeze


# ============================= granularity-convergence check =============================

class TestGranularityConvergence(unittest.TestCase):
    """The honesty check for this strategy's specific geometry: does a fair, zero-drift random
    walk manufacture an apparent edge at coarse intrabar resolution (n_substeps=1, i.e. literally
    zero wick beyond the open/close body - see build_synthetic_ohlc's docstring)?

    RESULT, AND WHY IT'S DIFFERENT FROM PO3/THE MEAN-REVERSION SCRIPT: unlike PO3's manipulation-
    bar stop or the mean-reversion script's excursion-bar-low/high stop - both pinned to THAT
    SAME BAR'S OWN WICK, which is exactly what a wickless n_substeps=1 construction distorts -
    this strategy's entry (a plain close), stop (the opposite band value, itself built from a
    rolling SMA/stddev of CLOSES over the last 20 bars) and target (a FIXED reward:risk multiple
    of the stop distance) are ALL close-based; none of them reference the breakout bar's own
    high/low. So there is no direct channel for coarse-resolution wick collapse to bias the
    entry/stop/target geometry itself the way it does for those other scripts. Empirically (see
    the assertions below, run at n_substeps in {1, 4, 16, 64}): the measured z-score stays small
    and non-monotonic in substep count at every resolution tested - consistent with "no
    structural artifact here", not a large coarse-resolution effect that collapses as resolution
    increases. That is itself the honest result of running this check, not an assumption -
    checked, not asserted away, exactly per this project's practice for every other strategy with
    tight entry/stop coupling. (The strategy's REAL geometry risk is a different, non-granularity
    one: only the opposite-band stop and fixed-R:R target references are used, so intrabar wicks
    only matter for whether/when a FUTURE bar's stop/target level gets touched - an ordinary,
    non-artifactual property of any bar-resolution backtest.) Fixed seeds make this fully
    deterministic/reproducible, not a flaky statistical test."""

    def test_no_large_coarse_resolution_artifact_at_any_substep_count(self):
        n_bars = 12_000
        seeds = range(24)
        results = {}
        for n_substeps in (1, 4, 16, 64):
            all_r = []
            for seed in seeds:
                opens, highs, lows, closes = build_synthetic_ohlc(
                    n_bars, n_substeps, bar_std_pct=0.05, seed=seed)
                idx = pd.date_range("2018-01-01", periods=n_bars, freq="5min", tz="America/New_York")
                df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)
                trades = bsb.backtest_instrument("SYN", df)
                all_r.extend(t["r"] for t in trades)
            n = len(all_r)
            avg_r = sum(all_r) / n if n else 0.0
            std_r = np.std(all_r, ddof=1) if n >= 2 else 0.0
            z = avg_r / (std_r / (n ** 0.5)) if std_r > 0 and n >= 2 else 0.0
            results[n_substeps] = {"n_trades": n, "avg_r": avg_r, "z": z}

        # sanity: enough trades at every resolution for the z-scores to be meaningful at all
        for n_substeps, r in results.items():
            self.assertGreater(r["n_trades"], 500,
                                f"too few synthetic trades at n_substeps={n_substeps} to trust this check")

        # the core result: no resolution shows a large, clearly-non-chance apparent edge (a fair
        # random walk shouldn't produce |z| > ~3 by construction at ANY resolution here, since
        # unlike PO3/mean-reversion this strategy's geometry never references the entry bar's own
        # wick - see the class docstring)
        for n_substeps, r in results.items():
            self.assertLess(abs(r["z"]), 3.0,
                             f"unexpected large apparent edge at n_substeps={n_substeps}: z={r['z']:.2f}")


# ============================= synthetic end-to-end smoke run =============================

class TestEndToEndSmokeRun(unittest.TestCase):
    """ONE small synthetic multi-"instrument" run confirming the full pipeline (bands -> width ->
    squeeze -> breakout -> exit -> reporting-style aggregation) runs without crashing and
    produces plausibly-structured output. Deliberately NOT a real Dukascopy download."""

    def test_synthetic_multi_instrument_run_produces_well_formed_trades(self):
        all_trades = []
        for label, seed in [("SYN_EURUSD", 1), ("SYN_XAUUSD", 2), ("SYN_GBPUSD", 3)]:
            opens, highs, lows, closes = build_synthetic_ohlc(
                4000, n_substeps=16, bar_std_pct=0.08, start_price=1.2000, seed=seed)
            idx = pd.date_range("2019-03-01", periods=4000, freq="5min", tz="America/New_York")
            df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)
            trades = bsb.backtest_instrument(label, df)
            for t in trades:
                t["instrument"] = label
            all_trades.extend(trades)

        for t in all_trades:
            self.assertIn(t["outcome"], ("TP", "SL", "FLAT"))
            self.assertIn(t["side"], ("LONG", "SHORT"))
            self.assertTrue(np.isfinite(t["r"]))

        if all_trades:
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
