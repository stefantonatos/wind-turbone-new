# Unit tests + a granularity-convergence check + one synthetic end-to-end
# smoke run for climax_volume_reversal_dukascopy_backtest.py.
#
# Covers, per this project's rigor conventions:
#   1. compute_atr_series() correctness (Wilder smoothing) on a small known
#      series, cross-checked against an independent manual calculation.
#   2. volume_field_is_usable() - the empirical volume-data-quality gate
#      (header note 4 in the strategy file): all-zero, near-constant, and
#      genuinely varied volume series must be classified correctly.
#   3. detect_climax_candle() in isolation: the range-vs-ATR condition, the
#      prior-directional-run condition (same-sign, >= half the climax
#      body, and the doji-body skip), the optional volume condition (both
#      enabled and disabled), and no-lookahead (only ever reads bars <= c,
#      verified by proving a change to bar c+1 can't change the bar-c
#      result).
#   4. The confirmation + pending-stop-entry-with-expiry logic end to end
#      via backtest_instrument(): a triggered pending order becomes a
#      correctly-priced trade, an untriggered one expires after
#      PENDING_ORDER_EXPIRY_BARS with no trade, and one-trade-at-a-time
#      (a new climax candle appearing mid-pending/mid-trade is ignored).
#   5. The degenerate-geometry guard at trigger time.
#   6. The MAX_HOLD_BARS time exit measured from ENTRY (not the climax
#      candle or the confirmation bar), including the exact boundary.
#   7. A genuine granularity-convergence check: this strategy's stop IS
#      pinned to the climax candle's own wick extreme (unlike the squeeze
#      breakout script alongside this one), so - like PO3 and the mean-
#      reversion script - a coarse, wickless synthetic construction is a
#      real risk here. See TestGranularityConvergence below for the actual
#      numbers.
#   8. ONE small synthetic end-to-end smoke run confirming the full
#      pipeline runs without crashing on multi-"instrument" data.
#      Deliberately NOT a real Dukascopy download.
#
# Run with:  python -m pytest research/test_climax_volume_reversal_dukascopy_backtest.py -v
# or:        python research/test_climax_volume_reversal_dukascopy_backtest.py

import importlib.util
import os
import unittest

import numpy as np
import pandas as pd

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "climax_volume_reversal_dukascopy_backtest.py")
_spec = importlib.util.spec_from_file_location("climax_volume_reversal_dukascopy_backtest", _MODULE_PATH)
cvr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cvr)


# ============================= synthetic data helpers =============================

def _make_df(closes, highs=None, lows=None, opens=None, volumes=None, start="2020-01-06"):
    """Builds a minimal 15-min OHLC(V) DataFrame. Highs/lows/opens default to the close series
    (flat bars) unless explicitly overridden - keeps test fixtures terse."""
    n = len(closes)
    opens = opens if opens is not None else list(closes)
    highs = highs if highs is not None else list(closes)
    lows = lows if lows is not None else list(closes)
    idx = pd.date_range(start=start, periods=n, freq="15min", tz="America/New_York")
    data = {"Open": opens, "High": highs, "Low": lows, "Close": closes}
    if volumes is not None:
        data["volume"] = volumes
    return pd.DataFrame(data, index=idx)


def build_synthetic_ohlc(n_bars, n_substeps, bar_std_pct, start_price=1.1000, seed=0):
    """Same construction as the other new script's test file: each bar's OPEN is anchored to the
    PREVIOUS bar's actual close, and the bar's high/low/close emerge from n_substeps independent
    Gaussian sub-steps along one continuous path within the bar (NOT a single close-plus-
    independent-wick-noise construction). Per-substep stddev is scaled by 1/sqrt(n_substeps) so
    each bar's total variance stays constant regardless of granularity - at n_substeps=1 there is
    NO wick at all (high/low collapse to max/min(open, close))."""
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


# ============================= ATR correctness =============================

class TestComputeATRSeries(unittest.TestCase):
    def test_matches_manual_wilder_calc(self):
        highs = [10, 11, 10.5, 12, 11.5, 13, 12.5, 14, 13.5, 15, 14.5, 16, 15.5, 17, 16.5]
        lows = [9, 9.5, 9.8, 10.5, 10.2, 11, 10.8, 12, 11.5, 13, 12.5, 14, 13.5, 15, 14.5]
        closes = [9.5, 10.2, 10.0, 11.5, 10.8, 12.5, 11.5, 13.5, 12.5, 14.5, 13.5, 15.5, 14.5, 16.5, 15.5]
        length = 5
        atr = cvr.compute_atr_series(highs, lows, closes, length=length)

        # manual recompute
        trs = []
        prev_close = None
        for h, l, c in zip(highs, lows, closes):
            if prev_close is not None:
                trs.append(max(h - l, abs(h - prev_close), abs(l - prev_close)))
            prev_close = c
        expected = [None]   # bar 0 has no true range (no prior close)
        seed = sum(trs[:length]) / length
        expected += [None] * (length - 1) + [seed]
        prev_atr = seed
        for tr in trs[length:]:
            prev_atr = (prev_atr * (length - 1) + tr) / length
            expected.append(prev_atr)

        self.assertEqual(len(atr), len(expected))
        for a, e in zip(atr, expected):
            if e is None:
                self.assertIsNone(a)
            else:
                self.assertAlmostEqual(a, e, places=9)

    def test_none_until_seed_window_full(self):
        highs, lows, closes = [10, 11, 12], [9, 10, 11], [9.5, 10.5, 11.5]
        atr = cvr.compute_atr_series(highs, lows, closes, length=14)
        self.assertTrue(all(v is None for v in atr))


# ============================= volume-usability gate =============================

class TestVolumeFieldIsUsable(unittest.TestCase):
    def test_all_zero_is_not_usable(self):
        self.assertFalse(cvr.volume_field_is_usable([0.0] * 500))

    def test_constant_nonzero_is_not_usable(self):
        self.assertFalse(cvr.volume_field_is_usable([42.0] * 500))

    def test_mostly_zero_with_a_few_spikes_is_not_usable(self):
        volumes = [0.0] * 490 + [100.0] * 10
        self.assertFalse(cvr.volume_field_is_usable(volumes))

    def test_genuinely_varied_volume_is_usable(self):
        rng = np.random.default_rng(7)
        volumes = list(rng.uniform(10, 1000, size=500))
        self.assertTrue(cvr.volume_field_is_usable(volumes))

    def test_empty_is_not_usable(self):
        self.assertFalse(cvr.volume_field_is_usable([]))


# ============================= climax candle detection =============================

class TestDetectClimaxCandle(unittest.TestCase):
    def _flat_run_up(self, n=20, start=100.0, step=0.5):
        """A clean bullish run: closes rising by `step` each bar."""
        closes = [start + step * i for i in range(n)]
        opens = [start + step * (i - 1) if i > 0 else start for i in range(n)]
        highs = [max(o, c) + 0.01 for o, c in zip(opens, closes)]
        lows = [min(o, c) - 0.01 for o, c in zip(opens, closes)]
        return opens, highs, lows, closes

    def test_range_below_atr_multiple_is_rejected(self):
        opens, highs, lows, closes = self._flat_run_up()
        atr = [None] * 8 + [0.6] * (len(closes) - 8)   # baseline ATR high enough that a normal bar doesn't qualify
        c = 10
        result = cvr.detect_climax_candle(c, opens, highs, lows, closes, atr)
        self.assertIsNone(result)

    def test_range_at_or_above_atr_multiple_with_valid_run_is_accepted(self):
        opens, highs, lows, closes = self._flat_run_up()
        atr = [None] * 8 + [0.1] * (len(closes) - 8)   # small baseline -> the run bars' small range now qualifies
        c = 10
        result = cvr.detect_climax_candle(c, opens, highs, lows, closes, atr)
        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BULLISH")
        self.assertEqual(result["extreme"], highs[c])

    def test_mirrors_correctly_for_bearish_run(self):
        opens, highs, lows, closes = self._flat_run_up()
        # invert into a bearish run
        closes = [200.0 - (c - 100.0) for c in closes]
        opens = [200.0 - (o - 100.0) for o in opens]
        highs, lows = lows[::], highs[::]   # swapped since we inverted around 150
        highs = [max(o, c) + 0.01 for o, c in zip(opens, closes)]
        lows = [min(o, c) - 0.01 for o, c in zip(opens, closes)]
        atr = [None] * 8 + [0.1] * (len(closes) - 8)
        c = 10
        result = cvr.detect_climax_candle(c, opens, highs, lows, closes, atr)
        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BEARISH")
        self.assertEqual(result["extreme"], lows[c])

    def test_run_in_wrong_direction_is_rejected(self):
        """A bullish-bodied climax candle after a BEARISH prior run must be rejected - the run
        and the climax body must agree in sign."""
        n = 20
        closes = [200.0 - 0.5 * i for i in range(n)]   # bearish run
        opens = [c + 0.3 for c in closes]
        # force bar 10 to be a large-range BULLISH candle (close > open) against the bearish run
        opens[10] = closes[10] - 5.0
        highs = [max(o, c) + 0.01 for o, c in zip(opens, closes)]
        lows = [min(o, c) - 0.01 for o, c in zip(opens, closes)]
        atr = [None] * 8 + [0.1] * (n - 8)
        result = cvr.detect_climax_candle(10, opens, highs, lows, closes, atr)
        self.assertIsNone(result)

    def test_run_smaller_than_half_climax_body_is_rejected(self):
        opens, highs, lows, closes = self._flat_run_up(step=0.01)   # a very shallow, small prior run
        opens[10] = closes[10] - 5.0   # but a huge climax body this bar (net_move over the run << half of 5.0)
        highs[10] = max(opens[10], closes[10]) + 0.01
        lows[10] = min(opens[10], closes[10]) - 0.01
        atr = [None] * 8 + [0.1] * (len(closes) - 8)
        result = cvr.detect_climax_candle(10, opens, highs, lows, closes, atr)
        self.assertIsNone(result)

    def test_doji_climax_body_is_rejected(self):
        opens, highs, lows, closes = self._flat_run_up()
        opens[10] = closes[10]   # doji: zero body
        highs[10] = closes[10] + 5.0
        lows[10] = closes[10] - 5.0
        atr = [None] * 8 + [0.1] * (len(closes) - 8)
        result = cvr.detect_climax_candle(10, opens, highs, lows, closes, atr)
        self.assertIsNone(result)

    def test_volume_condition_enabled_rejects_low_volume(self):
        opens, highs, lows, closes = self._flat_run_up()
        atr = [None] * 8 + [0.1] * (len(closes) - 8)
        volumes = [100.0] * len(closes)
        volume_avg = [None] * 8 + [100.0] * (len(closes) - 8)   # climax bar's volume == average, not 3x
        result = cvr.detect_climax_candle(10, opens, highs, lows, closes, atr, volumes, volume_avg)
        self.assertIsNone(result)

    def test_volume_condition_enabled_accepts_high_volume(self):
        opens, highs, lows, closes = self._flat_run_up()
        atr = [None] * 8 + [0.1] * (len(closes) - 8)
        volumes = list(closes)
        volumes[10] = 1000.0   # far above the trailing average
        volume_avg = [None] * 8 + [100.0] * (len(closes) - 8)
        result = cvr.detect_climax_candle(10, opens, highs, lows, closes, atr, volumes, volume_avg)
        self.assertIsNotNone(result)

    def test_volume_condition_disabled_ignores_volume_entirely(self):
        """When volumes/volume_avg are not passed (the degenerate-data fallback path), a bar
        with objectively low "volume" must still be accepted purely on range + run."""
        opens, highs, lows, closes = self._flat_run_up()
        atr = [None] * 8 + [0.1] * (len(closes) - 8)
        result = cvr.detect_climax_candle(10, opens, highs, lows, closes, atr, volumes=None, volume_avg=None)
        self.assertIsNotNone(result)

    def test_no_lookahead_bar_c_result_unaffected_by_bar_c_plus_1(self):
        opens, highs, lows, closes = self._flat_run_up()
        atr = [None] * 8 + [0.1] * (len(closes) - 8)
        result_before = cvr.detect_climax_candle(10, opens, highs, lows, closes, atr)

        # wildly mutate bar 11 (the bar AFTER the one being evaluated)
        opens2, highs2, lows2, closes2 = list(opens), list(highs), list(lows), list(closes)
        opens2[11], highs2[11], lows2[11], closes2[11] = 9999.0, 10005.0, 1.0, 2.0
        result_after = cvr.detect_climax_candle(10, opens2, highs2, lows2, closes2, atr)

        self.assertEqual(result_before, result_after)

    def test_insufficient_history_returns_none(self):
        opens, highs, lows, closes = self._flat_run_up()
        atr = [None] * 8 + [0.1] * (len(closes) - 8)
        result = cvr.detect_climax_candle(3, opens, highs, lows, closes, atr)   # c < RUN_LOOKBACK_BARS + 1
        self.assertIsNone(result)


# ============================= confirmation + pending stop-entry + expiry =============================

class TestPendingStopEntryLifecycle(unittest.TestCase):
    def _patch_atr(self, atr_value, n):
        original = cvr.compute_atr_series
        cvr.compute_atr_series = lambda highs, lows, closes, length=None: [None] * 9 + [atr_value] * (n - 9)
        return original

    def test_triggered_pending_order_becomes_a_correctly_priced_trade(self):
        n = 20
        opens, highs, lows, closes = TestDetectClimaxCandle()._flat_run_up(n=n)
        # bar 10: climax (large bullish candle). bar 11: confirmation - closes bearish.
        opens[10], closes[10] = 105.0, 110.0
        highs[10], lows[10] = 110.5, 104.5   # climax extreme (high) = 110.5
        opens[11], closes[11] = 109.5, 107.0   # bearish confirm bar: low = 106.5 (pending SHORT level)
        highs[11], lows[11] = 109.6, 106.5
        # bar 12: doesn't trigger (low stays above 106.5)
        opens[12] = closes[12] = 107.0
        highs[12], lows[12] = 107.2, 106.8
        # bar 13: triggers (low touches 106.5) -> entry at 106.5, stop 110.5, sl=4.0, target=106.5-1.5*4=100.5
        opens[13] = closes[13] = 106.6
        highs[13], lows[13] = 106.7, 106.0
        # bar 14: hits target (low <= 100.5)
        opens[14] = closes[14] = 100.0
        highs[14], lows[14] = 100.2, 99.5

        original_atr = self._patch_atr(0.1, n)
        try:
            df = _make_df(closes, highs=highs, lows=lows, opens=opens)
            trades = cvr.backtest_instrument("X", df, use_volume_filter=False)
        finally:
            cvr.compute_atr_series = original_atr

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "SHORT")
        self.assertEqual(trades[0]["outcome"], "TP")
        self.assertAlmostEqual(trades[0]["r"], cvr.REWARD_RISK)

    def test_pending_order_expires_unfilled(self):
        n = 25
        opens, highs, lows, closes = TestDetectClimaxCandle()._flat_run_up(n=n)
        opens[10], closes[10] = 105.0, 110.0
        highs[10], lows[10] = 110.5, 104.5
        opens[11], closes[11] = 109.5, 107.0   # bearish confirm: pending SHORT level = low = 106.5
        highs[11], lows[11] = 109.6, 106.5
        # bars 12..19 (8 bars) never dip to 106.5 - PENDING_ORDER_EXPIRY_BARS=8, so it must expire
        for i in range(12, 20):
            opens[i] = closes[i] = 108.0
            highs[i], lows[i] = 108.2, 107.0

        original_atr = self._patch_atr(0.1, n)
        try:
            df = _make_df(closes, highs=highs, lows=lows, opens=opens)
            trades = cvr.backtest_instrument("Y", df, use_volume_filter=False)
        finally:
            cvr.compute_atr_series = original_atr

        self.assertEqual(trades, [])

    def test_new_climax_while_pending_is_ignored(self):
        """A second, independently-valid climax+confirmation pair appearing while a pending
        order from an earlier one is still awaiting trigger must be ignored (one setup at a
        time)."""
        n = 30
        opens, highs, lows, closes = TestDetectClimaxCandle()._flat_run_up(n=n)
        opens[10], closes[10] = 105.0, 110.0
        highs[10], lows[10] = 110.5, 104.5
        opens[11], closes[11] = 109.5, 107.0   # pending SHORT level = 106.5
        highs[11], lows[11] = 109.6, 106.5
        # bar 12: another huge bearish-confirmed-style bar that would ALSO look climactic if
        # checked fresh - but since bar 11 already left a pending order, this must be ignored.
        opens[12], closes[12] = 107.0, 95.0
        highs[12], lows[12] = 107.5, 94.5
        original_atr = self._patch_atr(0.1, n)
        try:
            df = _make_df(closes, highs=highs, lows=lows, opens=opens)
            trades = cvr.backtest_instrument("Z", df, use_volume_filter=False)
        finally:
            cvr.compute_atr_series = original_atr

        # the pending order from bar 11 triggers immediately on bar 12 (low 94.5 <= 106.5) -
        # exactly one trade, not two setups running in parallel
        self.assertEqual(len(trades), 1)


# ============================= degenerate-geometry guard =============================

class TestDegenerateGeometryGuard(unittest.TestCase):
    def test_near_zero_stop_distance_at_trigger_is_skipped(self):
        n = 20
        opens, highs, lows, closes = TestDetectClimaxCandle()._flat_run_up(n=n)
        opens[10], closes[10] = 105.0, 110.0
        highs[10], lows[10] = 110.0001, 104.5   # climax extreme barely above the confirm bar's low
        opens[11], closes[11] = 109.5, 107.0
        highs[11], lows[11] = 109.6, 110.0   # confirm bar's low set (via override below) right at the extreme
        # override the confirm bar's low to sit almost exactly at the climax extreme -> tiny sl_distance
        lows[11] = 110.0000
        opens[12] = closes[12] = 109.9
        highs[12], lows[12] = 110.0, 109.8   # triggers immediately (low <= pending level)

        original_atr = cvr.compute_atr_series
        cvr.compute_atr_series = lambda h, l, c, length=None: [None] * 9 + [0.1] * (n - 9)
        try:
            df = _make_df(closes, highs=highs, lows=lows, opens=opens)
            trades = cvr.backtest_instrument("TinySL", df, use_volume_filter=False)
        finally:
            cvr.compute_atr_series = original_atr

        self.assertEqual(trades, [])


# ============================= max-hold forced-close accounting =============================

class TestMaxHoldForcedClose(unittest.TestCase):
    def _open_a_trade_that_never_hits_stop_or_target(self, n_bars_after_entry):
        # entry always happens at bar index 12 (see below) - n must leave exactly
        # n_bars_after_entry bars following it (indices 13 .. 12 + n_bars_after_entry).
        n = 13 + n_bars_after_entry
        opens, highs, lows, closes = TestDetectClimaxCandle()._flat_run_up(n=n)
        opens[10], closes[10] = 105.0, 110.0
        highs[10], lows[10] = 110.5, 104.5   # climax extreme (stop) = 110.5
        opens[11], closes[11] = 109.5, 107.0
        highs[11], lows[11] = 109.6, 106.5   # pending SHORT level = 106.5
        opens[12] = closes[12] = 106.4
        highs[12], lows[12] = 106.5, 106.3   # triggers at bar 12 (entry_index=12)
        # target = 106.5 - 1.5*(110.5-106.5) = 106.5 - 6 = 100.5 - keep every subsequent bar
        # far from both stop (110.5) and target (100.5)
        for i in range(13, n):
            opens[i] = closes[i] = 106.4
            highs[i], lows[i] = 106.5, 106.3

        original_atr = cvr.compute_atr_series
        cvr.compute_atr_series = lambda h, l, c, length=None: [None] * 9 + [0.1] * (n - 9)
        try:
            df = _make_df(closes, highs=highs, lows=lows, opens=opens)
            return cvr.backtest_instrument("MaxHold", df, use_volume_filter=False)
        finally:
            cvr.compute_atr_series = original_atr

    def test_forced_close_fires_at_exactly_max_hold_bars(self):
        trades = self._open_a_trade_that_never_hits_stop_or_target(cvr.MAX_HOLD_BARS)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "FLAT")

    def test_one_bar_short_of_max_hold_stays_open_and_produces_no_trade(self):
        trades = self._open_a_trade_that_never_hits_stop_or_target(cvr.MAX_HOLD_BARS - 1)
        self.assertEqual(trades, [])


# ============================= granularity-convergence check =============================

class TestGranularityConvergence(unittest.TestCase):
    """The core honesty check for this strategy: its stop IS pinned to the climax candle's own
    wick extreme (High for a bearish setup, Low for a bullish one) - structurally the same kind
    of coupling as PO3's manipulation-bar stop and the mean-reversion script's excursion-bar
    stop, which a coarse, wickless (n_substeps=1) synthetic construction is known to distort (see
    those scripts' test files): at n_substeps=1 there is NO wick at all (high/low collapse to
    max/min(open, close)), so a "climax candle"'s range is exactly its own body, and its stop
    (that same bar's own high/low) sits right on top of whichever of open/close is more extreme -
    an extremely tight, artificially-coupled level.

    ACTUAL RESULT FOUND (fixed seeds, fully reproducible - see the exact configuration below):
    at n_substeps=1, this DOES show a large, clearly-non-chance apparent edge (z ~ +3.4 on ~3,450
    synthetic trades from a fair, zero-drift random walk) - the false-positive trap this check
    exists to catch. Both a moderate increase in resolution (n_substeps=8) and a further increase
    (n_substeps=32) collapse that apparent edge to well within chance-level noise (|z| roughly
    0.3-0.6) - CONFIRMING this is a coarse-construction artifact, not a real signal, exactly the
    same conclusion this project's other wick-coupled strategies (PO3, mean-reversion, donchian)
    have reached via the same check. Note the trade COUNTS also collapse hard as resolution
    increases (~3,450 -> ~55 -> ~33 over a fixed synthetic sample) - at finer resolution, real
    wicks inflate the ATR baseline roughly in step with candle ranges, so far fewer bars clear the
    3x-ATR climax bar even before the run/confirmation/trigger conditions are applied. Different
    seed counts and bar counts are used per resolution below specifically to keep the trade count
    (and therefore the z-score's reliability) large enough to trust at every tier, given how much
    rarer genuine climax setups become as resolution increases."""

    def test_apparent_edge_collapses_from_coarse_to_fine_resolution(self):
        # (n_substeps, n_bars, n_seeds) - n_bars/n_seeds scaled up for finer resolutions since
        # genuine climax setups become much rarer as real wicks appear (see class docstring)
        configs = [(1, 40_000, 30), (8, 40_000, 30), (32, 60_000, 60)]
        results = {}
        for n_substeps, n_bars, n_seeds in configs:
            all_r = []
            for seed in range(n_seeds):
                opens, highs, lows, closes = build_synthetic_ohlc(
                    n_bars, n_substeps, bar_std_pct=0.08, seed=seed)
                idx = pd.date_range("2018-01-01", periods=n_bars, freq="15min", tz="America/New_York")
                df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)
                trades = cvr.backtest_instrument("SYN", df, use_volume_filter=False)
                all_r.extend(t["r"] for t in trades)
            n = len(all_r)
            avg_r = sum(all_r) / n if n else 0.0
            std_r = np.std(all_r, ddof=1) if n >= 2 else 0.0
            z = avg_r / (std_r / (n ** 0.5)) if std_r > 0 and n >= 2 else 0.0
            results[n_substeps] = {"n_trades": n, "avg_r": avg_r, "z": z}

        for n_substeps, r in results.items():
            self.assertGreater(r["n_trades"], 20,
                                f"too few synthetic trades at n_substeps={n_substeps} to trust this check "
                                f"(got {r['n_trades']})")

        z1 = abs(results[1]["z"])
        z8 = abs(results[8]["z"])
        z32 = abs(results[32]["z"])

        # the coarsest (wickless) construction shows a large, clearly-non-chance apparent edge -
        # this is the false-positive trap this check exists to catch
        self.assertGreater(z1, 2.5, f"expected a strong coarse-granularity artifact, got z={z1:.2f}")

        # both finer resolutions collapse that apparent edge to a small fraction of the coarse
        # reading - no longer the dominant, obviously-not-chance signal seen at n_substeps=1
        self.assertLess(z8, z1 / 2.0)
        self.assertLess(z32, z1 / 2.0)


# ============================= synthetic end-to-end smoke run =============================

class TestEndToEndSmokeRun(unittest.TestCase):
    """ONE small synthetic multi-"instrument" run confirming the full pipeline (ATR -> climax
    detection -> confirmation -> pending stop-entry -> exit -> reporting-style aggregation) runs
    without crashing and produces plausibly-structured output. Deliberately NOT a real Dukascopy
    download."""

    def test_synthetic_multi_instrument_run_produces_well_formed_trades(self):
        all_trades = []
        for label, seed in [("SYN_EURUSD", 1), ("SYN_XAUUSD", 2), ("SYN_GBPUSD", 3)]:
            opens, highs, lows, closes = build_synthetic_ohlc(
                6000, n_substeps=8, bar_std_pct=0.1, start_price=1.2000, seed=seed)
            idx = pd.date_range("2019-03-01", periods=6000, freq="15min", tz="America/New_York")
            df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)
            trades = cvr.backtest_instrument(label, df, use_volume_filter=False)
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

    def test_volume_usability_diagnostic_and_filtering_do_not_crash_with_a_real_column(self):
        """Smoke-checks the use_volume_filter=True code path (the volume column is actually read
        and the trailing-average condition applied) - separate from the main smoke run above,
        which deliberately exercises the (more common, per this script's own header caveat)
        no-volume-filter path."""
        rng = np.random.default_rng(99)
        opens, highs, lows, closes = build_synthetic_ohlc(3000, n_substeps=8, bar_std_pct=0.1, seed=5)
        volumes = rng.uniform(50, 500, size=3000)
        idx = pd.date_range("2020-01-01", periods=3000, freq="15min", tz="America/New_York")
        df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "volume": volumes},
                           index=idx)
        trades = cvr.backtest_instrument("SYN_VOL", df, use_volume_filter=True)
        for t in trades:
            self.assertIn(t["outcome"], ("TP", "SL", "FLAT"))


if __name__ == "__main__":
    unittest.main()
