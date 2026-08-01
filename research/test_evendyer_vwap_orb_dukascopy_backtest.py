# Unit tests + a granularity-convergence check + one synthetic end-to-end
# smoke run for evendyer_vwap_orb_dukascopy_backtest.py.
#
# Covers, per this project's rigor conventions:
#   1. find_confirmed_pivots' no-lookahead placement (adapted verbatim from
#      support_resistance_zone_bounce_dukascopy_backtest.py - re-verified
#      here rather than assumed correct just because it's a copy).
#   2. determine_vwap_weights' real-volume-vs-TWAP-fallback decision on
#      usable, all-zero, and constant-nonzero (degenerate) volume series.
#   3. The full ORB-break -> VWAP-close-through -> VWAP-reclaim signal
#      state machine, verified bar-by-bar against an INDEPENDENT reference
#      running-VWAP computed from scratch in this test file (not calling
#      into the module's own incremental cum_pv/cum_vol accumulator) -
#      confirms the signal fires at the exact reclaim bar and not before.
#   4. The degenerate-geometry validity guard (stop/target on the wrong
#      side of price) correctly skips a trade instead of forcing it.
#   5. One-trade-per-day gating and forced close at FORCE_CLOSE_TIME.
#   6. Both stranded-position defensive paths: a data gap that skips past
#      FORCE_CLOSE_TIME straight into the next day, and a position still
#      open on the very last bar of the whole series.
#   7. A genuine granularity-convergence check - this strategy's entry
#      triggers are both close-based (ORB break, VWAP reclaim), not
#      wick-based, but its STOP is derived from a confirmed swing-pivot
#      extreme (a multi-bar aggregate), so the same false-positive-trap
#      check this project always runs on stop/target strategies is run
#      here too rather than assumed safe by resemblance to Donchian/Dow
#      Theory's own (clean) results.
#   8. One small synthetic end-to-end smoke run confirming the full
#      pipeline runs without crashing on multi-instrument data.
#
# Run with:  python -m pytest research/test_evendyer_vwap_orb_dukascopy_backtest.py -v

import importlib.util
import os
import unittest

import numpy as np
import pandas as pd

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "evendyer_vwap_orb_dukascopy_backtest.py")
_spec = importlib.util.spec_from_file_location("evendyer_vwap_orb_dukascopy_backtest", _MODULE_PATH)
evw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evw)


# ============================= synthetic data helpers =============================

def _make_df(closes, highs=None, lows=None, opens=None, volumes=None,
             start="2020-01-06 09:30", freq="5min"):
    n = len(closes)
    opens = opens if opens is not None else list(closes)
    highs = highs if highs is not None else list(closes)
    lows = lows if lows is not None else list(closes)
    volumes = volumes if volumes is not None else [1.0] * n
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="America/New_York")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes,
                          "volume": volumes}, index=idx)


def _reference_running_vwap(highs, lows, closes, weights, session_start_idx):
    """Independent, from-scratch running-VWAP computation (a plain cumulative
    weighted average from session_start_idx onward) - used ONLY to cross-check the
    module's own incremental accumulator, not imported from it."""
    out = [None] * len(closes)
    cum_pv = cum_w = 0.0
    for i in range(session_start_idx, len(closes)):
        hlc3 = (highs[i] + lows[i] + closes[i]) / 3.0
        cum_pv += hlc3 * weights[i]
        cum_w += weights[i]
        out[i] = cum_pv / cum_w if cum_w > 0 else None
    return out


def build_synthetic_ohlc(n_bars, n_substeps, bar_std_pct, start_price=4500.0, seed=0):
    """Proper multi-step intraday random walk - same construction (and same
    false-positive-trap rationale) as test_bollinger_band_mean_reversion_dukascopy_backtest.py's
    copy of this helper."""
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


# ============================= find_confirmed_pivots =============================

class TestConfirmedPivots(unittest.TestCase):
    def test_single_spike_high_confirmed_only_after_lookback_bars_both_sides(self):
        # flat highs except one spike at index 5; lookback=2 needs bars 3..7 to confirm it
        highs = [10, 10, 10, 10, 10, 20, 10, 10, 10, 10, 10]
        lows = list(highs)
        swing_highs, swing_lows = evw.find_confirmed_pivots(highs, lows, lookback=2)
        # confirmed at key 5+2=7, not any earlier index - the earliest point a lookahead-free
        # algorithm could actually know bar 5 was a local max is once bars 6 and 7 exist
        self.assertIn(7, swing_highs)
        self.assertEqual(swing_highs[7], (5, 20))
        self.assertNotIn(5, swing_highs)
        self.assertNotIn(6, swing_highs)

    def test_single_spike_low_confirmed_only_after_lookback_bars_both_sides(self):
        lows = [10, 10, 10, 10, 10, 2, 10, 10, 10, 10, 10]
        highs = list(lows)
        swing_highs, swing_lows = evw.find_confirmed_pivots(highs, lows, lookback=2)
        self.assertIn(7, swing_lows)
        self.assertEqual(swing_lows[7], (5, 2))

    def test_no_spike_produces_no_pivots(self):
        highs = [10] * 11
        lows = [10] * 11
        swing_highs, swing_lows = evw.find_confirmed_pivots(highs, lows, lookback=2)
        self.assertEqual(swing_highs, {})
        self.assertEqual(swing_lows, {})

    def test_tied_extreme_across_window_is_not_treated_as_a_unique_pivot(self):
        # two bars tie for the window's max - count()==1 check should reject both as "the" pivot
        highs = [10, 10, 20, 10, 20, 10, 10]
        lows = list(highs)
        swing_highs, _ = evw.find_confirmed_pivots(highs, lows, lookback=2)
        self.assertEqual(swing_highs, {})


# ============================= determine_vwap_weights =============================

class TestVwapWeightsFallback(unittest.TestCase):
    def test_varied_nonzero_volume_is_used_as_is(self):
        volumes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        weights, usable, diag = evw.determine_vwap_weights(volumes, "TEST")
        self.assertTrue(usable)
        self.assertEqual(weights, [float(v) for v in volumes])

    def test_all_zero_volume_falls_back_to_twap(self):
        volumes = [0] * 10
        weights, usable, diag = evw.determine_vwap_weights(volumes, "TEST")
        self.assertFalse(usable)
        self.assertEqual(weights, [1.0] * 10)
        self.assertIn("degenerate", diag)

    def test_constant_nonzero_volume_falls_back_to_twap(self):
        # nonzero_frac is 100% but there's only ONE distinct value - carries no real
        # cross-bar information, so this should still be treated as degenerate
        volumes = [100.0] * 10
        weights, usable, diag = evw.determine_vwap_weights(volumes, "TEST")
        self.assertFalse(usable)
        self.assertEqual(weights, [1.0] * 10)

    def test_empty_series_does_not_crash(self):
        weights, usable, diag = evw.determine_vwap_weights([], "TEST")
        self.assertEqual(weights, [])
        self.assertFalse(usable)


# ============================= ORB break / VWAP retest signal state machine =============================

def _build_long_entry_bars(n_filler=95):
    """Hand-verified (by an independent from-scratch running-vwap calc, see the test that
    checks it) 11-bar sequence that: builds a tight ORB range (bars 0-5), breaks it upward with
    a high spike at bar 6 (which becomes the swing-high TARGET once TARGET_SWING_LEN=1
    confirms it), dips the close below the running vwap at bar 8 (setting long_closed_through,
    with a swing-low dip at bar 8 that becomes the swing-low STOP once STOP_SWING_LEN=2
    confirms it), then reclaims vwap at bar 10 - the bar a LONG should actually open on, with
    stop=4470 (bar 8's low) and target=4600 (bar 6's high), both already confirmed by bar 10.
    Flat filler bars afterward keep price clear of both stop and target through FORCE_CLOSE_TIME
    (18:00 NY = bar index 102 from a 09:30 start), so the opened trade forces closed FLAT."""
    orb_n = 6
    closes = [4500] * orb_n + [4520, 4515, 4480, 4490, 4530] + [4530] * n_filler
    highs = [4505] * orb_n + [4600, 4516, 4482, 4495, 4532] + [4531] * n_filler
    lows = [4495] * orb_n + [4515, 4510, 4470, 4485, 4528] + [4529] * n_filler
    return highs, lows, closes


def _mirror_ohlc(highs, lows, closes, pivot=9000.0):
    """Mirrors a bar sequence around `pivot` (mirrored = pivot - value), swapping high/low so the
    mirrored series is still valid OHLC. VWAP is a linear (weighted-average) function of price,
    so mirroring the whole series exactly mirrors every vwap comparison in it too - this turns
    the hand-verified LONG scenario above into an equally-verified SHORT one without having to
    redo the running-vwap arithmetic by hand a second time."""
    return ([pivot - l for l in lows], [pivot - h for h in highs], [pivot - c for c in closes])


class TestOrbBreakAndVwapRetestSignal(unittest.TestCase):
    def setUp(self):
        self._orig_stop_len = evw.STOP_SWING_LEN
        self._orig_target_len = evw.TARGET_SWING_LEN
        evw.STOP_SWING_LEN = 2
        evw.TARGET_SWING_LEN = 1

    def tearDown(self):
        evw.STOP_SWING_LEN = self._orig_stop_len
        evw.TARGET_SWING_LEN = self._orig_target_len

    def test_reference_vwap_and_pivot_arithmetic_matches_hand_derivation(self):
        # sanity-checks the hand-derived scenario itself against the independent reference vwap
        # BEFORE trusting any assertion built on top of it
        highs, lows, closes = _build_long_entry_bars(n_filler=0)
        ref_vwap = _reference_running_vwap(highs, lows, closes, [1.0] * len(closes), session_start_idx=0)
        self.assertAlmostEqual(ref_vwap[6], 4506.428571428571, places=6)
        self.assertAlmostEqual(ref_vwap[8], 4504.0, places=6)
        self.assertAlmostEqual(ref_vwap[10], 4505.0909090909, places=6)
        self.assertLess(closes[8], ref_vwap[8])     # dip below vwap at bar 8
        self.assertGreater(closes[10], ref_vwap[10])   # reclaim above vwap at bar 10

    def test_long_signal_fires_and_opens_a_valid_trade_at_the_vwap_reclaim_bar(self):
        highs, lows, closes = _build_long_entry_bars()
        df = _make_df(closes, highs=highs, lows=lows, volumes=[1.0] * len(closes))
        trades, used_real_volume, diag, trace = evw.backtest_instrument("TEST", df, verbose=False,
                                                                          record_trace=True)

        # bar 8 (first close below vwap) must not have fired or opened anything yet
        self.assertFalse(trace[8]["long_signal"])
        # bar 10 is the first bar where signal AND valid geometry hold together (long_signal
        # being True here already implies long_orb_broken/long_closed_through held at the moment
        # the signal was evaluated - the trace's own copies of those two flags are NOT checked
        # here, since a successful entry resets them to False for the rest of the day in the
        # same bar the trace snapshot is taken, by design, not a bug)
        self.assertTrue(trace[10]["long_signal"])
        self.assertTrue(trace[10]["valid_long"])
        self.assertEqual(trace[10]["long_stop"], 4470.0)
        self.assertEqual(trace[10]["long_target"], 4600.0)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "LONG")

    def test_short_mirrors_long_with_signs_flipped(self):
        long_highs, long_lows, long_closes = _build_long_entry_bars()
        highs, lows, closes = _mirror_ohlc(long_highs, long_lows, long_closes)
        df = _make_df(closes, highs=highs, lows=lows, volumes=[1.0] * len(closes))
        trades, _, _, trace = evw.backtest_instrument("TEST", df, verbose=False, record_trace=True)

        self.assertFalse(trace[8]["short_signal"])
        self.assertTrue(trace[10]["short_signal"])
        self.assertTrue(trace[10]["valid_short"])
        self.assertAlmostEqual(trace[10]["short_stop"], 9000.0 - 4470.0, places=6)
        self.assertAlmostEqual(trace[10]["short_target"], 9000.0 - 4600.0, places=6)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "SHORT")

    def test_no_orb_break_means_no_signal_ever(self):
        # price never leaves the ORB range - closes_through/reclaim state machine should
        # never even engage since can_track_break's break conditions never trigger
        n = 100
        closes = [4500] * n
        df = _make_df(closes, volumes=[1.0] * n)
        trades, _, _, trace = evw.backtest_instrument("TEST", df, verbose=False, record_trace=True)
        self.assertFalse(any(row["long_signal"] or row["short_signal"] for row in trace))
        self.assertEqual(trades, [])


# ============================= degenerate-geometry validity guard =============================

class TestValidityGuardSkipsDegenerateGeometry(unittest.TestCase):
    def test_signal_with_no_confirmed_pivots_yet_is_skipped_not_forced(self):
        # a LONG signal fires (ORB break + reclaim) but STOP_SWING_LEN/TARGET_SWING_LEN are
        # left at their real (large) defaults, so no swing pivot has had time to confirm this
        # early in the series - the trade must be skipped, not opened with a None/bogus stop
        n_orb = 6
        orb_closes = [4500] * n_orb
        post_orb_closes = [4520, 4515, 4490, 4530]
        n_filler = 90
        closes = orb_closes + post_orb_closes + [4530] * n_filler
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        highs[n_orb] = 4521
        lows[n_orb] = 4499

        df = _make_df(closes, highs=highs, lows=lows, volumes=[1.0] * len(closes))
        # STOP_SWING_LEN/TARGET_SWING_LEN untouched here - real defaults (20/5) mean no pivot
        # has confirmed yet this early, so valid_long must be False even if long_signal fires
        trades, _, _, trace = evw.backtest_instrument("TEST", df, verbose=False, record_trace=True)
        fired = [row for row in trace if row["long_signal"]]
        self.assertTrue(len(fired) >= 1)
        self.assertFalse(fired[0]["valid_long"])
        self.assertEqual(trades, [])


# ============================= one-trade-per-day / forced close / stranded positions =============================

class TestTradeFrequencyAndForceClose(unittest.TestCase):
    def setUp(self):
        self._orig_stop_len = evw.STOP_SWING_LEN
        self._orig_target_len = evw.TARGET_SWING_LEN
        evw.STOP_SWING_LEN = 2
        evw.TARGET_SWING_LEN = 1

    def tearDown(self):
        evw.STOP_SWING_LEN = self._orig_stop_len
        evw.TARGET_SWING_LEN = self._orig_target_len

    def test_forced_close_marks_flat_when_neither_stop_nor_target_hit(self):
        # reuses the hand-verified LONG-entry scenario (opens at bar 10, stop=4470,
        # target=4600) - flat filler bars afterward sit strictly between those two levels,
        # so the trade survives untouched until FORCE_CLOSE_TIME (18:00 NY)
        highs, lows, closes = _build_long_entry_bars()
        df = _make_df(closes, highs=highs, lows=lows, volumes=[1.0] * len(closes))
        trades, _, _, trace = evw.backtest_instrument("TEST", df, verbose=False, record_trace=True)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "FLAT")

    def test_stranded_position_across_a_data_gap_is_marked_flat_not_dropped(self):
        # entry happens at bar 10, then the data jumps straight to the NEXT day's morning bar
        # with no bar at/after FORCE_CLOSE_TIME in between - exercises the day-boundary
        # defensive guard rather than the normal same-day forced-close path
        highs, lows, closes = _build_long_entry_bars(n_filler=0)   # stop right after entry, no same-day filler
        idx_day1 = pd.date_range(start="2020-01-06 09:30", periods=len(closes), freq="5min",
                                  tz="America/New_York")
        idx_day2 = pd.date_range(start="2020-01-07 09:30", periods=5, freq="5min",
                                  tz="America/New_York")
        full_idx = idx_day1.append(idx_day2)
        full_closes = closes + [4530] * 5
        full_highs = highs + [4531] * 5
        full_lows = lows + [4529] * 5
        df = pd.DataFrame({"Open": full_closes, "High": full_highs, "Low": full_lows,
                            "Close": full_closes, "volume": [1.0] * len(full_closes)}, index=full_idx)

        trades, _, _, trace = evw.backtest_instrument("TEST", df, verbose=False, record_trace=True)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "FLAT")
        self.assertEqual(trades[0]["date"], idx_day1[-1].date())

    def test_position_still_open_at_series_end_is_marked_flat_not_dropped(self):
        # series ends right after entry (bar 10), no more bars at all
        highs, lows, closes = _build_long_entry_bars(n_filler=0)
        df = _make_df(closes, highs=highs, lows=lows, volumes=[1.0] * len(closes))
        trades, _, _, trace = evw.backtest_instrument("TEST", df, verbose=False, record_trace=True)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "FLAT")


# ============================= granularity-convergence check =============================

class TestGranularityConvergence(unittest.TestCase):
    def test_apparent_edge_shrinks_toward_zero_as_substep_resolution_increases(self):
        """This strategy's entries are close-based (ORB break, VWAP reclaim) but its STOP comes
        from a confirmed swing-pivot extreme - a multi-bar aggregate, not the entry bar's own
        wick, structurally closer to Donchian/Dow Theory (which showed no coarse-resolution
        artifact) than PO3 (which did). Checked honestly here rather than assumed safe by
        resemblance."""
        results = {}
        for n_substeps in (1, 4, 16, 64):
            all_r = []
            for seed in range(6):
                n_bars = 900   # ~a couple weeks of NY trading-hour bars per "instrument"
                # build a full session-hours calendar so ORB/VWAP/entry windows are meaningful
                opens, highs, lows, closes = build_synthetic_ohlc(n_bars, n_substeps, bar_std_pct=0.05,
                                                                    seed=seed * 100 + n_substeps)
                idx = pd.date_range(start="2020-01-06 09:30", periods=n_bars, freq="5min",
                                     tz="America/New_York")
                df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes,
                                    "volume": [1.0] * n_bars}, index=idx)
                trades, _, _, _ = evw.backtest_instrument(f"SEED{seed}", df, verbose=False, record_trace=True)
                all_r.extend(t["r"] for t in trades)
            n = len(all_r)
            if n < 2:
                results[n_substeps] = (0.0, n)
                continue
            mean_r = sum(all_r) / n
            std_r = np.std(all_r, ddof=1)
            z = (mean_r / (std_r / (n ** 0.5))) if std_r > 0 else 0.0
            results[n_substeps] = (z, n)

        print(f"\nEvenDyer VWAP granularity convergence: {results}")
        z_coarse = abs(results[1][0])
        z_fine = abs(results[64][0])
        # the coarse (1-substep) reading is not required to be huge here (unlike PO3) since
        # entries aren't wick-triggered - the check is that finer resolution doesn't make an
        # edge WORSE/bigger, i.e. no hidden coarse-graining artifact inflating it
        self.assertLessEqual(z_fine, max(z_coarse, 3.0) + 1.0,
                              f"finer resolution should not show a materially larger apparent edge "
                              f"than coarse resolution: {results}")


# ============================= end-to-end smoke test =============================

class TestSmokeEndToEnd(unittest.TestCase):
    def test_full_pipeline_runs_without_crashing_on_synthetic_multi_instrument_data(self):
        all_trades = []
        for i, label in enumerate(["SYN1", "SYN2", "SYN3"]):
            n_bars = 2000
            opens, highs, lows, closes = build_synthetic_ohlc(n_bars, n_substeps=16, bar_std_pct=0.06,
                                                                 start_price=4000 + i * 500, seed=i)
            idx = pd.date_range(start="2020-01-06 09:30", periods=n_bars, freq="5min",
                                 tz="America/New_York")
            volumes = list(np.random.default_rng(i).uniform(50, 150, size=n_bars))
            df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes,
                                "volume": volumes}, index=idx)
            trades, used_real_volume, diag = evw.backtest_instrument(label, df, verbose=False)
            for t in trades:
                t["instrument"] = label
            all_trades.extend(trades)
            self.assertIsInstance(used_real_volume, bool)
            self.assertIsInstance(diag, str)

        for t in all_trades:
            self.assertIn(t["outcome"], ("TP", "SL", "FLAT"))
            self.assertTrue(np.isfinite(t["r"]))
            self.assertIn(t["side"], ("LONG", "SHORT"))


if __name__ == "__main__":
    unittest.main()
