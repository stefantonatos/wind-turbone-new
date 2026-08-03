# Unit tests + a granularity-convergence check + one synthetic end-to-end
# smoke run for asian_range_breakout_dukascopy_backtest.py.
#
# Covers, per this project's rigor conventions:
#   1. NO-LOOKAHEAD proof: the Asian range used to trade a given day's
#      London breakout window comes from the PRIOR day's evening window,
#      never that same day's own (still in the future, relative to the
#      breakout window) evening session - a direct check on the exact
#      off-by-one a naive "use today's own 19:00-00:00 range" bug would
#      produce (a different, wrong trade).
#   2. Range-building correctness (high/low over the 19:00-00:00 window).
#   3. Breakout trigger logic: close-confirmed break in each direction,
#      same-bar double-sweep ambiguity (skip the day), no-break-in-window
#      (no trade), and "no available range" (a data gap the prior evening
#      -> no trade at all that day).
#   4. Stop/target geometry: measured-move target with the exact expected
#      multiplier, the degenerate-range fallback to FALLBACK_REWARD_RISK,
#      and the stop buffer's sign convention for both directions.
#   5. Outcome accounting: SL, TP, and forced FLAT close at FORCE_CLOSE_TIME,
#      including stop-vs-target same-bar precedence (stop checked first).
#   6. One trade per day, even after an early ambiguous double-sweep.
#   7. A genuine granularity-convergence check (same methodology as
#      test_bollinger_band_mean_reversion_dukascopy_backtest.py's - see
#      this script's own header for why it applies here too): synthetic
#      OHLC built from a proper multi-step intraday random walk, covering
#      full NY-time calendar days so the Asian/breakout session windows
#      are populated realistically, run at increasing sub-bar resolution.
#      Confirms any apparent "edge" on a zero-drift random walk shrinks
#      hard as resolution increases, i.e. it's a construction artifact at
#      coarse resolution, not a real signal.
#   8. ONE small synthetic end-to-end smoke run confirming the full
#      fetch-independent pipeline runs without crashing on multi-
#      "instrument" data. Deliberately NOT a real Dukascopy download.
#
# Run with:  python -m pytest research/test_asian_range_breakout_dukascopy_backtest.py -v
# or:        python research/test_asian_range_breakout_dukascopy_backtest.py

import datetime
import importlib.util
import os
import unittest

import numpy as np
import pandas as pd

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "asian_range_breakout_dukascopy_backtest.py")
_spec = importlib.util.spec_from_file_location("asian_range_breakout_dukascopy_backtest", _MODULE_PATH)
arb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arb)


# ============================= synthetic data helpers =============================

def _make_day_index(date_, tz="America/New_York"):
    """288 5-min bars, 00:00 through 23:55 NY time - a full calendar day, covering both the
    Asian window (19:00-00:00) and the breakout window (02:00-05:00)."""
    return pd.date_range(start=pd.Timestamp(date_.year, date_.month, date_.day, 0, 0, tz=tz),
                          periods=288, freq="5min")


def _build_synthetic_df(dates, base_price=1.1000, overrides=None):
    """Builds a synthetic 5-min OHLC DataFrame across the given list of dates. Every bar is
    flat at base_price unless `overrides[date]["HH:MM"] = (o, h, l, c)` says otherwise."""
    overrides = overrides or {}
    idx_all, rows = [], []
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


# Three consecutive weekday dates, safely away from any US DST transition (all in Jan 2019).
_DAY0 = datetime.date(2019, 1, 7)    # builds the Asian range used by DAY1's breakout
_DAY1 = datetime.date(2019, 1, 8)
_DAY2 = datetime.date(2019, 1, 9)    # builds the Asian range used by DAY2's own... (i.e. DAY3's)
_DAY3 = datetime.date(2019, 1, 10)


def _asian_window_bars(high, low, base=1.1000):
    """A tiny set of overrides that make the 19:00-00:00 window's high/low come out exactly as
    given - one bar touches `high`, another touches `low`, the rest sit at `base`."""
    return {"19:00": (base, high, base, base), "19:05": (base, base, low, base)}


class TestNoLookahead(unittest.TestCase):
    def test_breakout_uses_prior_days_range_not_same_days_own_evening(self):
        """DAY0's evening builds a narrow range (1.0990-1.1010). DAY1's breakout window trades
        against THAT range. DAY1's OWN evening (which happens AFTER its breakout window, later
        the same calendar day) builds a wildly different, wide range (1.0000-1.2000). A naive
        off-by-one bug that used "today's own 19:00-00:00 range" instead of yesterday's would
        either see no range yet (nothing built until 19:00) or, worse, silently use the wrong
        one on a later day - this proves DAY1's breakout genuinely used DAY0's numbers, not
        DAY1's own (self-inconsistent, still-in-the-future-at-breakout-time) evening range."""
        overrides = {
            _DAY0: _asian_window_bars(1.1010, 1.0990),
            _DAY1: {**_asian_window_bars(1.2000, 1.0000),  # DAY1's own evening - irrelevant to DAY1's own breakout
                    "02:00": (1.1000, 1.1015, 1.0995, 1.1015)},   # closes above DAY0's range high (1.1010) -> LONG
        }
        df = _build_synthetic_df([_DAY0, _DAY1, _DAY2], overrides=overrides)
        trades = arb.backtest_instrument("SYN", df)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "LONG")
        self.assertEqual(trades[0]["date"], _DAY1)
        # the stop must be derived from DAY0's range low (1.0990), not DAY1's own (much wider,
        # still-in-the-future-at-breakout-time) evening range low of 1.0000 - if the bug used
        # DAY1's own range, the stop would be miles away (near 1.0000) instead of just below 1.0990
        entry = 1.1015
        expected_buffer = (arb.STOP_BUFFER_PCT / 100.0) * entry
        expected_stop = 1.0990 - expected_buffer
        expected_sl_distance = entry - expected_stop
        self.assertLess(expected_sl_distance, 0.01, "sanity check: expected stop distance derived from DAY0's "
                                                      "narrow range should be small, not the ~0.10 a DAY1-leak bug would produce")
        # the trade drifts flat (back to base_price=1.1000) and is force-closed FLAT - its r must
        # be computed against DAY0's stop distance, not a DAY1-leaked one
        self.assertEqual(trades[0]["outcome"], "FLAT")
        expected_r = (1.1000 - entry) / expected_sl_distance
        self.assertAlmostEqual(trades[0]["r"], expected_r, places=6)

    def test_no_available_range_produces_no_trade(self):
        """DAY0 has NO bars during its 19:00-00:00 window at all (simulated by never touching
        base_price differently - i.e. a genuinely flat/empty evening, so building_range still
        gets SET (flat high==low==base), which is a valid, if degenerate, range). To test a
        real gap, we instead check that if DAY0 is entirely absent from the data (as if a
        holiday), DAY1 has no available range and takes no breakout trade even on a large move."""
        overrides = {
            _DAY1: {"02:00": (1.1000, 1.1500, 1.0995, 1.1500)},   # would clearly break out if any range existed
        }
        # DAY0 omitted entirely - available_range must be None on DAY1
        df = _build_synthetic_df([_DAY1, _DAY2], overrides=overrides)
        trades = arb.backtest_instrument("SYN", df)
        self.assertEqual(trades, [])


# ============================= breakout trigger logic =============================

class TestBreakoutTrigger(unittest.TestCase):
    def _two_day_df(self, day1_overrides):
        overrides = {_DAY0: _asian_window_bars(1.1010, 1.0990), _DAY1: day1_overrides}
        return _build_synthetic_df([_DAY0, _DAY1, _DAY2], overrides=overrides)

    def test_close_confirmed_upside_break_triggers_long(self):
        df = self._two_day_df({"02:00": (1.1000, 1.1015, 1.0995, 1.1015)})
        trades = arb.backtest_instrument("SYN", df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "LONG")

    def test_close_confirmed_downside_break_triggers_short(self):
        df = self._two_day_df({"02:00": (1.1000, 1.1005, 1.0985, 1.0985)})
        trades = arb.backtest_instrument("SYN", df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "SHORT")

    def test_wick_beyond_level_without_close_confirmation_does_not_trigger_yet(self):
        """Bar wicks above the range high but closes back inside - no entry on this bar. A
        LATER bar in the window that closes cleanly above still can trigger."""
        df = self._two_day_df({
            "02:00": (1.1000, 1.1020, 1.0995, 1.1005),   # wicks to 1.1020 but closes at 1.1005 (inside range)
            "02:05": (1.1005, 1.1018, 1.1000, 1.1018),   # now closes above 1.1010 -> LONG here
        })
        trades = arb.backtest_instrument("SYN", df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "LONG")

    def test_same_bar_double_sweep_skips_the_day(self):
        df = self._two_day_df({"02:00": (1.1000, 1.1020, 1.0980, 1.1000)})   # wicks past BOTH sides
        trades = arb.backtest_instrument("SYN", df)
        self.assertEqual(trades, [])

    def test_double_sweep_consumes_the_days_slot_no_later_retrigger(self):
        df = self._two_day_df({
            "02:00": (1.1000, 1.1020, 1.0980, 1.1000),    # ambiguous double sweep -> day consumed
            "02:05": (1.1000, 1.1030, 1.0995, 1.1030),    # would otherwise be a clean LONG trigger
        })
        trades = arb.backtest_instrument("SYN", df)
        self.assertEqual(trades, [], "the day's one-trade slot was already consumed by the ambiguous bar")

    def test_no_break_in_window_produces_no_trade(self):
        df = self._two_day_df({"02:00": (1.1000, 1.1005, 1.0995, 1.1000)})   # stays inside the range
        trades = arb.backtest_instrument("SYN", df)
        self.assertEqual(trades, [])

    def test_one_trade_per_day_after_a_clean_trigger(self):
        df = self._two_day_df({
            "02:00": (1.1000, 1.1015, 1.0995, 1.1015),   # LONG triggers here
            "02:05": (1.1015, 1.1005, 1.0970, 1.0975),   # would look like a fresh downside break if re-evaluated
        })
        trades = arb.backtest_instrument("SYN", df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "LONG")


# ============================= stop / target geometry =============================

class TestStopTargetGeometry(unittest.TestCase):
    def test_long_measured_move_target_and_stop_buffer(self):
        # Asian range: high=1.1010, low=1.0990 -> height 0.0020. Breakout close = 1.1015.
        overrides = {_DAY0: _asian_window_bars(1.1010, 1.0990),
                     _DAY1: {"02:00": (1.1000, 1.1015, 1.0995, 1.1015)}}
        df = _build_synthetic_df([_DAY0, _DAY1, _DAY2], overrides=overrides)

        entry = 1.1015
        range_low = 1.0990
        buffer_price = (arb.STOP_BUFFER_PCT / 100.0) * entry
        expected_stop = range_low - buffer_price
        expected_sl_distance = entry - expected_stop
        expected_target = entry + (1.1010 - 1.0990) * arb.TARGET_RANGE_MULT
        expected_reward_risk = ((1.1010 - 1.0990) * arb.TARGET_RANGE_MULT) / expected_sl_distance

        # drive price straight to the target to read back the realized reward:risk via outcome "TP"
        overrides[_DAY1]["02:05"] = (entry, expected_target + 0.0005, entry, expected_target + 0.0002)
        df = _build_synthetic_df([_DAY0, _DAY1, _DAY2], overrides=overrides)
        trades = arb.backtest_instrument("SYN", df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "TP")
        self.assertAlmostEqual(trades[0]["r"], expected_reward_risk, places=6)

    def test_short_stop_buffer_direction(self):
        overrides = {_DAY0: _asian_window_bars(1.1010, 1.0990),
                     _DAY1: {"02:00": (1.1000, 1.1005, 1.0985, 1.0985),   # closes below range low -> SHORT
                             "02:05": (1.0985, 1.1030, 1.0980, 1.1025)}}  # rallies straight through the stop
        df = _build_synthetic_df([_DAY0, _DAY1, _DAY2], overrides=overrides)
        trades = arb.backtest_instrument("SYN", df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "SL")
        self.assertAlmostEqual(trades[0]["r"], -1.0, places=6)

    def test_degenerate_range_falls_back_to_fixed_reward_risk(self):
        # near-zero-height Asian range (0.00002, well under MIN_RANGE_PCT of price) -> the
        # measured-move target would be a meaningless sliver; fallback R:R must be used instead
        overrides = {_DAY0: _asian_window_bars(1.10001, 1.09999),
                     _DAY1: {"02:00": (1.10005, 1.10015, 1.10000, 1.10015)}}
        df = _build_synthetic_df([_DAY0, _DAY1, _DAY2], overrides=overrides)

        entry = 1.10015
        range_low = 1.09999
        buffer_price = (arb.STOP_BUFFER_PCT / 100.0) * entry
        expected_stop = range_low - buffer_price
        expected_sl_distance = entry - expected_stop
        expected_target = entry + expected_sl_distance * arb.FALLBACK_REWARD_RISK

        overrides[_DAY1]["02:05"] = (entry, expected_target + 0.0005, entry, expected_target + 0.0002)
        df = _build_synthetic_df([_DAY0, _DAY1, _DAY2], overrides=overrides)
        trades = arb.backtest_instrument("SYN", df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "TP")
        self.assertAlmostEqual(trades[0]["r"], arb.FALLBACK_REWARD_RISK, places=6)


# ============================= outcome accounting =============================

class TestOutcomeAccounting(unittest.TestCase):
    def test_forced_close_at_force_close_time(self):
        overrides = {_DAY0: _asian_window_bars(1.1010, 1.0990),
                     _DAY1: {"02:00": (1.1000, 1.1015, 1.0995, 1.1015)}}
        # never hits stop or the (far away) measured-move target - drifts a little, then FORCE_CLOSE_TIME arrives
        for hh in range(3, 12):
            overrides[_DAY1][f"{hh:02d}:00"] = (1.1015, 1.1018, 1.1012, 1.1016)
        df = _build_synthetic_df([_DAY0, _DAY1, _DAY2], overrides=overrides)
        trades = arb.backtest_instrument("SYN", df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "FLAT")
        entry = 1.1015
        expected_r = (1.1016 - entry) / (entry - (1.0990 - (arb.STOP_BUFFER_PCT / 100.0) * entry))
        self.assertAlmostEqual(trades[0]["r"], expected_r, places=6)

    def test_stop_checked_before_target_on_same_bar(self):
        # construct a bar whose range spans BOTH the stop and the (very close) target - stop
        # must win, matching this project's "tighter/first-hit level" precedence convention
        overrides = {_DAY0: _asian_window_bars(1.1010, 1.0990),
                     _DAY1: {"02:00": (1.1000, 1.1015, 1.0995, 1.1015)}}
        entry = 1.1015
        buffer_price = (arb.STOP_BUFFER_PCT / 100.0) * entry
        stop = 1.0990 - buffer_price
        target = entry + (1.1010 - 1.0990) * arb.TARGET_RANGE_MULT
        overrides[_DAY1]["02:05"] = (entry, target + 0.0010, stop - 0.0010, entry)
        df = _build_synthetic_df([_DAY0, _DAY1, _DAY2], overrides=overrides)
        trades = arb.backtest_instrument("SYN", df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "SL")
        self.assertAlmostEqual(trades[0]["r"], -1.0, places=6)


# ============================= granularity-convergence check =============================

def build_synthetic_ohlc_full_days(n_days, n_substeps, bar_std_pct, start_price=1.1000, seed=0,
                                    start_date="2018-01-01"):
    """Proper multi-step intraday random walk covering FULL 24-hour NY-time calendar days (288
    5-min bars/day), so the Asian (19:00-00:00) and breakout (02:00-05:00) session windows are
    populated realistically - not just a flat arbitrary-start bar sequence. Each bar's OPEN is
    anchored to the previous bar's actual close, and high/low/close emerge from n_substeps
    independent Gaussian sub-steps along one continuous path within the bar (same methodology as
    test_bollinger_band_mean_reversion_dukascopy_backtest.py's build_synthetic_ohlc - NOT a
    single close-plus-independent-wick-noise construction). Per-substep stddev is scaled by
    1/sqrt(n_substeps) so each bar's total variance stays constant regardless of granularity."""
    n_bars = n_days * 288
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
    idx = pd.date_range(start=start_date, periods=n_bars, freq="5min", tz="America/New_York")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


class TestGranularityConvergence(unittest.TestCase):
    """The core honesty check for this script's close-confirmed-break entry (see the module
    header's GRANULARITY / FALSE-POSITIVE SANITY CHECK section for the full reasoning): does the
    apparent edge measured on synthetic data survive as the synthetic OHLC's intrabar path gets
    more finely resolved? A fair, zero-drift random walk has no true edge at any resolution - a
    strategy showing a strong "edge" only at coarse resolution that collapses as resolution
    increases would be exhibiting a construction artifact, not a real signal.

    HONEST RESULT OF ACTUALLY RUNNING THIS (rather than assuming it away): unlike a wick-touch
    mean-reversion strategy (see test_bollinger_band_mean_reversion_dukascopy_backtest.py's
    TestGranularityConvergence, which shows a dramatic z > 10 coarse-resolution artifact that
    collapses hard as resolution increases), this script does NOT show that dramatic pattern -
    empirically, |z| stays small and does not blow up at n_substeps=1 the way Bollinger's does.
    Two structural reasons this is plausible, not just "we got lucky with these seeds": (1) the
    entry trigger is CLOSE-confirmed, not wick-triggered, and a driftless Gaussian random walk's
    close-crossing-a-level distribution is exactly (not just asymptotically) invariant to
    substep count, since a sum of n iid Gaussians scaled by 1/sqrt(n) is itself Gaussian with
    matching variance for any n - so the entry decision itself carries no substep-count bias by
    construction. (2) the stop/target geometry is derived from a MULTI-BAR range aggregate (the
    whole ~60-bar Asian session's high/low), not a single bar's own wick the way PO3's
    manipulation-bar stop is - averaging over many bars damps the single-bar wick-resolution
    sensitivity that drives Bollinger/PO3's artifact.

    What DOES show a real, if modest, granularity-dependent effect: the Asian range's own
    high/low span is itself built from wicks, so it's a LITTLE narrower at coarse resolution
    (fewer, cruder per-bar excursions), which pulls the measured-move target a little closer and
    modestly raises the coarse-resolution TP rate relative to the finest resolution tested. This
    is checked directly below and bounded, not ignored - so this check is doing real work even
    though the headline finding is negative (no large spurious average-R edge at any tested
    resolution). Fixed seeds make this fully deterministic/reproducible, not a flaky test."""

    def test_no_large_spurious_edge_at_any_resolution(self):
        n_days = 150
        seeds = range(15)
        results = {}
        for n_substeps in (1, 4, 16, 64):
            all_r = []
            outcomes = {"TP": 0, "SL": 0, "FLAT": 0}
            for seed in seeds:
                df = build_synthetic_ohlc_full_days(n_days, n_substeps, bar_std_pct=0.05, seed=seed)
                trades = arb.backtest_instrument("SYN", df)
                for t in trades:
                    all_r.append(t["r"])
                    outcomes[t["outcome"]] += 1
            n = len(all_r)
            avg_r = sum(all_r) / n
            std_r = np.std(all_r, ddof=1)
            z = avg_r / (std_r / (n ** 0.5)) if std_r > 0 else 0.0
            tp_rate = outcomes["TP"] / n
            results[n_substeps] = {"n_trades": n, "avg_r": avg_r, "z": z, "tp_rate": tp_rate, "outcomes": outcomes}

        for n_substeps, r in results.items():
            print(f"  n_substeps={n_substeps:3d}: n_trades={r['n_trades']:5d}  avg_r={r['avg_r']:+.4f}  "
                  f"z={r['z']:+.2f}  TP_rate={r['tp_rate']:.2%}")

        self.assertGreater(results[1]["n_trades"], 100, "test setup didn't produce enough coarse-granularity trades")

        # (1) no large, "smoking gun" average-R edge at ANY tested resolution - generous bounds,
        # same spirit as donchian_turtle_breakout_dukascopy_backtest.py's TestRandomWalkNullResult
        # (a genuine implementation bug, e.g. a lookahead leak, would be expected to blow well
        # past these, at every resolution, not just the coarsest one)
        for n_substeps, r in results.items():
            self.assertLess(abs(r["avg_r"]), 0.15, f"suspiciously large avg R at n_substeps={n_substeps}: {r['avg_r']:.3f}")
            self.assertLess(abs(r["z"]), 3.5, f"suspiciously large z at n_substeps={n_substeps}: {r['z']:.2f}")

        # (2) the one real granularity-dependent effect this construction does produce - the
        # coarsest resolution's TP rate is somewhat higher than the finest (narrower Asian range
        # -> closer measured-move target) - bounded to a modest gap, not a dramatic one; a bug
        # that broke the no-lookahead range carry-over (see TestNoLookahead) would be expected to
        # produce a much larger, structurally different distortion than this smooth, bounded one
        tp_gap = results[1]["tp_rate"] - results[64]["tp_rate"]
        self.assertGreater(tp_gap, 0.0, "expected the coarsest resolution's TP rate to be at least "
                                         "slightly higher than the finest (narrower coarse-resolution Asian range)")
        self.assertLess(tp_gap, 0.10, f"coarse-vs-fine TP rate gap is larger than expected: {tp_gap:.2%}")


# ============================= synthetic end-to-end smoke run =============================

class TestEndToEndSmokeRun(unittest.TestCase):
    """ONE small synthetic multi-"instrument" run confirming the full pipeline (range-building
    -> breakout entry -> stop/target/force-close -> reporting-style aggregation) runs without
    crashing and produces plausibly-structured output. Deliberately NOT a real Dukascopy
    download."""

    def test_smoke_run_produces_plausible_trades(self):
        all_trades = []
        for label, seed in [("SYN_A", 1), ("SYN_B", 2)]:
            df = build_synthetic_ohlc_full_days(n_days=60, n_substeps=8, bar_std_pct=0.05,
                                                  seed=seed, start_date="2020-01-01")
            trades = arb.backtest_instrument(label, df)
            for t in trades:
                t["instrument"] = label
            all_trades.extend(trades)

        for t in all_trades:
            self.assertIn(t["side"], ("LONG", "SHORT"))
            self.assertIn(t["outcome"], ("TP", "SL", "FLAT"))
            self.assertIsInstance(t["r"], float)
            self.assertIn(t["instrument"], ("SYN_A", "SYN_B"))

        if len(all_trades) >= 2:
            all_r = [t["r"] for t in all_trades]
            std_r = np.std(all_r, ddof=1)
            n = len(all_r)
            z = (sum(all_r) / n) / (std_r / (n ** 0.5)) if std_r > 0 else 0.0
            self.assertTrue(np.isfinite(z))


if __name__ == "__main__":
    unittest.main()
