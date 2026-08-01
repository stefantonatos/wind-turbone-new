# Unit tests + one synthetic end-to-end smoke run for
# dow_theory_swing_structure_dukascopy_backtest.py.
#
# Covers, per this project's rigor conventions:
#   1. Confirmed-pivot detection correctness on find_confirmed_swing_pivots
#      directly (hand-computed small arrays) - same pattern as support_
#      resistance_zone_bounce_dukascopy_backtest.py's own pivot tests.
#   2. NO-LOOKAHEAD proofs, at TWO separate levels: (a) a pivot is never
#      usable before index (formation_index + SWING_LEN) - a synthetic
#      spike that a naive off-by-one bug would confirm a day early is
#      checked directly; (b) a fresh trend-structure transition confirmed
#      on day d is never ACTED ON before day d+1's first bar (the shift-
#      by-one-day convention) - a synthetic case where a naive "act same
#      day" bug would enter a day early, at a different/unavailable price,
#      is checked directly.
#   3. Trend-state transitions: HH+HL -> UP, LH+LL -> DOWN, mixed
#      structure (HH+LL or LH+HL) -> neither.
#   4. "Freshly confirmed" entry semantics: a transition fires an entry
#      signal exactly once (on the day the state FLIPS), not again on a
#      later day where the trend state merely continues to hold.
#   5. Entry mechanics: fires on the ACTION day's FIRST 5-min bar only
#      (not a later bar the same day), correct initial stop/sl_distance,
#      degenerate-stop guard (stop on the wrong side of entry -> skip).
#   6. Trailing-stop ratchet: a later, more favorable confirmed pivot
#      tightens the stop; the tightened level only becomes usable starting
#      the day after ITS OWN confirmation (not immediately); a stop touch
#      correctly closes the trade using the RATCHETED level, not the
#      original one.
#   7. One trade at a time (a fresh signal while already in a position is
#      ignored) and force-close accounting at the end of the data.
#   8. ONE small synthetic end-to-end smoke run (a genuine random walk,
#      not hand-crafted) confirming the full pipeline runs without
#      crashing and doesn't show a suspiciously large edge on pure noise -
#      same spirit as donchian_turtle_breakout_dukascopy_backtest.py's
#      TestRandomWalkNullResult. Deliberately NOT a real Dukascopy
#      download.
#
# Run with:  python -m pytest research/test_dow_theory_swing_structure_dukascopy_backtest.py -v
# or:        python research/test_dow_theory_swing_structure_dukascopy_backtest.py

import importlib.util
import os
import unittest

import numpy as np
import pandas as pd

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "dow_theory_swing_structure_dukascopy_backtest.py")
_spec = importlib.util.spec_from_file_location("dow_theory_swing_structure_dukascopy_backtest", _MODULE_PATH)
dow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dow)

SWING_LEN = dow.SWING_LEN   # 20 - use the real production constant throughout, not a test-only override


# ============================= synthetic data helpers =============================

def _make_daily_df(n, overrides, base_high=1.1002, base_low=1.0998, start="2016-01-04"):
    """A small daily OHLC DataFrame for testing compute_daily_swing_signals / find_confirmed_
    swing_pivots directly, bypassing the 5-min -> daily resample plumbing entirely. `overrides`
    is {index: (high, low)}; every other day sits flat at (base_high, base_low), which - by
    construction (repeated, tied values never pass the strict-uniqueness pivot test) - never
    itself produces a spurious pivot."""
    idx = pd.bdate_range(start=start, periods=n)
    highs = [base_high] * n
    lows = [base_low] * n
    for i, (h, l) in overrides.items():
        highs[i] = h
        lows[i] = l
    opens = list(lows)
    closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


FLAT_DAY = (1.1000, 1.1002, 1.0998, 1.1000)


def _flat_days(n, start="2016-01-04"):
    days = pd.bdate_range(start=start, periods=n)
    return [(d, [FLAT_DAY]) for d in days]


def _override_day(days_list, index, bars):
    new_list = list(days_list)
    d, _ = new_list[index]
    new_list[index] = (d, bars)
    return new_list


def _build_5min_df(day_specs):
    """day_specs: list of (date, [(o,h,l,c), ...]) - one or more 5-min bars per day, 5 minutes
    apart starting 09:30 NY time. Same helper as donchian_turtle_breakout_dukascopy_backtest.py's
    test file - the day's aggregate daily OHLC (via resample_daily) is exactly what a plain
    daily-bar feed for that day would produce."""
    rows, idx = [], []
    for date_, bars in day_specs:
        for k, (o, h, l, c) in enumerate(bars):
            ts = pd.Timestamp(date_.year, date_.month, date_.day, 9, 30, tz="America/New_York") + \
                 pd.Timedelta(minutes=5 * k)
            idx.append(ts)
            rows.append((o, h, l, c))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=pd.DatetimeIndex(idx))


def _make_random_walk_5min_df(n_days, bars_per_day, seed, start_price=1.1000, intrabar_vol=0.0006):
    """A genuine no-drift random walk with intrabar noise - same construction as donchian_turtle_
    breakout_dukascopy_backtest.py's test file, used only for the null-result sanity check below."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n_days)
    rows, idx = [], []
    price = start_price
    for d in dates:
        o_prev = price
        for k in range(bars_per_day):
            c = o_prev + rng.normal(0, intrabar_vol)
            o = o_prev
            hi = max(o, c) + abs(rng.normal(0, intrabar_vol / 2))
            lo = min(o, c) - abs(rng.normal(0, intrabar_vol / 2))
            ts = pd.Timestamp(d.year, d.month, d.day, 9, 30, tz="America/New_York") + pd.Timedelta(minutes=5 * k)
            idx.append(ts)
            rows.append((o, hi, lo, c))
            o_prev = c
        price = o_prev
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=pd.DatetimeIndex(idx))


# A reusable 4(+1)-pivot uptrend fixture, spaced far enough apart (with SWING_LEN=20 lookback on
# each side) that no pivot's confirmation window overlaps another's in a way that would create
# spurious/duplicate pivots - see the inline comments below for the exact non-overlap reasoning.
PIVOT_LOW1_IDX, LOW1 = 25, 1.0900
PIVOT_HIGH1_IDX, HIGH1 = 30, 1.1100
PIVOT_LOW2_IDX, LOW2 = 60, 1.0950     # a HIGHER low than LOW1
PIVOT_HIGH2_IDX, HIGH2 = 65, 1.1150   # a HIGHER high than HIGH1
PIVOT_LOW3_IDX, LOW3 = 110, 1.0980    # a further HIGHER low - continues the uptrend, no re-trigger expected

CONFIRM_LOW1_IDX = PIVOT_LOW1_IDX + SWING_LEN     # 45
CONFIRM_HIGH1_IDX = PIVOT_HIGH1_IDX + SWING_LEN   # 50
CONFIRM_LOW2_IDX = PIVOT_LOW2_IDX + SWING_LEN     # 80
CONFIRM_HIGH2_IDX = PIVOT_HIGH2_IDX + SWING_LEN   # 85  <- fresh_up fires here
CONFIRM_LOW3_IDX = PIVOT_LOW3_IDX + SWING_LEN     # 130 <- trend continues, stop ratchets, no re-trigger

ACTION_DAY_IDX = CONFIRM_HIGH2_IDX + 1    # 86 - the day the LONG is actually taken (shift-by-one)
RATCHET_ACTIVE_IDX = CONFIRM_LOW3_IDX + 1  # 131 - the day the ratcheted (LOW3) stop becomes usable


def _build_uptrend_days(n_total=145, start="2016-01-04"):
    days = _flat_days(n_total, start=start)
    days = _override_day(days, PIVOT_LOW1_IDX, [(1.1000, 1.1002, LOW1, 1.0950)])
    days = _override_day(days, PIVOT_HIGH1_IDX, [(1.1000, HIGH1, 1.0998, 1.1050)])
    days = _override_day(days, PIVOT_LOW2_IDX, [(1.1000, 1.1002, LOW2, 1.0980)])
    days = _override_day(days, PIVOT_HIGH2_IDX, [(1.1000, HIGH2, 1.0998, 1.1100)])
    if n_total > PIVOT_LOW3_IDX:
        days = _override_day(days, PIVOT_LOW3_IDX, [(1.1000, 1.1002, LOW3, 1.1000)])
    return days


def _uptrend_overrides_for_daily_df():
    return {PIVOT_LOW1_IDX: (1.1002, LOW1), PIVOT_HIGH1_IDX: (HIGH1, 1.0998),
            PIVOT_LOW2_IDX: (1.1002, LOW2), PIVOT_HIGH2_IDX: (HIGH2, 1.0998)}


# ============================= confirmed-pivot detection =============================

class TestSwingPivotDetection(unittest.TestCase):
    def test_unique_extreme_confirmed_at_formation_plus_lookback(self):
        highs = [1.10, 1.10, 1.10, 1.20, 1.10, 1.10, 1.10]
        lows = [1.00] * 7
        swing_highs, swing_lows = dow.find_confirmed_swing_pivots(highs, lows, lookback=3)
        # formation index 3, confirmed at index 3+3=6
        self.assertIn(6, swing_highs)
        self.assertEqual(swing_highs[6], (3, 1.20))
        self.assertEqual(swing_lows, {})

    def test_tied_extreme_is_not_a_pivot(self):
        highs = [1.10, 1.20, 1.10, 1.20, 1.10]   # two bars tie for the max within the window
        lows = [1.00] * 5
        swing_highs, _ = dow.find_confirmed_swing_pivots(highs, lows, lookback=2)
        self.assertEqual(swing_highs, {})

    def test_swing_low_mirrors_swing_high(self):
        highs = [1.10] * 7
        lows = [1.00, 1.00, 1.00, 0.90, 1.00, 1.00, 1.00]
        _, swing_lows = dow.find_confirmed_swing_pivots(highs, lows, lookback=3)
        self.assertIn(6, swing_lows)
        self.assertEqual(swing_lows[6], (3, 0.90))


# ============================= no-lookahead: pivot confirmation timing =============================

class TestNoLookaheadPivotConfirmation(unittest.TestCase):
    def test_pivot_not_usable_before_formation_plus_lookback(self):
        """A spike at index 10 must not appear in the confirmed-pivot dict at any index earlier
        than 10 + lookback - a direct check on the exact off-by-one a naive implementation could
        get wrong (e.g. confirming at `i` instead of `i + lookback`)."""
        n = 25
        highs = [1.1002] * n
        highs[10] = 5.0
        lows = [1.0998] * n
        swing_highs, _ = dow.find_confirmed_swing_pivots(highs, lows, lookback=5)
        self.assertNotIn(10, swing_highs)
        self.assertNotIn(14, swing_highs)   # one bar short of 10+5
        self.assertIn(15, swing_highs)
        self.assertEqual(swing_highs[15], (10, 5.0))

    def test_trend_state_and_stop_do_not_reflect_a_pivot_before_its_confirm_index(self):
        overrides = _uptrend_overrides_for_daily_df()
        daily = _make_daily_df(100, overrides)
        sig = dow.compute_daily_swing_signals(daily)

        # at CONFIRM_HIGH2_IDX - 1 (one day before the fresh-up confirmation), fewer than 2
        # confirmed highs exist yet - no trend state at all (stored as either None or NaN
        # depending on pandas' object-column inference - either way, "no state")
        self.assertTrue(pd.isna(sig["trend_state"].iloc[CONFIRM_HIGH2_IDX - 1]))
        # AT the confirm index itself, the (as-of, same-day) state is already UP...
        self.assertEqual(sig["trend_state"].iloc[CONFIRM_HIGH2_IDX], "UP")
        self.assertTrue(bool(sig["fresh_up"].iloc[CONFIRM_HIGH2_IDX]))
        # ...but the ACTIONABLE (shifted) flag only turns True the FOLLOWING day - not the same
        # day the confirmation itself becomes known
        self.assertFalse(bool(sig["fresh_up_act"].iloc[CONFIRM_HIGH2_IDX]))
        self.assertTrue(bool(sig["fresh_up_act"].iloc[CONFIRM_HIGH2_IDX + 1]))


# ============================= no-lookahead: shifted entry/trail timing (full pipeline) =============================

class TestNoLookaheadActionTiming(unittest.TestCase):
    def test_entry_does_not_fire_on_the_confirmation_day_itself(self):
        """A naive 'act the same day the structure confirms' bug would enter on the confirm day
        (array index 85) using that day's own not-yet-fully-known close. This proves the actual
        entry happens the NEXT trading day instead."""
        days = _build_uptrend_days()
        df = _build_5min_df(days)
        trades = dow.backtest_instrument("SYN", df)
        self.assertEqual(len(trades), 1)
        # if the bug fired a day early, the position's entry_date would be the confirm day
        # (days[CONFIRM_HIGH2_IDX][0]), not the action day (days[ACTION_DAY_IDX][0])
        self.assertEqual(trades[0]["date"], days[ACTION_DAY_IDX][0].date())

    def test_entry_fires_only_on_the_first_5min_bar_of_the_action_day(self):
        """The action day gets a SECOND, differently-priced bar later the same day - the entry
        price must come from the FIRST bar's close only. Both bars' High/Low are kept exactly at
        the surrounding flat-filler bounds (1.0998-1.1002) so this override doesn't itself spawn
        a brand-new, unintended swing pivot (any High/Low DEVIATING from an isolated flat-filler
        region always becomes a new local extreme in its own confirmation window - see the
        degenerate-stop test below, which deliberately works around the same effect)."""
        days = _build_uptrend_days()
        days = _override_day(days, ACTION_DAY_IDX,
                              [(1.1000, 1.1002, 1.0998, 1.1001),      # first bar - THIS close is the entry
                               (1.1001, 1.1002, 1.0998, 1.0999)])     # later same-day bar - must be ignored for entry pricing
        df = _build_5min_df(days)
        trades = dow.backtest_instrument("SYN", df)
        self.assertEqual(len(trades), 1)
        entry = 1.1001
        stop = LOW2
        sl_distance = entry - stop
        # drift flat afterward so the trade force-closes FLAT at the last available close,
        # letting us read the recorded entry/stop back out via the r calculation
        last_close = days[-1][1][-1][3]
        expected_r = (last_close - entry) / sl_distance
        self.assertEqual(trades[0]["outcome"], "FLAT")
        self.assertAlmostEqual(trades[0]["r"], expected_r, places=6)


# ============================= trend-state transitions =============================

class TestTrendStateTransitions(unittest.TestCase):
    def test_hh_plus_hl_is_uptrend(self):
        daily = _make_daily_df(100, _uptrend_overrides_for_daily_df())
        sig = dow.compute_daily_swing_signals(daily)
        self.assertEqual(sig["trend_state"].iloc[CONFIRM_HIGH2_IDX], "UP")

    def test_lh_plus_ll_is_downtrend(self):
        # mirror: a LOWER high then a LOWER low
        overrides = {PIVOT_LOW1_IDX: (1.1002, 1.0900), PIVOT_HIGH1_IDX: (1.1100, 1.0998),
                     PIVOT_LOW2_IDX: (1.1002, 1.0800),      # LOWER low than 1.0900
                     PIVOT_HIGH2_IDX: (1.1050, 1.0998)}     # LOWER high than 1.1100
        daily = _make_daily_df(100, overrides)
        sig = dow.compute_daily_swing_signals(daily)
        self.assertEqual(sig["trend_state"].iloc[CONFIRM_HIGH2_IDX], "DOWN")
        self.assertTrue(bool(sig["fresh_down"].iloc[CONFIRM_HIGH2_IDX]))
        self.assertFalse(bool(sig["fresh_up"].iloc[CONFIRM_HIGH2_IDX]))

    def test_mixed_structure_is_neither(self):
        # higher high, but LOWER low - not a clean HH+HL, and not LH+LL either
        overrides = {PIVOT_LOW1_IDX: (1.1002, 1.0900), PIVOT_HIGH1_IDX: (1.1100, 1.0998),
                     PIVOT_LOW2_IDX: (1.1002, 1.0800),      # LOWER low
                     PIVOT_HIGH2_IDX: (1.1150, 1.0998)}     # HIGHER high
        daily = _make_daily_df(100, overrides)
        sig = dow.compute_daily_swing_signals(daily)
        self.assertIsNone(sig["trend_state"].iloc[CONFIRM_HIGH2_IDX])
        self.assertFalse(bool(sig["fresh_up"].iloc[CONFIRM_HIGH2_IDX]))
        self.assertFalse(bool(sig["fresh_down"].iloc[CONFIRM_HIGH2_IDX]))


# ============================= "freshly confirmed" fires once =============================

class TestFreshTransitionOnlyOnce(unittest.TestCase):
    def test_state_continuing_to_hold_does_not_retrigger(self):
        overrides = _uptrend_overrides_for_daily_df()
        overrides[PIVOT_LOW3_IDX] = (1.1002, LOW3)   # a further higher low - trend continues
        daily = _make_daily_df(145, overrides)
        sig = dow.compute_daily_swing_signals(daily)

        self.assertTrue(bool(sig["fresh_up"].iloc[CONFIRM_HIGH2_IDX]))
        self.assertEqual(sig["trend_state"].iloc[CONFIRM_LOW3_IDX], "UP")
        # trend state still holds at the LOW3 confirmation, but it's NOT a fresh transition -
        # the state was already UP coming into this day
        self.assertFalse(bool(sig["fresh_up"].iloc[CONFIRM_LOW3_IDX]))
        # the trailing stop level DOES update, though (checked properly in the full-pipeline
        # ratchet test below) - this test isolates just the "no retrigger" claim
        self.assertEqual(sig["trail_stop_long_asof"].iloc[CONFIRM_LOW3_IDX], LOW3)


# ============================= entry mechanics (full pipeline) =============================

class TestEntryMechanics(unittest.TestCase):
    def test_long_entry_price_stop_and_sl_distance(self):
        days = _build_uptrend_days()
        df = _build_5min_df(days)
        trades = dow.backtest_instrument("SYN", df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "LONG")
        entry = FLAT_DAY[3]   # action day is a flat filler day -> close = 1.1000
        stop = LOW2
        sl_distance = entry - stop
        last_close = days[-1][1][-1][3]
        expected_r = (last_close - entry) / sl_distance
        self.assertAlmostEqual(trades[0]["r"], expected_r, places=6)

    def test_degenerate_stop_on_wrong_side_of_entry_is_skipped(self):
        """The action day's close happens to be BELOW the confirmed swing-low stop (a fast
        move between confirmation and action) - nonsensical geometry for a long, must be
        skipped rather than forced through. Data is cut short right after the action day (well
        under SWING_LEN=20 more trading days) so this same override - which necessarily deviates
        from the surrounding flat filler and would otherwise become a brand-new swing pivot of
        its own once SWING_LEN days pass - never gets the chance to confirm and produce an
        unrelated, LEGITIMATE later signal that would muddy this specific guard check."""
        days = _build_uptrend_days(n_total=ACTION_DAY_IDX + 5)
        days = _override_day(days, ACTION_DAY_IDX, [(1.0900, 1.0905, 1.0895, 1.0900)])   # closes below LOW2 (1.0950)
        df = _build_5min_df(days)
        trades = dow.backtest_instrument("SYN", df)
        self.assertEqual(trades, [])


# ============================= trailing-stop ratchet =============================

class TestTrailingStopRatchet(unittest.TestCase):
    def test_stop_ratchets_to_a_later_more_favorable_confirmed_low(self):
        days = _build_uptrend_days()
        # a touch bar strictly between the ORIGINAL stop (LOW2=1.0950) and the RATCHETED stop
        # (LOW3=1.0980) - only fires if the ratchet actually took effect
        touch_low = (LOW2 + LOW3) / 2   # 1.0965
        days = _override_day(days, RATCHET_ACTIVE_IDX, [(LOW3, LOW3 + 0.0005, touch_low, LOW3 - 0.0005)])
        df = _build_5min_df(days)
        trades = dow.backtest_instrument("SYN", df)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "STOP")
        entry = FLAT_DAY[3]
        original_sl_distance = entry - LOW2
        expected_r = (LOW3 - entry) / original_sl_distance   # closed at the RATCHETED level, r denominator unchanged
        self.assertAlmostEqual(trades[0]["r"], expected_r, places=6)
        # sanity: this is a smaller loss than an un-ratcheted stop-out would have been
        self.assertGreater(trades[0]["r"], -1.0)

    def test_ratcheted_level_not_usable_one_day_before_it_activates(self):
        """The exact same touch price placed ONE DAY EARLIER (before the ratchet is actually
        effective) must NOT trigger a stop-out - proving the ratchet's own shift-by-one-day
        timing is enforced, not just its existence."""
        days = _build_uptrend_days()
        touch_low = (LOW2 + LOW3) / 2
        early_idx = RATCHET_ACTIVE_IDX - 1   # still using the OLD stop (LOW2) this day
        days = _override_day(days, early_idx, [(LOW3, LOW3 + 0.0005, touch_low, LOW3 - 0.0005)])
        df = _build_5min_df(days)
        trades = dow.backtest_instrument("SYN", df)
        # touch_low (1.0965) is well above the still-active old stop (1.0950) - no stop-out here
        self.assertEqual(len(trades), 1)
        self.assertNotEqual(trades[0]["outcome"], "STOP")


# ============================= one trade at a time / only enter when flat =============================

class TestOnlyEnterWhenFlat(unittest.TestCase):
    def test_second_fresh_signal_while_in_a_position_is_ignored(self):
        # add a THIRD, later downtrend-completing pivot pair that would otherwise fire a fresh
        # SHORT signal while the LONG from CONFIRM_HIGH2_IDX is still open - must be ignored
        days = _build_uptrend_days(n_total=200)
        # override a late pivot low BELOW everything seen so far, followed by a lower high, to
        # attempt to build LH+LL structure - even if trend_state briefly reads DOWN somewhere,
        # position management must ignore it while already LONG (this is checked structurally:
        # only one trade total should ever appear)
        days = _override_day(days, 150, [(1.1000, 1.1002, 1.0500, 1.0600)])
        days = _override_day(days, 155, [(1.0700, 1.0998, 1.0998, 1.0700)])
        df = _build_5min_df(days)
        trades = dow.backtest_instrument("SYN", df)
        self.assertEqual(len(trades), 1)   # exactly one trade - the original LONG, held/force-closed
        self.assertEqual(trades[0]["side"], "LONG")


# ============================= force-close accounting =============================

class TestForceCloseAtEnd(unittest.TestCase):
    def test_open_position_force_closed_flat_at_last_bar(self):
        days = _build_uptrend_days(n_total=100)   # data ends well before any stop/trail could fire
        df = _build_5min_df(days)
        trades = dow.backtest_instrument("SYN", df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "FLAT")
        entry = FLAT_DAY[3]
        sl_distance = entry - LOW2
        last_close = days[-1][1][-1][3]
        expected_r = (last_close - entry) / sl_distance
        self.assertAlmostEqual(trades[0]["r"], expected_r, places=6)

    def test_no_confirmed_structure_yet_produces_no_trades(self):
        days = _flat_days(30)   # far too short for any SWING_LEN=20 pivot to even confirm
        df = _build_5min_df(days)
        trades = dow.backtest_instrument("SYN", df)
        self.assertEqual(trades, [])


# ============================= synthetic end-to-end smoke run + null check =============================

class TestEndToEndSmokeRunAndNullResult(unittest.TestCase):
    """ONE small synthetic multi-year random-walk run confirming the full pipeline (daily
    resample -> pivot confirmation -> trend structure -> entry -> trailing exit -> reporting-
    style aggregation) runs without crashing, and doesn't show a suspiciously large edge on pure
    noise - same spirit as donchian_turtle_breakout_dukascopy_backtest.py's
    TestRandomWalkNullResult. Deliberately NOT a real Dukascopy download."""

    def test_smoke_run_and_no_suspiciously_large_edge_on_pure_noise(self):
        all_trades = []
        for seed in range(6):
            df = _make_random_walk_5min_df(n_days=600, bars_per_day=12, seed=seed)
            trades = dow.backtest_instrument("SYN", df)
            for t in trades:
                t["instrument"] = f"SYN_{seed}"
            all_trades.extend(trades)

        for t in all_trades:
            self.assertIn(t["side"], ("LONG", "SHORT"))
            self.assertIn(t["outcome"], ("STOP", "FLAT"))
            self.assertIsInstance(t["r"], float)

        if len(all_trades) >= 10:
            all_r = [t["r"] for t in all_trades]
            n = len(all_r)
            avg_r = sum(all_r) / n
            std_r = np.std(all_r, ddof=1)
            z = avg_r / (std_r / (n ** 0.5)) if std_r > 0 else 0.0
            # generous bounds, same rationale as Donchian's own null check - a genuine
            # implementation bug (e.g. a lookahead leak) would be expected to blow well past these
            self.assertLess(abs(avg_r), 1.0, f"suspiciously large average R/trade on pure noise: {avg_r:.3f}")
            self.assertLess(abs(z), 3.5, f"suspiciously large z-score on pure noise: {z:.2f}")


if __name__ == "__main__":
    unittest.main()
