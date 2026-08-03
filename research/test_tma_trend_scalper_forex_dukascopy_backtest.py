# Unit tests for tma_trend_scalper_forex_dukascopy_backtest.py.
#
# Covers, per this project's rigor conventions:
#   1. Indicator fidelity to the Pine built-ins they stand in for - ta.sma, ta.rma (Wilder), ta.tr,
#      ta.atr, ta.rsi and ta.dmi's ADX - including the WARMUP RULE (undefined until the window is
#      full, never approximated from a shorter window), because getting the seed wrong shifts
#      every downstream value silently.
#   2. THE WEEKDAY OFF-BY-ONE. Pine's dayofweek is 1=Sunday, so the source's `>= 1 and <= 5` is
#      Sunday-to-Thursday, not Monday-to-Friday. Both modes are pinned explicitly, day by day,
#      because this is the single easiest thing in the port to "fix" by accident.
#   3. The session hour window, including its exclusive upper bound.
#   4. Pattern detection: 3-line strike and engulfing, both directions, plus the near-miss cases
#      that must NOT fire.
#   5. THE NEXT-BAR-OPEN FILL. The stop and target are computed from the signal bar's close while
#      the entry lands on the next bar's open, so the realised reward is NOT the nominal 2R. Pinned
#      in both directions of the gap, plus the skip when a gap puts the entry past its own stop.
#   6. MIN_STOP_PCT: a hairline signal candle is skipped rather than scored as a huge R-multiple.
#   7. One position at a time - signals during an open trade are not even evaluated.
#   8. Trade management: TP, SL, the pessimistic same-bar tie-break, and the mark-to-last-close of
#      a position still open at the end of the data (the source has no time-based exit at all).
#   9. A LONG/SHORT mirror check on a full scenario.
#  10. The webapp trade contract: every trade carries `date` and `stop_pct`.
#
# Run with:  python -m pytest research/test_tma_trend_scalper_forex_dukascopy_backtest.py -v

import contextlib
import datetime
import importlib.util
import math
import os
import unittest

import pandas as pd

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "tma_trend_scalper_forex_dukascopy_backtest.py")
_spec = importlib.util.spec_from_file_location("tma_trend_scalper_forex_dukascopy_backtest", _MODULE_PATH)
tma = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tma)


@contextlib.contextmanager
def config(**overrides):
    previous = {name: getattr(tma, name) for name in overrides}
    for name, value in overrides.items():
        setattr(tma, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(tma, name, value)


# ================================ indicators ================================

class TestMovingAverages(unittest.TestCase):
    def test_sma_is_undefined_during_warmup_then_exact(self):
        out = tma.sma_series([1.0, 2.0, 3.0, 4.0, 5.0], 3)
        self.assertEqual(out[:2], [None, None])
        self.assertAlmostEqual(out[2], 2.0)
        self.assertAlmostEqual(out[3], 3.0)
        self.assertAlmostEqual(out[4], 4.0)

    def test_rma_seeds_with_the_simple_average_then_recurses(self):
        values = [1.0, 2.0, 3.0, 10.0, 10.0]
        out = tma.rma_series(values, 3)
        self.assertEqual(out[:2], [None, None])
        self.assertAlmostEqual(out[2], 2.0)                      # seed = (1+2+3)/3
        self.assertAlmostEqual(out[3], (2.0 * 2 + 10.0) / 3)     # then Wilder's recursion
        self.assertAlmostEqual(out[4], (out[3] * 2 + 10.0) / 3)

    def test_rma_start_index_skips_leading_nones_rather_than_treating_them_as_zero(self):
        # A true-range series has no value on bar 0. Seeding from index 0 with a substituted 0.0
        # would drag the seed down by 1/length - this pins that it doesn't happen.
        out = tma.rma_series([None, 3.0, 3.0, 3.0], 3, start_index=1)
        self.assertEqual(out[:3], [None, None, None])
        self.assertAlmostEqual(out[3], 3.0)

    def test_rma_returns_all_none_when_there_is_not_enough_data(self):
        self.assertEqual(tma.rma_series([1.0, 2.0], 5), [None, None])

    def test_rma_matches_the_sources_smma_recurrence(self):
        # The Pine writes it as (prev*(len-1) + close)/len, which is what rma_series must produce.
        closes = [float(v) for v in range(1, 30)]
        out = tma.rma_series(closes, 5)
        prev = out[4]
        for i in range(5, len(closes)):
            prev = (prev * 4 + closes[i]) / 5
            self.assertAlmostEqual(out[i], prev)


class TestTrueRangeAndAtr(unittest.TestCase):
    def test_true_range_uses_the_widest_of_the_three_measures(self):
        highs = [10.0, 12.0, 11.0]
        lows = [9.0, 11.5, 8.0]
        closes = [9.5, 11.8, 8.5]
        tr = tma.true_range_series(highs, lows, closes)
        self.assertIsNone(tr[0])
        self.assertAlmostEqual(tr[1], 12.0 - 9.5)     # high - previous close beats high - low
        self.assertAlmostEqual(tr[2], 11.8 - 8.0)     # previous close - low

    def test_atr_of_a_constant_range_is_that_range(self):
        n = 60
        highs = [101.0] * n
        lows = [99.0] * n
        closes = [100.0] * n
        atr = tma.atr_series(highs, lows, closes, 14)
        self.assertAlmostEqual(atr[-1], 2.0)


class TestRsi(unittest.TestCase):
    def test_unbroken_advance_pins_rsi_at_100(self):
        closes = [100.0 + i for i in range(40)]
        self.assertAlmostEqual(tma.rsi_series(closes, 14)[-1], 100.0)

    def test_unbroken_decline_pins_rsi_at_0(self):
        closes = [100.0 - i for i in range(40)]
        self.assertAlmostEqual(tma.rsi_series(closes, 14)[-1], 0.0)

    def test_symmetric_alternation_hovers_around_50(self):
        # Perfectly alternating gains and losses of equal size: the two rma legs converge to the
        # same value, so RSI oscillates tightly around 50 rather than landing exactly on it (the
        # recursion is one bar out of phase with itself).
        closes = [100.0 + (1.0 if i % 2 else 0.0) for i in range(200)]
        self.assertAlmostEqual(tma.rsi_series(closes, 14)[-1], 50.0, delta=5.0)

    def test_rsi_is_undefined_before_its_window_fills(self):
        rsi = tma.rsi_series([100.0 + i for i in range(40)], 14)
        self.assertTrue(all(v is None for v in rsi[:14]))
        self.assertIsNotNone(rsi[14])


class TestAdx(unittest.TestCase):
    def test_clean_one_way_trend_produces_a_high_adx(self):
        n = 120
        closes = [100.0 + i for i in range(n)]
        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]
        adx = tma.adx_series(highs, lows, closes, 14)
        self.assertGreater(adx[-1], 90.0)

    def test_directionless_chop_produces_a_low_adx(self):
        n = 300
        closes = [100.0 + (i % 2) for i in range(n)]
        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]
        adx = tma.adx_series(highs, lows, closes, 14)
        self.assertLess(adx[-1], 25.0)

    def test_adx_stays_within_bounds_and_warms_up_late(self):
        n = 200
        closes = [100.0 + (i * 0.3 if i < 100 else 130.0 - (i - 100) * 0.2) for i in range(n)]
        highs = [c + 0.4 for c in closes]
        lows = [c - 0.4 for c in closes]
        adx = tma.adx_series(highs, lows, closes, 14)
        defined = [v for v in adx if v is not None]
        self.assertTrue(defined)
        self.assertTrue(all(0.0 <= v <= 100.0 for v in defined))
        # DX itself needs 14 bars, and the ADX is a 14-period rma OF that - so ~28 bars minimum.
        self.assertTrue(all(v is None for v in adx[:27]))


# ================================ session gate ================================

class TestSessionGate(unittest.TestCase):
    """2024-03-03 is a Sunday, so 03..09 March covers Sunday through Saturday in order."""

    WEEK = {"Sun": "2024-03-03", "Mon": "2024-03-04", "Tue": "2024-03-05", "Wed": "2024-03-06",
            "Thu": "2024-03-07", "Fri": "2024-03-08", "Sat": "2024-03-09"}

    def _at(self, day, hour):
        return pd.Timestamp(f"{self.WEEK[day]} {hour:02d}:00", tz="UTC")

    def test_source_mode_trades_sunday_to_thursday_and_never_friday(self):
        with config(WEEKDAY_FILTER_MODE=0):
            allowed = {d for d in self.WEEK if tma.in_session(self._at(d, 10))}
        self.assertEqual(allowed, {"Sun", "Mon", "Tue", "Wed", "Thu"})
        self.assertNotIn("Fri", allowed)

    def test_corrected_mode_trades_monday_to_friday(self):
        with config(WEEKDAY_FILTER_MODE=1):
            allowed = {d for d in self.WEEK if tma.in_session(self._at(d, 10))}
        self.assertEqual(allowed, {"Mon", "Tue", "Wed", "Thu", "Fri"})

    def test_the_two_modes_disagree_on_exactly_friday_and_sunday(self):
        with config(WEEKDAY_FILTER_MODE=0):
            source = {d for d in self.WEEK if tma.in_session(self._at(d, 10))}
        with config(WEEKDAY_FILTER_MODE=1):
            corrected = {d for d in self.WEEK if tma.in_session(self._at(d, 10))}
        self.assertEqual(source ^ corrected, {"Fri", "Sun"})

    def test_hour_window_is_inclusive_of_the_start_and_exclusive_of_the_end(self):
        with config(WEEKDAY_FILTER_MODE=1, SESSION_START_HOUR=7, SESSION_END_HOUR=15):
            self.assertFalse(tma.in_session(self._at("Mon", 6)))
            self.assertTrue(tma.in_session(self._at("Mon", 7)))
            self.assertTrue(tma.in_session(self._at("Mon", 14)))
            self.assertFalse(tma.in_session(self._at("Mon", 15)))


# ================================ scenario harness ================================
#
# A rising but WAVY warmup: the 21/50/200 SMMAs stack in order with real separation, ADX runs
# high, ATR is stable and non-zero, and momentum holds - while RSI oscillates in the 70s instead of
# pinning at 100. That last part is why the warmup has pullbacks rather than being a clean ramp:
# `rsiBullish` requires RSI to be above its OWN 50-period SMMA, and on an unbroken advance both sit
# at 100, so the condition becomes unsatisfiable and every scenario would silently produce zero
# trades. Bars are 5 minutes apart on a Wednesday, which both weekday modes allow.

WARMUP = 400
BASE = 1.0
STEP = 0.0004
WARMUP_CYCLE = (1.0, 1.0, 1.0, -0.6)   # three up bars, one pullback - net uptrend, RSI in the 70s


def _bar(open_, close, wick=0.15):
    return {"Open": open_, "High": max(open_, close) + STEP * wick,
            "Low": min(open_, close) - STEP * wick, "Close": close}


def build_scenario(pattern_bars, follow_through, start="2024-03-06 07:00"):
    """WARMUP wavy-uptrend bars, then `pattern_bars(level)`, then `follow_through(signal_bar)`.
    Returns the DataFrame and the index of the signal bar (the last of pattern_bars)."""
    rows = []
    level = BASE
    for i in range(WARMUP):
        delta = WARMUP_CYCLE[i % len(WARMUP_CYCLE)] * STEP * 0.6
        rows.append(_bar(level, level + delta))
        level += delta
    rows.extend(pattern_bars(level))
    signal_index = len(rows) - 1
    rows.extend(follow_through(rows[signal_index]))
    index = pd.date_range(start=pd.Timestamp(start, tz="UTC"), periods=len(rows), freq="5min")
    return pd.DataFrame(rows, index=index), signal_index


def _three_shallow_dips(level, dip=0.04):
    bars = []
    for _ in range(3):
        bars.append(_bar(level, level - STEP * dip))
        level -= STEP * dip
    return bars, level


def _bull_strike_pattern(level):
    """Three small red candles then a strong candle closing above the previous candle's open. The
    dips are shallow so momentum and RSI still read bullish at the signal bar."""
    bars, level = _three_shallow_dips(level)
    return bars + [_bar(level, bars[-1]["Open"] + STEP * 2.0)]


def _bull_strike_not_engulfing(level):
    """Same strike, but opening ABOVE the previous close so the engulfing test cannot also fire -
    isolates the strike branch of _pattern_name()."""
    bars, level = _three_shallow_dips(level)
    return bars + [_bar(level + STEP * 0.01, bars[-1]["Open"] + STEP * 2.0)]


def _hold_flat(signal_bar, n=6):
    """Bars that neither reach the stop nor the target - used when the test only cares about entry."""
    price = signal_bar["Close"]
    return [{"Open": price, "High": price + STEP * 0.02, "Low": price - STEP * 0.02, "Close": price}
            for _ in range(n)]


def run_scenario(df, label="TEST", **overrides):
    overrides.setdefault("WEEKDAY_FILTER_MODE", 1)
    overrides.setdefault("SESSION_START_HOUR", 0)
    overrides.setdefault("SESSION_END_HOUR", 24)
    with config(**overrides):
        return tma.backtest_instrument(label, df)


class TestScenarioHarness(unittest.TestCase):
    """If the harness stops producing a trade, every scenario test below becomes vacuous - so the
    harness itself is pinned first."""

    def test_the_ramp_plus_bull_strike_produces_exactly_one_long(self):
        df, _ = build_scenario(_bull_strike_pattern, _hold_flat)
        trades = run_scenario(df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "LONG")
        # This candle satisfies both branches at once (it opens at the previous close), which is
        # itself the "both" label's reason to exist.
        self.assertEqual(trades[0]["pattern"], "both")

    def test_a_strike_that_is_not_also_an_engulfing_is_labelled_a_strike(self):
        df, _ = build_scenario(_bull_strike_not_engulfing, _hold_flat)
        trades = run_scenario(df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["pattern"], "3-line strike")

    def test_each_trend_filter_actually_binds(self):
        df, _ = build_scenario(_bull_strike_pattern, _hold_flat)
        self.assertEqual(len(run_scenario(df)), 1)
        # Each of these makes one gate unsatisfiable; the trade must disappear in every case.
        self.assertEqual(run_scenario(df, ADX_MIN=101.0), [])
        self.assertEqual(run_scenario(df, ATR_MIN_MULT=1000.0), [])
        # "TEST" isn't a key in MIN_SEPARATION_ABS_BY_INSTRUMENT, so it falls back to
        # DEFAULT_MIN_SEPARATION_ABS - overriding that is how to make the separation gate bind here.
        self.assertEqual(run_scenario(df, DEFAULT_MIN_SEPARATION_ABS=100.0), [])
        self.assertEqual(run_scenario(df, SESSION_START_HOUR=23, SESSION_END_HOUR=24), [])


class TestPerInstrumentSeparationTable(unittest.TestCase):
    """MIN_SEPARATION_ABS_BY_INSTRUMENT holds the writeup's own per-pair values (0.001 for EURUSD/
    GBPUSD/AUDUSD, 0.10 for USDJPY) as absolute price distances, not a synthetic percentage - so the
    same scenario must be read differently depending on which instrument label is passed in."""

    def test_the_three_majors_share_the_tight_threshold(self):
        df, _ = build_scenario(_bull_strike_pattern, _hold_flat)
        for label in ("EURUSD", "GBPUSD", "AUDUSD"):
            trades = run_scenario(df, label=label)
            self.assertEqual(len(trades), 1, f"{label} should trade at the 0.001 threshold")

    def test_usdjpy_uses_a_stricter_threshold_and_rejects_the_same_scenario(self):
        # At the price scale this harness builds (~1.0), USDJPY's documented 0.10 absolute
        # separation is far wider than the SMMA stack the scenario actually produces - the same
        # setup that qualifies for EURUSD/GBPUSD/AUDUSD must NOT qualify for USDJPY.
        df, _ = build_scenario(_bull_strike_pattern, _hold_flat)
        self.assertEqual(run_scenario(df, label="USDJPY"), [])

    def test_an_unlisted_instrument_falls_back_to_the_default(self):
        df, _ = build_scenario(_bull_strike_pattern, _hold_flat)
        with_default = run_scenario(df, label="NOT_IN_THE_TABLE")
        with_fallback_overridden = run_scenario(df, label="NOT_IN_THE_TABLE", DEFAULT_MIN_SEPARATION_ABS=100.0)
        self.assertEqual(len(with_default), 1)
        self.assertEqual(with_fallback_overridden, [])


# ================================ patterns ================================

class TestPatternDetection(unittest.TestCase):
    def test_three_line_strike_needs_three_same_colour_candles_first(self):
        def only_two_red(level):
            bars = _bull_strike_pattern(level)
            bars[0] = _bar(bars[0]["Open"], bars[0]["Open"] + STEP * 0.04)   # break the run of three
            return bars
        df, _ = build_scenario(only_two_red, _hold_flat)
        trades = run_scenario(df)
        # It still qualifies as an engulfing; what must not happen is it being called a strike.
        self.assertTrue(all(t["pattern"] == "engulfing" for t in trades))

    def test_engulfing_is_recognised_on_its_own(self):
        def engulf(level):
            # Two filler up-bars so the three-candle lookback is defined but is not a run of three
            # same-coloured candles, then one down bar, then a candle engulfing it.
            first = _bar(level, level + STEP * 0.04)
            second = _bar(first["Close"], first["Close"] + STEP * 0.04)
            prev_open = second["Close"]
            prev_close = prev_open - STEP * 0.04
            prev = _bar(prev_open, prev_close)
            cur = _bar(prev_close - STEP * 0.01, prev_open + STEP * 2.0)
            return [first, second, prev, cur]
        df, _ = build_scenario(engulf, _hold_flat)
        trades = run_scenario(df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["pattern"], "engulfing")

    def test_no_pattern_means_no_trade_even_in_a_perfect_trend(self):
        def only_up_bars(level):
            bars = []
            for _ in range(4):
                bars.append(_bar(level, level + STEP * 0.6))
                level += STEP * 0.6
            return bars
        df, _ = build_scenario(only_up_bars, _hold_flat)
        self.assertEqual(run_scenario(df), [])


# ================================ the fill convention ================================

class TestNextBarOpenFill(unittest.TestCase):
    """FILL_AT_NEXT_OPEN defaults to 1: enter at the OPEN of the candle after the signal candle
    CLOSES. That is the writeup's own documented entry timing ('wait for the candle to close...
    enter at open of next candle'), not an accidental Pine quirk - reading the Pine alone, before
    the writeup existed, made it look like an unintended mismatch between the entry price and the
    prices the stop/target were computed from. It is not; it's the design. The realised risk:reward
    still isn't a clean 2:1 as a result of that gap (0.30R-4.62R on test data), which is a genuine
    property of the strategy, pinned here rather than hidden. Every test below sets FILL_AT_NEXT_OPEN
    explicitly rather than relying on the module's current default, so a future default change can't
    silently invalidate what's pinned here."""

    def _scenario_with_gap(self, gap_multiple):
        def follow(signal_bar):
            opening = signal_bar["Close"] + STEP * gap_multiple
            bars = [{"Open": opening, "High": opening + STEP * 0.02,
                      "Low": opening - STEP * 0.02, "Close": opening}]
            bars.extend(_hold_flat({"Close": opening}, n=6))
            return bars
        return build_scenario(_bull_strike_pattern, follow)

    def test_the_default_fills_at_the_next_bar_open_and_shrinks_a_favourable_gaps_risk(self):
        df, signal = self._scenario_with_gap(0.5)
        trades = run_scenario(df)          # no override - pins the actual shipped default
        self.assertEqual(len(trades), 1)
        t = trades[0]
        signal_close = df["Close"].iloc[signal]
        candle = df["High"].iloc[signal] - df["Low"].iloc[signal]
        expected_entry = df["Open"].iloc[signal + 1]
        expected_stop = signal_close - candle * tma.STOP_CANDLE_MULT
        self.assertAlmostEqual(t["entry"], expected_entry)
        self.assertAlmostEqual(t["sl_distance"], expected_entry - expected_stop)
        # Entry above the reference price: risk is larger than the nominal 2x candle, so the
        # reward multiple on a win is BELOW the intended 2.0 - a real property of the design.
        self.assertGreater(t["sl_distance"], candle * tma.STOP_CANDLE_MULT)

    def test_filling_at_the_signal_close_opt_out_gives_exactly_the_nominal_two_r(self):
        df, signal = self._scenario_with_gap(0.5)
        trades = run_scenario(df, FILL_AT_NEXT_OPEN=0)
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertAlmostEqual(t["entry"], df["Close"].iloc[signal])
        candle = df["High"].iloc[signal] - df["Low"].iloc[signal]
        self.assertAlmostEqual(t["sl_distance"], candle * tma.STOP_CANDLE_MULT)

    def test_the_two_fill_conventions_produce_different_risk(self):
        df, _ = self._scenario_with_gap(0.5)
        at_open = run_scenario(df, FILL_AT_NEXT_OPEN=1)[0]["sl_distance"]
        at_close = run_scenario(df, FILL_AT_NEXT_OPEN=0)[0]["sl_distance"]
        self.assertNotAlmostEqual(at_open, at_close)

    def test_a_gap_straight_through_the_stop_is_skipped_not_scored(self):
        # Opening far BELOW the intended long stop leaves entry < stop: a negative risk. Scoring it
        # would invent an instant winner out of an unfillable setup. Only the (default) next-open-
        # fill convention can produce this - filling at the signal close never gaps past its stop.
        df, _ = self._scenario_with_gap(-5.0)
        self.assertEqual(run_scenario(df), [])

    def test_hairline_signal_candle_is_skipped(self):
        def flat_signal(level):
            bars = _bull_strike_pattern(level)
            mid = bars[-1]["Close"]
            bars[-1] = {"Open": mid, "High": mid + 1e-9, "Low": mid - 1e-9, "Close": mid}
            return bars
        df, _ = build_scenario(flat_signal, _hold_flat)
        # Thin regardless of fill convention - the candle range itself is near-zero.
        self.assertEqual(run_scenario(df), [])
        self.assertEqual(run_scenario(df, FILL_AT_NEXT_OPEN=0), [])


# ================================ trade management ================================

class TestTradeManagement(unittest.TestCase):
    def _with_exit(self, reach):
        """`reach` maps the signal bar to the price the next-next bar should touch."""
        def follow(signal_bar):
            opening = signal_bar["Close"]
            entry_bar = {"Open": opening, "High": opening + STEP * 0.02,
                          "Low": opening - STEP * 0.02, "Close": opening}
            target_price = reach(signal_bar)
            hit = {"Open": opening, "High": max(opening, target_price),
                    "Low": min(opening, target_price), "Close": target_price}
            return [entry_bar, hit] + _hold_flat({"Close": target_price}, n=4)
        return build_scenario(_bull_strike_pattern, follow)

    def test_target_hit_pays_the_realised_reward_multiple(self):
        df, signal = self._with_exit(lambda b: b["Close"] + STEP * 20)
        trades = run_scenario(df)
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t["outcome"], "TP")
        candle = df["High"].iloc[signal] - df["Low"].iloc[signal]
        expected_reward = (df["Close"].iloc[signal] + candle * tma.TARGET_CANDLE_MULT) - t["entry"]
        self.assertAlmostEqual(t["r"], expected_reward / t["sl_distance"])

    def test_stop_hit_is_exactly_minus_one_r(self):
        df, _ = self._with_exit(lambda b: b["Close"] - STEP * 20)
        trades = run_scenario(df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "SL")
        self.assertAlmostEqual(trades[0]["r"], -1.0)

    def test_a_bar_spanning_both_is_resolved_as_the_stop(self):
        def follow(signal_bar):
            opening = signal_bar["Close"]
            entry_bar = {"Open": opening, "High": opening + STEP * 0.02,
                          "Low": opening - STEP * 0.02, "Close": opening}
            both = {"Open": opening, "High": opening + STEP * 30,
                     "Low": opening - STEP * 30, "Close": opening}
            return [entry_bar, both] + _hold_flat({"Close": opening}, n=4)
        df, _ = build_scenario(_bull_strike_pattern, follow)
        trades = run_scenario(df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "SL")

    def test_position_open_at_the_end_of_data_is_marked_to_the_last_close(self):
        df, _ = build_scenario(_bull_strike_pattern, lambda b: _hold_flat(b, n=3))
        trades = run_scenario(df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "FLAT")
        self.assertTrue(math.isfinite(trades[0]["r"]))

    def test_no_second_entry_while_a_position_is_open(self):
        # A second identical pattern immediately after the first must be ignored entirely.
        def follow(signal_bar):
            level = signal_bar["Close"]
            return _bull_strike_pattern(level) + _hold_flat({"Close": level}, n=6)
        df, _ = build_scenario(_bull_strike_pattern, follow)
        trades = run_scenario(df)
        self.assertEqual(len(trades), 1)


# ================================ daily trade cap ================================

class TestMaxTradesPerDay(unittest.TestCase):
    """The writeup's explicit, repeated rule ('One Trade Per Day (Maximum)... NOT 50 trades, NOT
    10 trades, just ONE') isn't enforced by the Pine code at all - it only blocks a second trade
    while the FIRST is still open. This scenario has the first trade hit its target and CLOSE, then
    a second, independent, fully-qualifying signal fire later the SAME calendar day."""

    def _two_signals_same_day(self):
        def first_signal(level):
            return _bull_strike_pattern(level)

        def rest_of_day(signal_bar):
            opening = signal_bar["Close"]
            fill_bar = _bar(opening, opening + STEP * 0.02)          # the fill bar itself
            target_reach = opening + STEP * 20                        # far enough to hit any target
            hit_bar = {"Open": opening, "High": target_reach,
                        "Low": opening - STEP * 0.02, "Close": target_reach}   # closes the 1st trade
            continuation = []                                          # keeps trend/momentum/RSI
            level = target_reach                                       # bullish for a 2nd signal
            for _ in range(6):
                continuation.append(_bar(level, level + STEP * 0.3))
                level += STEP * 0.3
            second_signal = _bull_strike_pattern(level)
            tail = _hold_flat({"Close": second_signal[-1]["Close"]}, n=6)
            return [fill_bar, hit_bar] + continuation + second_signal + tail

        return build_scenario(first_signal, rest_of_day)

    def test_the_default_cap_blocks_a_second_same_day_signal(self):
        df, _ = self._two_signals_same_day()
        trades = run_scenario(df)          # no override - pins the actual shipped default (cap=1)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "TP")

    def test_disabling_the_cap_allows_both_same_day_signals(self):
        df, _ = self._two_signals_same_day()
        trades = run_scenario(df, MAX_TRADES_PER_DAY_PER_INSTRUMENT=0)
        self.assertEqual(len(trades), 2)
        # Both genuinely land on the same calendar day - not a date-rollover coincidence.
        self.assertEqual(trades[0]["date"], trades[1]["date"])

    def test_a_higher_cap_allows_exactly_that_many(self):
        df, _ = self._two_signals_same_day()
        trades = run_scenario(df, MAX_TRADES_PER_DAY_PER_INSTRUMENT=2)
        self.assertEqual(len(trades), 2)

    def test_the_cap_resets_on_a_new_calendar_day(self):
        # Same two-signal shape, but with the second signal pushed into the following session
        # rather than later the same day - the cap must NOT carry over across the date boundary.
        def first_signal(level):
            return _bull_strike_pattern(level)

        def next_day_follow(signal_bar):
            opening = signal_bar["Close"]
            fill_bar = _bar(opening, opening + STEP * 0.02)
            target_reach = opening + STEP * 20
            hit_bar = {"Open": opening, "High": target_reach,
                        "Low": opening - STEP * 0.02, "Close": target_reach}
            # ~26 hours of continuation (312 5-min bars) pushes the second signal well past
            # midnight into a later calendar day. Reuses WARMUP_CYCLE's own pace (not a flatter
            # one) - a near-flat stretch this long lets the SMMAs converge and collapses the
            # separation gate, which isn't what this test is checking.
            continuation = []
            level = target_reach
            for k in range(312):
                delta = WARMUP_CYCLE[k % len(WARMUP_CYCLE)] * STEP * 0.6
                continuation.append(_bar(level, level + delta))
                level += delta
            second_signal = _bull_strike_pattern(level)
            tail = _hold_flat({"Close": second_signal[-1]["Close"]}, n=6)
            return [fill_bar, hit_bar] + continuation + second_signal + tail

        df, _ = build_scenario(first_signal, next_day_follow)
        trades = run_scenario(df)   # default cap=1, but the two signals are on different dates
        self.assertEqual(len(trades), 2)
        self.assertNotEqual(trades[0]["date"], trades[1]["date"])


# ================================ symmetry + contract ================================

class TestLongShortMirror(unittest.TestCase):
    def test_a_mirrored_downtrend_produces_the_mirrored_short(self):
        df_up, _ = build_scenario(_bull_strike_pattern, _hold_flat)
        pivot = 2.0
        df_down = pd.DataFrame({
            "Open": pivot - df_up["Open"], "High": pivot - df_up["Low"],
            "Low": pivot - df_up["High"], "Close": pivot - df_up["Close"],
        }, index=df_up.index)
        longs = run_scenario(df_up)
        shorts = run_scenario(df_down)
        self.assertEqual(len(longs), len(shorts))
        self.assertEqual(len(longs), 1)
        self.assertEqual(longs[0]["side"], "LONG")
        self.assertEqual(shorts[0]["side"], "SHORT")
        self.assertEqual(longs[0]["outcome"], shorts[0]["outcome"])
        self.assertAlmostEqual(longs[0]["r"], shorts[0]["r"])
        self.assertAlmostEqual(longs[0]["sl_distance"], shorts[0]["sl_distance"])


class TestWebappContract(unittest.TestCase):
    def test_every_trade_carries_date_and_stop_pct(self):
        df, _ = build_scenario(_bull_strike_pattern, _hold_flat)
        trades = run_scenario(df)
        self.assertTrue(trades)
        for t in trades:
            self.assertIsInstance(t["date"], datetime.date)
            self.assertGreater(t["stop_pct"], 0.0)
            self.assertAlmostEqual(t["stop_pct"], t["sl_distance"] / t["entry"])
            self.assertTrue(math.isfinite(t["r"]))
            self.assertIn(t["side"], ("LONG", "SHORT"))
            self.assertIn(t["pattern"], ("3-line strike", "engulfing", "both"))

    def test_date_is_the_entry_bar_not_the_exit_bar(self):
        df, signal = build_scenario(_bull_strike_pattern, lambda b: _hold_flat(b, n=600))
        trades = run_scenario(df)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["date"], df.index[signal + 1].date())

    def test_too_little_data_returns_no_trades(self):
        idx = pd.date_range("2024-03-06 07:00", periods=50, freq="5min", tz="UTC")
        df = pd.DataFrame({"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0}, index=idx)
        self.assertEqual(tma.backtest_instrument("TEST", df), [])


if __name__ == "__main__":
    unittest.main()
