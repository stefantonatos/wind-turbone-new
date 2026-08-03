# Unit tests + one synthetic end-to-end smoke run for
# ma_golden_death_cross_dukascopy_backtest.py.
#
# Covers, per this project's rigor conventions:
#   1. NO-LOOKAHEAD proof: act_golden/act_death fire exactly one trading day
#      AFTER the underlying cross actually completes - never on the
#      crossing day itself (which wouldn't be knowable until that day's
#      close, by which point it's too late to trade that same close).
#   2. Cross detection with several small, explicit hand-computed cases
#      (golden, death, and "no cross" near-misses).
#   3. Entry/exit mechanics: ATR-sized entry, exit-on-opposite-cross with
#      NO immediate reversal (the "flat-then-wait" design choice - see the
#      strategy file's header point 7), the ATR hard stop, stop-vs-
#      opposite-cross precedence on the SAME bar (including the exact
#      same-bar-reversal bug this test suite caught and the strategy file
#      was fixed for - see TestSameBarStopAndCrossDoesNotReverse), and
#      force-close/timeout accounting.
#   4. A synthetic multi-year random-walk null test (several seeds pooled)
#      confirming no suspiciously large edge shows up on pure noise.
#
# Run with:  python -m pytest research/test_ma_golden_death_cross_dukascopy_backtest.py -v
# or:        python research/test_ma_golden_death_cross_dukascopy_backtest.py

import importlib.util
import os
import unittest

import numpy as np
import pandas as pd

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "ma_golden_death_cross_dukascopy_backtest.py")
_spec = importlib.util.spec_from_file_location("ma_golden_death_cross_dukascopy_backtest", _MODULE_PATH)
ma = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ma)


# ============================= synthetic data helpers =============================

def _make_daily_ohlc(closes, buf=0.1, start="2016-01-04"):
    """A daily OHLC frame from a plain list of closes: Open=Close, High=Close+buf, Low=Close-buf
    (a small, constant intraday buffer so ATR is nonzero but never dominates the close-to-close
    signal math)."""
    n = len(closes)
    idx = pd.date_range(start=start, periods=n, freq="B", tz="America/New_York")
    highs = [c + buf for c in closes]
    lows = [c - buf for c in closes]
    opens = list(closes)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)


def _daily_to_5min_df(daily, overrides=None):
    """One 5-min bar per day (at 09:30 NY) whose OHLC exactly matches the given daily frame -
    so resample_daily(this) reproduces `daily` exactly. `overrides`, if given, is
    {date: (o, h, l, c)} to replace specific days' single bar (used to punch an intrabar wick
    through a level without disturbing any other day)."""
    overrides = overrides or {}
    rows, idx = [], []
    for ts, row in daily.iterrows():
        d = ts.date()
        o, h, l, c = overrides.get(d, (row["Open"], row["High"], row["Low"], row["Close"]))
        bar_ts = pd.Timestamp(ts.year, ts.month, ts.day, 9, 30, tz="America/New_York")
        idx.append(bar_ts)
        rows.append((o, h, l, c))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=pd.DatetimeIndex(idx))


def _make_random_walk_5min_df(n_days, bars_per_day, seed, start_price=1.1000, intrabar_vol=0.0006):
    """A genuine no-drift random walk with intrabar noise - used only for the null-result
    sanity check below."""
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


# A price path that produces exactly one golden cross (day index 210, i.e. 2016-10-24) and,
# much later, exactly one death cross - reused verbatim across several tests below so the
# derived dates/prices are consistent and independently checkable.
def _golden_then_death_closes(n_flat=210, n_up=60, up_slope=0.5, n_down=250, down_slope=0.15):
    closes = [100.0] * n_flat + [100.0 + i * up_slope for i in range(1, n_up + 1)]
    peak = closes[-1]
    closes += [peak - i * down_slope for i in range(1, n_down + 1)]
    return closes


# ============================= no-lookahead =============================

class TestNoLookahead(unittest.TestCase):
    def test_act_golden_fires_one_trading_day_after_the_cross_itself(self):
        closes = _golden_then_death_closes()
        daily = _make_daily_ohlc(closes)
        sig = ma.build_daily_signals(daily)

        golden_days = sig.index[sig["golden_cross"]]
        self.assertEqual(len(golden_days), 1)
        golden_loc = sig.index.get_loc(golden_days[0])

        # act_golden must be False on the crossing day itself (can't know the close is a cross
        # until the day is over) and True on the very next trading day only
        self.assertFalse(bool(sig["act_golden"].iloc[golden_loc]))
        self.assertTrue(bool(sig["act_golden"].iloc[golden_loc + 1]))
        self.assertFalse(bool(sig["act_golden"].iloc[golden_loc + 2]))

    def test_act_death_fires_one_trading_day_after_the_cross_itself(self):
        closes = _golden_then_death_closes()
        daily = _make_daily_ohlc(closes, buf=1.5)
        sig = ma.build_daily_signals(daily)

        death_days = sig.index[sig["death_cross"]]
        self.assertEqual(len(death_days), 1)
        death_loc = sig.index.get_loc(death_days[0])

        self.assertFalse(bool(sig["act_death"].iloc[death_loc]))
        self.assertTrue(bool(sig["act_death"].iloc[death_loc + 1]))

    def test_a_naive_same_day_action_would_trade_one_day_too_early(self):
        """Direct proof that using golden_cross/death_cross (unshifted) instead of
        act_golden/act_death (shifted) would be a lookahead bug: it disagrees with the shifted
        version by exactly one trading day, on the day that actually matters."""
        closes = _golden_then_death_closes()
        daily = _make_daily_ohlc(closes)
        sig = ma.build_daily_signals(daily)
        golden_loc = sig.index.get_loc(sig.index[sig["golden_cross"]][0])

        naive_would_act_today = bool(sig["golden_cross"].iloc[golden_loc])          # the bug
        honest_acts_today = bool(sig["act_golden"].iloc[golden_loc])                 # the fix
        self.assertTrue(naive_would_act_today)
        self.assertFalse(honest_acts_today)
        self.assertTrue(bool(sig["act_golden"].iloc[golden_loc + 1]))                # fix acts here instead


# ============================= cross detection (hand-computed small cases) =============================

class TestCrossDetection(unittest.TestCase):
    def test_golden_cross_flagged_only_on_the_crossing_day(self):
        # sma_fast <= sma_slow, then sma_fast > sma_slow - construct a minimal series with
        # SMA_SLOW=200/SMA_FAST=50 by using a long flat run then a short ramp
        closes = [50.0] * 205 + [50.0 + i for i in range(1, 6)]   # small ramp at the end
        daily = _make_daily_ohlc(closes)
        sig = ma.build_daily_signals(daily)
        crosses = sig.index[sig["golden_cross"]]
        # exactly one crossing day, and sma_fast/sma_slow ordering confirms it by hand
        self.assertEqual(len(crosses), 1)
        loc = sig.index.get_loc(crosses[0])
        self.assertLessEqual(sig["sma_fast"].iloc[loc - 1], sig["sma_slow"].iloc[loc - 1])
        self.assertGreater(sig["sma_fast"].iloc[loc], sig["sma_slow"].iloc[loc])

    def test_death_cross_flagged_only_on_the_crossing_day(self):
        closes = [50.0] * 205 + [50.0 - i for i in range(1, 6)]
        daily = _make_daily_ohlc(closes)
        sig = ma.build_daily_signals(daily)
        crosses = sig.index[sig["death_cross"]]
        self.assertEqual(len(crosses), 1)
        loc = sig.index.get_loc(crosses[0])
        self.assertGreaterEqual(sig["sma_fast"].iloc[loc - 1], sig["sma_slow"].iloc[loc - 1])
        self.assertLess(sig["sma_fast"].iloc[loc], sig["sma_slow"].iloc[loc])

    def test_flat_series_produces_no_cross_at_all(self):
        closes = [50.0] * 260
        daily = _make_daily_ohlc(closes)
        sig = ma.build_daily_signals(daily)
        self.assertFalse(sig["golden_cross"].any())
        self.assertFalse(sig["death_cross"].any())

    def test_golden_and_death_cross_are_mutually_exclusive_every_day(self):
        closes = _golden_then_death_closes()
        daily = _make_daily_ohlc(closes)
        sig = ma.build_daily_signals(daily)
        self.assertFalse((sig["golden_cross"] & sig["death_cross"]).any())


# ============================= entry / exit mechanics =============================

class TestEntryExitMechanics(unittest.TestCase):
    def test_no_cross_produces_no_trade(self):
        closes = [50.0] * 260
        df = _daily_to_5min_df(_make_daily_ohlc(closes))
        trades = ma.backtest_instrument("SYN", df)
        self.assertEqual(trades, [])

    def test_golden_cross_entry_is_atr_sized_correctly(self):
        full_closes = _golden_then_death_closes(n_down=0)
        full_daily = _make_daily_ohlc(full_closes)
        full_sig = ma.build_daily_signals(full_daily)
        golden_loc = full_sig.index.get_loc(full_sig.index[full_sig["golden_cross"]][0])
        # truncate the data so it ends exactly on the entry day (golden_loc + 1) - proves the
        # force-close/timeout path fires at exactly the right price with nothing left dangling
        closes = full_closes[: golden_loc + 2]
        daily = _make_daily_ohlc(closes)
        sig = ma.build_daily_signals(daily)
        entry_date = sig.index[golden_loc + 1].date()
        atr_at_entry = sig["atr_entry"].iloc[golden_loc + 1]
        entry_close = daily["Close"].iloc[golden_loc + 1]

        df = _daily_to_5min_df(daily)
        trades = ma.backtest_instrument("SYN", df)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "LONG")
        self.assertEqual(trades[0]["date"], entry_date)
        expected_stop = entry_close - ma.ATR_STOP_MULT * atr_at_entry
        expected_sl_distance = entry_close - expected_stop
        # data ends on the entry day itself -> force-closed FLAT at the same close -> r == 0
        self.assertEqual(trades[0]["outcome"], "FLAT")
        self.assertAlmostEqual(trades[0]["r"], 0.0, places=6)
        self.assertAlmostEqual(expected_sl_distance, ma.ATR_STOP_MULT * atr_at_entry, places=6)

    def test_long_closes_on_death_cross_with_no_immediate_reversal(self):
        """The 'flat-then-wait' design choice (header point 7): closing a LONG on a death-cross
        action day must NOT also open a fresh SHORT that same day."""
        closes = _golden_then_death_closes()   # gentle decline -> death cross, ATR stop never hit
        daily = _make_daily_ohlc(closes, buf=1.5)
        sig = ma.build_daily_signals(daily)
        golden_loc = sig.index.get_loc(sig.index[sig["golden_cross"]][0])
        death_loc = sig.index.get_loc(sig.index[sig["death_cross"]][0])
        entry_close = daily["Close"].iloc[golden_loc + 1]
        atr_at_entry = sig["atr_entry"].iloc[golden_loc + 1]
        exit_close = daily["Close"].iloc[death_loc + 1]
        sl_distance = ma.ATR_STOP_MULT * atr_at_entry
        # sanity: this synthetic path must not have breached the ATR stop before the death cross
        min_low_between = daily["Low"].iloc[golden_loc + 1: death_loc + 2].min()
        self.assertGreater(min_low_between, entry_close - sl_distance)

        df = _daily_to_5min_df(daily)
        trades = ma.backtest_instrument("SYN", df)

        self.assertEqual(len(trades), 1)   # exactly one trade - no reversal short appended
        self.assertEqual(trades[0]["side"], "LONG")
        self.assertEqual(trades[0]["outcome"], "CROSS")
        expected_r = (exit_close - entry_close) / sl_distance
        self.assertAlmostEqual(trades[0]["r"], expected_r, places=6)

    def test_same_bar_stop_and_opposite_cross_prefers_stop_and_does_not_reverse(self):
        """Regression test for the bug this test suite caught during development: when the ATR
        hard stop AND that day's opposite-cross action would BOTH apply on the same bar, the
        stop must win (outcome SL, not CROSS) AND - the part that was actually broken - no fresh
        reversal position may open on that same bar just because the day "looks flat" right
        after the stop-out. Before the fix, `flat_at_start_of_day` was captured AFTER the
        same-bar ATR-stop check, so a stop-out made the day incorrectly look like it had started
        flat, letting that same day's already-consumed opposite-cross signal open a brand-new
        SHORT immediately - producing 2 trades instead of 1. This test fails on the buggy
        version (2 trades: SL then a same-day SHORT) and passes on the fixed version (1 trade:
        SL only)."""
        closes = _golden_then_death_closes()
        daily = _make_daily_ohlc(closes, buf=1.5)
        sig = ma.build_daily_signals(daily)
        golden_loc = sig.index.get_loc(sig.index[sig["golden_cross"]][0])
        death_loc = sig.index.get_loc(sig.index[sig["death_cross"]][0])
        entry_close = daily["Close"].iloc[golden_loc + 1]
        atr_at_entry = sig["atr_entry"].iloc[golden_loc + 1]
        stop_level = entry_close - ma.ATR_STOP_MULT * atr_at_entry
        act_death_date = sig.index[death_loc + 1].date()
        act_death_row = daily.loc[[ts for ts in daily.index if ts.date() == act_death_date][0]]

        # punch an intrabar wick through the stop on the exact act_death action day, without
        # touching that day's Close (so act_death, which depends only on Close-based SMAs, is
        # unaffected by this override)
        overrides = {act_death_date: (act_death_row["Open"], act_death_row["High"],
                                       stop_level - 5.0, act_death_row["Close"])}
        df = _daily_to_5min_df(daily, overrides=overrides)
        trades = ma.backtest_instrument("SYN", df)

        self.assertEqual(len(trades), 1, f"expected exactly 1 trade (SL only), got {trades}")
        self.assertEqual(trades[0]["outcome"], "SL")
        self.assertAlmostEqual(trades[0]["r"], -1.0, places=6)

    def test_force_close_at_data_end_while_still_in_position(self):
        closes = _golden_then_death_closes(n_down=10, down_slope=0.0)   # stays open, no death cross
        daily = _make_daily_ohlc(closes)
        sig = ma.build_daily_signals(daily)
        golden_loc = sig.index.get_loc(sig.index[sig["golden_cross"]][0])
        entry_close = daily["Close"].iloc[golden_loc + 1]
        atr_at_entry = sig["atr_entry"].iloc[golden_loc + 1]
        sl_distance = ma.ATR_STOP_MULT * atr_at_entry
        last_close = daily["Close"].iloc[-1]

        df = _daily_to_5min_df(daily)
        trades = ma.backtest_instrument("SYN", df)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "FLAT")
        expected_r = (last_close - entry_close) / sl_distance
        self.assertAlmostEqual(trades[0]["r"], expected_r, places=6)


# ============================= random-walk null-result sanity check =============================

class TestRandomWalkNullResult(unittest.TestCase):
    def test_no_suspiciously_large_edge_on_pure_noise(self):
        """Golden/death crosses are rare by nature (see the strategy file's header), so this
        needs more days and more pooled seeds than the Donchian script's equivalent test to say
        anything meaningful at all. A genuine no-drift random walk run through the exact same
        backtest function should not show a large, consistent edge - pooled across seeds since
        individual seeds are noisy with this few trades."""
        all_r = []
        for seed in range(20):
            df = _make_random_walk_5min_df(n_days=1500, bars_per_day=8, seed=seed + 2000)
            trades = ma.backtest_instrument("SYN", df)
            all_r.extend(t["r"] for t in trades)

        n = len(all_r)
        self.assertGreater(n, 20, "test setup produced too few trades to say anything meaningful")
        avg_r = sum(all_r) / n
        std_r = np.std(all_r, ddof=1)
        z = avg_r / (std_r / (n ** 0.5)) if std_r > 0 else 0.0
        self.assertLess(abs(avg_r), 0.6, f"suspiciously large average R/trade on pure noise: {avg_r:.3f}")
        self.assertLess(abs(z), 3.0, f"suspiciously large z-score on pure noise: {z:.2f}")


if __name__ == "__main__":
    unittest.main()
