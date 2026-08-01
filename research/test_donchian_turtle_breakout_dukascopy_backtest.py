# Unit tests + one synthetic end-to-end smoke run for
# donchian_turtle_breakout_dukascopy_backtest.py.
#
# Covers, per this project's rigor conventions:
#   1. NO-LOOKAHEAD proof: a synthetic day with an extreme spike shows the
#      channel/ATR values attached to that SAME day never include the
#      spike (only the day after does) - a direct check on the exact
#      off-by-one bug this kind of rolling-window logic is prone to.
#   2. Channel computation with a hand-computed 20/10-day window value.
#   3. Breakout entry (LONG and SHORT), initial ATR stop sizing, trailing
#      exit, stop-vs-trailing precedence (same-bar), one-trade-at-a-time,
#      and force-close/timeout accounting - several small, targeted cases,
#      each isolating one behavior.
#   4. A synthetic multi-year random-walk null test (several seeds pooled)
#      confirming no suspiciously large edge shows up on pure noise - the
#      granularity/false-positive sanity check this project's other
#      scripts also run (see the header of ict_po3_forex_dukascopy_backtest.py
#      for why single-step synthetic OHLC construction can create fake
#      edges, and this file's own header for why the risk is smaller but
#      not assumed away for a swing system).
#
# Run with:  python -m pytest research/test_donchian_turtle_breakout_dukascopy_backtest.py -v
# or:        python research/test_donchian_turtle_breakout_dukascopy_backtest.py

import importlib.util
import os
import unittest

import numpy as np
import pandas as pd

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "donchian_turtle_breakout_dukascopy_backtest.py")
_spec = importlib.util.spec_from_file_location("donchian_turtle_breakout_dukascopy_backtest", _MODULE_PATH)
donchian = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(donchian)


# ============================= synthetic data helpers =============================

def _make_daily_df(n_days, start="2016-01-04"):
    return pd.date_range(start=start, periods=n_days, freq="B", tz="America/New_York")


def _build_5min_df(day_specs):
    """day_specs: list of (date, [(o,h,l,c), ...]) - one or more 5-min bars per day, 5 minutes
    apart starting 09:30 NY time. The day's aggregate daily OHLC (via resample_daily) is exactly
    what a plain daily-bar feed for that day would produce."""
    rows, idx = [], []
    for date_, bars in day_specs:
        for k, (o, h, l, c) in enumerate(bars):
            ts = pd.Timestamp(date_.year, date_.month, date_.day, 9, 30, tz="America/New_York") + \
                 pd.Timedelta(minutes=5 * k)
            idx.append(ts)
            rows.append((o, h, l, c))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=pd.DatetimeIndex(idx))


def _make_random_walk_5min_df(n_days, bars_per_day, seed, start_price=1.1000, intrabar_vol=0.0006):
    """A genuine no-drift random walk with intrabar noise (not a single-step synthetic OHLC
    construction) - used only for the null-result sanity check below."""
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


# ============================= no-lookahead + channel math =============================

class TestNoLookahead(unittest.TestCase):
    def test_channel_high_excludes_current_day_own_spike(self):
        n = 25
        idx = _make_daily_df(n)
        highs = [1.1002 + 0.00001 * i for i in range(n)]
        highs[20] = 5.0   # a huge one-day spike, day index 20
        lows = [h - 0.0004 for h in highs]
        opens = list(lows)
        closes = [(o + h) / 2 for o, h in zip(opens, highs)]
        daily = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)

        sig = donchian.build_daily_signals(daily)
        # channel_high attached to day 20 itself must be built ONLY from days 0..19 - the
        # spike sitting on day 20 must not leak into its own channel value
        self.assertLess(sig["channel_high"].iloc[20], 2.0)
        # channel_high attached to day 21 (the day after) MUST now reflect the day-20 spike,
        # since day 20 is history as of day 21
        self.assertEqual(sig["channel_high"].iloc[21], 5.0)

    def test_channel_low_excludes_current_day_own_spike(self):
        n = 25
        idx = _make_daily_df(n)
        lows = [1.0998 - 0.00001 * i for i in range(n)]
        lows[20] = 0.5   # a huge one-day downside spike
        highs = [l + 0.0004 for l in lows]
        opens = list(highs)
        closes = [(o + l) / 2 for o, l in zip(opens, lows)]
        daily = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)

        sig = donchian.build_daily_signals(daily)
        self.assertGreater(sig["channel_low"].iloc[20], 0.9)
        self.assertEqual(sig["channel_low"].iloc[21], 0.5)

    def test_atr_entry_excludes_current_day_own_true_range(self):
        n = 25
        idx = _make_daily_df(n)
        closes = [1.1000] * n
        highs = [1.1005] * n
        lows = [1.0995] * n
        highs[20] = 1.30   # a huge true-range day
        lows[20] = 0.90
        opens = list(closes)
        daily = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)

        sig = donchian.build_daily_signals(daily)
        atr_on_spike_day = sig["atr_entry"].iloc[20]     # built only from days < 20 - must be tiny
        atr_day_after = sig["atr_entry"].iloc[21]         # now includes day 20's huge TR - must jump
        self.assertLess(atr_on_spike_day, 0.002)
        self.assertGreater(atr_day_after, atr_on_spike_day * 5)

    def test_exit_channel_uses_shorter_prior_window_and_excludes_current_day(self):
        n = 15
        idx = _make_daily_df(n)
        lows = [1.0990 + 0.0001 * i for i in range(n)]   # rising lows
        lows[12] = 0.8   # spike low on day 12
        highs = [l + 0.0005 for l in lows]
        opens = list(highs)
        closes = [(o + l) / 2 for o, l in zip(opens, lows)]
        daily = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx)

        sig = donchian.build_daily_signals(daily)
        self.assertGreater(sig["exit_low"].iloc[12], 0.9)     # day 12's own spike not in its own exit_low
        self.assertEqual(sig["exit_low"].iloc[13], 0.8)       # day 13 sees it


# ============================= entry / exit mechanics =============================

class TestBreakoutMechanics(unittest.TestCase):
    """All cases share one base setup: 25 warmup days flat at (O=1.1000, H=1.1002, L=1.0998,
    C=1.1000). That warmup range makes ATR(14) converge to exactly 0.0004 (true range = H-L
    every day since price never gaps), so entry sizing (stop = entry -/+ 2*ATR) is exactly
    predictable by hand rather than merely plausible."""

    WARMUP_BAR = (1.1000, 1.1002, 1.0998, 1.1000)
    WARMUP_ATR = 0.0004   # hand-derived: constant TR = H-L = 0.0004 every warmup day

    def _warmup_specs(self, n=25, start="2016-01-04"):
        days = pd.bdate_range(start=start, periods=n + 20)   # plenty of trailing business days
        return [(d, [self.WARMUP_BAR]) for d in days[:n]], days[n:]

    def test_long_breakout_entry_sizing_and_flat_force_close(self):
        warmup, remaining_days = self._warmup_specs()
        breakout_day = remaining_days[0]
        next_day = remaining_days[1]
        specs = list(warmup)
        specs.append((breakout_day, [self.WARMUP_BAR, (1.1000, 1.1050, 1.0995, 1.1050)]))
        specs.append((next_day, [(1.1050, 1.1060, 1.1045, 1.1055)]))   # doesn't hit stop or trail

        df = _build_5min_df(specs)
        trades = donchian.backtest_instrument("SYN", df)

        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t["side"], "LONG")
        self.assertEqual(t["outcome"], "FLAT")
        entry = 1.1050
        stop = entry - 2 * self.WARMUP_ATR
        sl_distance = entry - stop
        expected_r = (1.1055 - entry) / sl_distance
        self.assertAlmostEqual(t["r"], expected_r, places=6)
        self.assertAlmostEqual(sl_distance, 0.0008, places=6)

    def test_short_breakout_entry_sizing(self):
        warmup, remaining_days = self._warmup_specs()
        breakout_day = remaining_days[0]
        specs = list(warmup)
        # close breaks BELOW channel_low (0.0998) -> SHORT
        specs.append((breakout_day, [self.WARMUP_BAR, (1.1000, 1.1002, 1.0950, 1.0950)]))

        df = _build_5min_df(specs)
        trades = donchian.backtest_instrument("SYN", df)

        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t["side"], "SHORT")
        entry = 1.0950
        stop = entry + 2 * self.WARMUP_ATR
        sl_distance = stop - entry
        # data ends on the entry bar itself -> force-closed FLAT at the same price -> r == 0
        self.assertEqual(t["outcome"], "FLAT")
        self.assertAlmostEqual(t["r"], 0.0, places=6)
        self.assertAlmostEqual(sl_distance, 0.0008, places=6)

    def test_initial_stop_takes_precedence_over_trailing_exit_same_bar(self):
        warmup, remaining_days = self._warmup_specs()
        breakout_day = remaining_days[0]
        pullback_day = remaining_days[1]
        specs = list(warmup)
        specs.append((breakout_day, [self.WARMUP_BAR, (1.1000, 1.1050, 1.0995, 1.1050)]))
        # entry=1.1050, stop=1.1042 (1.1050 - 2*0.0004). exit_low after this day is 1.0995 (the
        # prior 10 days' low, all still from warmup/breakout). A bar whose low dives to 1.0900
        # is BELOW BOTH the stop (1.1042) and the trailing exit (1.0995) - the stop must win.
        specs.append((pullback_day, [(1.1050, 1.1055, 0.9000, 1.0000)]))

        df = _build_5min_df(specs)
        trades = donchian.backtest_instrument("SYN", df)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "SL")
        self.assertAlmostEqual(trades[0]["r"], -1.0, places=6)

    def test_trailing_exit_fires_when_stop_not_touched(self):
        # Build a breakout then a steady uptrend so the trailing 10-day low channel rises
        # above the (fixed) initial ATR stop - proving TRAIL can fire on its own, distinct
        # from the ATR stop, once enough days have passed for the channel to catch up.
        warmup, remaining_days = self._warmup_specs(n=25)
        breakout_day = remaining_days[0]
        specs = list(warmup)
        specs.append((breakout_day, [self.WARMUP_BAR, (1.1000, 1.1050, 1.0995, 1.1050)]))
        uptrend_days = remaining_days[1:16]
        for i, d in enumerate(uptrend_days):
            lo = 1.1046 + i * 0.0010
            hi = lo + 0.0010
            specs.append((d, [(lo, hi, lo, hi)]))

        # find the first day where the trailing exit_low has climbed above the fixed stop
        # (1.1042), using the lower-level signal builder directly (not the full backtest loop)
        probe_df = _build_5min_df(specs)
        daily_sig = donchian.build_daily_signals(donchian.resample_daily(probe_df))
        stop_level = 1.1050 - 2 * self.WARMUP_ATR
        candidates = daily_sig[daily_sig["exit_low"] > stop_level]
        self.assertGreater(len(candidates), 0, "test setup didn't produce a day with exit_low above the stop")
        pullback_date = candidates.index[0].date()
        exit_low_that_day = candidates.iloc[0]["exit_low"]
        pullback_low = (stop_level + exit_low_that_day) / 2   # strictly between stop and exit_low

        # cut the synthetic feed to end right after replacing that day's bar with a pullback
        cut_specs = [(d, bars) for d, bars in specs if d.date() < pullback_date]
        cut_specs.append((pullback_date, [(exit_low_that_day + 0.001, exit_low_that_day + 0.002,
                                            pullback_low, exit_low_that_day + 0.0005)]))
        df = _build_5min_df(cut_specs)
        trades = donchian.backtest_instrument("SYN", df)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "TRAIL")
        entry = 1.1050
        sl_distance = entry - stop_level
        expected_r = (exit_low_that_day - entry) / sl_distance
        self.assertAlmostEqual(trades[0]["r"], expected_r, places=6)

    def test_one_trade_at_a_time_no_pyramiding_while_in_position(self):
        # while already LONG, a further bar whose close is still (or again) above channel_high
        # must NOT open a second concurrent position - the eventual single close must reflect
        # the ORIGINAL entry/stop, not a re-based one.
        warmup, remaining_days = self._warmup_specs()
        breakout_day = remaining_days[0]
        still_high_day = remaining_days[1]
        specs = list(warmup)
        specs.append((breakout_day, [self.WARMUP_BAR, (1.1000, 1.1050, 1.0995, 1.1050)]))
        # this day's close is even higher still - if pyramiding were (bugged) into happening,
        # a second trade would appear here
        specs.append((still_high_day, [(1.1050, 1.1080, 1.1048, 1.1080)]))

        df = _build_5min_df(specs)
        trades = donchian.backtest_instrument("SYN", df)

        self.assertEqual(len(trades), 1)   # exactly one trade, not two
        entry = 1.1050
        stop = entry - 2 * self.WARMUP_ATR
        sl_distance = entry - stop
        expected_r = (1.1080 - entry) / sl_distance   # force-closed FLAT at the last bar's close
        self.assertAlmostEqual(trades[0]["r"], expected_r, places=6)

    def test_no_channel_break_produces_no_trade(self):
        warmup, remaining_days = self._warmup_specs()
        specs = list(warmup)
        specs.append((remaining_days[0], [self.WARMUP_BAR, self.WARMUP_BAR]))
        df = _build_5min_df(specs)
        trades = donchian.backtest_instrument("SYN", df)
        self.assertEqual(trades, [])


# ============================= random-walk null-result sanity check =============================

class TestRandomWalkNullResult(unittest.TestCase):
    def test_no_suspiciously_large_edge_on_pure_noise(self):
        """A genuine no-drift random walk (with real intrabar noise, not a single flat bar per
        day - see this file's helper) run through the exact same backtest function should not
        show a large, consistent edge. Pooled across several independent seeds (per-seed
        results are individually noisy with this few trades - that's expected, not a failure by
        itself; it's the pooled figure that should look like noise)."""
        all_r = []
        for seed in range(10):
            df = _make_random_walk_5min_df(n_days=500, bars_per_day=24, seed=seed)
            trades = donchian.backtest_instrument("SYN", df)
            all_r.extend(t["r"] for t in trades)

        n = len(all_r)
        self.assertGreater(n, 20, "test setup produced too few trades to say anything meaningful")
        avg_r = sum(all_r) / n
        z = avg_r / (1 / (n ** 0.5))
        # generous bounds - this is a coarse "no smoking gun" check, not a rigorous statistical
        # proof of zero edge (small-sample trend-following results on random walks are noisy by
        # nature); a genuine implementation bug (e.g. a lookahead leak) would be expected to
        # blow well past these.
        self.assertLess(abs(avg_r), 0.6, f"suspiciously large average R/trade on pure noise: {avg_r:.3f}")
        self.assertLess(abs(z), 3.0, f"suspiciously large z-score on pure noise: {z:.2f}")


if __name__ == "__main__":
    unittest.main()
