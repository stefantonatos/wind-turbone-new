# Unit tests for bdm_orb_reversal_indices_dukascopy_backtest.py.
#
# Covers, per this project's rigor conventions:
#   1. The opening range is built from ORB-window bars ONLY - pre-open and post-ORB bars must not
#      leak into it.
#   2. Continuation entry on the first close outside the range: entry price, both stop modes,
#      target placement, and TP / SL outcomes at exactly +REWARD_RISK / -1R.
#   3. Reversal on a close back inside the range: the continuation is closed at that bar's close
#      for a PARTIAL R (not -1R), and the reversal's stop sits at the failed breakout's extreme
#      including the current bar.
#   4. THE ORDERING INVARIANT (the most important test in this file). On a bar that both stops the
#      continuation out intrabar AND closes back inside the range, the default configuration must
#      produce NO reversal, because TradingView's broker emulator fills the protective stop before
#      the script's close-based logic reads the position. The same bar WITH
#      ALLOW_REVERSAL_AFTER_CLOSE enabled must produce one. Getting this backwards silently turns
#      this into a different, busier strategy, so both directions are pinned.
#   5. MIN_STOP_PCT: a stop thinner than the floor is SKIPPED rather than scored (a near-zero stop
#      would otherwise mint a fake triple-digit R winner), and the reversal stays available for a
#      later bar - matching how the Pine's mintick guard behaves.
#   6. End-of-session flatten produces a fractional-R FLAT on the last in-window bar.
#   7. At most one continuation and one reversal per session, and no entry outside the trade window.
#   8. ENABLE_CONTINUATION off still detects the breakout and allows a reversal-only trade.
#   9. A LONG/SHORT mirror check - mirroring price around a pivot and swapping high/low turns a
#      hand-verified LONG scenario into an equally-verified SHORT one without re-deriving the
#      arithmetic by hand a second time (same technique as the VWAP ORB and London 3AM tests).
#  10. The webapp trade contract: every trade carries `date` and `stop_pct`, both of which were
#      real silent-failure bugs in this repo when omitted.
#  11. A smoke end-to-end run over synthetic multi-day data.
#
# Run with:  python -m pytest research/test_bdm_orb_reversal_indices_dukascopy_backtest.py -v

import contextlib
import datetime
import importlib.util
import math
import os
import unittest

import pandas as pd

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "bdm_orb_reversal_indices_dukascopy_backtest.py")
_spec = importlib.util.spec_from_file_location("bdm_orb_reversal_indices_dukascopy_backtest", _MODULE_PATH)
bdm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bdm)

TZ = "America/New_York"
SESSION_OPEN = pd.Timestamp("09:30").time()


@contextlib.contextmanager
def config(**overrides):
    """Temporarily set module-level constants, always restoring them - the same
    override-module-globals mechanism the webapp registry uses at runtime, so anything these tests
    pin is pinned for the app too."""
    previous = {name: getattr(bdm, name) for name in overrides}
    for name, value in overrides.items():
        setattr(bdm, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(bdm, name, value)


def make_day(bars, date="2024-03-04"):
    """`bars` maps "HH:MM" -> (high, low, close). Bars are 5 minutes apart starting at 09:00 so a
    pre-open bar can be included; any slot not listed is filled with a flat bar at 100.0."""
    start = pd.Timestamp(f"{date} 09:00", tz=TZ)
    index, rows = [], []
    for step in range(int(8 * 60 / 5)):          # 09:00 -> 17:00
        ts = start + pd.Timedelta(minutes=5 * step)
        key = ts.strftime("%H:%M")
        high, low, close = bars.get(key, (100.0, 100.0, 100.0))
        index.append(ts)
        rows.append({"open": close, "high": high, "low": low, "close": close})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index))


def run(bars, date="2024-03-04", **overrides):
    overrides.setdefault("SESSION_MINUTES", 30)   # window = 09:45, 09:50, 09:55 unless overridden
    with config(**overrides):
        return bdm.backtest_index("TEST", None, TZ, SESSION_OPEN, df=make_day(bars, date=date))


# A flat opening range of [99.0, 101.0] -> midpoint 100.0. Every scenario below builds on it, so
# the ORB numbers never have to be re-derived: mid = 100.0, and a continuation long entering at
# 101.5 risks exactly 1.5 with a 1R target at 103.0.
ORB = {"09:30": (101.0, 99.0, 100.0), "09:35": (100.5, 99.5, 100.0), "09:40": (100.8, 99.2, 100.0)}


def orb_with(extra):
    merged = dict(ORB)
    merged.update(extra)
    return merged


class TestOpeningRange(unittest.TestCase):
    def test_pre_open_and_post_orb_bars_do_not_widen_the_range(self):
        # A wild 09:25 bar and a wild 09:45 bar both sit outside the 0930-0945 ORB window. If
        # either leaked in, orb_high would be 120 and the 101.5 close would not be a breakout.
        trades = run(orb_with({"09:25": (120.0, 80.0, 100.0),
                                "09:45": (120.0, 80.0, 101.5),
                                "09:50": (103.2, 101.4, 103.1)}))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "TP")
        # entry 101.5, stop at the midpoint 100.0 -> risk 1.5, unaffected by the 80/120 bars
        self.assertAlmostEqual(trades[0]["sl_distance"], 1.5)

    def test_no_breakout_means_no_trade(self):
        self.assertEqual(run(orb_with({"09:45": (100.9, 99.1, 100.2),
                                        "09:50": (100.9, 99.1, 99.8)})), [])


class TestContinuation(unittest.TestCase):
    def test_long_continuation_take_profit(self):
        trades = run(orb_with({"09:45": (101.6, 100.9, 101.5),
                                "09:50": (103.2, 101.4, 103.1)}))
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual((t["side"], t["outcome"], t["trade_type"]), ("LONG", "TP", "continuation"))
        self.assertAlmostEqual(t["entry"], 101.5)
        self.assertAlmostEqual(t["sl_distance"], 1.5)          # 101.5 - 100.0 (midpoint)
        self.assertAlmostEqual(t["r"], 1.0)
        self.assertAlmostEqual(t["stop_pct"], 1.5 / 101.5)

    def test_long_continuation_stop_loss(self):
        trades = run(orb_with({"09:45": (101.6, 100.9, 101.5),
                                "09:50": (101.7, 99.9, 100.1)}),
                     ALLOW_REVERSAL_AFTER_CLOSE=0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "SL")
        self.assertAlmostEqual(trades[0]["r"], -1.0)

    def test_short_continuation_take_profit(self):
        trades = run(orb_with({"09:45": (98.9, 98.4, 98.5),      # closes below orb_low 99.0
                                "09:50": (98.6, 96.8, 96.9)}))
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual((t["side"], t["outcome"]), ("SHORT", "TP"))
        self.assertAlmostEqual(t["sl_distance"], 1.5)            # 100.0 - 98.5
        self.assertAlmostEqual(t["r"], 1.0)

    def test_opposite_side_stop_mode_uses_the_far_edge_of_the_range(self):
        trades = run(orb_with({"09:45": (101.6, 100.9, 101.5),
                                "09:50": (104.7, 101.4, 104.6)}),
                     CONTINUATION_STOP_AT_MID=0)
        self.assertEqual(len(trades), 1)
        # stop at orb_low 99.0 -> risk 2.5, target 104.0, reached by the 104.7 high
        self.assertAlmostEqual(trades[0]["sl_distance"], 2.5)
        self.assertEqual(trades[0]["outcome"], "TP")

    def test_reward_risk_multiple_scales_the_target(self):
        # At 2R the same 103.2 high is no longer enough to reach the 104.5 target, so the trade is
        # still open at the session end and flattens instead of taking profit.
        trades = run(orb_with({"09:45": (101.6, 100.9, 101.5),
                                "09:50": (103.2, 101.4, 103.1),
                                "09:55": (103.2, 102.9, 103.0)}),
                     REWARD_RISK=2.0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "FLAT")
        self.assertAlmostEqual(trades[0]["r"], (103.0 - 101.5) / 1.5)


class TestReversal(unittest.TestCase):
    def test_close_back_inside_closes_the_continuation_and_flips(self):
        trades = run(orb_with({"09:45": (101.6, 100.9, 101.5),
                                "09:50": (102.0, 100.4, 100.5),   # back inside, continuation alive
                                "09:55": (100.6, 98.9, 99.0)}))
        self.assertEqual(len(trades), 2)
        cont, rev = trades
        # The continuation is closed at the bar's CLOSE for a partial loss, not at its stop.
        self.assertEqual((cont["outcome"], cont["trade_type"]), ("REV_EXIT", "continuation"))
        self.assertAlmostEqual(cont["r"], (100.5 - 101.5) / 1.5)
        self.assertGreater(cont["r"], -1.0)
        # Reversal: short from 100.5, stop at the failed breakout's extreme (102.0, which includes
        # this bar's own high), risk 1.5, 1R target at 99.0 - hit by the 98.9 low.
        self.assertEqual((rev["side"], rev["outcome"], rev["trade_type"]), ("SHORT", "TP", "reversal"))
        self.assertAlmostEqual(rev["entry"], 100.5)
        self.assertAlmostEqual(rev["sl_distance"], 1.5)
        self.assertAlmostEqual(rev["r"], 1.0)

    def test_reversal_stop_uses_the_highest_high_since_the_breakout(self):
        # 102.9 stays under the continuation's 103.0 target, so the trade is still open when price
        # comes back - the extreme is carried forward across bars rather than read off the
        # reversal bar alone.
        trades = run(orb_with({"09:45": (101.6, 100.9, 101.5),
                                "09:50": (102.9, 101.0, 101.2),   # runs up but stays outside
                                "09:55": (101.3, 100.4, 100.5)}), # now closes back inside
                     SESSION_MINUTES=35)
        rev = next(t for t in trades if t["trade_type"] == "reversal")
        self.assertAlmostEqual(rev["sl_distance"], 102.9 - 100.5)

    def test_reversal_only_mode_when_continuation_is_disabled(self):
        trades = run(orb_with({"09:45": (101.6, 100.9, 101.5),
                                "09:50": (102.0, 100.4, 100.5),
                                "09:55": (100.6, 98.9, 99.0)}),
                     ENABLE_CONTINUATION=0)
        self.assertEqual(len(trades), 1)
        self.assertEqual((trades[0]["trade_type"], trades[0]["side"]), ("reversal", "SHORT"))

    def test_reversals_disabled_leaves_the_continuation_to_run(self):
        trades = run(orb_with({"09:45": (101.6, 100.9, 101.5),
                                "09:50": (102.0, 100.4, 100.5),
                                "09:55": (100.6, 98.9, 99.0)}),
                     ENABLE_REVERSALS=0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["trade_type"], "continuation")
        # 09:55 dips to 98.9, through the 100.0 midpoint stop
        self.assertEqual(trades[0]["outcome"], "SL")


class TestBracketBeforeSignalOrdering(unittest.TestCase):
    """The single most consequential fidelity detail in the port - see the module header."""

    # One bar that BOTH breaks the 100.0 midpoint stop (low 99.5) and closes back inside the
    # range (close 100.2). Everything else is held fixed between the two configurations.
    SCENARIO = {"09:45": (101.6, 100.9, 101.5),
                "09:50": (101.7, 99.5, 100.2),
                "09:55": (100.3, 97.0, 97.1)}

    def test_default_mode_produces_no_reversal_when_the_stop_fired_first(self):
        trades = run(orb_with(self.SCENARIO), ALLOW_REVERSAL_AFTER_CLOSE=0)
        self.assertEqual(len(trades), 1)
        self.assertEqual((trades[0]["trade_type"], trades[0]["outcome"]), ("continuation", "SL"))

    def test_allow_after_close_produces_the_reversal_on_the_same_bar(self):
        trades = run(orb_with(self.SCENARIO), ALLOW_REVERSAL_AFTER_CLOSE=1)
        self.assertEqual(len(trades), 2)
        self.assertEqual((trades[0]["trade_type"], trades[0]["outcome"]), ("continuation", "SL"))
        rev = trades[1]
        self.assertEqual((rev["trade_type"], rev["side"]), ("reversal", "SHORT"))
        self.assertAlmostEqual(rev["entry"], 100.2)
        self.assertAlmostEqual(rev["sl_distance"], 101.7 - 100.2)   # highest high since breakout
        self.assertEqual(rev["outcome"], "TP")                       # 1R target 98.7, low 97.0

    def test_the_two_modes_differ_only_because_of_the_ordering(self):
        # Same bars, one flag - if the bracket were evaluated AFTER the close-based signal, both
        # configurations would return the same two trades and this assertion would fail.
        self.assertNotEqual(len(run(orb_with(self.SCENARIO), ALLOW_REVERSAL_AFTER_CLOSE=0)),
                            len(run(orb_with(self.SCENARIO), ALLOW_REVERSAL_AFTER_CLOSE=1)))


class TestMinStopFloor(unittest.TestCase):
    def test_hairline_reversal_stop_is_skipped_not_scored(self):
        # Closing at 100.999 with a failed-breakout extreme of 101.0 is a 0.001-wide stop: scored
        # naively it would produce R-multiples in the hundreds off a single bar.
        trades = run(orb_with({"09:45": (101.0, 100.9, 100.95),
                                "09:50": (101.0, 100.9, 100.999),
                                "09:55": (101.0, 100.9, 100.95)}),
                     ENABLE_CONTINUATION=0)
        self.assertEqual(trades, [])

    def test_reversal_stays_available_after_a_skipped_hairline_bar(self):
        # 09:45 breaks out by a hair (close 101.0001, high the same), so 09:50 closing at 100.9999
        # is a genuine close-back-inside whose stop distance is 0.0002 - under the floor, skipped.
        # The setup must survive that skip: 09:55 comes back inside with a real 1.0-wide stop.
        trades = run(orb_with({"09:45": (101.0001, 100.9, 101.0001),
                                "09:50": (101.0001, 100.9, 100.9999),   # hairline stop -> skipped
                                "09:55": (101.5, 100.4, 100.5)}),        # real reversal
                     SESSION_MINUTES=35)
        rev = [t for t in trades if t["trade_type"] == "reversal"]
        self.assertEqual(len(rev), 1)
        self.assertAlmostEqual(rev[0]["entry"], 100.5)
        self.assertAlmostEqual(rev[0]["sl_distance"], 101.5 - 100.5)

    def test_floor_is_expressed_as_a_percentage_of_price(self):
        # A 1.5-wide stop on a ~101 price is 1.48%; raising the floor above that rejects the same
        # trade that a lower floor accepts.
        bars = orb_with({"09:45": (101.6, 100.9, 101.5), "09:50": (103.2, 101.4, 103.1)})
        self.assertEqual(len(run(bars, MIN_STOP_PCT=0.02)), 1)
        self.assertEqual(run(bars, MIN_STOP_PCT=2.0), [])


class TestSessionEnd(unittest.TestCase):
    def test_open_position_flattens_at_the_last_in_window_bar(self):
        trades = run(orb_with({"09:45": (101.6, 100.9, 101.5),
                                "09:50": (102.2, 101.4, 102.0),
                                "09:55": (102.4, 101.8, 102.3)}))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["outcome"], "FLAT")
        self.assertAlmostEqual(trades[0]["r"], (102.3 - 101.5) / 1.5)

    def test_flatten_can_be_disabled(self):
        trades = run(orb_with({"09:45": (101.6, 100.9, 101.5),
                                "09:50": (102.2, 101.4, 102.0),
                                "09:55": (102.4, 101.8, 102.3)}),
                     CLOSE_AT_SESSION_END=0)
        self.assertEqual(trades, [])

    def test_no_entry_after_the_trade_window_closes(self):
        # The breakout bar is at 10:00, one bar past a 30-minute session.
        self.assertEqual(run(orb_with({"10:00": (101.6, 100.9, 101.5),
                                        "10:05": (103.2, 101.4, 103.1)})), [])


class TestOnePerSession(unittest.TestCase):
    def test_at_most_one_continuation_and_one_reversal(self):
        # After the reversal completes, price breaks out and returns repeatedly - none of it may
        # produce a third trade.
        bars = orb_with({"09:45": (101.6, 100.9, 101.5),
                          "09:50": (102.0, 100.4, 100.5),
                          "09:55": (100.6, 98.9, 99.0),
                          "10:00": (105.0, 98.0, 104.0),
                          "10:05": (104.5, 99.5, 100.0),
                          "10:10": (105.0, 98.0, 104.0),
                          "10:15": (104.5, 99.5, 100.0)})
        trades = run(bars, SESSION_MINUTES=60)
        self.assertEqual(len(trades), 2)
        self.assertEqual([t["trade_type"] for t in trades], ["continuation", "reversal"])


class TestLongShortMirror(unittest.TestCase):
    """Mirroring every price around a pivot and swapping high/low must turn a LONG result into the
    identical SHORT result. Any asymmetry in the entry, stop, target, or exit logic shows up here
    without the arithmetic having to be hand-derived twice."""

    PIVOT = 100.0

    def _mirror(self, bars):
        return {k: (2 * self.PIVOT - low, 2 * self.PIVOT - high, 2 * self.PIVOT - close)
                for k, (high, low, close) in bars.items()}

    def test_continuation_and_reversal_mirror(self):
        bars = orb_with({"09:45": (101.6, 100.9, 101.5),
                          "09:50": (102.0, 100.4, 100.5),
                          "09:55": (100.6, 98.9, 99.0)})
        long_side = run(bars)
        short_side = run(self._mirror(bars))
        self.assertEqual(len(long_side), len(short_side))
        for a, b in zip(long_side, short_side):
            self.assertEqual(a["trade_type"], b["trade_type"])
            self.assertEqual(a["outcome"], b["outcome"])
            self.assertNotEqual(a["side"], b["side"])
            self.assertAlmostEqual(a["r"], b["r"])
            self.assertAlmostEqual(a["sl_distance"], b["sl_distance"])


class TestWebappContract(unittest.TestCase):
    def test_every_trade_carries_date_and_stop_pct(self):
        trades = run(orb_with({"09:45": (101.6, 100.9, 101.5),
                                "09:50": (102.0, 100.4, 100.5),
                                "09:55": (100.6, 98.9, 99.0)}))
        self.assertTrue(trades)
        for t in trades:
            self.assertIsInstance(t["date"], datetime.date)
            self.assertGreater(t["stop_pct"], 0.0)
            self.assertAlmostEqual(t["stop_pct"], t["sl_distance"] / t["entry"])
            self.assertTrue(math.isfinite(t["r"]))
            self.assertIn(t["side"], ("LONG", "SHORT"))

    def test_trade_date_is_the_session_date(self):
        trades = run(orb_with({"09:45": (101.6, 100.9, 101.5),
                                "09:50": (103.2, 101.4, 103.1)}), date="2024-06-12")
        self.assertEqual(trades[0]["date"], datetime.date(2024, 6, 12))


class TestMultiDaySmoke(unittest.TestCase):
    def test_runs_across_several_sessions_independently(self):
        frames = []
        for day in ("2024-03-04", "2024-03-05", "2024-03-06"):
            frames.append(make_day(orb_with({"09:45": (101.6, 100.9, 101.5),
                                              "09:50": (103.2, 101.4, 103.1)}), date=day))
        df = pd.concat(frames)
        with config(SESSION_MINUTES=30):
            trades = bdm.backtest_index("TEST", None, TZ, SESSION_OPEN, df=df)
        self.assertEqual(len(trades), 3)
        self.assertEqual(sorted({t["date"] for t in trades}),
                         [datetime.date(2024, 3, 4), datetime.date(2024, 3, 5), datetime.date(2024, 3, 6)])
        self.assertTrue(all(t["outcome"] == "TP" for t in trades))

    def test_empty_input_returns_no_trades(self):
        self.assertEqual(bdm.backtest_index("TEST", None, TZ, SESSION_OPEN, df=pd.DataFrame()), [])


if __name__ == "__main__":
    unittest.main()
