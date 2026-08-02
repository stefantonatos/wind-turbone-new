# Unit tests for stats.py's compounding functions - added alongside the fix for a real bug
# found on a live deployment: a strategy with a genuinely thin, cost-adjusted-negative edge run
# over a large (~41k) trade sample showed a "-10,977%" total return and (via the dollar equity
# curve) an implied negative account balance, both mathematically impossible for an account that
# risks a % of its CURRENT balance each trade. Root cause: total %/dollar-equity were computed by
# naively SUMMING r*risk_pct (equivalent to risking a fixed dollar amount off the STARTING
# balance forever) instead of COMPOUNDING (balance *= 1 + r*risk_pct/100), which can only ever
# approach zero, never cross it. The same bug was ALSO visible in the "Max drawdown" metric
# sitting right next to the (now-fixed) Total % on a live run ("-3157%"), so
# compounded_max_drawdown_pct is covered here too. Covers compounded_return_pct,
# compounded_return_ci, compounded_max_drawdown_pct, and the fixed dollar_equity_curve.
#
# Run with:  python -m pytest webapp/test_stats.py -v

import datetime
import unittest

import stats as stats_mod


class TestCompoundedReturnPct(unittest.TestCase):
    def test_matches_naive_sum_for_a_single_trade(self):
        # with exactly one trade, compounding and additive scaling are identical by construction
        trades = [{"r": 2.0}]
        self.assertAlmostEqual(stats_mod.compounded_return_pct(trades, risk_pct=1.0), 2.0, places=6)

    def test_matches_hand_computed_compounding_for_two_trades(self):
        # growth = (1 + 0.02) * (1 - 0.01) = 1.0098 -> +0.98%, NOT the naive (2 - 1) = +1.0%
        trades = [{"r": 2.0}, {"r": -1.0}]
        expected = ((1 + 0.02) * (1 - 0.01) - 1) * 100
        self.assertAlmostEqual(stats_mod.compounded_return_pct(trades, risk_pct=1.0), expected, places=6)
        self.assertNotAlmostEqual(stats_mod.compounded_return_pct(trades, risk_pct=1.0), 1.0, places=2)

    def test_never_crosses_minus_100_no_matter_how_many_losing_trades(self):
        # the exact regression this fix targets - a large sample of a genuinely bad, cost-adjusted
        # edge must show a bounded, realistic loss, not an impossible "-10,977%"-style number.
        # >= not > : at this scale (1 - 0.0068)**40000 underflows to ~3e-119, so -100.0 is the
        # correct (not just approximate) floating-point value - a real account this deep
        # underwater IS fully wiped out, "-100%" is the honest answer, not a bug.
        trades = [{"r": -0.68} for _ in range(40000)]
        pct = stats_mod.compounded_return_pct(trades, risk_pct=1.0)
        self.assertGreaterEqual(pct, -100.0)
        # the naive additive equivalent WOULD have been impossible - confirms this is a real fix,
        # not a no-op that happens to agree with the old behavior
        naive_equivalent = sum(t["r"] for t in trades) * 1.0
        self.assertLess(naive_equivalent, -100.0)

    def test_stays_strictly_above_minus_100_at_a_scale_precision_can_still_resolve(self):
        # same shape as the underflow test above, but at a small enough N that -100% is NOT the
        # nearest representable float, isolating the "approaches but never crosses" guarantee
        # from the separate (and also correct) underflow behavior at extreme N
        trades = [{"r": -0.68} for _ in range(50)]
        pct = stats_mod.compounded_return_pct(trades, risk_pct=1.0)
        self.assertGreater(pct, -100.0)

    def test_a_total_wipeout_trade_floors_growth_at_zero_not_negative(self):
        # a single trade whose r*risk_pct implies losing MORE than the whole balance (a large R
        # loss at a large risk_pct) must floor at -100%, not go further negative
        trades = [{"r": -1.0}]
        pct = stats_mod.compounded_return_pct(trades, risk_pct=150.0)   # -1R * 150% = -150% naively
        self.assertAlmostEqual(pct, -100.0, places=6)

    def test_profitable_edge_compounds_to_a_sensible_positive_return(self):
        trades = [{"r": 0.3} for _ in range(200)]
        pct = stats_mod.compounded_return_pct(trades, risk_pct=1.0)
        self.assertGreater(pct, 0.0)
        # compounding a positive edge should exceed the naive additive total (growth compounds
        # favorably on wins too - (1.003)^200 - 1 > 200 * 0.003)
        naive_equivalent = sum(t["r"] for t in trades) * 1.0
        self.assertGreater(pct, naive_equivalent)


class TestCompoundedReturnCi(unittest.TestCase):
    def test_ci_bounds_never_cross_minus_100(self):
        low, high = stats_mod.compounded_return_ci(-0.7, -0.6, risk_pct=1.0, n=40000)
        self.assertGreaterEqual(low, -100.0)   # underflows to exactly -100.0 at this scale - correct
        self.assertGreaterEqual(high, -100.0)
        self.assertLessEqual(low, high)

    def test_ci_matches_compounded_return_pct_for_a_constant_average(self):
        # if every trade earned exactly avg_r, compounded_return_ci's bound should match
        # compounded_return_pct on that constant series directly
        n = 50
        avg_r = 0.4
        trades = [{"r": avg_r}] * n
        direct = stats_mod.compounded_return_pct(trades, risk_pct=2.0)
        low, high = stats_mod.compounded_return_ci(avg_r, avg_r, risk_pct=2.0, n=n)
        self.assertAlmostEqual(low, direct, places=6)
        self.assertAlmostEqual(high, direct, places=6)


class TestCompoundedMaxDrawdownPct(unittest.TestCase):
    def test_bounded_to_100_even_on_a_catastrophic_loss_streak(self):
        # the exact second call site this fix targets - "Max drawdown" showed "-3157%" on a live
        # run right next to a correctly-capped "-100.00%" Total %, same underlying bug
        trades = [{"r": -0.68} for _ in range(40000)]
        dd = stats_mod.compounded_max_drawdown_pct(trades, risk_pct=1.0)
        self.assertLessEqual(dd, 100.0)
        self.assertGreaterEqual(dd, 0.0)
        # the naive additive equivalent (max_drawdown_r * risk_pct) WOULD have blown past 100
        naive_equivalent = 40000 * 0.68 * 1.0   # a monotonic loss streak's additive "drawdown"
        self.assertGreater(naive_equivalent, 100.0)

    def test_matches_hand_computed_drawdown_for_a_simple_up_then_down_sequence(self):
        # +10% then -10% (of the new, higher balance): peak = 1.10, trough = 1.10*0.90 = 0.99
        # drawdown = (1.10 - 0.99) / 1.10 * 100 = 10%
        trades = [{"r": 10.0}, {"r": -10.0}]
        dd = stats_mod.compounded_max_drawdown_pct(trades, risk_pct=1.0)
        self.assertAlmostEqual(dd, 10.0, places=6)

    def test_no_losses_means_zero_drawdown(self):
        trades = [{"r": 1.0} for _ in range(20)]
        dd = stats_mod.compounded_max_drawdown_pct(trades, risk_pct=1.0)
        self.assertAlmostEqual(dd, 0.0, places=6)


class TestDollarEquityCurve(unittest.TestCase):
    def test_equity_never_goes_negative_on_a_catastrophic_loss_streak(self):
        trades = [{"r": -0.68, "date": datetime.date(2020, 1, 1) + datetime.timedelta(days=i)}
                  for i in range(5000)]
        xs, equity, chronological = stats_mod.dollar_equity_curve(trades, risk_pct=1.0, starting_balance=10000.0)
        self.assertTrue(all(e >= 0.0 for e in equity))
        self.assertLess(equity[-1], 1.0)   # effectively wiped out, but not negative

    def test_matches_hand_computed_compounding_for_two_trades(self):
        trades = [{"r": 2.0, "date": datetime.date(2020, 1, 1)},
                  {"r": -1.0, "date": datetime.date(2020, 1, 2)}]
        xs, equity, chronological = stats_mod.dollar_equity_curve(trades, risk_pct=1.0, starting_balance=10000.0)
        self.assertTrue(chronological)
        self.assertAlmostEqual(equity[0], 10000.0 * 1.02, places=6)
        self.assertAlmostEqual(equity[1], 10000.0 * 1.02 * 0.99, places=6)

    def test_falls_back_to_sequence_order_when_trades_have_no_dates(self):
        trades = [{"r": 1.0}, {"r": -0.5}]
        xs, equity, chronological = stats_mod.dollar_equity_curve(trades, risk_pct=1.0, starting_balance=10000.0)
        self.assertFalse(chronological)
        self.assertEqual(len(equity), 2)


if __name__ == "__main__":
    unittest.main()
