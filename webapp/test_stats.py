# Unit tests for stats.py's compounding functions - added alongside the fix for a real bug
# found on a live deployment: a strategy with a genuinely thin, cost-adjusted-negative edge run
# over a large (~41k) trade sample showed a "-10,977%" total return and (via the dollar equity
# curve) an implied negative account balance, both mathematically impossible for an account that
# risks a % of its CURRENT balance each trade. Root cause: total %/dollar-equity were computed by
# naively SUMMING r*risk_pct (equivalent to risking a fixed dollar amount off the STARTING
# balance forever) instead of COMPOUNDING (balance *= 1 + r*risk_pct/100), which can only ever
# approach zero, never cross it. The same bug was ALSO visible in the "Max drawdown" metric
# sitting right next to the (now-fixed) Total % on a live run ("-3157%"), so
# compounded_max_drawdown_pct is covered here too. Also covers split_trades_for_holdout (the
# Compare-All out-of-sample ranking fix) and the previously-untested core functions
# (compute_stats, max_drawdown, apply_cost_adjustment, scale_trades_r, equity_curve,
# per_instrument_breakdown, normalize_trade_dates/trades_to_jsonable) - this whole file was the
# first real test coverage the webapp layer had; it grew alongside each bug this session found.
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


class TestSplitTradesForHoldout(unittest.TestCase):
    def test_empty_trades_returns_empty_both_sides(self):
        fit, holdout, is_date_based = stats_mod.split_trades_for_holdout([])
        self.assertEqual(fit, [])
        self.assertEqual(holdout, [])

    def test_date_based_split_puts_the_last_quarter_of_the_span_in_holdout(self):
        # a clean 100-day span, one trade/day - last 25% of DAYS (not of trade COUNT, though
        # they coincide here since it's one trade/day) should land in holdout
        trades = [{"r": 0.1, "date": datetime.date(2020, 1, 1) + datetime.timedelta(days=i)}
                  for i in range(100)]
        fit, holdout, is_date_based = stats_mod.split_trades_for_holdout(trades, holdout_fraction=0.25)
        self.assertTrue(is_date_based)
        self.assertEqual(len(fit) + len(holdout), 100)
        # threshold = day 0 + round(99 * 0.75) = day 74 -> holdout is days 74..99 (26 trades)
        self.assertEqual(len(holdout), 26)
        self.assertTrue(all(t["date"] >= trades[74]["date"] for t in holdout))
        self.assertTrue(all(t["date"] < trades[74]["date"] for t in fit))

    def test_date_based_split_uses_the_trades_own_span_not_a_caller_supplied_window(self):
        # trades only span 40 of a hypothetical wider fetch window - the split must be computed
        # from the trades' OWN min/max date, not some external range this function never sees
        trades = [{"r": 0.1, "date": datetime.date(2021, 6, 1) + datetime.timedelta(days=i)}
                  for i in range(40)]
        fit, holdout, is_date_based = stats_mod.split_trades_for_holdout(trades, holdout_fraction=0.5)
        self.assertEqual(len(fit) + len(holdout), 40)
        self.assertGreater(len(holdout), 0)
        self.assertGreater(len(fit), 0)

    def test_falls_back_to_positional_split_when_trades_have_no_dates(self):
        trades = [{"r": float(i)} for i in range(100)]   # no "date" key at all
        fit, holdout, is_date_based = stats_mod.split_trades_for_holdout(trades, holdout_fraction=0.25)
        self.assertFalse(is_date_based)
        self.assertEqual(len(fit), 75)
        self.assertEqual(len(holdout), 25)
        # the LAST 25% of the list order, not a random subset
        self.assertEqual(holdout[0]["r"], 75.0)
        self.assertEqual(fit[-1]["r"], 74.0)

    def test_fit_and_holdout_together_account_for_every_trade_exactly_once(self):
        trades = [{"r": 0.1, "date": datetime.date(2020, 1, 1) + datetime.timedelta(days=i)}
                  for i in range(37)]   # an awkward, non-round count on purpose
        fit, holdout, _ = stats_mod.split_trades_for_holdout(trades, holdout_fraction=0.25)
        self.assertEqual(len(fit) + len(holdout), len(trades))
        self.assertEqual(set(id(t) for t in fit) | set(id(t) for t in holdout), set(id(t) for t in trades))
        self.assertEqual(set(id(t) for t in fit) & set(id(t) for t in holdout), set())


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


class TestFixedFractionDollarEquityCurve(unittest.TestCase):
    """The non-compounding counterpart to dollar_equity_curve - every trade risks risk_pct% of the
    STARTING balance, not the current one, so this is the volatility-drag-free view: a symmetric
    50/50 R=1 sequence should net to (near) zero here even over thousands of trades, unlike the
    compounded version which drifts to a loss purely from compounding a fixed % of a moving
    balance (see TestDollarEquityCurve's sibling test for that contrast)."""

    def test_matches_hand_computed_addition_for_two_trades(self):
        trades = [{"r": 2.0, "date": datetime.date(2020, 1, 1)},
                  {"r": -1.0, "date": datetime.date(2020, 1, 2)}]
        xs, equity, chronological = stats_mod.fixed_fraction_dollar_equity_curve(
            trades, risk_pct=1.0, starting_balance=10000.0)
        self.assertTrue(chronological)
        # +2R then -1R at 1% risk of the ORIGINAL balance: +$200, then -$100 - additive, not
        # compounded off the new $10,200 balance (which the sibling compounding test IS off of).
        self.assertAlmostEqual(equity[0], 10200.0, places=6)
        self.assertAlmostEqual(equity[1], 10100.0, places=6)

    def test_a_perfectly_symmetric_coin_flip_sequence_nets_to_zero(self):
        # The exact effect this function exists to isolate: dollar_equity_curve's compounding
        # would drag this same sequence toward a loss (volatility drag); the additive version
        # must not, because a fixed-dollar bet has zero expected value on a symmetric bet.
        trades = ([{"r": 1.0} for _ in range(2500)] + [{"r": -1.0} for _ in range(2500)])
        xs, equity, chronological = stats_mod.fixed_fraction_dollar_equity_curve(
            trades, risk_pct=1.0, starting_balance=10000.0)
        self.assertAlmostEqual(equity[-1], 10000.0, places=6)

    def test_can_go_negative_unlike_the_compounding_version(self):
        # The whole documented tradeoff: no floor at zero. A long enough losing streak at a fixed
        # dollar risk drives the account negative - unrealistic, but an honest consequence of
        # never resizing risk down, shown rather than silently clipped.
        trades = [{"r": -1.0} for _ in range(150)]
        xs, equity, chronological = stats_mod.fixed_fraction_dollar_equity_curve(
            trades, risk_pct=1.0, starting_balance=10000.0)
        self.assertLess(equity[-1], 0.0)
        self.assertAlmostEqual(equity[-1], 10000.0 * (1.0 - 150 * 0.01), places=6)

    def test_falls_back_to_sequence_order_when_trades_have_no_dates(self):
        trades = [{"r": 1.0}, {"r": -0.5}]
        xs, equity, chronological = stats_mod.fixed_fraction_dollar_equity_curve(
            trades, risk_pct=1.0, starting_balance=10000.0)
        self.assertFalse(chronological)
        self.assertEqual(len(equity), 2)


class TestComputeStats(unittest.TestCase):
    def test_empty_trades_returns_none(self):
        self.assertIsNone(stats_mod.compute_stats([]))

    def test_basic_aggregates_match_hand_computation(self):
        trades = [{"r": 2.0, "outcome": "TP"}, {"r": -1.0, "outcome": "SL"}, {"r": 0.0, "outcome": "FLAT"}]
        s = stats_mod.compute_stats(trades)
        self.assertEqual(s["n_trades"], 3)
        self.assertAlmostEqual(s["total_r"], 1.0, places=6)
        self.assertAlmostEqual(s["avg_r"], 1.0 / 3, places=6)
        self.assertEqual(s["tp"], 1)
        self.assertEqual(s["sl"], 1)
        self.assertEqual(s["flat"], 1)
        self.assertAlmostEqual(s["tp_pct"], 100 / 3, places=6)
        # win_pct/loss_pct are BY R (r > 0), not by outcome label - here they happen to match
        # tp_pct/sl_pct exactly (1 winner, 2 non-winners) because this fixture's outcome labels
        # happen to line up with sign of r, but see TestWinLossPctIsByRNotOutcomeLabel below for
        # a fixture where they genuinely diverge (the whole point of these fields existing).
        self.assertAlmostEqual(s["win_pct"], 100 / 3, places=6)
        self.assertAlmostEqual(s["loss_pct"], 200 / 3, places=6)

    def test_win_loss_pct_is_by_r_not_outcome_label(self):
        # A trailing-stop strategy (Donchian/Dow Theory/Parabolic SAR convention) never produces a
        # "TP" or "SL" label - only "STOP"/"FLAT" - so tp_pct/sl_pct show 0% regardless of how
        # profitable the strategy actually is. win_pct/loss_pct must not have this blind spot:
        # caught live on a real deployed leaderboard (Parabolic SAR: +1.8% holdout return, 0%
        # "win rate" from the old tp_pct-based metric).
        trades = [{"r": 2.0, "outcome": "STOP"}, {"r": 1.0, "outcome": "STOP"}, {"r": -1.0, "outcome": "STOP"},
                  {"r": 0.0, "outcome": "FLAT"}]
        s = stats_mod.compute_stats(trades)
        self.assertEqual(s["tp"], 0)   # the old, blind-spotted metric - confirms the bug is real
        self.assertEqual(s["tp_pct"], 0.0)
        self.assertAlmostEqual(s["win_pct"], 50.0, places=6)    # 2 of 4 trades have r > 0
        self.assertAlmostEqual(s["loss_pct"], 50.0, places=6)   # r <= 0: the -1.0 and the 0.0 (breakeven)

    def test_breakeven_trade_does_not_count_as_a_win(self):
        trades = [{"r": 0.0}, {"r": 0.0}, {"r": 1.0}]
        s = stats_mod.compute_stats(trades)
        self.assertAlmostEqual(s["win_pct"], 100 / 3, places=6)
        self.assertAlmostEqual(s["loss_pct"], 200 / 3, places=6)

    def test_single_trade_has_zero_z_score_and_a_degenerate_ci(self):
        # stdev is undefined for n=1 - z-score and CI must not crash, and should collapse to the
        # point estimate rather than claim false precision
        s = stats_mod.compute_stats([{"r": 1.5, "outcome": "TP"}])
        self.assertEqual(s["z_score"], 0.0)
        self.assertEqual(s["avg_r_ci_low"], s["avg_r_ci_high"])
        self.assertEqual(s["avg_r_ci_low"], 1.5)

    def test_ci_widens_around_the_point_estimate_for_a_noisy_sample(self):
        trades = [{"r": r} for r in [2.0, -1.0, 3.0, -2.0, 1.0, -1.5, 2.5]]
        s = stats_mod.compute_stats(trades)
        self.assertLess(s["avg_r_ci_low"], s["avg_r"])
        self.assertGreater(s["avg_r_ci_high"], s["avg_r"])


class TestMaxDrawdown(unittest.TestCase):
    def test_never_dipping_below_the_running_peak_has_zero_drawdown(self):
        trades = [{"r": 1.0}, {"r": 1.0}, {"r": 1.0}]
        self.assertEqual(stats_mod.max_drawdown(trades), 0.0)

    def test_matches_hand_computed_peak_to_trough(self):
        # cumulative: 2, 1, 3, 0.5 -> peak hits 3 at step 3, trough of 0.5 at step 4 -> dd = 2.5
        trades = [{"r": 2.0}, {"r": -1.0}, {"r": 2.0}, {"r": -2.5}]
        self.assertAlmostEqual(stats_mod.max_drawdown(trades), 2.5, places=6)

    def test_orders_by_date_when_every_trade_has_one_even_if_list_order_differs(self):
        # list order is deliberately NOT chronological - max_drawdown must sort by date itself
        trades = [
            {"r": -2.5, "date": datetime.date(2020, 1, 4)},
            {"r": 2.0, "date": datetime.date(2020, 1, 1)},
            {"r": -1.0, "date": datetime.date(2020, 1, 2)},
            {"r": 2.0, "date": datetime.date(2020, 1, 3)},
        ]
        self.assertAlmostEqual(stats_mod.max_drawdown(trades), 2.5, places=6)


class TestApplyCostAdjustment(unittest.TestCase):
    def test_known_instrument_uses_its_own_typical_cost(self):
        trades = [{"r": 1.0, "stop_pct": 0.01, "instrument": "EURUSD"}]
        adjusted, n_unadjusted = stats_mod.apply_cost_adjustment(trades)
        self.assertEqual(n_unadjusted, 0)
        expected_r = 1.0 - (stats_mod.TYPICAL_COST_PCT_BY_INSTRUMENT["EURUSD"] / 100.0) / 0.01
        self.assertAlmostEqual(adjusted[0]["r"], expected_r, places=6)

    def test_unknown_instrument_falls_back_to_default_cost(self):
        trades = [{"r": 1.0, "stop_pct": 0.01, "instrument": "SOMETHING_NOT_IN_THE_TABLE"}]
        adjusted, _ = stats_mod.apply_cost_adjustment(trades)
        expected_r = 1.0 - (stats_mod.DEFAULT_COST_PCT / 100.0) / 0.01
        self.assertAlmostEqual(adjusted[0]["r"], expected_r, places=6)

    def test_trade_with_no_stop_pct_is_left_unadjusted_and_counted(self):
        trades = [{"r": 1.0, "instrument": "EURUSD"}]   # no "stop_pct" key at all
        adjusted, n_unadjusted = stats_mod.apply_cost_adjustment(trades)
        self.assertEqual(n_unadjusted, 1)
        self.assertEqual(adjusted[0]["r"], 1.0)   # untouched

    def test_does_not_mutate_the_input_trades(self):
        trades = [{"r": 1.0, "stop_pct": 0.01, "instrument": "EURUSD"}]
        stats_mod.apply_cost_adjustment(trades)
        self.assertEqual(trades[0]["r"], 1.0)   # original list/dicts unchanged


class TestScaleTradesR(unittest.TestCase):
    def test_multiplies_every_r_by_the_factor(self):
        trades = [{"r": 2.0}, {"r": -1.0}]
        scaled = stats_mod.scale_trades_r(trades, 1.5)
        self.assertAlmostEqual(scaled[0]["r"], 3.0, places=6)
        self.assertAlmostEqual(scaled[1]["r"], -1.5, places=6)

    def test_does_not_mutate_the_input(self):
        trades = [{"r": 2.0}]
        stats_mod.scale_trades_r(trades, 2.0)
        self.assertEqual(trades[0]["r"], 2.0)


class TestEquityCurve(unittest.TestCase):
    def test_cumulative_r_matches_hand_computation(self):
        trades = [{"r": 1.0}, {"r": -0.5}, {"r": 2.0}]
        xs, ys, chronological = stats_mod.equity_curve(trades)
        self.assertEqual(xs, [1, 2, 3])
        self.assertEqual(ys, [1.0, 0.5, 2.5])

    def test_sorts_chronologically_when_every_trade_has_a_date(self):
        trades = [{"r": 1.0, "date": datetime.date(2020, 1, 2)},
                  {"r": 2.0, "date": datetime.date(2020, 1, 1)}]
        xs, ys, chronological = stats_mod.equity_curve(trades)
        self.assertTrue(chronological)
        self.assertEqual(ys, [2.0, 3.0])   # 2020-01-01's trade (r=2.0) comes first


class TestPerInstrumentBreakdown(unittest.TestCase):
    def test_groups_and_sorts_by_total_r_descending(self):
        trades = [
            {"r": 1.0, "instrument": "EURUSD", "outcome": "TP"},
            {"r": -3.0, "instrument": "GBPUSD", "outcome": "SL"},
            {"r": 2.0, "instrument": "EURUSD", "outcome": "TP"},
        ]
        rows = stats_mod.per_instrument_breakdown(trades)
        self.assertEqual([r["instrument"] for r in rows], ["EURUSD", "GBPUSD"])
        self.assertAlmostEqual(rows[0]["total_r"], 3.0, places=3)
        self.assertEqual(rows[0]["trades"], 2)


class TestNormalizeAndJsonable(unittest.TestCase):
    def test_normalize_converts_iso_strings_to_date_objects(self):
        trades = [{"r": 1.0, "date": "2020-01-15"}]
        normalized = stats_mod.normalize_trade_dates(trades)
        self.assertEqual(normalized[0]["date"], datetime.date(2020, 1, 15))

    def test_normalize_leaves_real_date_objects_alone(self):
        d = datetime.date(2020, 1, 15)
        trades = [{"r": 1.0, "date": d}]
        normalized = stats_mod.normalize_trade_dates(trades)
        self.assertEqual(normalized[0]["date"], d)

    def test_jsonable_round_trips_through_normalize(self):
        trades = [{"r": 1.0, "date": datetime.date(2020, 1, 15)}]
        jsonable = stats_mod.trades_to_jsonable(trades)
        self.assertEqual(jsonable[0]["date"], "2020-01-15")
        back = stats_mod.normalize_trade_dates(jsonable)
        self.assertEqual(back[0]["date"], datetime.date(2020, 1, 15))


if __name__ == "__main__":
    unittest.main()
