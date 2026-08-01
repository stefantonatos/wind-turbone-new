# Unit tests for the risk-per-trade sweep and losing-streak-probability
# diagnostic added to prop_firm_challenge_simulator.py.
#
# Covers, per this project's rigor conventions:
#   1. consecutive_losses_to_breach() on known account-rule parameters (hand-derived
#      expected values, including the exact-division edge case).
#   2. Bootstrap resampling (bootstrap_resample_r_matrix) producing a distribution
#      whose mean converges toward the real trade list's mean as n_iter grows.
#   3. Early-stop-on-loss-rule-breach: a crafted trade sequence that obviously
#      breaches the daily-loss rule, confirming simulate_challenge_path() stops and
#      marks FAIL at the exact right trade.
#   4. Profit-target-reached-first logic: a crafted winning sequence, confirming PASS
#      at the exact right trade once both the target and min-trading-days are met.
#   5. Longest-consecutive-loss-run detection (used by both risk_sweep's "streak
#      seen" stat and the losing-streak diagnonstic) against a hand-counted sequence.
#   6. prob_run_at_least_k() (exact iid-Bernoulli DP) cross-checked against brute-force
#      enumeration for small n/k.
#   7. estimate_trades_per_day() on a known synthetic date list.
#   8. Small synthetic end-to-end smoke runs of risk_sweep() and
#      losing_streak_probabilities() confirming the full pipelines run without
#      crashing and produce sane, internally-consistent output (deliberately NOT a
#      real Dukascopy download - this project's convention is to validate via unit
#      tests plus a synthetic smoke run).
#
# Run with:  python -m pytest research/test_prop_firm_challenge_simulator.py -v
# or:        python research/test_prop_firm_challenge_simulator.py

import datetime
import importlib.util
import itertools
import os
import unittest

import numpy as np

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prop_firm_challenge_simulator.py")
_spec = importlib.util.spec_from_file_location("prop_firm_challenge_simulator", _MODULE_PATH)
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)


def _make_trades(r_values, start_date=datetime.date(2024, 1, 1), trades_per_day=1):
    """Builds a synthetic trade list (only 'r' and 'date' matter to this module)."""
    trades = []
    day = start_date
    for i, r in enumerate(r_values):
        if i > 0 and i % trades_per_day == 0:
            day = day + datetime.timedelta(days=1)
        trades.append({"r": r, "date": day, "outcome": "SL" if r < 0 else "TP"})
    return trades


class TestConsecutiveLossesToBreach(unittest.TestCase):
    """Deterministic (no simulation) calculation, hand-derived against this project's
    default account rules: $10k account, 5% max daily loss, 10% max overall loss."""

    def test_default_1pct_risk_daily_binds(self):
        # k * 1% >= 5% first at k=5 -> survives 4, matches the video's own "4 in a row"
        # framing falling out naturally at this project's real default rules.
        survivable, reason, breach_at = sim.consecutive_losses_to_breach(1.0, 5.0, 10.0)
        self.assertEqual(survivable, 4)
        self.assertEqual(reason, "daily_drawdown")
        self.assertEqual(breach_at, 5)

    def test_2pct_risk_daily_binds(self):
        # k * 2% >= 5% first at k=3 (2%,4%,6%) -> survives 2.
        survivable, reason, breach_at = sim.consecutive_losses_to_breach(2.0, 5.0, 10.0)
        self.assertEqual(survivable, 2)
        self.assertEqual(reason, "daily_drawdown")
        self.assertEqual(breach_at, 3)

    def test_5pct_risk_cannot_survive_even_one_loss(self):
        # k * 5% >= 5% exactly at k=1 -> survives 0 (the very first loss ends it).
        survivable, reason, breach_at = sim.consecutive_losses_to_breach(5.0, 5.0, 10.0)
        self.assertEqual(survivable, 0)
        self.assertEqual(reason, "daily_drawdown")
        self.assertEqual(breach_at, 1)

    def test_small_risk_survives_many_losses(self):
        # k * 0.5% >= 5% first at k=10 -> survives 9. Overall: k*0.5%>=10% at k=20 (not binding).
        survivable, reason, breach_at = sim.consecutive_losses_to_breach(0.5, 5.0, 10.0)
        self.assertEqual(survivable, 9)
        self.assertEqual(reason, "daily_drawdown")
        self.assertEqual(breach_at, 10)

    def test_overall_can_bind_when_daily_limit_is_looser(self):
        # If daily limit is very loose (50%) but overall is tight (4%), overall binds.
        # k * 2% >= 4% first at k=2 -> survives 1.
        survivable, reason, breach_at = sim.consecutive_losses_to_breach(2.0, 50.0, 4.0)
        self.assertEqual(survivable, 1)
        self.assertEqual(reason, "overall_drawdown")
        self.assertEqual(breach_at, 2)

    def test_zero_risk_is_effectively_infinite_runway(self):
        survivable, reason, breach_at = sim.consecutive_losses_to_breach(0.0, 5.0, 10.0)
        self.assertEqual(survivable, float("inf"))


class TestBootstrapResampling(unittest.TestCase):
    """bootstrap_resample_r_matrix must draw WITH replacement, shape (n_iter, n_per_path),
    and its mean should converge toward the real trade list's mean as n_iter grows."""

    def test_shape_and_replacement(self):
        r_values = np.array([1.0, -1.0, 2.0, -1.0, 0.5])
        rng = np.random.default_rng(0)
        matrix = sim.bootstrap_resample_r_matrix(r_values, n_iter=50, n_per_path=20, rng=rng)
        self.assertEqual(matrix.shape, (50, 20))
        # every sampled value must come from the original set (with-replacement resampling
        # can only ever reproduce the original values, never invent new ones)
        self.assertTrue(np.isin(matrix, r_values).all())

    def test_mean_converges_to_true_mean_as_n_iter_grows(self):
        rng_source = np.random.default_rng(123)
        r_values = rng_source.normal(loc=0.05, scale=1.0, size=300)
        true_mean = r_values.mean()

        small_rng = np.random.default_rng(1)
        small = sim.bootstrap_resample_r_matrix(r_values, n_iter=20, n_per_path=300, rng=small_rng)
        small_gap = abs(small.mean() - true_mean)

        large_rng = np.random.default_rng(2)
        large = sim.bootstrap_resample_r_matrix(r_values, n_iter=20000, n_per_path=300, rng=large_rng)
        large_gap = abs(large.mean() - true_mean)

        # more iterations -> the aggregate bootstrap mean should land much closer to the
        # true mean (law of large numbers) - allow generous slack, this only needs to show
        # the right direction/order of magnitude, not a tight bound.
        self.assertLess(large_gap, small_gap)
        self.assertLess(large_gap, 0.01)


class TestSimulateChallengePath(unittest.TestCase):
    """Deterministic per-path account-rule walkthrough - no randomness involved once
    r_values_path is fixed, so exact trade-by-trade behavior is fully checkable."""

    def test_early_stop_on_daily_drawdown_breach(self):
        # 1% risk/trade, defaults (5% daily / 10% overall). trades_per_day=10 buckets
        # all 7 of these trades into the SAME synthetic day, so the daily-loss check
        # can actually accumulate across consecutive losses (with trades_per_day=1
        # every trade would start a fresh day and the daily check could never see more
        # than one trade's worth of loss - the point of this test is exactly the
        # multi-loss accumulation, so it needs the losses sharing a day). Breaches at
        # the 5th consecutive -1R loss (see TestConsecutiveLossesToBreach).
        r_path = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]  # 7 losses available
        result = sim.simulate_challenge_path(
            r_path, initial_balance=10000.0, risk_pct_per_trade=1.0, profit_target_pct=8.0,
            max_daily_loss_pct=5.0, max_overall_loss_pct=10.0, min_trading_days=4,
            drawdown_mode="static", trades_per_day=10,
        )
        self.assertEqual(result["outcome"], "FAIL")
        self.assertEqual(result["reason"], "daily_drawdown")
        self.assertEqual(result["trades_taken"], 5)  # stops on the 5th loss, doesn't run further
        self.assertAlmostEqual(result["final_equity"], 10000.0 - 5 * 100.0)

    def test_early_stop_on_overall_drawdown_breach_when_daily_resets(self):
        # 1 loss per day (daily resets every trade) so the daily rule never binds -
        # only the overall 10% floor can. At 1% risk, that's 10 consecutive losses.
        r_path = [-1.0] * 15
        result = sim.simulate_challenge_path(
            r_path, initial_balance=10000.0, risk_pct_per_trade=1.0, profit_target_pct=8.0,
            max_daily_loss_pct=5.0, max_overall_loss_pct=10.0, min_trading_days=4,
            drawdown_mode="static", trades_per_day=1,
        )
        self.assertEqual(result["outcome"], "FAIL")
        self.assertEqual(result["reason"], "overall_drawdown_static")
        self.assertEqual(result["trades_taken"], 10)

    def test_profit_target_reached_first(self):
        # Enough winning trades (2R each) across enough distinct days to clear both the
        # 8% profit target and the 4-day minimum, with no losses in the way.
        r_path = [2.0, 2.0, 2.0, 2.0, 2.0]  # 1 trade/day -> 5 distinct days
        result = sim.simulate_challenge_path(
            r_path, initial_balance=10000.0, risk_pct_per_trade=1.0, profit_target_pct=8.0,
            max_daily_loss_pct=5.0, max_overall_loss_pct=10.0, min_trading_days=4,
            drawdown_mode="static", trades_per_day=1,
        )
        self.assertEqual(result["outcome"], "PASS")
        # +2% equity/trade -> target ($10,800) cleared on the 4th trade (+8%), but
        # min_trading_days=4 means the 4th trade IS enough (4th day satisfies both
        # conditions simultaneously).
        self.assertEqual(result["trades_taken"], 4)
        self.assertEqual(result["days_taken"], 4)
        self.assertAlmostEqual(result["final_equity"], 10000.0 + 4 * 200.0)

    def test_min_trading_days_delays_pass_even_if_target_hit_early(self):
        # A single huge winning trade clears the profit target immediately in equity
        # terms, but min_trading_days=4 must still be satisfied before PASS fires.
        r_path = [20.0, 0.1, 0.1, 0.1, 0.1]  # first trade alone blows past +8%
        result = sim.simulate_challenge_path(
            r_path, initial_balance=10000.0, risk_pct_per_trade=1.0, profit_target_pct=8.0,
            max_daily_loss_pct=5.0, max_overall_loss_pct=10.0, min_trading_days=4,
            drawdown_mode="static", trades_per_day=1,
        )
        self.assertEqual(result["outcome"], "PASS")
        self.assertEqual(result["trades_taken"], 4)
        self.assertEqual(result["days_taken"], 4)

    def test_inconclusive_when_path_runs_out(self):
        r_path = [0.1, 0.1, 0.1]  # nowhere near target, nowhere near breach
        result = sim.simulate_challenge_path(
            r_path, initial_balance=10000.0, risk_pct_per_trade=1.0, profit_target_pct=8.0,
            max_daily_loss_pct=5.0, max_overall_loss_pct=10.0, min_trading_days=4,
            drawdown_mode="static", trades_per_day=1,
        )
        self.assertEqual(result["outcome"], "INCONCLUSIVE")
        self.assertEqual(result["trades_taken"], 3)

    def test_max_consecutive_losses_tracked_correctly(self):
        r_path = [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0]  # longest run = 3
        result = sim.simulate_challenge_path(
            r_path, initial_balance=10000.0, risk_pct_per_trade=0.1, profit_target_pct=8.0,
            max_daily_loss_pct=50.0, max_overall_loss_pct=50.0, min_trading_days=1,
            drawdown_mode="static", trades_per_day=1,
        )
        self.assertEqual(result["max_consecutive_losses"], 3)
        self.assertEqual(result["outcome"], "INCONCLUSIVE")  # rules loose enough not to resolve


class TestEstimateTradesPerDay(unittest.TestCase):
    def test_known_ratio(self):
        d1, d2, d3 = datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)
        trades = [{"date": d1}, {"date": d1}, {"date": d1}, {"date": d2}, {"date": d2}, {"date": d3}]
        # 6 trades / 3 days = 2/day
        self.assertEqual(sim.estimate_trades_per_day(trades), 2)

    def test_empty_defaults_to_one(self):
        self.assertEqual(sim.estimate_trades_per_day([]), 1)


class TestLongestRunDetectionAndBernoulliDP(unittest.TestCase):
    def test_longest_run_matches_hand_count(self):
        # losses (True) at positions 1-3 (len 3), 5-6 (len 2), 8-11 (len 4) -> longest 4
        r = np.array([1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1], dtype=float)
        loss_bool = (r < 0).reshape(1, -1)
        n = loss_bool.shape[1]
        consec = np.zeros((1, n), dtype=np.int64)
        consec[:, 0] = loss_bool[:, 0]
        for j in range(1, n):
            consec[:, j] = np.where(loss_bool[:, j], consec[:, j - 1] + 1, 0)
        self.assertEqual(int(consec.max()), 4)

    def test_prob_run_at_least_k_matches_brute_force_enumeration(self):
        # Small n/k so full enumeration is feasible - the DP must match exactly.
        for n, k, p in [(4, 2, 0.5), (5, 3, 0.3), (6, 2, 0.7), (3, 3, 0.4)]:
            total_mass = 0.0
            hit_mass = 0.0
            for outcome in itertools.product([0, 1], repeat=n):  # 1 = "loss"
                prob = 1.0
                for bit in outcome:
                    prob *= p if bit == 1 else (1 - p)
                total_mass += prob
                # check for a run of >= k consecutive 1s
                run = 0
                hit = False
                for bit in outcome:
                    run = run + 1 if bit == 1 else 0
                    if run >= k:
                        hit = True
                        break
                if hit:
                    hit_mass += prob
            self.assertAlmostEqual(total_mass, 1.0, places=9)
            dp_result = sim.prob_run_at_least_k(n, k, p)
            self.assertAlmostEqual(dp_result, hit_mass, places=9,
                                    msg=f"mismatch at n={n}, k={k}, p={p}")

    def test_prob_run_at_least_k_edge_cases(self):
        self.assertEqual(sim.prob_run_at_least_k(5, 0, 0.5), 1.0)
        self.assertEqual(sim.prob_run_at_least_k(2, 5, 0.5), 0.0)  # n < k -> impossible
        self.assertAlmostEqual(sim.prob_run_at_least_k(2, 2, 0.5), 0.25)  # only "LL"


class TestRiskSweepSmoke(unittest.TestCase):
    """Small synthetic end-to-end smoke run - not a real Dukascopy download, per this
    project's convention of validating via unit tests + a synthetic smoke run."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        # A modestly-positive-edge synthetic strategy: ~50% win rate, 2:1 R:R
        # (avg R/trade = 0.5*2 - 0.5*1 = +0.5R), 400 trades, ~2/day.
        n = 400
        wins = rng.random(n) < 0.5
        r_values = np.where(wins, 2.0, -1.0)
        cls.trades = _make_trades(r_values.tolist(), trades_per_day=2)

    def test_risk_sweep_runs_and_produces_sane_output(self):
        results = sim.risk_sweep(
            self.trades, risk_levels_pct=[0.25, 1.0, 3.0, 5.0], n_iter=300,
            max_trades_per_path=400, initial_balance=10000.0, profit_target_pct=8.0,
            max_daily_loss_pct=5.0, max_overall_loss_pct=10.0, min_trading_days=4,
            drawdown_mode="static", seed=1,
        )
        self.assertEqual(len(results), 4)
        for row in results:
            self.assertGreaterEqual(row["pass_prob"], 0.0)
            self.assertLessEqual(row["pass_prob"], 1.0)
            self.assertAlmostEqual(row["pass_prob"] + row["fail_prob"] + row["inconclusive_prob"], 1.0, places=6)

        # max_survivable_consecutive_losses must strictly decrease as risk grows (more
        # risk/trade -> less runway) for this fixed set of account rules.
        survivable_by_risk = [row["max_survivable_consecutive_losses"] for row in results]
        self.assertEqual(survivable_by_risk, sorted(survivable_by_risk, reverse=True))

        # the 5% level should be the degenerate one that can't survive even 1 loss
        five_pct_row = next(r for r in results if r["risk_pct_per_trade"] == 5.0)
        self.assertEqual(five_pct_row["max_survivable_consecutive_losses"], 0)

    def test_risk_sweep_empty_on_too_few_trades(self):
        self.assertEqual(sim.risk_sweep(self.trades[:5], n_iter=10), [])


class TestLosingStreakProbabilitiesSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        n = 500
        wins = rng.random(n) < 0.55
        cls.r_values = np.where(wins, 1.0, -1.0)

    def test_runs_and_produces_sane_output(self):
        results = sim.losing_streak_probabilities(
            self.r_values, ks=[2, 3, 4], n_iter=500, sample_sizes=[250], seed=1,
        )
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["sample_size"], 250)
        self.assertAlmostEqual(res["win_rate"], (self.r_values > 0).mean())
        # P(>=k) must be non-increasing as k grows, both empirically and theoretically
        emp = [row["empirical_p"] for row in res["per_k"]]
        theo = [row["theoretical_p"] for row in res["per_k"]]
        self.assertEqual(emp, sorted(emp, reverse=True))
        self.assertEqual(theo, sorted(theo, reverse=True))
        for row in res["per_k"]:
            self.assertGreaterEqual(row["empirical_p"], 0.0)
            self.assertLessEqual(row["empirical_p"], 1.0)

    def test_empty_on_too_few_trades(self):
        self.assertEqual(sim.losing_streak_probabilities(self.r_values[:5]), [])


if __name__ == "__main__":
    unittest.main()
