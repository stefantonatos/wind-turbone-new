# Unit tests for research/optimization_engine.py - the shared, strategy-
# agnostic objective-function registry + search-strategy module used by both
# research/ict_po3_forex_dukascopy_optimization.py and
# research/day_trading_rauf_dukascopy_optimization.py.
#
# Covers, per this project's rigor conventions:
#   1. Every OBJECTIVES function against hand-crafted trade lists where the
#      correct answer is known in advance (constant-R, all-losers, all-
#      winners, a list with a known max drawdown by construction, n=0, n=1).
#   2. grid_search reproduces EXACTLY the same combos+order an old-style
#      hand-written double loop already gets on the same 2-parameter grid
#      shape both companion scripts use.
#   3. bayesian_search and genetic_search actually find a known optimum on a
#      constructed synthetic objective surface, using MEANINGFULLY FEWER
#      evaluations than grid_search's exhaustive count over the same space -
#      the efficiency claim proven numerically, not just "runs without
#      crashing".
#   4. bayesian_search's ImportError handling (optuna missing) never crashes,
#      and run_search's dispatch falls back to grid_search when that happens.
#
# Run with:  python -m pytest research/test_optimization_engine.py -v
# or:        python research/test_optimization_engine.py

import datetime
import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optimization_engine as engine


# ============================= OBJECTIVE FUNCTIONS =============================

def _trades(rs):
    return [{"r": r} for r in rs]


class TestTotalR(unittest.TestCase):
    def test_known_sum(self):
        self.assertAlmostEqual(engine.total_r(_trades([1.0, -0.5, 2.0])), 2.5)

    def test_empty(self):
        self.assertEqual(engine.total_r([]), 0.0)

    def test_single_trade(self):
        self.assertAlmostEqual(engine.total_r(_trades([3.0])), 3.0)

    def test_all_losers(self):
        self.assertAlmostEqual(engine.total_r(_trades([-1.0, -1.0, -1.0])), -3.0)

    def test_all_winners(self):
        self.assertAlmostEqual(engine.total_r(_trades([2.0, 2.0, 2.0])), 6.0)


class TestAvgR(unittest.TestCase):
    def test_known_mean(self):
        self.assertAlmostEqual(engine.avg_r(_trades([1.0, -0.5, 2.0])), 2.5 / 3.0)

    def test_empty_returns_zero_not_nan(self):
        self.assertEqual(engine.avg_r([]), 0.0)

    def test_single_trade_equals_that_trade(self):
        self.assertAlmostEqual(engine.avg_r(_trades([-2.0])), -2.0)

    def test_constant_r_list(self):
        self.assertAlmostEqual(engine.avg_r(_trades([0.25] * 10)), 0.25)


class TestSharpe(unittest.TestCase):
    def test_known_value_hand_computed(self):
        rs = [1.0, -1.0, 2.0, -1.0, 1.0]
        r = np.array(rs)
        expected = (r.mean() / r.std(ddof=1)) * np.sqrt(len(rs) / 2.0)
        self.assertAlmostEqual(engine.sharpe(_trades(rs), years=2.0), expected, places=9)

    def test_n_zero_returns_sentinel_not_crash(self):
        self.assertEqual(engine.sharpe([], years=1.0), engine.SHARPE_DEGENERATE_SENTINEL)

    def test_n_one_returns_sentinel_not_crash(self):
        self.assertEqual(engine.sharpe(_trades([1.0]), years=1.0), engine.SHARPE_DEGENERATE_SENTINEL)

    def test_constant_r_list_zero_std_returns_sentinel(self):
        # std(r) == 0 for a constant list regardless of sign - both must degrade to the sentinel,
        # never a ZeroDivisionError or NaN.
        self.assertEqual(engine.sharpe(_trades([0.5] * 10), years=3.0), engine.SHARPE_DEGENERATE_SENTINEL)
        self.assertEqual(engine.sharpe(_trades([-0.5] * 10), years=3.0), engine.SHARPE_DEGENERATE_SENTINEL)

    def test_missing_or_zero_years_returns_sentinel(self):
        rs = [1.0, -1.0, 2.0, -0.5]
        self.assertEqual(engine.sharpe(_trades(rs), years=None), engine.SHARPE_DEGENERATE_SENTINEL)
        self.assertEqual(engine.sharpe(_trades(rs), years=0.0), engine.SHARPE_DEGENERATE_SENTINEL)
        self.assertEqual(engine.sharpe(_trades(rs), years=-1.0), engine.SHARPE_DEGENERATE_SENTINEL)

    def test_never_returns_nan(self):
        for rs, years in [([], 1.0), ([1.0], 1.0), ([1.0] * 5, 1.0), ([1.0, -1.0, 2.0], None)]:
            score = engine.sharpe(_trades(rs), years=years)
            self.assertFalse(np.isnan(score))


class TestMaxDrawdownR(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(engine.max_drawdown_r([]), 0.0)

    def test_single_value_is_zero_regardless_of_sign(self):
        self.assertEqual(engine.max_drawdown_r([5.0]), 0.0)
        self.assertEqual(engine.max_drawdown_r([-5.0]), 0.0)

    def test_monotonic_increase_has_zero_drawdown(self):
        self.assertEqual(engine.max_drawdown_r([1.0, 1.0, 2.0, 0.5]), 0.0)

    def test_known_drawdown_by_construction(self):
        # cumsum: 3, 5, 2, 4, 1, 6 -> running max: 3, 5, 5, 5, 5, 6 -> dd: 0, 0, 3, 1, 4, 0
        # max drawdown = 4 (peak of 5 at index 1, trough of 1 at index 4)
        r = [3, 2, -3, 2, -3, 5]
        self.assertAlmostEqual(engine.max_drawdown_r(r), 4.0)

    def test_all_losers_drawdown_equals_all_but_first_loss(self):
        # cumsum monotonically decreasing from the first (least negative) point -> peak is always
        # the first point, drawdown at the end = r[0] - sum(r) = -(sum(r) - r[0])
        r = [-1.0] * 6
        expected = r[0] - sum(r)
        self.assertAlmostEqual(engine.max_drawdown_r(r), expected)


class TestCalmar(unittest.TestCase):
    def test_known_ratio(self):
        # total_r=8, max_drawdown=4 per TestMaxDrawdownR.test_known_drawdown_by_construction
        r = [3, 2, -3, 2, -3, 5]
        self.assertAlmostEqual(engine.calmar(_trades(r)), sum(r) / 4.0)

    def test_no_trades_is_neutral_zero(self):
        self.assertEqual(engine.calmar([]), 0.0)

    def test_single_winning_trade_is_positive_sentinel(self):
        self.assertEqual(engine.calmar(_trades([2.0])), engine.CALMAR_NO_DRAWDOWN_POSITIVE_SENTINEL)

    def test_single_losing_trade_is_negative_sentinel_not_masked_as_neutral(self):
        # a single loser has max_drawdown_r == 0 by TestMaxDrawdownR's single-value rule, but this
        # must NOT be reported as a neutral/good 0.0 - it's a real loss with an undefined ratio.
        self.assertEqual(engine.calmar(_trades([-2.0])), engine.CALMAR_NO_DRAWDOWN_NEGATIVE_SENTINEL)

    def test_all_winners_never_drawn_down_is_positive_sentinel(self):
        self.assertEqual(engine.calmar(_trades([1.0, 2.0, 0.5])), engine.CALMAR_NO_DRAWDOWN_POSITIVE_SENTINEL)

    def test_never_raises_or_nan(self):
        for rs in [[], [0.0], [1.0], [-1.0], [0.0, 0.0], [1.0, -1.0, 2.0, -3.0]]:
            score = engine.calmar(_trades(rs))
            self.assertFalse(np.isnan(score))


class TestWinRate(unittest.TestCase):
    def test_known_fraction(self):
        rs = [1.0, -1.0, 2.0, -1.0]   # 2 of 4 wins
        self.assertAlmostEqual(engine.win_rate(_trades(rs)), 0.5)

    def test_all_winners_is_one(self):
        self.assertAlmostEqual(engine.win_rate(_trades([1.0, 2.0, 0.5])), 1.0)

    def test_all_losers_is_zero(self):
        self.assertAlmostEqual(engine.win_rate(_trades([-1.0, -0.5])), 0.0)

    def test_breakeven_trade_does_not_count_as_a_win(self):
        self.assertAlmostEqual(engine.win_rate(_trades([0.0, 0.0, 1.0])), 1.0 / 3.0)

    def test_empty_is_zero(self):
        self.assertEqual(engine.win_rate([]), 0.0)

    def test_the_known_rauf_style_trap_high_win_rate_net_negative(self):
        # mirrors this project's real Day Trading Rauf finding: a >50% win rate that is still
        # net-negative because losers outsize winners - proves win_rate and total_r can and do
        # disagree on which cell looks "best", which is the whole point of this objective existing.
        rs = [0.3] * 6 + [-2.0] * 4   # 60% win rate, but total R = 1.8 - 8.0 = -6.2
        self.assertGreater(engine.win_rate(_trades(rs)), 0.5)
        self.assertLess(engine.total_r(_trades(rs)), 0.0)


def _dated_trades(pairs):
    """pairs: list of (r, date) - a "date" key alongside "r", since consistency_ratio (unlike
    every other OBJECTIVES function) needs each trade dated for period bucketing."""
    return [{"r": r, "date": d} for r, d in pairs]


class TestBucketTradesByPeriod(unittest.TestCase):
    """Period-bucketing correctness on hand-crafted trade lists with known R_p values - does NOT
    include zero-trade periods (that's consistency_ratio's own job via _all_period_keys, tested
    separately below)."""

    def test_monthly_bucketing_known_sums(self):
        import datetime
        trades = _dated_trades([
            (1.0, datetime.date(2020, 1, 5)),
            (0.5, datetime.date(2020, 1, 20)),
            (-2.0, datetime.date(2020, 3, 10)),
        ])
        buckets = engine.bucket_trades_by_period(trades, period="M")
        self.assertEqual(buckets, {(2020, 1): 1.5, (2020, 3): -2.0})

    def test_weekly_bucketing_known_sums(self):
        import datetime
        trades = _dated_trades([
            (1.0, datetime.date(2021, 1, 4)),    # Monday of ISO week 1
            (-0.5, datetime.date(2021, 1, 18)),  # ISO week 3
        ])
        buckets = engine.bucket_trades_by_period(trades, period="W")
        self.assertEqual(buckets, {(2021, 1): 1.0, (2021, 3): -0.5})

    def test_unknown_period_raises(self):
        import datetime
        with self.assertRaises(ValueError):
            engine.bucket_trades_by_period(_dated_trades([(1.0, datetime.date(2020, 1, 1))]), period="Q")


class TestConsistencyRatio(unittest.TestCase):
    """Period-bucketing correctness INCLUDING zero-trade periods, confidence-flag thresholds at
    the exact 12/24 boundaries, and nan-safety when std==0 - see consistency_ratio's own docstring
    for the honest ICIR-inspired-but-not-ICIR distinction this is testing the MECHANICS of, not the
    cross-sectional claim (there is none here)."""

    def _month(self, y, m, day=1):
        import datetime
        return datetime.date(y, m, day)

    def _multi_year_months(self, n, start_year=2020):
        """n consecutive calendar months as (year, month), starting Jan of start_year - handles
        year rollover so callers can safely build spans longer than 12 months."""
        out = []
        for i in range(n):
            y = start_year + i // 12
            m = i % 12 + 1
            out.append((y, m))
        return out

    def test_zero_trade_periods_are_included_with_r_p_zero(self):
        # trades in Jan and Mar only - Feb must still appear as a zero-R_p period, changing the
        # mean/std computed vs. a naive "only bucket occupied months" approach.
        trades = _dated_trades([(1.0, self._month(2020, 1)), (-2.0, self._month(2020, 3))])
        import statistics
        r_values_with_zero = [1.0, 0.0, -2.0]        # Jan, Feb (empty -> 0), Mar - what
                                                        # consistency_ratio is supposed to use
        r_values_without_zero = [1.0, -2.0]           # what a naive "only occupied buckets"
                                                        # approach would use instead - must differ
        self.assertNotEqual(statistics.mean(r_values_with_zero), statistics.mean(r_values_without_zero))
        self.assertNotEqual(statistics.stdev(r_values_with_zero), statistics.stdev(r_values_without_zero))

        result = engine.consistency_ratio(trades)
        self.assertEqual(result["n_periods"], 3)   # Jan, Feb, Mar - proves Feb (empty) was counted
        self.assertEqual(result["n_trades"], 2)
        # n_periods < 12 -> confidence "insufficient" and value forced to nan regardless of what
        # the raw ratio would have been (see docstring) - covered by the dedicated boundary tests
        # below; this test's own job is just proving the zero-period inclusion into R_p.
        self.assertEqual(result["confidence"], "insufficient")
        self.assertTrue(math.isnan(result["value"]))

    def test_a_strategy_active_every_period_beats_one_with_the_same_total_concentrated(self):
        # same total R (12.0) either spread evenly across 12 months (active every period, each
        # R_p=1.0) or concentrated into 2 big months (R_p=6.0 each) with 10 silent (zero-R_p)
        # months - the concentrated case's std is inflated by those 10 zeros relative to its own
        # mean far more than the even case's (which has zero std, the degenerate "too good"
        # guarded case - see test_std_zero_returns_nan_not_inf_or_zero). Demonstrated here directly
        # on the RAW (ungated) mean/std arithmetic, which is what actually differs; the gated
        # "value" for both is nan below n_periods=12/24 confidence thresholds regardless.
        even_r = [1.0] * 12
        concentrated_r = [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(sum(even_r), sum(concentrated_r))   # same total R by construction
        even_std = np.std(even_r, ddof=1)
        concentrated_std = np.std(concentrated_r, ddof=1)
        self.assertEqual(even_std, 0.0)          # perfectly consistent -> degenerate (guarded) case
        self.assertGreater(concentrated_std, 0.0)   # genuinely more volatile period-to-period

        even_trades = _dated_trades([(1.0, self._month(y, m)) for y, m in self._multi_year_months(12)])
        concentrated_trades = _dated_trades([(6.0, self._month(2020, 1)), (6.0, self._month(2020, 7))])
        even = engine.consistency_ratio(even_trades)
        concentrated = engine.consistency_ratio(concentrated_trades)
        self.assertEqual(even["n_periods"], 12)
        self.assertEqual(concentrated["n_periods"], 7)   # Jan..Jul span (the trades' own date range),
                                                          # only 2 of those 7 months occupied

    def test_std_zero_returns_nan_not_inf_or_zero(self):
        trades = _dated_trades([(1.0, self._month(y, m)) for y, m in self._multi_year_months(24)])
        result = engine.consistency_ratio(trades)
        self.assertEqual(result["n_periods"], 24)
        self.assertEqual(result["confidence"], "ok")
        self.assertTrue(math.isnan(result["value"]))
        self.assertNotEqual(result["value"], float("inf"))

    def test_confidence_boundary_at_exactly_12_periods_is_low_confidence(self):
        trades = _dated_trades([((-1) ** m, self._month(2020, m)) for m in range(1, 13)])
        result = engine.consistency_ratio(trades)
        self.assertEqual(result["n_periods"], 12)
        self.assertEqual(result["confidence"], "low_confidence")
        self.assertFalse(math.isnan(result["value"]))

    def test_confidence_boundary_at_11_periods_is_insufficient(self):
        trades = _dated_trades([((-1) ** m, self._month(2020, m)) for m in range(1, 12)])
        result = engine.consistency_ratio(trades)
        self.assertEqual(result["n_periods"], 11)
        self.assertEqual(result["confidence"], "insufficient")
        self.assertTrue(math.isnan(result["value"]))

    def test_confidence_boundary_at_exactly_24_periods_is_ok(self):
        dates_r = []
        for m in range(1, 25):
            y = 2020 + (m - 1) // 12
            mo = (m - 1) % 12 + 1
            dates_r.append(((-1) ** m * 1.5, self._month(y, mo)))
        trades = _dated_trades(dates_r)
        result = engine.consistency_ratio(trades)
        self.assertEqual(result["n_periods"], 24)
        self.assertEqual(result["confidence"], "ok")
        self.assertFalse(math.isnan(result["value"]))

    def test_confidence_boundary_at_23_periods_is_low_confidence(self):
        dates_r = []
        for m in range(1, 24):
            y = 2020 + (m - 1) // 12
            mo = (m - 1) % 12 + 1
            dates_r.append(((-1) ** m * 1.5, self._month(y, mo)))
        trades = _dated_trades(dates_r)
        result = engine.consistency_ratio(trades)
        self.assertEqual(result["n_periods"], 23)
        self.assertEqual(result["confidence"], "low_confidence")
        self.assertFalse(math.isnan(result["value"]))

    def test_empty_trades_is_insufficient_zero_periods(self):
        result = engine.consistency_ratio([])
        self.assertEqual(result["n_periods"], 0)
        self.assertEqual(result["n_trades"], 0)
        self.assertEqual(result["confidence"], "insufficient")
        self.assertTrue(math.isnan(result["value"]))

    def test_weekly_period_option(self):
        trades = _dated_trades([(1.0, self._month(2021, 1, 4)), (-1.0, self._month(2021, 1, 18))])
        result = engine.consistency_ratio(trades, period="W")
        self.assertEqual(result["n_periods"], 3)   # weeks 1, 2 (empty), 3
        self.assertEqual(result["n_trades"], 2)

    def test_returns_dict_not_float(self):
        result = engine.consistency_ratio(_dated_trades([(1.0, self._month(2020, 1))]))
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result), {"value", "n_periods", "n_trades", "confidence"})

    def test_registered_in_objectives(self):
        self.assertIs(engine.OBJECTIVES["consistency_ratio"], engine.consistency_ratio)


class TestZScore(unittest.TestCase):
    def test_known_value_hand_computed(self):
        rs = [1.0, -1.0, 2.0, -1.0, 1.0]
        r = np.array(rs)
        expected = r.mean() / (r.std(ddof=1) / np.sqrt(len(rs)))
        self.assertAlmostEqual(engine.zscore(_trades(rs)), expected, places=9)

    def test_is_not_the_old_broken_shortcut(self):
        # the OLD (buggy) formula elsewhere in this project was avg_r * sqrt(n), i.e. implicitly
        # assuming std(r) == 1. For a realistic stop/target-shaped R distribution (std well above
        # 1), the corrected zscore() here must be smaller in magnitude than that old shortcut.
        rs = [-1.0, -1.0, 2.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, 2.0] * 5   # 50 trades, std > 1
        r = np.array(rs)
        old_broken = r.mean() * np.sqrt(len(rs))
        corrected = engine.zscore(_trades(rs))
        self.assertGreater(r.std(ddof=1), 1.0)   # confirms this fixture actually exercises the bug
        self.assertLess(abs(corrected), abs(old_broken))

    def test_n_zero_and_n_one_are_neutral_zero_not_crash(self):
        self.assertEqual(engine.zscore([]), 0.0)
        self.assertEqual(engine.zscore(_trades([1.0])), 0.0)

    def test_constant_r_list_is_neutral_zero(self):
        self.assertEqual(engine.zscore(_trades([0.5] * 10)), 0.0)
        self.assertEqual(engine.zscore(_trades([-0.5] * 10)), 0.0)

    def test_never_nan(self):
        for rs in [[], [1.0], [1.0] * 5, [1.0, -1.0, 2.0, -0.5]]:
            self.assertFalse(np.isnan(engine.zscore(_trades(rs))))


class TestBonferroniAdjustedZThreshold(unittest.TestCase):
    def test_single_trial_reduces_to_ordinary_1_96_bar(self):
        self.assertAlmostEqual(engine.bonferroni_adjusted_z_threshold(1), 1.959963985, places=6)

    def test_threshold_rises_as_n_trials_grows(self):
        thresholds = [engine.bonferroni_adjusted_z_threshold(n) for n in [1, 2, 5, 16, 20, 100]]
        for prev, nxt in zip(thresholds, thresholds[1:]):
            self.assertLess(prev, nxt)

    def test_known_value_for_twenty_trials(self):
        # per_test_alpha = 1 - 0.95**(1/20); two-sided normal quantile at 1 - per_test_alpha/2
        import statistics
        per_test_alpha = 1.0 - 0.95 ** (1.0 / 20)
        expected = statistics.NormalDist().inv_cdf(1.0 - per_test_alpha / 2.0)
        self.assertAlmostEqual(engine.bonferroni_adjusted_z_threshold(20), expected, places=9)

    def test_invalid_n_trials_raises(self):
        with self.assertRaises(ValueError):
            engine.bonferroni_adjusted_z_threshold(0)
        with self.assertRaises(ValueError):
            engine.bonferroni_adjusted_z_threshold(-3)

    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            engine.bonferroni_adjusted_z_threshold(5, family_wise_alpha=0.0)
        with self.assertRaises(ValueError):
            engine.bonferroni_adjusted_z_threshold(5, family_wise_alpha=1.0)


class TestObjectivesRegistry(unittest.TestCase):
    def test_default_is_total_r(self):
        self.assertEqual(engine.DEFAULT_OBJECTIVE, "total_r")
        self.assertIs(engine.OBJECTIVES["total_r"], engine.total_r)

    def test_all_six_keys_present(self):
        # consistency_ratio was added on top of the original five - see TestConsistencyRatio below
        # and the LOCKBOX/decay-diagnostic additions at the bottom of this file for the rest of
        # this same follow-up pass.
        self.assertEqual(set(engine.OBJECTIVES),
                          {"total_r", "avg_r", "sharpe", "calmar", "win_rate", "consistency_ratio"})

    def test_get_objective_valid(self):
        self.assertIs(engine.get_objective("avg_r"), engine.avg_r)

    def test_get_objective_invalid_raises_clear_error(self):
        with self.assertRaises(KeyError):
            engine.get_objective("not_a_real_objective")


# ============================= grid_search =============================

class TestGridSearchEquivalence(unittest.TestCase):
    """Confirms grid_search generalizes the exact combos+order both companion
    scripts' hand-written double loops already produce, on the same
    2-parameter grid shape (STOP_BUFFER_PCT_GRID x FALLBACK_REWARD_RISK_GRID
    style)."""

    def test_same_combos_and_order_as_old_style_double_loop(self):
        sb_grid = [0.01, 0.02, 0.05, 0.1]
        frr_grid = [1.0, 1.5, 2.0, 2.5, 3.0]

        # the OLD style: a hand-written double loop, exactly as both companion scripts wrote it
        # before this refactor.
        old_style_combos = [(sb, frr) for sb in sb_grid for frr in frr_grid]

        def eval_fn(params):
            # deterministic score purely as a function of params, so equality of (params, score)
            # pairs is a real check, not a coincidence.
            return _trades([params["stop_buffer_pct"] * 100 + params["fallback_reward_risk"]])

        result = engine.grid_search({"stop_buffer_pct": sb_grid, "fallback_reward_risk": frr_grid},
                                     eval_fn, engine.total_r)

        new_combos = [(r["params"]["stop_buffer_pct"], r["params"]["fallback_reward_risk"])
                      for r in result["all"]]
        self.assertEqual(new_combos, old_style_combos)
        self.assertEqual(result["n_evals"], len(sb_grid) * len(frr_grid))

        # every score must match what the old-style loop would have computed directly, too.
        for r in result["all"]:
            expected = r["params"]["stop_buffer_pct"] * 100 + r["params"]["fallback_reward_risk"]
            self.assertAlmostEqual(r["score"], expected)

    def test_best_is_actually_the_max_score(self):
        param_grid = {"a": [1, 2, 3], "b": [10, 20]}

        def eval_fn(params):
            return _trades([params["a"] + params["b"]])

        result = engine.grid_search(param_grid, eval_fn, engine.total_r)
        self.assertEqual(result["best"]["params"], {"a": 3, "b": 20})
        self.assertAlmostEqual(result["best"]["score"], 23.0)

    def test_single_parameter_grid(self):
        param_grid = {"only": [5, 6, 7]}

        def eval_fn(params):
            return _trades([-params["only"]])

        result = engine.grid_search(param_grid, eval_fn, engine.total_r)
        self.assertEqual(result["n_evals"], 3)
        self.assertEqual(result["best"]["params"], {"only": 5})

    def test_empty_candidate_list_yields_empty_results(self):
        result = engine.grid_search({"a": [1, 2], "b": []}, lambda p: [], engine.total_r)
        self.assertEqual(result["n_evals"], 0)
        self.assertIsNone(result["best"])


# ============================= synthetic optimum surface =============================
#
# A 4-parameter, 5-values-each grid (5**4 = 625 combos) with a SINGLE known-best combo
# by construction: eval_fn returns one synthetic trade whose R is the negative Manhattan
# distance from a fixed target combo, so total_r is maximized (at exactly 0.0) only at the
# target and decreases smoothly (separably, one parameter at a time) everywhere else - a
# deliberately "findable" surface for a search strategy, not an adversarial one, since the
# actual claim under test is "these methods are more EFFICIENT on a space grid_search already
# covers", not "these methods solve arbitrarily hard landscapes".

_SURFACE_VALUES = [0, 1, 2, 3, 4]
_SURFACE_PARAM_GRID = {"p1": _SURFACE_VALUES, "p2": _SURFACE_VALUES, "p3": _SURFACE_VALUES, "p4": _SURFACE_VALUES}
_SURFACE_TARGET = {"p1": 2, "p2": 3, "p3": 1, "p4": 4}
_SURFACE_GRID_SIZE = len(_SURFACE_VALUES) ** len(_SURFACE_PARAM_GRID)   # 625


def _surface_eval_fn(params):
    distance = sum(abs(params[name] - _SURFACE_TARGET[name]) for name in _SURFACE_PARAM_GRID)
    return _trades([-float(distance)])


class TestGridSearchFindsTrueOptimum(unittest.TestCase):
    """Sanity check on the synthetic surface itself: exhaustive grid_search must find EXACTLY
    the constructed target (score 0.0), so the efficiency tests below have a known ground truth
    to compare bayesian_search/genetic_search against."""

    def test_grid_search_finds_exact_target(self):
        result = engine.grid_search(_SURFACE_PARAM_GRID, _surface_eval_fn, engine.total_r)
        self.assertEqual(result["n_evals"], _SURFACE_GRID_SIZE)
        self.assertAlmostEqual(result["best"]["score"], 0.0)
        self.assertEqual(result["best"]["params"], _SURFACE_TARGET)


class TestBayesianSearchEfficiency(unittest.TestCase):
    def test_finds_known_optimum_with_far_fewer_evals_than_grid(self):
        try:
            import optuna  # noqa: F401
        except ImportError:
            self.skipTest("optuna not installed in this environment")

        n_trials = 50
        result = engine.bayesian_search(_SURFACE_PARAM_GRID, _surface_eval_fn, engine.total_r,
                                         n_trials=n_trials, seed=1)
        self.assertIsNotNone(result)
        self.assertLessEqual(result["n_evals"], n_trials)

        # the actual efficiency claim: meaningfully fewer evaluations than grid_search's
        # exhaustive count over the identical space (625) - "meaningfully" pinned down here as
        # well under half.
        self.assertLess(result["n_evals"], _SURFACE_GRID_SIZE // 2)

        # tolerance: within 1 Manhattan-distance unit of the true optimum (score 0.0), i.e. score
        # >= -1.0 - documented here rather than requiring an exact hit, since TPE is stochastic.
        self.assertGreaterEqual(result["best"]["score"], -1.0)
        print(f"\n[bayesian efficiency] n_evals={result['n_evals']} (grid would need {_SURFACE_GRID_SIZE}), "
              f"best score={result['best']['score']}, best params={result['best']['params']}")


class TestGeneticSearchEfficiency(unittest.TestCase):
    def test_finds_known_optimum_with_far_fewer_evals_than_grid(self):
        result = engine.genetic_search(_SURFACE_PARAM_GRID, _surface_eval_fn, engine.total_r,
                                        population_size=15, generations=8, mutation_rate=0.2, seed=7)
        self.assertIsNotNone(result)
        raw_slots = 15 * 8
        self.assertLess(raw_slots, _SURFACE_GRID_SIZE // 2)   # even before dedup, meaningfully fewer
        self.assertLessEqual(result["n_evals"], raw_slots)     # dedup caching only ever reduces this further
        self.assertLess(result["n_evals"], _SURFACE_GRID_SIZE // 2)

        self.assertGreaterEqual(result["best"]["score"], -1.0)   # same documented tolerance as bayesian's test
        self.assertEqual(len(result["history"]), 8)
        # elitism means the best-seen score is monotonically non-decreasing generation over generation
        for prev, nxt in zip(result["history"], result["history"][1:]):
            self.assertLessEqual(prev, nxt)
        print(f"\n[genetic efficiency] n_evals={result['n_evals']} (grid would need {_SURFACE_GRID_SIZE}), "
              f"best score={result['best']['score']}, best params={result['best']['params']}, "
              f"history={result['history']}")

    def test_elitism_never_loses_the_best_individual(self):
        result = engine.genetic_search(_SURFACE_PARAM_GRID, _surface_eval_fn, engine.total_r,
                                        population_size=10, generations=12, mutation_rate=0.3, seed=99,
                                        elitism=True)
        # with elitism, the final generation's best score must be >= every earlier generation's best
        self.assertEqual(result["history"][-1], max(result["history"]))

    def test_single_parameter_param_grid_does_not_crash(self):
        param_grid = {"only": list(range(6))}

        def eval_fn(params):
            return _trades([-abs(params["only"] - 3)])

        result = engine.genetic_search(param_grid, eval_fn, engine.total_r,
                                        population_size=6, generations=4, seed=1)
        self.assertIsNotNone(result["best"])
        self.assertGreaterEqual(result["best"]["score"], -3)

    def test_deduplication_reduces_n_evals_below_raw_slots_on_small_space(self):
        # a tiny 2x2 space with a big population/many generations forces heavy repetition -
        # n_evals must be capped at 4 (the whole space), proving cache dedup actually works
        # rather than just being decorative.
        param_grid = {"a": [0, 1], "b": [0, 1]}

        def eval_fn(params):
            return _trades([float(params["a"] + params["b"])])

        result = engine.genetic_search(param_grid, eval_fn, engine.total_r,
                                        population_size=8, generations=10, mutation_rate=0.5, seed=3)
        self.assertLessEqual(result["n_evals"], 4)


# ============================= consistency_ratio wired through the search strategies =============================
#
# consistency_ratio uniquely returns a dict (not a float) - grid_search/bayesian_search/
# genetic_search must all still produce a plain-float "score" (via _normalize_score) for max()
# selection, preserve the full dict as "score_detail", and never let a nan/insufficient-data cell
# win a "best" comparison.

def _dated_trades_for_search(rs_dates):
    return [{"r": r, "date": d} for r, d in rs_dates]


class TestConsistencyRatioThroughSearchStrategies(unittest.TestCase):
    def test_grid_search_score_is_float_and_score_detail_preserves_the_dict(self):
        import datetime

        def eval_fn(params):
            # every combo gets the SAME 30-month, evenly-spread trade stream except the "a"
            # parameter scales R_p - enough periods (30 > 24) to get a real ("ok"-confidence,
            # non-nan) consistency_ratio value for every cell.
            scale = params["a"]
            return _dated_trades_for_search([
                (scale * (1.0 if m % 2 == 0 else -0.5), datetime.date(2020 + m // 12, m % 12 + 1, 1))
                for m in range(30)
            ])

        result = engine.grid_search({"a": [1, 2, 3]}, eval_fn, engine.consistency_ratio)
        for entry in result["all"]:
            self.assertIsInstance(entry["score"], float)
            self.assertIn("score_detail", entry)
            self.assertEqual(set(entry["score_detail"]), {"value", "n_periods", "n_trades", "confidence"})
        self.assertIsNotNone(result["best"])
        # scaling every R_p by a positive constant doesn't change mean/std's RATIO - all three
        # cells' consistency_ratio values (and therefore scores) should be equal here, so "best"
        # is just whichever grid_search's max() picked among ties (still must be a real float).
        self.assertFalse(math.isnan(result["best"]["score"]))

    def test_insufficient_data_cell_never_wins_against_a_real_cell(self):
        import datetime

        def eval_fn(params):
            if params["a"] == "few":
                # only 3 trades/periods - n_periods < 12 -> confidence "insufficient" -> value nan
                return _dated_trades_for_search([(1.0, datetime.date(2020, m, 1)) for m in (1, 2, 3)])
            # 30 periods, non-constant R_p -> a real, finite consistency_ratio value
            return _dated_trades_for_search([
                (5.0 if m % 2 == 0 else -1.0, datetime.date(2020 + m // 12, m % 12 + 1, 1))
                for m in range(30)
            ])

        result = engine.grid_search({"a": ["few", "many"]}, eval_fn, engine.consistency_ratio)
        few_entry = next(r for r in result["all"] if r["params"]["a"] == "few")
        many_entry = next(r for r in result["all"] if r["params"]["a"] == "many")

        self.assertEqual(few_entry["score"], float("-inf"))   # nan normalized to -inf, never nan itself
        self.assertFalse(math.isnan(few_entry["score"]))
        self.assertGreater(many_entry["score"], few_entry["score"])
        self.assertIs(result["best"], many_entry)   # the real cell must win, never the insufficient one

    def test_genetic_search_never_lets_nan_score_win(self):
        import datetime

        def eval_fn(params):
            if params["a"] == 0:
                return []   # 0 trades -> consistency_ratio "insufficient", value nan
            return _dated_trades_for_search([
                (1.0 if m % 2 == 0 else -1.0, datetime.date(2020 + m // 12, m % 12 + 1, 1))
                for m in range(24)
            ])

        result = engine.genetic_search({"a": [0, 1]}, eval_fn, engine.consistency_ratio,
                                        population_size=4, generations=3, seed=1)
        self.assertIsNotNone(result["best"])
        self.assertFalse(math.isnan(result["best"]["score"]))
        self.assertEqual(result["best"]["params"]["a"], 1)

    def test_bayesian_search_handles_dict_objective_without_crashing_optuna(self):
        try:
            import optuna  # noqa: F401
        except ImportError:
            self.skipTest("optuna not installed in this environment")
        import datetime

        def eval_fn(params):
            return _dated_trades_for_search([
                (params["a"] * (1.0 if m % 2 == 0 else -0.5), datetime.date(2020 + m // 12, m % 12 + 1, 1))
                for m in range(30)
            ])

        # Optuna's study.optimize() requires a plain numeric return from the objective callback -
        # a dict-valued objective_fn would break it if _normalize_score weren't applied BEFORE
        # returning from the trial closure, not just at "best" selection time.
        result = engine.bayesian_search({"a": [1, 2, 3]}, eval_fn, engine.consistency_ratio, n_trials=5, seed=1)
        self.assertIsNotNone(result)
        for entry in result["all"]:
            self.assertIsInstance(entry["score"], float)


# ============================= bayesian_search ImportError handling =============================

class TestBayesianSearchMissingOptuna(unittest.TestCase):
    def test_returns_none_and_does_not_crash_when_optuna_missing(self):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "optuna" or name.startswith("optuna."):
                raise ImportError("simulated: optuna not installed")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            result = engine.bayesian_search({"a": [1, 2]}, lambda p: _trades([1.0]), engine.total_r)
        self.assertIsNone(result)

    def test_run_search_falls_back_to_grid_when_bayesian_unavailable(self):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "optuna" or name.startswith("optuna."):
                raise ImportError("simulated: optuna not installed")
            return real_import(name, *args, **kwargs)

        param_grid = {"a": [1, 2, 3]}

        def eval_fn(params):
            return _trades([float(params["a"])])

        with mock.patch("builtins.__import__", side_effect=fake_import):
            result = engine.run_search("bayesian", param_grid, eval_fn, engine.total_r)

        # fell back to an exhaustive grid_search over the same 3-value space
        self.assertEqual(result["method"], "grid")
        self.assertEqual(result["n_evals"], 3)
        self.assertEqual(result["best"]["params"], {"a": 3})


class TestRunSearchDispatch(unittest.TestCase):
    def test_unknown_method_raises_value_error(self):
        with self.assertRaises(ValueError):
            engine.run_search("not_a_real_method", {"a": [1]}, lambda p: [], engine.total_r)

    def test_grid_dispatch(self):
        result = engine.run_search("grid", {"a": [1, 2]}, lambda p: _trades([float(p["a"])]), engine.total_r)
        self.assertEqual(result["method"], "grid")

    def test_genetic_dispatch(self):
        result = engine.run_search("genetic", {"a": [1, 2, 3]}, lambda p: _trades([float(p["a"])]),
                                    engine.total_r, population_size=4, generations=2, seed=1)
        self.assertEqual(result["method"], "genetic")


class TestPrintSearchComparison(unittest.TestCase):
    def test_runs_without_crashing_on_mixed_results(self):
        grid_result = engine.grid_search({"a": [1, 2]}, lambda p: _trades([float(p["a"])]), engine.total_r)
        none_result = None
        engine.print_search_comparison({"grid": grid_result, "bayesian": none_result})


# ============================= estimate_decay =============================

def _decay_fold_data(n_folds, true_phi, metric0=1.0, oos_years=1, trades_per_month=2, start_year=2019):
    """Builds `n_folds` synthetic fold entries, each with a 12-month OOS window whose avg_r per
    pooled month-since-fit follows metric_t = metric0 * true_phi**t EXACTLY (every trade within a
    given month is given the exact same R, so avg_r for that month equals metric_t exactly, with
    no sampling noise to fit around) - lets estimate_decay's recovered phi/half_life be checked
    against a KNOWN ground truth, not just "some plausible-looking number"."""
    fold_data = []
    for f in range(n_folds):
        oos_start = datetime.date(start_year + f, 1, 1)
        trades = []
        for t in range(1, 12 * oos_years + 1):
            value = metric0 * (true_phi ** t)
            month = ((oos_start.month - 1 + (t - 1)) % 12) + 1
            year = oos_start.year + (oos_start.month - 1 + (t - 1)) // 12
            d = datetime.date(year, month, 1)
            trades.extend([{"r": value, "date": d}] * trades_per_month)
        fold_data.append({"oos_start": oos_start, "oos_trades": trades})
    return fold_data


class TestPoolOosTradesByMonth(unittest.TestCase):
    def test_pools_across_folds_by_months_since_fit(self):
        fold_data = [
            {"oos_start": datetime.date(2020, 1, 1),
             "oos_trades": [{"r": 1.0, "date": datetime.date(2020, 1, 15)},   # month 1
                             {"r": 2.0, "date": datetime.date(2020, 2, 10)}]},  # month 2
            {"oos_start": datetime.date(2021, 1, 1),
             "oos_trades": [{"r": 5.0, "date": datetime.date(2021, 1, 20)}]},   # month 1
        ]
        pooled, n_folds_used = engine.pool_oos_trades_by_month(fold_data)
        self.assertEqual(n_folds_used, 2)
        self.assertEqual(len(pooled[1]), 2)   # month 1 trades pooled from BOTH folds
        self.assertEqual(len(pooled[2]), 1)
        self.assertEqual({t["r"] for t in pooled[1]}, {1.0, 5.0})

    def test_fold_with_no_oos_trades_does_not_count_toward_n_folds_used(self):
        fold_data = [
            {"oos_start": datetime.date(2020, 1, 1), "oos_trades": []},
            {"oos_start": datetime.date(2021, 1, 1),
             "oos_trades": [{"r": 1.0, "date": datetime.date(2021, 1, 5)}]},
        ]
        pooled, n_folds_used = engine.pool_oos_trades_by_month(fold_data)
        self.assertEqual(n_folds_used, 1)


class TestEstimateDecay(unittest.TestCase):
    def test_recovers_known_phi_and_half_life_within_tolerance(self):
        true_phi = 0.75
        fold_data = _decay_fold_data(n_folds=6, true_phi=true_phi, metric0=2.0)
        result = engine.estimate_decay(fold_data, metric="avg_r", min_folds=4)

        self.assertEqual(result["confidence"], "ok")
        self.assertEqual(result["n_folds_used"], 6)
        self.assertLess(result["slope"], 0.0)   # decaying series -> negative linear trend
        self.assertIsNotNone(result["phi"])
        # exact recovery (within floating-point tolerance) since every pooled point lies EXACTLY
        # on the true exponential curve by construction (see _decay_fold_data) - documented
        # tolerance: 1e-6 relative, comfortably tighter than any real-world use would need.
        self.assertAlmostEqual(result["phi"], true_phi, places=6)
        expected_half_life = math.log(0.5) / math.log(true_phi)
        self.assertAlmostEqual(result["half_life_months"], expected_half_life, places=6)

    def test_flat_series_has_no_reportable_half_life(self):
        # phi=1.0 (no decay at all) is OUTSIDE the open interval (0, 1) - half_life/phi must stay
        # None even though the linear slope is (near) zero, not a fabricated "infinite half-life".
        fold_data = _decay_fold_data(n_folds=6, true_phi=1.0, metric0=1.5)
        result = engine.estimate_decay(fold_data, metric="avg_r", min_folds=4)
        self.assertEqual(result["confidence"], "ok")
        self.assertIsNone(result["phi"])
        self.assertIsNone(result["half_life_months"])
        self.assertAlmostEqual(result["slope"], 0.0, places=6)

    def test_growth_series_has_no_reportable_half_life(self):
        # phi > 1 is growth, not decay - must not be reported as a "half-life" (which only makes
        # sense for genuine decay).
        fold_data = _decay_fold_data(n_folds=6, true_phi=1.3, metric0=0.5)
        result = engine.estimate_decay(fold_data, metric="avg_r", min_folds=4)
        self.assertIsNone(result["phi"])
        self.assertIsNone(result["half_life_months"])
        self.assertGreater(result["slope"], 0.0)   # linear slope still correctly shows the growth

    def test_min_folds_gate_blocks_too_few_folds(self):
        fold_data = _decay_fold_data(n_folds=3, true_phi=0.7, metric0=1.0)   # below default min_folds=4
        result = engine.estimate_decay(fold_data, metric="avg_r", min_folds=4)
        self.assertEqual(result["confidence"], "insufficient_folds")
        self.assertEqual(result["n_folds_used"], 3)
        self.assertIsNone(result["phi"])
        self.assertIsNone(result["half_life_months"])
        self.assertTrue(math.isnan(result["slope"]))

    def test_min_folds_gate_exactly_at_boundary_passes(self):
        fold_data = _decay_fold_data(n_folds=4, true_phi=0.7, metric0=1.0)
        result = engine.estimate_decay(fold_data, metric="avg_r", min_folds=4)
        self.assertEqual(result["confidence"], "ok")
        self.assertEqual(result["n_folds_used"], 4)

    def test_custom_min_folds_parameter_is_honored(self):
        fold_data = _decay_fold_data(n_folds=5, true_phi=0.7, metric0=1.0)
        result = engine.estimate_decay(fold_data, metric="avg_r", min_folds=6)
        self.assertEqual(result["confidence"], "insufficient_folds")

    def test_unknown_metric_raises_clear_error(self):
        fold_data = _decay_fold_data(n_folds=4, true_phi=0.7, metric0=1.0)
        with self.assertRaises(KeyError):
            engine.estimate_decay(fold_data, metric="not_a_real_metric", min_folds=4)

    def test_consistency_ratio_as_metric_runs_without_crashing(self):
        # a larger trades_per_month so each pooled month has enough trades for consistency_ratio
        # to not immediately degrade to "insufficient" within estimate_decay's own per-bucket call.
        fold_data = _decay_fold_data(n_folds=6, true_phi=0.8, metric0=1.0, trades_per_month=6)
        result = engine.estimate_decay(fold_data, metric="consistency_ratio", min_folds=4)
        self.assertIn(result["confidence"], ("ok", "insufficient_folds"))


# ============================= split_lockbox =============================

class TestSplitLockbox(unittest.TestCase):
    def test_default_twelve_months_datetime_inputs(self):
        fetch_start = datetime.datetime(2016, 1, 1)
        fetch_end = datetime.datetime(2025, 1, 1)
        search_start, search_end, lockbox_start, lockbox_end = engine.split_lockbox(fetch_start, fetch_end)
        self.assertEqual(search_start, fetch_start)
        self.assertEqual(search_end, datetime.datetime(2024, 1, 1))
        self.assertEqual(lockbox_start, datetime.datetime(2024, 1, 1))
        self.assertEqual(lockbox_end, fetch_end)
        self.assertEqual(search_end, lockbox_start)   # adjacency guarantee

    def test_date_inputs_preserve_type(self):
        fetch_start = datetime.date(2016, 1, 1)
        fetch_end = datetime.date(2025, 1, 1)
        search_start, search_end, lockbox_start, lockbox_end = engine.split_lockbox(fetch_start, fetch_end, 12)
        self.assertIsInstance(search_end, datetime.date)
        self.assertEqual(lockbox_start, datetime.date(2024, 1, 1))

    def test_six_month_lockbox(self):
        fetch_start = datetime.datetime(2020, 1, 1)
        fetch_end = datetime.datetime(2021, 1, 1)
        _, search_end, lockbox_start, _ = engine.split_lockbox(fetch_start, fetch_end, lockbox_months=6)
        self.assertEqual(search_end, datetime.datetime(2020, 7, 1))
        self.assertEqual(lockbox_start, datetime.datetime(2020, 7, 1))

    def test_lockbox_months_not_a_multiple_of_twelve(self):
        fetch_start = datetime.datetime(2016, 1, 1)
        fetch_end = datetime.datetime(2025, 3, 1)
        _, search_end, lockbox_start, lockbox_end = engine.split_lockbox(fetch_start, fetch_end, lockbox_months=15)
        self.assertEqual(lockbox_start, datetime.datetime(2023, 12, 1))
        self.assertEqual(lockbox_end, fetch_end)

    def test_day_of_month_rollover_clamped_to_valid_day(self):
        # Mar 31 minus 1 month has no "Feb 31" - must clamp to Feb's own last valid day.
        fetch_start = datetime.datetime(2019, 1, 1)
        fetch_end = datetime.datetime(2020, 3, 31)
        _, search_end, lockbox_start, _ = engine.split_lockbox(fetch_start, fetch_end, lockbox_months=1)
        self.assertEqual(lockbox_start, datetime.datetime(2020, 2, 29))   # 2020 is a leap year

    def test_zero_or_negative_lockbox_months_raises(self):
        fetch_start = datetime.datetime(2016, 1, 1)
        fetch_end = datetime.datetime(2025, 1, 1)
        with self.assertRaises(ValueError):
            engine.split_lockbox(fetch_start, fetch_end, lockbox_months=0)
        with self.assertRaises(ValueError):
            engine.split_lockbox(fetch_start, fetch_end, lockbox_months=-3)

    def test_lockbox_months_consuming_entire_range_raises(self):
        fetch_start = datetime.datetime(2016, 1, 1)
        fetch_end = datetime.datetime(2016, 6, 1)
        with self.assertRaises(ValueError):
            engine.split_lockbox(fetch_start, fetch_end, lockbox_months=12)

    def test_lockbox_months_consuming_more_than_the_entire_range_raises(self):
        fetch_start = datetime.datetime(2016, 1, 1)
        fetch_end = datetime.datetime(2017, 1, 1)
        with self.assertRaises(ValueError):
            engine.split_lockbox(fetch_start, fetch_end, lockbox_months=24)


# ============================= lockbox_confirm =============================

class TestLockboxConfirm(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ledger_path = os.path.join(self.tmpdir, "lockbox_ledger.json")
        self.lockbox_start = datetime.datetime(2024, 1, 1)
        self.lockbox_end = datetime.datetime(2025, 1, 1)

    def _passing_backtest_fn(self, start, end):
        # positive avg_r by construction; dated (lockbox_confirm's internal consistency_ratio
        # call needs every trade dated, same as every trade dict elsewhere in this project).
        dates = [datetime.date(2024, 1, 5), datetime.date(2024, 3, 5), datetime.date(2024, 5, 5),
                 datetime.date(2024, 7, 5), datetime.date(2024, 9, 5)]
        return _dated_trades_for_search(list(zip([0.5, -0.2, 0.8, -0.1, 0.6], dates)))

    def _failing_backtest_fn(self, start, end):
        dates = [datetime.date(2024, 1, 5), datetime.date(2024, 3, 5), datetime.date(2024, 5, 5)]
        return _dated_trades_for_search(list(zip([-0.5, -0.2, -0.8], dates)))

    def test_first_call_runs_and_appends_ledger_record(self):
        result = engine.lockbox_confirm("strat_a", {"x": 1}, self._passing_backtest_fn,
                                         self.lockbox_start, self.lockbox_end, ledger_path=self.ledger_path)
        self.assertTrue(result["passed"])
        self.assertEqual(result["n_trades"], 5)
        self.assertAlmostEqual(result["avg_r"], sum([0.5, -0.2, 0.8, -0.1, 0.6]) / 5)

        self.assertTrue(os.path.exists(self.ledger_path))
        with open(self.ledger_path) as f:
            records = json.load(f)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["strategy_id"], "strat_a")
        self.assertTrue(records[0]["passed"])

    def test_negative_avg_r_fails(self):
        result = engine.lockbox_confirm("strat_b", {"x": 1}, self._failing_backtest_fn,
                                         self.lockbox_start, self.lockbox_end, ledger_path=self.ledger_path)
        self.assertFalse(result["passed"])

    def test_second_call_for_same_strategy_id_raises_and_does_not_call_backtest_fn(self):
        engine.lockbox_confirm("strat_c", {"x": 1}, self._passing_backtest_fn,
                                self.lockbox_start, self.lockbox_end, ledger_path=self.ledger_path)

        calls = []

        def tracking_backtest_fn(start, end):
            calls.append((start, end))
            return self._passing_backtest_fn(start, end)

        with self.assertRaises(engine.LockboxAlreadyUsedError):
            engine.lockbox_confirm("strat_c", {"x": 2}, tracking_backtest_fn,
                                    self.lockbox_start, self.lockbox_end, ledger_path=self.ledger_path)
        self.assertEqual(calls, [])   # backtest_fn must never even be invoked on the refused attempt

    def test_ledger_persists_across_two_separate_calls_not_just_in_memory(self):
        # first call - fresh process-level state each time (no shared object between these two
        # calls beyond the ledger FILE itself), proving persistence is genuinely on disk.
        engine.lockbox_confirm("strat_d", {"x": 1}, self._passing_backtest_fn,
                                self.lockbox_start, self.lockbox_end, ledger_path=self.ledger_path)
        with open(self.ledger_path) as f:
            records_after_first = json.load(f)
        self.assertEqual(len(records_after_first), 1)

        # a SECOND, unrelated strategy_id's call must see the FIRST record already on disk (by
        # re-reading the file, not an in-memory cache) and append to it, not overwrite it.
        engine.lockbox_confirm("strat_e", {"x": 1}, self._passing_backtest_fn,
                                self.lockbox_start, self.lockbox_end, ledger_path=self.ledger_path)
        with open(self.ledger_path) as f:
            records_after_second = json.load(f)
        self.assertEqual(len(records_after_second), 2)
        self.assertEqual({r["strategy_id"] for r in records_after_second}, {"strat_d", "strat_e"})

    def test_failed_attempt_still_counts_as_used_up(self):
        engine.lockbox_confirm("strat_f", {"x": 1}, self._failing_backtest_fn,
                                self.lockbox_start, self.lockbox_end, ledger_path=self.ledger_path)
        with self.assertRaises(engine.LockboxAlreadyUsedError):
            engine.lockbox_confirm("strat_f", {"x": 1}, self._passing_backtest_fn,
                                    self.lockbox_start, self.lockbox_end, ledger_path=self.ledger_path)

    def test_insufficient_consistency_data_does_not_veto_a_positive_result(self):
        # too few trades for consistency_ratio to report anything ("insufficient") must not, by
        # itself, fail an otherwise-positive-avg-R lockbox result.
        def backtest_fn(start, end):
            return _dated_trades_for_search([(0.5, datetime.date(2024, 1, 5)),
                                              (0.3, datetime.date(2024, 1, 20))])
        result = engine.lockbox_confirm("strat_g", {"x": 1}, backtest_fn,
                                         self.lockbox_start, self.lockbox_end, ledger_path=self.ledger_path)
        self.assertEqual(result["consistency"]["confidence"], "insufficient")
        self.assertTrue(result["passed"])

    def test_zero_trades_does_not_pass(self):
        result = engine.lockbox_confirm("strat_h", {"x": 1}, lambda s, e: [],
                                         self.lockbox_start, self.lockbox_end, ledger_path=self.ledger_path)
        self.assertEqual(result["n_trades"], 0)
        self.assertFalse(result["passed"])   # avg_r is 0.0, not > 0.0


# ============================= WALK-FORWARD FOLDS =============================

class TestWalkForwardFolds(unittest.TestCase):
    def test_rolling_folds_step_forward_and_stop_before_overrunning_fetch_end(self):
        folds = list(engine.walk_forward_folds(datetime.date(2016, 1, 1), datetime.date(2020, 1, 1),
                                                 is_years=1, oos_years=1, step_years=1))
        self.assertEqual(folds, [
            (datetime.date(2016, 1, 1), datetime.date(2017, 1, 1), datetime.date(2017, 1, 1), datetime.date(2018, 1, 1)),
            (datetime.date(2017, 1, 1), datetime.date(2018, 1, 1), datetime.date(2018, 1, 1), datetime.date(2019, 1, 1)),
            (datetime.date(2018, 1, 1), datetime.date(2019, 1, 1), datetime.date(2019, 1, 1), datetime.date(2020, 1, 1)),
        ])

    def test_accepts_datetime_not_just_date(self):
        folds = list(engine.walk_forward_folds(datetime.datetime(2016, 1, 1), datetime.datetime(2019, 1, 1),
                                                 is_years=2, oos_years=1, step_years=1))
        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0], (datetime.date(2016, 1, 1), datetime.date(2018, 1, 1),
                                     datetime.date(2018, 1, 1), datetime.date(2019, 1, 1)))

    def test_no_folds_when_range_too_short(self):
        folds = list(engine.walk_forward_folds(datetime.date(2016, 1, 1), datetime.date(2017, 1, 1),
                                                 is_years=1, oos_years=1, step_years=1))
        self.assertEqual(folds, [])


# ============================= MONTE CARLO =============================

class TestMonteCarloBootstrap(unittest.TestCase):
    def test_all_winners_never_shows_a_negative_path(self):
        result = engine.monte_carlo_bootstrap([1.0, 2.0, 1.5] * 20, n_iter=500, seed=1)
        self.assertGreater(result["total_r_p05"], 0)
        self.assertEqual(result["p_total_r_leq_0"], 0.0)

    def test_all_losers_never_shows_a_positive_path(self):
        result = engine.monte_carlo_bootstrap([-1.0, -2.0, -0.5] * 20, n_iter=500, seed=1)
        self.assertLess(result["total_r_p95"], 0)
        self.assertEqual(result["p_total_r_leq_0"], 1.0)

    def test_empty_input_returns_nans_not_a_crash(self):
        result = engine.monte_carlo_bootstrap([], n_iter=100)
        self.assertTrue(math.isnan(result["total_r_p50"]))

    def test_deterministic_with_a_seed(self):
        r1 = engine.monte_carlo_bootstrap([0.5, -1.0, 2.0, -0.3], n_iter=200, seed=7)
        r2 = engine.monte_carlo_bootstrap([0.5, -1.0, 2.0, -0.3], n_iter=200, seed=7)
        self.assertEqual(r1, r2)


class TestMonteCarloShuffle(unittest.TestCase):
    def test_total_r_is_identical_across_every_path_a_permutation_cant_change_the_sum(self):
        result = engine.monte_carlo_shuffle([0.5, -1.0, 2.0, -0.3], n_iter=300, seed=3)
        expected_total = sum([0.5, -1.0, 2.0, -0.3])
        self.assertAlmostEqual(result["total_r_p05"], expected_total, places=6)
        self.assertAlmostEqual(result["total_r_p95"], expected_total, places=6)

    def test_empty_input_returns_nans_not_a_crash(self):
        result = engine.monte_carlo_shuffle([], n_iter=100)
        self.assertTrue(math.isnan(result["max_dd_p50"]))


# ============================= N-DIMENSIONAL CLUSTER / PLATEAU CHECK =============================

def _grid_result_entry(params, rs):
    return {"params": params, "trades": _trades(rs), "score": engine.total_r(_trades(rs))}


class TestBuildClusterFeatures(unittest.TestCase):
    def test_one_parameter_grid_produces_two_columns(self):
        param_grid = {"x": [5, 10, 15]}
        all_results = [_grid_result_entry({"x": v}, [0.1 * i]) for i, v in enumerate([5, 10, 15], start=1)]
        features = engine.build_cluster_features(all_results, param_grid)
        self.assertEqual(features.shape, (3, 2))
        # x=5 is the first grid point -> normalized position 0.0; x=15 is the last -> 1.0
        self.assertAlmostEqual(features[0][0], 0.0)
        self.assertAlmostEqual(features[2][0], 1.0)

    def test_two_parameter_grid_produces_three_columns(self):
        param_grid = {"a": [1, 2], "b": [10, 20, 30]}
        all_results = [_grid_result_entry({"a": a, "b": b}, [1.0]) for a in [1, 2] for b in [10, 20, 30]]
        features = engine.build_cluster_features(all_results, param_grid)
        self.assertEqual(features.shape, (6, 3))


class TestNeighborPlateauCheck(unittest.TestCase):
    def test_one_parameter_plateau_when_neighbors_are_decent(self):
        # avg R/trade: 0.05, 0.10 (peak), 0.09 - both neighbors of the peak are the same sign and
        # within decent_ratio of the peak, so this is a plateau, not an isolated spike.
        param_grid = {"x": [5, 10, 15, 20]}
        all_results = [
            _grid_result_entry({"x": 5}, [0.05]),
            _grid_result_entry({"x": 10}, [0.10]),
            _grid_result_entry({"x": 15}, [0.09]),
            _grid_result_entry({"x": 20}, [-0.5]),
        ]
        result = engine.neighbor_plateau_check(all_results, param_grid, rank_by="avg_r")
        self.assertEqual(result["best_params"], {"x": 10})
        self.assertEqual(len(result["neighbors"]), 2)   # x=10's only in-grid neighbors are x=5 and x=15
        self.assertEqual(result["verdict"], "PLATEAU")

    def test_one_parameter_isolated_spike_when_neighbors_are_weak(self):
        param_grid = {"x": [5, 10, 15]}
        all_results = [
            _grid_result_entry({"x": 5}, [0.001]),
            _grid_result_entry({"x": 10}, [1.0]),
            _grid_result_entry({"x": 15}, [-0.5]),
        ]
        result = engine.neighbor_plateau_check(all_results, param_grid, rank_by="avg_r")
        self.assertEqual(result["verdict"], "ISOLATED SPIKE / overfit warning")

    def test_single_cell_grid_has_no_neighbors(self):
        param_grid = {"x": [10]}
        all_results = [_grid_result_entry({"x": 10}, [1.0])]
        result = engine.neighbor_plateau_check(all_results, param_grid)
        self.assertEqual(result["neighbors"], [])
        self.assertIn("no in-grid neighbors", result["verdict"])

    def test_two_parameter_grid_checks_up_to_four_neighbors(self):
        # matches both existing companion scripts' original up/down/left/right convention - the
        # center of a 3x3 grid has exactly 4 immediate neighbors (one step per axis, per direction).
        param_grid = {"a": [1, 2, 3], "b": [10, 20, 30]}
        all_results = [_grid_result_entry({"a": a, "b": b}, [0.1]) for a in [1, 2, 3] for b in [10, 20, 30]]
        result = engine.neighbor_plateau_check(all_results, param_grid, rank_by="avg_r")
        self.assertEqual(result["best_params"], {"a": 1, "b": 10})   # first max() match, all tied at 0.1
        self.assertLessEqual(len(result["neighbors"]), 4)


class TestRunClusterAnalysis(unittest.TestCase):
    def test_returns_none_without_crashing_when_sklearn_missing(self):
        param_grid = {"x": [5, 10, 15]}
        all_results = [_grid_result_entry({"x": v}, [0.1]) for v in [5, 10, 15]]
        with mock.patch.dict("sys.modules", {"sklearn": None, "sklearn.cluster": None}):
            result = engine.run_cluster_analysis(all_results, param_grid)
        self.assertIsNone(result)

    def test_finds_a_cluster_containing_the_best_cell_when_sklearn_available(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")
        param_grid = {"x": [5, 10, 15, 20, 25]}
        all_results = [_grid_result_entry({"x": v}, [r])
                        for v, r in zip([5, 10, 15, 20, 25], [0.1, 0.2, 0.8, 0.2, 0.1])]
        result = engine.run_cluster_analysis(all_results, param_grid, k=2)
        self.assertIsNotNone(result)
        best_entry = max(all_results, key=lambda e: engine.total_r(e["trades"]))
        self.assertIn(best_entry, result["best_cluster_members"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
