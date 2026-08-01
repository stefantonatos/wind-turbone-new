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

import os
import sys
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

    def test_all_five_keys_present(self):
        self.assertEqual(set(engine.OBJECTIVES), {"total_r", "avg_r", "sharpe", "calmar", "win_rate"})

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
