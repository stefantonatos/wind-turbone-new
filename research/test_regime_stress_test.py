# Unit tests + a granularity-convergence check + two mocked real-strategy
# end-to-end smoke runs for regime_stress_test.py.
#
# Covers, per this project's rigor conventions:
#   1. fit_regime_model recovers something close to the true parameters of a
#      KNOWN 2-regime synthetic return process (we generate it, so we know
#      the true means/vols/transition probabilities), and its BIC-based
#      regime-count selection actually picks 2 for this obviously-2-regime
#      input rather than just always returning max_regimes - a sanity check
#      that the selection logic is doing real work, not defaulting upward.
#   2. fit_regime_model's ImportError handling - hmmlearn missing must print
#      a clear skip message and return None, never crash, matching this
#      project's existing sklearn/matplotlib/optuna-missing pattern.
#   3. generate_synthetic_ohlc_path's structural invariants (Low <= Open,
#      Close <= High always; each bar's open anchored exactly to the
#      previous bar's close - no lookahead, no gap) plus a genuine
#      granularity-convergence check, reusing the EXACT diagnostic pattern
#      already validated in research/test_rsi_mean_reversion_dukascopy_
#      backtest.py's own granularity-convergence test (same z-score-of-
#      apparent-edge methodology, same real strategy function, same
#      averaged-across-finer-granularities comparison since this project's
#      RSI strategy - unlike Bollinger's - is already documented there as
#      having a noisier, less strictly-monotonic coarse-granularity
#      artifact): confirms the "single-bar edge" artifact this project has
#      already been burned by once shrinks hard as sub_steps_per_bar grows,
#      i.e. it's a construction artifact, not a real signal, in THIS
#      module's regime-driven construction too.
#   4. run_regime_stress_test's percentile/outlier-flagging logic
#      (_percentile_and_verdict), unit tested directly against a fully
#      controlled synthetic distribution where the correct percentile and
#      verdict are known exactly, for both "higher is better" (total_r,
#      avg_r) and "lower is better" (max_drawdown) metrics.
#   5. TWO full mocked end-to-end smoke tests, one per real, already-existing
#      strategy wired in as this module's proof-of-concept -
#      ict_po3_forex_dukascopy_backtest.backtest_instrument(label, df) and
#      rsi_mean_reversion_dukascopy_backtest.backtest_instrument(label, df) -
#      IMPORTED, not reimplemented, both called through run_regime_stress_test
#      against regime-generated synthetic data end-to-end, confirming the
#      whole pipeline (fit -> simulate n_paths -> re-run the real strategy on
#      each -> percentile/verdict) produces well-formed output without
#      crashing. Deliberately NOT a real Dukascopy download - this project's
#      convention is to validate via unit tests plus a synthetic smoke run.
#
# Both wired-in strategies were picked because their backtest_instrument(label, df)
# functions are clean, minimal callback shapes (label + OHLC DataFrame in,
# a list of trade dicts with an "r" key out) with no other required
# arguments - exactly the shape run_regime_stress_test's backtest_fn callback
# needs, and both already fetch their own data separately from
# backtest_instrument itself, so passing a synthetic df straight in is a
# genuine, unmodified call into each script's real strategy logic.
#
# Run with:  python -m pytest research/test_regime_stress_test.py -v
# or:        python research/test_regime_stress_test.py

import importlib.util
import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_THIS_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rst = _load("regime_stress_test", "regime_stress_test.py")
po3 = _load("ict_po3_forex_dukascopy_backtest", "ict_po3_forex_dukascopy_backtest.py")
rsi_mod = _load("rsi_mean_reversion_dukascopy_backtest", "rsi_mean_reversion_dukascopy_backtest.py")

try:
    import hmmlearn  # noqa: F401
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

_skip_no_hmmlearn = unittest.skipUnless(
    HMMLEARN_AVAILABLE, "hmmlearn not installed in this environment - fit_regime_model's own "
                        "ImportError handling is tested separately via a simulated import failure")


# ============================= synthetic data helpers =============================

def _simulate_known_2regime_returns(n, seed, trans, means, stds):
    """Generates a return series from a KNOWN, caller-specified 2-state Markov-switching
    process (fixed transition matrix, fixed per-state mean/std) - the ground truth this test
    module's fit_regime_model recovery test checks the fit against. Independent of
    generate_synthetic_ohlc_path/fit_regime_model - built directly from numpy so there is no
    circular dependency between "what we're testing" and "the data we test it on"."""
    rng = np.random.default_rng(seed)
    state = 0
    states = np.empty(n, dtype=int)
    for i in range(n):
        if i > 0:
            state = rng.choice(2, p=trans[state])
        states[i] = state
    returns = np.array([rng.normal(means[s], stds[s]) for s in states])
    return returns, states


def _flat_regime_model(std, mean=0.0):
    """A hand-built, single-state 'regime model' dict of the exact shape fit_regime_model
    returns (n_states/means/stds/transmat/startprob) - used to test
    generate_synthetic_ohlc_path in isolation from HMM fitting, with a fair, zero-drift
    process (matching the premise the granularity-convergence check below relies on: a
    process with no true edge at any resolution)."""
    return {
        "n_states": 1,
        "means": np.array([mean]),
        "stds": np.array([std]),
        "transmat": np.array([[1.0]]),
        "startprob": np.array([1.0]),
    }


def _two_state_regime_model(means, stds, trans, startprob=(0.5, 0.5)):
    return {
        "n_states": 2,
        "means": np.asarray(means, dtype=float),
        "stds": np.asarray(stds, dtype=float),
        "transmat": np.asarray(trans, dtype=float),
        "startprob": np.asarray(startprob, dtype=float),
    }


# ============================= fit_regime_model =============================

@_skip_no_hmmlearn
class TestFitRegimeModelRecoversKnownProcess(unittest.TestCase):
    """The core honesty check for fit_regime_model: fit it to a return series generated from a
    KNOWN 2-regime process and confirm the recovered means/vols/transition persistence are
    close to the ground truth - not just "a model with 2 states came out", but a model whose
    parameters are actually recognizable as the process that generated the data."""

    @classmethod
    def setUpClass(cls):
        cls.true_trans = np.array([[0.97, 0.03], [0.04, 0.96]])
        cls.true_means = [0.0015, -0.0010]     # regime 0: calm/trending up, regime 1: crisis/high-vol down-drift
        cls.true_stds = [0.0040, 0.0180]
        cls.returns, cls.true_states = _simulate_known_2regime_returns(
            6000, seed=7, trans=cls.true_trans, means=cls.true_means, stds=cls.true_stds)
        cls.fit = rst.fit_regime_model(cls.returns, n_regimes=2, seed=1, n_init=6)

    def test_returns_a_result_not_none(self):
        self.assertIsNotNone(self.fit, "hmmlearn is installed in this environment - fit_regime_model "
                                        "must not skip")
        self.assertEqual(self.fit["n_states"], 2)

    def test_recovers_low_vol_and_high_vol_means_and_stds_within_tolerance(self):
        # sort BOTH recovered and true regimes by std ascending before comparing, since hmmlearn
        # state indices are not guaranteed to line up with the order the true process was defined in
        order = self.fit["vol_rank_order"]
        recovered_means = self.fit["means"][order]
        recovered_stds = self.fit["stds"][order]
        true_order = np.argsort(self.true_stds)
        true_means_sorted = np.array(self.true_means)[true_order]
        true_stds_sorted = np.array(self.true_stds)[true_order]

        np.testing.assert_allclose(recovered_stds, true_stds_sorted, rtol=0.15)
        # means are noisier to recover than stds at this sample size - looser but still
        # meaningful tolerance, and same-sign is checked explicitly (a flipped-sign mean would
        # be a real recovery failure, not just estimation noise)
        for recovered, true in zip(recovered_means, true_means_sorted):
            self.assertEqual(np.sign(recovered), np.sign(true),
                              f"recovered mean {recovered:+.5f} has the wrong sign vs true {true:+.5f}")
        np.testing.assert_allclose(recovered_means, true_means_sorted, atol=0.0012)

    def test_recovers_high_persistence_transition_matrix(self):
        # both true diagonal entries are >= 0.96 (a persistent, regime-switching-not-i.i.d.
        # process) - confirm the fit recovered similarly high self-transition probabilities,
        # not something close to i.i.d. (diagonal ~= 1/n_states)
        transmat = self.fit["transmat"]
        for state_idx in range(2):
            self.assertGreater(transmat[state_idx, state_idx], 0.85,
                                "fitted transition matrix lost the true process's high persistence")

    def test_regime_labels_and_summary_are_human_readable(self):
        self.assertEqual(set(self.fit["state_labels"].values()), {"low-vol", "high-vol"})
        self.assertIn("low-vol", self.fit["summary"])
        self.assertIn("high-vol", self.fit["summary"])
        # never expose a bare, unlabeled state index as the only identifier in the summary
        self.assertIn("mean=", self.fit["summary"])
        self.assertIn("std=", self.fit["summary"])


@_skip_no_hmmlearn
class TestFitRegimeModelBicSelection(unittest.TestCase):
    """Confirms BIC-based regime-count selection (n_regimes=None) actually picks 2 for this
    obviously-2-regime synthetic input, scanning 2..4 - i.e. the selection logic is doing real
    comparative work and isn't just always returning max_regimes."""

    def test_bic_selects_two_regimes_not_max_regimes(self):
        trans = np.array([[0.97, 0.03], [0.04, 0.96]])
        returns, _ = _simulate_known_2regime_returns(
            6000, seed=7, trans=trans, means=[0.0015, -0.0010], stds=[0.0040, 0.0180])
        fit = rst.fit_regime_model(returns, n_regimes=None, max_regimes=4, seed=1, n_init=5)
        self.assertEqual(fit["n_states"], 2, f"expected BIC to pick 2 regimes, got {fit['n_states']} "
                                              f"(bic_scan={fit['bic_scan']})")
        self.assertIsNotNone(fit["bic_scan"])
        self.assertEqual(set(fit["bic_scan"].keys()), {2, 3, 4})
        # the actual selection criterion: 2's BIC must be the strict minimum across the scan
        self.assertEqual(min(fit["bic_scan"], key=fit["bic_scan"].get), 2)


class TestFitRegimeModelMissingDependency(unittest.TestCase):
    """hmmlearn missing must never crash a calling script - a simulated ImportError (rather than
    requiring an environment that genuinely lacks hmmlearn) confirms fit_regime_model prints a
    clear skip message and returns None, matching this project's existing sklearn/matplotlib/
    optuna-missing handling pattern (see research/optimization_engine.py's bayesian_search)."""

    def test_missing_hmmlearn_returns_none_without_crashing(self):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "hmmlearn" or name.startswith("hmmlearn."):
                raise ImportError("simulated missing hmmlearn")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            result = rst.fit_regime_model(np.random.default_rng(0).normal(0, 0.001, 500))
        self.assertIsNone(result)


# ============================= generate_synthetic_ohlc_path =============================

class TestGenerateSyntheticOhlcPathInvariants(unittest.TestCase):
    """Structural, no-lookahead-discipline checks that must hold for ANY regime model, any
    seed: Low <= Open, Close <= High on every bar, and each bar's open anchored EXACTLY to the
    previous bar's actual close (no gap, no lookahead) - the same discipline this project's
    other synthetic-OHLC test suites already enforce."""

    def test_low_high_bracket_open_and_close_on_every_bar(self):
        regime_model = _two_state_regime_model(means=[0.0002, -0.0004], stds=[0.0006, 0.0035],
                                                trans=[[0.98, 0.02], [0.05, 0.95]])
        for seed in range(5):
            rng = np.random.default_rng(seed)
            df = rst.generate_synthetic_ohlc_path(regime_model, n_bars=1500, bar_interval_minutes=5,
                                                    start_price=1.1000, sub_steps_per_bar=25, rng=rng)
            self.assertTrue((df["Low"] <= df["Open"]).all())
            self.assertTrue((df["Low"] <= df["Close"]).all())
            self.assertTrue((df["High"] >= df["Open"]).all())
            self.assertTrue((df["High"] >= df["Close"]).all())
            self.assertTrue((df["Low"] <= df["High"]).all())
            self.assertTrue(np.isfinite(df.to_numpy()).all())

    def test_open_anchored_exactly_to_previous_close(self):
        regime_model = _flat_regime_model(std=0.0008)
        rng = np.random.default_rng(3)
        df = rst.generate_synthetic_ohlc_path(regime_model, n_bars=500, bar_interval_minutes=5,
                                                start_price=1.2000, sub_steps_per_bar=10, rng=rng)
        np.testing.assert_allclose(df["Open"].to_numpy()[1:], df["Close"].to_numpy()[:-1])
        self.assertEqual(df["Open"].iloc[0], 1.2000)

    def test_output_shape_matches_project_ohlc_convention(self):
        regime_model = _flat_regime_model(std=0.0005)
        df = rst.generate_synthetic_ohlc_path(regime_model, n_bars=300, bar_interval_minutes=5,
                                                start_price=1.1, sub_steps_per_bar=8,
                                                rng=np.random.default_rng(0))
        self.assertEqual(list(df.columns[:4]), ["Open", "High", "Low", "Close"])
        self.assertEqual(len(df), 300)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertIsNotNone(df.index.tz)

    def test_all_prices_strictly_positive(self):
        # log-space compounding (see this module's header) must never let price cross zero,
        # unlike a naive additive-noise construction could over a long enough series
        regime_model = _two_state_regime_model(means=[0.0, 0.0], stds=[0.02, 0.06], trans=[[0.9, 0.1], [0.1, 0.9]])
        rng = np.random.default_rng(9)
        df = rst.generate_synthetic_ohlc_path(regime_model, n_bars=4000, bar_interval_minutes=5,
                                                start_price=1.1, sub_steps_per_bar=5, rng=rng)
        self.assertTrue((df.to_numpy() > 0).all())


class TestGranularityConvergence(unittest.TestCase):
    """Reuses the EXACT diagnostic pattern already validated in
    research/test_rsi_mean_reversion_dukascopy_backtest.py's own granularity-convergence test
    (same z-score-of-apparent-edge methodology, same real strategy function
    rsi_mean_reversion_dukascopy_backtest.backtest_instrument, same averaged-across-finer-
    granularities comparison and the same thresholds that test file's own header explains are
    appropriate for THIS particular strategy's noisier, less strictly-monotonic coupling to
    single-bar wick geometry) - applied to THIS module's regime-driven construction instead of
    that file's own build_synthetic_ohlc. A single, zero-drift, single-state "regime" is used so
    the underlying process has no true edge at any resolution by construction: any apparent
    edge measured is necessarily a construction artifact, and it must shrink hard as
    sub_steps_per_bar grows, confirming generate_synthetic_ohlc_path avoids the single-step
    close-plus-wick-noise trap this project has already been burned by once (see this module's
    header)."""

    def test_apparent_edge_shrinks_as_substep_count_increases(self):
        n_bars = 10_000
        seeds = range(20)
        regime_model = _flat_regime_model(std=0.0005)   # bar_std_pct=0.05 equivalent - same scale
                                                          # the RSI test suite's own check uses
        results = {}
        for n_substeps in (1, 4, 16, 64):
            all_r = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                df = rst.generate_synthetic_ohlc_path(regime_model, n_bars=n_bars, bar_interval_minutes=5,
                                                        start_price=1.1000, sub_steps_per_bar=n_substeps, rng=rng)
                trades = rsi_mod.backtest_instrument("SYN", df)
                all_r.extend(t["r"] for t in trades)
            n = len(all_r)
            avg_r = sum(all_r) / n
            std_r = np.std(all_r, ddof=1)
            z = avg_r / (std_r / (n ** 0.5)) if std_r > 0 else 0.0
            results[n_substeps] = {"n_trades": n, "avg_r": avg_r, "z": z}

        z1 = abs(results[1]["z"])
        avg_z_finer = (abs(results[4]["z"]) + abs(results[16]["z"]) + abs(results[64]["z"])) / 3.0

        self.assertGreater(z1, 3.0, f"expected a coarse-granularity artifact, got z={z1:.2f}")
        self.assertLess(avg_z_finer, z1 / 1.5,
                         f"expected the finer-granularity average ({avg_z_finer:.2f}) well below "
                         f"the coarse reading ({z1:.2f})")
        self.assertLess(avg_z_finer, 2.5,
                         "expected finer granularities to stay within noise-level significance")
        self.results = results


# ============================= _percentile_and_verdict =============================

class TestPercentileAndVerdict(unittest.TestCase):
    """run_regime_stress_test's percentile/outlier-flagging logic, unit tested directly against
    a fully controlled synthetic distribution (1..100) where the correct percentile and verdict
    are known exactly by construction, for both metric senses (higher-is-better and
    lower-is-better)."""

    def setUp(self):
        self.sim = np.arange(1, 101, dtype=float)   # 1..100 inclusive

    def test_higher_is_better_high_value_outperforms(self):
        pct, verdict = rst._percentile_and_verdict(95.0, self.sim, higher_is_better=True)
        self.assertAlmostEqual(pct, 95.0)
        self.assertEqual(verdict, "OUTPERFORMS")

    def test_higher_is_better_low_value_underperforms(self):
        pct, verdict = rst._percentile_and_verdict(5.0, self.sim, higher_is_better=True)
        self.assertAlmostEqual(pct, 5.0)
        self.assertEqual(verdict, "UNDERPERFORMS")

    def test_higher_is_better_middle_value_is_typical(self):
        pct, verdict = rst._percentile_and_verdict(50.0, self.sim, higher_is_better=True)
        self.assertAlmostEqual(pct, 50.0)
        self.assertEqual(verdict, "TYPICAL")

    def test_lower_is_better_flips_the_verdict_sense(self):
        # for a "lower is better" metric like max_drawdown, a LOW percentile (real value smaller
        # than almost every simulated one) is GOOD -> OUTPERFORMS, and a HIGH percentile is BAD
        pct_low, verdict_low = rst._percentile_and_verdict(5.0, self.sim, higher_is_better=False)
        pct_high, verdict_high = rst._percentile_and_verdict(95.0, self.sim, higher_is_better=False)
        self.assertEqual(verdict_low, "OUTPERFORMS")
        self.assertEqual(verdict_high, "UNDERPERFORMS")

    def test_exact_percentile_values_for_known_positions(self):
        for real_value, expected_pct in [(1.0, 1.0), (25.0, 25.0), (100.0, 100.0)]:
            pct, _ = rst._percentile_and_verdict(real_value, self.sim, higher_is_better=True)
            self.assertAlmostEqual(pct, expected_pct)

    def test_empty_distribution_does_not_crash(self):
        pct, verdict = rst._percentile_and_verdict(10.0, [])
        self.assertIsNone(pct)
        self.assertIn("no simulated paths", verdict)


# ============================= run_regime_stress_test - mocked backtest_fn =============================

class TestRunRegimeStressTestOrchestration(unittest.TestCase):
    """run_regime_stress_test's own orchestration (fit -> simulate n_paths -> aggregate ->
    percentile/verdict), tested with a small, fast, fully deterministic mock backtest_fn so this
    test is about the ORCHESTRATION LOGIC, not about hmmlearn's fit quality or a real strategy's
    signal generation (those are covered separately above and in the real-strategy smoke tests
    below)."""

    @_skip_no_hmmlearn
    def test_well_formed_output_with_mocked_backtest_fn(self):
        regime_model = _two_state_regime_model(means=[0.0003, -0.0002], stds=[0.0008, 0.0030],
                                                trans=[[0.97, 0.03], [0.04, 0.96]])
        real_df = rst.generate_synthetic_ohlc_path(regime_model, n_bars=1200, bar_interval_minutes=5,
                                                     start_price=1.1, sub_steps_per_bar=20,
                                                     rng=np.random.default_rng(42))

        def mock_backtest_fn(label, df):
            # one deterministic pseudo-trade per call: r = total path return in "R units" of a
            # fixed 1% stop - exercises the same (label, df) -> list_of_trade_dicts contract
            # backtest_fn must satisfy, without depending on any real strategy's own signal logic
            total_ret = (df["Close"].iloc[-1] / df["Open"].iloc[0]) - 1.0
            return [{"side": "LONG", "outcome": "FLAT", "r": float(total_ret / 0.01)}]

        result = rst.run_regime_stress_test(mock_backtest_fn, real_df, "SYN", n_paths=10,
                                             sub_steps_per_bar=10, seed=1, n_regimes=2,
                                             show_progress=False)
        self.assertIsNotNone(result)
        self.assertEqual(result["n_paths"], 10)
        self.assertEqual(len(result["path_stats"]), 10)
        for key in ("total_r", "avg_r", "n_trades", "max_drawdown"):
            self.assertIn(key, result["distributions"])
            self.assertEqual(len(result["distributions"][key]), 10)
            self.assertTrue(np.isfinite(result["distributions"][key]).all())
        for key in ("total_r", "avg_r", "max_drawdown"):
            self.assertIn(key, result["percentiles"])
            self.assertGreaterEqual(result["percentiles"][key], 0.0)
            self.assertLessEqual(result["percentiles"][key], 100.0)
            self.assertIn(result["verdicts"][key], ("OUTPERFORMS", "UNDERPERFORMS", "TYPICAL"))
        self.assertIn("total_r", result["real_stats"])

    def test_missing_hmmlearn_propagates_none_without_crashing(self):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "hmmlearn" or name.startswith("hmmlearn."):
                raise ImportError("simulated missing hmmlearn")
            return real_import(name, *args, **kwargs)

        idx = pd.date_range("2020-01-01", periods=200, freq="5min", tz="America/New_York")
        closes = 1.1 + np.cumsum(np.random.default_rng(0).normal(0, 0.0002, 200))
        real_df = pd.DataFrame({"Open": closes, "High": closes + 0.0002,
                                 "Low": closes - 0.0002, "Close": closes}, index=idx)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            result = rst.run_regime_stress_test(lambda label, df: [], real_df, "SYN", n_paths=3,
                                                  show_progress=False)
        self.assertIsNone(result)


# ============================= real-strategy end-to-end smoke tests =============================

@_skip_no_hmmlearn
class TestEndToEndSmokeRunWithRealStrategies(unittest.TestCase):
    """The proof-of-concept this module was built for: TWO already-existing, real strategies -
    ict_po3_forex_dukascopy_backtest.backtest_instrument and rsi_mean_reversion_dukascopy_
    backtest.backtest_instrument (both imported, not reimplemented) - wired into
    run_regime_stress_test end-to-end against regime-generated synthetic data. Deliberately
    small (few thousand bars, few paths) for test speed; the pipeline's correctness, not runtime
    performance at production scale, is what's being validated here."""

    @classmethod
    def setUpClass(cls):
        regime_model = _two_state_regime_model(means=[0.0003, -0.0002], stds=[0.0007, 0.0030],
                                                trans=[[0.97, 0.03], [0.05, 0.95]], startprob=(0.6, 0.4))
        cls.real_df = rst.generate_synthetic_ohlc_path(
            regime_model, n_bars=3000, bar_interval_minutes=5, start_price=1.15,
            sub_steps_per_bar=30, rng=np.random.default_rng(21))

    def _assert_well_formed(self, result, n_paths):
        self.assertIsNotNone(result)
        self.assertEqual(result["n_paths"], n_paths)
        self.assertEqual(result["regime_model"]["n_states"], 2)
        for key in ("total_r", "avg_r", "n_trades", "max_drawdown"):
            dist = result["distributions"][key]
            self.assertEqual(len(dist), n_paths)
            self.assertTrue(np.isfinite(dist).all())
        for key, verdict in result["verdicts"].items():
            self.assertIn(verdict, ("OUTPERFORMS", "UNDERPERFORMS", "TYPICAL"))
        real_trades = result["real_stats"]
        self.assertIsInstance(real_trades["n_trades"], int)
        self.assertGreaterEqual(real_trades["n_trades"], 0)

    def test_ict_po3_strategy_wired_in_end_to_end(self):
        result = rst.run_regime_stress_test(po3.backtest_instrument, self.real_df, "SYN",
                                             n_paths=5, sub_steps_per_bar=15, seed=3, n_regimes=2,
                                             show_progress=False)
        self._assert_well_formed(result, n_paths=5)

    def test_rsi_mean_reversion_strategy_wired_in_end_to_end(self):
        result = rst.run_regime_stress_test(rsi_mod.backtest_instrument, self.real_df, "SYN",
                                             n_paths=5, sub_steps_per_bar=15, seed=3, n_regimes=2,
                                             show_progress=False)
        self._assert_well_formed(result, n_paths=5)


if __name__ == "__main__":
    unittest.main()
