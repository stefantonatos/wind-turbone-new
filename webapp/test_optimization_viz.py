import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optimization_viz as viz


class _Result:
    def __init__(self, heatmap=None, monte_carlo=None, cluster_verdict=None, walk_forward=None):
        self.heatmap = heatmap
        self.monte_carlo = monte_carlo
        self.cluster_verdict = cluster_verdict
        self.walk_forward = walk_forward


def _mc_df():
    return pd.DataFrame([
        {"range_minutes": 15, "reward_risk": 1.0, "n_trades": 400, "avg_r": 0.05,
         "boot_total_r_p5": 4.0, "boot_total_r_p50": 20.0, "boot_total_r_p95": 36.0,
         "p_total_r_le_0": 0.02},
        {"range_minutes": 30, "reward_risk": 2.0, "n_trades": 380, "avg_r": -0.01,
         "boot_total_r_p5": -18.0, "boot_total_r_p50": -4.0, "boot_total_r_p95": 9.0,
         "p_total_r_le_0": 0.71},
    ])


def _wf_df():
    return pd.DataFrame([
        {"fold": 1, "oos_start": "2019-01-01", "oos_end": "2020-01-01",
         "is_avg_r": 0.12, "oos_avg_r": 0.03},
        {"fold": 2, "oos_start": "2020-01-01", "oos_end": "2021-01-01",
         "is_avg_r": 0.15, "oos_avg_r": -0.04},
    ])


class TestPipelineRail(unittest.TestCase):
    def test_every_stage_renders_with_a_word_not_only_a_colour(self):
        html = viz.render_pipeline({0: "done", 1: "running", 2: "pending", 3: "failed", 4: "sealed"})
        for word in ("COMPLETE", "RUNNING", "WAITING", "FAILED", "SEALED"):
            self.assertIn(word, html)

    def test_all_five_stage_titles_are_present(self):
        html = viz.render_pipeline({})
        for _num, title, _hint in viz.STAGES:
            self.assertIn(title, html)

    def test_running_stage_gets_the_live_class_and_others_do_not(self):
        html = viz.render_pipeline({1: "running"})
        self.assertEqual(html.count("optviz-live"), 2)  # the keyframes rule + the one live stage

    def test_stats_render_under_the_title_when_supplied(self):
        html = viz.render_pipeline({0: "done"}, stats={0: "196 cells scored"})
        self.assertIn("196 cells scored", html)

    def test_stage_detail_is_html_escaped(self):
        html = viz.render_pipeline({0: "done"}, stats={0: "<script>x</script>"})
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)


class TestStatesFromResult(unittest.TestCase):
    def test_missing_outputs_are_skipped_not_silently_shown_as_done(self):
        states = viz.states_from_result(_Result(heatmap=pd.DataFrame([[1.0]])))
        self.assertEqual(states[0], "done")
        self.assertEqual(states[1], "skipped")
        self.assertEqual(states[2], "skipped")
        self.assertEqual(states[3], "skipped")

    def test_an_empty_dataframe_counts_as_skipped_not_done(self):
        states = viz.states_from_result(_Result(monte_carlo=pd.DataFrame()))
        self.assertEqual(states[1], "skipped")

    def test_lockbox_defaults_to_sealed_when_never_used(self):
        self.assertEqual(viz.states_from_result(_Result())[4], "sealed")
        self.assertEqual(viz.states_from_result(_Result(), lockbox_status={"passed": True})[4], "done")


class TestHeatmap(unittest.TestCase):
    def test_zero_is_pinned_to_the_neutral_midpoint(self):
        """An all-negative grid must not render half-green. zmid=0 with symmetric
        bounds is what guarantees that; without it plotly centres on the data."""
        df = pd.DataFrame([[-0.05, -0.02], [-0.09, -0.01]],
                          index=["RANGE=15m", "RANGE=30m"], columns=["RR=1", "RR=2"])
        fig = viz.heatmap_figure(df)
        trace = fig.data[0]
        self.assertEqual(trace.zmid, 0.0)
        self.assertEqual(trace.zmin, -trace.zmax)
        self.assertGreater(trace.zmax, 0)

    def test_diverging_scale_has_a_neutral_grey_midpoint_not_a_hue(self):
        mid = dict(viz.DIVERGING_SCALE)[0.5]
        self.assertEqual(mid, viz.DIVERGING_MID)
        r, g, b = (int(mid[i:i + 2], 16) for i in (1, 3, 5))
        self.assertLess(max(r, g, b) - min(r, g, b), 60, "midpoint should read as neutral, not a hue")

    def test_returns_none_rather_than_an_empty_chart(self):
        self.assertIsNone(viz.heatmap_figure(None))
        self.assertIsNone(viz.heatmap_figure(pd.DataFrame()))


class TestMonteCarlo(unittest.TestCase):
    def test_headline_counts_only_cells_whose_p5_clears_zero(self):
        h = viz.monte_carlo_headline(_mc_df())
        self.assertEqual(h["cleared"], 1)
        self.assertEqual(h["total"], 2)
        self.assertAlmostEqual(h["share"], 0.5)

    def test_a_cell_whose_band_merely_straddles_zero_does_not_count(self):
        df = _mc_df()
        df.loc[0, "boot_total_r_p5"] = -0.5      # median still strongly positive
        self.assertEqual(viz.monte_carlo_headline(df)["cleared"], 0)

    def test_figure_marks_break_even_and_plots_one_row_per_cell(self):
        fig = viz.monte_carlo_figure(_mc_df())
        median_trace = [t for t in fig.data if t.mode == "markers"][0]
        self.assertEqual(len(median_trace.x), 2)
        self.assertTrue(any(s.type == "line" and s.x0 == 0 and s.x1 == 0
                            for s in fig.layout.shapes), "break-even reference line missing")

    def test_missing_bootstrap_columns_degrade_to_none(self):
        self.assertIsNone(viz.monte_carlo_figure(pd.DataFrame([{"avg_r": 0.1}])))
        self.assertIsNone(viz.monte_carlo_headline(pd.DataFrame([{"avg_r": 0.1}])))


class TestWalkForward(unittest.TestCase):
    def test_headline_counts_profitable_out_of_sample_folds_only(self):
        h = viz.walk_forward_headline(_wf_df())
        self.assertEqual(h["positive"], 1)
        self.assertEqual(h["total"], 2)

    def test_in_sample_being_positive_never_inflates_the_headline(self):
        """Both folds fit well in-sample; only one transferred. The headline must
        report 1, because in-sample is fitted by construction."""
        df = _wf_df()
        df["is_avg_r"] = [0.9, 0.9]
        self.assertEqual(viz.walk_forward_headline(df)["positive"], 1)

    def test_figure_has_both_series_labelled_so_identity_is_not_colour_alone(self):
        fig = viz.walk_forward_figure(_wf_df())
        named = [t.name for t in fig.data if t.name]
        self.assertIn("in-sample (fitted)", named)
        self.assertIn("out-of-sample (real)", named)

    def test_one_axis_only(self):
        fig = viz.walk_forward_figure(_wf_df())
        self.assertNotIn("xaxis2", fig.layout)
        for trace in fig.data:
            self.assertIn(getattr(trace, "xaxis", None), (None, "x"))

    def test_non_finite_values_become_gaps_not_zeros(self):
        df = _wf_df()
        df.loc[0, "oos_avg_r"] = float("nan")
        fig = viz.walk_forward_figure(df)
        oos = [t for t in fig.data if t.name == "out-of-sample (real)"][0]
        self.assertIn(None, list(oos.x))
        self.assertNotIn(0.0, [v for v in oos.x if v is not None])


class TestLockboxPanel(unittest.TestCase):
    def test_sealed_state_reads_as_never_opened(self):
        html = viz.lockbox_panel(sealed=True, window="2024-01-01 to 2025-01-01")
        self.assertIn("SEALED", html)
        self.assertIn("Never opened", html)
        self.assertIn("2024-01-01", html)

    def test_opened_state_reports_the_recorded_attempt(self):
        html = viz.lockbox_panel(sealed=False, prior={"timestamp": "2026-08-04"})
        self.assertIn("OPENED", html)
        self.assertIn("2026-08-04", html)


if __name__ == "__main__":
    unittest.main()
