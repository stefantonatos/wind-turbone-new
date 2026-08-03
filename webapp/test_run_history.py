import os
import shutil
import tempfile
import unittest

import run_history
import stats as stats_mod


def _trade(r, instrument="EURUSD", stop_pct=0.001):
    return {"r": r, "instrument": instrument, "stop_pct": stop_pct, "date": "2026-01-01"}


class TestAppendRunUsesCostAdjustedSummary(unittest.TestCase):
    """History/Gallery cards must show the SAME number "View full results" shows by default -
    that page's own "Apply typical trading costs" checkbox defaults to checked, so a raw/uncosted
    summary here would show a rosier headline total than clicking into the very same run reveals.
    Confirmed live: a real run whose raw total_r was +116.75 (positive) showed -99.60% total
    once costs were applied on the Results page - the card and the full view must agree."""

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self._orig_dir = run_history.HISTORY_DIR
        self._orig_file = run_history.HISTORY_FILE
        run_history.HISTORY_DIR = self._tmp_dir
        run_history.HISTORY_FILE = os.path.join(self._tmp_dir, "runs.jsonl")

    def tearDown(self):
        run_history.HISTORY_DIR = self._orig_dir
        run_history.HISTORY_FILE = self._orig_file
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_summary_total_r_matches_cost_adjusted_not_raw(self):
        # Every trade has a small positive raw r, but with a tight stop_pct the per-trade cost
        # deduction outweighs it - raw sum is positive, cost-adjusted sum is negative. This is
        # exactly the shape of the real bug report (raw +116.75R, cost-adjusted -99.60%).
        trades = [_trade(r=0.05) for _ in range(200)]
        raw_total = sum(t["r"] for t in trades)
        self.assertGreater(raw_total, 0, "test fixture should have a positive raw total")

        cost_adjusted, _ = stats_mod.apply_cost_adjustment(trades)
        expected_total_r = sum(t["r"] for t in cost_adjusted)
        self.assertLess(expected_total_r, 0, "test fixture should flip negative after costs")

        run_id = run_history.append_run(
            strategy_name="Test Strategy", instruments=["EURUSD"],
            start_date="2026-01-01", end_date="2026-06-01", params={}, trades=trades,
        )

        rows = run_history.load_runs()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["run_id"], run_id)
        self.assertAlmostEqual(row["total_r"], expected_total_r, places=6)
        self.assertNotAlmostEqual(row["total_r"], raw_total, places=2)
        self.assertAlmostEqual(row["avg_r"], expected_total_r / len(trades), places=6)

    def test_max_drawdown_also_uses_cost_adjusted_trades(self):
        trades = [_trade(r=0.05) for _ in range(50)]
        cost_adjusted, _ = stats_mod.apply_cost_adjustment(trades)
        expected_dd = stats_mod.max_drawdown(cost_adjusted)

        run_history.append_run(
            strategy_name="Test Strategy", instruments=["EURUSD"],
            start_date="2026-01-01", end_date="2026-06-01", params={}, trades=trades,
        )
        row = run_history.load_runs()[0]
        self.assertAlmostEqual(row["max_drawdown_r"], expected_dd, places=6)

    def test_raw_trades_are_still_stored_unadjusted_for_the_results_page_toggle(self):
        trades = [_trade(r=0.05) for _ in range(10)]
        run_id = run_history.append_run(
            strategy_name="Test Strategy", instruments=["EURUSD"],
            start_date="2026-01-01", end_date="2026-06-01", params={}, trades=trades,
        )
        loaded_trades = run_history.load_trades_for_run(run_id)
        self.assertEqual([t["r"] for t in loaded_trades], [0.05] * 10)


if __name__ == "__main__":
    unittest.main()
