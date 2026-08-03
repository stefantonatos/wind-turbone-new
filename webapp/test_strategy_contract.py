# Structural contract tests: every strategy wired into the app must emit the per-trade fields the
# webapp's own scoring silently depends on.
#
# WHY THIS FILE EXISTS: two strategies shipped for weeks quietly violating this contract, and
# because both failure modes are SILENT (no crash, no warning - just a different, wrong number in
# the same column as everyone else's right one) neither was caught by the 433 research-layer tests
# or by any amount of looking at the UI:
#
#   1. support_resistance_zone_bounce never recorded "stop_pct". stats.apply_cost_adjustment skips
#      any trade without it (`if r is not None and stop_pct:`), so that strategy alone was scored
#      GROSS of trading costs while the other 15 were scored net - and it ranked 2nd of 11 on a real
#      Compare-All leaderboard largely because of it.
#   2. orb_indices never recorded "date". stats.split_trades_for_holdout falls back to a POSITIONAL
#      split when any trade lacks one, and since trades are concatenated per-instrument, its
#      "holdout" was really "the last ~1.5 of 6 indices over the whole period" - a cross-INSTRUMENT
#      split presented in the same column as ten genuine cross-TIME holdouts.
#
# Both are one-line omissions with leaderboard-corrupting consequences, which is exactly the kind of
# thing a contract test is for. This is deliberately a SOURCE-level check rather than a run-the-
# strategy check: running all 16 needs real Dukascopy data (unavailable in CI/sandbox), and the
# thing actually worth pinning down is "does this script ever write the key", which the source
# answers definitively and instantly.
#
# Run with:  python -m pytest webapp/test_strategy_contract.py -v

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import dukascopy_python

# registry imports every research module at build time, and several fetch on import-adjacent paths -
# stub the network out before importing it, same pattern the webapp's own preview harness uses.
dukascopy_python.fetch = lambda *a, **k: pd.DataFrame(
    columns=["open", "high", "low", "close", "volume"])

import registry  # noqa: E402


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _source_for(strategy):
    path = os.path.join(REPO_ROOT, "research", strategy.module_name.replace("research.", "") + ".py")
    with open(path) as f:
        return f.read()


def _writes_key(source, key):
    """True if the module ever writes `"<key>":` into a dict literal - i.e. actually emits it on a
    trade, as opposed to merely mentioning it in a comment or reading it back."""
    return re.search(r'"%s"\s*:' % re.escape(key), source) is not None


class TestEveryStrategyRecordsStopPct(unittest.TestCase):
    """Without stop_pct, stats.apply_cost_adjustment silently leaves the trade uncosted - so the
    strategy competes on the leaderboard without paying the trading costs everyone else pays."""

    def test_all_registry_strategies_emit_stop_pct(self):
        missing = [s.name for s in registry.STRATEGIES if not _writes_key(_source_for(s), "stop_pct")]
        self.assertEqual(missing, [], f"strategies not emitting stop_pct (they'd be scored gross of "
                                       f"costs while every other strategy is scored net): {missing}")


class TestEveryStrategyRecordsDate(unittest.TestCase):
    """Without a per-trade date, stats.split_trades_for_holdout silently degrades from a
    cross-TIME holdout to a positional one over a per-instrument-concatenated list."""

    def test_all_registry_strategies_emit_date(self):
        missing = [s.name for s in registry.STRATEGIES if not _writes_key(_source_for(s), "date")]
        self.assertEqual(missing, [], f"strategies not emitting a per-trade date (their 'holdout' "
                                       f"would be a cross-instrument split, not out-of-sample in "
                                       f"time): {missing}")


class TestContractCheckerItself(unittest.TestCase):
    """The checker is only worth anything if it actually distinguishes the two cases - a test that
    passes vacuously would have hidden both original bugs just as well as no test at all."""

    def test_detects_a_key_written_into_a_dict(self):
        self.assertTrue(_writes_key('trades.append({"stop_pct": x / y})', "stop_pct"))

    def test_ignores_a_key_only_mentioned_in_prose(self):
        self.assertFalse(_writes_key("# stop_pct is what the cost model keys off", "stop_pct"))

    def test_ignores_a_key_only_read_back(self):
        self.assertFalse(_writes_key('sp = t.get("stop_pct")\nif t["stop_pct"]:', "stop_pct"))


if __name__ == "__main__":
    unittest.main()
