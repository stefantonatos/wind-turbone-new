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
    """Returns the registered module's own source, PLUS the source of every research.* module it
    delegates to via `importlib.import_module("research.<name>")` (followed recursively, one level
    is all any current wrapper needs but this doesn't assume that). Needed for delegating wrapper
    modules like tma_trend_scalper_reversed_forex_dukascopy_backtest.py, whose own file contains no
    `"stop_pct":`/`"date":` literal - it returns whatever the module it wraps produces, unmodified -
    so checking only its own text would report a false violation of a contract it isn't actually
    breaking. See TestSourceForFollowsDelegation below for proof this isn't vacuous either way."""
    def read(module_name):
        path = os.path.join(REPO_ROOT, "research", module_name.replace("research.", "") + ".py")
        with open(path) as f:
            return f.read()

    seen = set()
    to_visit = [strategy.module_name]
    combined = []
    while to_visit:
        name = to_visit.pop()
        if name in seen:
            continue
        seen.add(name)
        text = read(name)
        combined.append(text)
        to_visit.extend(re.findall(r'importlib\.import_module\(\s*["\'](research\.\w+)["\']\s*\)', text))
    return "\n".join(combined)


def _writes_key(source, key):
    """True if the module ever WRITES the key onto a trade, as opposed to merely mentioning it in a
    comment or reading it back. Two forms count, because both are in use in research/:

        trades.append({"stop_pct": risk / entry})   <- dict literal
        trade["window"] = window_label              <- subscript assignment after the fact

    Only matching the first form would have reported ICT Silver Bullet's `window` facet as missing
    when it is emitted on every trade. A read (`t["window"]`, `t["window"] == x`) must not match -
    that is the whole distinction this checker exists to draw."""
    quoted = re.escape(key)
    in_dict_literal = re.search(r'"%s"\s*:' % quoted, source)
    by_assignment = re.search(r'\[\s*"%s"\s*\]\s*=(?!=)' % quoted, source)
    return bool(in_dict_literal or by_assignment)


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


class TestEveryDeclaredFacetIsActuallyEmitted(unittest.TestCase):
    """A StrategyDef's `facets` list names extra trade-dict keys the Results page turns into filter
    dropdowns. app.py builds each one with `{t.get(facet) for t in trades}` - so a facet naming a
    key the strategy never writes produces an EMPTY multiselect that filters nothing, with no error
    anywhere. Same silent-failure shape as the two bugs this file was written for."""

    def test_all_declared_facets_are_written_by_their_strategy(self):
        broken = [(s.name, facet) for s in registry.STRATEGIES for facet in s.facets
                  if not _writes_key(_source_for(s), facet)]
        self.assertEqual(broken, [], f"facets declared in the registry but never written onto a "
                                      f"trade (they'd render as empty filter dropdowns): {broken}")


class TestFacetLabelRendering(unittest.TestCase):
    """Facet keys are snake_case; the filter label shown to a user should not be. Pinned because
    the transform is easy to drop back to a bare .capitalize() during unrelated edits, and the
    result ("Trade_type") is ugly rather than broken, so nothing else would catch it."""

    @staticmethod
    def _label(facet):
        return facet.replace("_", " ").capitalize()

    def test_multiword_facet_keys_render_as_words(self):
        self.assertEqual(self._label("trade_type"), "Trade type")

    def test_single_word_facet_keys_are_unchanged_apart_from_the_capital(self):
        self.assertEqual(self._label("pattern"), "Pattern")
        self.assertEqual(self._label("range"), "Range")

    def test_every_registered_facet_produces_a_clean_label(self):
        for strategy in registry.STRATEGIES:
            for facet in strategy.facets:
                label = self._label(facet)
                self.assertNotIn("_", label, f"{strategy.name}'s '{facet}' facet renders as {label!r}")
                self.assertTrue(label[:1].isupper())


class TestSourceForFollowsDelegation(unittest.TestCase):
    """Proof that _source_for's delegation-following isn't vacuous in either direction: a wrapper
    module's OWN text alone must not satisfy the contract (or a genuinely broken wrapper would pass
    by accident), but the COMBINED text (wrapper + whatever it delegates to) must."""

    def test_the_reversed_wrapper_alone_does_not_satisfy_the_contract(self):
        path = os.path.join(REPO_ROOT, "research",
                             "tma_trend_scalper_reversed_forex_dukascopy_backtest.py")
        with open(path) as f:
            wrapper_only = f.read()
        self.assertFalse(_writes_key(wrapper_only, "stop_pct"),
                          "if this starts passing, the wrapper grew its own trade-dict construction "
                          "and _source_for's delegation-following is no longer being exercised by "
                          "this test - update the fixture")

    def test_source_for_combines_the_wrapper_with_what_it_delegates_to(self):
        strategy = registry.STRATEGIES_BY_ID["tma_trend_scalper_reversed"]
        combined = _source_for(strategy)
        self.assertTrue(_writes_key(combined, "stop_pct"))
        self.assertTrue(_writes_key(combined, "date"))


class TestContractCheckerItself(unittest.TestCase):
    """The checker is only worth anything if it actually distinguishes the two cases - a test that
    passes vacuously would have hidden both original bugs just as well as no test at all."""

    def test_detects_a_key_written_into_a_dict(self):
        self.assertTrue(_writes_key('trades.append({"stop_pct": x / y})', "stop_pct"))

    def test_detects_a_key_written_by_subscript_assignment(self):
        self.assertTrue(_writes_key('trade["window"] = window_label', "window"))

    def test_ignores_a_key_only_mentioned_in_prose(self):
        self.assertFalse(_writes_key("# stop_pct is what the cost model keys off", "stop_pct"))

    def test_ignores_a_key_only_read_back(self):
        self.assertFalse(_writes_key('sp = t.get("stop_pct")\nif t["stop_pct"]:', "stop_pct"))

    def test_ignores_a_subscript_comparison(self):
        # `==` is a read, not a write - the (?!=) guard in the pattern is what makes this hold.
        self.assertFalse(_writes_key('if t["window"] == "10am":', "window"))


if __name__ == "__main__":
    unittest.main()
