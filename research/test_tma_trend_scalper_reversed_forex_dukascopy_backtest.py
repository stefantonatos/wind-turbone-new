# Unit tests for tma_trend_scalper_reversed_forex_dukascopy_backtest.py.
#
# WHY THIS WRAPPER EXISTS AT ALL (see its own module docstring for the full version): the webapp's
# plain-run and Compare-All code paths never apply a ParamSpec's declared default to a strategy's
# module before running it, so two registry entries pointing at the SAME already-imported module
# would run identically regardless of what either ParamSpec claims. This wrapper is a genuinely
# separate module specifically so its REVERSE_SIGNALS=True actually takes effect - these tests
# exist to prove that delegation is correct, not to re-test the base strategy's own trading rules
# (already covered exhaustively by test_tma_trend_scalper_forex_dukascopy_backtest.py).
#
# These tests import the REAL production base module (research.tma_trend_scalper_forex_dukascopy_
# backtest via the normal package path) rather than a separately-loaded copy, because that is
# EXACTLY the module object the wrapper's own `_base` reference points at - anything else would
# not actually be testing what ships. Every test that mutates the shared module's globals restores
# them afterward, since it is genuinely shared, live state.
#
# Run with:  python -m pytest research/test_tma_trend_scalper_reversed_forex_dukascopy_backtest.py -v

import contextlib
import datetime
import importlib.util
import os
import unittest

import pandas as pd

from research import tma_trend_scalper_forex_dukascopy_backtest as base

_WRAPPER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "tma_trend_scalper_reversed_forex_dukascopy_backtest.py")
_spec = importlib.util.spec_from_file_location(
    "tma_trend_scalper_reversed_forex_dukascopy_backtest", _WRAPPER_PATH)
reversed_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reversed_mod)


@contextlib.contextmanager
def base_config(**overrides):
    """Same override-and-restore convention as the base module's own test file's `config()`, but
    operating on the REAL production module (research.tma_trend_scalper_forex_dukascopy_backtest)
    since that is what this wrapper actually delegates to - not the base test file's own separately
    -loaded copy, which is a different Python object entirely and would not exercise the wrapper's
    real delegation path."""
    previous = {name: getattr(base, name) for name in overrides}
    for name, value in overrides.items():
        setattr(base, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


STEP = 0.0004
BASE_PRICE = 1.0
WARMUP = 400
WARMUP_CYCLE = (1.0, 1.0, 1.0, -0.6)


def _bar(open_, close, wick=0.15):
    return {"Open": open_, "High": max(open_, close) + STEP * wick,
            "Low": min(open_, close) - STEP * wick, "Close": close}


def _bull_strike_scenario(start="2024-03-06 07:00"):
    """A minimal standalone version of the base test file's own scenario builder - a wavy uptrend
    warmup (so RSI doesn't pin at 100, same reason the base test file's own harness needs it) then
    a 3-line-strike signal that fires exactly one LONG in the un-reversed direction."""
    rows = []
    level = BASE_PRICE
    for i in range(WARMUP):
        delta = WARMUP_CYCLE[i % len(WARMUP_CYCLE)] * STEP * 0.6
        rows.append(_bar(level, level + delta))
        level += delta
    dip = STEP * 0.04
    for _ in range(3):
        rows.append(_bar(level, level - dip))
        level -= dip
    prev_open = rows[-1]["Open"]
    rows.append(_bar(level, prev_open + STEP * 2.0))
    for _ in range(6):
        p = rows[-1]["Close"]
        rows.append({"Open": p, "High": p + STEP * 0.02, "Low": p - STEP * 0.02, "Close": p})
    index = pd.date_range(start=pd.Timestamp(start, tz="UTC"), periods=len(rows), freq="5min")
    return pd.DataFrame(rows, index=index)


class TestDelegation(unittest.TestCase):
    def test_reversed_trade_mirrors_the_base_modules_own_output(self):
        df = _bull_strike_scenario()
        with base_config(WEEKDAY_FILTER_MODE=1, SESSION_START_HOUR=0, SESSION_END_HOUR=24):
            normal = base.backtest_instrument("TEST", df)
            reversed_trades = reversed_mod.backtest_instrument("TEST", df)
        self.assertEqual(len(normal), 1)
        self.assertEqual(len(reversed_trades), 1)
        self.assertEqual(normal[0]["side"], "LONG")
        self.assertEqual(reversed_trades[0]["side"], "SHORT")
        self.assertAlmostEqual(normal[0]["entry"], reversed_trades[0]["entry"])
        self.assertAlmostEqual(normal[0]["sl_distance"], reversed_trades[0]["sl_distance"])

    def test_instruments_are_shared_with_the_base_module(self):
        self.assertEqual(reversed_mod.INSTRUMENTS, base.INSTRUMENTS)


class TestStateRestoration(unittest.TestCase):
    def test_reverse_signals_is_false_before_and_after_a_call(self):
        df = _bull_strike_scenario()
        self.assertFalse(base.REVERSE_SIGNALS)
        with base_config(WEEKDAY_FILTER_MODE=1, SESSION_START_HOUR=0, SESSION_END_HOUR=24):
            reversed_mod.backtest_instrument("TEST", df)
        self.assertFalse(base.REVERSE_SIGNALS)

    def test_repeated_calls_never_leak_reverse_signals_as_true(self):
        df = _bull_strike_scenario()
        with base_config(WEEKDAY_FILTER_MODE=1, SESSION_START_HOUR=0, SESSION_END_HOUR=24):
            for _ in range(5):
                reversed_mod.backtest_instrument("TEST", df)
                self.assertFalse(base.REVERSE_SIGNALS)

    def test_reverse_signals_is_restored_even_if_the_base_call_raises(self):
        def _boom(label, df):
            raise RuntimeError("simulated failure inside the base strategy")

        original = base.backtest_instrument
        base.backtest_instrument = _boom
        try:
            self.assertFalse(base.REVERSE_SIGNALS)
            with self.assertRaises(RuntimeError):
                reversed_mod.backtest_instrument("TEST", pd.DataFrame())
            self.assertFalse(base.REVERSE_SIGNALS)
        finally:
            base.backtest_instrument = original

    def test_a_pre_existing_true_value_is_restored_to_true_not_clobbered_to_false(self):
        # The restore uses whatever REVERSE_SIGNALS held BEFORE the call, not a hardcoded False -
        # pinned so a future refactor can't silently change that to an unconditional reset.
        df = _bull_strike_scenario()
        base.REVERSE_SIGNALS = True
        try:
            with base_config(WEEKDAY_FILTER_MODE=1, SESSION_START_HOUR=0, SESSION_END_HOUR=24):
                reversed_mod.backtest_instrument("TEST", df)
            self.assertTrue(base.REVERSE_SIGNALS)
        finally:
            base.REVERSE_SIGNALS = False


class TestFetchPropagation(unittest.TestCase):
    def test_fetch_instrument_data_copies_wrapper_globals_onto_the_base_module(self):
        seen = {}

        def _stub(label, instrument_const):
            seen["FETCH_START"] = base.FETCH_START
            seen["FETCH_END"] = base.FETCH_END
            seen["CACHE_DIR"] = base.CACHE_DIR
            return None

        original = base.fetch_instrument_data
        base.fetch_instrument_data = _stub
        original_start, original_end, original_cache = (
            reversed_mod.FETCH_START, reversed_mod.FETCH_END, reversed_mod.CACHE_DIR)
        try:
            reversed_mod.FETCH_START = datetime.datetime(2020, 1, 1)
            reversed_mod.FETCH_END = datetime.datetime(2021, 1, 1)
            reversed_mod.CACHE_DIR = "/tmp/some-sentinel-cache-dir"
            reversed_mod.fetch_instrument_data("EURUSD", object())
            self.assertEqual(seen["FETCH_START"], datetime.datetime(2020, 1, 1))
            self.assertEqual(seen["FETCH_END"], datetime.datetime(2021, 1, 1))
            self.assertEqual(seen["CACHE_DIR"], "/tmp/some-sentinel-cache-dir")
        finally:
            base.fetch_instrument_data = original
            reversed_mod.FETCH_START, reversed_mod.FETCH_END, reversed_mod.CACHE_DIR = (
                original_start, original_end, original_cache)


if __name__ == "__main__":
    unittest.main()
