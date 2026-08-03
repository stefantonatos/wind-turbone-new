# TMA TREND SCALPER - REVERSED. NOT a new strategy: this is
# tma_trend_scalper_forex_dukascopy_backtest.py's exact rules, run with every signal flipped to
# the opposite side (REVERSE_SIGNALS - see that module's own docstring for the full reasoning on
# why "this loses, so trade the opposite" needs testing through the same rigor as everything else,
# not trusted on sight, and for the project's own history of that exact experiment failing once
# already on the legacy ORB strategy).
#
# WHY THIS IS A SEPARATE FILE RATHER THAN JUST A DIFFERENT DEFAULT ON THE SAME MODULE: the webapp's
# plain "Run Backtest" and "Run All Strategies" (Compare All) code paths do NOT apply any
# ParamSpec-declared override to a strategy's module before running it - both currently call each
# strategy's runner with an EMPTY overrides dict (confirmed by reading webapp/app.py directly), so
# a ParamSpec's "default" value is decorative documentation for a plain run, not something that
# actually gets set. Two registry entries pointing at the SAME already-imported module would
# therefore run IDENTICALLY regardless of what either one's ParamSpec claims - there would be no
# way to actually get the reversed variant to behave differently without hand-editing the .py file
# between runs. A genuinely separate module, with its own independent Python module namespace, is
# the only way this takes effect automatically - including inside Compare All, which is the entire
# point (getting this variant through the out-of-sample holdout split and the Šidák-corrected
# significance bar automatically, rather than reading one flattering full-period number).
#
# HOW IT WORKS: delegates every real computation to the base module's own already-tested functions
# (fetch_instrument_data, backtest_instrument, every indicator) rather than re-implementing
# anything - this file has no independent trading logic of its own to get wrong. Each call
# temporarily sets the base module's REVERSE_SIGNALS to True, delegates, then restores it to False
# in a `finally` block immediately after. Safe under this app's execution model specifically
# because Compare All runs every strategy SEQUENTIALLY in one process (confirmed in
# webapp/app.py's render_compare_all_section: a plain `for strategy in STRATEGIES:` loop, never
# concurrent) - so there is never a moment where this module's mutation of the shared base module's
# global could be visible to a different strategy's own run.

import importlib

_base = importlib.import_module("research.tma_trend_scalper_forex_dukascopy_backtest")

INSTRUMENTS = _base.INSTRUMENTS
FETCH_START = _base.FETCH_START
FETCH_END = _base.FETCH_END
CACHE_DIR = _base.CACHE_DIR   # placeholder - the webapp registry's _run_with_own_cache runner
                                # overwrites this attribute on THIS module before every run


def fetch_instrument_data(label, instrument_const):
    # The base module's fetch_instrument_data reads FETCH_START/FETCH_END/CACHE_DIR from ITS OWN
    # global namespace (ordinary Python closure-over-module-globals behaviour, not this module's) -
    # so this module's current values (set onto THIS module's globals by _run_with_own_cache) are
    # copied across before delegating, every call, in case the webapp changed the date range or
    # cache location since the last one.
    _base.FETCH_START = FETCH_START
    _base.FETCH_END = FETCH_END
    _base.CACHE_DIR = CACHE_DIR
    return _base.fetch_instrument_data(label, instrument_const)


def backtest_instrument(label, df):
    previous = _base.REVERSE_SIGNALS
    _base.REVERSE_SIGNALS = True
    try:
        return _base.backtest_instrument(label, df)
    finally:
        _base.REVERSE_SIGNALS = previous   # restored even if backtest_instrument raises


def main():
    print("TMA TREND SCALPER - REVERSED (every signal flipped to the opposite side)")
    print("This is a control for the base strategy, not a separately-designed strategy - see this ")
    print("file's own header for why 'it loses, so trade the opposite' needs the same scrutiny as ")
    print("a real strategy, and this project's own history of that experiment failing once already ")
    print("(quantconnect/main.py's header: a reversed ORB variant profitable on a 17-day sample ")
    print("lost money too on a full year of real data).\n")

    global FETCH_START, FETCH_END, CACHE_DIR
    FETCH_START, FETCH_END, CACHE_DIR = _base.FETCH_START, _base.FETCH_END, _base.CACHE_DIR

    all_trades = []
    for label, const in INSTRUMENTS:
        try:
            df = fetch_instrument_data(label, const)
        except Exception as exc:
            print(f"{label}: failed ({exc})")
            continue
        if df is None or df.empty:
            print(f"{label}: no data")
            continue
        instrument_trades = backtest_instrument(label, df)
        for t in instrument_trades:
            t["instrument"] = label
            all_trades.append(t)
        print(f"{label}: {len(instrument_trades)} trades")

    if not all_trades:
        print("\nNo trades produced - check the output above.")
        return

    n = len(all_trades)
    total_r = sum(t["r"] for t in all_trades)
    print(f"\n{n} trades, {total_r:+.2f}R total, {total_r / n:+.4f}R/trade (BEFORE costs)")


if __name__ == "__main__":
    main()
