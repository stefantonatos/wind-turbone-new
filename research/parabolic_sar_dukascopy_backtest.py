# Parabolic SAR (Wilder, "New Concepts in Technical Trading Systems", 1978) stop-and-reverse
# trend-following backtest via Dukascopy, 2016-2025.
#
# SOURCE: the recursive SAR/trend/extreme-point/acceleration-factor formula below is adapted
# from je-suis-tm/quant-trading's "Parabolic SAR backtest.py" (github.com/je-suis-tm/
# quant-trading, Apache License 2.0 - a permissively-licensed repo, unlike scraping someone's
# proprietary TradingView Pine script). That repo is itself just one Python coding of Wilder's
# original, decades-old, public-domain indicator formula - the initial/step/end acceleration
# factors (0.02/0.02/0.2) are Wilder's own published defaults, not that repo's invention.
#
# THE TRADING LOGIC ON TOP IS NOT COPIED FROM THAT REPO. Its own strategy layer is a simplified
# demo: long-or-flat only (position = 1 or 0, never short), no stop-loss, no R-multiple
# accounting - `positions = real_sar < Close`. This project's trade schema needs a real stop and
# a real R-multiple regardless, and long-only-with-no-risk-framing doesn't map onto that without
# inventing something anyway. So this uses the CANONICAL Wilder stop-and-reverse system instead
# - always in the market once started, alternating long and short, with the SAR line itself
# acting as the trailing stop - which is both the standard textbook description of "trading
# Parabolic SAR" (this is not a strategy this project invented) and the natural way to fit it
# into an R-multiple framework (the SAR level IS the stop, by construction).
#
# THE RECURSIVE FORMULA (Wilder's, ported faithfully - see compute_parabolic_sar - just
# reindexed onto plain Python lists in place of the reference's pandas .at[] loop):
#   sar[i] = the day-i SAR level projected using ONLY day i-1's own state (sar/af/ep) and days
#            i-1/i-2's highs or lows as a clip - never day i's own bar, so this is not lookahead;
#            day i's own high/low is only used to check whether TODAY's price crosses that
#            already-fixed level, exactly like every other stop-check in this project.
#   trend[i] = a signed run-length counter (+1, +2, +3... in an uptrend; -1, -2, -3... in a
#              downtrend) - abs(trend[i])==1 marks the FIRST bar of a newly flipped trend, i.e.
#              a reversal day.
#   ep[i] = the extreme point of the CURRENT trend (running highest high in an uptrend, lowest
#           low in a downtrend), reset to just today's own extreme on a reversal day.
#   af[i] = the acceleration factor - resets to INITIAL_AF on a reversal, then increases by
#           STEP_AF (capped at END_AF) each time a fresh trend extreme is made.
#   real_sar[i] = ep[i-1] on a reversal day (Wilder's own rule: a freshly-reversed trend's SAR
#                 starts conservatively at the JUST-ENDED trend's extreme point, not at the raw
#                 crossing price - deliberately wider, to avoid an immediate re-whipsaw), else
#                 the plain sar[i].
#
# MAPPING ONTO THIS PROJECT'S R-MULTIPLE TRADE SCHEMA (this part - not the SAR math itself - IS
# this port's own choice, since the reference repo's strategy layer doesn't use R at all):
#   - A reversal day (abs(trend[i])==1) closes any open position and opens the opposite one, both
#     at sar[i] - the level today's price actually crossed to trigger the flip (the tradable
#     fill price; the reversal always fires from a stop-and-reverse ORDER resting at that exact
#     level, same "filled at the precomputed level" convention as every other script here).
#   - The new position's initial stop is real_sar[i] (=ep[i-1], Wilder's own conservative
#     post-reversal level) - sl_distance = abs(entry - stop), fixed at entry, same convention as
#     Donchian/Turtle's ATR stop. R at the NEXT reversal (or a forced FLAT close at series end)
#     is computed against this fixed basis, even though the SAR trailing level moves every day
#     between now and then - exactly Donchian's own "fixed sl_distance vs. a moving trailing
#     exit" model, not a new pattern invented for this script.
#   - The very first position only opens on the FIRST real reversal in the series (not the
#     arbitrary 1-bar bootstrap trend[1], which is just an initialization artifact, not a real
#     signal) - a documented simplification, not silently assumed.
#   - MIN_SL_PCT skips (does not force through) a reversal whose fresh stop distance is
#     degenerately small - same guard, same "skip rather than force" convention as
#     bollinger_band_mean_reversion_dukascopy_backtest.py.
#
# DAILY BARS ONLY, no intrabar 5-min stop-checking (unlike Donchian/Turtle in this project): the
# flip determination itself is inherently a daily-bar computation (Wilder's own clip against the
# PRIOR TWO DAILY bars' highs/lows), so recomputing it at 5-min granularity would mean deriving a
# materially different, much noisier indicator, not "the same Parabolic SAR checked more
# precisely." Same daily-only choice already made by
# support_resistance_zone_bounce_dukascopy_backtest.py in this project, for the same reason.
#
# NO PARAMETER GRID SEARCH - Wilder's own published defaults (0.02/0.02/0.2), tested for
# temporal consistency (split-period check), not tuned to this data. Swing/position system -
# expect far fewer, much longer-held trades than this project's intraday scripts.

# !pip install --upgrade dukascopy-python -q   # uncomment this line in Colab

import datetime

import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

INSTRUMENTS = [
    ("EURUSD", dki.INSTRUMENT_FX_MAJORS_EUR_USD),
    ("GBPUSD", dki.INSTRUMENT_FX_MAJORS_GBP_USD),
    ("USDJPY", dki.INSTRUMENT_FX_MAJORS_USD_JPY),
    ("XAUUSD", dki.INSTRUMENT_FX_METALS_XAU_USD),
]

FETCH_START = datetime.datetime(2016, 1, 1)
FETCH_END = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_DAY_1
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

INITIAL_AF = 0.02   # Wilder's own published defaults - not tuned, not this repo's invention
STEP_AF = 0.02
END_AF = 0.2

MIN_SL_PCT = 0.02   # floor on stop distance, as a % of entry - guards against a degenerate near-zero SAR gap

# Illustrative round-trip cost scenarios, as a percentage of entry price - NOT measured real spread
# data, just a few bracketing assumptions to see how much cost this edge can absorb before it
# disappears, same convention introduced in support_resistance_zone_bounce_dukascopy_backtest.py.
COST_PCT_SCENARIOS = [0.0, 0.01, 0.03, 0.05]


def fetch_daily_ohlc(instrument_const):
    df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, FETCH_START, FETCH_END)
    if df.empty:
        return None
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def compute_parabolic_sar(highs, lows, closes):
    """Wilder's recursive SAR - see the file header for the full derivation. Returns
    (sar, real_sar, trend, ep, af), each a plain list the same length as `closes`, with indices
    0 and (for `trend`) index 1's neighbors left at their initialized defaults (0 / 0.0) since
    the recursion only produces a meaningful value from index 1 onward for the seeded fields and
    index 2 onward for everything computed inside the loop."""
    n = len(closes)
    trend = [0] * n
    sar = [0.0] * n
    real_sar = [0.0] * n
    ep = [0.0] * n
    af = [0.0] * n

    if n < 2:
        return sar, real_sar, trend, ep, af

    trend[1] = 1 if closes[1] > closes[0] else -1
    sar[1] = highs[0] if trend[1] > 0 else lows[0]
    real_sar[1] = sar[1]
    ep[1] = highs[1] if trend[1] > 0 else lows[1]
    af[1] = INITIAL_AF

    for i in range(2, n):
        projected = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])

        if trend[i - 1] < 0:
            sar[i] = max(projected, highs[i - 1], highs[i - 2])
            trend[i] = 1 if sar[i] < highs[i] else trend[i - 1] - 1
        else:
            sar[i] = min(projected, lows[i - 1], lows[i - 2])
            trend[i] = -1 if sar[i] > lows[i] else trend[i - 1] + 1

        if trend[i] < 0:
            ep[i] = lows[i] if trend[i] == -1 else min(lows[i], ep[i - 1])
        else:
            ep[i] = highs[i] if trend[i] == 1 else max(highs[i], ep[i - 1])

        if abs(trend[i]) == 1:
            real_sar[i] = ep[i - 1]
            af[i] = INITIAL_AF
        else:
            real_sar[i] = sar[i]
            af[i] = af[i - 1] if ep[i] == ep[i - 1] else min(END_AF, af[i - 1] + STEP_AF)

    return sar, real_sar, trend, ep, af


def backtest_instrument(label, df):
    """Runs the stop-and-reverse trading layer described in the file header over one
    instrument's daily bars. `df` must have Open/High/Low/Close columns and a tz-aware
    DatetimeIndex (as produced by fetch_daily_ohlc). Returns a list of trade dicts."""
    highs, lows, closes = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
    times = df.index
    n = len(closes)
    if n < 3:
        return []

    sar, real_sar, trend, ep, af = compute_parabolic_sar(highs, lows, closes)

    trades = []
    # BOOTSTRAP: the trend already underway from bar 1 (trend[1], seeded from just a 1-bar close
    # comparison - see compute_parabolic_sar) is a real position under Wilder's own "always in
    # the market" convention, not something to wait out. Skipping it would silently miss the
    # entire first trend whenever it happens to run long without ever reversing (confirmed by a
    # steady-uptrend smoke test producing zero trades before this fix) - an unavoidable
    # initialization artifact of any indicator needing history to compute (same category as
    # Donchian's warmup period here), entered immediately at bar 1's own close rather than
    # silently skipped, and documented rather than assumed away.
    #
    # Its seed stop (real_sar[1] = high[0] or low[0]) is ALWAYS close to close[1] by construction
    # (they're one bar apart with no accumulated acceleration factor yet) - MIN_SL_PCT's usual
    # "skip a thin signal" convention would reject this bootstrap position on essentially every
    # run, defeating its purpose. So the bootstrap is the one place this script WIDENS a thin
    # stop to the floor instead of skipping (same widen-not-skip convention ict_po3 uses for its
    # own unavoidable edge case) - a real position always opens here, just never with a
    # near-zero risk basis.
    boot_side = "LONG" if trend[1] > 0 else "SHORT"
    boot_entry = closes[1]
    boot_sl_distance = abs(boot_entry - real_sar[1])
    min_sl = (MIN_SL_PCT / 100.0) * boot_entry
    if boot_sl_distance < min_sl:
        boot_sl_distance = min_sl
    boot_stop = boot_entry - boot_sl_distance if boot_side == "LONG" else boot_entry + boot_sl_distance
    position = {"side": boot_side, "entry": boot_entry, "stop": boot_stop,
                "sl_distance": boot_sl_distance, "entry_date": times[1].date(), "entry_time": times[1]}

    for i in range(2, n):
        if abs(trend[i]) != 1:
            continue   # trend continuing - nothing to do on a non-reversal day

        reversal_side = "LONG" if trend[i] > 0 else "SHORT"
        fill_price = sar[i]      # the level today's price actually crossed to trigger the flip
        new_stop = real_sar[i]   # Wilder's conservative post-reversal stop (=ep[i-1])

        if position is not None:
            old_side = position["side"]
            pnl = (fill_price - position["entry"]) if old_side == "LONG" else (position["entry"] - fill_price)
            trades.append({"side": old_side, "outcome": "SAR", "r": pnl / position["sl_distance"],
                            "date": position["entry_date"], "stop_pct": position["sl_distance"] / position["entry"],
                            "entry_price": position["entry"], "stop_price": position["stop"], "target_price": None,
                            "exit_price": fill_price, "entry_time": position["entry_time"], "exit_time": times[i]})
            position = None

        sl_distance = abs(fill_price - new_stop)
        min_sl = (MIN_SL_PCT / 100.0) * fill_price
        if sl_distance < min_sl:
            continue   # degenerate fresh stop - skip rather than force a bad trade through

        position = {"side": reversal_side, "entry": fill_price, "stop": new_stop,
                    "sl_distance": sl_distance, "entry_date": times[i].date(), "entry_time": times[i]}

    if position is not None:
        last_close = closes[-1]
        pnl = (last_close - position["entry"]) if position["side"] == "LONG" else (position["entry"] - last_close)
        trades.append({"side": position["side"], "outcome": "FLAT", "r": pnl / position["sl_distance"],
                        "date": position["entry_date"], "stop_pct": position["sl_distance"] / position["entry"],
                        "entry_price": position["entry"], "stop_price": position["stop"], "target_price": None,
                        "exit_price": last_close, "entry_time": position["entry_time"], "exit_time": times[-1]})

    return trades


def main():
    years = (FETCH_END - FETCH_START).days / 365
    print(f"Downloading {len(INSTRUMENTS)} instruments (daily bars) from Dukascopy over ~{years:.0f} years "
          f"({FETCH_START.date()} to {FETCH_END.date()}).\n")

    data = {}
    for label, instrument_const in INSTRUMENTS:
        print(f"{label}...", end=" ")
        try:
            df = fetch_daily_ohlc(instrument_const)
        except Exception as exc:
            print(f"failed ({exc})")
            continue
        if df is None:
            print("no data")
            continue
        data[label] = df
        print(f"{len(df)} daily bars")

    if not data:
        print("No data downloaded - check output above.")
        return

    all_trades = []
    for label, df in data.items():
        trades = backtest_instrument(label, df)
        for t in trades:
            t["instrument"] = label
        all_trades.extend(trades)

    if not all_trades:
        print("\nNo trades at all - no reversal ever cleared MIN_SL_PCT on this data.")
        return

    total_r = sum(t["r"] for t in all_trades)
    n_trades = len(all_trades)
    print("\n" + "=" * 70)
    print(f"PARABOLIC SAR (stop-and-reverse, daily) - {n_trades} trades, "
          f"{FETCH_START.date()} to {FETCH_END.date()}")
    print("=" * 70)
    print(f"Total R: {total_r:+.2f}   Avg R/trade: {total_r/n_trades:+.4f}")

    sar_exit = sum(1 for t in all_trades if t["outcome"] == "SAR")
    flat = sum(1 for t in all_trades if t["outcome"] == "FLAT")
    wins = sum(1 for t in all_trades if t["r"] > 0)
    print(f"Outcome breakdown: SAR reversal {sar_exit} ({sar_exit/n_trades*100:.1f}%)  "
          f"FLAT(timeout) {flat} ({flat/n_trades*100:.1f}%)  Win rate {wins/n_trades*100:.1f}%")

    per_instrument = {}
    for t in all_trades:
        per_instrument.setdefault(t["instrument"], []).append(t["r"])
    print("\nPer instrument:")
    for label, rs in sorted(per_instrument.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {label:10s}: {len(rs):4d} trades, {sum(rs):+8.2f}R")

    all_r = [t["r"] for t in all_trades]
    if n_trades >= 2:
        std_r = np.std(all_r, ddof=1)
        z = (total_r / n_trades) / (std_r / (n_trades ** 0.5)) if std_r > 0 else 0.0
    else:
        z = 0.0
    print(f"\nApprox z-score: {z:.2f} (rule of thumb: |z| > 1.96 for ~95% confidence this isn't chance)")
    if n_trades < 100:
        print(f"CAVEAT: only {n_trades} trades - too few to trust regardless of the z-score.")

    all_trades_sorted = sorted(all_trades, key=lambda t: t["date"])
    midpoint_date = all_trades_sorted[len(all_trades_sorted) // 2]["date"]
    first_half = [t for t in all_trades_sorted if t["date"] < midpoint_date]
    second_half = [t for t in all_trades_sorted if t["date"] >= midpoint_date]
    print(f"\nSPLIT-PERIOD CHECK (same fixed rule, not tuned to this data - does it hold up in both halves?):")
    for label, half in [("First half", first_half), ("Second half", second_half)]:
        if not half:
            continue
        r = sum(t["r"] for t in half)
        n = len(half)
        half_r = [t["r"] for t in half]
        if n >= 2:
            std_half = np.std(half_r, ddof=1)
            z_half = (r / n) / (std_half / (n ** 0.5)) if std_half > 0 else 0.0
        else:
            z_half = 0.0
        print(f"  {label} ({half[0]['date']} to {half[-1]['date']}): {n} trades, {r:+.2f}R, "
              f"{r/n:+.4f}R/trade, z={z_half:.2f}")

    print(f"\nNOTE ON MULTIPLE COMPARISONS: this project has shipped many strategies, several with "
          f"their own parameter grid searches - a single script's z-score in isolation isn't strong "
          f"evidence, since data-snooping risk compounds across every strategy and parameter "
          f"combination tried project-wide, not just this one.")

    print(f"\nCOST SENSITIVITY (illustrative round-trip spread scenarios, NOT measured real spread data):")
    for cost_pct in COST_PCT_SCENARIOS:
        cost_adjusted_total = sum(t["r"] - (cost_pct / 100.0) / t["stop_pct"] for t in all_trades)
        print(f"  {cost_pct:.2f}% round-trip cost: {cost_adjusted_total:+.2f}R total, "
              f"{cost_adjusted_total/n_trades:+.4f}R/trade")
    print(f"  If the total goes negative well before 0.05%, this edge is too thin to survive real "
          f"execution costs - check your actual broker's spread on each instrument against these numbers.")

    print("\nNo commission/spread/slippage modeled. Reversal fills are simulated at the exact computed "
          "SAR crossing level - real fills, especially in a fast move, would usually be worse. This is a "
          "swing/position system, always in the market once started: expect far fewer, much longer-held "
          "trades than this project's intraday scripts, with a small number of large trend trades likely "
          "dominating the total (classic SAR behavior - frequent small whipsaw losses, occasional big "
          "trend wins).")


if __name__ == "__main__":
    main()
