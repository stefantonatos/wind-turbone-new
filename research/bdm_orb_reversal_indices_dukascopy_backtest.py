# "BIG DADDY MAX ORB" - opening-range breakout WITH a failed-breakout reversal, ported from the
# TradingView Pine v6 strategy of that name (part of a public "Scam or Slam" series that codes up
# popular trading concepts and backtests them).
#
# WHY PORT IT HERE RATHER THAN TRUST THE TRADINGVIEW REPORT
#
# The published report shows +$3,312.50 (+6.63%) with a $1,746.50 max drawdown on $50,000 of
# initial capital. That is a real number, but it is not a measurement of edge, for one specific and
# decisive reason: the Pine runs `default_qty_type = strategy.fixed, default_qty_value = 1`, i.e.
# ONE CONTRACT PER TRADE, while the stop distance is set by the width of that morning's opening
# range. A quiet-open day risks a few points; a CPI-morning open risks many times that. Fixed size
# with variable stop distance means every trade contributes a DIFFERENT dollar amount of risk, so
# the equity curve is a sum of unequal bets. A strategy can post a positive dollar total while
# having negative average R (a few wide-range winners paying for many narrow-range losers), or the
# reverse. The dollar total cannot distinguish those, and the report shows only the dollar total.
#
# This project scores everything in R - profit measured in units of the risk actually taken on that
# trade - precisely so that question has an answer. Porting the rules here puts them through the
# identical pipeline as the rest of the catalog: R-multiples, the per-instrument cost model, the
# out-of-sample holdout split, the multiple-comparisons bar, and the random-entry control that says
# what a zero-edge system scores under the same costs.
#
# Two further limits of the source report worth naming, since they bound how much it can support:
#   - It is ONE instrument over ONE date range with ONE parameter set. This port runs six indices.
#   - Slippage is set to 1 tick. That is optimistic for a market order filling a stop during a
#     failed opening-range breakout, which is exactly when the book is thinnest. Costs here are
#     applied by the app's own model instead, and the Cost Sensitivity tab shows the multiplier at
#     which the verdict flips - which is the honest way to handle a number nobody has measured.
#
# THE RULES (faithful to the Pine, deviations listed explicitly at the bottom of this header)
#
#   1. Build the opening range from the first ORB_MINUTES of the local cash session.
#   2. CONTINUATION: the first bar to CLOSE outside that range enters in the breakout's direction,
#      at that bar's close. Stop is the range midpoint (or the opposite side of the range - see
#      CONTINUATION_STOP_AT_MID). Target is REWARD_RISK x the stop distance.
#   3. REVERSAL: if price later CLOSES back inside the range, the breakout is treated as failed -
#      the continuation trade is closed at that bar's close and an opposite-direction trade opens,
#      stopped at the extreme the failed breakout reached, targeting REWARD_RISK x that distance.
#   4. At most one continuation and one reversal per day. Flat at the session close.
#
# THE ONE SUBTLE RULE THAT DOES MOST OF THE WORK. By default the Pine's
# `allowReversalAfterContinuationClosedInput` is OFF, and the reversal condition additionally
# requires the continuation position to still be open. That is not a detail - it is the difference
# between two different strategies. With it off, the reversal only fires while the continuation
# trade is still alive, which (with a midpoint stop) means price closed back inside the range but
# ABOVE the midpoint. With it on, any close back inside the range triggers a reversal, including
# after the continuation already stopped out. TradingView's broker emulator fills the protective
# stop intrabar BEFORE the script's close-based logic reads the position, so a bar that both
# stops the continuation out and closes back inside the range yields NO reversal in the default
# mode. This port reproduces that ordering exactly (see backtest_index: bracket first, signal
# second) - getting it backwards silently converts this into a different, busier strategy.
#
# DELIBERATE DEVIATIONS FROM THE PINE, each with its reason:
#   - MIN_STOP_PCT floor replaces Pine's `risk > syminfo.mintick` guard. A one-tick check is
#     nearly no check at all: a reversal entry that closes a hair below the failed-breakout high
#     has a near-zero stop distance and therefore a near-infinite R-multiple, which would dominate
#     the average and make the whole result an artifact of one bar. Trades whose stop is thinner
#     than MIN_STOP_PCT of price are skipped rather than scored.
#   - The end-of-session flatten happens on the LAST bar inside the trade window rather than the
#     first bar after it, so no fill is ever taken on a bar outside the window being tested.
#   - Six indices with their own local session opens rather than one hardcoded New York open. The
#     Pine's 0930-0945 default is a New York session; running DAX or FTSE against New York clock
#     times would be testing an arbitrary mid-session window, not an opening range.
#
# NOT MODELLED: the Pine's per-contract commission and tick slippage are not reproduced here.
# Costs are applied downstream by the app's per-instrument model so this strategy is charged on the
# same basis as every other one in the catalog - mixing two cost conventions inside one leaderboard
# is exactly the comparability bug that made an earlier version of that leaderboard unreadable.

# !pip install --upgrade dukascopy-python -q   # uncomment this line in Colab

import datetime

import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

# (label, instrument constant, IANA timezone, local cash-session open) - same six indices and the
# same per-index session mapping as orb_indices_dukascopy_backtest.py, so the two ORB variants in
# this catalog are directly comparable rather than differing in their instrument set as well as
# their rules.
INDICES = [
    ("SP500", dki.INSTRUMENT_IDX_AMERICA_E_SANDP_500, "America/New_York", pd.Timestamp("09:30").time()),
    ("NASDAQ100", dki.INSTRUMENT_IDX_AMERICA_E_NQ_100, "America/New_York", pd.Timestamp("09:30").time()),
    ("DOWJONES", dki.INSTRUMENT_IDX_AMERICA_E_D_J_IND, "America/New_York", pd.Timestamp("09:30").time()),
    ("DAX", dki.INSTRUMENT_IDX_EUROPE_E_DAAX, "Europe/Berlin", pd.Timestamp("09:00").time()),
    ("FTSE100", dki.INSTRUMENT_IDX_EUROPE_E_FUTSEE_100, "Europe/London", pd.Timestamp("08:00").time()),
    ("NIKKEI225", dki.INSTRUMENT_IDX_ASIA_E_N225JAP, "Asia/Tokyo", pd.Timestamp("09:00").time()),
]

START_DATE = datetime.datetime(2024, 1, 1)
END_DATE = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_MIN_5
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

ORB_MINUTES = 15            # Pine default: 0930-0945
SESSION_MINUTES = 390       # open + 6.5h = 1600 NY. Trade window is [open+ORB_MINUTES, open+SESSION_MINUTES)
REWARD_RISK = 1.0           # Pine's "Take Profit RR" input

CONTINUATION_STOP_AT_MID = 1   # 1 = ORB midpoint (Pine default), 0 = opposite side of the ORB
ENABLE_CONTINUATION = 1
ENABLE_REVERSALS = 1
ALLOW_REVERSAL_AFTER_CLOSE = 0  # Pine default OFF - see the header note; this flag changes the
                                 # strategy's character more than any other input here
CLOSE_AT_SESSION_END = 1

MIN_STOP_PCT = 0.02         # % of price. Floor on stop distance - see header (replaces mintick guard)

# Illustrative round-trip cost scenarios as a % of entry price, for this script's own standalone
# printout only. The webapp applies its own per-instrument measured cost model instead.
COST_PCT_SCENARIOS = [0.0, 0.01, 0.03, 0.05]


def to_local_time(index, tz_name):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(tz_name)


def _shift_time(base_time, minutes):
    return (datetime.datetime.combine(datetime.date.min, base_time)
            + datetime.timedelta(minutes=minutes)).time()


def _resolve_stop(side, orb_high, orb_low, orb_mid):
    if CONTINUATION_STOP_AT_MID:
        return orb_mid
    return orb_low if side == "LONG" else orb_high


def backtest_index(label, instrument_const, tz_name, session_start, df=None):
    """One continuation + at most one reversal per session, per the Pine's state machine.

    `df` is an injection point for tests (and for any caller that already holds the bars) - when
    None the data is fetched here, matching orb_indices_dukascopy_backtest.py's backtest_index
    signature so the webapp can drive both ORB variants through the same runner."""
    if df is None:
        df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE,
                                     START_DATE, END_DATE)
    if df is None or df.empty:
        return []
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df.index = to_local_time(df.index, tz_name)

    highs, lows, closes = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
    times = df.index
    n = len(closes)

    orb_end = _shift_time(session_start, ORB_MINUTES)
    session_end = _shift_time(session_start, SESSION_MINUTES)

    # Group bar positions by local calendar date first. A session that wraps past local midnight
    # would break this grouping, which is why the six indices above all open in their own local
    # morning - none of these sessions cross a date boundary in their own timezone.
    by_day = {}
    for i in range(n):
        by_day.setdefault(times[i].date(), []).append(i)

    trades = []
    for day, day_indices in by_day.items():
        orb_bars = [i for i in day_indices if session_start <= times[i].time() < orb_end]
        window = [i for i in day_indices if orb_end <= times[i].time() < session_end]
        if not orb_bars or not window:
            continue

        orb_high = max(highs[i] for i in orb_bars)
        orb_low = min(lows[i] for i in orb_bars)
        if not (orb_high > orb_low):
            continue
        orb_mid = (orb_high + orb_low) / 2.0

        breakout_dir = 0
        reversal_taken = False
        failed_extreme = None
        position = None

        for i in window:
            # (a) BRACKET FIRST. The protective stop/target are live orders during this bar and
            # fill intrabar, before any close-based decision is made. Stop is checked before
            # target - this project's standard pessimistic same-bar tie-break, and also
            # TradingView's own default assumption when a bar spans both.
            if position is not None:
                is_long = position["side"] == "LONG"
                hit_stop = lows[i] <= position["stop"] if is_long else highs[i] >= position["stop"]
                hit_target = highs[i] >= position["target"] if is_long else lows[i] <= position["target"]
                if hit_stop:
                    trades.append(_close_trade(position, -1.0, "SL", day))
                    position = None
                elif hit_target:
                    trades.append(_close_trade(position, REWARD_RISK, "TP", day))
                    position = None

            # (b) FIRST BREAKOUT. Only while flat and only once per session, matching the Pine's
            # `canDetectFirstBreakout` guard.
            if breakout_dir == 0 and position is None:
                if closes[i] > orb_high:
                    breakout_dir, failed_extreme = 1, highs[i]
                elif closes[i] < orb_low:
                    breakout_dir, failed_extreme = -1, lows[i]
                if breakout_dir != 0:
                    if ENABLE_CONTINUATION:
                        side = "LONG" if breakout_dir == 1 else "SHORT"
                        entry = closes[i]
                        stop = _resolve_stop(side, orb_high, orb_low, orb_mid)
                        position = _open_trade(side, entry, stop, "continuation")
                    # A breakout bar closes OUTSIDE the range by definition, so the reversal
                    # condition cannot also be true here - nothing further to check this bar.
                    continue

            if breakout_dir == 0:
                continue

            # (c) Track how far the failed breakout ran. Updated before the reversal check, so the
            # reversal's stop includes the current bar's extreme - same order as the Pine.
            if not reversal_taken:
                if breakout_dir == 1:
                    failed_extreme = highs[i] if failed_extreme is None else max(failed_extreme, highs[i])
                else:
                    failed_extreme = lows[i] if failed_extreme is None else min(failed_extreme, lows[i])

            # (d) REVERSAL on a close back inside the range.
            if not ENABLE_REVERSALS or reversal_taken:
                continue
            if not (orb_low < closes[i] < orb_high):
                continue
            continuation_open = position is not None and position["kind"] == "continuation"
            if not (not ENABLE_CONTINUATION or ALLOW_REVERSAL_AFTER_CLOSE or continuation_open):
                continue

            side = "SHORT" if breakout_dir == 1 else "LONG"
            entry = closes[i]
            stop = failed_extreme
            candidate = _open_trade(side, entry, stop, "reversal")
            if candidate is None:
                continue  # stop too thin to be a real trade - skipped, and the reversal stays
                          # available for a later bar (the Pine's mintick guard behaves the same way)
            if continuation_open:
                trades.append(_close_trade(position, _unrealized_r(position, closes[i]), "REV_EXIT", day))
            position = candidate
            reversal_taken = True

        if position is not None and CLOSE_AT_SESSION_END:
            last_close = closes[window[-1]]
            trades.append(_close_trade(position, _unrealized_r(position, last_close), "FLAT", day))

    return trades


def _open_trade(side, entry, stop, kind):
    """Returns None when the stop is thinner than MIN_STOP_PCT of price - see the header for why a
    mintick-sized floor isn't enough once results are scored in R rather than in dollars."""
    if entry <= 0 or stop is None:
        return None
    risk = entry - stop if side == "LONG" else stop - entry
    if risk <= 0 or (risk / entry) * 100.0 < MIN_STOP_PCT:
        return None
    target = entry + risk * REWARD_RISK if side == "LONG" else entry - risk * REWARD_RISK
    return {"side": side, "entry": entry, "stop": stop, "target": target, "risk": risk, "kind": kind}


def _unrealized_r(position, price):
    pnl = (price - position["entry"]) if position["side"] == "LONG" else (position["entry"] - price)
    return pnl / position["risk"]


def _close_trade(position, exit_r, outcome, day):
    # stop_pct and date are both REQUIRED by the webapp: without stop_pct the cost model silently
    # skips this strategy (scoring it gross while everything else is net), and without date the
    # out-of-sample holdout degrades to a positional split. Both of those were real, silent bugs in
    # this repo - see webapp/test_strategy_contract.py, which now fails the build if either is
    # missing from any registered strategy.
    return {
        "side": position["side"], "outcome": outcome, "r": exit_r, "date": day,
        "entry": position["entry"], "sl_distance": position["risk"],
        "stop_pct": position["risk"] / position["entry"],
        "trade_type": position["kind"],
    }


def main():
    print(f"BIG DADDY MAX ORB (continuation + failed-breakout reversal) - {START_DATE.date()} to {END_DATE.date()}")
    print(f"ORB {ORB_MINUTES}min, TP {REWARD_RISK}R, continuation stop = "
          f"{'ORB midpoint' if CONTINUATION_STOP_AT_MID else 'opposite ORB side'}, "
          f"reversal-after-close {'ON' if ALLOW_REVERSAL_AFTER_CLOSE else 'OFF'}\n")

    all_trades = []
    for label, const, tz_name, session_start in INDICES:
        try:
            index_trades = backtest_index(label, const, tz_name, session_start)
        except Exception as exc:
            print(f"{label}: failed ({exc})")
            continue
        for t in index_trades:
            t["instrument"] = label
            all_trades.append(t)
        print(f"{label}: {len(index_trades)} trades")

    if not all_trades:
        print("\nNo trades produced - check the output above.")
        return

    n = len(all_trades)
    total_r = sum(t["r"] for t in all_trades)
    print(f"\n{n} trades, {total_r:+.2f}R total, {total_r / n:+.4f}R/trade (BEFORE costs)")

    for kind in ("continuation", "reversal"):
        subset = [t for t in all_trades if t["trade_type"] == kind]
        if subset:
            sub_r = sum(t["r"] for t in subset)
            print(f"  {kind:<13} {len(subset):>5} trades, {sub_r:+9.2f}R, {sub_r / len(subset):+.4f}R/trade")

    print("\nCOST SENSITIVITY (% of entry price, round trip):")
    for cost_pct in COST_PCT_SCENARIOS:
        adj = sum(t["r"] - (cost_pct / 100.0) / t["stop_pct"] for t in all_trades)
        print(f"  {cost_pct:.2f}%: {adj:+.2f}R total, {adj / n:+.4f}R/trade")

    print("\nThe split above is the thing to read first. Continuation and reversal are two different "
          "bets sharing one script - if the total is positive but one leg is negative, the strategy "
          "being tested is really the other leg, and the reported total understates it while the "
          "combined rule set understates nothing about the leg that is losing.")


if __name__ == "__main__":
    main()
