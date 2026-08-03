# "TMA TREND SCALPER" - a triple-SMMA trend filter + candlestick reversal pattern + RSI
# confirmation, ported from a TradingView Pine v6 strategy of that name.
#
# WHAT IT DOES: during the London session, require a clean 21/50/200 SMMA stack (fast above
# medium above slow, each separated by a minimum distance), a trending ADX, live volatility, price
# on the correct side of the 200, and short-term momentum agreeing. When all of that holds, wait
# for a 3-line-strike or engulfing reversal pattern candle to CLOSE, then enter at the OPEN of the
# next candle - confirmed by RSI on the correct side of 50 and of its own 50-period SMMA. Stop is
# 2x the signal candle's range, target is 4x - a 1:2 risk:reward. At most ONE trade per instrument
# per day, whether it wins, loses, or is still open at day end.
#
# THIS PORT IS BUILT AGAINST TWO SOURCES, NOT ONE: the Pine v6 script itself, and a separate plain-
# English strategy writeup (from the same creator's material) describing what the script is meant
# to do. They disagree in two places, and the writeup - the stated INTENT - wins both times, because
# it is more specific than the code and the code's gap looks like an omission rather than a choice:
#
#   - The writeup is explicit that entry happens at the OPEN OF THE NEXT CANDLE after the signal
#     candle CLOSES ("wait for the candle to close... enter at open of next candle"). The Pine code
#     produces exactly this by NOT setting process_orders_on_close (defaults to false), so an order
#     placed on the signal bar fills on the following bar's open while stopLoss/takeProfit are
#     still computed from the signal bar's close and range. This looked, on first reading the Pine
#     alone, like an accidental mismatch between the entry price and the prices the stop/target were
#     computed from - it is not; it is the documented, intended design. FILL_AT_NEXT_OPEN defaults
#     to 1 for that reason. The realised R:R still is not a clean 2:1 as a result (0.30R-4.62R on
#     test data, per the entry/reference-price gap) - that is a real property of the strategy, not a
#     bug to hide. Set FILL_AT_NEXT_OPEN = 0 to fill at the signal close instead and isolate exactly
#     what the next-bar-open convention is worth.
#
#   - The writeup states "One Trade Per Day (Maximum)" as a hard, repeatedly emphasized rule ("NOT
#     50 trades, NOT 10 trades, just ONE") - and its own trade checklist asks "Is this my first
#     trade today?" as a precondition. The pasted Pine code never enforces this: it only blocks a
#     SECOND trade while the FIRST is still open (`strategy.position_size == 0`), so a fresh signal
#     later the same day, after the first trade has already closed by TP or SL, is free to fire
#     again in the code as written. MAX_TRADES_PER_DAY_PER_INSTRUMENT below enforces the documented
#     rule rather than the code's gap, since testing the documented strategy is the point of this
#     port. Set it to 0 to allow unlimited same-day re-entries and measure the Pine-literal version.
#
# TWO MORE THINGS THE PORT FOUND, both switchable or corrected as noted:
#
#   - THE WEEKDAY FILTER EXCLUDES FRIDAY AND INCLUDES SUNDAY. `dayofweek >= 1 and dayofweek <= 5`
#     looks like Monday-to-Friday. In Pine, dayofweek returns 1 for SUNDAY through 7 for Saturday,
#     so 1..5 is Sunday-to-Thursday. On forex the 07:00-15:00 Sunday window is closed, so the
#     practical effect is that the strategy trades Monday through THURSDAY and never trades Friday
#     - roughly a fifth of the available sessions, silently dropped, and never mentioned in the
#     writeup either (which says plainly "Days: Weekdays Only (Monday - Friday)"). Both sources
#     agree the INTENT is Monday-Friday, so WEEKDAY_FILTER_MODE defaults to 1 (real Mon-Fri). Set
#     it to 0 to reproduce the Pine's literal Sun-Thu/no-Friday behaviour, e.g. to reconcile a
#     result against the source's own TradingView report.
#
#   - minDist IS AN ABSOLUTE PRICE NUMBER, PER-PAIR IN THE WRITEUP BUT HARDCODED TO ONE VALUE IN
#     THE PINE. The Pine sets `minDist = 0.001` with a comment saying to adjust it per pair; the
#     writeup actually gives that table - EURUSD/GBPUSD/AUDUSD -> 0.001, USDJPY (and other JPY
#     crosses) -> 0.10. Ported here as MIN_SEPARATION_ABS_BY_INSTRUMENT, using the writeup's own
#     values for the four instruments this port trades, rather than either hardcoding the Pine's
#     single EURUSD-calibrated number across every pair (silently switching the filter off on
#     USDJPY) or inventing a synthetic percentage of my own.
#
# ALSO NOT PORTED, deliberately: the writeup's position-size inputs contradict each other in the
# Pine (`default_qty_type = strategy.percent_of_equity, default_qty_value = 100` overridden by
# `qty = 1` on every entry - one unit, not 100% of equity), which is why the source's own dollar
# equity curve doesn't reflect its own settings; irrelevant here since this port scores everything
# in R-multiples, which are size-independent by construction. `ema2 = ta.ema(close, 2)` is computed
# and plotted in the Pine but never used in a single condition, so it is not ported. There is no
# time-based exit of any kind beyond the one-trade-per-day cap, so a position opened near the end of
# a session can still be open well into a later one - kept faithfully, since it's a real property of
# the strategy worth measuring, not a bug to paper over.
#
# COST NOTE: the source assumes 0.03% commission per side, i.e. 0.06% round trip. That is roughly
# EIGHT TIMES this project's measured EURUSD figure (0.0074%). Costs are applied downstream by the
# webapp's per-instrument model rather than reproduced here, so this strategy is charged on the
# same basis as every other one in the catalog - but if the source's own report looked survivable
# at 0.06%, that is a point in its favour worth keeping in mind.

# !pip install --upgrade dukascopy-python -q   # uncomment this line in Colab

import calendar
import datetime
import logging
import os
import pickle

import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

from tqdm.auto import tqdm


class _SuppressDukascopyInfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING


logging.getLogger("DUKASCRIPT").addFilter(_SuppressDukascopyInfoFilter())

CACHE_DIR = "/content/drive/MyDrive/dukascopy_cache" if os.path.isdir("/content/drive/MyDrive") else "dukascopy_cache"
FETCH_CHUNK_MONTHS = 3

# The four instruments the writeup's own minDist table gives explicit values for, led by AUDUSD -
# "AUD/USD Recommended" is the writeup's own words for the pair this strategy was actually designed
# around. Not this project's usual EURUSD/GBPUSD/USDJPY/XAUUSD set - gold is never mentioned in the
# writeup at all, so it's dropped in favour of the pair the strategy is actually calibrated to.
INSTRUMENTS = [
    ("EURUSD", dki.INSTRUMENT_FX_MAJORS_EUR_USD),
    ("GBPUSD", dki.INSTRUMENT_FX_MAJORS_GBP_USD),
    ("AUDUSD", dki.INSTRUMENT_FX_MAJORS_AUD_USD),
    ("USDJPY", dki.INSTRUMENT_FX_MAJORS_USD_JPY),
]

FETCH_START = datetime.datetime(2022, 1, 1)
FETCH_END = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_MIN_5
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

# --- session ---
# The source uses Pine's `hour`, which reads in the CHART's timezone - unspecified, and for a
# forex chart usually the exchange default rather than London. Dukascopy bars are UTC, so 07:00-
# 15:00 UTC is used here: that is 08:00-16:00 London in summer and 07:00-15:00 in winter, i.e. the
# London session either way, which is plainly what "London Session Only" intended.
SESSION_START_HOUR = 7
SESSION_END_HOUR = 15
WEEKDAY_FILTER_MODE = 1   # 0 = as the source behaves (Sun-Thu, no Friday); 1 = Mon-Fri (default -
                           # this project trades the real London week, not the source's off-by-one).
                           # See note 1.

# --- trend stack ---
SMMA_FAST_LEN = 21
SMMA_MED_LEN = 50
SMMA_SLOW_LEN = 200
# The writeup's own per-pair minDist table (absolute price units, exactly as the Pine's `minDist`
# constant is used) - NOT a synthetic percentage. DEFAULT_MIN_SEPARATION_ABS is used for any
# instrument not in the table (shouldn't happen for INSTRUMENTS above; kept for safety only).
MIN_SEPARATION_ABS_BY_INSTRUMENT = {
    "EURUSD": 0.001,
    "GBPUSD": 0.001,
    "AUDUSD": 0.001,
    "USDJPY": 0.10,
}
DEFAULT_MIN_SEPARATION_ABS = 0.001

# --- regime filters ---
ADX_LEN = 14
ADX_MIN = 25.0
ATR_LEN = 14
ATR_AVG_LEN = 50
ATR_MIN_MULT = 0.7

# --- momentum ---
MOMENTUM_SMA_LEN = 5
MOMENTUM_LOOKBACK = 5

# --- RSI confirmation ---
RSI_LEN = 14
RSI_SMMA_LEN = 50

# --- risk ---
STOP_CANDLE_MULT = 2.0
TARGET_CANDLE_MULT = 4.0
FILL_AT_NEXT_OPEN = 1     # 1 = enter at the next bar's open after the signal bar closes (the
                           # writeup's documented, intended entry timing - default); 0 = fill at
                           # the signal bar's close instead, isolating what that convention is worth.
MIN_STOP_PCT = 0.005      # % of price. A signal candle with a near-zero range would otherwise
                           # produce a hairline stop and an enormous R-multiple off one bar.
MAX_TRADES_PER_DAY_PER_INSTRUMENT = 1   # the writeup's "One Trade Per Day (Maximum)" rule - the
                                          # Pine code itself doesn't enforce this (see header). 0
                                          # disables the cap and allows unlimited same-day re-entries,
                                          # matching the Pine exactly as written instead.

COST_PCT_SCENARIOS = [0.0, 0.01, 0.03, 0.05]


def _month_chunks(start, end, months_per_chunk):
    chunk_start = start
    while chunk_start < end:
        month_index = chunk_start.month - 1 + months_per_chunk
        target_year = chunk_start.year + month_index // 12
        target_month = month_index % 12 + 1
        target_day = min(chunk_start.day, calendar.monthrange(target_year, target_month)[1])
        chunk_end = chunk_start.replace(year=target_year, month=target_month, day=target_day)
        yield chunk_start, min(chunk_end, end)
        chunk_start = chunk_end


def fetch_instrument_data(label, instrument_const):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{label}_5min_{FETCH_START.date()}_{FETCH_END.date()}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    chunks = []
    for chunk_start, chunk_end in tqdm(list(_month_chunks(FETCH_START, FETCH_END, FETCH_CHUNK_MONTHS)),
                                        desc=f"{label}: downloading {DUKASCOPY_INTERVAL} bars", unit="chunk"):
        chunk = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE,
                                        chunk_start, chunk_end)
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return None
    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    return df


# ======================================================================================
# indicators. Every one of these matches the Pine built-in it stands in for, including the
# seeding rule - ta.rma/ta.sma are undefined for their warmup period rather than being
# approximated from a shorter window, and getting that wrong shifts every downstream value.
# ======================================================================================

def sma_series(values, length, start_index=0):
    """Simple moving average, None until `length` real values are available from start_index."""
    n = len(values)
    out = [None] * n
    window_sum = 0.0
    count = 0
    for i in range(start_index, n):
        if values[i] is None:
            return out
        window_sum += values[i]
        count += 1
        if count > length:
            window_sum -= values[i - length]
            count = length
        if count == length:
            out[i] = window_sum / length
    return out


def rma_series(values, length, start_index=0):
    """Pine's ta.rma / Wilder's smoothing / this project's SMMA: seeded with the simple average of
    the first `length` values, then recursive. `start_index` skips leading Nones (e.g. a true-range
    series that has no value on bar 0) so the seed is built from real numbers rather than from
    substituted zeros, which would silently drag the whole series toward zero."""
    n = len(values)
    out = [None] * n
    usable = [v for v in values[start_index:] if v is not None]
    if len(usable) < length:
        return out
    seed_end = start_index + length
    out[seed_end - 1] = sum(values[start_index:seed_end]) / length
    prev = out[seed_end - 1]
    for i in range(seed_end, n):
        prev = (prev * (length - 1) + values[i]) / length
        out[i] = prev
    return out


def true_range_series(highs, lows, closes):
    """None on bar 0 (no previous close), matching Pine's ta.tr behaviour for the purposes of
    every rma() built on top of it."""
    out = [None]
    for i in range(1, len(closes)):
        out.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
    return out


def atr_series(highs, lows, closes, length):
    return rma_series(true_range_series(highs, lows, closes), length, start_index=1)


def adx_series(highs, lows, closes, length):
    """Wilder's ADX, matching Pine's ta.dmi(length, length): directional movement smoothed by rma,
    normalised by the smoothed true range, then the DX itself smoothed by rma again."""
    n = len(closes)
    tr = true_range_series(highs, lows, closes)
    plus_dm = [None] * n
    minus_dm = [None] * n
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    trur = rma_series(tr, length, start_index=1)
    plus_sm = rma_series(plus_dm, length, start_index=1)
    minus_sm = rma_series(minus_dm, length, start_index=1)

    dx = [None] * n
    first_dx = None
    for i in range(n):
        if trur[i] is None or plus_sm[i] is None or minus_sm[i] is None or trur[i] == 0:
            continue
        plus = 100.0 * plus_sm[i] / trur[i]
        minus = 100.0 * minus_sm[i] / trur[i]
        total = plus + minus
        dx[i] = 100.0 * abs(plus - minus) / (total if total != 0 else 1.0)
        if first_dx is None:
            first_dx = i
    if first_dx is None:
        return [None] * n
    return rma_series(dx, length, start_index=first_dx)


def rsi_series(closes, length):
    """Pine's ta.rsi: rma of upward and downward changes."""
    n = len(closes)
    gains = [None] * n
    losses = [None] * n
    for i in range(1, n):
        change = closes[i] - closes[i - 1]
        gains[i] = max(change, 0.0)
        losses[i] = max(-change, 0.0)
    avg_gain = rma_series(gains, length, start_index=1)
    avg_loss = rma_series(losses, length, start_index=1)
    out = [None] * n
    for i in range(n):
        if avg_gain[i] is None or avg_loss[i] is None:
            continue
        if avg_loss[i] == 0:
            out[i] = 100.0
        elif avg_gain[i] == 0:
            out[i] = 0.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def in_session(ts):
    """The source's session gate, including its off-by-one weekday filter. Pine's dayofweek is
    1=Sunday..7=Saturday; pandas' weekday is 0=Monday..6=Sunday, so the source's `1..5` maps to
    pandas weekdays {6, 0, 1, 2, 3} = Sunday through Thursday. See note 1 in the header."""
    if not (SESSION_START_HOUR <= ts.hour < SESSION_END_HOUR):
        return False
    if WEEKDAY_FILTER_MODE:
        return ts.weekday() <= 4          # Monday-Friday, what the source meant
    return ts.weekday() in (6, 0, 1, 2, 3)  # Sunday-Thursday, what the source does


def backtest_instrument(label, df):
    opens, highs, lows, closes = (df["Open"].tolist(), df["High"].tolist(),
                                   df["Low"].tolist(), df["Close"].tolist())
    times = df.index
    n = len(closes)
    if n < SMMA_SLOW_LEN + RSI_SMMA_LEN + 10:
        return []

    smma_fast = rma_series(closes, SMMA_FAST_LEN)
    smma_med = rma_series(closes, SMMA_MED_LEN)
    smma_slow = rma_series(closes, SMMA_SLOW_LEN)
    adx = adx_series(highs, lows, closes, ADX_LEN)
    atr = atr_series(highs, lows, closes, ATR_LEN)
    atr_first = next((i for i, a in enumerate(atr) if a is not None), n)
    atr_avg = sma_series(atr, ATR_AVG_LEN, start_index=atr_first)
    mom_sma = sma_series(closes, MOMENTUM_SMA_LEN)
    rsi = rsi_series(closes, RSI_LEN)
    rsi_first = next((i for i, r in enumerate(rsi) if r is not None), n)
    rsi_smma = rma_series(rsi, RSI_SMMA_LEN, start_index=rsi_first)

    min_separation_abs = MIN_SEPARATION_ABS_BY_INSTRUMENT.get(label, DEFAULT_MIN_SEPARATION_ABS)

    trades = []
    position = None
    current_day = None
    trades_opened_today = 0
    i = max(SMMA_SLOW_LEN, MOMENTUM_LOOKBACK + 1, 3)
    while i < n:
        bar_day = times[i].date()
        if bar_day != current_day:
            current_day = bar_day
            trades_opened_today = 0

        # --- manage an open position first: the protective bracket is live during this bar and
        # fills intrabar, before any new signal is evaluated. Stop before target on a bar that
        # spans both - this project's standard pessimistic tie-break.
        if position is not None:
            is_long = position["side"] == "LONG"
            hit_stop = lows[i] <= position["stop"] if is_long else highs[i] >= position["stop"]
            hit_target = highs[i] >= position["target"] if is_long else lows[i] <= position["target"]
            if hit_stop:
                trades.append(_close_trade(position, -1.0, "SL", times[i]))
                position = None
            elif hit_target:
                trades.append(_close_trade(position, position["reward_r"], "TP", times[i]))
                position = None
            if position is not None:
                i += 1
                continue    # one position at a time - no signal is even looked at while in a trade

        if not in_session(times[i]):
            i += 1
            continue
        if MAX_TRADES_PER_DAY_PER_INSTRUMENT and trades_opened_today >= MAX_TRADES_PER_DAY_PER_INSTRUMENT:
            i += 1
            continue
        if any(v is None for v in (smma_fast[i], smma_med[i], smma_slow[i], adx[i], atr[i],
                                    atr_avg[i], mom_sma[i], rsi[i], rsi_smma[i])):
            i += 1
            continue

        # ABSOLUTE price distance, exactly like the Pine's own `minDist` constant - NOT scaled by
        # price - since these are the writeup's own per-pair literal values (0.001 for EURUSD/
        # GBPUSD/AUDUSD, 0.10 for USDJPY), already calibrated per instrument at the source.
        separation = min_separation_abs
        bull_stack = smma_fast[i] > smma_med[i] + separation and smma_med[i] > smma_slow[i] + separation
        bear_stack = smma_fast[i] < smma_med[i] - separation and smma_med[i] < smma_slow[i] - separation
        trending = adx[i] > ADX_MIN
        volatile = atr[i] > atr_avg[i] * ATR_MIN_MULT
        bull_mom = closes[i] > mom_sma[i] and closes[i] > closes[i - MOMENTUM_LOOKBACK]
        bear_mom = closes[i] < mom_sma[i] and closes[i] < closes[i - MOMENTUM_LOOKBACK]

        bull_quality = bull_stack and trending and closes[i] > smma_slow[i] and bull_mom and volatile
        bear_quality = bear_stack and trending and closes[i] < smma_slow[i] and bear_mom and volatile
        if not (bull_quality or bear_quality):
            i += 1
            continue

        # 3-line strike: three candles the same colour, then one closing through the open of the
        # candle immediately before it. Engulfing: opens beyond the previous candle and closes
        # through its open. Both taken straight from the source, including its exact comparisons.
        three_up = closes[i - 3] > opens[i - 3] and closes[i - 2] > opens[i - 2] and closes[i - 1] > opens[i - 1]
        three_down = closes[i - 3] < opens[i - 3] and closes[i - 2] < opens[i - 2] and closes[i - 1] < opens[i - 1]
        bull_strike = three_down and closes[i] > opens[i - 1]
        bear_strike = three_up and closes[i] < opens[i - 1]
        bull_engulf = opens[i] <= closes[i - 1] and opens[i] < opens[i - 1] and closes[i] > opens[i - 1]
        bear_engulf = opens[i] >= closes[i - 1] and opens[i] > opens[i - 1] and closes[i] < opens[i - 1]

        rsi_bull = rsi[i] > 50 and rsi[i] > rsi_smma[i]
        rsi_bear = rsi[i] < 50 and rsi[i] < rsi_smma[i]

        side = None
        if bull_quality and (bull_strike or bull_engulf) and rsi_bull:
            side, pattern = "LONG", _pattern_name(bull_strike, bull_engulf)
        elif bear_quality and (bear_strike or bear_engulf) and rsi_bear:
            side, pattern = "SHORT", _pattern_name(bear_strike, bear_engulf)
        if side is None:
            i += 1
            continue

        # Stop and target come off the SIGNAL bar's close and range; the fill (by default) lands on
        # the NEXT bar's open - the writeup's own documented entry timing (see header). Both prices
        # are fixed before the entry price is known, which is why the realised R:R isn't a clean 2:1.
        candle_size = highs[i] - lows[i]
        reference = closes[i]
        if side == "LONG":
            stop = reference - candle_size * STOP_CANDLE_MULT
            target = reference + candle_size * TARGET_CANDLE_MULT
        else:
            stop = reference + candle_size * STOP_CANDLE_MULT
            target = reference - candle_size * TARGET_CANDLE_MULT

        fill_index = i + 1 if FILL_AT_NEXT_OPEN else i
        if fill_index >= n:
            break
        entry = opens[fill_index] if FILL_AT_NEXT_OPEN else closes[i]

        risk = entry - stop if side == "LONG" else stop - entry
        reward = target - entry if side == "LONG" else entry - target
        # A gap through the intended stop leaves an entry already past it, or a target already
        # reached. Neither is a tradeable setup - skipped rather than scored as a fake instant win.
        if risk <= 0 or reward <= 0 or (risk / entry) * 100.0 < MIN_STOP_PCT:
            i += 1
            continue

        position = {"side": side, "entry": entry, "stop": stop, "target": target, "risk": risk,
                     "reward_r": reward / risk, "pattern": pattern, "entry_time": times[fill_index]}
        trades_opened_today += 1   # counts against the SIGNAL bar's day - see MAX_TRADES_PER_DAY_
                                     # PER_INSTRUMENT above; the entry itself may land moments into
                                     # the next bar but not, in practice, a different calendar day
        i = fill_index    # management begins on the fill bar itself, same as the broker emulator

    if position is not None:
        # No time-based exit exists in the source, so a position open at the end of the data is
        # marked to the last close rather than being dropped (dropping it would quietly discard the
        # worst trades, which are exactly the ones that never resolve).
        last_close = closes[n - 1]
        pnl = (last_close - position["entry"]) if position["side"] == "LONG" else (position["entry"] - last_close)
        trades.append(_close_trade(position, pnl / position["risk"], "FLAT", times[n - 1]))

    return trades


def _pattern_name(is_strike, is_engulfing):
    if is_strike and is_engulfing:
        return "both"
    return "3-line strike" if is_strike else "engulfing"


def _close_trade(position, exit_r, outcome, exit_time):
    # stop_pct and date are REQUIRED by the webapp - without stop_pct the cost model silently skips
    # the strategy (scoring it gross while everything else is net) and without date the
    # out-of-sample holdout degrades to a positional split. Both were real silent bugs in this
    # repo; webapp/test_strategy_contract.py now fails the build if either is missing.
    return {
        "side": position["side"], "outcome": outcome, "r": exit_r,
        "date": position["entry_time"].date(),
        "entry": position["entry"], "sl_distance": position["risk"],
        "stop_pct": position["risk"] / position["entry"],
        "pattern": position["pattern"],
    }


def main():
    print(f"TMA TREND SCALPER - {FETCH_START.date()} to {FETCH_END.date()}")
    print(f"Session {SESSION_START_HOUR:02d}:00-{SESSION_END_HOUR:02d}:00 UTC, weekday mode "
          f"{'Mon-Fri (corrected)' if WEEKDAY_FILTER_MODE else 'Sun-Thu (as the source behaves)'}, "
          f"fill at {'next open' if FILL_AT_NEXT_OPEN else 'signal close'}\n")

    all_trades = []
    for label, const in tqdm(INSTRUMENTS, desc="Instruments", unit="instrument"):
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
    wins = [t for t in all_trades if t["r"] > 0]
    print(f"\n{n} trades, {total_r:+.2f}R total, {total_r / n:+.4f}R/trade (BEFORE costs), "
          f"{len(wins) / n * 100:.1f}% winners")

    realised = [t["r"] for t in all_trades if t["outcome"] == "TP"]
    if realised:
        print(f"\nActual reward on winners: {min(realised):.2f}R to {max(realised):.2f}R "
              f"(mean {sum(realised) / len(realised):.2f}R) - the source intends a flat 2R, and the "
              f"spread here is the next-bar-open fill moving the entry away from the price the stop "
              f"and target were calculated from.")

    print("\nBY PATTERN:")
    for pattern in ("3-line strike", "engulfing", "both"):
        subset = [t for t in all_trades if t["pattern"] == pattern]
        if subset:
            sub_r = sum(t["r"] for t in subset)
            print(f"  {pattern:<14} {len(subset):>5} trades, {sub_r:+9.2f}R, {sub_r / len(subset):+.4f}R/trade")

    print("\nCOST SENSITIVITY (% of entry price, round trip):")
    for cost_pct in COST_PCT_SCENARIOS:
        adj = sum(t["r"] - (cost_pct / 100.0) / t["stop_pct"] for t in all_trades)
        print(f"  {cost_pct:.2f}%: {adj:+.2f}R total, {adj / n:+.4f}R/trade")
    print("\nThe source's own settings assume 0.03% per side (0.06% round trip) - read the 0.05% row "
          "as roughly its own assumption, and the 0.01% row as closer to what a prop firm raw-spread "
          "account actually charges.")


if __name__ == "__main__":
    main()
