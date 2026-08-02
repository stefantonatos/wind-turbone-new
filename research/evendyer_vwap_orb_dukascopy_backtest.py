# "EvenDyer VWAP FREE - Scam Or Slam" backtest via Dukascopy, 2016-2025.
#
# A TradingView Pine v6 strategy from the same "Scam or Slam" series as
# day_trading_rauf_dukascopy_backtest.py. LESSON CARRIED FORWARD FROM THAT
# SCRIPT: Rauf was tested on forex/gold and came back clearly negative even
# though TradingView's own strategy tester showed it positive - the user
# traced this to Rauf actually being an index-futures strategy, and testing
# it on the wrong asset class entirely undermined the comparison. THIS
# script is even less ambiguous about its intended market: its opening-
# range default window is "0930-1000" New York time - the literal first 30
# minutes of the US stock market's regular session. There is no forex
# reading of that default. So this is tested on Dukascopy's US index CFDs
# (SP500 / NASDAQ100 / DOWJONES - the same three already used successfully
# in orb_indices_dukascopy_backtest.py), not currency pairs.
#
# Distinct from every other script in this project's "range then confirm"
# family (Rauf's 3-candle reversal, ict_po3's fade, ict_silver_bullet's
# sweep -> MSS -> FVG chain): this one enters WITH an opening-range break's
# direction, not against it, and the confirmation is a VWAP pullback-then-
# reclaim, not a candle pattern or fair value gap. Stops/targets also come
# from real market-structure swing pivots (two independent lookback
# lengths), not a fixed R:R or the opposite side of a range - the first
# script here to do that.
#
# OPERATIONALIZING the Pine source's default configuration only:
#   1. ORB: high/low of price during ORB_SESSION (0930-1000 NY, 30 min).
#      "Ready" (usable for the rest of the day) once that window ends.
#   2. NY SESSION VWAP: volume-weighted HLC3, accumulating fresh from the
#      START of NY_VWAP_SESSION (0930-1800 NY) each day and going flat/na
#      outside that window - this is the ONLY vwap the entry logic uses.
#      The Pine source also draws a "Midnight VWAP" (accumulates from
#      00:00 NY) but it is purely a visual overlay never referenced by any
#      entry/exit condition in the source - intentionally not built here.
#   3. ENTRY WINDOW: entries only allowed inside ENTRY_SESSION (1000-1800
#      NY), which starts exactly as the ORB window ends (script default).
#   4. ORB BREAK TRACKING: once "can track break" (ORB ready, NOT inside
#      the ORB window itself, ORB high/low known), a close above the ORB
#      high sets a STICKY longOrbBroken flag for the rest of the day; a
#      close below the ORB low sets shortOrbBroken. Both are independent -
#      a day can see both flip true at different times.
#   5. VWAP RETEST - "Close Through Then Reclaim" mode only (the script's
#      default; the alternative "Touch VWAP" mode is intentionally not
#      built, same default-only scope as every prior script here). Once in
#      the entry window with the matching ORB-break flag already true:
#        LONG:  a close below vwap sets a sticky longClosedThroughVwap
#               flag; the long signal fires the bar that flag is true AND
#               the close is back above vwap (break up, pull back through
#               vwap, reclaim vwap upward - a continuation-after-pullback
#               model, not a plain breakout or a fade).
#        SHORT: mirror image (close above vwap sets the sticky flag, entry
#               fires on close back below vwap).
#   6. STOP/TARGET FROM CONFIRMED SWING PIVOTS - genuinely different from
#      every prior script here (no fixed R:R, no ATR, no "opposite side of
#      a range"). TWO separate pivot passes, using the exact confirmed-
#      pivot-with-no-lookahead pattern from
#      support_resistance_zone_bounce_dukascopy_backtest.py's
#      find_confirmed_pivots (a pivot at index i is only usable starting at
#      bar i + lookback, once both sides exist to confirm it):
#        - STOP_SWING_LEN (20 bars each side, default) for the stop.
#        - TARGET_SWING_LEN (5 bars each side, default - tighter/shorter
#          than the stop's) for the target.
#      LONG: stop = most recent confirmed swing LOW (stop lookback) minus
#      BUFFER_AMOUNT; target = most recent confirmed swing HIGH (target
#      lookback). SHORT is the mirror. BUFFER_AMOUNT default is 0 (Pine's
#      own default - no padding beyond the swing extreme), kept as-is
#      rather than inventing a nonzero value. Pivot state is tracked
#      CONTINUOUSLY across the whole bar series (matching Pine's `var`
#      persistence for ta.pivothigh/ta.pivotlow) - NOT reset at day
#      boundaries the way the ORB/VWAP/break-flag state is.
#      VALIDITY CHECK before taking the trade (this matters - the two
#      swing lookbacks are computed independently and don't know about
#      each other, so their combination can produce a nonsensical
#      geometry): LONG requires stop < close < target; SHORT requires
#      target < close < stop. If not, the trade is skipped entirely rather
#      than forced through broken - same convention as the S/R zone bounce
#      script's opposite-edge validity check.
#   7. TRADE FREQUENCY: MAX_TRADES_PER_DAY = 1 (script default) AND only
#      enter when flat - both must hold. On any entry, both ORB-break flags
#      and both VWAP-retest sticky flags reset to false for the rest of the
#      day (matters in principle for a higher MAX_TRADES_PER_DAY; with the
#      default of 1 plus the flat-only gate it just means one trade/day).
#   8. FORCED CLOSE: the Pine source has NO time-based force-close - its
#      strategy.exit() stop/limit orders stay live indefinitely once
#      placed. For a clean, tractable daily backtest, this port forces any
#      open position closed at FORCE_CLOSE_TIME = 18:00 NY (the entry
#      window's own end time) the same day it was opened, market-on-close.
#      This is a deliberate simplification (documented, not silently
#      assumed) - it's consistent with how every other script in this
#      project's ICT/ORB family closes out same-day, and the Pine source's
#      own session config (everything scoped to NY trading hours) implies
#      same-day trading is the intent even though the code doesn't literally
#      enforce it. Forced closes are recorded "FLAT" with
#      r = pnl / original_stop_distance, this project's standard.
#   9. COSTS: the Pine source's own strategy() declaration models real
#      costs (commission_value = 2.50 cash-per-contract, slippage = 1
#      tick) - unlike most strategies found for this project, this one's
#      author actually accounted for them. This port does NOT replicate
#      that (a fixed cash-per-contract commission doesn't translate
#      meaningfully into this project's %-of-price R-multiple framework),
#      which, if anything, makes this port's numbers optimistic relative
#      to the original's own modeling - flagged the same way every other
#      script here flags "no commission/spread/slippage modeled."
#
# VOLUME / VWAP CAVEAT (read before trusting any VWAP-gated signal here):
# VWAP is volume-weighted by definition. Dukascopy's fetch() DataFrame has
# a "volume" column for index CFDs same as FX (see orb_indices_dukascopy_
# backtest.py's own volume-filter caveat), but spot/CFD index feeds often
# carry zero or otherwise degenerate volume since there's no centralized
# exchange tape behind them - unlike real futures. This port tries the
# literal volume-weighted formula FIRST; determine_vwap_weights() checks
# the fetched volume for basic usability (a meaningful nonzero fraction and
# more than one distinct nonzero value) and prints a one-time diagnostic
# per instrument. If volume looks unusable, it falls back to an equal-
# weighted TWAP (every bar weighted 1.0 instead of by volume) as an
# honest, clearly-flagged approximation - NOT a silent substitution. No
# live network access exists in this sandbox, so which branch real
# Dukascopy index data actually takes could not be verified here; the
# mocked tests below exercise both branches so the fallback logic itself
# is at least known to work correctly.
#
# No parameter grid search - one fixed, default configuration, tested for
# whether it holds up (split-period check), not tuned to this data.

# !pip install --upgrade dukascopy-python tqdm -q   # uncomment this line in Colab

import calendar
import datetime
import logging
import os
import pickle

import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

from tqdm.auto import tqdm   # auto-picks the Colab/Jupyter widget bar when available, a plain terminal bar otherwise

# dukascopy_python logs an "INFO:DUKASCRIPT:current timestamp:..." line for every internal
# download chunk - useful for debugging a stuck fetch, just noisy for normal runs. A plain
# logger.setLevel(WARNING) does NOT work here: the library's own fetch() call resets the
# "DUKASCRIPT" logger's level back to INFO internally on every single call (confirmed by
# reading dukascopy_python/__init__.py's _get_custom_logger - it unconditionally calls
# logger.setLevel(...) each time), silently undoing a one-time setLevel before it ever helps.
# A logging Filter survives that reset (the library never touches .filters), so that's what's
# used instead. Remove this filter to see the raw log again.
class _SuppressDukascopyInfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING


logging.getLogger("DUKASCRIPT").addFilter(_SuppressDukascopyInfoFilter())

# Fetched data is cached to disk per (instrument, interval, date range) - the FIRST run of a
# given range still has to download it all, but every run after that (e.g. after tweaking a
# parameter below, or a DIFFERENT script that happens to need the same instrument/interval/
# range) loads from disk instantly instead of re-downloading ~650k+ bars per instrument. Auto-
# detects a mounted Google Drive (run `from google.colab import drive; drive.mount('/content/
# drive')` once at the top of your notebook, before this cell) and uses that instead of the
# ephemeral local disk if present - this is what makes the cache survive runtime resets AND
# get shared across every script in this project that uses the same instruments/date range,
# not just repeat runs of this one file. Falls back to a local (session-only) cache if Drive
# isn't mounted, so this still works without any setup, just without the persistence.
CACHE_DIR = "/content/drive/MyDrive/dukascopy_cache" if os.path.isdir("/content/drive/MyDrive") else "dukascopy_cache"
FETCH_CHUNK_MONTHS = 3   # how finely to split the download for progress-bar granularity

INSTRUMENTS = [
    ("SP500", dki.INSTRUMENT_IDX_AMERICA_E_SANDP_500),
    ("NASDAQ100", dki.INSTRUMENT_IDX_AMERICA_E_NQ_100),
    ("DOWJONES", dki.INSTRUMENT_IDX_AMERICA_E_D_J_IND),
]

FETCH_START = datetime.datetime(2016, 1, 1)
FETCH_END = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_MIN_5
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

NY_TZ = "America/New_York"

# --- session windows, all NY time (script defaults) - half-open [start, end) throughout,
# matching this project's existing session-window convention (see orb_indices_dukascopy_backtest.py) ---
ORB_SESSION = (pd.Timestamp("09:30").time(), pd.Timestamp("10:00").time())
NY_VWAP_SESSION = (pd.Timestamp("09:30").time(), pd.Timestamp("18:00").time())
ENTRY_SESSION = (pd.Timestamp("10:00").time(), pd.Timestamp("18:00").time())
FORCE_CLOSE_TIME = pd.Timestamp("18:00").time()   # see header note 8 - not in the Pine source, a documented simplification

STOP_SWING_LEN = 20     # bars each side, confirmed-pivot lookback used for the STOP
TARGET_SWING_LEN = 5    # bars each side, confirmed-pivot lookback used for the TARGET (tighter than the stop's)
BUFFER_AMOUNT = 0.0     # price units beyond the swing extreme - Pine default is 0, kept as-is (see header note 6)
MAX_TRADES_PER_DAY = 1

# Illustrative round-trip cost scenarios, as a percentage of entry price - NOT measured real spread
# data, just a few bracketing assumptions to see how much cost this edge can absorb before it
# disappears, same convention introduced in support_resistance_zone_bounce_dukascopy_backtest.py.
COST_PCT_SCENARIOS = [0.0, 0.01, 0.03, 0.05]

VOLUME_USABLE_MIN_NONZERO_FRACTION = 0.5   # heuristic threshold - see determine_vwap_weights()


def to_ny_time(index):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(NY_TZ)


def _month_chunks(start, end, months_per_chunk):
    """Splits [start, end) into months_per_chunk-sized pieces, advancing by calendar month
    while keeping the same day-of-month as `start` where possible. BUG FIX: a plain
    `chunk_start.replace(month=...)` raises ValueError("day is out of range for month")
    whenever start falls on the 29th-31st and the target month is shorter (e.g. start on
    Jan 31 + 3 months -> April 31, which doesn't exist) - this silently broke ANY date range
    whose start date landed on one of those days, which is a completely ordinary thing for a
    user-picked "last N days" range to do. Clamps to the target month's actual last day
    instead, the same convention date-arithmetic libraries like dateutil use for
    add-N-months."""
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
    """Downloads in FETCH_CHUNK_MONTHS-sized pieces (shows real progress instead of one long
    silent call) and caches the combined result to disk, so re-running this script after
    changing a strategy parameter below loads instantly instead of re-downloading everything."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{label}_5min_{FETCH_START.date()}_{FETCH_END.date()}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    chunks = []
    chunk_bounds = list(_month_chunks(FETCH_START, FETCH_END, FETCH_CHUNK_MONTHS))
    for chunk_start, chunk_end in tqdm(chunk_bounds, desc=f"{label}: downloading {DUKASCOPY_INTERVAL} bars",
                                        unit="chunk"):
        chunk = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE,
                                        chunk_start, chunk_end)
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return None
    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")].sort_index()   # chunk boundaries may overlap by one bar
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df.index = to_ny_time(df.index)

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    return df


def find_confirmed_pivots(highs, lows, lookback):
    """Returns two dicts: {confirmed_at_index: (pivot_index, price)} for
    swing highs and swing lows. A pivot at index i is only placed at key
    i + lookback - the earliest point its formation could actually be
    known, since confirming it requires the `lookback` bars AFTER it.
    Adapted verbatim from support_resistance_zone_bounce_dukascopy_backtest.py."""
    n = len(highs)
    swing_highs, swing_lows = {}, {}
    for i in range(lookback, n - lookback):
        window_highs = highs[i - lookback:i + lookback + 1]
        if highs[i] == max(window_highs) and window_highs.count(highs[i]) == 1:
            swing_highs[i + lookback] = (i, highs[i])
        window_lows = lows[i - lookback:i + lookback + 1]
        if lows[i] == min(window_lows) and window_lows.count(lows[i]) == 1:
            swing_lows[i + lookback] = (i, lows[i])
    return swing_highs, swing_lows


def determine_vwap_weights(volumes, label):
    """VWAP is volume-weighted by definition - if the fetched volume series looks unusable
    (mostly zero, or a single constant nonzero value carrying no real information), fall back
    to an equal-weighted TWAP (every bar weighted 1.0) instead of feeding a degenerate series
    into the VWAP formula. See the file header's VOLUME / VWAP CAVEAT for why this can't be
    verified against real data in this sandbox."""
    n = len(volumes)
    if n == 0:
        return [], False, f"[{label}] volume diagnostic: no bars to check"
    nonzero = [v for v in volumes if v and v > 0]
    nonzero_frac = len(nonzero) / n
    distinct_nonzero = len(set(nonzero))
    usable = nonzero_frac >= VOLUME_USABLE_MIN_NONZERO_FRACTION and distinct_nonzero > 1
    diagnostic = (f"[{label}] volume diagnostic: {nonzero_frac * 100:.1f}% of bars have volume > 0, "
                  f"{distinct_nonzero} distinct nonzero value(s) -> "
                  + ("using real volume-weighted VWAP" if usable
                     else "volume looks degenerate, falling back to equal-weighted TWAP"))
    weights = [float(v) for v in volumes] if usable else [1.0] * n
    return weights, usable, diagnostic


def backtest_instrument(label, df, verbose=True, record_trace=False):
    """Core strategy logic - see the file header for the full rule set. Returns
    (trades, used_real_volume, volume_diagnostic_string), plus a per-bar trace list of internal
    signal/flag state as a 4th element when record_trace=True (testing hook only - main() never
    sets this, so its normal 3-value unpacking is unaffected)."""
    highs = df["High"].tolist()
    lows = df["Low"].tolist()
    closes = df["Close"].tolist()
    volumes_raw = df["volume"].tolist() if "volume" in df.columns else [0] * len(df)
    times = df.index
    n = len(closes)
    src = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(n)]   # hlc3, the Pine source's VWAP source

    weights, used_real_volume, diag = determine_vwap_weights(volumes_raw, label)
    if verbose:
        print(f"  {diag}")

    stop_swing_highs, stop_swing_lows = find_confirmed_pivots(highs, lows, STOP_SWING_LEN)
    target_swing_highs, target_swing_lows = find_confirmed_pivots(highs, lows, TARGET_SWING_LEN)

    trades = []
    trace = [] if record_trace else None
    current_day = None
    open_trade = None

    orb_high = orb_low = None
    orb_ready = False
    prev_in_orb = False
    prev_in_vwap = False

    ny_cum_pv = ny_cum_vol = None

    trades_today = 0
    long_orb_broken = short_orb_broken = False
    long_closed_through = short_closed_through = False

    # swing-pivot state is CONTINUOUS across the whole series (matches Pine's `var` persistence
    # for ta.pivothigh/ta.pivotlow) - deliberately NOT reset at day boundaries below.
    last_stop_swing_high = last_stop_swing_low = None
    last_target_swing_high = last_target_swing_low = None

    for i in range(n):
        t = times[i]
        today = t.date()
        tod = t.time()

        if today != current_day:
            if open_trade is not None:
                # defensive only: a position should always be resolved by the prior day's
                # FORCE_CLOSE_TIME below - guards against a data gap leaving a trade stranded
                # across the day boundary, same pattern as day_trading_rauf_dukascopy_backtest.py.
                side = open_trade["side"]
                last_close = closes[i - 1]
                pnl = (last_close - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - last_close)
                trades.append({"side": side, "outcome": "FLAT", "r": pnl / open_trade["sl_distance"],
                                "date": times[i - 1].date(),
                                "stop_pct": open_trade["sl_distance"] / open_trade["entry"]})
                open_trade = None
            current_day = today
            orb_high = orb_low = None
            orb_ready = False
            trades_today = 0
            long_orb_broken = short_orb_broken = False
            long_closed_through = short_closed_through = False

        # --- manage an already-open trade: stop / target / forced close (stop checked first
        # on an ambiguous same-bar hit, this project's standard pessimistic tie-break) ---
        if open_trade is not None:
            side = open_trade["side"]
            stop, target = open_trade["stop"], open_trade["target"]
            hi, lo = highs[i], lows[i]
            hit_stop = lo <= stop if side == "LONG" else hi >= stop
            hit_target = hi >= target if side == "LONG" else lo <= target
            stop_pct = open_trade["sl_distance"] / open_trade["entry"]
            if hit_stop:
                trades.append({"side": side, "outcome": "SL", "r": -1.0, "date": today, "stop_pct": stop_pct})
                open_trade = None
            elif hit_target:
                trades.append({"side": side, "outcome": "TP", "r": open_trade["reward_risk"], "date": today,
                               "stop_pct": stop_pct})
                open_trade = None
            elif tod >= FORCE_CLOSE_TIME:
                pnl = (closes[i] - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - closes[i])
                trades.append({"side": side, "outcome": "FLAT", "r": pnl / open_trade["sl_distance"], "date": today,
                               "stop_pct": stop_pct})
                open_trade = None

        in_orb = ORB_SESSION[0] <= tod < ORB_SESSION[1]
        in_vwap_sess = NY_VWAP_SESSION[0] <= tod < NY_VWAP_SESSION[1]
        in_entry_sess = ENTRY_SESSION[0] <= tod < ENTRY_SESSION[1]

        orb_start = in_orb and not prev_in_orb
        orb_end = (not in_orb) and prev_in_orb
        vwap_start = in_vwap_sess and not prev_in_vwap

        # --- ORB range ---
        if orb_start:
            orb_high, orb_low = highs[i], lows[i]
        if in_orb:
            orb_high = highs[i] if orb_high is None else max(orb_high, highs[i])
            orb_low = lows[i] if orb_low is None else min(orb_low, lows[i])
        if orb_end:
            orb_ready = True

        # --- NY session VWAP ---
        if vwap_start:
            ny_cum_pv = src[i] * weights[i]
            ny_cum_vol = weights[i]
        elif in_vwap_sess:
            if ny_cum_vol is None:
                ny_cum_pv, ny_cum_vol = src[i] * weights[i], weights[i]   # defensive; shouldn't happen with clean data
            else:
                ny_cum_pv += src[i] * weights[i]
                ny_cum_vol += weights[i]
        else:
            ny_cum_pv = ny_cum_vol = None
        ny_vwap = (ny_cum_pv / ny_cum_vol) if (in_vwap_sess and ny_cum_vol and ny_cum_vol > 0) else None

        # --- swing pivots become usable the instant they're confirmed at this index ---
        if i in stop_swing_highs:
            last_stop_swing_high = stop_swing_highs[i][1]
        if i in stop_swing_lows:
            last_stop_swing_low = stop_swing_lows[i][1]
        if i in target_swing_highs:
            last_target_swing_high = target_swing_highs[i][1]
        if i in target_swing_lows:
            last_target_swing_low = target_swing_lows[i][1]

        # --- ORB break tracking (sticky for the rest of the day) ---
        can_track_break = orb_ready and not in_orb and orb_high is not None and orb_low is not None
        can_look_for_entry = can_track_break and in_vwap_sess and in_entry_sess and ny_vwap is not None

        if can_track_break and closes[i] > orb_high:
            long_orb_broken = True
        if can_track_break and closes[i] < orb_low:
            short_orb_broken = True

        # --- VWAP retest, "Close Through Then Reclaim" mode ---
        long_signal = False
        short_signal = False

        if can_look_for_entry and long_orb_broken:
            if closes[i] < ny_vwap:
                long_closed_through = True
            long_signal = long_closed_through and closes[i] > ny_vwap

        if can_look_for_entry and short_orb_broken:
            if closes[i] > ny_vwap:
                short_closed_through = True
            short_signal = short_closed_through and closes[i] < ny_vwap

        # --- stop/target from confirmed swing pivots, with validity check ---
        long_stop = (last_stop_swing_low - BUFFER_AMOUNT) if last_stop_swing_low is not None else None
        long_target = last_target_swing_high
        short_stop = (last_stop_swing_high + BUFFER_AMOUNT) if last_stop_swing_high is not None else None
        short_target = last_target_swing_low

        price = closes[i]
        valid_long = (long_stop is not None and long_target is not None and long_stop < price < long_target)
        valid_short = (short_stop is not None and short_target is not None and short_target < price < short_stop)

        can_trade = open_trade is None and trades_today < MAX_TRADES_PER_DAY
        long_entry = long_signal and valid_long and can_trade
        short_entry = short_signal and valid_short and can_trade

        if long_entry:
            sl_distance = price - long_stop
            reward_risk = (long_target - price) / sl_distance
            open_trade = {"side": "LONG", "entry": price, "stop": long_stop, "target": long_target,
                          "sl_distance": sl_distance, "reward_risk": reward_risk}
            trades_today += 1
            long_orb_broken = short_orb_broken = False
            long_closed_through = short_closed_through = False
        elif short_entry:
            sl_distance = short_stop - price
            reward_risk = (price - short_target) / sl_distance
            open_trade = {"side": "SHORT", "entry": price, "stop": short_stop, "target": short_target,
                          "sl_distance": sl_distance, "reward_risk": reward_risk}
            trades_today += 1
            long_orb_broken = short_orb_broken = False
            long_closed_through = short_closed_through = False

        prev_in_orb = in_orb
        prev_in_vwap = in_vwap_sess

        if record_trace:
            trace.append({"time": t, "long_signal": long_signal, "short_signal": short_signal,
                          "long_orb_broken": long_orb_broken, "short_orb_broken": short_orb_broken,
                          "long_closed_through": long_closed_through,
                          "short_closed_through": short_closed_through, "ny_vwap": ny_vwap,
                          "in_orb": in_orb, "in_entry_sess": in_entry_sess,
                          "long_stop": long_stop, "long_target": long_target,
                          "short_stop": short_stop, "short_target": short_target,
                          "valid_long": valid_long, "valid_short": valid_short})

    if open_trade is not None:
        # the series ended mid-trade (e.g. FETCH_END cuts off before FORCE_CLOSE_TIME on the
        # last day) - mark to the last available close rather than drop the trade silently.
        side = open_trade["side"]
        last_close = closes[-1]
        pnl = (last_close - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - last_close)
        trades.append({"side": side, "outcome": "FLAT", "r": pnl / open_trade["sl_distance"],
                        "date": times[-1].date(), "stop_pct": open_trade["sl_distance"] / open_trade["entry"]})

    if record_trace:
        return trades, used_real_volume, diag, trace
    return trades, used_real_volume, diag


def main():
    years = (FETCH_END - FETCH_START).days / 365
    print(f"Downloading {len(INSTRUMENTS)} instruments from Dukascopy over ~{years:.0f} years "
          f"({FETCH_START.date()} to {FETCH_END.date()}) - cached to disk after the first run, "
          f"so this is only slow once.\n")

    data = {}
    for label, instrument_const in tqdm(INSTRUMENTS, desc="Instruments", unit="instrument"):
        try:
            df = fetch_instrument_data(label, instrument_const)
        except Exception as exc:
            print(f"{label}: failed ({exc})")
            continue
        if df is None:
            print(f"{label}: no data")
            continue
        data[label] = df
        print(f"{label}: {len(df)} bars")

    if not data:
        print("No data downloaded - check output above.")
        return

    all_trades = []
    for label, df in data.items():
        trades, used_real_volume, diag = backtest_instrument(label, df)
        for t in trades:
            t["instrument"] = label
        all_trades.extend(trades)

    if not all_trades:
        print("\nNo trades at all - an ORB break followed by a valid VWAP close-through-then-reclaim "
              "with a sane swing-pivot stop/target never occurred in this data.")
        return

    total_r = sum(t["r"] for t in all_trades)
    n_trades = len(all_trades)
    print("\n" + "=" * 70)
    print(f"EVENDYER VWAP ORB (Scam Or Slam) - {n_trades} trades, {FETCH_START.date()} to {FETCH_END.date()}")
    print("=" * 70)
    print(f"Total R: {total_r:+.2f}   Avg R/trade: {total_r/n_trades:+.4f}")

    tp = sum(1 for t in all_trades if t["outcome"] == "TP")
    sl = sum(1 for t in all_trades if t["outcome"] == "SL")
    flat = sum(1 for t in all_trades if t["outcome"] == "FLAT")
    print(f"Outcome breakdown: TP {tp} ({tp/n_trades*100:.1f}%)  SL {sl} ({sl/n_trades*100:.1f}%)  "
          f"FLAT {flat} ({flat/n_trades*100:.1f}%)")

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
    for lbl, half in [("First half", first_half), ("Second half", second_half)]:
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
        print(f"  {lbl} ({half[0]['date']} to {half[-1]['date']}): {n} trades, {r:+.2f}R, "
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

    print("\nNo commission/spread/slippage modeled (the original Pine source DID model $2.50/contract "
          "commission + 1 tick slippage in its own strategy() declaration - not replicated here, see "
          "header note 9). Entry is a simulated market order at the close of the bar that reclaims VWAP - "
          "real fills would be worse.")


if __name__ == "__main__":
    main()
