# Climax Volume Reversal backtest via Dukascopy, 2016-2025.
#
# SOURCE HONESTY UP FRONT: this is ported from a strategy-pitch commentary
# (paraphrased, not this project's own idea) that claims "123% return over
# 11 years, 11% max drawdown" for the underlying rule set. That claim is
# NOT reproduced or assumed here in any way - it's exactly the kind of
# marketed backtest number this project treats with total skepticism (see
# day_trading_rauf_dukascopy_backtest.py's header for the same posture
# toward a different source). The RULES below are ported faithfully; the
# RESULT reported by this script's own main() is this project's own,
# independently computed number on real Dukascopy data, whatever it turns
# out to be - not a reproduction of the source's figure.
#
# THE IDEA: a volume-and-range EXHAUSTION candle at the peak of a clear
# directional run, followed by a reversal - "climax" in the classic sense
# (a blow-off top/bottom on a candle that's unusually large and unusually
# busy relative to recent bars), then trade the reversal once it's
# confirmed, with a stop-entry (not a market order) so the trade is only
# actually taken if price genuinely continues through the confirmation
# level on a later bar.
#
# OPERATIONALIZING each piece (the source's own phrasing was loose in
# places - see each numbered note for exactly which interpretation was
# picked and why, rather than guessing silently):
#
#   1. TIMEFRAME: dukascopy_python.INTERVAL_MIN_15 fetched DIRECTLY (not
#      resampled from 5-min bars used elsewhere in this project) - the
#      source specifically says "measured and built on the 15[-minute
#      chart]", and resampling from 5-min would produce subtly different
#      OHLC than a native 15-min fetch would (5-min bar boundaries don't
#      necessarily line up with a resample's aggregation in exactly the
#      way Dukascopy's own native 15-min bars are built), so this fetches
#      the native interval directly rather than approximating it.
#
#   2. CLIMAX CANDLE - RANGE CONDITION: (High - Low) of the candle is at
#      least CLIMAX_RANGE_ATR_MULT (3.0) times the 15-min ATR(14), Wilder-
#      smoothed - the exact same recursive formula as compute_atr_series()
#      in orb_indices_optimization_and_ml.py (and duplicated verbatim here,
#      per this project's "duplicate, don't import, cite the source"
#      convention - see donchian_turtle_breakout_dukascopy_backtest.py for
#      precedent). The baseline ATR used is the value AS OF THE END OF THE
#      PRIOR BAR (atr[c-1], not atr[c]) - using the climax candle's own ATR
#      value would let its own huge true range partially inflate the very
#      baseline it's being judged against (Wilder smoothing folds each
#      bar's TR into the next ATR value), which is a subtle lookahead-
#      adjacent leak the same way donchian's ATR is deliberately shifted by
#      a full period for its entry-day value.
#
#   3. CLIMAX CANDLE - PRIOR-RUN CONDITION ("at the peak of a bullish/
#      bearish run"): this is the single most ambiguous phrase in the
#      source, so it's operationalized as a concrete, checkable condition
#      rather than assumed true of every large-range candle (a huge range
#      after a flat, directionless market is NOT what "peak of a run"
#      means): the net close-to-close move over the RUN_LOOKBACK_BARS (8)
#      bars immediately BEFORE the climax candle (i.e. closes[c-1] -
#      closes[c-1-RUN_LOOKBACK_BARS]) must (a) have the SAME SIGN as the
#      climax candle's own body (close[c] - open[c]) - a bullish run capped
#      by a bullish (or at least non-bearish-bodied) climax candle, mirror
#      for bearish - and (b) be at least HALF the size of the climax
#      candle's own body. A doji climax candle (body == 0) has no
#      meaningful sign and is skipped entirely - there's no clean "bullish
#      run climax" vs "bearish run climax" call to make from a body of
#      exactly zero.
#
#   4. CLIMAX CANDLE - VOLUME CONDITION, AND THE DATA-AVAILABILITY CAVEAT:
#      the source's literal language is "volume... at least 3 times [the
#      trailing average]" (CLIMAX_VOLUME_MULT). Whether Dukascopy's
#      `volume` field for FX/metals pairs is even meaningfully non-constant
#      is a real, already-known risk for this data source (see this
#      project's ORB scripts, which fall back to an all-zero volume list
#      when the column is missing at all). This script does NOT hardcode
#      an assumption either way: `volume_field_is_usable()` below inspects
#      the ACTUAL fetched data at run time (fraction of non-zero bars and
#      number of distinct values) and `main()` prints exactly what it
#      found and which path it took, for every instrument, before running
#      a single backtest. If the volume field turns out usable across all
#      four instruments, the volume condition is layered on as an ADDITION
#      to the range/run conditions (matching the source's literal "times
#      three" language exactly); if it's degenerate (all-zero, near-
#      constant, or too sparse to trust) for even one instrument, the
#      volume condition is DROPPED FOR THE ENTIRE RUN and the script
#      relies on the range-vs-ATR + prior-run conditions alone, so every
#      instrument is judged by the same rule (a per-instrument split
#      policy would make the cross-instrument comparison apples-to-oranges
#      and isn't worth the complexity for a data field this uncertain).
#      This mirrors this project's established practice of falling back
#      to a range/impulse-only definition when a volume field can't be
#      trusted (see orb_indices_optimization_and_ml.py's
#      `volumes = df["volume"].tolist() if "volume" in df.columns else
#      [0] * len(df)` guard) - documented plainly here rather than silently
#      assumed either way. NOTE: this sandboxed development environment
#      has no network egress to Dukascopy to check this in advance, so the
#      decision genuinely is made empirically at run time, not pre-baked
#      from prior knowledge - see main()'s printed diagnostic on a real
#      run for the actual answer.
#      VOLUME_AVG_BARS approximates "trailing 15 TRADING DAYS of volume"
#      as VOLUME_AVG_DAYS (15) * BARS_PER_DAY_15MIN (96, i.e. 24 hours of
#      continuous 15-min bars) = 1440 bars - FX/metals trade continuously
#      Sun evening through Fri evening, not in discrete exchange sessions,
#      so "bars per day" here is an approximation of continuous-market
#      time, not a literal exchange trading day; noted rather than assumed
#      exact.
#
#   5. CONFIRMATION: the very NEXT 15-min bar (bar c+1, the one immediately
#      after the climax candle - not any later bar) must close in the
#      OPPOSITE direction of the climax candle's own body (a bearish close
#      after a bullish-run climax candle -> bearish/SHORT setup; mirror for
#      bullish). If the immediately-next bar doesn't confirm, the climax
#      candle is simply discarded - this is a one-shot check on the very
#      next bar, not a search over some later window.
#
#   6. ENTRY - A STOP-ENTRY, NOT A MARKET ORDER: once confirmed, a PENDING
#      order is set at the confirmation bar's own LOW (bearish setup) or
#      HIGH (bullish setup) - i.e. "dribbles into" the pending order only
#      if a SUBSEQUENT bar actually trades through that level, matching the
#      source's stop-entry description rather than assuming an immediate
#      fill at the confirmation bar's close. Watched starting the bar AFTER
#      the confirmation bar (the confirmation bar's own low/high isn't
#      known as a level to place an order at until that bar has actually
#      closed) for up to PENDING_ORDER_EXPIRY_BARS (8) bars; if untriggered
#      by then, the pending order is cancelled and nothing is traded.
#
#   7. STOP - the climax candle's OWN extreme: High (bearish setup) or Low
#      (bullish setup) - not the confirmation bar's extreme and not an ATR
#      multiple, matching the source's plain "stop above/below the climax
#      candle" description.
#
#   8. TARGET - fixed REWARD_RISK (1.5), matching the source's literal
#      "1.5R" language.
#
#   9. TIME EXIT - force-closed (FLAT, r = pnl / sl_distance) MAX_HOLD_HOURS
#      (2 hours = 8 bars of 15-min data) AFTER ENTRY (not after the climax
#      candle, and not after the confirmation bar - the source's stated
#      window is measured from when the trade is actually live).
#
#  10. NO NEWS/SPREAD FILTER - the source doesn't specify one in a way this
#      project could implement without a news feed it doesn't have (same
#      caveat convention as every other simplification in this project -
#      see e.g. PO3's same-day-vs-prior-session simplification). Not
#      modeled; a genuine simplification, stated plainly rather than
#      silently assumed away.
#
#  11. ONE TRADE (AND ONE PENDING ORDER) AT A TIME - a climax candle
#      detected while a position is open OR a pending order is already
#      awaiting trigger is simply not acted on; matches this project's
#      convention everywhere else of a single flat/pending/open state
#      machine, not queued or parallel setups.
#
#  12. DEGENERATE-GEOMETRY GUARD: skip (don't force) if sl_distance falls
#      below MIN_SL_PCT of entry, same convention as every other script
#      here.
#
# No parameter grid search - one fixed rule set, tested for whether it
# holds up via a split-period honesty check, per-instrument breakdown, and
# a CORRECT z-score (see the project-wide z-score fix note below), not
# tuned to this data.
#
# GRANULARITY / FALSE-POSITIVE SANITY CHECK (this project's established
# practice - see ict_po3_forex_dukascopy_backtest.py's header for why a
# single-step synthetic-bar OHLC construction can manufacture a fake edge):
# this strategy's stop IS pinned to the climax candle's own wick extreme
# (structurally similar to PO3's manipulation-bar-derived stop and the
# mean-reversion script's excursion-bar stop), so this risk is real here,
# unlike the squeeze breakout script alongside this one (whose stop is
# close-based, not wick-based). CONFIRMED via a synthetic, zero-drift,
# multi-step random walk (test_climax_volume_reversal_dukascopy_backtest.py's
# TestGranularityConvergence): at coarse (wickless, n_substeps=1) resolution
# this DOES show a large, clearly-non-chance apparent edge (z ~ +3.4 on
# ~3,450 synthetic trades) - exactly the false-positive trap this check
# exists to catch - which collapses to chance-level noise (|z| ~ 0.3-0.6) at
# finer resolutions (n_substeps=8 and 32). That's the same conclusion this
# project's other wick-coupled strategies have reached: the coarse "edge" is
# a construction artifact, not a real signal. See the test file for the
# full numbers and the exact reproducible configuration.

# !pip install --upgrade dukascopy-python -q   # uncomment this line in Colab

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
# download chunk. A plain logger.setLevel(WARNING) does NOT survive this: the library's own
# fetch() call resets the "DUKASCRIPT" logger's level back to INFO internally on every single
# call. A logging Filter survives that reset (the library never touches .filters), so that's
# what's used instead - see day_trading_rauf_dukascopy_backtest.py for the original writeup.
class _SuppressDukascopyInfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING


logging.getLogger("DUKASCRIPT").addFilter(_SuppressDukascopyInfoFilter())

# Fetched data is cached to disk per (instrument, interval, date range) - see
# day_trading_rauf_dukascopy_backtest.py for the full writeup. Auto-detects a mounted Google
# Drive and uses that instead of the ephemeral local disk if present, so the cache survives
# runtime resets and is shared across every script in this project using the same range.
CACHE_DIR = "/content/drive/MyDrive/dukascopy_cache" if os.path.isdir("/content/drive/MyDrive") else "dukascopy_cache"
FETCH_CHUNK_MONTHS = 3   # how finely to split the download for progress-bar granularity

INSTRUMENTS = [
    ("EURUSD", dki.INSTRUMENT_FX_MAJORS_EUR_USD),
    ("GBPUSD", dki.INSTRUMENT_FX_MAJORS_GBP_USD),
    ("USDJPY", dki.INSTRUMENT_FX_MAJORS_USD_JPY),
    ("XAUUSD", dki.INSTRUMENT_FX_METALS_XAU_USD),
]

FETCH_START = datetime.datetime(2016, 1, 1)
FETCH_END = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_MIN_15   # native 15-min fetch - see header note 1
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

ATR_LENGTH = 14                     # Wilder ATR length on 15-min bars
CLIMAX_RANGE_ATR_MULT = 3.0         # climax candle's range must be >= this many ATRs
RUN_LOOKBACK_BARS = 8               # bars of prior directional run required before a climax candle
CLIMAX_VOLUME_MULT = 3.0            # climax candle's volume must be >= this many times its trailing average
VOLUME_AVG_DAYS = 15                # "trailing 15-day" average volume window, in approximate trading days
BARS_PER_DAY_15MIN = 96             # 24h of continuous 15-min bars - see header note 4 on why this is an approximation
VOLUME_AVG_BARS = VOLUME_AVG_DAYS * BARS_PER_DAY_15MIN
PENDING_ORDER_EXPIRY_BARS = 8       # bars after confirmation the stop-entry order is left resting
REWARD_RISK = 1.5                   # fixed reward:risk target, matching the source's "1.5R"
MAX_HOLD_HOURS = 2                  # force-close this many hours after ENTRY (not after the climax candle)
MAX_HOLD_BARS = int(MAX_HOLD_HOURS * 60 / 15)   # = 8 bars of 15-min data
MIN_SL_PCT = 0.02                   # floor on stop distance, as a % of entry - same convention as other scripts

# Illustrative round-trip cost scenarios, as a percentage of entry price - NOT measured real spread
# data, just a few bracketing assumptions to see how much cost this edge can absorb before it
# disappears, same convention introduced in support_resistance_zone_bounce_dukascopy_backtest.py.
COST_PCT_SCENARIOS = [0.0, 0.01, 0.03, 0.05]


def to_ny_time(index):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert("America/New_York")


def _month_chunks(start, end, months_per_chunk):
    chunk_start = start
    while chunk_start < end:
        month_index = chunk_start.month - 1 + months_per_chunk
        chunk_end = chunk_start.replace(year=chunk_start.year + month_index // 12, month=month_index % 12 + 1)
        yield chunk_start, min(chunk_end, end)
        chunk_start = chunk_end


def fetch_instrument_data(label, instrument_const):
    """Downloads in FETCH_CHUNK_MONTHS-sized pieces and caches the combined result to disk -
    see day_trading_rauf_dukascopy_backtest.py for the full writeup on why. Keeps the `volume`
    column (if Dukascopy returns one) rather than dropping it, since volume_field_is_usable()
    needs it."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{label}_15min_{FETCH_START.date()}_{FETCH_END.date()}.pkl")
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


def compute_atr_series(highs, lows, closes, length=ATR_LENGTH):
    """Wilder's ATR - identical formula to compute_atr_series() in
    orb_indices_optimization_and_ml.py (duplicated, not imported, per this project's convention -
    see donchian_turtle_breakout_dukascopy_backtest.py for precedent): the first `length` true
    ranges are seeded as a simple average, then smoothed one bar at a time:
    atr = (prev_atr * (length - 1) + tr) / length. atr[i] is None until the seed window is full."""
    n = len(closes)
    atr = [None] * n
    tr_seed = []
    atr_val = None
    prev_close = None
    for i in range(n):
        if prev_close is not None:
            tr = max(highs[i] - lows[i], abs(highs[i] - prev_close), abs(lows[i] - prev_close))
            if atr_val is None:
                tr_seed.append(tr)
                if len(tr_seed) >= length:
                    atr_val = sum(tr_seed) / length
            else:
                atr_val = (atr_val * (length - 1) + tr) / length
        atr[i] = atr_val
        prev_close = closes[i]
    return atr


def volume_field_is_usable(volumes, min_nonzero_frac=0.5, min_unique_values=5):
    """Decides, from the ACTUAL fetched data, whether a `volume` series is meaningful enough to
    build a trading condition on - see header note 4 for the full rationale on why this is
    determined empirically at run time rather than assumed either way. A field that's all (or
    mostly) zero, or takes on only a handful of distinct values (e.g. a constant placeholder), is
    treated as degenerate. `min_nonzero_frac` and `min_unique_values` are deliberately generous
    (this is a coarse "is this even remotely real data" check, not a data-quality scorer)."""
    n = len(volumes)
    if n == 0:
        return False
    arr = np.asarray(volumes, dtype=float)
    nonzero_frac = float(np.mean(arr > 0))
    n_unique = len(np.unique(arr))
    return nonzero_frac >= min_nonzero_frac and n_unique >= min_unique_values


def detect_climax_candle(c, opens, highs, lows, closes, atr, volumes=None, volume_avg=None,
                          range_atr_mult=CLIMAX_RANGE_ATR_MULT, run_lookback_bars=RUN_LOOKBACK_BARS,
                          volume_mult=CLIMAX_VOLUME_MULT):
    """Checks whether bar index `c` is a valid climax candle - range vs ATR (header note 2), a
    genuine prior directional run (header note 3), and, if `volumes`/`volume_avg` are both
    provided (i.e. the volume condition is enabled for this run - header note 4), volume vs its
    trailing average (header note 4). Returns None if not a climax candle, else a dict
    {"direction": "BULLISH" or "BEARISH" (the climax candle's OWN body direction), "extreme":
    the candle's High (BULLISH) or Low (BEARISH), used as the eventual stop level}.
    NO LOOKAHEAD: only ever reads bars at index <= c, and the ATR baseline used is atr[c-1] (the
    prior bar's ATR, not the climax candle's own - see header note 2)."""
    if c < run_lookback_bars + 1:
        return None   # not enough history for the prior-run window

    baseline_atr = atr[c - 1]
    if baseline_atr is None or baseline_atr <= 0:
        return None

    candle_range = highs[c] - lows[c]
    if candle_range < range_atr_mult * baseline_atr:
        return None

    body = closes[c] - opens[c]
    if body == 0:
        return None   # doji climax candle - no clean run-direction call to make (header note 3)

    net_move = closes[c - 1] - closes[c - 1 - run_lookback_bars]
    same_sign = (net_move > 0 and body > 0) or (net_move < 0 and body < 0)
    if not same_sign or abs(net_move) < 0.5 * abs(body):
        return None

    if volumes is not None and volume_avg is not None:
        vol_avg_c = volume_avg[c]
        if vol_avg_c is None or vol_avg_c <= 0:
            return None
        if volumes[c] < volume_mult * vol_avg_c:
            return None

    if body > 0:
        return {"direction": "BULLISH", "extreme": highs[c]}
    return {"direction": "BEARISH", "extreme": lows[c]}


def backtest_instrument(label, df, use_volume_filter):
    highs = df["High"].tolist()
    lows = df["Low"].tolist()
    closes = df["Close"].tolist()
    opens = df["Open"].tolist()
    times = df.index
    n = len(closes)

    atr = compute_atr_series(highs, lows, closes)

    volumes = volume_avg = None
    if use_volume_filter:
        volumes = df["volume"].tolist() if "volume" in df.columns else [0.0] * n
        # trailing average, PURELY historical (bars c - VOLUME_AVG_BARS .. c - 1, excluding c
        # itself) - a rolling mean shifted by one bar, same no-lookahead convention as the ATR
        # baseline above.
        vol_series = pd.Series(volumes, dtype=float)
        avg = vol_series.rolling(VOLUME_AVG_BARS).mean().shift(1)
        volume_avg = [None if pd.isna(x) else float(x) for x in avg]

    trades = []
    state = None   # None (flat) | {"type": "pending", ...} | {"type": "open", ...}

    for i in range(n):
        if state is not None and state["type"] == "open":
            side = state["side"]
            stop, target = state["stop"], state["target"]
            hi, lo = highs[i], lows[i]
            hit_stop = hi >= stop if side == "SHORT" else lo <= stop
            hit_target = lo <= target if side == "SHORT" else hi >= target
            bars_held = i - state["entry_index"]
            outcome = exit_r = None
            if hit_stop:
                outcome, exit_r = "SL", -1.0
            elif hit_target:
                outcome, exit_r = "TP", REWARD_RISK
            elif bars_held >= MAX_HOLD_BARS:
                pnl = (state["entry"] - closes[i]) if side == "SHORT" else (closes[i] - state["entry"])
                outcome, exit_r = "FLAT", pnl / state["sl_distance"]
            if outcome is not None:
                trades.append({"side": side, "outcome": outcome, "r": exit_r, "date": times[i].date(),
                               "stop_pct": state["sl_distance"] / state["entry"]})
                state = None
            continue   # one trade at a time - don't look for a new setup on a bar we just managed

        if state is not None and state["type"] == "pending":
            side = state["side"]
            level = state["pending_level"]
            triggered = lows[i] <= level if side == "SHORT" else highs[i] >= level
            if triggered:
                entry = level
                stop = state["climax_extreme"]
                sl_distance = (stop - entry) if side == "SHORT" else (entry - stop)
                min_sl = (MIN_SL_PCT / 100.0) * entry
                valid_geometry = sl_distance >= min_sl and ((stop > entry) if side == "SHORT" else (stop < entry))
                if valid_geometry:
                    target = entry - REWARD_RISK * sl_distance if side == "SHORT" else entry + REWARD_RISK * sl_distance
                    state = {"type": "open", "side": side, "entry": entry, "stop": stop, "target": target,
                              "sl_distance": sl_distance, "entry_index": i}
                else:
                    state = None   # degenerate geometry - skip rather than force a bad trade through
            else:
                state["bars_waited"] += 1
                if state["bars_waited"] >= PENDING_ORDER_EXPIRY_BARS:
                    state = None   # pending order expired unfilled
            continue

        # state is None (flat, no pending order) - look for a climax candle at bar i - 1,
        # confirmed by bar i's own close direction (header notes 2-5). Checking climax_bar = i-1
        # against confirm_bar = i (the CURRENT bar) rather than climax_bar = i / confirm_bar =
        # i+1 keeps this a strict single forward pass with no lookahead into future bars: by the
        # time we're processing bar i, bar i-1 has already fully closed.
        climax_bar = i - 1
        if climax_bar < 0:
            continue

        climax = detect_climax_candle(climax_bar, opens, highs, lows, closes, atr, volumes, volume_avg)
        if climax is None:
            continue

        # confirmation: bar i (current) closes opposite the climax candle's own direction
        confirm_bearish = closes[i] < opens[i]
        confirm_bullish = closes[i] > opens[i]
        if climax["direction"] == "BULLISH" and confirm_bearish:
            side = "SHORT"
            pending_level = lows[i]
        elif climax["direction"] == "BEARISH" and confirm_bullish:
            side = "LONG"
            pending_level = highs[i]
        else:
            continue   # no confirmation on the very next bar - climax candle discarded (header note 5)

        state = {"type": "pending", "side": side, "pending_level": pending_level,
                  "climax_extreme": climax["extreme"], "bars_waited": 0}

    return trades


def main():
    years = (FETCH_END - FETCH_START).days / 365
    print(f"Downloading {len(INSTRUMENTS)} instruments from Dukascopy over ~{years:.0f} years "
          f"({FETCH_START.date()} to {FETCH_END.date()}), native 15-min bars - cached to disk "
          f"after the first run, so this is only slow once.\n")

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

    # --- decide, empirically, whether the volume condition is usable (header note 4) ---
    print("\nVOLUME FIELD DIAGNOSTIC (deciding whether CLIMAX_VOLUME_MULT is applied this run):")
    per_instrument_usable = {}
    for label, df in data.items():
        volumes = df["volume"].tolist() if "volume" in df.columns else []
        usable = volume_field_is_usable(volumes)
        per_instrument_usable[label] = usable
        if volumes:
            nz = float(np.mean(np.asarray(volumes, dtype=float) > 0))
            n_unique = len(np.unique(np.asarray(volumes, dtype=float)))
            print(f"  {label:10s}: usable={usable}  (nonzero_frac={nz:.3f}, unique_values={n_unique})")
        else:
            print(f"  {label:10s}: usable={usable}  (no `volume` column at all)")

    use_volume_filter = bool(data) and all(per_instrument_usable.values())
    if use_volume_filter:
        print("-> volume field looks usable on every instrument - applying the CLIMAX_VOLUME_MULT "
              "condition as an ADDITION to range/run, matching the source's literal 'volume times 3' language.")
    else:
        print("-> volume field is degenerate (or missing) on at least one instrument - DROPPING the volume "
              "condition for this ENTIRE run (every instrument judged by the same rule) and relying on the "
              "range-vs-ATR + prior-run conditions alone. This is a real, documented data-source limitation, "
              "not a silent assumption - see this script's header note 4.")

    all_trades = []
    for label, df in data.items():
        trades = backtest_instrument(label, df, use_volume_filter)
        for t in trades:
            t["instrument"] = label
        all_trades.extend(trades)

    if not all_trades:
        print("\nNo trades at all - a climax candle followed by a confirmed, triggered stop-entry never "
              "occurred in this data.")
        return

    total_r = sum(t["r"] for t in all_trades)
    n_trades = len(all_trades)
    print("\n" + "=" * 70)
    print(f"CLIMAX VOLUME REVERSAL - {n_trades} trades, {FETCH_START.date()} to {FETCH_END.date()}")
    print("=" * 70)
    print(f"Total R: {total_r:+.2f}   Avg R/trade: {total_r/n_trades:+.4f}")
    print("(This is THIS PROJECT'S OWN independently computed result - not the source's claimed "
          "'123% return / 11% max drawdown' figure, which is not reproduced or assumed here.)")

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

    # CORRECT z-score: (mean R) / (standard error of the mean), i.e.
    # z = (total_r / n) / (std(r, ddof=1) / sqrt(n)) - NOT the old buggy avg_r * sqrt(n) shortcut
    # this project caught and fixed project-wide (see ict_po3_forex_dukascopy_backtest.py).
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

    print("\nNo commission/spread/slippage modeled. Entry is a simulated stop-entry fill at the exact "
          "confirmation-bar level once a later bar's high/low touches it - real fills would be worse "
          "(a genuine stop order in a fast-continuing move typically fills through the level, not exactly "
          "at it). No news/spread filter either - this project has no news feed to build one from (header "
          "note 10).")


if __name__ == "__main__":
    main()
