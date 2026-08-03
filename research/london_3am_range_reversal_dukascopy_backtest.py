# "3:00 AM key time" London-session range-reversal backtest via Dukascopy, 2016-2025.
#
# SOURCE IS DIFFERENT FROM EVERY OTHER ICT-STYLE SCRIPT IN THIS PROJECT: ict_po3 and
# ict_silver_bullet were operationalized from Pine source code with exact numeric
# thresholds. This one comes from a trader narrating a discretionary process on video - no
# code, no numbers. Several concepts are named repeatedly but never given a rule ("relatively
# equal lows", "speed and distance"), and one entry condition is stated as REQUIRED but never
# actually defined: SMT (an inter-market divergence check) is described as needed ("all you're
# looking for is SMT and displacement") without ever naming a second instrument or a divergence
# rule to check against it.
#
# SMT IS DELIBERATELY NOT IMPLEMENTED HERE. Asked directly how to handle it, the answer was
# "idk" - so this takes the more conservative branch: build the parts that ARE mechanical
# (range sweep + displacement + premium/discount entry) and skip the one requirement the
# source never defines, rather than inventing an SMT rule wholesale and presenting it as if it
# came from the video. Every other threshold below that needed a number reuses this project's
# OWN already-established definition of an analogous concept elsewhere (see each stage's note)
# - applying a precedented technical definition to a concept the source names but doesn't
# quantify is a materially different, more defensible thing than fabricating one from nothing.
#
# OPERATIONALIZING (mapping the video's own vocabulary onto exact rules):
#   1. DEALING RANGE: the high/low of price during RANGE_START-RANGE_END (00:00-02:00 NY) - a
#      same-day simplification analogous to ict_po3's own ACCUMULATION window (see that
#      script's header note for the same tradeoff). The video's "connected range" traced from
#      an earlier price leg on the chart isn't reproducible without inventing its own
#      swing-detection rule, so this reuses PO3's precedent for the same underlying problem.
#   2. KEY-TIME SWEEP ("manipulation"): the first 5-min bar within SESSION_START-SESSION_END
#      (02:00-04:30 NY - starts an hour before the video's literal "3:00am" specifically to
#      cover its own "if 2am already manipulated, 3am distributes instead" case; either way,
#      the FIRST sweep inside this window is the one the video is describing) whose high
#      exceeds the dealing range's high, or whose low undercuts its low, is the sweep. A single
#      bar breaching BOTH sides at once (a wide/volatile bar) skips that day entirely -
#      genuinely ambiguous which direction "the" manipulation was, same guard ict_po3 uses for
#      the same situation.
#   3. NEW DEALING RANGE / EQUILIBRIUM: sweeping the range high forms a new range from that
#      swept high down to the ORIGINAL range low (mirror image for a low-side sweep) - the
#      video's own "this high to this low is your new dealing range" framing. EQUILIBRIUM is
#      the 50% midpoint of this new range - the first source's stated target ("you're only
#      looking for a trade to 50%"), still computed below since #8 depends on it, though it's
#      no longer this script's actual TARGET - see #8.
#   4. DISPLACEMENT: ict_silver_bullet's own "market structure shift" is anchored to the swing
#      BEFORE its sweep. This video's own worked example anchors it AFTER the sweep instead -
#      "you have one, two, three [candles] that also traded above this range high... this
#      5-minute candle closes below all of them." Followed literally rather than borrowed from
#      Silver Bullet's different anchor: POST_SWEEP_CLUSTER_BARS candles immediately after the
#      sweep (3, matching the video's own count) set a local high/low; DISPLACEMENT is the
#      first later close that breaks back through that level in the reversal direction, found
#      within DISPLACEMENT_SEARCH_BARS bars.
#   5. PREMIUM/DISCOUNT GATE: displacement only counts while price is still on the sweep side
#      of equilibrium (above it after a high-side sweep, below it after a low-side sweep) - the
#      video's explicit validity condition ("this is a valid entry... because it is in a
#      premium"), enforced as a hard filter here, not just a caption.
#   6. ENTRY: a simulated market order at the displacement bar's own close - the video is
#      explicit no retracement/retest is needed ("you don't need to wait for anything else...
#      as soon as we displace, you have an entry"), unlike Silver Bullet's FVG-midpoint
#      limit-order model.
#   7. STOP: beyond the sweep bar's own wick, small buffer - same STOP_BUFFER_PCT convention as
#      every other script here.
#   8. TARGET: the FULL opposite-range boundary (the ORIGINAL range low for a high-side sweep,
#      ORIGINAL range high for a low-side sweep) - NOT equilibrium. Revised from an earlier pass
#      that only ever targeted equilibrium, after a second, independent source made the same
#      claim about this exact strategy and additionally said the target can be "50%... or the
#      connected range low/high", i.e. the far side, not just the midpoint. Equilibrium is
#      DELIBERATELY not tried as a fallback when the full-range target misses MIN_REWARD_RISK,
#      despite that being a real, considered option here: it provably could never have passed
#      anyway, since the premium/discount gate (#5) already guarantees entry sits on the
#      profitable side of equilibrium, which makes reward-to-full-range = reward-to-equilibrium
#      + the (always positive) half-range width - strictly bigger, every time, off the same
#      risk. If the full-range target's reward:risk is below MIN_REWARD_RISK, the trade is
#      SKIPPED rather than resized or forced to a fallback ratio - both sources are explicit
#      about this ("don't try force a 1:5 with this strategy... at least a 1:2"), unlike
#      ict_po3's fallback-to-fixed-R:R handling of the same situation.
#
# One trade/day/instrument (the first sweep found; no retry if displacement fails). No
# parameter grid search - like ict_po3 and evendyer_vwap_orb, this is a single fixed rule set
# tested for temporal consistency (split-period check), not tuned to this data.

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
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_MIN_5
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

# --- session windows, all NY time (same-day simplification - see header note 1) ---
RANGE_START = pd.Timestamp("00:00").time()
RANGE_END = pd.Timestamp("02:00").time()
SESSION_START = pd.Timestamp("02:00").time()   # widened before the video's literal 3:00am - see header note 2
SESSION_END = pd.Timestamp("04:30").time()

POST_SWEEP_CLUSTER_BARS = 3     # the "one, two, three candles" the source counts explicitly
DISPLACEMENT_SEARCH_BARS = 12   # 1 hour to find the break-of-cluster displacement candle
MAX_HOLD_BARS = 288             # 24 hours max hold once filled, then FLAT at last close

STOP_BUFFER_PCT = 0.02          # small buffer beyond the sweep's wick, same scale as other scripts here
MIN_RANGE_PCT = 0.02            # floor on the dealing range size, guards against a degenerate near-zero range
MIN_REWARD_RISK = 2.0           # source's own stated floor ("at least a 1:2") - skipped, not resized, below this

# Illustrative round-trip cost scenarios, as a percentage of entry price - NOT measured real spread
# data, just a few bracketing assumptions to see how much cost this edge can absorb before it
# disappears, same convention introduced in support_resistance_zone_bounce_dukascopy_backtest.py.
COST_PCT_SCENARIOS = [0.0, 0.01, 0.03, 0.05]


def to_ny_time(index):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert("America/New_York")


def fetch_instrument_data(instrument_const):
    df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, FETCH_START, FETCH_END)
    if df.empty:
        return None
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df.index = to_ny_time(df.index)
    return df


def find_range_reversal_trade(highs, lows, closes, times, day_bars, n):
    """One day, one instrument - returns a trade dict or None if any stage fails. `day_bars` is
    this day's bar indices in ascending time order (see backtest_instrument)."""
    range_high = range_low = None
    session_bars = []
    for idx in day_bars:
        tod = times[idx].time()
        if RANGE_START <= tod < RANGE_END:
            range_high = highs[idx] if range_high is None else max(range_high, highs[idx])
            range_low = lows[idx] if range_low is None else min(range_low, lows[idx])
        elif SESSION_START <= tod < SESSION_END:
            session_bars.append(idx)

    if range_high is None or range_low is None or range_low <= 0:
        return None
    range_pct = (range_high - range_low) / range_low * 100.0
    if range_pct < MIN_RANGE_PCT:
        return None

    # --- stage 1: liquidity sweep of one side of the dealing range ---
    i_sweep = sweep_side = None
    for idx in session_bars:
        broke_high = highs[idx] > range_high
        broke_low = lows[idx] < range_low
        if broke_high and broke_low:
            return None   # ambiguous same-bar double break - same guard as ict_po3's manipulation stage
        if broke_high:
            i_sweep, sweep_side = idx, "SHORT"
            break
        if broke_low:
            i_sweep, sweep_side = idx, "LONG"
            break
    if i_sweep is None:
        return None

    sweep_extreme = highs[i_sweep] if sweep_side == "SHORT" else lows[i_sweep]
    if sweep_side == "SHORT":
        new_top, new_bottom = sweep_extreme, range_low
    else:
        new_top, new_bottom = range_high, sweep_extreme
    equilibrium = (new_top + new_bottom) / 2.0

    # --- stage 2: post-sweep consolidation cluster defines the local structure to be displaced ---
    cluster_start = i_sweep + 1
    cluster_end = min(i_sweep + POST_SWEEP_CLUSTER_BARS, n - 1)
    if cluster_end < cluster_start:
        return None
    cluster_idx = range(cluster_start, cluster_end + 1)
    if sweep_side == "SHORT":
        cluster_ref = min(lows[j] for j in cluster_idx)
    else:
        cluster_ref = max(highs[j] for j in cluster_idx)

    # --- stage 3: displacement - first close past the cluster that breaks its level in the
    # reversal direction, only while still on the sweep side of equilibrium (premium/discount gate) ---
    i_entry = None
    for j in range(cluster_end + 1, min(cluster_end + DISPLACEMENT_SEARCH_BARS, n - 1) + 1):
        price = closes[j]
        still_valid = (price > equilibrium) if sweep_side == "SHORT" else (price < equilibrium)
        if not still_valid:
            break   # already crossed equilibrium before displacing - this setup is invalidated
        displaced = (price < cluster_ref) if sweep_side == "SHORT" else (price > cluster_ref)
        if displaced:
            i_entry = j
            break
    if i_entry is None:
        return None

    # --- stage 4: entry, stop, target ---
    entry = closes[i_entry]
    buffer_price = (STOP_BUFFER_PCT / 100.0) * entry
    if sweep_side == "SHORT":
        stop = sweep_extreme + buffer_price
        sl_distance = stop - entry
        target = new_bottom
    else:
        stop = sweep_extreme - buffer_price
        sl_distance = entry - stop
        target = new_top

    if sl_distance <= 0:
        return None

    # TARGET: the FULL opposite-range boundary, not equilibrium - a second source (a different
    # video, same claim independently) confirms the target isn't always just 50%: "target 50%...
    # or the connected range low/high". Equilibrium is deliberately NOT tried as a fallback when
    # the full-range target misses MIN_REWARD_RISK, because it provably never would have passed
    # anyway: the premium/discount gate above already guarantees entry sits strictly on the
    # profitable side of equilibrium (reward-to-equilibrium > 0 is a precondition for i_entry to
    # exist at all), and reward-to-full-range = reward-to-equilibrium + the (always positive)
    # half-range width - so for the same entry/stop, the full-range reward:risk is provably always
    # >= equilibrium's. A "try full-range, fall back to equilibrium" version would have an
    # equilibrium branch that can mathematically never execute; simplified away here rather than
    # carried as dead code - see test_london_3am_range_reversal_dukascopy_backtest.py's
    # TestFullRangeTarget.test_full_range_reward_risk_is_always_at_least_equilibriums for the
    # proof-by-test.
    target_mode = "full_range"
    reward = (entry - target) if sweep_side == "SHORT" else (target - entry)
    if reward <= 0:
        return None
    reward_risk = reward / sl_distance
    if reward_risk < MIN_REWARD_RISK:
        return None

    # --- stage 5: manage the trade forward ---
    outcome = exit_r = None
    p = i_entry + 1
    end_p = min(i_entry + MAX_HOLD_BARS, n - 1)
    while p <= end_p:
        hi, lo = highs[p], lows[p]
        hit_stop = hi >= stop if sweep_side == "SHORT" else lo <= stop
        hit_target = lo <= target if sweep_side == "SHORT" else hi >= target
        if hit_stop:
            outcome, exit_r = "SL", -1.0
            break
        if hit_target:
            outcome, exit_r = "TP", reward_risk
            break
        p += 1
    else:
        p = end_p

    if outcome is None:
        last_close = closes[p]
        pnl = (entry - last_close) if sweep_side == "SHORT" else (last_close - entry)
        outcome, exit_r = "FLAT", pnl / sl_distance
        exit_price = last_close
    else:
        exit_price = stop if outcome == "SL" else target

    return {"side": sweep_side, "outcome": outcome, "r": exit_r, "date": times[i_sweep].date(),
            "stop_pct": sl_distance / entry, "target_mode": target_mode,
            "entry_price": entry, "stop_price": stop, "target_price": target,
            "exit_price": exit_price, "entry_time": times[i_entry], "exit_time": times[p]}


def backtest_instrument(label, df):
    highs, lows, closes = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
    times = df.index
    n = len(closes)

    days_index = {}
    for idx in range(n):
        days_index.setdefault(times[idx].date(), []).append(idx)

    trades = []
    for day in sorted(days_index):
        trade = find_range_reversal_trade(highs, lows, closes, times, days_index[day], n)
        if trade is not None:
            trades.append(trade)
    return trades


def main():
    years = (FETCH_END - FETCH_START).days / 365
    print(f"Downloading {len(INSTRUMENTS)} instruments from Dukascopy over ~{years:.0f} years "
          f"({FETCH_START.date()} to {FETCH_END.date()}) - cached to disk after the first run.\n")
    print("NOTE: SMT is NOT implemented - the source video requires it but never defines a second "
          "instrument or a divergence rule, so this only trades the mechanical parts (range sweep, "
          "post-sweep displacement, premium/discount gate). See the file header for why.\n")

    data = {}
    for label, instrument_const in INSTRUMENTS:
        print(f"{label}...", end=" ")
        try:
            df = fetch_instrument_data(instrument_const)
        except Exception as exc:
            print(f"failed ({exc})")
            continue
        if df is None:
            print("no data")
            continue
        data[label] = df
        print(f"{len(df)} bars")

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
        print("\nNo trades at all - the 90-minute window never produced a clean sweep -> post-sweep "
              "cluster -> displacement chain while still inside premium/discount on this data.")
        return

    total_r = sum(t["r"] for t in all_trades)
    n_trades = len(all_trades)
    print("\n" + "=" * 70)
    print(f"LONDON 3AM RANGE REVERSAL - {n_trades} trades, {FETCH_START.date()} to {FETCH_END.date()}")
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

    print("\nNo commission/spread/slippage modeled. Entry is a simulated market order at the "
          "displacement candle's own close - real fills would be worse. SMT is not modeled at all - "
          "see the file header; treat this as a test of the strategy's mechanical half only.")


if __name__ == "__main__":
    main()
