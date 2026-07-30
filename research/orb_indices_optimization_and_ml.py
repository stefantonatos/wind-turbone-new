# Proper walk-forward parameter optimization + ML scoring for ORB on
# indices - the one candidate out of everything tested in this project
# that's shown real (if not yet proven) promise: +0.012R/trade unfiltered,
# +0.043R/trade with the range/impulse/volume/regime filters, over
# 1500/548 trades on 6 real indices via Dukascopy.
#
# Two things this does, both structured to avoid the classic overfitting
# trap of "grid search until something looks good, then trust it":
#
#   1. PARAMETER OPTIMIZATION: grid search over RANGE_MINUTES and
#      REWARD_RISK on an IN-SAMPLE period only (2021-2023), pick the best
#      combination by in-sample total R, then report how that SAME
#      combination performs on a completely separate OUT-OF-SAMPLE period
#      (2024, the same year already tested elsewhere in this project) -
#      if out-of-sample performance is much worse than in-sample, that's
#      direct evidence of overfitting, not a reason to keep searching for
#      a better number.
#
#   2. ML SCORING: same approach already proven for forex in
#      quantconnect/orb_datagen.py / train_orb_model.ipynb / orb_ml.py -
#      strip the hand-picked filters, record every raw close-confirmed
#      breakout's features and real outcome across the in-sample period,
#      train a classifier, and check its R-multiple performance on the
#      SAME held-out out-of-sample period the optimization step uses -
#      apples-to-apples against the hand-tuned version, not a different
#      test window.
#
# Only two structural parameters are grid-searched (not the filter
# thresholds too) - keeping the search space small relative to the data
# is itself a guard against overfitting; a search over 6+ parameters
# would very likely just find noise that happens to fit 2021-2023.

# !pip install --upgrade dukascopy-python scikit-learn -q   # uncomment in Colab

import datetime

import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

INDICES = [
    ("SP500", dki.INSTRUMENT_IDX_AMERICA_E_SANDP_500, "America/New_York", pd.Timestamp("09:30").time()),
    ("NASDAQ100", dki.INSTRUMENT_IDX_AMERICA_E_NQ_100, "America/New_York", pd.Timestamp("09:30").time()),
    ("DOWJONES", dki.INSTRUMENT_IDX_AMERICA_E_D_J_IND, "America/New_York", pd.Timestamp("09:30").time()),
    ("DAX", dki.INSTRUMENT_IDX_EUROPE_E_DAAX, "Europe/Berlin", pd.Timestamp("09:00").time()),
    ("FTSE100", dki.INSTRUMENT_IDX_EUROPE_E_FUTSEE_100, "Europe/London", pd.Timestamp("08:00").time()),
    ("NIKKEI225", dki.INSTRUMENT_IDX_ASIA_E_N225JAP, "Asia/Tokyo", pd.Timestamp("09:00").time()),
]

FETCH_START = datetime.datetime(2021, 1, 1)
FETCH_END = datetime.datetime(2025, 1, 1)
SPLIT_DATE = datetime.date(2024, 1, 1)   # everything before this = in-sample, on/after = out-of-sample
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_MIN_5
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

# --- fixed strategy config (the parts NOT being grid-searched) ---
ENTRY_WINDOW_MINUTES = 180
SESSION_HOLD_HOURS = 8
ENTRY_BUFFER_PCT = 0.02
MIN_RANGE_PCT = 0.05
REVERSE_SIGNALS = False

ATR_LEN = 14
RANGE_ATR_FILTER = True
MIN_RANGE_ATR_MULT = 0.5
MAX_RANGE_ATR_MULT = 3.0
IMPULSE_FILTER = True
RANGE_AVG_LEN = 20
IMPULSE_RANGE_MULT = 1.3
VOLUME_FILTER = True
VOLUME_AVG_LEN = 20
RVOL_MULT = 1.3
VOLATILITY_REGIME_FILTER = True
ATR_BASELINE_LEN = 100
LOW_VOL_MULT = 0.7
HIGH_VOL_MULT = 1.5
ALLOWED_REGIMES = {"normal", "high"}

# --- grid search space (kept deliberately small - see header comment) ---
RANGE_MINUTES_GRID = [10, 15, 20, 30]
REWARD_RISK_GRID = [0.5, 1.0, 1.5, 2.0]

FEATURE_COLUMNS = [
    "range_vs_atr", "impulse_ratio", "atr_regime_ratio", "volume_ratio",
    "session_minute", "day_of_week", "direction",
]


def to_local_time(index, tz_name):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(tz_name)


def persisted_avg(values, length, start_index=0):
    n = len(values)
    out = [None] * n
    usable = values[start_index:]
    if len(usable) < length:
        return out
    seed = sum(usable[:length]) / length
    out[start_index + length - 1] = seed
    prev = seed
    for i in range(start_index + length, n):
        prev = (prev * (length - 1) + values[i]) / length
        out[i] = prev
    return out


def compute_atr_series(highs, lows, closes, length):
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


def volatility_regime(current_atr, baseline):
    if current_atr is None or baseline is None or baseline <= 0:
        return "normal"
    if current_atr > HIGH_VOL_MULT * baseline:
        return "high"
    if current_atr < LOW_VOL_MULT * baseline:
        return "low"
    return "normal"


def fetch_index_data(instrument_const, tz_name):
    """Downloads ONCE for the full FETCH_START-FETCH_END range - grid
    search and ML both reuse this same in-memory data rather than
    re-fetching per parameter combination, which would be enormously
    wasteful (16 grid combos x 6 indices x re-download each time)."""
    df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, FETCH_START, FETCH_END)
    if df.empty:
        return None
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df.index = to_local_time(df.index, tz_name)
    return df


def precompute_indicators(df):
    """Everything that doesn't depend on RANGE_MINUTES/REWARD_RISK - computed
    once per index, reused across every grid combination."""
    highs, lows, closes = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
    volumes = df["volume"].tolist() if "volume" in df.columns else [0] * len(df)

    atr = compute_atr_series(highs, lows, closes, ATR_LEN)
    bar_ranges = [h - l for h, l in zip(highs, lows)]
    range_avg = persisted_avg(bar_ranges, RANGE_AVG_LEN)
    volume_avg = persisted_avg(volumes, VOLUME_AVG_LEN)
    atr_first_valid = next((idx for idx, a in enumerate(atr) if a is not None), len(atr))
    atr_baseline = persisted_avg([a if a is not None else 0.0 for a in atr], ATR_BASELINE_LEN,
                                  start_index=atr_first_valid)
    return {
        "highs": highs, "lows": lows, "closes": closes, "volumes": volumes,
        "atr": atr, "range_avg": range_avg, "volume_avg": volume_avg, "atr_baseline": atr_baseline,
    }


def run_backtest(df, ind, tz_name, session_start, range_minutes, reward_risk,
                  apply_filters=True, split_before=None, split_after=None, collect_candidates=False):
    """Core ORB loop, parameterized by range_minutes/reward_risk for the
    grid search. apply_filters=False + collect_candidates=True switches
    into ML-datagen mode: every close-confirmed breakout is taken and
    labeled, filters only computed as features, matching this project's
    existing quantconnect/orb_datagen.py approach.

    split_before/split_after restrict which calendar dates are processed
    (in-sample vs out-of-sample), without needing to re-slice or
    re-download the underlying data."""
    highs, lows, closes = ind["highs"], ind["lows"], ind["closes"]
    volumes, atr = ind["volumes"], ind["atr"]
    range_avg, volume_avg, atr_baseline = ind["range_avg"], ind["volume_avg"], ind["atr_baseline"]
    times = df.index
    n = len(closes)

    range_end = (datetime.datetime.combine(datetime.date.min, session_start)
                 + datetime.timedelta(minutes=range_minutes)).time()
    entry_end = (datetime.datetime.combine(datetime.date.min, session_start)
                 + datetime.timedelta(minutes=range_minutes + ENTRY_WINDOW_MINUTES)).time()
    session_end = (datetime.datetime.combine(datetime.date.min, session_start)
                   + datetime.timedelta(hours=SESSION_HOLD_HOURS)).time()

    trades = []
    current_day = None
    range_high = range_low = None
    traded_today = False
    session_start_dt = None
    i = 0
    while i < n:
        t = times[i]
        today = t.date()
        tod = t.time()

        if split_before is not None and today >= split_before:
            i += 1
            continue
        if split_after is not None and today < split_after:
            i += 1
            continue

        if today != current_day:
            current_day = today
            range_high = range_low = None
            traded_today = False

        if traded_today:
            i += 1
            continue

        if session_start <= tod < range_end:
            range_high = highs[i] if range_high is None else max(range_high, highs[i])
            range_low = lows[i] if range_low is None else min(range_low, lows[i])
            if session_start_dt is None or today != session_start_dt:
                session_start_dt = today
            i += 1
            continue

        if range_high is None:
            i += 1
            continue

        if tod >= entry_end:
            traded_today = True
            i += 1
            continue

        buffer_price = (ENTRY_BUFFER_PCT / 100.0) * closes[i]
        price = closes[i]
        buy_setup = price > range_high + buffer_price
        sell_setup = price < range_low - buffer_price

        if REVERSE_SIGNALS:
            buy_setup, sell_setup = sell_setup, buy_setup

        if not buy_setup and not sell_setup:
            i += 1
            continue

        range_size = range_high - range_low
        current_atr = atr[i]

        if apply_filters:
            if RANGE_ATR_FILTER and current_atr:
                if range_size < MIN_RANGE_ATR_MULT * current_atr or range_size > MAX_RANGE_ATR_MULT * current_atr:
                    traded_today = True
                    i += 1
                    continue
            if VOLATILITY_REGIME_FILTER:
                regime = volatility_regime(current_atr, atr_baseline[i])
                if regime not in ALLOWED_REGIMES:
                    traded_today = True
                    i += 1
                    continue
            if IMPULSE_FILTER and range_avg[i]:
                if (highs[i] - lows[i]) < IMPULSE_RANGE_MULT * range_avg[i]:
                    traded_today = True
                    i += 1
                    continue
            if VOLUME_FILTER and volume_avg[i] and volume_avg[i] > 0:
                if volumes[i] < RVOL_MULT * volume_avg[i]:
                    traded_today = True
                    i += 1
                    continue

        sl_distance = max(range_size, (MIN_RANGE_PCT / 100.0) * price)
        tp_distance = sl_distance * reward_risk
        side = "LONG" if buy_setup else "SHORT"
        entry = price
        stop = entry - sl_distance if side == "LONG" else entry + sl_distance
        target = entry + tp_distance if side == "LONG" else entry - tp_distance

        traded_today = True
        outcome, exit_r = None, None
        j = i + 1
        while j < n and times[j].date() == today and times[j].time() < session_end:
            hi, lo = highs[j], lows[j]
            hit_stop = lo <= stop if side == "LONG" else hi >= stop
            hit_target = hi >= target if side == "LONG" else lo <= target
            if hit_stop:
                outcome, exit_r = "SL", -1.0
                break
            if hit_target:
                outcome, exit_r = "TP", reward_risk
                break
            j += 1
        else:
            j = min(j, n - 1)

        if outcome is None:
            last_close = closes[j]
            pnl = (last_close - entry) if side == "LONG" else (entry - last_close)
            outcome, exit_r = "FLAT", pnl / sl_distance

        trade = {"side": side, "outcome": outcome, "r": exit_r}

        if collect_candidates:
            minutes_since_open = (datetime.datetime.combine(datetime.date.min, tod)
                                   - datetime.datetime.combine(datetime.date.min, session_start)).total_seconds() / 60
            trade["features"] = {
                "range_vs_atr": (range_size / current_atr) if current_atr else np.nan,
                "impulse_ratio": ((highs[i] - lows[i]) / range_avg[i]) if range_avg[i] else np.nan,
                "atr_regime_ratio": (current_atr / atr_baseline[i]) if (current_atr and atr_baseline[i]) else np.nan,
                "volume_ratio": (volumes[i] / volume_avg[i]) if (volume_avg[i] and volume_avg[i] > 0) else np.nan,
                "session_minute": minutes_since_open,
                "day_of_week": t.weekday(),
                "direction": 1 if side == "LONG" else -1,
            }
            trade["label"] = 1 if exit_r > 0 else 0

        trades.append(trade)
        i = j + 1

    return trades


def main():
    print(f"This runs a {len(RANGE_MINUTES_GRID)}x{len(REWARD_RISK_GRID)} grid search across "
          f"{len(INDICES)} indices over 4 years of 5-min data, plus ML training - expect roughly "
          f"15-20 minutes total, not a hang.\n")
    print(f"Downloading {len(INDICES)} indices from Dukascopy ({FETCH_START.date()} to {FETCH_END.date()})...")
    data = {}
    for label, instrument_const, tz_name, session_start in INDICES:
        df = fetch_index_data(instrument_const, tz_name)
        if df is None:
            print(f"  {label}: no data")
            continue
        data[label] = (df, precompute_indicators(df), tz_name, session_start)
        print(f"  {label}: {len(df)} bars")

    if not data:
        print("No data downloaded - check output above.")
        return

    # ================= PART 1: parameter optimization =================
    print("\n" + "=" * 70)
    print(f"PART 1: grid search on IN-SAMPLE period ({FETCH_START.date()} to {SPLIT_DATE}), "
          f"validated on OUT-OF-SAMPLE ({SPLIT_DATE} to {FETCH_END.date()})")
    print("=" * 70)

    grid_results = []
    for range_minutes in RANGE_MINUTES_GRID:
        for reward_risk in REWARD_RISK_GRID:
            total_r_in_sample = 0.0
            n_trades_in_sample = 0
            for label, (df, ind, tz_name, session_start) in data.items():
                trades = run_backtest(df, ind, tz_name, session_start, range_minutes, reward_risk,
                                       apply_filters=True, split_before=SPLIT_DATE)
                total_r_in_sample += sum(t["r"] for t in trades)
                n_trades_in_sample += len(trades)
            grid_results.append({
                "range_minutes": range_minutes, "reward_risk": reward_risk,
                "in_sample_total_r": total_r_in_sample, "in_sample_trades": n_trades_in_sample,
                "in_sample_avg_r": total_r_in_sample / n_trades_in_sample if n_trades_in_sample else 0.0,
            })
            print(f"  RANGE_MINUTES={range_minutes:3d}  REWARD_RISK={reward_risk:.1f}  "
                  f"-> {n_trades_in_sample:4d} trades, {total_r_in_sample:+8.2f}R in-sample")

    grid_df = pd.DataFrame(grid_results).sort_values("in_sample_total_r", ascending=False)
    print("\nTop 5 in-sample combinations:")
    print(grid_df.head(5).to_string(index=False))

    best = grid_df.iloc[0]
    best_range_minutes, best_reward_risk = int(best["range_minutes"]), float(best["reward_risk"])
    print(f"\nBest in-sample: RANGE_MINUTES={best_range_minutes}, REWARD_RISK={best_reward_risk} "
          f"-> {best['in_sample_total_r']:+.2f}R over {best['in_sample_trades']:.0f} trades")

    total_r_oos, n_trades_oos = 0.0, 0
    per_index_oos = {}
    for label, (df, ind, tz_name, session_start) in data.items():
        trades = run_backtest(df, ind, tz_name, session_start, best_range_minutes, best_reward_risk,
                               apply_filters=True, split_after=SPLIT_DATE)
        r = sum(t["r"] for t in trades)
        total_r_oos += r
        n_trades_oos += len(trades)
        per_index_oos[label] = (len(trades), r)

    print(f"\nSAME combination, OUT-OF-SAMPLE ({SPLIT_DATE} to {FETCH_END.date()}):")
    if n_trades_oos:
        print(f"  {n_trades_oos} trades, {total_r_oos:+.2f}R, {total_r_oos/n_trades_oos:+.4f}R/trade")
    else:
        print("  0 trades")
    for label, (n, r) in per_index_oos.items():
        print(f"    {label}: {n} trades, {r:+.2f}R")

    in_sample_avg = best["in_sample_avg_r"]
    oos_avg = total_r_oos / n_trades_oos if n_trades_oos else float("nan")
    print(f"\nIn-sample avg R/trade: {in_sample_avg:+.4f}   Out-of-sample avg R/trade: {oos_avg:+.4f}")
    if n_trades_oos and oos_avg < in_sample_avg * 0.5:
        print("Out-of-sample is much weaker than in-sample - treat the grid search result with real "
              "suspicion, this is what overfitting looks like.")

    # ================= PART 2: ML scoring =================
    print("\n" + "=" * 70)
    print("PART 2: ML classifier scoring raw breakouts (filters off, all candidates labeled), "
          "using the best RANGE_MINUTES from Part 1")
    print("=" * 70)

    in_sample_candidates, oos_candidates = [], []
    for label, (df, ind, tz_name, session_start) in data.items():
        in_sample_candidates.extend(run_backtest(df, ind, tz_name, session_start, best_range_minutes, 1.0,
                                                   apply_filters=False, split_before=SPLIT_DATE,
                                                   collect_candidates=True))
        oos_candidates.extend(run_backtest(df, ind, tz_name, session_start, best_range_minutes, 1.0,
                                            apply_filters=False, split_after=SPLIT_DATE,
                                            collect_candidates=True))

    print(f"In-sample candidates: {len(in_sample_candidates)}   Out-of-sample candidates: {len(oos_candidates)}")

    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        print("scikit-learn not installed - run '!pip install scikit-learn -q' and re-run. Skipping ML step.")
        return

    train_df = pd.DataFrame([{**c["features"], "label": c["label"]} for c in in_sample_candidates])
    test_df = pd.DataFrame([{**c["features"], "label": c["label"], "r": c["r"]} for c in oos_candidates])

    train_df = train_df.dropna(subset=FEATURE_COLUMNS + ["label"])
    test_clean = test_df.dropna(subset=FEATURE_COLUMNS + ["label"])
    print(f"After dropping incomplete-feature rows: train={len(train_df)}, test={len(test_clean)}")

    if len(train_df) < 30 or len(test_clean) < 10:
        print("Not enough clean candidates to train/evaluate a model meaningfully. Stopping here.")
        return

    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"])

    test_clean = test_clean.copy()
    test_clean["pred_proba"] = model.predict_proba(test_clean[FEATURE_COLUMNS])[:, 1]

    baseline_r = test_clean["r"].sum()
    print(f"\nBaseline (every raw breakout, no ML filtering): {len(test_clean)} trades, "
          f"{baseline_r:+.2f}R, {baseline_r/len(test_clean):+.4f}R/trade")

    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        filtered = test_clean[test_clean["pred_proba"] >= threshold]
        if len(filtered) == 0:
            print(f"  threshold {threshold:.2f}: 0 trades")
            continue
        total_r = filtered["r"].sum()
        print(f"  threshold {threshold:.2f}: {len(filtered):4d} trades, {total_r:+8.2f}R, "
              f"{total_r/len(filtered):+.4f}R/trade")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print(f"\nFeature importances:\n{importances}")

    print(f"\nCompare this out-of-sample R/trade against Part 1's hand-tuned-filter out-of-sample result "
          f"({oos_avg:+.4f}R/trade) - same test period, same indices, different selection method. "
          f"Whichever wins here is only a real finding if it's not a razor-thin difference on a few "
          f"hundred out-of-sample trades.")


if __name__ == "__main__":
    main()
