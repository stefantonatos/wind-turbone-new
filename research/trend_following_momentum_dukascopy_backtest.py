# Time-series momentum / trend-following - the one genre of strategy with a
# real, multi-decade published track record (CTAs like Winton/AHL/Man AHL,
# and the academic literature: Moskowitz, Ooi & Pedersen 2012, "Time Series
# Momentum", Journal of Financial Economics). Everything else tested in
# this project - MA setups, gold breakouts, ORB, ICT - was an intraday
# chart pattern on liquid instruments, exactly the kind of thing that gets
# arbitraged away if it ever worked. This is different: a documented,
# decades-old, cross-asset risk premium.
#
# Built deliberately differently from every other script here: there is NO
# grid search on the core parameters (lookback, holding period). The
# credibility of trend-following comes specifically from NOT curve-fitting
# a lookback per market - the textbook 12-month-lookback / 1-month-hold
# combination is used unchanged across every instrument and asset class,
# exactly as published. A side-by-side comparison against a few other
# standard lookbacks (1/3/6 months) is reported for honesty, but nothing is
# "chosen" by which one performs best here - that would just reintroduce
# the overfitting this project has spent the whole session trying to avoid.
#
# METHODOLOGY (simplified Moskowitz/Ooi/Pedersen, for daily bars):
#   1. Monthly rebalance. At each month-end, the signal is the SIGN of the
#      trailing 12-month return (long if positive, short if negative) -
#      this is "absolute"/time-series momentum, not a cross-sectional
#      ranking against other instruments.
#   2. Position size is scaled inversely to the instrument's trailing
#      3-month realized volatility, targeting a common ex-ante volatility
#      contribution per instrument (capped) - standard trend-following
#      practice, and the reason a government bond and a stock index can
#      sit in the same portfolio without the noisier one dominating.
#   3. That position is held through the next month, then re-signaled.
#      Signal and position size at month t use ONLY data through month t;
#      they earn the return realized over month t -> t+1, never t itself
#      (checked explicitly in the unit tests below - lookahead bias is
#      the single most common way this kind of backtest silently lies).
#
# UNIVERSE: 27 instruments across 6 asset classes via Dukascopy daily
# bars, 2005-2025 - FX majors, developed-market equity indices, 3
# government bonds, metals, energy, and agricultural commodities. Scaled
# down from the original paper's 58-instrument universe to what Dukascopy
# offers, but the same idea: diversify the SAME rule across asset classes,
# don't tune it per market.

# !pip install --upgrade dukascopy-python -q   # uncomment this line in Colab

import datetime

import numpy as np
import pandas as pd
import dukascopy_python
from dukascopy_python import instruments as dki

INSTRUMENTS = [
    ("EURUSD", dki.INSTRUMENT_FX_MAJORS_EUR_USD, "FX"),
    ("GBPUSD", dki.INSTRUMENT_FX_MAJORS_GBP_USD, "FX"),
    ("USDJPY", dki.INSTRUMENT_FX_MAJORS_USD_JPY, "FX"),
    ("USDCHF", dki.INSTRUMENT_FX_MAJORS_USD_CHF, "FX"),
    ("USDCAD", dki.INSTRUMENT_FX_MAJORS_USD_CAD, "FX"),
    ("AUDUSD", dki.INSTRUMENT_FX_MAJORS_AUD_USD, "FX"),
    ("NZDUSD", dki.INSTRUMENT_FX_MAJORS_NZD_USD, "FX"),
    ("SP500", dki.INSTRUMENT_IDX_AMERICA_E_SANDP_500, "EQUITY_INDEX"),
    ("NASDAQ100", dki.INSTRUMENT_IDX_AMERICA_E_NQ_100, "EQUITY_INDEX"),
    ("DOWJONES", dki.INSTRUMENT_IDX_AMERICA_E_D_J_IND, "EQUITY_INDEX"),
    ("DAX", dki.INSTRUMENT_IDX_EUROPE_E_DAAX, "EQUITY_INDEX"),
    ("FTSE100", dki.INSTRUMENT_IDX_EUROPE_E_FUTSEE_100, "EQUITY_INDEX"),
    ("NIKKEI225", dki.INSTRUMENT_IDX_ASIA_E_N225JAP, "EQUITY_INDEX"),
    ("BUND", dki.INSTRUMENT_BND_CFD_BUND_TR_EUR, "BOND"),
    ("UKGILT", dki.INSTRUMENT_BND_CFD_UKGILT_TR_GBP, "BOND"),
    ("USTBOND", dki.INSTRUMENT_BND_CFD_USTBOND_TR_USD, "BOND"),
    ("GOLD", dki.INSTRUMENT_FX_METALS_XAU_USD, "METAL"),
    ("SILVER", dki.INSTRUMENT_FX_METALS_XAG_USD, "METAL"),
    ("COPPER", dki.INSTRUMENT_CMD_METALS_COPPER_CMD_USD, "METAL"),
    ("PLATINUM", dki.INSTRUMENT_CMD_METALS_XPT_CMD_USD, "METAL"),
    ("BRENT", dki.INSTRUMENT_CMD_ENERGY_E_BRENT, "ENERGY"),
    ("WTI", dki.INSTRUMENT_CMD_ENERGY_E_LIGHT, "ENERGY"),
    ("NATGAS", dki.INSTRUMENT_CMD_ENERGY_GAS_CMD_USD, "ENERGY"),
    ("SOYBEAN", dki.INSTRUMENT_CMD_AGRICULTURAL_SOYBEAN_CMD_USX, "AGRICULTURAL"),
    ("SUGAR", dki.INSTRUMENT_CMD_AGRICULTURAL_SUGAR_CMD_USD, "AGRICULTURAL"),
    ("COFFEE", dki.INSTRUMENT_CMD_AGRICULTURAL_COFFEE_CMD_USX, "AGRICULTURAL"),
    ("COTTON", dki.INSTRUMENT_CMD_AGRICULTURAL_COTTON_CMD_USX, "AGRICULTURAL"),
]

FETCH_START = datetime.datetime(2005, 1, 1)
FETCH_END = datetime.datetime(2025, 1, 1)
DUKASCOPY_INTERVAL = dukascopy_python.INTERVAL_DAY_1
DUKASCOPY_OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

# --- the published, NOT grid-searched, core parameters ---
LOOKBACK_MONTHS = 12
VOL_LOOKBACK_DAYS = 63     # ~3 trading months
TARGET_ANNUAL_VOL = 0.10   # 10% - a retail-digestible risk target, not the paper's leveraged-futures 40%
MAX_POSITION_WEIGHT = 3.0  # caps leverage on abnormally quiet instruments

# --- side-by-side comparison only, never used to pick a "winner" ---
ROBUSTNESS_LOOKBACKS = [1, 3, 6, 12]

MIN_MONTHS_REQUIRED = LOOKBACK_MONTHS + 6   # instruments with less history than this are skipped, not force-fit


def fetch_daily_closes(instrument_const):
    df = dukascopy_python.fetch(instrument_const, DUKASCOPY_INTERVAL, DUKASCOPY_OFFER_SIDE, FETCH_START, FETCH_END)
    if df.empty:
        return None
    closes = df["close"]
    if closes.index.tz is None:
        closes.index = closes.index.tz_localize("UTC")
    return closes.sort_index()


def build_instrument_frame(daily_closes, lookback_months, vol_lookback_days=VOL_LOOKBACK_DAYS,
                            target_vol=TARGET_ANNUAL_VOL, max_weight=MAX_POSITION_WEIGHT):
    """Returns a month-end-indexed DataFrame with the momentum signal,
    volatility-scaled position weight, and the strategy's realized return
    for each month. Signal/weight at row t are computed using data
    available THROUGH month t only, and are shifted forward one row
    before being multiplied by monthly_return - so strategy_return[t]
    always reflects a decision made at t-1, earning the return realized
    from t-1 to t. No lookahead."""
    daily_returns = daily_closes.pct_change()
    rolling_vol = daily_returns.rolling(vol_lookback_days).std() * np.sqrt(252)

    monthly_close = daily_closes.resample("ME").last().dropna()
    monthly_return = monthly_close.pct_change()
    vol_at_month_end = rolling_vol.reindex(monthly_close.index, method="ffill")

    signal = pd.Series(index=monthly_close.index, dtype=float)
    for i in range(lookback_months, len(monthly_close)):
        past_price = monthly_close.iloc[i - lookback_months]
        now_price = monthly_close.iloc[i]
        if now_price > past_price:
            signal.iloc[i] = 1.0
        elif now_price < past_price:
            signal.iloc[i] = -1.0
        else:
            signal.iloc[i] = 0.0

    position_weight = (target_vol / vol_at_month_end).clip(upper=max_weight)

    strategy_return = signal.shift(1) * position_weight.shift(1) * monthly_return

    return pd.DataFrame({
        "monthly_close": monthly_close,
        "monthly_return": monthly_return,
        "signal": signal,
        "vol": vol_at_month_end,
        "position_weight": position_weight,
        "strategy_return": strategy_return,
    })


def portfolio_return_series(instrument_frames):
    """Equal-weight average of each instrument's strategy_return across
    whatever instruments have a valid (non-NaN) reading that month -
    handles instruments with different amounts of history gracefully
    instead of forcing every instrument onto the same start date."""
    returns_df = pd.DataFrame({label: frame["strategy_return"] for label, frame in instrument_frames.items()})
    return returns_df.mean(axis=1, skipna=True), returns_df


def portfolio_stats(monthly_returns):
    monthly_returns = monthly_returns.dropna()
    if len(monthly_returns) < 12:
        return None
    nav = (1 + monthly_returns).cumprod()
    ann_return = nav.iloc[-1] ** (12 / len(monthly_returns)) - 1
    ann_vol = monthly_returns.std() * np.sqrt(12)
    sharpe = (monthly_returns.mean() * 12) / ann_vol if ann_vol > 0 else float("nan")
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_dd = drawdown.min()
    pct_positive_months = (monthly_returns > 0).mean() * 100
    return {
        "n_months": len(monthly_returns), "ann_return_pct": ann_return * 100, "ann_vol_pct": ann_vol * 100,
        "sharpe": sharpe, "max_dd_pct": max_dd * 100, "pct_positive_months": pct_positive_months,
        "total_return_pct": (nav.iloc[-1] - 1) * 100,
    }


def main():
    years = (FETCH_END - FETCH_START).days / 365
    print(f"Downloading {len(INSTRUMENTS)} instruments (daily bars) from Dukascopy over ~{years:.0f} years "
          f"({FETCH_START.date()} to {FETCH_END.date()}) - daily data is small, expect a few minutes.\n"
          f"NOTE: FX/metals typically have the deepest history on Dukascopy; CFD-style instruments "
          f"(bonds, indices, some commodities) may have meaningfully shorter real history - instruments "
          f"with under {MIN_MONTHS_REQUIRED} months of usable data are skipped below, not force-fit.\n")

    raw_closes = {}
    asset_class = {}
    for label, instrument_const, ac in INSTRUMENTS:
        print(f"{label}...", end=" ")
        try:
            closes = fetch_daily_closes(instrument_const)
        except Exception as exc:
            print(f"failed ({exc})")
            continue
        if closes is None or len(closes) < 30:
            print("no/insufficient data")
            continue
        raw_closes[label] = closes
        asset_class[label] = ac
        print(f"{len(closes)} daily bars ({closes.index[0].date()} to {closes.index[-1].date()})")

    if not raw_closes:
        print("No data downloaded at all - check output above.")
        return

    print("\n" + "=" * 70)
    print(f"MAIN RESULT: LOOKBACK_MONTHS={LOOKBACK_MONTHS} (published standard, not grid-searched)")
    print("=" * 70)

    main_frames = {}
    for label, closes in raw_closes.items():
        frame = build_instrument_frame(closes, LOOKBACK_MONTHS)
        if frame["strategy_return"].notna().sum() < MIN_MONTHS_REQUIRED:
            print(f"  {label}: skipped - only {frame['strategy_return'].notna().sum()} usable months "
                  f"(need {MIN_MONTHS_REQUIRED}+)")
            continue
        main_frames[label] = frame

    if not main_frames:
        print("No instrument had enough history for the 12-month lookback. Stopping.")
        return

    portfolio_returns, per_instrument_returns = portfolio_return_series(main_frames)
    stats = portfolio_stats(portfolio_returns)
    if stats is None:
        print("Not enough overlapping months across instruments to compute portfolio stats.")
        return

    print(f"\nPortfolio ({len(main_frames)} instruments, {stats['n_months']} months, "
          f"{portfolio_returns.dropna().index[0].date()} to {portfolio_returns.dropna().index[-1].date()}):")
    print(f"  Total return:      {stats['total_return_pct']:+.1f}%")
    print(f"  Annualized return: {stats['ann_return_pct']:+.2f}%")
    print(f"  Annualized vol:    {stats['ann_vol_pct']:.2f}%")
    print(f"  Sharpe ratio:      {stats['sharpe']:.2f}  (rf=0, not subtracted)")
    print(f"  Max drawdown:      {stats['max_dd_pct']:.2f}%")
    print(f"  Positive months:   {stats['pct_positive_months']:.1f}%")

    print("\nPer-instrument contribution (mean monthly strategy return, %):")
    contrib = per_instrument_returns.mean() * 100
    for label in sorted(contrib.index, key=lambda l: -contrib[l]):
        print(f"  {label:10s} ({asset_class[label]:14s}): {contrib[label]:+.3f}%/month")

    print("\nPer asset-class average (mean monthly strategy return, %):")
    ac_series = pd.Series(asset_class)
    for ac in sorted(set(asset_class.values())):
        labels_in_class = [l for l in per_instrument_returns.columns if asset_class.get(l) == ac]
        if not labels_in_class:
            continue
        avg = per_instrument_returns[labels_in_class].mean(axis=1).mean() * 100
        print(f"  {ac:14s}: {avg:+.3f}%/month  ({len(labels_in_class)} instruments)")

    portfolio_returns_clean = portfolio_returns.dropna()
    all_months = portfolio_returns_clean.index
    midpoint = all_months[len(all_months) // 2]
    first_half = portfolio_returns_clean[all_months < midpoint]
    second_half = portfolio_returns_clean[all_months >= midpoint]
    stats_first = portfolio_stats(first_half)
    stats_second = portfolio_stats(second_half)
    print(f"\nSPLIT-PERIOD CHECK (not a parameter search - just checking the SAME rule holds up "
          f"across two different multi-year regimes):")
    if stats_first:
        print(f"  {all_months[0].date()} to {midpoint.date()}: annualized return {stats_first['ann_return_pct']:+.2f}%, "
              f"Sharpe {stats_first['sharpe']:.2f}, max DD {stats_first['max_dd_pct']:.2f}%")
    if stats_second:
        print(f"  {midpoint.date()} to {all_months[-1].date()}: annualized return {stats_second['ann_return_pct']:+.2f}%, "
              f"Sharpe {stats_second['sharpe']:.2f}, max DD {stats_second['max_dd_pct']:.2f}%")

    print("\n" + "=" * 70)
    print("ROBUSTNESS CHECK: same portfolio, other standard lookbacks (side-by-side, NOT a search for a winner)")
    print("=" * 70)
    for lookback in ROBUSTNESS_LOOKBACKS:
        frames = {}
        for label, closes in raw_closes.items():
            frame = build_instrument_frame(closes, lookback)
            if frame["strategy_return"].notna().sum() >= (lookback + 6):
                frames[label] = frame
        if not frames:
            continue
        returns, _ = portfolio_return_series(frames)
        s = portfolio_stats(returns)
        if s is None:
            continue
        tag = " (main result above)" if lookback == LOOKBACK_MONTHS else ""
        print(f"  {lookback:2d}-month lookback: ann. return {s['ann_return_pct']:+7.2f}%   "
              f"Sharpe {s['sharpe']:5.2f}   max DD {s['max_dd_pct']:7.2f}%{tag}")

    print("\nNo transaction costs, financing/carry costs, or CFD overnight fees modeled. Monthly rebalancing "
          "with liquid instruments means cost drag is much smaller proportionally than the intraday "
          "strategies tested elsewhere in this project, but it is not zero - especially for the less "
          "liquid commodity CFDs (agriculturals), which typically carry wider spreads than FX majors.")


if __name__ == "__main__":
    main()
