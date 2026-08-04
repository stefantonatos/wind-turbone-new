// TMA Trend Scalper (vFinal) - the rules the live alerts actually run on.
//
// WHY THIS IS A SEPARATE FILE FROM strategy.js: strategy.js holds the ORIGINAL,
// simpler setup (21/50/200 stack + pattern + RSI>50, with a 6-bar trend-persistence
// confirm) and is imported by backtester/backtest.js. Replacing it in place would
// silently change what that backtester measures. This module is additive; the old
// one is left exactly as it was.
//
// WHAT CHANGED vs strategy.js, and why the old alerts were not accurate for this
// strategy - every one of these gates exists in vFinal and in none of the old code:
//
//   ADX(14) > 25            trend strength
//   ATR(14) > 0.7 x SMA50   volatility, so dead sessions are skipped
//   close vs SMA(5), close[5]   short-term momentum agreeing with the trend
//   SMMA separation > minDist   per-pair, so a "stack" of three touching lines
//                               does not count as a trend
//   RSI > 50 AND > SMMA(50) of RSI   the old code checked only RSI > 50
//   session 07:00-15:00 UTC, Mon-Fri   the old window was 08:00-02:30 London
//   one trade per instrument per day
//
// The indicator maths is a direct port of
// research/tma_trend_scalper_forex_dukascopy_backtest.py, which was itself checked
// against Pine's ta.rma / ta.atr / ta.dmi / ta.rsi. It is deliberately NOT
// re-derived here - the Python file is the reference implementation and any change
// belongs in both.

export const SMMA_FAST = 21;
export const SMMA_MED = 50;
export const SMMA_SLOW = 200;
export const RSI_LEN = 14;
export const RSI_SMMA_LEN = 50;
export const ADX_LEN = 14;
export const ADX_MIN = 25.0;
export const ATR_LEN = 14;
export const ATR_AVG_LEN = 50;
export const ATR_MIN_MULT = 0.7;
export const MOMENTUM_SMA_LEN = 5;
export const MOMENTUM_LOOKBACK = 5;
export const STOP_CANDLE_MULT = 2.0;
export const TARGET_CANDLE_MULT = 4.0;

export const SESSION_START_HOUR = 7; // UTC, inclusive
export const SESSION_END_HOUR = 15; // UTC, exclusive

// Bars of history needed before the first evaluable signal: the 200 SMMA plus the
// 50-period RSI SMMA stacked on top of a 14-period RSI, plus slack.
export const MIN_BARS = SMMA_SLOW + RSI_SMMA_LEN + 10;

// ---------------------------------------------------------------------------
// Primitives. Each returns an array the same length as its input, with null
// wherever there is not yet enough history - never a substituted zero, which
// would quietly drag every average built on top of it toward zero.
// ---------------------------------------------------------------------------

export function smaSeries(values, length, startIndex = 0) {
  const n = values.length;
  const out = new Array(n).fill(null);
  let windowSum = 0;
  let count = 0;
  for (let i = startIndex; i < n; i++) {
    if (values[i] == null) return out;
    windowSum += values[i];
    count += 1;
    if (count > length) {
      windowSum -= values[i - length];
      count = length;
    }
    if (count === length) out[i] = windowSum / length;
  }
  return out;
}

// Pine's ta.rma / Wilder smoothing / this project's SMMA: seeded with the simple
// average of the first `length` values from startIndex, then recursive.
export function rmaSeries(values, length, startIndex = 0) {
  const n = values.length;
  const out = new Array(n).fill(null);
  let usable = 0;
  for (let i = startIndex; i < n; i++) if (values[i] != null) usable += 1;
  if (usable < length) return out;

  const seedEnd = startIndex + length;
  let sum = 0;
  for (let i = startIndex; i < seedEnd; i++) sum += values[i];
  let prev = sum / length;
  out[seedEnd - 1] = prev;
  for (let i = seedEnd; i < n; i++) {
    prev = (prev * (length - 1) + values[i]) / length;
    out[i] = prev;
  }
  return out;
}

export function trueRangeSeries(candles) {
  const out = [null];
  for (let i = 1; i < candles.length; i++) {
    const c = candles[i];
    const prevClose = candles[i - 1].close;
    out.push(Math.max(c.high - c.low, Math.abs(c.high - prevClose), Math.abs(c.low - prevClose)));
  }
  return out;
}

export function atrSeries(candles, length = ATR_LEN) {
  return rmaSeries(trueRangeSeries(candles), length, 1);
}

// Wilder's ADX, matching Pine's ta.dmi(len, len): directional movement smoothed by
// rma, normalised by smoothed true range, then the DX itself smoothed by rma again.
export function adxSeries(candles, length = ADX_LEN) {
  const n = candles.length;
  const tr = trueRangeSeries(candles);
  const plusDM = new Array(n).fill(null);
  const minusDM = new Array(n).fill(null);
  for (let i = 1; i < n; i++) {
    const up = candles[i].high - candles[i - 1].high;
    const down = candles[i - 1].low - candles[i].low;
    plusDM[i] = up > down && up > 0 ? up : 0;
    minusDM[i] = down > up && down > 0 ? down : 0;
  }

  const trur = rmaSeries(tr, length, 1);
  const plusSm = rmaSeries(plusDM, length, 1);
  const minusSm = rmaSeries(minusDM, length, 1);

  const dx = new Array(n).fill(null);
  let firstDx = null;
  for (let i = 0; i < n; i++) {
    if (trur[i] == null || plusSm[i] == null || minusSm[i] == null || trur[i] === 0) continue;
    const plus = (100 * plusSm[i]) / trur[i];
    const minus = (100 * minusSm[i]) / trur[i];
    const total = plus + minus;
    dx[i] = (100 * Math.abs(plus - minus)) / (total !== 0 ? total : 1);
    if (firstDx == null) firstDx = i;
  }
  if (firstDx == null) return new Array(n).fill(null);
  return rmaSeries(dx, length, firstDx);
}

// Pine's ta.rsi: rma of upward and downward changes.
export function rsiSeries(closes, length = RSI_LEN) {
  const n = closes.length;
  const gains = new Array(n).fill(null);
  const losses = new Array(n).fill(null);
  for (let i = 1; i < n; i++) {
    const change = closes[i] - closes[i - 1];
    gains[i] = Math.max(change, 0);
    losses[i] = Math.max(-change, 0);
  }
  const avgGain = rmaSeries(gains, length, 1);
  const avgLoss = rmaSeries(losses, length, 1);
  const out = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    if (avgGain[i] == null || avgLoss[i] == null) continue;
    if (avgLoss[i] === 0) out[i] = 100;
    else if (avgGain[i] === 0) out[i] = 0;
    else out[i] = 100 - 100 / (1 + avgGain[i] / avgLoss[i]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

// `time` is TwelveData's "YYYY-MM-DD HH:MM:SS" string, already requested with
// timezone=UTC, so it is parsed as UTC rather than in the Worker's local zone.
export function parseUTC(timeStr) {
  return new Date(`${timeStr.replace(" ", "T")}Z`);
}

export const BAR_MINUTES = 5;

// TwelveData stamps a bar with its OPEN time and returns the in-progress bar as the
// most recent value. So the newest candle is still forming until open + 5 minutes.
// This matters twice over: an early-warning pass must evaluate the forming bar (that
// is the whole point), and a confirmation pass must NOT - it has to score the bar
// that actually closed, or it re-reads a provisional bar and calls it confirmed.
export function minutesToClose(candle, now = new Date()) {
  const closeMs = parseUTC(candle.time).getTime() + BAR_MINUTES * 60000;
  return (closeMs - now.getTime()) / 60000;
}

export function isForming(candle, now = new Date()) {
  return minutesToClose(candle, now) > 0;
}

/**
 * Splits a series into the bars that have definitely closed and the one still
 * forming, if any. `closed` is always safe to evaluate as a finished signal.
 */
export function splitCandles(candles, now = new Date()) {
  if (candles.length === 0) return { closed: [], forming: null };
  const last = candles[candles.length - 1];
  if (isForming(last, now)) {
    return { closed: candles.slice(0, -1), forming: last };
  }
  return { closed: candles, forming: null };
}

export function inSession(timeStr) {
  const d = parseUTC(timeStr);
  const day = d.getUTCDay(); // 0=Sunday .. 6=Saturday
  if (day === 0 || day === 6) return false;
  const hour = d.getUTCHours();
  return hour >= SESSION_START_HOUR && hour < SESSION_END_HOUR;
}

// ---------------------------------------------------------------------------
// The setup
// ---------------------------------------------------------------------------

/**
 * Evaluates the vFinal setup against the LAST candle in `candles`
 * (ascending-chronological). `minDist` is the per-pair absolute SMMA separation
 * (0.001 for EURUSD/GBPUSD/AUDUSD-class pairs, 0.10 for JPY pairs).
 *
 * Always returns the full gate breakdown, not just a boolean, so the alert and the
 * /debug endpoint can both say WHICH condition blocked a signal rather than
 * reporting a bare "no setup".
 */
export function evaluateTMA(candles, { minDist = 0.001 } = {}) {
  const n = candles.length;
  if (n < MIN_BARS) return { ok: false, reason: `need ${MIN_BARS} bars, have ${n}` };

  const closes = candles.map((c) => c.close);
  const opens = candles.map((c) => c.open);

  const smmaFast = rmaSeries(closes, SMMA_FAST, 0);
  const smmaMed = rmaSeries(closes, SMMA_MED, 0);
  const smmaSlow = rmaSeries(closes, SMMA_SLOW, 0);
  const adx = adxSeries(candles, ADX_LEN);
  const atr = atrSeries(candles, ATR_LEN);
  const atrFirst = atr.findIndex((v) => v != null);
  const atrAvg = smaSeries(atr, ATR_AVG_LEN, atrFirst < 0 ? 0 : atrFirst);
  const momSma = smaSeries(closes, MOMENTUM_SMA_LEN, 0);
  const rsi = rsiSeries(closes, RSI_LEN);
  const rsiFirst = rsi.findIndex((v) => v != null);
  const rsiSmma = rmaSeries(rsi, RSI_SMMA_LEN, rsiFirst < 0 ? 0 : rsiFirst);

  const i = n - 1;
  const need = [smmaFast[i], smmaMed[i], smmaSlow[i], adx[i], atr[i], atrAvg[i], momSma[i], rsi[i], rsiSmma[i]];
  if (need.some((v) => v == null)) return { ok: false, reason: "indicators not warmed up" };

  const c = candles[i];
  const sessionOK = inSession(c.time);

  const bullStack = smmaFast[i] > smmaMed[i] + minDist && smmaMed[i] > smmaSlow[i] + minDist;
  const bearStack = smmaFast[i] < smmaMed[i] - minDist && smmaMed[i] < smmaSlow[i] - minDist;
  const trending = adx[i] > ADX_MIN;
  const volatile_ = atr[i] > atrAvg[i] * ATR_MIN_MULT;
  const bullMom = closes[i] > momSma[i] && closes[i] > closes[i - MOMENTUM_LOOKBACK];
  const bearMom = closes[i] < momSma[i] && closes[i] < closes[i - MOMENTUM_LOOKBACK];

  const bullQuality = bullStack && trending && closes[i] > smmaSlow[i] && bullMom && volatile_;
  const bearQuality = bearStack && trending && closes[i] < smmaSlow[i] && bearMom && volatile_;

  const threeUp = closes[i - 3] > opens[i - 3] && closes[i - 2] > opens[i - 2] && closes[i - 1] > opens[i - 1];
  const threeDown = closes[i - 3] < opens[i - 3] && closes[i - 2] < opens[i - 2] && closes[i - 1] < opens[i - 1];
  const bullStrike = threeDown && closes[i] > opens[i - 1];
  const bearStrike = threeUp && closes[i] < opens[i - 1];
  const bullEngulf = opens[i] <= closes[i - 1] && opens[i] < opens[i - 1] && closes[i] > opens[i - 1];
  const bearEngulf = opens[i] >= closes[i - 1] && opens[i] > opens[i - 1] && closes[i] < opens[i - 1];

  const rsiBull = rsi[i] > 50 && rsi[i] > rsiSmma[i];
  const rsiBear = rsi[i] < 50 && rsi[i] < rsiSmma[i];

  const buySetup = sessionOK && bullQuality && (bullStrike || bullEngulf) && rsiBull;
  const sellSetup = sessionOK && bearQuality && (bearStrike || bearEngulf) && rsiBear;

  const side = buySetup ? "BUY" : sellSetup ? "SELL" : null;
  const pattern = side === "BUY"
    ? (bullStrike ? "3-Line Strike" : "Engulfing")
    : side === "SELL"
      ? (bearStrike ? "3-Line Strike" : "Engulfing")
      : null;

  // Stop is 2x the signal candle's own range, target 4x - so 2:1, with the size
  // varying bar to bar. That is the rule as documented; it is also why position
  // size has to be derived from the stop distance rather than fixed.
  const candleSize = c.high - c.low;
  const levels = side
    ? {
        entry: c.close,
        stop: side === "BUY" ? c.close - candleSize * STOP_CANDLE_MULT : c.close + candleSize * STOP_CANDLE_MULT,
        target: side === "BUY" ? c.close + candleSize * TARGET_CANDLE_MULT : c.close - candleSize * TARGET_CANDLE_MULT,
        candleSize,
      }
    : null;

  return {
    ok: true,
    side,
    pattern,
    levels,
    // Full breakdown, in the same order and with the same direction-awareness as the
    // confluence table in pine/tma-trend-scalper.pine, so the two can be read side by
    // side row for row when checking the bot against the chart.
    //
    // DIRECTION-AWARE ON PURPOSE. Every gate below the stack is asked about the side
    // the stack actually permits - "did a BULLISH pattern fire", not "did any pattern
    // fire". A bearish 3-Line Strike inside a bullish stack is not a pass; reporting
    // it as one made this table disagree with the Pine's, which is exactly the
    // comparison it exists to support.
    gates: (() => {
      const dir = bullStack ? "bull" : bearStack ? "bear" : null;
      const pick = (bull, bear) => (dir === "bull" ? bull : dir === "bear" ? bear : false);
      return {
        session: sessionOK,
        stack: dir !== null,
        adx: trending,
        volatility: volatile_,
        priceVs200: pick(closes[i] > smmaSlow[i], closes[i] < smmaSlow[i]),
        momentum: pick(bullMom, bearMom),
        pattern: pick(bullStrike || bullEngulf, bearStrike || bearEngulf),
        rsi: pick(rsiBull, rsiBear),
      };
    })(),
    values: {
      adx: adx[i],
      rsi: rsi[i],
      rsiSmma: rsiSmma[i],
      atrRatio: atrAvg[i] !== 0 ? atr[i] / atrAvg[i] : null,
      smmaFast: smmaFast[i],
      smmaMed: smmaMed[i],
      smmaSlow: smmaSlow[i],
      close: c.close,
      time: c.time,
    },
    direction: bullQuality ? "bull" : bearQuality ? "bear" : "none",
  };
}

/** The first gate that is false, for a human-readable "why no signal" line. */
export function firstBlockingGate(gates) {
  for (const [name, ok] of Object.entries(gates)) {
    if (!ok) return name;
  }
  return null;
}
