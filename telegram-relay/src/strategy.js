// Pure indicator / strategy functions shared between the live Cloudflare
// Worker (src/index.js) and the offline backtester (backtester/backtest.js).
// Keeping this logic in one module guarantees both consumers evaluate the
// exact same trend/arrow/RSI rules - no risk of the backtest drifting from
// what the live Worker actually alerts on.

export const MA_LENS = { fast: 21, mid: 50, slow: 200 };
export const RSI_LEN = 14;

// Wilder's RSI - matches Pine's built-in rsi().
export function wilderRSI(closes, len) {
  const rsi = new Array(closes.length).fill(null);
  if (closes.length < len + 1) return rsi;
  let gainSum = 0;
  let lossSum = 0;
  for (let i = 1; i <= len; i++) {
    const change = closes[i] - closes[i - 1];
    if (change >= 0) gainSum += change;
    else lossSum -= change;
  }
  let avgGain = gainSum / len;
  let avgLoss = lossSum / len;
  rsi[len] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  for (let i = len + 1; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1];
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? -change : 0;
    avgGain = (avgGain * (len - 1) + gain) / len;
    avgLoss = (avgLoss * (len - 1) + loss) / len;
    rsi[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  }
  return rsi;
}

// Matches the smma pattern from the source Pine scripts: seeded with an
// SMA(len), then Wilder-style smoothing from there on.
export function smoothedMA(values, len) {
  const out = new Array(values.length).fill(null);
  for (let i = len - 1; i < values.length; i++) {
    if (out[i - 1] == null) {
      let sum = 0;
      for (let j = i - len + 1; j <= i; j++) sum += values[j];
      out[i] = sum / len;
    } else {
      out[i] = (out[i - 1] * (len - 1) + values[i]) / len;
    }
  }
  return out;
}

// CONFIRM_BARS requires the trend condition to hold for this many consecutive
// 5-minute bars (not just the current one) before it counts. A single sharp
// spike can flip the raw MA stack for one bar even while the broader move is
// still the other way - requiring it to persist filters that out.
export const CONFIRM_BARS = 6; // 30 minutes at 5-min candles

export function computeTrend(closes, maLens = MA_LENS, confirmBars = CONFIRM_BARS) {
  const ma21 = smoothedMA(closes, maLens.fast);
  const ma50 = smoothedMA(closes, maLens.mid);
  const ma200 = smoothedMA(closes, maLens.slow);
  const n = closes.length;
  const start = n - confirmBars;
  if (start < 0) return "none";

  let allUp = true;
  let allDown = true;
  for (let i = start; i < n; i++) {
    if (ma21[i] == null || ma50[i] == null || ma200[i] == null) return "none";
    const up = closes[i] > ma200[i] && ma21[i] > ma50[i] && ma50[i] > ma200[i];
    const down = closes[i] < ma200[i] && ma21[i] < ma50[i] && ma50[i] < ma200[i];
    if (!up) allUp = false;
    if (!down) allDown = false;
  }
  if (allUp) return "up";
  if (allDown) return "down";
  return "none";
}

export function computeArrows(candles) {
  const n = candles.length - 1;
  if (n < 3) return { bull: false, bear: false };
  const c0 = candles[n];
  const c1 = candles[n - 1];
  const c2 = candles[n - 2];
  const c3 = candles[n - 3];

  const strike3Bull = c3.close < c3.open && c2.close < c2.open && c1.close < c1.open && c0.close > c1.open;
  const strike3Bear = c3.close > c3.open && c2.close > c2.open && c1.close > c1.open && c0.close < c1.open;

  const engulfBull = c0.open <= c1.close && c0.open < c1.open && c0.close > c1.open;
  const engulfBear = c0.open >= c1.close && c0.open > c1.open && c0.close < c1.open;

  return { bull: strike3Bull || engulfBull, bear: strike3Bear || engulfBear };
}

// Combines trend + arrow + RSI into the buy/sell setup decision. `candles`
// must be ascending-chronological, and the setup is evaluated against the
// *last* candle in the array (i.e. call with candles.slice(0, i + 1) to
// evaluate as-of bar i).
export function evaluateSetup(candles, { rsiLen = RSI_LEN, maLens = MA_LENS } = {}) {
  const closes = candles.map((c) => c.close);
  const trend = computeTrend(closes, maLens);
  const arrows = computeArrows(candles);
  const rsi = wilderRSI(closes, rsiLen);
  const currentRSI = rsi[rsi.length - 1];

  const buySetup = trend === "up" && arrows.bull && currentRSI != null && currentRSI > 50;
  const sellSetup = trend === "down" && arrows.bear && currentRSI != null && currentRSI < 50;

  return { trend, arrows, currentRSI, buySetup, sellSetup };
}

// --- Donchian channel breakout (Turtle Trading style) ---
// The most credibly-evidenced mechanical strategy out of everything
// researched: a real documented 1980s track record. Breakout of an N-bar
// high/low, with an ATR-based stop instead of the candle-range heuristic
// used above - ATR is the standard volatility measure this style of
// system actually uses.

export const DONCHIAN_LEN = 20; // bars in the breakout channel
export const ATR_LEN = 14;

// Wilder's ATR - same smoothing pattern as wilderRSI.
export function atr(candles, len = ATR_LEN) {
  const out = new Array(candles.length).fill(null);
  if (candles.length < len + 1) return out;

  const tr = new Array(candles.length).fill(null);
  for (let i = 1; i < candles.length; i++) {
    const c = candles[i];
    const prevClose = candles[i - 1].close;
    tr[i] = Math.max(c.high - c.low, Math.abs(c.high - prevClose), Math.abs(c.low - prevClose));
  }

  let sum = 0;
  for (let i = 1; i <= len; i++) sum += tr[i];
  let avg = sum / len;
  out[len] = avg;
  for (let i = len + 1; i < candles.length; i++) {
    avg = (avg * (len - 1) + tr[i]) / len;
    out[i] = avg;
  }
  return out;
}

// Highest high / lowest low over the `len` bars BEFORE the current one
// (excludes the current bar, so today's own high/low can't count as its
// own breakout level - that would make every bar trivially a "breakout").
export function donchianChannel(candles, len = DONCHIAN_LEN) {
  const n = candles.length;
  const i = n - 1;
  if (i - len < 0) return { upper: null, lower: null };

  let upper = -Infinity;
  let lower = Infinity;
  for (let j = i - len; j < i; j++) {
    if (candles[j].high > upper) upper = candles[j].high;
    if (candles[j].low < lower) lower = candles[j].low;
  }
  return { upper, lower };
}

// candles must be ascending-chronological; evaluated against the last candle.
export function evaluateBreakout(candles, { donchianLen = DONCHIAN_LEN, atrLen = ATR_LEN } = {}) {
  const current = candles[candles.length - 1];
  const { upper, lower } = donchianChannel(candles, donchianLen);
  const atrSeries = atr(candles, atrLen);
  const currentATR = atrSeries[atrSeries.length - 1];

  const buySetup = upper != null && currentATR != null && current.close > upper;
  const sellSetup = lower != null && currentATR != null && current.close < lower;

  return { upper, lower, currentATR, buySetup, sellSetup };
}
