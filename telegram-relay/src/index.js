// Polls TwelveData for closed 5-minute forex candles, recomputes the setup
// (trend-aligned 3 Line Strike / Engulfing arrow + RSI vs 50, gated by a
// 21/50/200 smoothed-MA trend stack), and messages Telegram when it fires.
// Runs on a Cloudflare Cron Trigger (see wrangler.toml) - no TradingView
// alert / paid plan involved.

const PAIRS = [
  { symbol: "EUR/USD", pip: 0.0001 },
  { symbol: "GBP/USD", pip: 0.0001 },
  { symbol: "USD/JPY", pip: 0.01 },
];

const INTERVAL = "5min";
const OUTPUT_SIZE = 300; // bars of warmup history for the 200-period MA
const RSI_LEN = 14;
const MA_LENS = { fast: 21, mid: 50, slow: 200 };

// Trading window: 08:00 to 02:30 (next day), Europe/London local time.
// Wraps past midnight, so "active" means >= start OR <= end.
function isWithinTradingWindow(now = new Date()) {
  const parts = new Intl.DateTimeFormat("en-GB", {
    timeZone: "Europe/London",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).formatToParts(now);
  const hour = Number(parts.find((p) => p.type === "hour").value);
  const minute = Number(parts.find((p) => p.type === "minute").value);
  const minutesNow = hour * 60 + minute;
  const startMinutes = 8 * 60; // 08:00
  const endMinutes = 2 * 60 + 30; // 02:30
  return minutesNow >= startMinutes || minutesNow <= endMinutes;
}

// Wilder's RSI - matches Pine's built-in rsi().
function wilderRSI(closes, len) {
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
function smoothedMA(values, len) {
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

function computeTrend(closes) {
  const ma21 = smoothedMA(closes, MA_LENS.fast);
  const ma50 = smoothedMA(closes, MA_LENS.mid);
  const ma200 = smoothedMA(closes, MA_LENS.slow);
  const i = closes.length - 1;
  if (ma21[i] == null || ma50[i] == null || ma200[i] == null) return "none";
  if (closes[i] > ma200[i] && ma21[i] > ma50[i] && ma50[i] > ma200[i]) return "up";
  if (closes[i] < ma200[i] && ma21[i] < ma50[i] && ma50[i] < ma200[i]) return "down";
  return "none";
}

function computeArrows(candles) {
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

async function fetchCandles(symbol, apiKey) {
  const url = `https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(symbol)}&interval=${INTERVAL}&outputsize=${OUTPUT_SIZE}&apikey=${apiKey}`;
  const resp = await fetch(url);
  const data = await resp.json();
  if (!data.values) {
    throw new Error(`TwelveData error for ${symbol}: ${data.message || JSON.stringify(data)}`);
  }
  // TwelveData returns most-recent-first; we want ascending chronological order.
  return data.values
    .map((v) => ({
      time: v.datetime,
      open: Number(v.open),
      high: Number(v.high),
      low: Number(v.low),
      close: Number(v.close),
    }))
    .reverse();
}

async function sendTelegram(env, text) {
  const resp = await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: env.TELEGRAM_CHAT_ID, text, parse_mode: "Markdown" }),
  });
  if (!resp.ok) {
    throw new Error(`Telegram error: ${await resp.text()}`);
  }
}

async function checkPair(env, pair) {
  const candles = await fetchCandles(pair.symbol, env.TWELVEDATA_API_KEY);
  if (candles.length < MA_LENS.slow + 5) {
    return { symbol: pair.symbol, skipped: "not enough history" };
  }

  const closes = candles.map((c) => c.close);
  const trend = computeTrend(closes);
  const arrows = computeArrows(candles);
  const rsi = wilderRSI(closes, RSI_LEN);
  const current = candles[candles.length - 1];
  const currentRSI = rsi[rsi.length - 1];

  const buySetup = trend === "up" && arrows.bull && currentRSI > 50;
  const sellSetup = trend === "down" && arrows.bear && currentRSI < 50;

  if (!buySetup && !sellSetup) {
    return { symbol: pair.symbol, skipped: "no setup", trend, currentRSI };
  }

  const dedupeKey = `${pair.symbol}:${current.time}`;
  if (env.ALERT_STATE) {
    const already = await env.ALERT_STATE.get(dedupeKey);
    if (already) {
      return { symbol: pair.symbol, skipped: "already alerted this candle" };
    }
  }

  const side = buySetup ? "BUY" : "SELL";
  const rangePips = (current.high - current.low) / pair.pip;
  const slPips = Math.round(rangePips * 2 * 10) / 10;
  const tpPips = Math.round(slPips * 2 * 10) / 10;
  const emoji = side === "BUY" ? "\u{1F7E2}" : "\u{1F534}";

  const text = [
    `${emoji} *${side} SETUP* - ${pair.symbol}`,
    `Candle close (${INTERVAL}): ${current.close}`,
    `Time: ${current.time} UTC`,
    `Trend: ${trend}, RSI: ${currentRSI.toFixed(1)}`,
    `Suggested SL: ${slPips} pips, TP: ${tpPips} pips (2:1)`,
  ].join("\n");

  await sendTelegram(env, text);

  if (env.ALERT_STATE) {
    await env.ALERT_STATE.put(dedupeKey, "1", { expirationTtl: 60 * 60 * 24 });
  }

  return { symbol: pair.symbol, alerted: side };
}

async function runAllChecks(env) {
  if (!isWithinTradingWindow()) {
    return { skipped: "outside trading window" };
  }
  const results = [];
  for (const pair of PAIRS) {
    try {
      results.push(await checkPair(env, pair));
    } catch (err) {
      results.push({ symbol: pair.symbol, error: String(err) });
    }
  }
  return { results };
}

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(runAllChecks(env));
  },

  // Manual trigger for testing: GET /?secret=YOUR_SECRET
  async fetch(request, env) {
    const url = new URL(request.url);
    const secret = url.searchParams.get("secret");
    if (!env.WEBHOOK_SECRET || secret !== env.WEBHOOK_SECRET) {
      return new Response("Unauthorized", { status: 401 });
    }
    if (url.searchParams.get("ping") === "1") {
      await sendTelegram(env, "✅ Test message from the forex setup alert worker.");
      return new Response("Sent test message", { status: 200 });
    }
    const summary = await runAllChecks(env);
    return new Response(JSON.stringify(summary, null, 2), {
      headers: { "Content-Type": "application/json" },
    });
  },
};
