// Polls TwelveData for closed 5-minute forex candles, recomputes the setup
// (trend-aligned 3 Line Strike / Engulfing arrow + RSI vs 50, gated by a
// 21/50/200 smoothed-MA trend stack), and messages Telegram when it fires.
// Runs on a Cloudflare Cron Trigger (see wrangler.toml) - no TradingView
// alert / paid plan involved.
//
// The indicator/strategy math itself (RSI, smoothed MAs, trend gate, arrow
// patterns, buy/sell setup combination) lives in ./strategy.js so the
// offline backtester (backtester/backtest.js) can reuse the exact same
// logic instead of a re-derived copy.

import { MA_LENS, evaluateSetup } from "./strategy.js";

const PAIRS = [
  { symbol: "EUR/USD", pip: 0.0001 },
  { symbol: "GBP/USD", pip: 0.0001 },
  { symbol: "USD/JPY", pip: 0.01 },
];

const INTERVAL = "5min";
const OUTPUT_SIZE = 300; // bars of warmup history for the 200-period MA

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

  const current = candles[candles.length - 1];
  const { trend, currentRSI, buySetup, sellSetup } = evaluateSetup(candles);

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
