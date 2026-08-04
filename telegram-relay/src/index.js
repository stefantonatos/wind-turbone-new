// Polls TwelveData for closed 5-minute forex candles, recomputes the TMA Trend
// Scalper (vFinal) setup, and messages Telegram with the trade levels and a chart
// image when it fires. Runs on a Cloudflare Cron Trigger (see wrangler.toml).
//
// WHAT REPLACED WHAT: this used to alert on the ORIGINAL, simpler setup in
// ./strategy.js - a 21/50/200 stack + pattern + RSI>50, over an 08:00-02:30 London
// window. That is a materially different strategy from the one being forward-tested
// now: it has no ADX gate, no ATR volatility gate, no momentum check, no minimum
// SMMA separation, and it compares RSI to 50 only rather than to its own SMMA(50).
// Those alerts were firing on setups the current strategy would reject, which is why
// they were not accurate. The rules now live in ./tma-strategy.js, whose indicator
// maths was diffed against research/tma_trend_scalper_forex_dukascopy_backtest.py
// and matches it to zero floating-point difference across SMMA/ADX/ATR/RSI.
//
// ./strategy.js is left untouched - backtester/backtest.js still imports it.

import { MIN_BARS, SESSION_END_HOUR, SESSION_START_HOUR, evaluateTMA, firstBlockingGate, inSession, parseUTC } from "./tma-strategy.js";

// TwelveData free tier is 800 requests/day and 8/minute. The cron fires every 5
// minutes but only spends quota inside the session, so the real budget is:
//   8 session hours x 12 fires/hour = 96 fires/day, x N pairs = 96N requests/day.
// 8 pairs = 768/day, which fits under 800 with enough headroom for the /ping and
// manual-trigger endpoints. Going past 8 breaks the daily cap AND collides with the
// 8-requests-per-minute limit, since one fire calls every pair back to back.
// `minDist` is the strategy's own per-pair SMMA separation - absolute price, so JPY
// pairs need a different number, not a scaled one.
const PAIRS = [
  { symbol: "AUD/USD", pip: 0.0001, minDist: 0.001 },
  { symbol: "EUR/USD", pip: 0.0001, minDist: 0.001 },
  { symbol: "GBP/USD", pip: 0.0001, minDist: 0.001 },
  { symbol: "NZD/USD", pip: 0.0001, minDist: 0.001 },
  { symbol: "USD/CAD", pip: 0.0001, minDist: 0.001 },
  { symbol: "USD/CHF", pip: 0.0001, minDist: 0.001 },
  { symbol: "USD/JPY", pip: 0.01, minDist: 0.10 },
  { symbol: "EUR/JPY", pip: 0.01, minDist: 0.10 },
];

const INTERVAL = "5min";
const OUTPUT_SIZE = 400; // > MIN_BARS (200 SMMA + 50-period RSI SMMA + slack)

// chart-img.com renders a real TradingView chart server-side. The Worker cannot
// screenshot TradingView itself, so this is the closest thing to "the chart you are
// looking at". Free tier is rate-limited; at roughly one signal per pair per day
// that is not a constraint. If the key is missing or the call fails the alert still
// goes out as text - a missing picture must never cost you the signal.
const CHART_IMG_ENDPOINT = "https://api.chart-img.com/v2/tradingview/advanced-chart";

async function resolveSecret(binding) {
  if (binding == null) return undefined;
  if (typeof binding === "string") return binding;
  if (typeof binding.get === "function") return await binding.get();
  return undefined;
}

function twelveDataSymbolToTradingView(symbol) {
  // "AUD/USD" -> "FX:AUDUSD"
  return `FX:${symbol.replace("/", "")}`;
}

async function fetchCandles(symbol, apiKey, outputSize = OUTPUT_SIZE) {
  const url = `https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(symbol)}&interval=${INTERVAL}&outputsize=${outputSize}&timezone=UTC&apikey=${apiKey}`;
  const resp = await fetch(url);
  const data = await resp.json();
  if (!data.values) {
    throw new Error(`TwelveData error for ${symbol}: ${data.message || JSON.stringify(data)}`);
  }
  // TwelveData returns most-recent-first; ascending chronological is what the
  // strategy expects.
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

async function fetchChartImage(env, symbol, side, levels) {
  const key = await resolveSecret(env.CHARTIMG_API_KEY);
  if (!key) return null;

  const body = {
    symbol: twelveDataSymbolToTradingView(symbol),
    interval: "5m",
    theme: "dark",
    width: 800,
    height: 500,
    timezone: "Etc/UTC",
    studies: [
      { name: "Moving Average", input: { length: 21 }, override: { "Plot.color": "rgb(255,255,255)" } },
      { name: "Moving Average", input: { length: 50 }, override: { "Plot.color": "rgb(0,255,0)" } },
      { name: "Moving Average", input: { length: 200 }, override: { "Plot.color": "rgb(255,0,0)" } },
      { name: "Relative Strength Index", input: { length: 14 } },
    ],
    drawings: [
      { name: "Horizontal Line", input: { price: levels.entry, text: "ENTRY" }, override: { lineColor: "rgb(255,255,255)" } },
      { name: "Horizontal Line", input: { price: levels.stop, text: "SL" }, override: { lineColor: "rgb(255,0,0)" } },
      { name: "Horizontal Line", input: { price: levels.target, text: "TP" }, override: { lineColor: "rgb(0,255,0)" } },
    ],
  };

  const resp = await fetch(CHART_IMG_ENDPOINT, {
    method: "POST",
    headers: { "x-api-key": key, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    throw new Error(`chart-img ${resp.status}: ${(await resp.text()).slice(0, 200)}`);
  }
  return await resp.arrayBuffer();
}

async function sendTelegramText(env, text) {
  const token = await resolveSecret(env.TELEGRAM_BOT_TOKEN);
  const chatId = await resolveSecret(env.TELEGRAM_CHAT_ID);
  const resp = await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: chatId, text, parse_mode: "Markdown" }),
  });
  if (!resp.ok) throw new Error(`Telegram sendMessage: ${await resp.text()}`);
}

async function sendTelegramPhoto(env, imageBuffer, caption) {
  const token = await resolveSecret(env.TELEGRAM_BOT_TOKEN);
  const chatId = await resolveSecret(env.TELEGRAM_CHAT_ID);
  const form = new FormData();
  form.append("chat_id", chatId);
  form.append("caption", caption);
  form.append("parse_mode", "Markdown");
  form.append("photo", new Blob([imageBuffer], { type: "image/png" }), "chart.png");
  const resp = await fetch(`https://api.telegram.org/bot${token}/sendPhoto`, { method: "POST", body: form });
  if (!resp.ok) throw new Error(`Telegram sendPhoto: ${await resp.text()}`);
}

function round(value, pip) {
  const decimals = pip === 0.01 ? 3 : 5;
  return Number(value).toFixed(decimals);
}

function buildMessage(pair, result) {
  const { side, pattern, levels, values } = result;
  const emoji = side === "BUY" ? "\u{1F7E2}" : "\u{1F534}";
  const slPips = Math.abs(levels.entry - levels.stop) / pair.pip;
  const tpPips = Math.abs(levels.target - levels.entry) / pair.pip;
  const ageMin = Math.round((Date.now() - parseUTC(values.time).getTime()) / 60000);

  return [
    `${emoji} *${side}* — ${pair.symbol}  \`5m\``,
    "",
    `*Entry* \`${round(levels.entry, pair.pip)}\``,
    `*SL*    \`${round(levels.stop, pair.pip)}\`  (${slPips.toFixed(1)} pips)`,
    `*TP*    \`${round(levels.target, pair.pip)}\`  (${tpPips.toFixed(1)} pips, 2:1)`,
    "",
    `Pattern: ${pattern}`,
    `RSI ${values.rsi.toFixed(1)} (vs SMMA ${values.rsiSmma.toFixed(1)})`,
    `ADX ${values.adx.toFixed(1)}  ·  ATR ${(values.atrRatio * 100).toFixed(0)}% of avg`,
    `Candle ${values.time} UTC (~${ageMin} min ago)`,
    "",
    `_Risk 1% of account. Stop is 2x the signal candle — size the position from the stop distance, not a fixed lot._`,
  ].join("\n");
}

async function checkPair(env, pair) {
  const apiKey = await resolveSecret(env.TWELVEDATA_API_KEY);
  const candles = await fetchCandles(pair.symbol, apiKey);
  if (candles.length < MIN_BARS) {
    return { symbol: pair.symbol, skipped: `not enough history (${candles.length}/${MIN_BARS})` };
  }

  const result = evaluateTMA(candles, { minDist: pair.minDist });
  if (!result.ok) return { symbol: pair.symbol, skipped: result.reason };

  if (!result.side) {
    return {
      symbol: pair.symbol,
      skipped: "no setup",
      blockedBy: firstBlockingGate(result.gates),
      direction: result.direction,
      adx: Number(result.values.adx.toFixed(1)),
      rsi: Number(result.values.rsi.toFixed(1)),
    };
  }

  const current = candles[candles.length - 1];

  // Dedupe on the candle, and separately enforce the strategy's own "one trade per
  // day maximum" rule per instrument - without it a trending session can fire
  // several times and the alerts stop matching the documented strategy.
  const dayKey = `${pair.symbol}:day:${current.time.slice(0, 10)}`;
  const candleKey = `${pair.symbol}:bar:${current.time}`;
  if (env.ALERT_STATE) {
    if (await env.ALERT_STATE.get(candleKey)) {
      return { symbol: pair.symbol, skipped: "already alerted this candle" };
    }
    if (await env.ALERT_STATE.get(dayKey)) {
      return { symbol: pair.symbol, skipped: "one trade per day already sent" };
    }
  }

  const caption = buildMessage(pair, result);
  let imageSent = false;
  let imageError = null;
  try {
    const image = await fetchChartImage(env, pair.symbol, result.side, result.levels);
    if (image) {
      await sendTelegramPhoto(env, image, caption);
      imageSent = true;
    }
  } catch (err) {
    imageError = String(err);
  }
  if (!imageSent) {
    // The signal is the product; the picture is a nicety. Never drop one for the other.
    await sendTelegramText(env, caption + (imageError ? "\n\n_(chart image unavailable)_" : ""));
  }

  if (env.ALERT_STATE) {
    await env.ALERT_STATE.put(candleKey, "1", { expirationTtl: 60 * 60 * 24 });
    await env.ALERT_STATE.put(dayKey, "1", { expirationTtl: 60 * 60 * 36 });
  }

  return { symbol: pair.symbol, alerted: result.side, pattern: result.pattern, imageSent, imageError };
}

async function runAllChecks(env, { force = false } = {}) {
  const nowIso = new Date().toISOString().replace("T", " ").slice(0, 19);
  if (!force && !inSession(nowIso)) {
    return { skipped: `outside session (${SESSION_START_HOUR}:00-${SESSION_END_HOUR}:00 UTC, Mon-Fri)`, now: `${nowIso} UTC` };
  }
  const results = [];
  for (const pair of PAIRS) {
    try {
      results.push(await checkPair(env, pair));
    } catch (err) {
      results.push({ symbol: pair.symbol, error: String(err) });
    }
  }
  return { now: `${nowIso} UTC`, results };
}

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(runAllChecks(env));
  },

  async fetch(request, env) {
    const url = new URL(request.url);
    const secret = url.searchParams.get("secret");
    const expectedSecret = await resolveSecret(env.WEBHOOK_SECRET);
    if (!expectedSecret || secret !== expectedSecret) {
      return new Response("Unauthorized", { status: 401 });
    }

    // GET /?secret=...&ping=1 - confirm Telegram delivery works at all.
    if (url.searchParams.get("ping") === "1") {
      await sendTelegramText(env, "✅ TMA Trend Scalper alert worker is alive.");
      return new Response("Sent test message", { status: 200 });
    }

    // GET /?secret=...&testchart=AUD/USD - confirm the chart-img key works and you
    // like how the picture looks, WITHOUT waiting for a real signal.
    const testChart = url.searchParams.get("testchart");
    if (testChart) {
      const pair = PAIRS.find((p) => p.symbol === testChart) || PAIRS[0];
      const apiKey = await resolveSecret(env.TWELVEDATA_API_KEY);
      const candles = await fetchCandles(pair.symbol, apiKey);
      const last = candles[candles.length - 1];
      const size = last.high - last.low;
      const levels = { entry: last.close, stop: last.close - size * 2, target: last.close + size * 4, candleSize: size };
      try {
        const image = await fetchChartImage(env, pair.symbol, "BUY", levels);
        if (!image) return new Response("CHARTIMG_API_KEY not set", { status: 400 });
        await sendTelegramPhoto(env, image, `🧪 *Test chart* — ${pair.symbol}\nNot a signal. Levels are illustrative.`);
        return new Response("Sent test chart", { status: 200 });
      } catch (err) {
        return new Response(`chart-img failed: ${err}`, { status: 502 });
      }
    }

    // GET /?secret=...&debug=1 - evaluate every pair right now, ignoring the session
    // gate, and report which condition is blocking each one. This is the endpoint to
    // compare against the Pine confluence table when verifying signals.
    const debug = url.searchParams.get("debug") === "1";
    const summary = await runAllChecks(env, { force: debug });
    return new Response(JSON.stringify(summary, null, 2), {
      headers: { "Content-Type": "application/json" },
    });
  },
};
