// Polls TwelveData for 5-minute forex candles, runs the TMA Trend Scalper (vFinal),
// and messages Telegram. Two passes per candle, on two Cloudflare Cron Triggers
// (see wrangler.toml):
//
//   EARLY  at :03 / :08 / :13 ...  - 2 minutes before the candle closes. Evaluates
//                                    the STILL-FORMING bar and, if it currently
//                                    qualifies, sends a "forming" heads-up so you
//                                    can be at the screen for the close.
//   CLOSE  at :00 / :05 / :10 ...  - evaluates the bar that just CLOSED, but only
//                                    for pairs the early pass flagged. Tells you
//                                    the signal either confirmed or evaporated.
//
// THE HONEST CAVEAT, and why the two messages are worded so differently: mid-candle
// every input is provisional. The bar's own high/low can still move, which changes
// the stop distance; RSI, ADX and the pattern can all flip. A forming signal is a
// "be ready", never an entry. Only the CLOSE pass reflects the strategy's real rule,
// which is evaluated on a closed bar.
//
// WHAT REPLACED WHAT: this used to alert on the ORIGINAL, simpler setup - a 21/50/200
// stack + pattern + RSI>50 over an 08:00-02:30 London window, with no ADX gate, no ATR
// volatility gate, no momentum check, no minimum SMMA separation, and RSI compared to
// 50 rather than to its own SMMA(50). Those alerts fired on setups the current
// strategy rejects. That logic has since been deleted outright rather than left beside
// this one, so there is no second file in the repo that could be mistaken for the
// rules the bot actually runs.

import {
  MIN_BARS, SESSION_END_HOUR, SESSION_START_HOUR,
  evaluateTMA, firstBlockingGate, inSession, minutesToClose, parseUTC, rmaSeries, splitCandles,
} from "./tma-strategy.js";

// TwelveData free tier: 800 requests/day, 8/minute.
//
// BUDGET, with the early pass running every 5 minutes across an 8-hour session:
//   8h x 12 fires/h = 96 early fires/day, x 6 pairs = 576 requests/day.
//   The CLOSE pass only spends a call for a pair the early pass actually flagged -
//   a handful a day - so the realistic total is ~600/day against the 800 cap.
//
// Six pairs rather than eight is deliberate. Eight comes to ~793/day, which fits on
// paper but leaves seven spare calls: a single manual ?debug=1 (one call per pair)
// would tip it over the cap and the alerts would then fail silently for the rest of
// the day. Six leaves ~200 spare. The per-minute limit caps this at 8 regardless,
// since one fire calls every pair back to back.
//
// `minDist` is the strategy's own per-pair SMMA separation. It is an ABSOLUTE price
// distance, so JPY pairs need a different number, not a scaled one.
const PAIRS = [
  { symbol: "AUD/USD", pip: 0.0001, minDist: 0.001 },
  { symbol: "EUR/USD", pip: 0.0001, minDist: 0.001 },
  { symbol: "GBP/USD", pip: 0.0001, minDist: 0.001 },
  { symbol: "NZD/USD", pip: 0.0001, minDist: 0.001 },
  { symbol: "USD/CAD", pip: 0.0001, minDist: 0.001 },
  { symbol: "USD/JPY", pip: 0.01, minDist: 0.10 },
];

const INTERVAL = "5min";
const OUTPUT_SIZE = 400; // > MIN_BARS (200 SMMA + 50-period RSI SMMA + slack)

// Which cron fired. wrangler.toml registers the early pass first.
const CRON_EARLY = "3,8,13,18,23,28,33,38,43,48,53,58 * * * *";

// QuickChart renders the chart server-side. No account and no API key, which is the
// whole reason it is here rather than chart-img - see fetchChartImage below. If the
// call fails for any reason the alert still goes out as text; a missing picture must
// never cost you the signal.
const QUICKCHART_ENDPOINT = "https://quickchart.io/chart";

async function resolveSecret(binding) {
  if (binding == null) return undefined;
  if (typeof binding === "string") return binding;
  if (typeof binding.get === "function") return await binding.get();
  return undefined;
}

const tvSymbol = (symbol) => `FX:${symbol.replace("/", "")}`;

async function fetchCandles(symbol, apiKey, env, outputSize = OUTPUT_SIZE) {
  const url = `https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(symbol)}&interval=${INTERVAL}&outputsize=${outputSize}&timezone=UTC&apikey=${apiKey}`;
  const resp = await fetch(url);
  const data = await resp.json();
  if (!data.values) {
    throw new Error(`TwelveData error for ${symbol}: ${data.message || JSON.stringify(data)}`);
  }
  await countCall(env);
  return data.values
    .map((v) => ({
      time: v.datetime,
      open: Number(v.open),
      high: Number(v.high),
      low: Number(v.low),
      close: Number(v.close),
    }))
    .reverse(); // TwelveData returns most-recent-first
}

// Rolling count of API calls per UTC day, so the debug endpoint can show real usage
// against the cap instead of leaving you to trust the arithmetic in the comment above.
async function countCall(env) {
  if (!env.ALERT_STATE) return;
  const key = `quota:${new Date().toISOString().slice(0, 10)}`;
  const current = Number((await env.ALERT_STATE.get(key)) || 0);
  await env.ALERT_STATE.put(key, String(current + 1), { expirationTtl: 60 * 60 * 48 });
}

async function quotaUsed(env) {
  if (!env.ALERT_STATE) return null;
  const key = `quota:${new Date().toISOString().slice(0, 10)}`;
  return Number((await env.ALERT_STATE.get(key)) || 0);
}

// How many bars of context the chart shows. Enough to read the swing that produced
// the setup without shrinking the signal candle to a hairline.
const CHART_BARS = 60;

function chartConfig(symbol, candles, levels, side, { candlestick }) {
  const window = candles.slice(-CHART_BARS);
  const closes = candles.map((c) => c.close);
  const smmaAt = (len) => {
    const out = rmaSeries(closes, len, 0);
    return out.slice(-CHART_BARS);
  };
  const labels = window.map((c) => c.time.slice(11, 16));
  const flat = (value) => window.map(() => value);

  const priceSeries = candlestick
    ? {
        type: "candlestick",
        label: symbol,
        data: window.map((c, i) => ({ x: i, o: c.open, h: c.high, l: c.low, c: c.close })),
        color: { up: "#00e6a0", down: "#ff4d6a", unchanged: "#9aa7b8" },
      }
    : {
        type: "line",
        label: symbol,
        data: window.map((c) => c.close),
        borderColor: "#e6edf3",
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
      };

  const line = (label, data, color, width = 1.5, dash = undefined) => ({
    type: "line", label, data, borderColor: color, borderWidth: width,
    pointRadius: 0, fill: false, borderDash: dash,
  });

  return {
    type: candlestick ? "candlestick" : "line",
    data: {
      labels,
      datasets: [
        priceSeries,
        line("SMMA 21", smmaAt(21), "#ffffff"),
        line("SMMA 50", smmaAt(50), "#00e6a0"),
        line("SMMA 200", smmaAt(200), "#ff4d6a"),
        line(`Entry ${levels.entry}`, flat(levels.entry), "#00c2ff", 1.5, [6, 4]),
        line("SL", flat(levels.stop), "#ff4d6a", 1.5, [4, 4]),
        line("TP", flat(levels.target), "#00e6a0", 1.5, [4, 4]),
      ],
    },
    options: {
      plugins: {
        title: { display: true, text: `${symbol} 5m — ${side}`, color: "#e6edf3", font: { size: 16 } },
        legend: { labels: { color: "#9aa7b8", boxWidth: 12, font: { size: 10 } } },
      },
      scales: {
        x: { ticks: { color: "#5c6b7f", maxTicksLimit: 8, font: { size: 9 } }, grid: { color: "rgba(255,255,255,0.06)" } },
        y: { ticks: { color: "#5c6b7f", font: { size: 9 } }, grid: { color: "rgba(255,255,255,0.06)" }, position: "right" },
      },
    },
  };
}

/**
 * Renders the signal on a chart via QuickChart.
 *
 * WHY NOT chart-img: it would have produced a real TradingView-style image, but its
 * only sign-in path is Google OAuth, which loops on iOS Safari - and the account here
 * is driven from a phone. An image source you cannot get a key for is not an image
 * source. QuickChart needs no account and no key at all.
 *
 * POST rather than a GET URL: the config carries 60 bars of OHLC plus four overlay
 * series, which blows past practical URL length limits. Posting returns the PNG bytes
 * directly, which is also the same shape the Telegram upload already wanted.
 *
 * CANDLESTICK WITH A LINE FALLBACK: candlesticks come from a Chart.js plugin whose
 * availability is not guaranteed. If that render fails, this retries once as a plain
 * line of closes rather than dropping the picture - the moving-average stack and the
 * entry/SL/TP levels, which are most of the value, survive either way.
 */
async function fetchChartImage(env, symbol, candles, levels, side) {
  const render = async (candlestick) => {
    const resp = await fetch(QUICKCHART_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chart: chartConfig(symbol, candles, levels, side, { candlestick }),
        width: 900,
        height: 500,
        format: "png",
        backgroundColor: "#0a0e17",
        version: "4",
      }),
    });
    if (!resp.ok) throw new Error(`quickchart ${resp.status}: ${(await resp.text()).slice(0, 200)}`);
    return await resp.arrayBuffer();
  };

  try {
    return await render(true);
  } catch (err) {
    return await render(false);
  }
}

async function sendText(env, text) {
  const token = await resolveSecret(env.TELEGRAM_BOT_TOKEN);
  const chatId = await resolveSecret(env.TELEGRAM_CHAT_ID);
  const resp = await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: chatId, text, parse_mode: "Markdown" }),
  });
  if (!resp.ok) throw new Error(`Telegram sendMessage: ${await resp.text()}`);
}

async function sendPhoto(env, imageBuffer, caption) {
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

/** Sends the picture when it is available, and the words regardless. */
async function deliver(env, caption, levels, symbol, candles, side) {
  try {
    const image = await fetchChartImage(env, symbol, candles, levels, side);
    if (image) {
      await sendPhoto(env, image, caption);
      return { imageSent: true };
    }
  } catch (err) {
    await sendText(env, `${caption}\n\n_(chart image unavailable)_`);
    return { imageSent: false, imageError: String(err) };
  }
  await sendText(env, caption);
  return { imageSent: false };
}

const round = (v, pip) => Number(v).toFixed(pip === 0.01 ? 3 : 5);

function levelLines(pair, levels) {
  const slPips = Math.abs(levels.entry - levels.stop) / pair.pip;
  const tpPips = Math.abs(levels.target - levels.entry) / pair.pip;
  return [
    `*Entry* \`${round(levels.entry, pair.pip)}\``,
    `*SL*    \`${round(levels.stop, pair.pip)}\`  (${slPips.toFixed(1)} pips)`,
    `*TP*    \`${round(levels.target, pair.pip)}\`  (${tpPips.toFixed(1)} pips, 2:1)`,
  ];
}

function formingMessage(pair, result, minsLeft) {
  const arrow = result.side === "BUY" ? "\u{1F7E2}" : "\u{1F534}";
  return [
    `\u{23F3} *FORMING — ${result.side}* ${arrow} ${pair.symbol}`,
    `*${minsLeft.toFixed(1)} min to close.* Get ready — do not enter yet.`,
    "",
    ...levelLines(pair, result.levels),
    "",
    `Pattern: ${result.pattern}`,
    `RSI ${result.values.rsi.toFixed(1)} (vs SMMA ${result.values.rsiSmma.toFixed(1)})  ·  ADX ${result.values.adx.toFixed(1)}`,
    "",
    `_Provisional. The bar's high/low can still move, which changes the stop, and RSI/ADX/the pattern can all flip before it closes. You'll get a confirm or a cancel at the close._`,
  ].join("\n");
}

function confirmedMessage(pair, result) {
  const arrow = result.side === "BUY" ? "\u{1F7E2}" : "\u{1F534}";
  const ageMin = Math.round((Date.now() - parseUTC(result.values.time).getTime()) / 60000);
  return [
    `\u{2705} *CONFIRMED — ${result.side}* ${arrow} ${pair.symbol}  \`5m closed\``,
    "",
    ...levelLines(pair, result.levels),
    "",
    `Pattern: ${result.pattern}`,
    `RSI ${result.values.rsi.toFixed(1)} (vs SMMA ${result.values.rsiSmma.toFixed(1)})`,
    `ADX ${result.values.adx.toFixed(1)}  ·  ATR ${(result.values.atrRatio * 100).toFixed(0)}% of avg`,
    `Candle ${result.values.time} UTC (~${ageMin} min ago)`,
    "",
    `_Risk 1%. Stop is 2x the signal candle — size from the stop distance, not a fixed lot._`,
  ].join("\n");
}

function cancelledMessage(pair, flagged, result) {
  const blocked = result?.gates ? firstBlockingGate(result.gates) : "unknown";
  return [
    `\u{274C} *CANCELLED — ${flagged.side}* ${pair.symbol}`,
    `The setup did not survive the close. Do not enter.`,
    "",
    `Failed on: *${blocked}*`,
  ].join("\n");
}

const dayStamp = (t) => t.slice(0, 10);

// ---------------------------------------------------------------------------
// PASS 1 - early warning on the forming bar
// ---------------------------------------------------------------------------

async function earlyPass(env) {
  const apiKey = await resolveSecret(env.TWELVEDATA_API_KEY);
  const results = [];
  for (const pair of PAIRS) {
    try {
      const candles = await fetchCandles(pair.symbol, apiKey, env);
      const { forming } = splitCandles(candles);
      if (!forming) {
        results.push({ symbol: pair.symbol, skipped: "no forming bar yet" });
        continue;
      }
      // Evaluate the series INCLUDING the forming bar - that provisional bar is
      // exactly what we want to know about here.
      const result = evaluateTMA(candles, { minDist: pair.minDist });
      if (!result.ok) { results.push({ symbol: pair.symbol, skipped: result.reason }); continue; }
      if (!result.side) {
        results.push({ symbol: pair.symbol, skipped: "no setup", blockedBy: firstBlockingGate(result.gates) });
        continue;
      }

      const dayKey = `${pair.symbol}:day:${dayStamp(forming.time)}`;
      const warnKey = `${pair.symbol}:warned:${forming.time}`;
      if (env.ALERT_STATE) {
        if (await env.ALERT_STATE.get(dayKey)) {
          results.push({ symbol: pair.symbol, skipped: "one trade per day already sent" });
          continue;
        }
        if (await env.ALERT_STATE.get(warnKey)) {
          results.push({ symbol: pair.symbol, skipped: "already warned on this bar" });
          continue;
        }
      }

      const minsLeft = minutesToClose(forming);
      const delivery = await deliver(env, formingMessage(pair, result, minsLeft), result.levels, pair.symbol, candles, result.side);

      if (env.ALERT_STATE) {
        await env.ALERT_STATE.put(warnKey, "1", { expirationTtl: 60 * 30 });
        // Hand the CLOSE pass a note of what to re-check, so it only spends an API
        // call on pairs that actually flagged.
        await env.ALERT_STATE.put(
          `pending:${pair.symbol}`,
          JSON.stringify({ side: result.side, bar: forming.time }),
          { expirationTtl: 60 * 20 });
      }
      results.push({ symbol: pair.symbol, warned: result.side, minsLeft: Number(minsLeft.toFixed(1)), ...delivery });
    } catch (err) {
      results.push({ symbol: pair.symbol, error: String(err) });
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// PASS 2 - confirm or cancel, only for pairs the early pass flagged
// ---------------------------------------------------------------------------

async function closePass(env) {
  if (!env.ALERT_STATE) return [{ skipped: "no KV binding, cannot track pending warnings" }];
  const apiKey = await resolveSecret(env.TWELVEDATA_API_KEY);
  const results = [];

  for (const pair of PAIRS) {
    const raw = await env.ALERT_STATE.get(`pending:${pair.symbol}`);
    if (!raw) continue; // nothing flagged - costs no API call, which is the point
    const flagged = JSON.parse(raw);
    await env.ALERT_STATE.delete(`pending:${pair.symbol}`);

    try {
      const candles = await fetchCandles(pair.symbol, apiKey, env);
      // Score only bars that have actually closed. Re-reading the forming bar here
      // would report a provisional setup as confirmed, which is the exact failure
      // this two-pass split exists to avoid.
      const { closed } = splitCandles(candles);
      const last = closed[closed.length - 1];
      if (!last || last.time !== flagged.bar) {
        results.push({ symbol: pair.symbol, skipped: `expected closed bar ${flagged.bar}, newest closed is ${last?.time}` });
        continue;
      }

      const result = evaluateTMA(closed, { minDist: pair.minDist });
      if (result.ok && result.side === flagged.side) {
        const delivery = await deliver(env, confirmedMessage(pair, result), result.levels, pair.symbol, closed, result.side);
        await env.ALERT_STATE.put(`${pair.symbol}:day:${dayStamp(last.time)}`, "1", { expirationTtl: 60 * 60 * 36 });
        results.push({ symbol: pair.symbol, confirmed: result.side, ...delivery });
      } else {
        await sendText(env, cancelledMessage(pair, flagged, result));
        results.push({ symbol: pair.symbol, cancelled: flagged.side, blockedBy: result.ok ? firstBlockingGate(result.gates) : result.reason });
      }
    } catch (err) {
      results.push({ symbol: pair.symbol, error: String(err) });
    }
  }
  return results;
}

async function runPass(env, which, { force = false } = {}) {
  const nowIso = new Date().toISOString().replace("T", " ").slice(0, 19);
  if (!force && !inSession(nowIso)) {
    return { pass: which, skipped: `outside session (${SESSION_START_HOUR}:00-${SESSION_END_HOUR}:00 UTC, Mon-Fri)`, now: `${nowIso} UTC` };
  }
  const results = which === "early" ? await earlyPass(env) : await closePass(env);
  return { pass: which, now: `${nowIso} UTC`, quotaUsedToday: await quotaUsed(env), results };
}

export default {
  async scheduled(event, env, ctx) {
    const which = event.cron === CRON_EARLY ? "early" : "close";
    ctx.waitUntil(runPass(env, which));
  },

  async fetch(request, env) {
    const url = new URL(request.url);
    const secret = url.searchParams.get("secret");
    const expected = await resolveSecret(env.WEBHOOK_SECRET);
    if (!expected || secret !== expected) return new Response("Unauthorized", { status: 401 });

    if (url.searchParams.get("ping") === "1") {
      await sendText(env, "✅ TMA Trend Scalper alert worker is alive.");
      return new Response("Sent test message", { status: 200 });
    }

    // See the chart image without waiting for a signal.
    const testChart = url.searchParams.get("testchart");
    if (testChart) {
      const pair = PAIRS.find((p) => p.symbol === testChart) || PAIRS[0];
      const apiKey = await resolveSecret(env.TWELVEDATA_API_KEY);
      const candles = await fetchCandles(pair.symbol, apiKey, env);
      const last = candles[candles.length - 1];
      const size = last.high - last.low;
      const levels = { entry: last.close, stop: last.close - size * 2, target: last.close + size * 4 };
      try {
        const image = await fetchChartImage(env, pair.symbol, candles, levels, "BUY");
        await sendPhoto(env, image, `🧪 *Test chart* — ${pair.symbol}\nNot a signal. Levels are illustrative.`);
        return new Response("Sent test chart", { status: 200 });
      } catch (err) {
        return new Response(`chart render failed: ${err}`, { status: 502 });
      }
    }

    // ?debug=1 - evaluate every pair now, ignoring the session gate, reporting the
    // first failing gate per pair. Gate names match the confluence-table rows in
    // pine/tma-trend-scalper.pine, so the bot and the chart can be compared directly.
    // Costs one API call per pair, which is why the quota counter is reported back.
    if (url.searchParams.get("debug") === "1") {
      const apiKey = await resolveSecret(env.TWELVEDATA_API_KEY);
      const rows = [];
      for (const pair of PAIRS) {
        try {
          const candles = await fetchCandles(pair.symbol, apiKey, env);
          const { closed, forming } = splitCandles(candles);
          const onClosed = evaluateTMA(closed, { minDist: pair.minDist });
          const onForming = forming ? evaluateTMA(candles, { minDist: pair.minDist }) : null;
          // DATA FRESHNESS, measured rather than assumed. Being early is the entire
          // point of the FORMING alert, so the two numbers that decide whether it can
          // work at all are reported here instead of taken on trust:
          //   formingBar null  -> the feed only publishes bars after they close, so
          //                       there is nothing to warn about early and the early
          //                       pass can never fire. Not a bug in the strategy.
          //   feedLagSeconds   -> how stale the newest bar is versus the wall clock.
          //                       A lag near or above the 2-minute head start means
          //                       the "early" warning is not actually early.
          const newest = candles.at(-1);
          const feedLagSeconds = newest
            ? Math.round((Date.now() - parseUTC(newest.time).getTime()) / 1000)
            : null;
          rows.push({
            symbol: pair.symbol,
            closedBar: closed.at(-1)?.time,
            formingBar: forming?.time ?? null,
            minsToClose: forming ? Number(minutesToClose(forming).toFixed(1)) : null,
            newestBar: newest?.time ?? null,
            feedLagSeconds,
            closed: onClosed.ok ? { side: onClosed.side, blockedBy: firstBlockingGate(onClosed.gates), gates: onClosed.gates, adx: Number(onClosed.values.adx.toFixed(1)), rsi: Number(onClosed.values.rsi.toFixed(1)), rsiSmma: Number(onClosed.values.rsiSmma.toFixed(1)) } : onClosed.reason,
            forming: onForming?.ok ? { side: onForming.side, blockedBy: firstBlockingGate(onForming.gates) } : null,
          });
        } catch (err) {
          rows.push({ symbol: pair.symbol, error: String(err) });
        }
      }
      return Response.json({
        now: new Date().toISOString(),
        inSession: inSession(new Date().toISOString().replace("T", " ").slice(0, 19)),
        quotaUsedToday: await quotaUsed(env),
        quotaCapPerDay: 800,
        pairs: rows,
      });
    }

    // Manually drive either pass: ?pass=early or ?pass=close
    const which = url.searchParams.get("pass") === "close" ? "close" : "early";
    const summary = await runPass(env, which, { force: url.searchParams.get("force") === "1" });
    return Response.json(summary);
  },
};
