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
  LONDON_END, LONDON_START, MIN_BARS, SESSION_END_HOUR, SESSION_START_HOUR,
  evaluateTMA, firstBlockingGate, inSession, minutesToClose, parseUTC, rmaSeries, splitCandles,
} from "./tma-strategy.js";

// TwelveData free tier: 800 requests/day, 8/minute.
//
// BUDGET. The session gate is currently WIDE OPEN (all hours, see tma-strategy.js),
// which triples what the 07:00-15:00 window cost and forces the pair count down:
//
//   24h x 12 fires/h = 288 early fires/day
//     x 2 pairs =   576/day  fits, ~200 spare
//     x 3 pairs =   864/day  OVER - goes silent partway through the day
//     x 6 pairs = 1,728/day  OVER by more than the entire cap
//
// So all-hours and six pairs cannot both hold. Hours were the explicit ask, so the
// pair list pays for it: AUD/USD (the strategy's own recommended pair) and EUR/USD.
// Restoring the 07:00-15:00 window frees the budget for six again.
//
// Going silent is the failure that matters here - the quota runs out mid-morning and
// the bot simply stops alerting, with nothing to say it has. ?debug=1 reports
// quotaUsedToday against the cap so that state is visible rather than inferred.
//
// `minDist` is the strategy's own per-pair SMMA separation. It is an ABSOLUTE price
// distance, so JPY pairs need a different number, not a scaled one.
const PAIRS = [
  { symbol: "AUD/USD", pip: 0.0001, minDist: 0.001 },
  { symbol: "EUR/USD", pip: 0.0001, minDist: 0.001 },
];

const INTERVAL = "5min";
const OUTPUT_SIZE = 400; // > MIN_BARS (200 SMMA + 50-period RSI SMMA + slack)

// Which cron fired. wrangler.toml registers the early pass first.
const CRON_EARLY = "3,8,13,18,23,28,33,38,43,48,53,58 * * * *";

// Bumped whenever the deployed behaviour changes, and reported by /?health=1. Without
// a marker like this there is no way to tell a Worker running new code from one still
// serving a stale deployment - the dashboard shows a version hash that means nothing
// against a git commit.
const BUILD = "tma-vfinal-2026-08-04-two-pass";

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

/**
 * X AXIS, and the trap that cost a render.
 *
 * A candlestick dataset reads `x` as a TIMESTAMP IN MILLISECONDS, not as a position.
 * The first version passed x: 0,1,2..., so every candle was plotted on 1 Jan 1970 -
 * the axis said "1970" - and the overlay lines, which were index-aligned arrays
 * against `labels`, had no valid x at all on that scale and were silently dropped.
 * They showed in the legend and never appeared on the chart.
 *
 * So the two chart types need genuinely different shapes, not one shape with a flag:
 *   candlestick -> every dataset is {x: epochMs, ...} on a time scale
 *   line        -> plain arrays aligned to `labels` on the default category scale
 * Trying to share one shape across both is what produced a chart with no indicators.
 */
function chartConfig(symbol, candles, levels, side, { candlestick }) {
  const window = candles.slice(-CHART_BARS);
  const closes = candles.map((c) => c.close);
  const smmaAt = (len) => rmaSeries(closes, len, 0).slice(-CHART_BARS);
  const at = window.map((c) => parseUTC(c.time).getTime());

  const overlays = [
    { label: "SMMA 21", values: smmaAt(21), color: "#ffffff", width: 1.5 },
    { label: "SMMA 50", values: smmaAt(50), color: "#00e6a0", width: 1.5 },
    { label: "SMMA 200", values: smmaAt(200), color: "#ff4d6a", width: 1.5 },
    { label: `Entry ${levels.entry}`, values: window.map(() => levels.entry), color: "#00c2ff", width: 1.5, dash: [6, 4] },
    { label: "SL", values: window.map(() => levels.stop), color: "#ff4d6a", width: 1.5, dash: [4, 4] },
    { label: "TP", values: window.map(() => levels.target), color: "#00e6a0", width: 1.5, dash: [4, 4] },
  ];

  // THE SIGNAL MARKER. A horizontal entry line gives the price but not the bar, and
  // on 60 candles "which one actually fired" is the first thing you look for. This is
  // a single triangle sitting on the signal candle - up under a BUY, down over a
  // SELL, matching the arrows the Pine indicator plots.
  const signalIdx = window.length - 1;
  const marker = (pointed) => ({
    type: "line",
    label: `${side} signal`,
    data: pointed
      ? [{ x: at[signalIdx], y: levels.entry }]
      : window.map((_, i) => (i === signalIdx ? levels.entry : null)),
    borderColor: "rgba(0,0,0,0)",
    backgroundColor: side === "BUY" ? "#00e6a0" : "#ff4d6a",
    pointStyle: side === "BUY" ? "triangle" : "triangle",
    pointRadius: 9,
    pointRotation: side === "BUY" ? 0 : 180,
    pointBorderColor: "#0a0e17",
    pointBorderWidth: 2,
    showLine: false,
    fill: false,
  });

  const asLine = (o, pointed) => ({
    type: "line",
    label: o.label,
    data: pointed ? o.values.map((v, i) => ({ x: at[i], y: v })) : o.values,
    borderColor: o.color,
    borderWidth: o.width,
    borderDash: o.dash,
    pointRadius: 0,
    fill: false,
    spanGaps: true,
  });


  const common = {
    plugins: {
      title: { display: true, text: `${symbol} 5m — ${side}`, color: "#e6edf3", font: { size: 16 } },
      legend: { labels: { color: "#9aa7b8", boxWidth: 12, font: { size: 10 } } },
    },
    scales: {
      y: { ticks: { color: "#5c6b7f", font: { size: 9 } }, grid: { color: "rgba(255,255,255,0.06)" }, position: "right" },
    },
  };

  if (candlestick) {
    return {
      type: "candlestick",
      data: {
        datasets: [
          {
            type: "candlestick",
            label: symbol,
            data: window.map((c, i) => ({ x: at[i], o: c.open, h: c.high, l: c.low, c: c.close })),
            color: { up: "#00e6a0", down: "#ff4d6a", unchanged: "#9aa7b8" },
            borderColor: { up: "#00e6a0", down: "#ff4d6a", unchanged: "#9aa7b8" },
          },
          ...overlays.map((o) => asLine(o, true)),
          marker(true),
        ],
      },
      options: {
        ...common,
        scales: {
          ...common.scales,
          x: {
            type: "time",
            time: { unit: "minute", stepSize: 15, displayFormats: { minute: "HH:mm" } },
            ticks: { color: "#5c6b7f", maxTicksLimit: 8, font: { size: 9 } },
            grid: { color: "rgba(255,255,255,0.06)" },
          },
        },
      },
    };
  }

  return {
    type: "line",
    data: {
      labels: window.map((c) => c.time.slice(11, 16)),
      datasets: [
        { type: "line", label: symbol, data: window.map((c) => c.close), borderColor: "#e6edf3", borderWidth: 2, pointRadius: 0, fill: false },
        ...overlays.map((o) => asLine(o, false)),
        marker(false),
      ],
    },
    options: {
      ...common,
      scales: {
        ...common.scales,
        x: { ticks: { color: "#5c6b7f", maxTicksLimit: 8, font: { size: 9 } }, grid: { color: "rgba(255,255,255,0.06)" } },
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

const outsideLondonNote = (result) => (result.inLondonSession ? [] : [
  "",
  `\u{26A0}\uFE0F _Outside ${LONDON_START}:00-${LONDON_END}:00 London. The strategy would NOT take this - session filter is open for testing._`,
]);

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
    ...outsideLondonNote(result),
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
    ...outsideLondonNote(result),
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

    // UNAUTHENTICATED HEALTH CHECK, and why it has to be unauthenticated: every other
    // endpoint needs WEBHOOK_SECRET, so when that binding goes missing there is no way
    // in to find out that it is missing. The first Git deploy did exactly that - it
    // stripped the Secrets Store bindings, and the only symptom available was a bare
    // "Unauthorized", indistinguishable from mistyping the password.
    //
    // It reports which bindings EXIST and what code is running. Never any value, never
    // any market data - a binding name and a boolean give an attacker nothing they
    // could not guess from the public repo.
    if (url.searchParams.get("health") === "1") {
      return Response.json({
        build: BUILD,
        bindings: {
          WEBHOOK_SECRET: Boolean(await resolveSecret(env.WEBHOOK_SECRET)),
          TELEGRAM_BOT_TOKEN: Boolean(await resolveSecret(env.TELEGRAM_BOT_TOKEN)),
          TELEGRAM_CHAT_ID: Boolean(await resolveSecret(env.TELEGRAM_CHAT_ID)),
          TWELVEDATA_API_KEY: Boolean(await resolveSecret(env.TWELVEDATA_API_KEY)),
          ALERT_STATE_KV: Boolean(env.ALERT_STATE),
        },
        pairs: PAIRS.map((p) => p.symbol),
        session: `${SESSION_START_HOUR}:00-${SESSION_END_HOUR}:00 Europe/London, Mon-Fri`,
      });
    }

    const secret = url.searchParams.get("secret");
    const expected = await resolveSecret(env.WEBHOOK_SECRET);
    // Two different failures, two different fixes. Collapsing them into one message
    // sent us chasing a wrong password when the binding was simply gone.
    if (!expected) {
      return new Response(
        "WEBHOOK_SECRET is not bound to this Worker - nothing to check the URL against.\n" +
        "This is a deploy/config problem, not a wrong password.\n" +
        "Try /?health=1 to see which bindings are missing.", { status: 503 });
    }
    if (secret !== expected) {
      return new Response("Unauthorized - WEBHOOK_SECRET is bound, but the ?secret= in this URL does not match it.", { status: 401 });
    }

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
