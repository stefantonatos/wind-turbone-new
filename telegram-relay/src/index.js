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
// WHICH RULES: the BASE strategy - 21/50/200 SMMA stack, close on the right side of
// the 200, 3-Line Strike or Engulfing, RSI above 50 and above its own SMMA(50). The
// chop filters (ADX, ATR, momentum, SMMA separation) are off; see EXTRA_FILTERS in
// tma-strategy.js. pine/tma-trend-scalper.pine draws exactly this, so the chart and
// the alerts should agree bar for bar. If either is changed, change both.

import {
  BAR_MINUTES, LONDON_END, LONDON_START, MIN_BARS, SESSION_END_HOUR, SESSION_START_HOUR,
  evaluateTMA, firstBlockingGate, inSession, minutesToClose, parseUTC, rmaSeries, splitCandles,
} from "./tma-strategy.js";

// TwelveData free tier: 800 requests/day, 8/minute.
//
// BUDGET, recomputed for two reasons at once:
//
//   1. Session is back to 07:00-15:00 London (8h/weekday), not all-hours - a 3x cut.
//   2. closePass no longer skips pairs the early pass didn't flag (see closePass for
//      why that was a real bug, not a saving) - so it now costs the SAME as the early
//      pass, not "free most of the time" the way the old budget comment assumed.
//
//   8h x 12 fires/h = 96 fires/day, PER PASS, and there are two passes:
//     96 x 2 passes x pairs = 192 x pairs calls/day (weekdays; weekends fetch nothing)
//
//     x 3 pairs =   576/day  fits, ~220 spare
//     x 4 pairs =   768/day  fits, ~30 spare - tight, but London-only is a hard
//                            floor under the cap either way, unlike all-hours x 3+.
//     x 6 pairs = 1,152/day  OVER even under London-only, now that close pass is not
//                            free - six needs either fewer fires/bar or a paid tier.
//
// Four pairs, chosen for London-session liquidity: the original two plus GBP/USD (the
// London session's own currency) and NZD/USD. ?debug=1 reports quotaUsedToday against
// the cap - going silent mid-session is the failure that matters, so that number is
// worth a glance if alerts feel like they've gone quiet.
//
// `minDist` is the strategy's own per-pair SMMA separation. It is an ABSOLUTE price
// distance, so JPY pairs need a different number, not a scaled one.
//
// `spreadPips` is a typical retail round-trip spread, used ONLY to warn when a stop
// is too tight to be real. It is not a cost model - the backtester has one of those.
const PAIRS = [
  { symbol: "AUD/USD", pip: 0.0001, minDist: 0.001, spreadPips: 1.2 },
  { symbol: "EUR/USD", pip: 0.0001, minDist: 0.001, spreadPips: 0.8 },
  { symbol: "GBP/USD", pip: 0.0001, minDist: 0.001, spreadPips: 1.0 },
  { symbol: "NZD/USD", pip: 0.0001, minDist: 0.001, spreadPips: 1.8 },
];

// SIZING GUARD RAILS, added after a live alert asked for 9.60 lots on a 1.4 pip stop.
//
// The stop is 2x the signal candle's own range. On a bar that is three minutes old
// and has moved 0.7 pips, that is a 1.4 pip stop - and since lots = risk / stop, a
// stop approaching zero sends the size to infinity. The £99.97 of risk was correct;
// the position needed to carry it was 960,000 AUD, about 67:1 on a £10,000 account.
// Two things were wrong with it and neither was the arithmetic:
//
//   1. No broker would accept it. UK retail leverage on major FX is capped at 30:1.
//   2. A 1.4 pip stop sits INSIDE the ~1.2 pip spread. Entry alone puts the trade
//      most of the way to its stop, so it is not a stop, it is a coin flip on the
//      first tick.
//
// So the size is capped at MAX_LEVERAGE and the alert says when the cap bit, and a
// stop under MIN_STOP_SPREAD_MULT x the spread is called out as untradeable rather
// than quietly sized up. The strategy's own stop rule is NOT changed - that belongs
// in the backtest, not in a patch to the alerter.
const MAX_LEVERAGE = 30;
const MIN_STOP_SPREAD_MULT = 3;

// The base strategy has no daily cap - that was one of the added rules, so it comes
// off with the rest of them. Expect more than one alert per pair per day now; that is
// the intended effect, not a dedupe bug. The per-BAR dedupe (`warned:` keys) stays
// either way, so a single bar still cannot alert twice.
const ONE_TRADE_PER_DAY = false;

const INTERVAL = "5min";
const OUTPUT_SIZE = 400; // > MIN_BARS (200 SMMA + 50-period RSI SMMA + slack)

// Which cron fired. wrangler.toml registers the early pass first.
const CRON_EARLY = "3,8,13,18,23,28,33,38,43,48,53,58 * * * *";

// Bumped whenever the deployed behaviour changes, and reported by /?health=1. Without
// a marker like this there is no way to tell a Worker running new code from one still
// serving a stale deployment - the dashboard shows a version hash that means nothing
// against a git commit.
const BUILD = "tma-base-2026-08-04-close-pass-fix";

// QuickChart renders the chart server-side. No account and no API key, which is the
// whole reason it is here rather than chart-img - see fetchChartImage below. If the
// call fails for any reason the alert still goes out as text; a missing picture must
// never cost you the signal.
const QUICKCHART_ENDPOINT = "https://quickchart.io/chart";

// POSITION SIZING. Overridable from the Cloudflare dashboard as plain text variables
// (Settings -> Variables and Secrets, type Text) so the balance keeps up with the
// account without a code change.
const DEFAULT_ACCOUNT_BALANCE = 10000;
const DEFAULT_ACCOUNT_CURRENCY = "GBP";
const DEFAULT_RISK_PCT = 1.0;
const MIN_LOT = 0.01;   // smallest size most brokers accept
const LOT_UNITS = 100000;

function sizingConfig(env) {
  const num = (v, fallback) => {
    const n = Number(v);
    return Number.isFinite(n) && n > 0 ? n : fallback;
  };
  const currency = (env.ACCOUNT_CURRENCY || DEFAULT_ACCOUNT_CURRENCY).toUpperCase();
  return {
    balance: num(env.ACCOUNT_BALANCE, DEFAULT_ACCOUNT_BALANCE),
    riskPct: num(env.RISK_PCT, DEFAULT_RISK_PCT),
    currency,
    symbolCcy: { GBP: "£", EUR: "€", USD: "$" }[currency] || `${currency} `,
  };
}

async function resolveSecret(binding) {
  if (binding == null) return undefined;
  if (typeof binding === "string") return binding;
  if (typeof binding.get === "function") return await binding.get();
  return undefined;
}

const tvSymbol = (symbol) => `FX:${symbol.replace("/", "")}`;

/**
 * Account currency -> USD, cached for a day in KV.
 *
 * Both watched pairs are USD-quoted, so a pip is worth USD while the risk budget is
 * in pounds. Skipping the conversion would undersize every trade by ~27% and, worse,
 * do it silently - the number would look perfectly reasonable.
 *
 * Returns null rather than guessing if the rate cannot be had. A missing lot size is
 * an inconvenience; a confidently wrong one is a bad trade.
 */
async function accountToUsd(env, apiKey, currency) {
  if (currency === "USD") return 1;
  const key = `fx:${currency}USD:${new Date().toISOString().slice(0, 10)}`;
  if (env.ALERT_STATE) {
    const cached = await env.ALERT_STATE.get(key);
    if (cached) return Number(cached);
  }
  try {
    const resp = await fetch(
      `https://api.twelvedata.com/price?symbol=${currency}/USD&apikey=${apiKey}`);
    const data = await resp.json();
    await countCall(env);
    const rate = Number(data.price);
    if (!Number.isFinite(rate) || rate <= 0) return null;
    if (env.ALERT_STATE) await env.ALERT_STATE.put(key, String(rate), { expirationTtl: 60 * 60 * 25 });
    return rate;
  } catch {
    return null;
  }
}

/**
 * Lots to risk `riskPct` of `balance` given the stop distance.
 *
 * lots = risk_in_quote_ccy / (stop_pips x value_per_pip_per_lot)
 * where a standard lot is 100,000 units, so on a USD-quoted pair one pip per lot is
 * 100,000 x 0.0001 = $10.
 *
 * Rounded DOWN to the broker's 0.01 step, so the rounding always risks slightly less
 * than intended rather than slightly more.
 */
export function computeLots({ balance, riskPct, toUsd, stopDistance, pip, price, spreadPips }) {
  if (!toUsd || !(stopDistance > 0)) return null;
  const riskAccount = balance * (riskPct / 100);
  const riskUsd = riskAccount * toUsd;
  const stopPips = stopDistance / pip;
  const usdPerPipPerLot = LOT_UNITS * pip;
  const raw = riskUsd / (stopPips * usdPerPipPerLot);

  // Leverage cap. On a USD-quoted pair one lot is LOT_UNITS of base currency, worth
  // LOT_UNITS x price in USD, and the account converts to USD at `toUsd`. Both sides
  // are in USD here on purpose - comparing a GBP balance against a USD notional is
  // how a 30:1 cap silently becomes 40:1.
  const capped = price > 0
    ? (MAX_LEVERAGE * balance * toUsd) / (LOT_UNITS * price)
    : Infinity;
  const wanted = Math.min(raw, capped);

  const lots = Math.floor(wanted / MIN_LOT) * MIN_LOT;
  return {
    lots: Number(lots.toFixed(2)),
    rawLots: raw,
    stopPips,
    riskAccount,
    riskUsd,
    // What the rounded-down size actually risks - not what was asked for.
    actualRiskUsd: lots * stopPips * usdPerPipPerLot,
    belowMinimum: wanted < MIN_LOT,
    // Reported so the message can say the size was cut and why, rather than showing a
    // number that no longer matches the stated 1% and letting you work it out.
    leverageCapped: raw > capped,
    maxLotsAtCap: capped,
    notionalUsd: lots * LOT_UNITS * price,
    // A stop this tight is not a stop. Flagged, never silently sized around.
    stopInsideSpread: spreadPips > 0 && stopPips < spreadPips * MIN_STOP_SPREAD_MULT,
    spreadPips,
  };
}

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

// How many bars of context the chart shows. Was 60 on a 900px-wide image - once the
// axis, its labels and the right-side padding are subtracted, that is roughly 13px
// per candle, which is a hairline on a phone screen and exactly the "can't tell"
// complaint. Fewer bars, bigger image: ~830px of usable plot / 32 bars = ~26px each,
// more than double the width per candle, still enough bars either side of the signal
// to read the swing that produced it.
const CHART_BARS = 32;

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
  // Sit the arrow OUTSIDE the candle - below the low on a BUY, above the high on a
  // SELL - the way the Pine indicator places its own. Drawn at the entry price it
  // lands on the candle body in the same colour as an up-bar and disappears into it.
  // Offset scales with the visible candle range so it clears the wick at any zoom.
  const visibleRange =
    Math.max(...window.map((c) => c.high)) - Math.min(...window.map((c) => c.low)) || 1;
  const signalBar = window[signalIdx];
  const markerY = side === "BUY"
    ? signalBar.low - visibleRange * 0.08
    : signalBar.high + visibleRange * 0.08;
  const marker = (pointed) => ({
    type: "line",
    label: `${side} signal`,
    data: pointed
      ? [{ x: at[signalIdx], y: markerY }]
      : window.map((_, i) => (i === signalIdx ? markerY : null)),
    borderColor: "rgba(0,0,0,0)",
    backgroundColor: side === "BUY" ? "#00e6a0" : "#ff4d6a",
    pointStyle: side === "BUY" ? "triangle" : "triangle",
    pointRadius: 11,
    pointRotation: side === "BUY" ? 0 : 180,
    pointBorderColor: "#e6edf3",
    pointBorderWidth: 2,
    showLine: false,
    fill: false,
    clip: false,
    order: -1,
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


  // Both zones spelled out for the signal bar, so the picture can be lined up against
  // a TradingView chart on any timezone setting without arithmetic.
  const lastAt = new Date(at[at.length - 1]);
  const hhmm = (tz) => new Intl.DateTimeFormat("en-GB", {
    timeZone: tz, hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  }).format(lastAt);
  const titleLines = [
    `${symbol} 5m — ${side}`,
    `signal bar ${hhmm("UTC")} UTC  ·  ${hhmm("Europe/London")} London  ·  axis is UTC`,
  ];

  const common = {
    // The signal is always the LAST bar, which lands hard against the right-hand
    // y-axis. Without this padding its marker is drawn on top of the axis and reads
    // as missing - the first attempt looked like the marker had not been added at all.
    layout: { padding: { right: 28, top: 4 } },
    plugins: {
      // THE TITLE CARRIES THE CLOCK, and it has to, because four of them are in play:
      // QuickChart renders this axis in UTC (measured, not assumed - a 12:20-17:20 UTC
      // window came back labelled "1 PM" to "5 PM"), the strategy's session is London,
      // and the phone reading the alert is on neither. Comparing this picture against
      // a TradingView chart on a third zone made the bot look two hours stale when it
      // was thirty seconds behind. Bare "HH:mm" on an axis is not a time; a time needs
      // its zone attached.
      title: { display: true, text: titleLines, color: "#e6edf3", font: { size: 14 } },
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
        width: 1200,
        height: 640,
        // Rendered at 2x and let Telegram scale down, rather than rendered at 1x and
        // stretched up - the second one is what "squashed" looks like on a retina
        // phone screen even when the layout math is fine.
        devicePixelRatio: 2,
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

/**
 * WORDS FIRST, PICTURE SECOND. This used to render the chart and then send it with
 * the alert as its caption, which put a QuickChart render and a PNG upload - 3 to 5
 * seconds - in front of the only thing you can act on. On a warning whose entire
 * value is a two-minute head start, that was spending 3% of the window on a picture.
 *
 * It also made the countdown wrong. "1.7 min to close" was computed before the
 * render, so it arrived reading 1.7 when the truth was nearer 1.4 - the one number
 * the message exists to give you, quietly stale.
 *
 * So the text is sent the moment the evaluation is done, and the chart follows as a
 * second message. Two notifications instead of one is the price; the alternative was
 * a slow single one. A failed render now costs only the picture - the words have
 * already gone.
 */
async function deliver(env, caption, levels, symbol, candles, side) {
  const startedAt = Date.now();
  await sendText(env, caption);
  const textMs = Date.now() - startedAt;

  try {
    const image = await fetchChartImage(env, symbol, candles, levels, side);
    await sendPhoto(env, image, `\u{1F4C8} ${symbol} — ${side}`);
    return { imageSent: true, textMs, totalMs: Date.now() - startedAt };
  } catch (err) {
    // Deliberately silent to Telegram. The alert is already delivered, and a second
    // message saying a picture failed is noise on top of the thing that matters.
    return { imageSent: false, imageError: String(err), textMs, totalMs: Date.now() - startedAt };
  }
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

// The lot size, plus what it ACTUALLY risks after rounding to the broker's step -
// which is never exactly the 1% asked for, and is worth showing rather than implying.
function sizeLines(sizing, cfg) {
  if (!sizing) {
    return ["", `_Lot size unavailable - could not convert ${cfg.currency} to USD._`];
  }
  if (sizing.belowMinimum) {
    return [
      "",
      `\u{26A0}\uFE0F *Lot ${sizing.rawLots.toFixed(3)} is below the ${MIN_LOT} minimum.*`,
      `Trading ${MIN_LOT} would risk ~${cfg.symbolCcy}${(sizing.actualRiskUsd / cfg.toUsd).toFixed(2)}, more than your ${cfg.riskPct}%.`,
    ];
  }
  const actualPct = (sizing.actualRiskUsd / cfg.toUsd / cfg.balance) * 100;
  const lines = [
    "",
    `*Lot size* \`${sizing.lots.toFixed(2)}\`  (${sizing.stopPips.toFixed(1)} pip stop)`,
    `Risks ${cfg.symbolCcy}${(sizing.actualRiskUsd / cfg.toUsd).toFixed(2)} of ${cfg.symbolCcy}${cfg.balance.toLocaleString()} = ${actualPct.toFixed(2)}%`,
  ];

  // The two ways this number can be a lie, each said plainly rather than left for you
  // to notice from the size looking odd.
  if (sizing.stopInsideSpread) {
    lines.push(
      "",
      `\u{26D4} *DO NOT TRADE — the stop is inside the spread.*`,
      `${sizing.stopPips.toFixed(1)} pip stop vs a ~${sizing.spreadPips} pip spread. You would be most of the way to the stop the moment you enter.`,
      `The signal candle was only ${(sizing.stopPips / 2).toFixed(1)} pips tall, and the stop is 2x that.`);
  }
  if (sizing.leverageCapped) {
    lines.push(
      "",
      `\u{26A0}️ *Size capped at ${MAX_LEVERAGE}:1 leverage.*`,
      `1% of the account wanted ${sizing.rawLots.toFixed(2)} lots; the cap allows ${sizing.maxLotsAtCap.toFixed(2)}.`,
      `Actual risk above is what ${sizing.lots.toFixed(2)} lots really risks — less than ${cfg.riskPct}%.`);
  }
  return lines;
}

const outsideLondonNote = (result) => (result.inLondonSession ? [] : [
  "",
  `\u{26A0}\uFE0F _Outside ${LONDON_START}:00-${LONDON_END}:00 London. The strategy would NOT take this - session filter is open for testing._`,
]);

function formingMessage(pair, result, minsLeft, closeAt, sizing, cfg) {
  const arrow = result.side === "BUY" ? "\u{1F7E2}" : "\u{1F534}";
  // A countdown starts ageing the moment it is sent - by the time the notification is
  // read it is already wrong, and there is no way to tell by how much. The absolute
  // close time does not age, so it is the one to act on; the countdown stays only
  // because it reads faster at a glance. London, because that is the clock you are on.
  const closeLondon = new Intl.DateTimeFormat("en-GB", {
    timeZone: "Europe/London", hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23",
  }).format(closeAt);
  return [
    `\u{23F3} *FORMING — ${result.side}* ${arrow} ${pair.symbol}`,
    `*Closes ${closeLondon} London* (~${minsLeft.toFixed(1)} min). Get ready — do not enter yet.`,
    "",
    ...levelLines(pair, result.levels),
    "",
    `Pattern: ${result.pattern}`,
    `RSI ${result.values.rsi.toFixed(1)} (vs SMMA ${result.values.rsiSmma.toFixed(1)})  ·  ADX ${result.values.adx.toFixed(1)}`,
    "",
    ...sizeLines(sizing, cfg),
    "",
    `_Provisional — the bar's high/low can still move, which changes the stop AND therefore this lot size. Final numbers come at the close._`,
    ...outsideLondonNote(result),
  ].join("\n");
}

function confirmedMessage(pair, result, sizing, cfg, { warned = false } = {}) {
  const arrow = result.side === "BUY" ? "\u{1F7E2}" : "\u{1F534}";
  const ageMin = Math.round((Date.now() - parseUTC(result.values.time).getTime()) / 60000);
  // "CONFIRMED" only means something if there was an earlier warning to confirm.
  // Most signals now arrive with no warning before them, because most patterns cannot
  // exist until the bar closes - calling those "confirmed" would imply a heads-up that
  // was never sent.
  const heading = warned
    ? `\u{2705} *CONFIRMED — ${result.side}* ${arrow} ${pair.symbol}  \`5m closed\``
    : `\u{1F6A8} *SIGNAL — ${result.side}* ${arrow} ${pair.symbol}  \`5m closed\``;
  return [
    heading,
    "",
    ...levelLines(pair, result.levels),
    "",
    `Pattern: ${result.pattern}`,
    `RSI ${result.values.rsi.toFixed(1)} (vs SMMA ${result.values.rsiSmma.toFixed(1)})`,
    `ADX ${result.values.adx.toFixed(1)}  ·  ATR ${(result.values.atrRatio * 100).toFixed(0)}% of avg`,
    `Candle ${result.values.time} UTC (~${ageMin} min ago)`,
    "",
    ...sizeLines(sizing, cfg),
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
        if (ONE_TRADE_PER_DAY && (await env.ALERT_STATE.get(dayKey))) {
          results.push({ symbol: pair.symbol, skipped: "one trade per day already sent" });
          continue;
        }
        if (await env.ALERT_STATE.get(warnKey)) {
          results.push({ symbol: pair.symbol, skipped: "already warned on this bar" });
          continue;
        }
      }

      const cfg = sizingConfig(env);
      cfg.toUsd = await accountToUsd(env, apiKey, cfg.currency);
      const sizing = computeLots({
        balance: cfg.balance, riskPct: cfg.riskPct, toUsd: cfg.toUsd,
        stopDistance: Math.abs(result.levels.entry - result.levels.stop), pip: pair.pip,
        price: result.levels.entry, spreadPips: pair.spreadPips,
      });
      // Read the clock HERE, after the FX lookup, not before it. accountToUsd can hit
      // the network on the first call of the day, and a countdown measured before a
      // network call is a countdown that ships already wrong.
      const minsLeft = minutesToClose(forming);
      const closeAt = new Date(parseUTC(forming.time).getTime() + BAR_MINUTES * 60000);
      const delivery = await deliver(env, formingMessage(pair, result, minsLeft, closeAt, sizing, cfg), result.levels, pair.symbol, candles, result.side);

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

/**
 * EVERY PAIR, EVERY BAR. This used to open with
 *
 *     const raw = await env.ALERT_STATE.get(`pending:${pair.symbol}`);
 *     if (!raw) continue;   // nothing flagged - costs no API call, which is the point
 *
 * and that one line is why a signal could appear on the chart and the bot say nothing
 * at all. A pair was only ever scored at the close if the EARLY pass had already
 * flagged it three minutes before, on a bar that was 60% formed.
 *
 * Most of these patterns cannot exist before the close. An engulfing candle is not
 * engulfing until it has a close to compare; a 3-Line Strike needs the fourth bar's
 * close to clear open[1]. So the setups that only become true at the close - which is
 * the ordinary case, not an edge case - were never flagged early, therefore never
 * checked at the close, therefore never reported. Not late. Never.
 *
 * Saving an API call was the stated reason. It was saving them on exactly the bars
 * that mattered.
 *
 * The pending flag still does its old job of telling CONFIRMED from CANCELLED, but it
 * no longer decides whether to look.
 */
async function closePass(env) {
  if (!env.ALERT_STATE) return [{ skipped: "no KV binding, cannot dedupe alerts" }];
  const apiKey = await resolveSecret(env.TWELVEDATA_API_KEY);
  const results = [];

  for (const pair of PAIRS) {
    const raw = await env.ALERT_STATE.get(`pending:${pair.symbol}`);
    const flagged = raw ? JSON.parse(raw) : null;
    if (raw) await env.ALERT_STATE.delete(`pending:${pair.symbol}`);

    try {
      const candles = await fetchCandles(pair.symbol, apiKey, env);
      // Score only bars that have actually closed. Re-reading the forming bar here
      // would report a provisional setup as confirmed, which is the exact failure
      // this two-pass split exists to avoid.
      const { closed } = splitCandles(candles);
      const last = closed[closed.length - 1];
      if (!last) {
        results.push({ symbol: pair.symbol, skipped: "no closed bar" });
        continue;
      }

      // Dedupe on the BAR, not on the pass. The close cron and a manual ?pass=close
      // can both land on the same bar, and the alert must go out once.
      const sentKey = `${pair.symbol}:sent:${last.time}`;
      if (await env.ALERT_STATE.get(sentKey)) {
        results.push({ symbol: pair.symbol, skipped: `already alerted on ${last.time}` });
        continue;
      }

      const result = evaluateTMA(closed, { minDist: pair.minDist });

      // A flag that named a DIFFERENT bar is stale - the early pass warned about a bar
      // that has since been superseded. Cancelling on it would be a message about the
      // wrong candle, so it is dropped rather than reported against this one.
      const flagMatches = flagged && flagged.bar === last.time;

      if (!result.ok || !result.side) {
        if (flagMatches) {
          await sendText(env, cancelledMessage(pair, flagged, result));
          results.push({ symbol: pair.symbol, cancelled: flagged.side, blockedBy: result.ok ? firstBlockingGate(result.gates) : result.reason });
        } else {
          results.push({ symbol: pair.symbol, noSignal: result.ok ? firstBlockingGate(result.gates) : result.reason });
        }
        continue;
      }

      if (flagMatches && flagged.side !== result.side) {
        await sendText(env, cancelledMessage(pair, flagged, result));
        results.push({ symbol: pair.symbol, cancelled: flagged.side, blockedBy: `flipped to ${result.side}` });
        continue;
      }

      const cfg = sizingConfig(env);
      cfg.toUsd = await accountToUsd(env, apiKey, cfg.currency);
      const sizing = computeLots({
        balance: cfg.balance, riskPct: cfg.riskPct, toUsd: cfg.toUsd,
        stopDistance: Math.abs(result.levels.entry - result.levels.stop), pip: pair.pip,
        price: result.levels.entry, spreadPips: pair.spreadPips,
      });
      const delivery = await deliver(
        env, confirmedMessage(pair, result, sizing, cfg, { warned: flagMatches }),
        result.levels, pair.symbol, closed, result.side);

      await env.ALERT_STATE.put(sentKey, "1", { expirationTtl: 60 * 60 * 2 });
      await env.ALERT_STATE.put(`${pair.symbol}:day:${dayStamp(last.time)}`, "1", { expirationTtl: 60 * 60 * 36 });
      results.push({ symbol: pair.symbol, confirmed: result.side, warnedEarly: flagMatches, ...delivery });
    } catch (err) {
      results.push({ symbol: pair.symbol, error: String(err) });
    }
  }
  return results;
}

/**
 * `scheduledTime` is when Cloudflare INTENDED to fire, which is not when it did.
 * Cron Triggers are best-effort and drift by seconds to minutes, and that drift comes
 * straight off a two-minute head start. Recorded here so "the alert was late" is a
 * measurement with a number on it rather than an argument - cronDriftMs separates a
 * late trigger from slow work inside the pass, and they have completely different
 * fixes. Every KV write and Telegram call in this worker is instrumented the same way
 * for the same reason.
 */
async function runPass(env, which, { force = false, scheduledTime = null } = {}) {
  const startedAt = Date.now();
  const nowIso = new Date().toISOString().replace("T", " ").slice(0, 19);
  const cronDriftMs = scheduledTime ? startedAt - scheduledTime : null;

  if (!force && !inSession(nowIso)) {
    return { pass: which, skipped: `outside session (${SESSION_START_HOUR}:00-${SESSION_END_HOUR}:00 UTC, Mon-Fri)`, now: `${nowIso} UTC`, cronDriftMs };
  }
  const results = which === "early" ? await earlyPass(env) : await closePass(env);
  return {
    pass: which,
    now: `${nowIso} UTC`,
    cronDriftMs,
    passMs: Date.now() - startedAt,
    quotaUsedToday: await quotaUsed(env),
    results,
  };
}

export default {
  async scheduled(event, env, ctx) {
    const which = event.cron === CRON_EARLY ? "early" : "close";
    ctx.waitUntil(runPass(env, which, { scheduledTime: event.scheduledTime }));
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
        sizing: (() => { const c = sizingConfig(env); return { balance: c.balance, currency: c.currency, riskPct: c.riskPct }; })(),
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
