# wind-turbone-new

**The main thing in this repo is `webapp/` — a Streamlit strategy-backtesting
app.** It runs 19 trading strategies (plus a random-entry control) against real
historical Dukascopy data, deducts realistic prop-firm trading costs, ranks
them on an out-of-sample holdout, and shows how much of each result is the
strategy versus the cost assumption. A separate **Momentum** page runs a
monthly-rebalanced trend-following portfolio across 27 instruments, which
reports as a NAV curve rather than a trade list.

```bash
pip install -r webapp/requirements.txt
streamlit run webapp/app.py
```

See **[`webapp/README.md`](webapp/README.md)** for how it works, what the
numbers mean, and the one-time setup that makes run history survive restarts.

Read this first if you are looking at the output: most strategies in the
catalog lose money once realistic costs are applied. That is the expected
result for simple technical rules, not a bug — and the app now ships a
random-entry control specifically so you can tell "this strategy has no edge"
apart from "the cost model is too harsh", which produce identical-looking
tables and are not otherwise distinguishable.

### Layout

| path | what it is | status |
|---|---|---|
| `webapp/` | the Streamlit backtesting app | **active — this is the product** |
| `research/` | the strategy backtests + optimization pipelines the app runs | **active** |
| `telegram-relay/`, `pine/` | the earlier Telegram/TradingView alert bot (documented below) | legacy |
| `backtester/`, `quantconnect/`, `copier/` | earlier experiments — a JS backtester, QuantConnect ports, and an MT5 trade copier | legacy, not wired to anything |

Everything below this line documents the **legacy alert bot**, which was this
repo's original purpose and is kept for reference. It is independent of the
backtesting app and neither one imports the other.

---

## Legacy: Telegram forex alert bot

Forex trade-setup alerts — fully free, no TradingView paid plan required.
A Cloudflare Worker polls real forex price data on a schedule, recomputes
your setup itself, and messages you on Telegram when it fires. You still
place the trade manually in MetaTrader 5.

## The setup being detected

The alerts run the **TMA Trend Scalper (vFinal)** — the strategy currently
being forward-tested. All seven gates must hold on a closed 5-minute candle:

| Gate | Rule |
|---|---|
| Session | 07:00–15:00 UTC, Monday–Friday |
| Trend stack | SMMA 21/50/200 in order, each separated by at least `minDist` |
| Trend strength | ADX(14) > 25 |
| Volatility | ATR(14) > 70% of its own 50-bar average |
| Position | price on the correct side of the 200 SMMA |
| Momentum | close vs SMA(5) and vs close[5], agreeing with the trend |
| Pattern | 3 Line Strike **or** Engulfing, matching the trend |
| RSI | RSI(14) past 50 **and** past its own SMMA(50) |

Then: **one alert per instrument per day**, stop at 2× the signal candle's
range, target at 4× (2:1). Position size comes from the stop distance, not a
fixed lot — that is what "risk 1%" means when the stop moves every bar.

### Two alerts per signal: a heads-up, then a verdict

| | fires | says |
|---|---|---|
| ⏳ **FORMING** | :03, :08, :13 … — 2 min before the bar closes | the setup qualifies *right now*. Get to the screen. **Do not enter.** |
| ✅ **CONFIRMED** | :00, :05, :10 … — just after it closes | it survived. Levels are final. |
| ❌ **CANCELLED** | same pass | it did not, and which gate broke it. |

The split exists because **mid-candle every input is provisional** — the bar's
own high and low can still move, which changes the stop distance, and RSI, ADX
and the pattern can all flip before the close. The strategy's rule is evaluated
on a *closed* bar, so only the CONFIRMED message reflects it. The FORMING one is
a timer, not a signal.

The close pass only spends an API call on pairs the early pass actually flagged,
which is what makes two passes per candle affordable on the free tier at all.

> **Note on the earlier version of this bot.** It alerted on a simpler setup:
> the same 21/50/200 stack, pattern and `RSI > 50`, over an 08:00–02:30 London
> window. It had no ADX gate, no volatility gate, no momentum check, no minimum
> SMMA separation, and it never compared RSI to its own average. It therefore
> fired on setups the current strategy rejects. That logic still lives in
> `telegram-relay/src/strategy.js` because `backtester/backtest.js` imports it;
> the live alerts now use `telegram-relay/src/tma-strategy.js` instead.

Monitors **6 pairs** — AUD/USD, EUR/USD, GBP/USD, NZD/USD, USD/CAD, USD/JPY.
That number is set by the free TwelveData tier, not by preference:

| pairs | requests/day | headroom under the 800 cap |
|---|---|---|
| 6 | ~600 | ~200 |
| 8 | ~793 | **7** |

Eight fits on paper, but seven spare calls means a single manual `?debug=1`
(one call per pair) tips it over and the alerts then fail silently for the rest
of the day. Six leaves real room. The 8-requests-per-minute limit caps it at 8
regardless, since one fire calls every pair back to back.

JPY pairs use `minDist: 0.10` rather than `0.001` — it is an absolute price
distance, so it does not scale across quote currencies.

The Worker counts its own API calls per UTC day into KV and reports
`quotaUsedToday` from `?debug=1`, so you can check real usage rather than trust
the arithmetic above.

`pine/tma-trend-scalper.pine` is the same rules as a TradingView strategy, with
a live confluence table so you can see which gate is blocking a signal and
check the bot against the chart row by row.

`pine/combined-setup-alert.pine` mirrors the same logic as a TradingView
indicator, purely so you can visually sanity-check the `BUY`/`SELL`
labels against what the bot sends you. It is **not** wired to any
TradingView alert — no paid plan needed anywhere in this setup.

`pine/combined-setup-strategy.pine` is the same rules again, but declared
as a `strategy()` so TradingView's own Strategy Tester (Performance
Summary, List of Trades) can backtest it directly on the chart. Free-plan
history is limited to ~5000 bars (~2-3 weeks on 5-min candles), and the
dollar P&L it shows isn't precise for forex without proper lot sizing —
treat win rate and trade count as the numbers worth comparing against
`backtester/backtest.js`'s output, not the $ figures.

## Backtesting on QuantConnect (free, real historical data)

`quantconnect/main.py` is the same strategy again, ported to QuantConnect's
free cloud backtester (Python/LEAN). This is worth using instead of (or
alongside) `backtester/backtest.js` because QuantConnect has real forex
history going back years — our own backtester has only tested 17 days of
manually-copied EUR/USD data so far, which is a small sample. The indicator
math was checked line-for-line against `strategy.js` on the same real
EUR/USD data and produced byte-identical signals (128/128 matching
timestamps, sides, and RSI values) before being shipped here, so this
isn't a re-derived guess — it's a verified port.

1. Sign up free at https://www.quantconnect.com (email only, no card).
2. Create a new Algorithm Project (Python).
3. Delete the default code, paste in `quantconnect/main.py`.
4. Click **Backtest**. It defaults to EUR/USD, all of 2024.
5. To test the reversed direction (the variant that backtested profitably
   on our own 17-day sample — see git history for that result), change
   `self.REVERSE_SIGNALS = False` to `True` near the top and re-run.

Only one trade is held at a time in this version (a new signal is ignored
while a previous trade is still open) — slightly different from
`backtester/backtest.js`, which opens an independent trade on every
qualifying bar even if overlapping. This is closer to how a real account
would actually be managed.

## 1. Get a free TwelveData API key

1. Sign up at https://twelvedata.com/pricing (Basic/free plan — email
   only, no card).
2. Copy your API key from the dashboard.
3. Free tier: 800 requests/day, 8/minute. At 5-minute candles this
   comfortably covers 3 pairs across an ~18.5-hour window
   (3 × ~222 requests/day ≈ 666, under the 800 cap). Adding pairs or
   widening the window eats into that budget — do the math before
   changing either.

## 2. Create a Telegram bot

1. In Telegram, message **@BotFather**.
2. Send `/newbot`, follow the prompts (name, then a username ending in
   `bot`).
3. BotFather replies with a **bot token** — copy it.
4. Message your new bot anything (e.g. "hi") so it's allowed to message
   you back.
5. Get your **chat ID**: message **@userinfobot**, or open
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser
   after messaging your bot and read the `chat.id` field.

## 3. Deploy the relay (Cloudflare Worker, free tier)

```bash
cd telegram-relay
npm install
npx wrangler login                       # opens a browser to authorize
npx wrangler kv namespace create ALERT_STATE
# paste the returned id into wrangler.toml, replacing REPLACE_WITH_KV_NAMESPACE_ID

npx wrangler secret put TWELVEDATA_API_KEY
npx wrangler secret put TELEGRAM_BOT_TOKEN
npx wrangler secret put TELEGRAM_CHAT_ID
npx wrangler secret put WEBHOOK_SECRET    # any random string you make up, for testing access
npx wrangler secret put CHARTIMG_API_KEY  # optional - free key from https://chart-img.com

npx wrangler deploy
```

`CHARTIMG_API_KEY` is what puts a TradingView-style chart image on each alert.
It is optional by design: if it is missing, or the call fails, or you run out of
free-tier quota, the alert still goes out as text. A missing picture must never
cost you the signal.

**Never paste any of these tokens into a chat, a commit, or a code file.**
`wrangler secret put` prompts for the value and stores it encrypted with
Cloudflare — it never touches the repo.

The Cron Trigger (`*/5 * * * *`, every 5 minutes) is defined in
`wrangler.toml` and starts running automatically once deployed — nothing
else to wire up.

## 4. Test it

```bash
BASE="https://forex-setup-alerts.<your-subdomain>.workers.dev"
S="<your WEBHOOK_SECRET>"

# 1. Confirm Telegram delivery works at all:
curl "$BASE/?secret=$S&ping=1"

# 2. Confirm the chart image works, and that you like how it looks,
#    WITHOUT waiting for a real signal:
curl "$BASE/?secret=$S&testchart=AUD/USD"

# 3. Evaluate every pair right now, ignoring the session gate, and see
#    exactly which condition is blocking each one (also reports quota used):
curl "$BASE/?secret=$S&debug=1"

# 4. Drive either pass by hand. force=1 ignores the session window.
curl "$BASE/?secret=$S&pass=early&force=1"
curl "$BASE/?secret=$S&pass=close&force=1"
```

`debug=1` is the one to use when checking the bot against TradingView. For every
pair it reports either the signal, or `blockedBy` naming the first failing gate
along with the live ADX and RSI. Those gate names line up one-for-one with the
rows of the confluence table in `pine/tma-trend-scalper.pine`, so you can put the
two side by side and see whether they agree.

Without `debug=1` the same endpoint respects the session window, which is what
the cron does.

## Changing pairs, timeframe, or window

Most of this lives in `telegram-relay/src/index.js`:

- `PAIRS` — symbol list and pip size per symbol
- `INTERVAL` — candle timeframe (must be a value TwelveData supports:
  `1min`, `5min`, `15min`, `30min`, `1h`, `4h`, ...)
- `isWithinTradingWindow` — active hours

Indicator periods (`RSI_LEN`, `MA_LENS`, `CONFIRM_BARS`) live in
`telegram-relay/src/strategy.js` instead, since that file is shared
with the backtester.

Redeploy with `npx wrangler deploy` after any change.
