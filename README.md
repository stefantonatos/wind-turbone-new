# wind-turbone-new

**The main thing in this repo is `webapp/` — a Streamlit strategy-backtesting
app.** It runs 17 trading strategies (plus a random-entry control) against real
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

- **Trend gate (must hold)**: 21/50/200 smoothed moving averages stacked
  in trend order, with price on the matching side of the 200 — only
  bullish signals count in an uptrend, only bearish in a downtrend.
- **Arrow**: a 3 Line Strike or Engulfing Candle pattern, matching the
  trend direction.
- **RSI confirm**: RSI(14) above 50 for a buy, below 50 for a sell.
- Fires **once per closed 5-minute candle**, only during your trading
  window (08:00–02:30 Europe/London, covering your waking hours — edit
  `isWithinTradingWindow` in `telegram-relay/src/index.js` if that
  changes).
- Monitors **EUR/USD, GBP/USD, USD/JPY** (edit the `PAIRS` array in the
  same file to change the list — more pairs costs more of the free API
  quota, see below).
- The Telegram message includes a suggested SL (2× the signal candle's
  range) and TP (2:1 reward:risk), per the strategy's rule — you still
  decide and place the actual trade.

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

npx wrangler deploy
```

The Cron Trigger (`*/5 * * * *`, every 5 minutes) is defined in
`wrangler.toml` and starts running automatically once deployed — nothing
else to wire up.

## 4. Test it

```bash
# Confirm Telegram delivery works at all:
curl "https://forex-setup-alerts.<your-subdomain>.workers.dev/?secret=<your WEBHOOK_SECRET>&ping=1"

# Manually run a full check right now (outside the cron schedule) and see the result per pair:
curl "https://forex-setup-alerts.<your-subdomain>.workers.dev/?secret=<your WEBHOOK_SECRET>"
```

The second command returns JSON showing, per pair, whether it alerted,
skipped (and why — outside trading window, no setup, already alerted
this candle), or errored.

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
