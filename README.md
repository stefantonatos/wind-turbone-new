# wind-turbone-new

Forex trade-setup alerts: a combined TradingView Pine Script fires when
your arrow pattern and RSI confirmation line up, and a small relay
forwards that alert to Telegram so you get notified instantly (you still
place the trade manually in MetaTrader 5).

## How it works

1. `pine/combined-setup-alert.pine` is a TradingView indicator that
   watches for your setup:
   - **Arrow**: a 3 Line Strike or Engulfing Candle pattern (bullish or
     bearish)
   - **RSI confirm**: RSI above its own 50-period smoothed average
     (bullish) or below it (bearish)
   - **Buy Setup** = bullish arrow + bullish RSI confirm
   - **Sell Setup** = bearish arrow + bearish RSI confirm
2. When the condition fires, TradingView sends a webhook (JSON) to a
   small Cloudflare Worker in `telegram-relay/`.
3. The Worker forwards a formatted message to your Telegram chat via a
   bot.

## 1. Add the Pine Script to TradingView

1. Open TradingView → Pine Editor → New blank indicator.
2. Paste the contents of `pine/combined-setup-alert.pine`.
3. Replace `REPLACE_WITH_YOUR_SECRET` (two occurrences, near the bottom)
   with a random string you make up — this is a shared secret so random
   people can't spam your bot even if they find your webhook URL. Use
   the same value later as `WEBHOOK_SECRET` when deploying the relay.
4. Click **Save**, then **Add to Chart**. Keep your existing two
   indicators on the chart too if you still want to see them visually —
   this script only adds `BUY`/`SELL` labels when both conditions align.

## 2. Create a Telegram bot

1. In Telegram, message **@BotFather**.
2. Send `/newbot`, follow the prompts (name, then a username ending in
   `bot`).
3. BotFather replies with a **bot token** — copy it, you'll need it in
   step 4.
4. Message your new bot anything (e.g. "hi") so it's allowed to message
   you back.
5. Get your **chat ID**: message **@userinfobot** (or open
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser
   after messaging your bot, and read the `chat.id` field from the
   JSON response).

## 3. Deploy the relay (Cloudflare Worker, free tier)

```bash
cd telegram-relay
npm install
npx wrangler login          # opens a browser to authorize
npx wrangler secret put TELEGRAM_BOT_TOKEN
npx wrangler secret put TELEGRAM_CHAT_ID
npx wrangler secret put WEBHOOK_SECRET   # same value you put in the Pine Script
npx wrangler deploy
```

`wrangler deploy` prints a URL like
`https://forex-setup-alerts.<your-subdomain>.workers.dev` — that's your
webhook URL.

## 4. Create the TradingView alerts

1. Right-click the chart → **Add Alert** (or the alarm-clock icon).
2. **Condition**: `Setup Alert` → `Buy Setup`. Trigger: **Once Per Bar
   Close** (recommended, avoids repainting/false triggers mid-candle).
3. Under **Notifications**, enable **Webhook URL** and paste your
   deployed Worker URL.
4. Leave the **Message** field as the default (it's pre-filled from the
   script's `alertcondition` message) — it already contains the JSON
   payload with your secret.
5. Save. Repeat steps 1–4 for `Sell Setup`.

Note: webhook alerts require your TradingView plan to support them —
most current paid plans do; check if the Webhook URL field is greyed
out on your plan.

## 5. Test it

- In the Pine Editor, temporarily change a condition to something that
  will trigger soon, or wait for a real signal.
- You can also test the relay directly:

```bash
curl -X POST https://forex-setup-alerts.<your-subdomain>.workers.dev \
  -H "Content-Type: application/json" \
  -d '{"side":"BUY","symbol":"EURUSD","interval":"15","price":"1.0850","time":"2026-07-28T12:00:00Z","secret":"<your secret>"}'
```

You should get a Telegram message within a couple seconds.
