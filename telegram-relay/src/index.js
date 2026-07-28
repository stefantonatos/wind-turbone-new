export default {
  async fetch(request, env) {
    if (request.method !== "POST") {
      return new Response("Method Not Allowed", { status: 405 });
    }

    let payload;
    try {
      payload = await request.json();
    } catch (err) {
      return new Response("Invalid JSON", { status: 400 });
    }

    if (!env.WEBHOOK_SECRET || payload.secret !== env.WEBHOOK_SECRET) {
      return new Response("Unauthorized", { status: 401 });
    }

    const side = payload.side === "SELL" ? "SELL" : "BUY";
    const emoji = side === "BUY" ? "\u{1F7E2}" : "\u{1F534}";
    const text = [
      `${emoji} *${side} SETUP*`,
      `Symbol: ${payload.symbol ?? "?"}`,
      `Timeframe: ${payload.interval ?? "?"}`,
      `Price: ${payload.price ?? "?"}`,
      `Time: ${payload.time ?? "?"}`,
    ].join("\n");

    const tgResp = await fetch(
      `https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chat_id: env.TELEGRAM_CHAT_ID,
          text,
          parse_mode: "Markdown",
        }),
      }
    );

    if (!tgResp.ok) {
      const errText = await tgResp.text();
      return new Response(`Telegram error: ${errText}`, { status: 502 });
    }

    return new Response("OK", { status: 200 });
  },
};
