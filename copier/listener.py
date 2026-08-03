# Watches a Telegram channel for new messages, parses each one as a trade
# signal, and places the corresponding order in MT5. Run this continuously
# on the same Windows machine/VPS where MT5 is installed and logged into
# your broker account (with Algo Trading enabled in the terminal).
#
# First run will prompt for your phone number and the login code Telegram
# sends you - that's Telethon logging in as your own account so it can read
# a channel you're a member of. After that it reuses config.SESSION_NAME's
# session file, no more prompts.

import asyncio

import MetaTrader5 as mt5
from telethon import TelegramClient, events

import config
from executor import calc_lot_size, connect as mt5_connect, place_market_order
from parser import parse_signal

client = TelegramClient(config.SESSION_NAME, config.API_ID, config.API_HASH)


def resolve_symbol(canonical_symbol: str) -> str:
    return config.SYMBOL_MAP.get(canonical_symbol, canonical_symbol) + config.SYMBOL_SUFFIX


def decide_lots(symbol: str, signal) -> float:
    if config.USE_RISK_SIZING and signal.entry is not None and signal.sl is not None:
        sl_distance = abs(signal.entry - signal.sl)
        risk_amount = config.ACCOUNT_EQUITY_FOR_SIZING * config.RISK_PERCENT
        return calc_lot_size(symbol, sl_distance, risk_amount)
    return config.FIXED_LOT_SIZE


@client.on(events.NewMessage(chats=config.CHANNEL))
async def on_message(event):
    text = event.raw_text
    signal = parse_signal(text)
    if signal is None:
        return

    print(f"\n[{event.date}] parsed: {signal}")

    symbol = resolve_symbol(signal.symbol)
    tp = signal.tps[0] if signal.tps else None
    lots = decide_lots(symbol, signal)

    try:
        place_market_order(
            symbol, signal.side, lots,
            sl=signal.sl, tp=tp,
            dry_run=config.DRY_RUN,
        )
    except Exception as exc:
        print(f"  order failed: {exc}")


async def main():
    mt5_connect()
    account = mt5.account_info()
    if account is None:
        raise RuntimeError(f"MT5 connected but no account logged in: {mt5.last_error()}")
    print(f"MT5 connected: account #{account.login}, balance {account.balance} {account.currency}")
    print(f"DRY_RUN = {config.DRY_RUN}")

    await client.start()
    print(f"Listening on {config.CHANNEL} ...")
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
