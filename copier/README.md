# Telegram signal copier (MT5)

Watches a Telegram channel for trade calls and places the matching order
directly in your MetaTrader 5 account. Free - no TradingView, no paid API,
no VPS required (though a VPS improves uptime since this only copies
signals while it's running).

## Why this needs to run on a Windows PC, not a cloud function

Two things here can't run in something like a Cloudflare Worker:

- **Reading the channel** requires a real Telegram login (Telethon, the
  library used here) that most signal channels won't let a bot join - so
  it logs in as *your* Telegram account instead, the same access you'd
  have reading it manually.
- **Placing the order** uses MetaQuotes' official `MetaTrader5` Python
  package, which talks to a MetaTrader 5 terminal running on the *same
  machine* - it's a local IPC connection, not a network API, and
  MetaQuotes only ships Windows builds of it.

So both pieces need to run continuously on the Windows PC (or Windows VPS)
where your MT5 terminal is installed and logged into your broker.

## Setup

1. **Install MT5** if you haven't, log into your broker account (start
   with a **demo account** - see the warning below), and enable
   `Tools > Options > Expert Advisors > Allow Algo Trading`.

2. **Get Telegram API credentials** (free): go to
   https://my.telegram.org/apps, log in with your phone number, create an
   app. Note the `api_id` and `api_hash`.

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Configure**: copy `config.example.py` to `config.py` and fill in:
   - `API_ID` / `API_HASH` from step 2
   - `CHANNEL` - the signal channel's `@username`, or its numeric chat ID
     for a private channel (join it yourself first via the invite link -
     Telethon can only read channels you're already in)
   - `SYMBOL_MAP` / `SYMBOL_SUFFIX` if your broker names symbols
     differently than the channel does (e.g. `EURUSD.m`)
   - Leave `DRY_RUN = True` for now

5. **Check the parser against real messages first.** Copy 5-10 actual
   messages from the target channel into `test_parser.py`'s `SAMPLES`
   list, then run:
   ```
   python test_parser.py
   ```
   `parser.py` is a tolerant best-effort parser, not a verified port of
   this specific channel's format - fix its regexes (symbol aliases, entry
   / SL / TP patterns) until every sample message parses correctly. Don't
   skip this: a signal that silently fails to parse just gets ignored (a
   missed trade), but one that parses *wrong* - flipped side, wrong
   symbol, missing stop loss - places a real order you didn't intend.

6. **Run the listener**:
   ```
   python listener.py
   ```
   First run asks for your phone number and the login code Telegram sends
   you - that's Telethon logging in, one time. After that it reuses the
   session file automatically.

   With `DRY_RUN = True`, it prints the MT5 order it *would* send for
   every parsed signal without sending it. Compare a handful of these
   against the channel's actual messages before doing anything else.

7. **Go live**: once you trust the parsing, set `DRY_RUN = False` in
   `config.py`. Consider starting on the demo account for a while even
   after that, before pointing it at a live account.

## Position sizing

`FIXED_LOT_SIZE` (default) sends every trade at a flat lot size regardless
of the signal's stop distance. `USE_RISK_SIZING = True` instead sizes each
trade so that hitting the stop loses roughly `RISK_PERCENT` of
`ACCOUNT_EQUITY_FOR_SIZING` - but only works when the signal includes both
an entry price and a stop loss; falls back to `FIXED_LOT_SIZE` otherwise.

## Multiple take-profits

If a signal has TP1/TP2/TP3, only TP1 is currently sent as the order's
take profit; the rest are parsed (`signal.tps`) but unused. Splitting
volume across multiple TPs, or moving the stop to breakeven after TP1
hits, would need `listener.py` to track open positions across messages -
not built yet, since it depends on how the specific channel structures
multi-TP calls.

## The actual risk here

This places real orders with real money, automatically, based on someone
else's calls, the moment a message hits the channel. Nothing here
evaluates whether a given call is any good - it only automates typing the
order in exactly as fast as it can be parsed. Start on a demo account,
verify DRY_RUN output against real signals for a while, and know that
enabling this is a bet on the channel's *track record*, not on anything
this code assesses on your behalf.
