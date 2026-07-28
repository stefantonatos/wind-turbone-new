# Copy this file to config.py and fill in your own values.
# NEVER commit config.py - it holds your Telegram API credentials, and
# listener.py will also create a copier_session.session file the first
# time it logs in, which is a live login token for your Telegram account.
# Both are gitignored already; keep it that way.

# From https://my.telegram.org/apps - create an app there (free) to get
# these. This logs in as YOUR Telegram account (not a bot), because bots
# generally can't be added to arbitrary signal channels - this is the same
# access a human reading the channel manually would have.
API_ID = 0
API_HASH = ""

SESSION_NAME = "copier_session"

# The channel/group to copy signals from. Use the @username for a public
# channel, or the numeric chat ID for a private one (you must already be a
# member - Telethon can't join a private channel for you, use the invite
# link yourself first).
CHANNEL = "@some_signal_channel"

# Broker-specific symbol translation. Signal channels often use aliases
# ("GOLD") or names your broker doesn't (some append suffixes like
# "EURUSD.m"). parser.py normalizes aliases to a canonical name (e.g.
# GOLD -> XAUUSD); this maps that canonical name to what your broker
# actually calls it, if different. Leave a symbol out to use it unchanged.
SYMBOL_MAP = {
    # "XAUUSD": "XAUUSD.m",
}
SYMBOL_SUFFIX = ""  # applied after SYMBOL_MAP, e.g. ".m" for every symbol

# Safety default: log what WOULD be sent to MT5 instead of sending it.
# Only flip to False once test_parser.py and a few real listener.py runs
# show signals parsing correctly.
DRY_RUN = True

# Position sizing: either a fixed lot size, or risk-based sizing (needs the
# signal to include both an entry price and a stop loss - if either is
# missing, falls back to FIXED_LOT_SIZE).
FIXED_LOT_SIZE = 0.01
USE_RISK_SIZING = False
RISK_PERCENT = 0.01              # fraction of ACCOUNT_EQUITY_FOR_SIZING risked per trade
ACCOUNT_EQUITY_FOR_SIZING = 1000  # or read mt5.account_info().equity live - see listener.py
