# Parses free-text Telegram trade signal messages into a structured Signal.
#
# Signal-provider channels don't agree on a format, so this is a tolerant
# best-effort parser built around the patterns most channels use in
# practice (BUY/SELL + pair + optional entry + SL + TP/TP1/TP2...). It is
# a first pass, not a verified port of any specific channel's format - run
# test_parser.py against real messages copied from the target channel and
# tighten SYMBOL_ALIASES / the regexes below to match what actually shows
# up before trusting this in listener.py with DRY_RUN off.

import re
from dataclasses import dataclass, field
from typing import Optional

SIDE_WORDS = {
    "BUY LIMIT": "BUY",
    "BUY STOP": "BUY",
    "SELL LIMIT": "SELL",
    "SELL STOP": "SELL",
    "BUY": "BUY",
    "LONG": "BUY",
    "SELL": "SELL",
    "SHORT": "SELL",
}

# Canonical symbol per alias. Broker-specific suffixes (e.g. "XAUUSD.m") are
# applied later in listener.py via config.SYMBOL_SUFFIX / config.SYMBOL_MAP,
# not here - this stays broker-agnostic.
SYMBOL_ALIASES = {
    "GOLD": "XAUUSD",
    "XAUUSD": "XAUUSD",
    "SILVER": "XAGUSD",
    "XAGUSD": "XAGUSD",
    "US30": "US30",
    "DOW": "US30",
    "NAS100": "NAS100",
    "NASDAQ": "NAS100",
    "SPX500": "SPX500",
    "US500": "SPX500",
    "GER40": "GER40",
    "DAX": "GER40",
    "BTCUSD": "BTCUSD",
    "BITCOIN": "BTCUSD",
}

FOREX_PAIR_RE = re.compile(r"\b([A-Z]{3})[/\s]?([A-Z]{3})\b")


@dataclass
class Signal:
    side: str
    symbol: str
    entry: Optional[float] = None       # None = market order
    entry_high: Optional[float] = None  # far end of a quoted entry range, if any
    sl: Optional[float] = None
    tps: list = field(default_factory=list)
    raw: str = ""


def _find_side(upper: str) -> Optional[str]:
    # Check multi-word forms ("BUY LIMIT") before the bare "BUY"/"SELL" they
    # contain, so a pending-order signal doesn't get misread as a market one.
    for word in sorted(SIDE_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(word)}\b", upper):
            return SIDE_WORDS[word]
    return None


def _find_symbol(upper: str) -> Optional[str]:
    for alias, canonical in SYMBOL_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", upper):
            return canonical
    m = FOREX_PAIR_RE.search(upper)
    if m:
        return m.group(1) + m.group(2)
    return None


def _find_price_pair(upper: str, *labels):
    for label in labels:
        m = re.search(rf"{label}[:\s@]*([0-9]+\.?[0-9]*)(?:\s*[-/]\s*([0-9]+\.?[0-9]*))?", upper)
        if m:
            lo = float(m.group(1))
            hi = float(m.group(2)) if m.group(2) else None
            return lo, hi
    return None, None


def parse_signal(text: str) -> Optional[Signal]:
    upper = text.upper()

    side = _find_side(upper)
    if side is None:
        return None

    symbol = _find_symbol(upper)
    if symbol is None:
        return None

    entry, entry_high = _find_price_pair(upper, "ENTRY", "@", "AT")
    sl, _ = _find_price_pair(upper, "SL", "S/L", "STOP LOSS", "STOPLOSS")

    # No \s* between "TP" and the optional 1-2 digit index: if it were
    # allowed, a plain "TP 38700" (no index) let the greedy index group eat
    # into the price itself (e.g. captured "38700" as index="387", price="00").
    # Requiring the index to be glued directly to "TP" avoids that; the
    # tradeoff is "TP 1: 188" (space before the index) isn't recognized as
    # indexed - rare enough to accept for a first-pass parser.
    tps = [float(m.group(2)) for m in re.finditer(r"TP(\d{1,2})?[:\s]+([0-9]+\.?[0-9]*)", upper)]
    if not tps:
        tp, _ = _find_price_pair(upper, "TAKE PROFIT", "T/P")
        if tp is not None:
            tps.append(tp)

    return Signal(side=side, symbol=symbol, entry=entry, entry_high=entry_high, sl=sl, tps=tps, raw=text)
