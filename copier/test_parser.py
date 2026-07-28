# Sanity check for parser.py against a handful of common signal-channel
# message formats. These samples are made up to cover the common shapes -
# they are NOT taken from any real channel. Once you have the real channel
# picked, paste 5-10 of its actual messages in here (replacing or alongside
# these) and fix parser.py until every one comes out right before turning
# DRY_RUN off in config.py.

from parser import parse_signal

SAMPLES = [
    "BUY EURUSD @ 1.0950\nSL 1.0900\nTP 1.1000",
    "🟢 SELL GOLD\nEntry: 2350-2355\nSL: 2365\nTP1: 2340\nTP2: 2330\nTP3: 2320",
    "GBPJPY SHORT\nEntry 189.50\nStop Loss 190.20\nTake Profit 188.00",
    "US30 buy now\nsl 38500\ntp 38700",
    "just chatting, no trade here today",
]

if __name__ == "__main__":
    for text in SAMPLES:
        signal = parse_signal(text)
        print("-" * 60)
        print(text)
        print("->", signal)
