# Offline backtester

`backtest.js` walks a CSV of 5-minute candles bar by bar and simulates the
same buy/sell setup the live Cloudflare Worker (`telegram-relay/src/index.js`)
alerts on: 21/50/200 smoothed-MA trend gate, 3-Line-Strike/Engulfing arrow,
RSI(14) vs 50 confirm. It imports the indicator/strategy functions directly
from `../telegram-relay/src/strategy.js`, so it is always testing the exact
same logic the Worker runs live - not a re-derived copy that could drift out
of sync.

It makes **no network calls**. Everything runs offline against a local CSV
file, using only Node's built-in modules (`fs`, `path`, `url`) - no
`npm install` required.

## CSV format

A header row followed by data rows, oldest candle first (ascending
chronological order):

```
datetime,open,high,low,close
2026-01-05 00:00,1.10000,1.10005,1.09995,1.10002
2026-01-05 00:05,1.10002,1.10010,1.09998,1.10006
...
```

Notes:

- One row per 5-minute candle. The script needs at least 200 candles before
  it can evaluate anything (the 200-period trend MA needs that much history
  to warm up), so very short files will be rejected with an error.
- Column **order** doesn't actually matter - `backtest.js` matches columns by
  header name (case-insensitive), so `open,high,low,close,datetime` works
  just as well as the order shown above. A separate `date` + `time` column
  pair (instead of one combined `datetime` column) is also accepted and will
  be concatenated automatically.
- Comma, semicolon, and tab delimiters are all auto-detected.
- Extra columns (tick volume, real volume, spread, etc., which MT5 exports
  often include) are simply ignored.
- If your export doesn't match this and the parser can't find the columns it
  needs, it prints an error naming which columns it couldn't identify - open
  the CSV in a text editor, rename the header row to match the names above
  (or reorder/re-export), and re-run.

## Exporting matching data from MetaTrader 5

MT5's built-in exporter is the "History Center":

1. In the MT5 terminal, open **Tools > History Center** (or press `F2`).
2. Select the symbol (e.g. `EURUSD`) and the **M5** timeframe in the tree.
3. Make sure enough history is loaded/downloaded for the date range you want
   to test (MT5 may need to download older bars from your broker first -
   there's usually a "Download"/"Request" button in this dialog if the bars
   aren't cached locally yet).
4. Click **Export** (or **Export Bars**, depending on your MT5 build) and
   save as a `.csv` file.
5. MT5's default export typically writes separate `Date` and `Time` columns
   (and sometimes uses `;` or `\t` as the separator, or omits a header row
   entirely depending on version/broker terminal). This script tolerates
   split date/time columns and different delimiters, but if it still can't
   parse the file, open it in a spreadsheet program and add/rename a header
   row with `datetime,open,high,low,close` (or `date,time,open,high,low,close`)
   so the columns are unambiguous, then re-save as CSV.

If your MT5 build doesn't expose "Export" directly from History Center, the
equivalent path is: right-click the symbol in the **Market Watch** panel >
**Charts** or **Specification**, open the M5 chart, then use
**File > Save As** on the chart, or open History Center from there - the
menu wording varies a little by MT5 build/broker skin, but "History Center"
plus an "Export" button is the standard route in every version.

## Running it

```
cd backtester
node backtest.js fixtures/sample.csv
```

Optional flags (all have defaults):

```
node backtest.js path/to/candles.csv --pip=0.0001 --balance=10000 --risk=1
```

- `--pip` - price value of one pip for this instrument (default `0.0001`;
  use `0.01` for JPY pairs, matching the `pip` values already used per-pair
  in `telegram-relay/src/index.js`).
- `--balance` / `--risk` - starting account balance and risk-per-trade
  percentage, used only to print an optional simulated account-growth
  figure alongside the pips/R stats. Purely informational.

## What it simulates

For every bar where the setup fires:

- Entry = that candle's close.
- Stop-loss distance = 2x the signal candle's high-low range.
- Take-profit distance = 4x the signal candle's range (2:1 reward:risk, as
  prescribed by the strategy).
- The script then scans forward through subsequent candles' high/low to see
  which level is reached first.
  - **If a single forward candle's range contains both the SL and TP
    levels, the backtester assumes the worse outcome (SL hit first)** -
    this is a conservative assumption since only OHLC data is available,
    not tick-by-tick price movement, so the true intrabar order can't be
    known for certain.
  - Trades that never hit either level before the CSV data runs out are
    reported as "still open" and excluded from the win-rate/pips
    statistics (but counted and listed separately).
- Every bar that independently satisfies the setup opens its own simulated
  trade - there's no cooldown suppressing a setup that persists across
  several consecutive bars, matching how the live Worker's per-poll
  evaluation logic itself works.

## Sample fixture

`fixtures/sample.csv` is **synthetic, engineered data** (not real market
data), meant only to prove the backtester runs end-to-end and produces sane
output. It contains a seeded random-walk warmup, an engineered sustained
uptrend with a bullish engulfing reversal, and an engineered sustained
downtrend with a bearish engulfing reversal, so `node backtest.js
fixtures/sample.csv` reliably produces both BUY and SELL signals to exercise
the whole pipeline. Do not treat its results as a real strategy performance
figure - swap in real MT5-exported history to backtest for real.
