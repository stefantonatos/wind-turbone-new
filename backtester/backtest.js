#!/usr/bin/env node
// Offline, walk-forward backtester for the trend + arrow + RSI setup that
// telegram-relay/src/index.js alerts on live. Reuses the exact same
// indicator/strategy functions from ../telegram-relay/src/strategy.js so
// this can never silently drift from what the live Worker evaluates.
//
// Usage:
//   node backtest.js path/to/candles.csv [--pip=0.0001] [--balance=10000] [--risk=1]
//
// See README.md in this directory for the expected CSV format and how to
// export matching data from MetaTrader 5's History Center.
//
// No network calls are made anywhere in this script - it only reads the
// local CSV file passed on the command line.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { MA_LENS, evaluateSetup } from "../telegram-relay/src/strategy.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const args = { csvPath: null, pip: 0.0001, balance: 10000, riskPercent: 1 };
  const positional = [];
  for (const raw of argv) {
    if (raw.startsWith("--pip=")) args.pip = Number(raw.slice("--pip=".length));
    else if (raw.startsWith("--balance=")) args.balance = Number(raw.slice("--balance=".length));
    else if (raw.startsWith("--risk=")) args.riskPercent = Number(raw.slice("--risk=".length));
    else positional.push(raw);
  }
  args.csvPath = positional[0] || null;
  return args;
}

// ---------------------------------------------------------------------------
// CSV loading
// ---------------------------------------------------------------------------

// Picks whichever of comma / semicolon / tab actually splits the header line
// into the most fields. MT5 exports commonly use comma, but some regional
// Excel setups re-export with semicolons, and old-style History Center dumps
// use tabs - this keeps the parser tolerant of all three without any flags.
function detectDelimiter(headerLine) {
  const candidates = [",", ";", "\t"];
  let best = ",";
  let bestCount = 1;
  for (const d of candidates) {
    const count = headerLine.split(d).length;
    if (count > bestCount) {
      bestCount = count;
      best = d;
    }
  }
  return best;
}

function cleanHeaderCell(cell) {
  return cell.trim().toLowerCase().replace(/^[<"']+|[>"']+$/g, "");
}

// Expected format (see README.md): a header row followed by rows of
//   datetime,open,high,low,close
// in ascending chronological order (oldest first), one row per 5-minute
// candle. Column ORDER is not assumed - columns are matched by header name
// (case-insensitive, tolerant of MT5-style "<OPEN>" wrapping) so a
// differently-ordered MT5 export still works as long as the header names are
// recognizable. A separate "date" + "time" column pair is also accepted and
// concatenated into one datetime string. Any extra columns (tick volume,
// real volume, spread, etc., which MT5 often includes) are ignored.
function parseCSV(filePath) {
  const raw = fs.readFileSync(filePath, "utf8");
  const lines = raw.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) {
    throw new Error("CSV file has no data rows (need a header row plus at least one candle).");
  }

  const delim = detectDelimiter(lines[0]);
  const header = lines[0].split(delim).map(cleanHeaderCell);

  const idx = {};
  header.forEach((h, i) => {
    if (h.includes("datetime")) idx.datetime = i;
    else if (h === "date" || (h.includes("date") && !h.includes("update"))) idx.date = i;
    else if (h === "time" || (h.includes("time") && !h.includes("datetime"))) idx.time = i;
    else if (h.includes("open")) idx.open = i;
    else if (h.includes("high")) idx.high = i;
    else if (h.includes("low")) idx.low = i;
    else if (h.includes("close")) idx.close = i;
  });

  if (idx.open == null || idx.high == null || idx.low == null || idx.close == null) {
    throw new Error(
      `Could not find open/high/low/close columns in header: "${lines[0]}". ` +
        "Expected column names containing open, high, low, close (see README.md)."
    );
  }
  if (idx.datetime == null && idx.date == null) {
    throw new Error(
      `Could not find a datetime (or date) column in header: "${lines[0]}". ` +
        "Expected a column named datetime, or separate date/time columns."
    );
  }

  const candles = [];
  let skipped = 0;
  for (let li = 1; li < lines.length; li++) {
    const cols = lines[li].split(delim).map((c) => c.trim().replace(/^["']|["']$/g, ""));
    if (cols.length < header.length) {
      skipped++;
      continue;
    }
    const datetime =
      idx.datetime != null ? cols[idx.datetime] : `${cols[idx.date]}${idx.time != null ? " " + cols[idx.time] : ""}`;
    const candle = {
      time: datetime,
      open: Number(cols[idx.open]),
      high: Number(cols[idx.high]),
      low: Number(cols[idx.low]),
      close: Number(cols[idx.close]),
    };
    if ([candle.open, candle.high, candle.low, candle.close].some((v) => !Number.isFinite(v))) {
      skipped++;
      continue;
    }
    candles.push(candle);
  }

  if (skipped > 0) {
    console.error(`Warning: skipped ${skipped} malformed/non-numeric row(s) while parsing ${filePath}.`);
  }

  return candles;
}

// ---------------------------------------------------------------------------
// Backtest engine
// ---------------------------------------------------------------------------

// Walks forward bar by bar. At each bar i (once the 200-period MA has enough
// history to be valid) it evaluates the setup using only candles[0..i] -
// exactly the information the live Worker would have had at that point in
// time - via the shared evaluateSetup() from strategy.js. When a setup
// fires, it opens a simulated trade at that candle's close and scans forward
// through the *following* candles to see whether stop-loss or take-profit is
// hit first.
//
// Assumptions worth flagging:
//  - Every bar that independently satisfies the setup opens its own trade.
//    There is no cooldown/deduplication across consecutive bars (the live
//    Worker only dedupes by "already alerted this exact candle timestamp",
//    it does not suppress a setup persisting across several bars either) -
//    so a strong trending run can produce several overlapping simulated
//    trades. This mirrors what the live alerting logic would actually fire.
//  - If a single forward-scanned candle's high/low range contains BOTH the
//    SL and the TP level, we assume the worse outcome: SL is treated as hit
//    first. This is the conservative assumption commonly used in backtests
//    that only have OHLC (not tick-level) data, since we cannot know the
//    true intrabar order in which price traveled.
//  - Trades still open when the CSV data runs out are recorded as "open"
//    and excluded from win-rate/pips stats, but counted and reported
//    separately.
function runBacktest(candles, { pip = 0.0001 } = {}) {
  const trades = [];
  const minIndex = MA_LENS.slow - 1; // first bar index where ma200 is valid

  for (let i = minIndex; i < candles.length; i++) {
    const windowCandles = candles.slice(0, i + 1);
    const { trend, currentRSI, buySetup, sellSetup } = evaluateSetup(windowCandles);
    if (!buySetup && !sellSetup) continue;

    const signalCandle = candles[i];
    const range = signalCandle.high - signalCandle.low;
    if (!(range > 0)) continue; // guard against degenerate/bad candle data

    const side = buySetup ? "BUY" : "SELL";
    const entry = signalCandle.close;
    const slDistance = range * 2; // SL = 2x signal candle range
    const tpDistance = range * 4; // TP = 4x signal candle range (2:1 reward:risk)

    const sl = side === "BUY" ? entry - slDistance : entry + slDistance;
    const tp = side === "BUY" ? entry + tpDistance : entry - tpDistance;

    let outcome = "open";
    let exitPrice = null;
    let exitIndex = null;

    for (let j = i + 1; j < candles.length; j++) {
      const c = candles[j];
      const hitSL = side === "BUY" ? c.low <= sl : c.high >= sl;
      const hitTP = side === "BUY" ? c.high >= tp : c.low <= tp;

      if (hitSL) {
        // Same-bar-hits-both assumption: SL wins (see comment above).
        outcome = "loss";
        exitPrice = sl;
        exitIndex = j;
        break;
      }
      if (hitTP) {
        outcome = "win";
        exitPrice = tp;
        exitIndex = j;
        break;
      }
    }

    const markPrice = outcome === "open" ? candles[candles.length - 1].close : exitPrice;
    const pips = side === "BUY" ? (markPrice - entry) / pip : (entry - markPrice) / pip;
    const riskPips = slDistance / pip;
    const rMultiple = riskPips > 0 ? pips / riskPips : 0;

    trades.push({
      index: i,
      time: signalCandle.time,
      side,
      trend,
      rsi: currentRSI,
      entry,
      sl,
      tp,
      outcome, // "win" | "loss" | "open"
      exitTime: exitIndex != null ? candles[exitIndex].time : null,
      pips,
      rMultiple,
    });
  }

  return trades;
}

function computeStreaks(closedTradesInOrder) {
  let curWin = 0;
  let curLoss = 0;
  let maxWin = 0;
  let maxLoss = 0;
  for (const t of closedTradesInOrder) {
    if (t.outcome === "win") {
      curWin++;
      curLoss = 0;
    } else {
      curLoss++;
      curWin = 0;
    }
    maxWin = Math.max(maxWin, curWin);
    maxLoss = Math.max(maxLoss, curLoss);
  }
  return { maxWin, maxLoss };
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

function fmt(n, digits = 1) {
  return Number.isFinite(n) ? n.toFixed(digits) : "n/a";
}

function printReport(candles, trades, args) {
  const closed = trades.filter((t) => t.outcome !== "open");
  const open = trades.filter((t) => t.outcome === "open");
  const wins = closed.filter((t) => t.outcome === "win");
  const losses = closed.filter((t) => t.outcome === "loss");
  const totalPips = closed.reduce((sum, t) => sum + t.pips, 0);
  const avgR = closed.length ? closed.reduce((sum, t) => sum + t.rMultiple, 0) / closed.length : 0;
  const winRate = closed.length ? (wins.length / closed.length) * 100 : 0;
  const { maxWin, maxLoss } = computeStreaks(closed);

  console.log("=".repeat(60));
  console.log("Forex setup backtest report");
  console.log("=".repeat(60));
  console.log(`Candles loaded:     ${candles.length}`);
  if (candles.length) {
    console.log(`Date range:         ${candles[0].time}  ->  ${candles[candles.length - 1].time}`);
  }
  console.log(`Pip size used:      ${args.pip}`);
  console.log("-".repeat(60));
  console.log(`Total signals:      ${trades.length}`);
  console.log(`  Wins:             ${wins.length}`);
  console.log(`  Losses:           ${losses.length}`);
  console.log(`  Still open:       ${open.length} (excluded from win-rate/pips stats below)`);
  console.log(`Win rate:           ${fmt(winRate)}% (of ${closed.length} closed trades)`);
  console.log(`Total pips:         ${fmt(totalPips)}`);
  console.log(`Average R-multiple: ${fmt(avgR, 2)}R`);
  console.log(`Largest win streak: ${maxWin}`);
  console.log(`Largest loss streak:${" ".repeat(0)} ${maxLoss}`);
  console.log("-".repeat(60));

  if (closed.length && Number.isFinite(args.balance) && Number.isFinite(args.riskPercent) && args.riskPercent > 0) {
    let balance = args.balance;
    for (const t of closed) {
      const riskAmount = balance * (args.riskPercent / 100);
      balance += riskAmount * t.rMultiple;
    }
    const returnPct = ((balance - args.balance) / args.balance) * 100;
    console.log(`Account growth simulation (${args.riskPercent}% risk per trade, starting ${args.balance}):`);
    console.log(`  Final balance:    ${fmt(balance, 2)}`);
    console.log(`  Total return:     ${fmt(returnPct, 2)}%`);
    console.log("-".repeat(60));
  }

  console.log("Individual trades:");
  if (!trades.length) {
    console.log("  (none)");
  }
  for (const t of trades) {
    const rsiStr = Number.isFinite(t.rsi) ? t.rsi.toFixed(1) : "n/a";
    console.log(
      `  [${t.outcome.toUpperCase().padEnd(4)}] ${t.time}  ${t.side}  trend=${t.trend}  rsi=${rsiStr}  ` +
        `entry=${t.entry.toFixed(5)}  sl=${t.sl.toFixed(5)}  tp=${t.tp.toFixed(5)}  ` +
        `pips=${fmt(t.pips)}  R=${fmt(t.rMultiple, 2)}` +
        (t.exitTime ? `  exit=${t.exitTime}` : "")
    );
  }
  console.log("=".repeat(60));
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.csvPath) {
    console.error("Usage: node backtest.js path/to/candles.csv [--pip=0.0001] [--balance=10000] [--risk=1]");
    process.exit(1);
  }

  const resolvedPath = path.resolve(process.cwd(), args.csvPath);
  if (!fs.existsSync(resolvedPath)) {
    console.error(`File not found: ${resolvedPath}`);
    process.exit(1);
  }

  const candles = parseCSV(resolvedPath);
  const minRequired = MA_LENS.slow;
  if (candles.length < minRequired) {
    console.error(
      `Not enough candles (${candles.length}) - need at least ${minRequired} for the 200-period trend MA to warm up.`
    );
    process.exit(1);
  }

  const trades = runBacktest(candles, { pip: args.pip });
  printReport(candles, trades, args);
}

main();
