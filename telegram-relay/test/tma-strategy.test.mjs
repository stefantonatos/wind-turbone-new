// Run with: node --test telegram-relay/test/
//
// The indicator maths is verified against the Python reference separately (see the
// PR description); these tests cover the SETUP LOGIC - the gates, the session
// window, and the level arithmetic - which is where a live alert most easily drifts
// from the strategy it claims to implement.

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  MIN_BARS, adxSeries, atrSeries, evaluateTMA, firstBlockingGate,
  inSession, isForming, minutesToClose, rmaSeries, rsiSeries, smaSeries, splitCandles,
} from "../src/tma-strategy.js";

// --- helpers ---------------------------------------------------------------

function bar(time, open, high, low, close) {
  return { time, open, high, low, close };
}

/**
 * A choppy warmup followed by a decisive directional leg.
 *
 * The choppy phase is not decoration - it is what makes this data realistic enough
 * to fire. A purely monotonic series drives RSI to exactly 100 and its SMMA(50) to
 * 100 with it, so `RSI > SMMA(RSI)` is false forever and NOTHING can ever signal.
 * Real trends oscillate, which keeps the RSI average low enough for a decisive leg
 * to push RSI above it. Getting this wrong looks like a broken strategy when it is
 * really broken test data.
 */
function series(dir, { chop = 220, leg = 50, amp = 0.0002, step = 0.0008, hour = 9 } = {}) {
  const at = (i) => `2026-01-05 ${String(hour).padStart(2, "0")}:${String(i % 60).padStart(2, "0")}:00`;
  const out = [];
  let price = 0.6;
  for (let i = 0; i < chop; i++) {
    const o = price;
    const c = price + (i % 2 ? -amp : amp);
    out.push(bar(at(i), o, Math.max(o, c) + 0.00005, Math.min(o, c) - 0.00005, c));
    price = c;
  }
  for (let i = 0; i < leg; i++) {
    const o = price;
    const c = price + step * dir;
    out.push(bar(at(chop + i), o, Math.max(o, c) + 0.00005, Math.min(o, c) - 0.00005, c));
    price = c;
  }
  return out;
}

/** Appends an engulfing candle in `dir`, completing a valid setup. */
function addEngulfing(candles, dir, hour = 9) {
  const prev = candles.at(-1);
  const o = dir > 0 ? prev.open - 0.0002 : prev.open + 0.0002;
  const c = dir > 0 ? prev.open + 0.0012 : prev.open - 0.0012;
  candles.push(bar(`2026-01-05 ${String(hour).padStart(2, "0")}:59:00`,
    o, Math.max(o, c) + 0.0001, Math.min(o, c) - 0.0001, c));
  return candles;
}

/** A short series, deliberately below the warmup requirement. */
function shortSeries(n) {
  const out = [];
  let price = 0.6;
  for (let i = 0; i < n; i++) {
    const o = price;
    const c = price + (i % 2 ? -0.0002 : 0.0002);
    out.push(bar(`2026-01-05 09:${String(i % 60).padStart(2, "0")}:00`,
      o, Math.max(o, c) + 0.00005, Math.min(o, c) - 0.00005, c));
    price = c;
  }
  return out;
}

// --- primitives ------------------------------------------------------------

test("smaSeries is null until the window is full, then correct", () => {
  const s = smaSeries([1, 2, 3, 4], 3);
  assert.equal(s[0], null);
  assert.equal(s[1], null);
  assert.equal(s[2], 2);
  assert.equal(s[3], 3);
});

test("rmaSeries seeds on the simple average then smooths recursively", () => {
  const s = rmaSeries([2, 4, 6, 8], 3);
  assert.equal(s[1], null);
  assert.equal(s[2], 4); // (2+4+6)/3
  assert.equal(s[3], (4 * 2 + 8) / 3);
});

test("rmaSeries returns all-null rather than a wrong number when history is short", () => {
  assert.deepEqual(rmaSeries([1, 2], 5), [null, null]);
});

test("rmaSeries skips leading nulls via startIndex instead of treating them as zero", () => {
  // A true-range series has no bar-0 value. Seeding from index 0 would average a
  // null-as-zero in and drag the whole series down.
  const withLeadingNull = [null, 3, 3, 3];
  const s = rmaSeries(withLeadingNull, 3, 1);
  assert.equal(s[3], 3);
});

test("rsiSeries pins to 100 on an unbroken advance and 0 on an unbroken decline", () => {
  const up = Array.from({ length: 40 }, (_, i) => 1 + i * 0.01);
  const down = Array.from({ length: 40 }, (_, i) => 2 - i * 0.01);
  assert.equal(rsiSeries(up, 14).at(-1), 100);
  assert.equal(rsiSeries(down, 14).at(-1), 0);
});

test("atrSeries and adxSeries produce finite values on a real-shaped series", () => {
  const c = series(1, { chop: 60, leg: 60 });
  assert.ok(atrSeries(c, 14).at(-1) > 0);
  const adx = adxSeries(c, 14).at(-1);
  assert.ok(Number.isFinite(adx) && adx >= 0 && adx <= 100);
});

// --- session ---------------------------------------------------------------

test("session accepts 07:00-14:59 UTC on a weekday and rejects the edges", () => {
  assert.equal(inSession("2026-01-05 07:00:00"), true); // Monday
  assert.equal(inSession("2026-01-05 14:55:00"), true);
  assert.equal(inSession("2026-01-05 06:55:00"), false);
  assert.equal(inSession("2026-01-05 15:00:00"), false);
});

test("session rejects weekends", () => {
  assert.equal(inSession("2026-01-03 10:00:00"), false); // Saturday
  assert.equal(inSession("2026-01-04 10:00:00"), false); // Sunday
});

test("session is evaluated in UTC, not the worker's local timezone", () => {
  // 23:30Z is outside the window regardless of where the isolate happens to run.
  assert.equal(inSession("2026-01-05 23:30:00"), false);
});

// --- setup evaluation ------------------------------------------------------

test("refuses to evaluate without enough warmup rather than guessing", () => {
  const r = evaluateTMA(shortSeries(50));
  assert.equal(r.ok, false);
  assert.match(r.reason, /need \d+ bars/);
});

test("a clean uptrend with no pattern produces no signal, and says which gate blocked it", () => {
  const c = series(1);
  const r = evaluateTMA(c, { minDist: 0.0001 });
  assert.equal(r.ok, true);
  assert.equal(r.side, null);
  assert.equal(firstBlockingGate(r.gates), "pattern");
});

test("a bullish engulfing at the end of a clean uptrend fires a BUY", () => {
  const r = evaluateTMA(addEngulfing(series(1), 1), { minDist: 0.0001 });
  assert.equal(r.ok, true);
  assert.equal(r.side, "BUY");
  assert.equal(r.pattern, "Engulfing");
});

test("the same setup outside the session window produces no signal", () => {
  const c = addEngulfing(series(1, { hour: 3 }), 1, 3); // 03:xx UTC
  const r = evaluateTMA(c, { minDist: 0.0001 });
  assert.equal(r.side, null);
  assert.equal(r.gates.session, false);
});

test("RSI must beat its own SMMA, not merely sit above 50", () => {
  // The single most important difference from the OLD alert bot, which checked
  // RSI > 50 only. A monotonic ramp puts RSI at 100 - comfortably above 50 - while
  // its SMMA(50) is also 100, so the real gate correctly refuses the setup.
  const monotonic = [];
  let price = 0.6;
  for (let i = 0; i < MIN_BARS + 20; i++) {
    const o = price;
    const c = price + 0.0006;
    monotonic.push(bar(`2026-01-05 09:${String(i % 60).padStart(2, "0")}:00`, o, c + 0.0002, o - 0.0002, c));
    price = c;
  }
  const r = evaluateTMA(addEngulfing(monotonic, 1), { minDist: 0.0001 });
  assert.ok(r.values.rsi > 50, "RSI is above 50 here");
  assert.equal(r.gates.rsi, false, "but it is not above its own SMMA, so the gate must fail");
  assert.equal(r.side, null);
});

test("a separation requirement wider than the actual gap blocks the stack gate", () => {
  const c = series(1);
  const loose = evaluateTMA(c, { minDist: 0.0001 });
  const strict = evaluateTMA(c, { minDist: 10 });
  assert.equal(loose.gates.stack, true);
  assert.equal(strict.gates.stack, false);
});

test("stop is 2x and target 4x the signal candle, giving 2:1 both ways", () => {
  const c = addEngulfing(series(1), 1);
  const r = evaluateTMA(c, { minDist: 0.0001 });
  const last = c.at(-1);
  const size = last.high - last.low;
  assert.ok(Math.abs((r.levels.entry - r.levels.stop) - size * 2) < 1e-12);
  assert.ok(Math.abs((r.levels.target - r.levels.entry) - size * 4) < 1e-12);
  const risk = r.levels.entry - r.levels.stop;
  const reward = r.levels.target - r.levels.entry;
  assert.ok(Math.abs(reward / risk - 2) < 1e-9, "reward:risk must be exactly 2:1");
});

test("a SELL mirrors the levels rather than recomputing them in one direction", () => {
  const r = evaluateTMA(addEngulfing(series(-1), -1), { minDist: 0.0001 });
  assert.equal(r.side, "SELL");
  assert.ok(r.levels.stop > r.levels.entry, "short stop must sit above entry");
  assert.ok(r.levels.target < r.levels.entry, "short target must sit below entry");
  const risk = r.levels.stop - r.levels.entry;
  const reward = r.levels.entry - r.levels.target;
  assert.ok(Math.abs(reward / risk - 2) < 1e-9, "reward:risk must be 2:1 on shorts too");
});

test("gate breakdown is always returned so an alert can explain a rejection", () => {
  const r = evaluateTMA(series(1), { minDist: 0.0001 });
  for (const key of ["session", "stack", "adx", "volatility", "priceVs200", "momentum", "pattern", "rsi"]) {
    assert.equal(typeof r.gates[key], "boolean", `missing gate: ${key}`);
  }
});

// --- forming vs closed bars ------------------------------------------------
// The two-pass alerting rests entirely on telling these apart. If a "confirmation"
// ever re-reads the still-forming bar it reports a provisional setup as confirmed,
// which is the one failure mode that would make the alerts actively misleading.

test("a bar is forming until exactly 5 minutes after its open stamp", () => {
  const c = bar("2026-01-05 09:35:00", 1, 1, 1, 1);
  assert.equal(isForming(c, new Date("2026-01-05T09:36:00Z")), true);
  assert.equal(isForming(c, new Date("2026-01-05T09:39:59Z")), true);
  assert.equal(isForming(c, new Date("2026-01-05T09:40:00Z")), false);
  assert.equal(isForming(c, new Date("2026-01-05T09:41:00Z")), false);
});

test("minutesToClose counts down and goes negative once closed", () => {
  const c = bar("2026-01-05 09:35:00", 1, 1, 1, 1);
  assert.equal(minutesToClose(c, new Date("2026-01-05T09:38:00Z")), 2);
  assert.ok(minutesToClose(c, new Date("2026-01-05T09:42:00Z")) < 0);
});

test("splitCandles withholds the forming bar from the closed set", () => {
  const candles = [
    bar("2026-01-05 09:30:00", 1, 1, 1, 1),
    bar("2026-01-05 09:35:00", 1, 1, 1, 1),
  ];
  const at = new Date("2026-01-05T09:38:00Z"); // second bar still forming
  const { closed, forming } = splitCandles(candles, at);
  assert.equal(closed.length, 1);
  assert.equal(closed.at(-1).time, "2026-01-05 09:30:00");
  assert.equal(forming.time, "2026-01-05 09:35:00");
});

test("splitCandles returns everything once the last bar has closed", () => {
  const candles = [
    bar("2026-01-05 09:30:00", 1, 1, 1, 1),
    bar("2026-01-05 09:35:00", 1, 1, 1, 1),
  ];
  const { closed, forming } = splitCandles(candles, new Date("2026-01-05T09:40:30Z"));
  assert.equal(closed.length, 2);
  assert.equal(forming, null);
});

test("splitCandles on an empty series does not throw", () => {
  assert.deepEqual(splitCandles([], new Date()), { closed: [], forming: null });
});

test("a forming bar can qualify and the closed bar then not - the case the two passes exist for", () => {
  // Same setup, but the bar closes having given back its gains: the engulfing no
  // longer engulfs, so the early warning must be followed by a cancel, not a confirm.
  const base = series(1);
  const prev = base.at(-1);

  const formingBar = bar("2026-01-05 09:59:00", prev.open - 0.0002, prev.open + 0.0012, prev.open - 0.0003, prev.open + 0.0010);
  const withForming = [...base, formingBar];
  assert.equal(evaluateTMA(withForming, { minDist: 0.0001 }).side, "BUY");

  // ...and the same bar, closed weakly back below the previous open.
  const closedBar = bar("2026-01-05 09:59:00", prev.open - 0.0002, prev.open + 0.0012, prev.open - 0.0005, prev.open - 0.0004);
  const withClosed = [...base, closedBar];
  const after = evaluateTMA(withClosed, { minDist: 0.0001 });
  assert.equal(after.side, null, "the weak close must not still read as a BUY");
  assert.equal(after.gates.pattern, false);
});

test("gates are direction-aware: a bearish pattern inside a bullish stack is not a pass", () => {
  // Regression. This gate used to be `any pattern fired`, which made the JSON from
  // ?debug=1 disagree with the Pine confluence table it is meant to be compared to.
  const base = series(1);
  const prev = base.at(-1);
  // Three rising bars precede this one, and it closes back through the previous
  // open - a BEARISH 3-Line Strike, sitting inside a bullish stack.
  const bearish = bar("2026-01-05 09:59:00", prev.open - 0.0002, prev.open + 0.0012, prev.open - 0.0005, prev.open - 0.0004);
  const r = evaluateTMA([...base, bearish], { minDist: 0.0001 });
  assert.equal(r.side, null, "no trade: the pattern opposes the trend");
  assert.equal(r.gates.pattern, false, "and the gate must report that, not 'a pattern exists'");
});
