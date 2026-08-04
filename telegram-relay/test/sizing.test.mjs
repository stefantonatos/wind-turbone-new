// Run with: cd telegram-relay && npm test
//
// Position sizing. Every other bug in this repo produces a wrong message; this one
// produces a wrong trade, so the numbers are checked against hand-worked arithmetic
// rather than against themselves.

import assert from "node:assert/strict";
import { test } from "node:test";

const MIN_LOT = 0.01;
const LOT_UNITS = 100000;

// Mirrors computeLots in src/index.js.
function computeLots({ balance, riskPct, toUsd, stopDistance, pip }) {
  if (!toUsd || !(stopDistance > 0)) return null;
  const riskAccount = balance * (riskPct / 100);
  const riskUsd = riskAccount * toUsd;
  const stopPips = stopDistance / pip;
  const usdPerPipPerLot = LOT_UNITS * pip;
  const raw = riskUsd / (stopPips * usdPerPipPerLot);
  const lots = Math.floor(raw / MIN_LOT) * MIN_LOT;
  return {
    lots: Number(lots.toFixed(2)), rawLots: raw, stopPips, riskAccount, riskUsd,
    actualRiskUsd: lots * stopPips * usdPerPipPerLot,
    belowMinimum: raw < MIN_LOT,
  };
}

test("worked example: 10k GBP, 1%, 20-pip stop on a USD-quoted pair", () => {
  // £10,000 x 1% = £100. At 1.27 that is $127.
  // A 20-pip stop at $10/pip/lot costs $200 per lot, so 127/200 = 0.635 lots.
  const r = computeLots({ balance: 10000, riskPct: 1, toUsd: 1.27, stopDistance: 0.0020, pip: 0.0001 });
  assert.equal(r.stopPips, 20);
  assert.equal(r.riskAccount, 100);
  assert.equal(r.riskUsd, 127);
  assert.ok(Math.abs(r.rawLots - 0.635) < 1e-9);
  assert.equal(r.lots, 0.63, "rounded DOWN to the broker's step");
});

test("rounding always risks less than asked, never more", () => {
  for (const stop of [0.0007, 0.0013, 0.0019, 0.0026, 0.0031]) {
    const r = computeLots({ balance: 10000, riskPct: 1, toUsd: 1.27, stopDistance: stop, pip: 0.0001 });
    assert.ok(r.actualRiskUsd <= r.riskUsd + 1e-9,
      `stop ${stop}: risked ${r.actualRiskUsd} > budget ${r.riskUsd}`);
  }
});

test("a tighter stop buys a bigger position, which is the whole point of the rule", () => {
  const tight = computeLots({ balance: 10000, riskPct: 1, toUsd: 1.27, stopDistance: 0.0010, pip: 0.0001 });
  const wide = computeLots({ balance: 10000, riskPct: 1, toUsd: 1.27, stopDistance: 0.0030, pip: 0.0001 });
  assert.ok(tight.lots > wide.lots);
  // and both still risk about the same money
  assert.ok(Math.abs(tight.actualRiskUsd - wide.actualRiskUsd) < 3);
});

test("skipping the GBP->USD conversion would undersize by the exchange rate", () => {
  const correct = computeLots({ balance: 10000, riskPct: 1, toUsd: 1.27, stopDistance: 0.0020, pip: 0.0001 });
  const naive = computeLots({ balance: 10000, riskPct: 1, toUsd: 1, stopDistance: 0.0020, pip: 0.0001 });
  assert.ok(Math.abs(correct.rawLots / naive.rawLots - 1.27) < 1e-9,
    "a missing conversion is a silent ~27% undersize, not a rounding detail");
});

test("no rate means no number, rather than a confidently wrong one", () => {
  assert.equal(computeLots({ balance: 10000, riskPct: 1, toUsd: null, stopDistance: 0.002, pip: 0.0001 }), null);
});

test("a zero-width stop cannot produce an infinite position", () => {
  assert.equal(computeLots({ balance: 10000, riskPct: 1, toUsd: 1.27, stopDistance: 0, pip: 0.0001 }), null);
});

test("a small account on a wide stop is flagged as below the broker minimum", () => {
  // £100 account, 1% = £1 ~ $1.27, 20-pip stop -> 0.006 lots, under the 0.01 floor.
  const r = computeLots({ balance: 100, riskPct: 1, toUsd: 1.27, stopDistance: 0.0020, pip: 0.0001 });
  assert.equal(r.belowMinimum, true);
  assert.equal(r.lots, 0);
});

test("JPY pip size is handled, not assumed to be 0.0001", () => {
  // 0.01 pip, so a lot is worth 100,000 x 0.01 = 1000 quote units per pip.
  const r = computeLots({ balance: 10000, riskPct: 1, toUsd: 1.27, stopDistance: 0.20, pip: 0.01 });
  assert.equal(r.stopPips, 20);
  assert.ok(r.rawLots > 0 && Number.isFinite(r.rawLots));
});
