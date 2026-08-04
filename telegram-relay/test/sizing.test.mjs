// Run with: cd telegram-relay && npm test
//
// Position sizing. Every other bug in this repo produces a wrong message; this one
// produces a wrong trade, so the numbers are checked against hand-worked arithmetic
// rather than against themselves.
//
// THIS IMPORTS THE REAL FUNCTION. It used to hold a copy of computeLots with a
// comment saying "mirrors src/index.js", which meant the suite was testing the copy:
// the leverage cap and the spread check were added to the real one and every test
// here still passed, green and meaningless. A test that re-implements its subject
// verifies nothing except that the author can type it twice.

import assert from "node:assert/strict";
import { test } from "node:test";

import { computeLots } from "../src/index.js";

const MIN_LOT = 0.01;

// --- the original arithmetic ------------------------------------------------
// No `price` is passed in these, so the leverage cap is inactive and they measure
// the risk maths alone, exactly as before.

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

// --- guard rails ------------------------------------------------------------
// Taken from a real alert: AUD/USD, entry 0.70403, stop 0.70389. A 1.4 pip stop, and
// the message that went out asked for 9.60 lots - 960,000 AUD against a £10,000
// account. The risk arithmetic was right; the trade was impossible.

const LIVE = {
  balance: 10000, riskPct: 1, toUsd: 1.345, pip: 0.0001,
  price: 0.70403, spreadPips: 1.2,
};

test("the live 1.4 pip stop is flagged as inside the spread", () => {
  const r = computeLots({ ...LIVE, stopDistance: 0.00014 });
  assert.equal(r.stopPips.toFixed(1), "1.4");
  assert.equal(r.stopInsideSpread, true,
    "1.4 pips against a 1.2 pip spread is not a stop and must be called out");
});

test("a normal stop on the same pair is neither capped nor flagged", () => {
  const r = computeLots({ ...LIVE, stopDistance: 0.0012 }); // 12 pips
  assert.equal(r.stopInsideSpread, false);
  assert.equal(r.leverageCapped, false);
  const pct = (r.actualRiskUsd / LIVE.toUsd / LIVE.balance) * 100;
  assert.ok(pct > 0.9 && pct <= 1.0, `should still risk ~1%, got ${pct}%`);
});

test("leverage is capped at 30:1 instead of the 9.60 lots the risk maths asked for", () => {
  const r = computeLots({ ...LIVE, stopDistance: 0.00014 });
  // The number that actually went out in the alert.
  assert.ok(r.rawLots > 9 && r.rawLots < 10, `rawLots ${r.rawLots}`);
  assert.equal(r.leverageCapped, true);
  assert.ok(r.lots < r.rawLots, "the capped size must be smaller than the requested size");

  // Both sides in USD on purpose - a GBP balance against a USD notional turns a 30:1
  // cap into roughly 40:1 without anything looking wrong.
  const accountUsd = LIVE.balance * LIVE.toUsd;
  assert.ok(r.notionalUsd <= accountUsd * 30 + 1e-6,
    `notional $${r.notionalUsd} exceeds 30x $${accountUsd}`);
});

test("a capped trade reports what it REALLY risks, not the 1% that was requested", () => {
  const r = computeLots({ ...LIVE, stopDistance: 0.00014 });
  const pct = (r.actualRiskUsd / LIVE.toUsd / LIVE.balance) * 100;
  // Once the cap bites the position can no longer carry the intended risk. Reporting
  // "1%" at that point would be the message lying about the trade it just sent.
  assert.ok(pct < 1.0, `capped trade should risk less than 1%, got ${pct}%`);
});

test("omitting price leaves the cap inactive, so old call sites are unaffected", () => {
  const r = computeLots({ balance: 10000, riskPct: 1, toUsd: 1.345, stopDistance: 0.00014, pip: 0.0001 });
  assert.equal(r.leverageCapped, false);
  assert.equal(r.stopInsideSpread, false);
  assert.ok(Math.abs(r.lots - Math.floor(r.rawLots / MIN_LOT) * MIN_LOT) < 1e-9);
});
