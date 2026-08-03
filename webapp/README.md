# Strategy Backtests - web UI

A Streamlit dashboard over this project's `research/*.py` backtest scripts. Pick a
strategy, instruments, and a date range, click **Run Backtest**, and watch it run a
real backtest against live Dukascopy data - then browse the results as charts and
tables instead of a scrolling terminal wall of text. Four pages, top nav (no
sidebar): **Run Backtest**, **Browse Strategies** (the full catalog, plus a "Compare
All Strategies" leaderboard that runs every strategy over the same date range and
ranks them), **Gallery** (every past run as a sortable/filterable card with a
red/green equity-curve thumbnail and an editable name), and **History** (the full
detail view for any past run).

Runs locally (`streamlit run webapp/app.py`) or deployed to Streamlit Community
Cloud. See "Persisting history across restarts" below if deployed - Streamlit Cloud's
local disk does not survive a redeploy or sleep/wake cycle on its own.

## Run it

```bash
pip install -r webapp/requirements.txt
streamlit run webapp/app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## What to know before you click "Run Backtest"

- **Every run fetches real historical data live from Dukascopy** via each strategy's
  own `research/*.py` module - nothing is mocked or precomputed. The date range
  defaults to a short 6-month window on purpose: a first-time fetch of a wide range (the
  research scripts themselves often default to 2005/2010/2016 through 2025) can take
  many minutes even with caching. Widen the range deliberately once you know what
  you're doing.
- Repeated runs over a range you've already fetched are fast - every raw Dukascopy
  fetch is cached to disk under `webapp/cache/` (see `webapp/data_cache.py`), and the
  "Rauf" strategy additionally reuses its own existing pickle-based cache
  (`research/day_trading_rauf_dukascopy_backtest.py`'s `CACHE_DIR`, redirected into
  `webapp/cache/rauf_dukascopy_cache/` instead of the repo root).
- **The underlying research scripts model no commission, spread, or slippage at all.**
  This app deducts it afterwards, by default, everywhere - see "How to read the
  numbers" below, because that deduction is doing a lot of work.
- Toggling a filter on the results page (outcome, instrument, side, date sub-range,
  session/range) never re-fetches data - it only recomputes stats over the trade list
  already produced by your last "Run Backtest" click.

## How to read the numbers

Three things will change how you interpret every result in this app.

**1. Costs are deducted by default, and they dominate.** Each trade's R is reduced by
`(cost_pct / 100) / stop_pct` using researched prop-firm round-trip costs (see
`stats.py`'s `TYPICAL_COST_PCT_BY_INSTRUMENT` for the figures and their sourcing gaps).
Because the divisor is the stop distance, a tight-stop strategy pays a *large* cost in R
terms - often the same order of magnitude as its entire measured edge. When that's true,
"this strategy has no edge" and "this cost estimate is too harsh" produce identical
headline numbers. **The Cost Sensitivity tab exists to separate them**: it re-scores the
same trades from 0x to 2x the modelled cost and tells you the multiplier at which the
verdict flips. Those cost figures have never been validated against a real filled broker
statement - doing that once is worth more than any further backtesting.

**2. There is a random-entry control in the catalog, and it is the yardstick.**
`⊘ Random Entry (control, not a strategy)` flips a coin on each eligible bar and takes a
symmetric 1:1 bet, so it has no edge by construction. Run it over the same range as
anything else. A strategy that doesn't clearly beat it hasn't demonstrated an edge; if
*everything* clusters around it, the leaderboard is measuring costs rather than
strategies. It's seeded, so it can't be quietly re-rolled.

**3. High trade counts wreck compounded returns even at zero edge.** At 1% risk per
trade, a system with a tiny negative expectancy compounds to near -100% over tens of
thousands of trades regardless of how good the rules are. A "-98%" next to 10,000 trades
and a "-98%" next to 300 trades are not the same claim. Check trade count and the
per-trade average, not just the headline.

Related: the Compare All leaderboard applies a **multiple-comparisons correction**. Running
N strategies against the same data is N tests, so the significance bar is |z| > ~2.9-3.0 for
a catalog this size, not the familiar 1.96 - at 1.96 you'd expect ~1 in 20 to look "significant"
by chance even if every strategy were worthless. The leaderboard also flags any strategy
whose result is **not directly comparable** to the others (costs that couldn't be applied,
or a holdout that couldn't be split by time).

## Persisting history across restarts (optional, recommended if deployed)

Local disk (`webapp/run_history_data/`) is the fast read/write path for the lifetime
of one running process, but **Streamlit Community Cloud wipes it on every redeploy
and every sleep/wake cycle after inactivity** - without external storage, every saved
run (and any Gallery names you've set) is lost the next time the app restarts. This
is a platform limitation, not something local files alone can fix.

`webapp/github_storage.py` adds an optional, best-effort backing store: it pushes
every completed run to a dedicated branch of this same GitHub repo, and pulls it back
on the next cold start. Nothing here touches the app's own code or the branch it
deploys from - only that dedicated history branch. If it isn't configured, everything
falls back to local-disk-only behavior exactly as before, so this is entirely opt-in.

**One-time setup:**

1. On GitHub: **Settings -> Developer settings -> Personal access tokens ->
   Fine-grained tokens -> Generate new token**. Scope it to just this repository,
   with Repository permissions -> **Contents: Read and write**. Copy the token
   (starts with `github_pat_`) - GitHub only shows it once.
2. On Streamlit Cloud: open this app -> **Settings -> Secrets**, and add:
   ```toml
   GITHUB_TOKEN = "github_pat_..."
   GITHUB_REPO = "owner/repo"
   ```
   (e.g. `GITHUB_REPO = "stefantonatos/wind-turbone-new"`). `GITHUB_BRANCH` is
   optional and defaults to `webapp-history-data`.
3. Save - the app restarts once, and from then on every completed run is pushed to
   that branch and pulled back automatically on the next cold start. The Gallery page
   shows a warning banner whenever this isn't configured yet, so it's obvious at a
   glance whether history will actually survive a restart.

**The same setup above also speeds up every restart, not just run history.** Every
raw Dukascopy fetch this app makes is cached to local disk (`webapp/cache/`) for the
life of one running process - but that cache is wiped on the exact same redeploy/
sleep-wake cycle as everything else, which is why a "warm" app can suddenly take
minutes again after sitting idle. `data_cache.py` now pushes each fetched chunk to the
same dedicated GitHub branch (binary-safe, since a pickled price DataFrame isn't UTF-8
text - see `github_storage.read_file_bytes`/`write_file_bytes`) and pulls it back
before ever hitting Dukascopy again on a cold start. No separate setup - the same
`GITHUB_TOKEN`/`GITHUB_REPO` secrets above cover both.

**The real cap here is smaller than it looks, and it is a genuine limit rather than a
tuning knob.** GitHub's Contents API rejects anything much over ~1MB, and base64 inflates
the payload by a third on top of that, so `data_cache._MAX_GITHUB_BLOB_BYTES` sits at
700KB. Run-history JSON compresses roughly 290x and fits comfortably. Pickled float64 OHLC
frames compress about 1.2x - they are near-incompressible - so a wide date range's price
cache genuinely does not fit this backend and stays local-only, getting re-fetched on the
next cold start. Oversized content now fails fast with a distinct error before any network
call rather than burning a doomed request per chunk. Fixing this properly needs an object
store or the Git Data blobs API, not a larger constant.

## Strategies in the registry

18 total: ICT Power of Three, Scam or Slam (Day Trading Rauf), Donchian/Turtle Breakout,
MA Golden/Death Cross, Bollinger Band Mean-Reversion, RSI Mean-Reversion, Asian Range
Breakout, Dow Theory Swing Structure, Bollinger Squeeze Breakout, Climax Volume Reversal,
Support/Resistance Zone Bounce, Parabolic SAR (Stop-and-Reverse), ICT Silver Bullet,
London 3AM Range Reversal, ORB (indices), Big Daddy Max ORB + Failed-Breakout Reversal
(indices), EvenDyer VWAP ORB, and TMA Trend Scalper. Plus the random-entry control, and a
**Momentum** page outside the registry (see the top-level README).

Big Daddy Max ORB is a port of a public TradingView Pine strategy. Two things about it are
worth knowing before reading its numbers. First, the source's own published result
(+6.63%) was produced at a **fixed one-contract size while the stop distance varies with
each morning's opening range** - so it is a sum of unequal bets and cannot answer whether
the average trade made money per unit of risk. Scoring it in R, as this app does, is the
whole reason to port it rather than trust the screenshot. Second, it is really *two*
strategies sharing one script: a breakout leg and a failed-breakout reversal leg that bets
the opposite way. Every trade is tagged `continuation` or `reversal`, and the **Trade type**
filter separates them - read them apart before reading the total, or a profitable leg and a
losing one will average into a meaningless middle.

TMA Trend Scalper is also a Pine port, and porting it surfaced three things in the source
worth knowing, all reproduced rather than quietly fixed and all switchable from the
parameters. Its weekday filter (`dayofweek >= 1 and <= 5`) reads like Monday-Friday, but
Pine numbers **Sunday** as day 1 - so it trades Sunday through Thursday and **never trades
Friday**. Its entries fill at the next bar's open while the stop and target were computed
from the previous bar's close, so the advertised 1:2 risk:reward is an intention rather
than a measured property - the realised reward on winners varies well above and below 2R.
And its "minimum SMMA separation" is an absolute price number (0.001), which is ~0.009% of
EURUSD but ~0.00004% of gold, so on anything but a EUR-priced pair that filter is
effectively switched off; it is expressed here as a percentage instead. There is also no
time-based exit at all, so one position can sit open for days blocking every later signal.

Parabolic SAR is the first strategy in this catalog sourced from an actual open-source
repository (je-suis-tm/quant-trading, Apache 2.0) rather than a Pine script or a video
transcript - though even there, only the recursive SAR indicator formula itself (Wilder's
original, decades-old, public-domain algorithm) was ported; the source repo's own trading
layer is a simplified long-only demo with no stop-loss or R-multiple accounting, so the
stop-and-reverse trading logic on top is this project's own, following the standard
textbook description of "trading Parabolic SAR." Always in the market once started,
alternating long/short on every SAR flip - expect frequent small whipsaw losses punctuated
by occasional large trend-following wins, not a smooth equity curve.

EvenDyer VWAP ORB is US index CFDs only (SP500/NASDAQ100/DOWJONES) - its opening-range and
VWAP session windows are scoped to US equity trading hours, there's no forex reading of
those defaults. It enters WITH an opening-range break's direction after a VWAP
pullback-then-reclaim, with stop/target from confirmed swing pivots rather than a fixed
R:R - the first strategy in this catalog to size trades that way.

London 3AM Range Reversal is the first strategy here sourced from a video walkthrough
rather than Pine code (now two independent ones, corroborating each other) - and it
shows: several stated concepts ("relatively equal lows", "speed and distance") have no
numeric definition anywhere in either source, and one entry condition (SMT) is stated
as required but never actually defined (no named second instrument, no divergence
rule). SMT is deliberately NOT implemented - this strategy only trades the mechanical
half (a 00:00-02:00 NY dealing range, a 02:00-04:30 sweep of one side, a post-sweep
displacement, and a trade toward the FULL opposite side of the range, not just its 50%
midpoint - the second source's "50%... or the connected range low" made the full-range
target the one actually used, since it mathematically always dominates the 50% one
whenever it's reachable at all). See the script's own header for the full reasoning
before trusting its numbers as a faithful test of "the strategy" as marketed.

Three of these (Donchian, MA cross, Dow Theory) are multi-day SWING/POSITION systems on
daily channels/pivots rather than intraday, so their date-range default is ~3
years instead of the 6-month default every intraday strategy uses - these need real
history to produce more than a couple of signals, and a couple of them (MA cross, Dow
Theory) are inherently rare/selective by design (a handful to a few dozen trades per
instrument over 9 years is normal, not a bug - see each script's own header).

Bollinger Band Mean-Reversion and Bollinger Squeeze Breakout are the mechanical
OPPOSITE of each other (fade a band touch vs. trade a breakout after a volatility
squeeze) - named and captioned distinctly on purpose, don't conflate them. Climax
Volume Reversal is the one strategy on native 15-min bars (every other strategy here
uses 5-min); it also decides at run time, empirically, whether its volume condition is
usable at all (checked against the actual fetched data for every selected instrument,
dropped for the whole run if any instrument's volume field looks degenerate) - the
webapp reproduces that exact check, never bypasses it.

`trend_following_momentum_dukascopy_backtest.py` is deliberately excluded: it produces
a monthly-rebalanced portfolio return series (NAV/Sharpe/drawdown), not the R-multiple
trade list every other strategy and this whole results UI is built around.

## Layout

- `app.py` - the Streamlit app (Run Backtest, Browse Strategies, Gallery, History pages).
- `registry.py` - the only place that imports `research/*.py` modules and calls their
  existing fetch/backtest functions. Every entry documents exactly which function and
  constant names it relies on, straight from reading that module's source.
- `data_cache.py` - the generic disk cache used by any research module that doesn't
  already have its own, with the same best-effort GitHub-backed durability layered on
  top as run history (binary-safe - see `github_storage.py`'s `*_bytes` variants) so a
  Streamlit Cloud restart doesn't force a full Dukascopy re-fetch.
- `optimization.py` - surfaces the "Optimization & Robustness" tab. For the two
  strategies that currently have a real companion script (PO3, Rauf), it calls their
  actual grid-search/Monte-Carlo/cluster/walk-forward functions for real - gated behind
  an explicit "Run full optimization & robustness pass" button in the UI, since these
  are genuinely heavy (the scripts' own headers warn 30 minutes to well over an hour).
  Also surfaces a signal-decay/half-life diagnostic (`research/optimization_engine.py`'s
  `estimate_decay`, pooled across the walk-forward folds) alongside the walk-forward
  table, and a separate **Lockbox Confirmation** section - a genuine one-shot, ever,
  ledger-enforced final holdout check (`split_lockbox`/`lockbox_confirm`), gated behind
  its own explicit "I understand this can only be run ONCE" checkbox before the button
  even appears. Both companion optimization pipelines are now also windowed to exclude
  the lockbox period from their own grid search and walk-forward folds, matching the
  underlying scripts' own `main()` - otherwise running "Run full optimization pass"
  here would have already leaked the lockbox window before the user ever got to confirm
  it. The webapp's lockbox ledger lives at `webapp/run_history_data/webapp_lockbox_
  ledger.json` - deliberately NOT `research/lockbox_ledger.json` (research/ is
  read-only for this webapp, and a casual click here must never consume the real,
  canonical, one-time-ever lockbox attempt meant for an actual research run - see
  `optimization.py`'s own comment on `WEBAPP_LOCKBOX_LEDGER_PATH` for the full
  reasoning; this mirrors the project's own test suite, which does the same thing with
  a throwaway ledger path). Any strategy without a hand-wired pipeline falls back to a
  lightweight generic detector that shows a plain "not available yet" state until a
  matching `research/<strategy>_optimization.py` lands and exposes recognizable
  function names.
- `stats.py` - pure functions over an already-computed trade list (summary stats
  including max drawdown, per-instrument breakdown, R and dollar equity curves) - no
  fetching or backtesting here.
- `run_history.py` - append-only local run history (`run_history_data/runs.jsonl`
  plus one `<run_id>.trades.json` per run for full trade-level re-viewing), with
  best-effort GitHub-backed persistence layered on top - see `github_storage.py` and
  "Persisting history across restarts" above.
- `github_storage.py` - optional GitHub Contents API read/write for run history AND
  the price-data cache, entirely opt-in (falls back to local-disk-only when not
  configured). Text (`read_file`/`write_file`) and binary-safe (`read_file_bytes`/
  `write_file_bytes`) variants share the same underlying base64 content plumbing.
- `.streamlit/config.toml` + `style.py` - the visual theme: a light, near-monochrome
  palette with one accent blue, hairline borders (never drop shadows), no gradients,
  a capped left-aligned content column, and a small set of chart color constants
  (`style.ACCENT`/`CATEGORICAL`/`GOOD`/`CRITICAL`/diverging & sequential scales) used
  explicitly on every Plotly figure and styled table instead of library defaults.

## Adding a new strategy later

Add one `StrategyDef` to `registry.py`'s `STRATEGIES` list, with a small `_run_xxx`
function following the same shape as the existing ones (read the new
`research/*.py` module fully first, then wire its exact fetch/backtest function and
constant names in - never guess them). Nothing else in the app needs to change.

## Filters, on both the Results tab and History re-view

Outcome, instrument, side, any strategy-specific facet (Rauf's range, Silver Bullet's
window), and a date sub-range slider when the trades have dates - all client-side over
the trade list already produced by the last run, never a re-fetch.
