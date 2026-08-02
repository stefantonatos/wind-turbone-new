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
- **No commission, spread, or slippage is modeled** by any of the underlying strategies
  - this matches the caveat every research script already carries at the bottom of its
  own printed report. Real trading results would be worse than what you see here.
- Toggling a filter on the results page (outcome, instrument, side, date sub-range,
  session/range) never re-fetches data - it only recomputes stats over the trade list
  already produced by your last "Run Backtest" click.

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

## Strategies in the registry

13 total: ICT Power of Three, Scam or Slam (Day Trading Rauf), Donchian/Turtle Breakout,
MA Golden/Death Cross, Bollinger Band Mean-Reversion, RSI Mean-Reversion, Asian Range
Breakout, Dow Theory Swing Structure, Bollinger Squeeze Breakout, Climax Volume Reversal,
Support/Resistance Zone Bounce, ICT Silver Bullet, and ORB (indices).

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
  already have its own.
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
- `github_storage.py` - optional GitHub Contents API read/write for run history,
  entirely opt-in (falls back to local-disk-only when not configured).
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
