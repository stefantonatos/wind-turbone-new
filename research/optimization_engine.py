# Shared, strategy-agnostic parameter-search + objective-function engine.
#
# WHY THIS EXISTS: research/ict_po3_forex_dukascopy_optimization.py and
# research/day_trading_rauf_dukascopy_optimization.py each independently
# implement the same 4-step methodology (grid search -> Monte Carlo -> cluster
# analysis -> rolling walk-forward), duplicating the grid-search loop and
# always picking the winner by raw total R. This module factors two of the
# ideas from a third-party backtesting video referenced in this project's task
# history out of that duplicated code into one reusable place:
#
#   1. more than one SEARCH STRATEGY over a parameter grid - grid (exhaustive),
#      Bayesian (Optuna/TPE), genetic (hand-rolled) - selectable per script via
#      a SEARCH_METHOD config value, instead of only ever doing a full grid
#      sweep.
#   2. a selectable OBJECTIVE FUNCTION to rank parameter combos by - total R
#      (today's default in both companion scripts, preserved as this module's
#      default too), average R/trade, a trade-based Sharpe-like ratio, Calmar
#      (return over max drawdown), or win rate - instead of always picking the
#      "best" combo by total R alone.
#
# HONEST SCOPE CAVEAT (read this before reaching for bayesian_search or
# genetic_search): both companion scripts' parameter spaces are TINY - 2
# parameters, 16-20 total combos (4x4 or 4x5). At that size, exhaustive grid
# search is not just adequate, it is the BETTER choice: it is deterministic,
# leaves no cell unexplored, and introduces no sampling noise to explain away.
# Bayesian and genetic search exist here because this project's parameter
# spaces will plausibly grow (3+ parameters, or wide continuous-like discrete
# ranges) as more strategies get their own optimization companion scripts, and
# at THAT size grid search's combo count grows multiplicatively while Bayesian
# and genetic search's evaluation budget does not have to. Using either one on
# today's 2-parameter grids is a demonstration of the mechanism, not a genuine
# improvement over grid search there - SEARCH_METHOD defaults to "grid" in
# both companion scripts for exactly this reason, and that default should stay
# "grid" until a script's parameter space actually grows past what exhaustive
# search can comfortably cover.
#
# STRATEGY-AGNOSTIC BY DESIGN: this module never imports from, or otherwise
# depends on, either companion script, Dukascopy, or any strategy's backtest
# logic. Every search function takes a generic `param_grid` (dict of
# param_name -> list of discrete candidate values) and an `eval_fn(params) ->
# list_of_trade_dicts` callback supplied by the CALLER - this module only ever
# calls that callback and scores what comes back. The companion scripts import
# FROM this module; this module imports nothing from them.

# !pip install --upgrade optuna -q   # uncomment in Colab - only needed for bayesian_search;
# grid_search and genetic_search need nothing beyond numpy (already a dependency of both
# companion scripts). Missing optuna never crashes anything here - see bayesian_search below.

import itertools
import random
import statistics

import numpy as np


# ============================================================================
# OBJECTIVE FUNCTIONS
# ============================================================================
#
# Every objective function has the signature (trades, years) -> float, where:
#   trades: a list of trade dicts, each with at least an "r" key (a float
#           R-multiple), in chronological order (the order a normal backtest
#           loop naturally produces them in - none of these functions re-sort).
#   years:  the span, in years, of the window the trades were drawn from. Only
#           `sharpe` actually uses this; the others accept and ignore it so
#           every objective can be called through the exact same signature.
#
# All of them return a plain float score that a search strategy MAXIMIZES -
# there is no "lower is better" objective in this registry; if one is ever
# added, either negate its natural sense internally or make that convention
# explicit in its own docstring.

def _r_values(trades):
    return np.asarray([t["r"] for t in trades], dtype=float)


def max_drawdown_r(r_values):
    """Largest peak-to-trough drop in the cumulative R sum along `r_values` in
    the order given (chronological, by convention). This is the SAME
    definition already used by both companion scripts' Monte Carlo code
    (monte_carlo_bootstrap/monte_carlo_shuffle in each): cumsum -> running max
    via np.maximum.accumulate -> (running_max - cumsum).max(). Deliberately
    re-implemented here rather than imported from either script, since this
    module must not depend on them - but the arithmetic is copied exactly, not
    redefined, precisely so `calmar` below means the same thing here as
    "max drawdown" already means in this project's Monte Carlo output.
    Returns 0.0 for an empty or single-trade list (see calmar's docstring for
    why a single trade is a special case, not just "small n")."""
    r = np.asarray(r_values, dtype=float)
    if len(r) == 0:
        return 0.0
    cum = np.cumsum(r)
    running_max = np.maximum.accumulate(cum)
    return float((running_max - cum).max())


def total_r(trades, years=None):
    """Sum of every trade's R-multiple. This is today's existing, already-
    verified default behavior in both companion scripts (the grid winner has
    always been picked by raw total R) - kept as this registry's default
    (OBJECTIVES["total_r"]) specifically so that behavior does not regress."""
    return float(_r_values(trades).sum()) if trades else 0.0


def avg_r(trades, years=None):
    """Mean R-multiple per trade. 0.0 for an empty trade list (no crash, no
    NaN) - an empty cell has no evidence either way, and 0.0 keeps it out of
    an optimizer's way without pretending it's the worst possible outcome."""
    if not trades:
        return 0.0
    return float(_r_values(trades).mean())


# Sentinel for sharpe's degenerate cases (fewer than 2 trades, or zero
# variance in R across every trade). A trade-based Sharpe-like ratio is
# mathematically undefined in both cases (you cannot estimate a standard
# deviation from 0 or 1 sample, and dividing by a std of exactly 0 is a
# division by zero) - rather than let either case raise, or silently produce
# NaN/inf that would then corrupt a `max(..., key=...)` comparison in a search
# strategy, both degenerate cases return this large negative sentinel. That
# specific choice (negative, not 0, not a large positive number) is
# deliberate: a search strategy MAXIMIZES this objective, so a degenerate cell
# - one with too little evidence to compute a meaningful risk-adjusted number
# at all - must never look attractive next to a real, well-evidenced cell,
# including a real cell with a merely mediocre or negative Sharpe. It is
# "we don't trust this number" pushed as far from "pick me" as reasonably
# encodable in a float, not a value that pretends to mean something risk-wise.
SHARPE_DEGENERATE_SENTINEL = -1.0e9


def sharpe(trades, years):
    """A TRADE-BASED Sharpe-like ratio - NOT a classic time-series Sharpe
    ratio. There is no equity curve sampled at fixed time intervals anywhere
    in this project's backtests, only a list of discrete trade R-multiples
    with irregular real-world spacing between them, so the classic
    (mean-return / std-return) * sqrt(periods-per-year) formula is adapted
    here to trade COUNT annualized by the window's span instead of a fixed
    sampling period:

        (mean(r) / std(r)) * sqrt(n_trades / years)

    where std(r) is the SAMPLE standard deviation (ddof=1, i.e. divided by
    n-1, the conventional choice when std is being estimated from a sample
    rather than treated as a known population parameter - consistent with how
    a real Sharpe ratio is normally computed from a finite return sample).

    Degenerate cases (see SHARPE_DEGENERATE_SENTINEL above for why this
    specific value): fewer than 2 trades (std is undefined with 0 or 1
    sample), OR std(r) == 0 (every trade had an identical R-multiple - the
    ratio's denominator is exactly zero, whether that constant R was a win or
    a loss) OR years is None/<=0 (nothing to annualize against). All three
    return SHARPE_DEGENERATE_SENTINEL rather than raising or producing NaN/inf
    - this function never crashes and never returns NaN into a comparison."""
    r = _r_values(trades)
    n = len(r)
    if n < 2 or years is None or years <= 0:
        return SHARPE_DEGENERATE_SENTINEL
    std = float(r.std(ddof=1))
    if std <= 0.0:
        return SHARPE_DEGENERATE_SENTINEL
    return float((r.mean() / std) * np.sqrt(n / years))


# Sentinels for calmar's no-drawdown edge case - see calmar's docstring.
CALMAR_NO_DRAWDOWN_POSITIVE_SENTINEL = 1.0e9
CALMAR_NO_DRAWDOWN_NEGATIVE_SENTINEL = -1.0e9


def calmar(trades, years=None):
    """total_r / max_drawdown_r, where max_drawdown_r is defined exactly as in
    max_drawdown_r() above (matching both companion scripts' existing Monte
    Carlo drawdown definition, not a new one).

    Guarded divide-by-zero: max_drawdown_r() returns exactly 0.0 whenever the
    cumulative R curve never has a single down-tick from its own running peak
    - either because every trade was a win (or breakeven) in sequence, giving
    a genuinely monotonic curve with no real drawdown to divide by, OR because
    there are 0 or 1 trades, where a "peak-to-trough drop" is trivially
    undefined/zero regardless of that one trade's sign (a single trade is
    always exactly at its own running peak by construction - this is a
    distinct edge case from "many trades with zero drawdown", not just a
    smaller version of it). Both trigger this same zero-drawdown branch, so it
    is split by the SIGN of total_r rather than by trade count, which is what
    actually distinguishes "a real, if short, winning run" from "a single (or
    zero) trade for which this ratio is simply not computable":
      - total_r > 0  -> CALMAR_NO_DRAWDOWN_POSITIVE_SENTINEL (as good as this
        ratio can express - real gains, no observed drawdown at all, whether
        that's because of genuine streak-strength or too little data to have
        dipped yet; both cases the search strategy should be free to prefer).
      - total_r < 0  -> CALMAR_NO_DRAWDOWN_NEGATIVE_SENTINEL (this only
        happens for a single losing trade, or an empty/degenerate window where
        max_drawdown_r's own single-point definition doesn't register a
        drawdown even though the outcome was a loss - never let that look like
        a "good, no-drawdown" cell).
      - total_r == 0 (includes the empty-trade-list case) -> 0.0, neutral."""
    r = _r_values(trades)
    total = float(r.sum()) if len(r) else 0.0
    dd = max_drawdown_r(r)
    if dd <= 0.0:
        if total > 0.0:
            return CALMAR_NO_DRAWDOWN_POSITIVE_SENTINEL
        if total < 0.0:
            return CALMAR_NO_DRAWDOWN_NEGATIVE_SENTINEL
        return 0.0
    return total / dd


def win_rate(trades, years=None):
    """Fraction of trades with r > 0 (breakeven, r == 0 exactly, counts as a
    non-win, matching this project's Monte Carlo convention elsewhere of
    treating "<= 0" as the not-a-win side of a threshold). 0.0 for an empty
    trade list.

    *** KNOWN TRAP - READ BEFORE SELECTING THIS OBJECTIVE ***
    Optimizing for win rate ALONE ignores payout size entirely, and this is
    not a hypothetical risk in this project - it is exactly what already
    happened here: the Day Trading Rauf base backtest
    (research/day_trading_rauf_dukascopy_backtest.py) has a real, measured
    51.6% win rate and is STILL net-negative overall, because its losers
    outsize its winners on average (see that script's and its optimization
    companion's headers for the full numbers). A parameter combo that
    maximizes win_rate can trivially do so by taking tiny, high-probability
    wins against large, rare losses and come out net-negative in total R -
    this objective exists in this registry to be SELECTABLE for exactly that
    kind of inspection (e.g. "does the highest-win-rate corner of this grid
    also happen to be profitable, or is it a payout-size trap like Rauf's
    fixed-parameter baseline already was?"), not because it is being quietly
    recommended as a good thing to optimize for on its own. Any script
    offering "win_rate" as an OBJECTIVE option must surface this same caveat
    in its own printed output when that objective is selected - the registry
    docstring alone is not enough if a user never reads this file."""
    if not trades:
        return 0.0
    r = _r_values(trades)
    return float(np.mean(r > 0.0))


OBJECTIVES = {
    "total_r": total_r,     # DEFAULT - matches both companion scripts' existing, already-verified behavior
    "avg_r": avg_r,
    "sharpe": sharpe,
    "calmar": calmar,
    "win_rate": win_rate,   # see the loud trap warning in win_rate()'s own docstring above
}

DEFAULT_OBJECTIVE = "total_r"


def get_objective(name):
    """Looks up `name` in OBJECTIVES with a clear error listing valid options
    on a miss, instead of a bare KeyError - both companion scripts resolve
    their OBJECTIVE config string through this, so a typo'd config value fails
    loudly and immediately rather than deep inside a search loop."""
    try:
        return OBJECTIVES[name]
    except KeyError:
        raise KeyError(f"Unknown objective {name!r} - choose one of {sorted(OBJECTIVES)}") from None


# ============================================================================
# SIGNIFICANCE / MULTIPLE-TESTING HELPERS
# ============================================================================
#
# These are REPORTING statistics for a verdict section, not OBJECTIVES a
# search strategy is ever handed as its objective_fn (there is no "zscore" key
# in the OBJECTIVES registry above, and there should not be one - actively
# searching a parameter grid FOR the highest z-score is exactly the kind of
# multiple-testing-blind mining bonferroni_adjusted_z_threshold below exists
# to guard against; z-score belongs downstream of a search, describing
# whatever cell was already selected by total_r/avg_r/sharpe/calmar/win_rate,
# not driving the selection itself).

def zscore(trades):
    """Sample z-score of a list of trade R-multiples: mean(r) / (std(r, ddof=1)
    / sqrt(n)), equivalently mean(r) * sqrt(n) / std(r, ddof=1).

    THIS IS A CORRECTED REPLACEMENT for a shortcut that shipped elsewhere in
    this project (research/ict_po3_forex_dukascopy_backtest.py and
    research/day_trading_rauf_dukascopy_backtest.py both compute
    `z = (total_r / n_trades) / (1 / sqrt(n_trades))`, i.e. `avg_r *
    sqrt(n_trades)` with NO division by the trades' actual sample standard
    deviation at all - equivalent to implicitly assuming std(r) == 1 exactly).
    For a stop/target payout structure clustered around -1R and some positive
    R, the real sample std is typically well above 1 (a simple 50/50 Bernoulli
    at (-1, +2) already has std ~= 1.5), so that shortcut systematically
    INFLATES every z-score computed with it by roughly that same factor. This
    function is the corrected version, used by both optimization companion
    scripts' verdict sections below - it is a straight division by the actual
    std(r, ddof=1), not a stand-in constant.

    Degenerate cases: fewer than 2 trades (std is undefined with 0 or 1
    sample) or std(r, ddof=1) == 0 (every trade had an identical R-multiple -
    the standard error is exactly 0) both return 0.0. Unlike sharpe()'s large
    negative sentinel above, 0.0 (neutral, not "actively bad") is the right
    degenerate value HERE specifically because zscore is never handed to a
    search strategy as an objective_fn to maximize - nothing is ever choosing
    "the highest z-score cell", so there is no risk of a degenerate 0.0
    looking artificially attractive next to a real result the way there would
    be if this were used as a search objective."""
    r = _r_values(trades)
    n = len(r)
    if n < 2:
        return 0.0
    std = float(r.std(ddof=1))
    if std <= 0.0:
        return 0.0
    return float(r.mean() / (std / np.sqrt(n)))


def bonferroni_adjusted_z_threshold(n_trials, family_wise_alpha=0.05):
    """The per-test two-sided |z| bar a single result needs to clear for a
    family of `n_trials` roughly-independent tests to hold an overall
    (family-wise) false-positive rate of `family_wise_alpha` (default 5%) -
    the standard Bonferroni correction, applied to a normal z-score.

    WHY THIS EXISTS: neither optimization companion script's grid search (nor
    any other fixed-rule strategy script in this project) currently corrects
    for the fact that MANY combos are being tested against the same data in
    one run (16-20 grid cells per script here alone, before counting every
    OTHER strategy script's own single-point test) - interpreting each cell's
    z-score against the naive |z| > 1.96 rule of thumb as if it were the only
    test ever run against that data is exactly the setup that produces false
    "this looks real" verdicts by chance alone (see Bailey/Borwein/Zhu/López
    de Prado's deflated Sharpe ratio / probability-of-backtest-overfitting
    literature). This helper gives each script's verdict section an honest,
    adjusted bar to compare against, using THAT RUN's own evaluated-combo
    count as `n_trials` - a defensible per-run lower bound, not a claim to
    have corrected for every strategy this project has ever tested (a fully
    project-wide correction would need a running total across every script's
    every run, which this helper deliberately leaves to the caller to supply
    if they want a stricter, cross-script `n_trials` instead).

    Formula: per_test_alpha = 1 - (1 - family_wise_alpha) ** (1 / n_trials),
    then the two-sided z bar is the standard normal's (1 - per_test_alpha / 2)
    quantile. Uses the Python standard library's statistics.NormalDist (exact,
    no extra dependency - available since Python 3.8) rather than scipy, since
    the standard library already provides this at full precision; scipy's
    scipy.stats.norm.ppf would give the identical value if a caller already
    has scipy loaded, but nothing here requires installing it.

    n_trials=1 must reduce to the ordinary single-test 1.96 bar (verified by
    test_optimization_engine.py) - the whole point of this function is that it
    generalizes, not replaces, the familiar single-test threshold."""
    if n_trials < 1:
        raise ValueError("bonferroni_adjusted_z_threshold: n_trials must be >= 1")
    if not (0.0 < family_wise_alpha < 1.0):
        raise ValueError("bonferroni_adjusted_z_threshold: family_wise_alpha must be in (0, 1)")
    per_test_alpha = 1.0 - (1.0 - family_wise_alpha) ** (1.0 / n_trials)
    return float(statistics.NormalDist().inv_cdf(1.0 - per_test_alpha / 2.0))


# ============================================================================
# SEARCH STRATEGIES
# ============================================================================
#
# All three share the same inputs and the same output shape:
#
#   param_grid: dict of param_name -> list of discrete candidate values,
#               exactly how STOP_BUFFER_PCT_GRID/FALLBACK_REWARD_RISK_GRID
#               etc. are already defined in both companion scripts - just
#               collected into one dict, e.g.
#               {"stop_buffer_pct": [...], "fallback_reward_risk": [...]}.
#   eval_fn:    params_dict -> list_of_trade_dicts. Supplied by the calling
#               script. This is the ONLY point of contact this module has with
#               a strategy's actual backtest logic - it never touches
#               Dukascopy data or backtest rules itself.
#   objective_fn: one of OBJECTIVES[...] (or any callable with the same
#               (trades, years) -> float signature).
#   years:      passed straight through to objective_fn on every evaluation.
#
# Return value (all three): a dict with
#   "method":   "grid" | "bayesian" | "genetic"
#   "all":      list of {"params": dict, "trades": [...], "score": float} -
#               every DISTINCT combo actually evaluated (grid_search: every
#               combo in param_grid, in exhaustive order; bayesian_search:
#               every trial Optuna ran; genetic_search: every distinct
#               chromosome evaluated across the whole run, deduplicated - see
#               genetic_search's own docstring)
#   "best":     the single entry of "all" with the highest score (None if
#               "all" is empty)
#   "n_evals":  len(results["all"]) - how many DISTINCT eval_fn calls this
#               search actually made; this is the number to compare across
#               methods for the efficiency claim (see print_search_comparison)
#
# genetic_search additionally returns "history": the best score seen in each
# generation, in order - a convergence trace, for a print or plot.


def grid_search(param_grid, eval_fn, objective_fn, years=None, show_progress=False, desc="grid search"):
    """Exhaustive itertools.product over every combo in param_grid, evaluated
    in the exact order a hand-written nested for-loop over the same
    param_grid (outer loop = first key, inner loop = last key) would produce
    - this generalizes both companion scripts' hardcoded 2-parameter double
    loops (`[(sb, frr) for sb in SB_GRID for frr in FRR_GRID]`) to any number
    of parameters without changing that iteration order, which is what lets
    grid_search be a drop-in replacement for those loops (see
    test_optimization_engine.py's exact-equivalence test)."""
    names = list(param_grid.keys())
    value_lists = [param_grid[name] for name in names]
    combos = list(itertools.product(*value_lists))

    iterator = combos
    if show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(combos, desc=desc, unit="combo", leave=False)
        except ImportError:
            pass

    all_results = []
    for combo in iterator:
        params = dict(zip(names, combo))
        trades = eval_fn(params)
        score = objective_fn(trades, years)
        all_results.append({"params": params, "trades": trades, "score": score})

    best = max(all_results, key=lambda r: r["score"]) if all_results else None
    return {"method": "grid", "all": all_results, "best": best, "n_evals": len(all_results)}


def bayesian_search(param_grid, eval_fn, objective_fn, years=None, n_trials=20, seed=42, show_progress=False):
    """Optuna (TPE sampler) over the SAME discrete candidate lists param_grid
    already defines - each trial samples every parameter via
    trial.suggest_categorical(name, candidate_values), never a wider
    continuous range. This is a deliberate choice, not a missed opportunity to
    "do real Bayesian optimization": the point of offering this as an
    alternative SEARCH STRATEGY is a fair, apples-to-apples comparison against
    grid_search over the identical space it exhaustively covers (fewer
    evaluations to explore the same candidate set), not a different, larger
    problem than grid_search is solving.

    Wrapped in try/except ImportError - if optuna isn't installed, prints a
    clear skip message and returns None, matching this project's existing
    sklearn/matplotlib-missing handling pattern elsewhere (see e.g.
    ict_po3_forex_dukascopy_optimization.py's sklearn_cluster_analysis). This
    function never crashes a calling script over a missing optional
    dependency - callers that get None back should fall back to grid_search
    (both companion scripts' dispatch helpers do exactly this)."""
    try:
        import optuna
    except ImportError:
        print("optuna not installed - skipping Bayesian search ('!pip install optuna -q' and "
              "re-run for this option). Falling back to grid search is the caller's responsibility "
              "(both companion scripts' SEARCH_METHOD dispatch does this automatically).")
        return None

    optuna.logging.set_verbosity(optuna.logging.WARNING)   # this project's convention is a quiet default,
                                                             # matching the DUKASCRIPT logger suppression elsewhere

    names = list(param_grid.keys())
    all_results = []
    seen = {}   # combo tuple -> index into all_results, so a trial Optuna happens to repeat
                # doesn't get scored/evaluated twice or double-counted in n_evals

    def objective(trial):
        params = {name: trial.suggest_categorical(name, param_grid[name]) for name in names}
        key = tuple(params[name] for name in names)
        if key in seen:
            return all_results[seen[key]]["score"]
        trades = eval_fn(params)
        score = objective_fn(trades, years)
        seen[key] = len(all_results)
        all_results.append({"params": params, "trades": trades, "score": score})
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)

    best = max(all_results, key=lambda r: r["score"]) if all_results else None
    return {"method": "bayesian", "all": all_results, "best": best, "n_evals": len(all_results)}


def genetic_search(param_grid, eval_fn, objective_fn, years=None, population_size=12, generations=8,
                    mutation_rate=0.15, tournament_size=3, elitism=True, seed=42, show_progress=False):
    """Hand-rolled genetic algorithm - no DEAP or other heavy GA dependency;
    this project prefers lean, well-understood custom code, and this search
    space (a handful of discrete parameters) is small enough that a simple GA
    is easy to implement correctly and easy to unit test directly.

    Chromosome = one integer gene per parameter, each gene an INDEX into that
    parameter's own candidate list (not the value itself) - e.g. for
    param_grid={"stop_buffer_pct": [0.01, 0.02, 0.05, 0.1], "fallback_reward_risk":
    [1.0, 1.5, 2.0, 2.5, 3.0]}, chromosome [2, 4] decodes to
    {"stop_buffer_pct": 0.05, "fallback_reward_risk": 3.0}.

    Per generation: tournament selection (tournament_size random contenders,
    the fittest wins) picks two parents, single-point crossover splices them
    at one random gene boundary, then random-reset mutation independently
    gives each gene a `mutation_rate` chance of being replaced with a fresh
    random index (not nudged - a full reset, so mutation can jump anywhere in
    that parameter's candidate list, not just to a neighboring value).
    Elitism: the current generation's single best chromosome is copied
    UNCHANGED into the next generation (no crossover, no mutation applied to
    it), so the best score seen so far can never regress from one generation
    to the next.

    Evaluation caching: a chromosome (by its exact index tuple) is only ever
    passed to eval_fn once per run - if tournament selection, crossover, or
    elitism produces the same chromosome again later, its score is looked up
    instead of re-evaluated. This is both a real efficiency win (fewer actual
    eval_fn calls for the same genetic search) and the reason "all"/"n_evals"
    in the return value reflect DISTINCT chromosomes evaluated, not
    population_size * generations raw slots.

    Returns the standard {"method", "all", "best", "n_evals"} shape plus
    "history": a list of length `generations`, the best score seen in that
    generation (a convergence trace - print it, or plot it, to show the
    search actually improving over generations rather than just asserting
    that it does)."""
    names = list(param_grid.keys())
    value_lists = [param_grid[name] for name in names]
    sizes = [len(v) for v in value_lists]
    if any(s == 0 for s in sizes):
        raise ValueError("genetic_search: every parameter in param_grid needs at least one candidate value")

    rng = random.Random(seed)
    all_results = []
    cache = {}   # chromosome tuple -> index into all_results

    def chromosome_to_params(chrom):
        return {name: value_lists[i][chrom[i]] for i, name in enumerate(names)}

    def score_chromosome(chrom):
        key = tuple(chrom)
        if key in cache:
            return all_results[cache[key]]["score"]
        params = chromosome_to_params(chrom)
        trades = eval_fn(params)
        score = objective_fn(trades, years)
        cache[key] = len(all_results)
        all_results.append({"params": params, "trades": trades, "score": score})
        return score

    def random_chromosome():
        return [rng.randrange(s) for s in sizes]

    def tournament_select(population, scored):
        contenders = rng.sample(range(len(population)), min(tournament_size, len(population)))
        best_i = max(contenders, key=lambda i: scored[i])
        return population[best_i]

    population = [random_chromosome() for _ in range(population_size)]
    history = []

    gen_iter = range(generations)
    if show_progress:
        try:
            from tqdm.auto import tqdm
            gen_iter = tqdm(gen_iter, desc="genetic search", unit="generation")
        except ImportError:
            pass

    for _ in gen_iter:
        scored = [score_chromosome(c) for c in population]
        best_idx = max(range(len(population)), key=lambda i: scored[i])
        best_chrom = population[best_idx]
        history.append(scored[best_idx])

        next_population = [list(best_chrom)] if elitism else []
        while len(next_population) < population_size:
            parent_a = tournament_select(population, scored)
            parent_b = tournament_select(population, scored)
            if len(names) > 1:
                cut = rng.randrange(1, len(names))
                child = parent_a[:cut] + parent_b[cut:]
            else:
                child = list(parent_a)
            child = [rng.randrange(sizes[i]) if rng.random() < mutation_rate else gene
                     for i, gene in enumerate(child)]
            next_population.append(child)

        population = next_population

    best = max(all_results, key=lambda r: r["score"]) if all_results else None
    return {"method": "genetic", "all": all_results, "best": best, "n_evals": len(all_results), "history": history}


# ============================================================================
# dispatch + comparison helpers
# ============================================================================

SEARCH_METHODS = ("grid", "bayesian", "genetic")


def run_search(method, param_grid, eval_fn, objective_fn, years=None, **kwargs):
    """Dispatches to grid_search/bayesian_search/genetic_search by name -
    shared by both companion scripts' SEARCH_METHOD config so the dispatch
    logic (including optuna's graceful fallback) lives in exactly one place.
    Unknown `method` raises ValueError immediately (loudly, at dispatch time)
    rather than silently doing nothing."""
    if method == "grid":
        return grid_search(param_grid, eval_fn, objective_fn, years=years, **kwargs)
    if method == "bayesian":
        result = bayesian_search(param_grid, eval_fn, objective_fn, years=years, **kwargs)
        if result is None:
            print("run_search: bayesian search unavailable (see message above) - falling back to grid_search "
                  "over the same param_grid so the calling script still gets a usable result.")
            return grid_search(param_grid, eval_fn, objective_fn, years=years)
        return result
    if method == "genetic":
        return genetic_search(param_grid, eval_fn, objective_fn, years=years, **kwargs)
    raise ValueError(f"Unknown SEARCH_METHOD {method!r} - choose one of {SEARCH_METHODS}")


def print_search_comparison(results_by_method, param_names=None):
    """Takes {"grid": grid_search_result, "genetic": genetic_search_result, ...}
    (any subset/superset of SEARCH_METHODS, any dict of labels -> result dicts
    from grid_search/bayesian_search/genetic_search) and prints how many
    evaluations each took and what score/params each found - the same
    "was this the same or did it drift" honesty this project's walk-forward
    step already applies to in-sample vs out-of-sample, applied here to
    search-method-vs-search-method instead."""
    print("\nSearch method comparison:")
    print(f"  {'method':<12}{'n_evals':>10}   {'best score':>12}   best params")
    for label, result in results_by_method.items():
        if result is None or result.get("best") is None:
            print(f"  {label:<12}{'--':>10}   {'no result':>12}")
            continue
        best = result["best"]
        print(f"  {label:<12}{result['n_evals']:>10}   {best['score']:>+12.4f}   {best['params']}")
