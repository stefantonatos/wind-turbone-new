# "Optimization & Robustness" tab support.
#
# Companion optimization scripts (research/<strategy>_optimization.py) are being built
# by other agents in parallel and none existed on disk as of this webapp build (checked
# at build time - see registry._find_optimization_module, which every StrategyDef in
# registry.py runs against the current research/ contents). This module defines:
#
#   - a best-effort introspection layer that, once a companion module DOES exist, looks
#     for a small set of conventional function names and calls them the same read-only
#     way every other strategy in this project is called (import + call, never edited,
#     never reloaded).
#   - a clean "not available yet" result when no companion module exists yet, or one
#     exists but doesn't expose anything this layer recognizes - never a crash, never a
#     fabricated number.
#
# WHY A DETECTOR RATHER THAN A FIXED INTEGRATION: none of the four expected outputs
# (parameter-stability heatmap, Monte Carlo percentile bands, cluster/plateau-vs-spike
# verdict, walk-forward per-fold table) can be correctly wired against a function
# signature that doesn't exist on disk yet - hardcoding a guessed name would either
# silently no-op (if wrong) or misrepresent the script's real methodology by calling
# something that isn't actually it. Once a real research/<x>_optimization.py lands, add
# its exact, VERIFIED function name (read that file fully first, same rule as every other
# module in this project) to CANDIDATE_FUNCTION_NAMES below - the loop picks it up with no
# other changes needed anywhere in this file or in app.py.

import importlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from data_cache import cached_dukascopy_fetch


@dataclass
class OptimizationResult:
    available: bool
    reason: str = ""
    heatmap: Optional[object] = None       # expected shape: pandas DataFrame, param grid x param grid -> metric
    monte_carlo: Optional[object] = None   # expected shape: dict/DataFrame of percentile bands or samples
    cluster_verdict: Optional[str] = None  # expected shape: short string verdict ("plateau" vs "spike")
    walk_forward: Optional[object] = None  # expected shape: pandas DataFrame, one row per fold + pass/fail
    extra_notes: Optional[str] = None      # combined walk-forward-efficiency summary line, when known


# Conventional names this layer looks for on a companion module, per methodology step.
# Extend this dict - don't rewrite the detection loop - once a real optimization script
# lands and its actual function name is confirmed by reading it.
CANDIDATE_FUNCTION_NAMES = {
    "heatmap": ["parameter_stability_heatmap", "build_heatmap", "grid_search_heatmap", "heatmap"],
    "monte_carlo": ["monte_carlo_simulation", "run_monte_carlo", "monte_carlo_bands", "monte_carlo"],
    "cluster_verdict": ["cluster_verdict", "plateau_vs_spike", "classify_stability"],
    "walk_forward": ["walk_forward_validation", "run_walk_forward", "walk_forward"],
}


# ---------------------------------------------------------------------------
# Real, hand-wired pipelines for the two companion optimization modules that
# actually exist on disk as of this build - read fully before wiring in, exact
# function/constant names confirmed from source (not guessed):
#
# research/ict_po3_forex_dukascopy_optimization.py exposes: INSTRUMENTS,
# FETCH_START/FETCH_END, fetch_instrument_data(label, instrument_const),
# precompute_indicators(df), run_grid_search(data) -> 20 cell dicts (keys:
# stop_buffer_pct, fallback_reward_risk, total_r, n_trades, avg_r, trades_r,
# per_instrument), build_grid_matrix(grid_results) -> (sb_list, frr_list,
# avg_r_matrix, total_r_matrix), run_monte_carlo_all_cells(grid_results) -> one
# dict per cell with "bootstrap"/"shuffle" summaries (keys: total_r_p5/p50/p95,
# dd_p5/p50/p95, p_total_r_le_0), neighbor_plateau_check(grid_results) -> dict
# (best_stop_buffer_pct, best_fallback_reward_risk, best_avg_r, verdict, ...),
# sklearn_cluster_analysis(grid_results) -> dict or None (sklearn missing),
# generate_walk_forward_folds(start_year, end_year) -> list of fold dicts,
# run_walk_forward(data, folds) -> (fold_results, combined_oos_r),
# compute_walk_forward_efficiency(fold_results, combined_oos_r) -> dict
# (combined_oos_total_r, combined_oos_n_trades, combined_oos_avg_r,
# mean_is_avg_r, wfe).
#
# research/day_trading_rauf_dukascopy_optimization.py exposes: INSTRUMENTS
# (label, dukascopy instrument CONSTANT NAME string, resolved lazily),
# FETCH_START/FETCH_END, STOP_BUFFER_PCT_GRID, CONFIRMATION_CANDLES_GRID,
# MC_ITERATIONS, fetch_instrument_data(label, instrument_const_name),
# precompute_arrays(df), and one convenience entry point,
# run_full_pipeline(all_arrays, stop_buffer_grid, confirmation_grid,
# fetch_start, fetch_end, mc_iterations=..., make_plots=..., show_progress=...,
# verbose=...) -> dict with keys grid_results, mc_results, cluster_result,
# neighbor_result, wf_result (wf_result keys: fold_rows, combined_total_r,
# combined_n_trades, combined_avg_r, mean_is_avg_r, wfe, wfe_pass).
#
# BOTH are genuinely heavy (full multi-year grid search + 2000-iteration Monte
# Carlo per cell + a re-gridded rolling walk-forward) - the modules' own
# headers warn 30 minutes to well over an hour on real data. app.py gates
# calling either of these behind an explicit "Run full optimization pass"
# button, never automatically on tab open.
# ---------------------------------------------------------------------------

def _po3_pipeline(module):
    data = {}
    with cached_dukascopy_fetch():
        for label, const in module.INSTRUMENTS:
            df = module.fetch_instrument_data(label, const)
            if df is None or df.empty:
                continue
            data[label] = module.precompute_indicators(df)
    if not data:
        return OptimizationResult(available=False, reason="no data could be fetched for any instrument")

    grid_results = module.run_grid_search(data)
    sb_list, frr_list, avg_r_matrix, _total_r_matrix = module.build_grid_matrix(grid_results)
    heatmap_df = pd.DataFrame(avg_r_matrix, index=[f"SB={v:g}" for v in sb_list],
                                columns=[f"FRR={v:g}" for v in frr_list])

    mc_results = module.run_monte_carlo_all_cells(grid_results)
    mc_rows = []
    for cell, mc in zip(grid_results, mc_results):
        boot = mc.get("bootstrap") or {}
        mc_rows.append({
            "stop_buffer_pct": cell["stop_buffer_pct"], "fallback_reward_risk": cell["fallback_reward_risk"],
            "n_trades": cell["n_trades"], "avg_r": cell["avg_r"],
            "boot_total_r_p5": boot.get("total_r_p5"), "boot_total_r_p50": boot.get("total_r_p50"),
            "boot_total_r_p95": boot.get("total_r_p95"), "p_total_r_le_0": boot.get("p_total_r_le_0"),
        })
    mc_df = pd.DataFrame(mc_rows).sort_values("avg_r", ascending=False).reset_index(drop=True)

    neighbor = module.neighbor_plateau_check(grid_results)
    cluster_verdict = (f"Best cell: STOP_BUFFER_PCT={neighbor['best_stop_buffer_pct']:g}, "
                        f"FALLBACK_REWARD_RISK={neighbor['best_fallback_reward_risk']:g} "
                        f"(avg R/trade={neighbor['best_avg_r']:+.4f}). "
                        f"{neighbor['n_decent_neighbors']}/{neighbor['n_neighbors']} immediate grid neighbors "
                        f"are decent -> {neighbor['verdict']}.")
    try:
        cluster = module.sklearn_cluster_analysis(grid_results)
        if cluster is not None:
            cluster_verdict += (f" KMeans cluster containing the best cell: {cluster['cluster_size']} of "
                                 f"{len(grid_results)} cells, mean avg R/trade={cluster['cluster_mean_avg_r']:+.4f}.")
        else:
            cluster_verdict += " (scikit-learn not installed - cluster analysis skipped, neighbor check above still stands.)"
    except Exception as exc:
        cluster_verdict += f" (cluster analysis raised {exc} - skipped.)"

    folds = module.generate_walk_forward_folds(module.FETCH_START.year, module.FETCH_END.year)
    fold_results, combined_oos_r = module.run_walk_forward(data, folds)
    wfe_stats = module.compute_walk_forward_efficiency(fold_results, combined_oos_r)
    wf_df = pd.DataFrame(fold_results)

    wfe = wfe_stats["wfe"]
    if wfe_stats["mean_is_avg_r"] == 0 or (isinstance(wfe, float) and np.isnan(wfe)):
        wfe_text = "undefined (mean in-sample avg R/trade is ~0)"
    else:
        wfe_text = f"{wfe:.3f} -> {'PASS' if wfe >= module.WFE_PASS_THRESHOLD else 'FAIL'} the >=0.5 rule of thumb"
    extra_notes = (f"Combined out-of-sample across {len(fold_results)} rolling folds: "
                   f"{wfe_stats['combined_oos_n_trades']} trades, {wfe_stats['combined_oos_total_r']:+.2f}R, "
                   f"{wfe_stats['combined_oos_avg_r']:+.4f}R/trade. Walk-Forward Efficiency = {wfe_text}. "
                   f"{module.WFE_PASS_THRESHOLD:.0%} is a standard rule-of-thumb threshold, not proof of "
                   f"robustness either way.")

    return OptimizationResult(available=True, reason="full", heatmap=heatmap_df, monte_carlo=mc_df,
                                cluster_verdict=cluster_verdict, walk_forward=wf_df, extra_notes=extra_notes)


def _rauf_pipeline(module):
    all_arrays = {}
    with cached_dukascopy_fetch():
        for label, const_name in module.INSTRUMENTS:
            df = module.fetch_instrument_data(label, const_name)
            if df is None or df.empty:
                continue
            all_arrays[label] = module.precompute_arrays(df)
    if not all_arrays:
        return OptimizationResult(available=False, reason="no data could be fetched for any instrument")

    results = module.run_full_pipeline(all_arrays, module.STOP_BUFFER_PCT_GRID, module.CONFIRMATION_CANDLES_GRID,
                                         module.FETCH_START, module.FETCH_END, mc_iterations=module.MC_ITERATIONS,
                                         make_plots=False, show_progress=False, verbose=False)
    grid_results = results["grid_results"]
    heatmap_df = pd.DataFrame([
        {"stop_buffer_pct": c["stop_buffer_pct"], "confirmation_candles": c["confirmation_candles"], "avg_r": c["avg_r"]}
        for c in grid_results
    ]).pivot(index="stop_buffer_pct", columns="confirmation_candles", values="avg_r")

    mc_rows = []
    for cell, mc in zip(grid_results, results["mc_results"]):
        boot = mc["bootstrap"]
        mc_rows.append({
            "stop_buffer_pct": cell["stop_buffer_pct"], "confirmation_candles": cell["confirmation_candles"],
            "n_trades": cell["n_trades"], "avg_r": cell["avg_r"],
            "boot_total_r_p05": boot["total_r_p05"], "boot_total_r_p50": boot["total_r_p50"],
            "boot_total_r_p95": boot["total_r_p95"], "p_total_r_leq_0": boot["p_total_r_leq_0"],
        })
    mc_df = pd.DataFrame(mc_rows).sort_values("avg_r", ascending=False).reset_index(drop=True)

    neighbor = results["neighbor_result"]
    bc = neighbor["best_cell"]
    cluster_verdict = (f"Best cell: STOP_BUFFER_PCT={bc['stop_buffer_pct']:g}, "
                        f"CONFIRMATION_CANDLES={bc['confirmation_candles']} (avg R/trade={bc['avg_r']:+.4f}). "
                        f"{neighbor['verdict']}.")
    cluster = results["cluster_result"]
    if cluster is not None:
        cluster_verdict += (f" KMeans cluster containing the best cell: {cluster['best_cluster_size']} of "
                             f"{len(grid_results)} cells, mean avg R/trade={cluster['best_cluster_mean_avg_r']:+.4f}.")
    else:
        cluster_verdict += " (scikit-learn not installed - cluster analysis skipped, neighbor check above still stands.)"

    wf = results["wf_result"]
    wf_df = pd.DataFrame(wf["fold_rows"])
    wfe = wf["wfe"]
    if isinstance(wfe, float) and np.isnan(wfe):
        wfe_text = "undefined (mean in-sample avg R/trade is ~0)"
    else:
        wfe_text = f"{wfe:.3f} -> {'PASS' if wf['wfe_pass'] else 'FAIL'} the >=0.5 rule of thumb"
    extra_notes = (f"Combined out-of-sample: {wf['combined_n_trades']} trades, {wf['combined_total_r']:+.2f}R, "
                   f"{wf['combined_avg_r']:+.4f}R/trade. Walk-Forward Efficiency = {wfe_text}. "
                   f"{module.WFE_PASS_THRESHOLD:.0%} is a standard rule-of-thumb threshold, not proof of "
                   f"robustness either way.")

    return OptimizationResult(available=True, reason="full", heatmap=heatmap_df, monte_carlo=mc_df,
                                cluster_verdict=cluster_verdict, walk_forward=wf_df, extra_notes=extra_notes)


# strategy id -> (companion module name, real pipeline function). Populated for the two
# companion scripts that exist as of this build; extend this dict (with its own small
# hand-written pipeline function above, following the same shape) the next time a new
# companion optimization script lands and its exact function names have been confirmed
# by reading it fully - do not repurpose the generic detector below for a known, heavy,
# real pipeline, since the generic path calls whatever it finds immediately with no
# "this is expensive" gate.
KNOWN_PIPELINES = {
    "po3": ("research.ict_po3_forex_dukascopy_optimization", _po3_pipeline),
    "rauf": ("research.day_trading_rauf_dukascopy_optimization", _rauf_pipeline),
}


def known_pipeline_module_name(strategy_id):
    entry = KNOWN_PIPELINES.get(strategy_id)
    return entry[0] if entry else None


def run_known_pipeline(strategy_id):
    """Actually runs the full, real, heavy 4-step pipeline for a strategy with a hand-wired
    adapter above. Caller (app.py) is responsible for gating this behind an explicit button
    and a spinner - this function does the real work and can take a long time. Never raises;
    returns an OptimizationResult with available=False and a reason on any failure."""
    entry = KNOWN_PIPELINES.get(strategy_id)
    if entry is None:
        return OptimizationResult(available=False, reason="no hand-wired pipeline for this strategy")
    module_name, pipeline_fn = entry
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return OptimizationResult(available=False, reason=f"import_failed: {exc}")
    try:
        return pipeline_fn(module)
    except Exception as exc:
        return OptimizationResult(available=False, reason=f"pipeline run failed: {exc}")


def load_optimization_result(optimization_module_name):
    """optimization_module_name is either None (the registry found no companion file for
    the currently selected strategy) or a research.<x>_optimization dotted path that DOES
    exist on disk. Returns an OptimizationResult. Never raises."""
    if optimization_module_name is None:
        return OptimizationResult(available=False, reason="no_companion_module")

    try:
        module = importlib.import_module(optimization_module_name)
    except Exception as exc:
        return OptimizationResult(available=False, reason=f"import_failed: {exc}")

    found = {}
    for step, names in CANDIDATE_FUNCTION_NAMES.items():
        for name in names:
            fn = getattr(module, name, None)
            if callable(fn):
                found[step] = fn
                break

    if not found:
        return OptimizationResult(
            available=False,
            reason=(
                f"{optimization_module_name} exists but doesn't expose any function matching "
                f"the conventional names this webapp currently knows to look for - see "
                f"webapp/optimization.py's CANDIDATE_FUNCTION_NAMES."
            ),
        )

    result = OptimizationResult(available=True, reason="partial" if len(found) < len(CANDIDATE_FUNCTION_NAMES) else "full")
    try:
        if "heatmap" in found:
            result.heatmap = found["heatmap"]()
        if "monte_carlo" in found:
            result.monte_carlo = found["monte_carlo"]()
        if "cluster_verdict" in found:
            result.cluster_verdict = found["cluster_verdict"]()
        if "walk_forward" in found:
            result.walk_forward = found["walk_forward"]()
    except Exception as exc:
        return OptimizationResult(available=False, reason=f"found matching function(s) but calling failed: {exc}")

    return result
