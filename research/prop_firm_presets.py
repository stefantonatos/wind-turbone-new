# Real, sourced prop-firm evaluation rule sets - used by prop_firm_challenge_simulator.py's
# multi-phase Monte Carlo to answer "what's my actual % chance of passing THIS firm's real
# rules, at different risk-per-trade levels" instead of one generic made-up rule set.
#
# SOURCING DISCIPLINE (same standard this project already applies to the news-calendar-filter
# investigation in orb_indices_optimization_and_ml.py - real, checkable numbers only, honest
# gaps where a number couldn't be confirmed, never a plausible-looking guess):
# - Every preset below cites the URL(s) it was checked against.
# - Every preset's `source_note` says plainly which fields are solid vs. uncertain.
# - Prop firm rules change frequently and firms run promotions/rule variants - treat these as
#   a solid STARTING POINT for the simulation, not a live-synced feed. Re-verify against the
#   firm's own current rules page before making a real decision based on this.
#
# SCOPE: only %-of-account, FX/CFD-style evaluation firms are included here (FTMO, FundedNext,
# The5ers) - they match this project's forex/CFD/index-CFD trade data and its %-of-account
# risk-sizing model. Futures-focused firms (Apex Trader Funding, TopStep) were deliberately
# left out: their rules are denominated in flat DOLLAR amounts on specific futures-contract
# account sizes, not %-of-account, and they evaluate on futures instruments this project
# doesn't backtest - forcing them into this %-based model would be exactly the kind of
# asset-class mismatch this project already got burned by once (see the Day Trading Rauf
# forex-vs-index-futures lesson in day_trading_rauf_dukascopy_backtest.py's header).
#
# Researched via web search, August 2026. Each phase's fields map directly onto
# prop_firm_challenge_simulator.py's simulate_challenge_path() parameters:
#   profit_target_pct, max_daily_loss_pct, max_overall_loss_pct, min_trading_days, drawdown_mode

PROP_FIRM_PRESETS = {
    "ftmo_2step": {
        "display_name": "FTMO Challenge (2-Step)",
        "source_urls": ["https://ftmo.com/en/trading-objectives/"],
        "source_note": (
            "Profit targets (10% / 5%), max daily loss (5%), and max drawdown (10%, trailing) are "
            "well-corroborated across FTMO's own trading-objectives page and multiple independent "
            "prop-firm rule trackers. Minimum trading days (4) is commonly cited but some sources "
            "describe it as a total-across-both-phases figure rather than per-phase - modeled here "
            "as 4 minimum days PER PHASE (the more conservative/stricter reading, i.e. this may "
            "slightly UNDERSTATE real pass probability if the true rule is 4 total). FTMO removed "
            "its overall time limit in recent years - not modeled as a constraint here either way."
        ),
        "initial_balance": 10000.0,
        "phases": [
            {"name": "Phase 1 (Challenge)", "profit_target_pct": 10.0, "max_daily_loss_pct": 5.0,
             "max_overall_loss_pct": 10.0, "min_trading_days": 4, "drawdown_mode": "trailing"},
            {"name": "Phase 2 (Verification)", "profit_target_pct": 5.0, "max_daily_loss_pct": 5.0,
             "max_overall_loss_pct": 10.0, "min_trading_days": 4, "drawdown_mode": "trailing"},
        ],
    },
    "fundednext_2step": {
        "display_name": "FundedNext Challenge (2-Step)",
        "source_urls": ["https://www.simtrade.io/prop-firms/fundednext"],
        "source_note": (
            "Profit targets (10% / 5%), daily loss limit (5%), and STATIC max drawdown (10%, "
            "floor locked at 90% of starting balance, never trails) are consistently reported. "
            "Minimum trading days (5) applies; sourced as unclear whether per-phase or total - "
            "modeled here as 5 PER PHASE (conservative/stricter reading, same caveat as FTMO "
            "above). No overall time limit modeled."
        ),
        "initial_balance": 10000.0,
        "phases": [
            {"name": "Phase 1", "profit_target_pct": 10.0, "max_daily_loss_pct": 5.0,
             "max_overall_loss_pct": 10.0, "min_trading_days": 5, "drawdown_mode": "static"},
            {"name": "Phase 2", "profit_target_pct": 5.0, "max_daily_loss_pct": 5.0,
             "max_overall_loss_pct": 10.0, "min_trading_days": 5, "drawdown_mode": "static"},
        ],
    },
    "the5ers_hypergrowth_1step": {
        "display_name": "The5ers Hyper Growth (1-Step)",
        "source_urls": ["https://the5ers.com/prop-firm-drawdown-rules-explained-daily-max-and-trailing-limits-in-2026/"],
        "source_note": (
            "Profit target (10%), daily loss limit (3%), and max drawdown (6%) for the single-step "
            "Hyper Growth program are clearly stated. TWO FIELDS ARE GENUINELY UNCERTAIN from public "
            "sources as of this research and are called out honestly rather than guessed: (1) whether "
            "the 6% max drawdown is static or trailing was not clearly confirmed - modeled here as "
            "TRAILING, the stricter/more conservative assumption, so real pass probability may be "
            "somewhat HIGHER than this simulation shows if it's actually static; (2) a minimum-trading-"
            "days figure was not found in the sources checked - modeled as 0 (no minimum). RESOLVED: "
            "The5ers' own challenge-comparison table lists 'Minimum Profitable Days: -' for Hyper "
            "Growth, confirming there is no minimum, so the 0 modeled here is correct rather than an "
            "assumption. The static-vs-trailing question (1) remains open. Re-verify against "
            "The5ers' current rules page before relying on this preset for a real decision."
        ),
        "initial_balance": 10000.0,
        "phases": [
            {"name": "Single Step", "profit_target_pct": 10.0, "max_daily_loss_pct": 3.0,
             "max_overall_loss_pct": 6.0, "min_trading_days": 0, "drawdown_mode": "trailing"},
        ],
    },
    "the5ers_highstakes_2step": {
        "display_name": "The5ers High Stakes (2-Step)",
        "source_urls": ["https://the5ers.com/"],
        "source_note": (
            "From The5ers' own challenge-comparison table: profit targets 8% then 5%, daily loss "
            "('Daily Pause Limit') 5% in both phases, max drawdown 10% in both phases, and a minimum "
            "of 3 PROFITABLE days per phase. Two caveats stated rather than hidden: (1) the table "
            "says 'Minimum Profitable Days', which is a stricter condition than the 'minimum trading "
            "days' this simulator models - a day with a small loss counts toward a trading-day "
            "requirement but NOT toward a profitable-day one, so pass probability here may be "
            "somewhat OVERSTATED; (2) static-vs-trailing drawdown is not stated in that table and is "
            "modeled as TRAILING (the stricter reading), same convention as the Hyper Growth preset "
            "above. Note this program disallows news trading, which this simulator does not model at "
            "all - a strategy that trades through releases would face a rule this simulation ignores."
        ),
        "initial_balance": 10000.0,
        "phases": [
            {"name": "Phase 1", "profit_target_pct": 8.0, "max_daily_loss_pct": 5.0,
             "max_overall_loss_pct": 10.0, "min_trading_days": 3, "drawdown_mode": "trailing"},
            {"name": "Phase 2", "profit_target_pct": 5.0, "max_daily_loss_pct": 5.0,
             "max_overall_loss_pct": 10.0, "min_trading_days": 3, "drawdown_mode": "trailing"},
        ],
    },
    "the5ers_bootcamp_3step": {
        "display_name": "The5ers Bootcamp (3-Step)",
        "source_urls": ["https://the5ers.com/"],
        "source_note": (
            "From The5ers' own challenge-comparison table: 6% profit target in each of three phases, "
            "5% max drawdown in each, and no minimum profitable-days requirement. THE DAILY LIMIT IS "
            "THE UNCERTAIN FIELD: the table shows a dash for 'Daily Pause Limit' across all three "
            "phases, which most plausibly means this program has none - modeled here as effectively "
            "unlimited (set equal to the 5% max drawdown, so the overall limit binds first and the "
            "daily one never independently fails a path). If a daily limit does in fact exist, this "
            "preset OVERSTATES pass probability. Verify before relying on it. Structurally this is "
            "the most demanding of the three programs despite the lowest per-phase target: three "
            "sequential phases each with only 5% of drawdown room means three independent chances to "
            "be knocked out, and pass probabilities multiply."
        ),
        "initial_balance": 10000.0,
        "phases": [
            {"name": "Phase 1", "profit_target_pct": 6.0, "max_daily_loss_pct": 5.0,
             "max_overall_loss_pct": 5.0, "min_trading_days": 0, "drawdown_mode": "trailing"},
            {"name": "Phase 2", "profit_target_pct": 6.0, "max_daily_loss_pct": 5.0,
             "max_overall_loss_pct": 5.0, "min_trading_days": 0, "drawdown_mode": "trailing"},
            {"name": "Phase 3", "profit_target_pct": 6.0, "max_daily_loss_pct": 5.0,
             "max_overall_loss_pct": 5.0, "min_trading_days": 0, "drawdown_mode": "trailing"},
        ],
    },
}


def get_preset(preset_id):
    """Returns a deep-enough copy (the phases list is rebuilt fresh) so a caller mutating the
    returned dict's phase entries never corrupts the shared PROP_FIRM_PRESETS module state."""
    preset = PROP_FIRM_PRESETS[preset_id]
    return {
        "display_name": preset["display_name"],
        "source_urls": list(preset["source_urls"]),
        "source_note": preset["source_note"],
        "initial_balance": preset["initial_balance"],
        "phases": [dict(phase) for phase in preset["phases"]],
    }


def list_presets():
    """Returns [(preset_id, display_name), ...] in a fixed, stable order - for populating a
    dropdown/selector without depending on dict ordering guarantees across Python versions."""
    return [(preset_id, PROP_FIRM_PRESETS[preset_id]["display_name"])
            for preset_id in ("ftmo_2step", "fundednext_2step", "the5ers_hypergrowth_1step")]
