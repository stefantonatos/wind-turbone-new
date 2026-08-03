# Local run history - one JSON line per completed backtest run. Deliberately simple
# (JSON-lines, not a database) since this is a single-user local tool: append on write,
# read-all-then-filter on the History/Gallery pages. Each row is a summary; the full trade
# list is also persisted (as a sibling .trades.json file) so a past run can be re-viewed in
# full, not just its summary numbers.
#
# PERSISTENCE ACROSS RESTARTS: local disk here is the fast read/write path for the lifetime
# of one running process, but Streamlit Community Cloud wipes it on every redeploy/sleep-wake
# cycle - see github_storage.py for the optional, best-effort GitHub-repo-backed durability
# layer that survives that. Every write below also pushes to GitHub when configured; every
# read pulls from GitHub only when the local copy is missing (a cold start), so a configured
# session still does all its actual reads/writes against local disk, not the network, once
# it's warm. Never lets a GitHub failure break a real backtest result - always best-effort.

import json
import os
import time
import uuid

import github_storage
import stats as stats_mod

HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_history_data")
HISTORY_FILE = os.path.join(HISTORY_DIR, "runs.jsonl")
_GITHUB_RUNS_PATH = "webapp_history/runs.jsonl"


def _trades_path(run_id):
    return os.path.join(HISTORY_DIR, f"{run_id}.trades.json")


def _github_trades_path(run_id):
    return f"webapp_history/{run_id}.trades.json"


def _sync_runs_from_github_if_needed():
    """Pulls the runs summary file down from GitHub on a cold start (local copy missing) -
    best-effort, silently gives up on any failure so a misconfigured/unreachable GitHub never
    blocks the app from working with whatever local history it does have."""
    if os.path.isfile(HISTORY_FILE) or not github_storage.is_configured():
        return
    try:
        content = github_storage.read_file(_GITHUB_RUNS_PATH)
    except Exception as exc:
        print(f"GitHub history pull failed (continuing with local-only history): {exc}")
        return
    if content:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            f.write(content)


def _push_to_github(path, local_path, message):
    if not github_storage.is_configured():
        return
    try:
        with open(local_path) as f:
            github_storage.write_file(path, f.read(), message)
    except Exception as exc:
        # the run itself already succeeded and is saved locally for this session - a GitHub
        # sync failure (bad token, rate limit, network) must never take that away
        print(f"GitHub history push failed (run is still saved locally): {exc}")


def append_run(strategy_name, instruments, start_date, end_date, params, trades, name=None):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    n = len(trades)
    # Summary stats (what History/Gallery cards show) are computed from COST-ADJUSTED trades,
    # matching what "View full results" shows by DEFAULT (its own "Apply typical trading costs"
    # checkbox defaults to checked) - a raw/uncosted total here would show a rosier headline
    # number on the card than clicking into the very same run reveals, which is exactly the
    # "sign flips when you open it" surprise the project's own cost-modeling feature exists to
    # prevent, not a second, differently-scoped number. The raw trades themselves are still
    # stored as-is below, so the Results page's own toggle can still show the uncosted view too.
    cost_adjusted_trades, _n_unadjusted = stats_mod.apply_cost_adjustment(trades)
    total_r = sum(t.get("r", 0.0) for t in cost_adjusted_trades)
    row = {
        "run_id": run_id,
        "timestamp": time.time(),
        "name": name or f"{strategy_name} - {start_date}",
        "strategy": strategy_name,
        "instruments": instruments,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "params": params,
        "n_trades": n,
        "total_r": total_r,
        "avg_r": (total_r / n) if n else 0.0,
        "max_drawdown_r": stats_mod.max_drawdown(cost_adjusted_trades),
    }
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")
    with open(_trades_path(run_id), "w") as f:
        json.dump(trades, f)

    _push_to_github(_GITHUB_RUNS_PATH, HISTORY_FILE, f"Add run {run_id} ({strategy_name})")
    _push_to_github(_github_trades_path(run_id), _trades_path(run_id), f"Add trades for run {run_id}")
    return run_id


def load_runs():
    """Newest first."""
    _sync_runs_from_github_if_needed()
    if not os.path.isfile(HISTORY_FILE):
        return []
    rows = []
    with open(HISTORY_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
    return rows


def rename_run(run_id, new_name):
    """Rewrites the matching row's "name" field. JSON-lines has no in-place update, so this
    reads every row, patches the one that matches, and rewrites the whole file - fine at the
    scale a single personal tool's history ever reaches. Returns True if a matching row was
    found and updated, False otherwise."""
    rows = load_runs()
    found = False
    for r in rows:
        if r.get("run_id") == run_id:
            r["name"] = new_name
            found = True
            break
    if not found:
        return False
    rows.sort(key=lambda r: r.get("timestamp", 0))   # restore append order before rewriting
    with open(HISTORY_FILE, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    _push_to_github(_GITHUB_RUNS_PATH, HISTORY_FILE, f"Rename run {run_id}")
    return True


def load_trades_for_run(run_id):
    path = _trades_path(run_id)
    if not os.path.isfile(path) and github_storage.is_configured():
        try:
            content = github_storage.read_file(_github_trades_path(run_id))
            if content:
                os.makedirs(HISTORY_DIR, exist_ok=True)
                with open(path, "w") as f:
                    f.write(content)
        except Exception as exc:
            print(f"GitHub trade-detail pull failed for run {run_id}: {exc}")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)
