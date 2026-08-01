# Local run history - one JSON line per completed backtest run. Deliberately simple
# (JSON-lines, not a database) since this is a single-user local tool: append on write,
# read-all-then-filter on the History page. Each row is a summary; the full trade list is
# also persisted (as a sibling .trades.json file) so a past run can be re-viewed in full,
# not just its summary numbers.

import json
import os
import time
import uuid

HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_history_data")
HISTORY_FILE = os.path.join(HISTORY_DIR, "runs.jsonl")


def _trades_path(run_id):
    return os.path.join(HISTORY_DIR, f"{run_id}.trades.json")


def append_run(strategy_name, instruments, start_date, end_date, params, trades):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    n = len(trades)
    total_r = sum(t.get("r", 0.0) for t in trades)
    row = {
        "run_id": run_id,
        "timestamp": time.time(),
        "strategy": strategy_name,
        "instruments": instruments,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "params": params,
        "n_trades": n,
        "total_r": total_r,
        "avg_r": (total_r / n) if n else 0.0,
    }
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")
    with open(_trades_path(run_id), "w") as f:
        json.dump(trades, f)
    return run_id


def load_runs():
    """Newest first."""
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


def load_trades_for_run(run_id):
    path = _trades_path(run_id)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)
