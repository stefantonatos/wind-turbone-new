# Optional GitHub-repo-backed persistence for run history, so backtest history survives
# Streamlit Cloud restarts. Streamlit Community Cloud wipes the app's local disk on every
# redeploy AND on every sleep/wake cycle after inactivity - local JSON files alone (see
# run_history.py) do not survive that. Everything in this module is BEST-EFFORT and OPTIONAL:
# if the secrets below aren't configured, is_configured() returns False and run_history.py
# falls straight back to pure local-disk storage, exactly like before this module existed - so
# this is always safe to import, even before the one-time setup below is done.
#
# ONE-TIME SETUP (do this once in the Streamlit Cloud dashboard, not in code):
#   1. On GitHub: Settings -> Developer settings -> Personal access tokens -> Fine-grained
#      tokens -> Generate new token. Scope it to just this repository, with Repository
#      permissions -> Contents: Read and write. Copy the token (starts with "github_pat_" or
#      "ghp_") - GitHub only shows it once.
#   2. On Streamlit Cloud: open this app -> Settings -> Secrets, and add:
#        GITHUB_TOKEN = "<the token from step 1>"
#        GITHUB_REPO = "<owner>/<repo>"      # e.g. "stefantonatos/wind-turbone-new"
#      GITHUB_BRANCH is optional (defaults to "webapp-history-data" below) - a DEDICATED
#      branch, separate from whatever branch the app code itself deploys from, so every
#      history-save commit doesn't clutter the app's own commit history or trigger a redeploy.
#   3. Save - the app restarts once, and from then on every completed backtest run is pushed
#      to that branch and pulled back on the next cold start.
#
# Nothing here ever touches the app's OWN code or the branch it deploys from - only the
# dedicated history branch/paths below.

import base64
import os

import requests
import streamlit as st

GITHUB_API = "https://api.github.com"
DEFAULT_BRANCH = "webapp-history-data"
_TIMEOUT = 15


def _secret(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass   # st.secrets raises if no secrets.toml exists at all - treat that as "not configured"
    return os.environ.get(name, default)


def is_configured():
    return bool(_secret("GITHUB_TOKEN") and _secret("GITHUB_REPO"))


def _headers():
    return {"Authorization": f"Bearer {_secret('GITHUB_TOKEN')}", "Accept": "application/vnd.github+json"}


def _repo():
    return _secret("GITHUB_REPO")


def _branch():
    return _secret("GITHUB_BRANCH", DEFAULT_BRANCH)


def _ensure_branch_exists():
    """Creates the dedicated history branch off the repo's current default branch, if it
    doesn't exist yet. Idempotent - a second call once the branch exists is a cheap no-op
    (one GET that returns 200)."""
    repo = _repo()
    branch = _branch()
    r = requests.get(f"{GITHUB_API}/repos/{repo}/branches/{branch}", headers=_headers(), timeout=_TIMEOUT)
    if r.status_code == 200:
        return
    repo_info = requests.get(f"{GITHUB_API}/repos/{repo}", headers=_headers(), timeout=_TIMEOUT)
    repo_info.raise_for_status()
    default_branch = repo_info.json()["default_branch"]
    ref = requests.get(f"{GITHUB_API}/repos/{repo}/git/ref/heads/{default_branch}",
                        headers=_headers(), timeout=_TIMEOUT)
    ref.raise_for_status()
    sha = ref.json()["object"]["sha"]
    create = requests.post(f"{GITHUB_API}/repos/{repo}/git/refs", headers=_headers(),
                            json={"ref": f"refs/heads/{branch}", "sha": sha}, timeout=_TIMEOUT)
    if create.status_code not in (201, 422):   # 422 = already exists (race with another process) - fine
        create.raise_for_status()


def read_file(path):
    """Returns the file's text content, or None if it doesn't exist on the history branch yet.
    Raises on any other failure (auth, network, etc.) - callers treat that as best-effort and
    fall back to local disk, never let a GitHub hiccup break the app itself."""
    repo = _repo()
    r = requests.get(f"{GITHUB_API}/repos/{repo}/contents/{path}", headers=_headers(),
                      params={"ref": _branch()}, timeout=_TIMEOUT)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    return base64.b64decode(data["content"]).decode("utf-8")


def write_file(path, content, message):
    """Creates or updates `path` on the dedicated history branch with `content` (text)."""
    _ensure_branch_exists()
    repo = _repo()
    # GitHub's Contents API refuses to update an existing file without its current sha (this
    # is what makes a concurrent-write conflict loud - a stale sha gets a 409/422, not a
    # silent overwrite) - so always look it up fresh right before writing.
    existing = requests.get(f"{GITHUB_API}/repos/{repo}/contents/{path}", headers=_headers(),
                             params={"ref": _branch()}, timeout=_TIMEOUT)
    sha = existing.json()["sha"] if existing.status_code == 200 else None

    payload = {"message": message, "branch": _branch(),
               "content": base64.b64encode(content.encode("utf-8")).decode("ascii")}
    if sha:
        payload["sha"] = sha
    r = requests.put(f"{GITHUB_API}/repos/{repo}/contents/{path}", headers=_headers(),
                      json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
