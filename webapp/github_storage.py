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
import gzip
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


# GitHub's Contents API (the create/update-a-file endpoint used below) is documented for files up
# to 1 MB; above that the docs direct you to the Git Data blobs API instead. Exceeding it does NOT
# fail loudly in an obvious way - it comes back as a bare "422 Unprocessable Entity", which is easy
# to mistake for an auth or branch problem.
#
# THIS WAS A REAL, SILENT PRODUCTION FAILURE: with the price cache pushing one ~1.25 MB pickle per
# fetched chunk, EVERY push 422'd, forever. The app's own guard was set at 8 MB - eight times the
# real ceiling - so nothing ever tripped it, the failures only ever reached stdout, and the Gallery
# banner (which checks that a token EXISTS, not that writes SUCCEED) stayed silent. The user had
# done the token setup correctly and been told it was working while nothing was being cached at
# all, which is exactly why every restart still re-downloaded everything from scratch.
#
# Payload here is base64, which inflates raw bytes by 4/3, so the raw ceiling is ~750 KB. Held a
# little under that for the JSON envelope around it.
MAX_CONTENT_BYTES = 700 * 1024

# Content is gzipped before upload. This is close to free for the case that matters most - run
# history and trade JSON are extremely repetitive and compress by ~290x, taking a 41k-trade run
# from ~4.6 MB (hopeless) to ~16 KB (trivial). It does NOT rescue pickled float64 OHLC frames,
# which are near-incompressible (~1.2x); see data_cache.py for how that case is handled instead.
_GZIP_MAGIC = b"\x1f\x8b"


class ContentTooLargeError(ValueError):
    """Raised when content cannot fit through the Contents API even after compression. A distinct
    type so callers can tell 'this will never work, stop retrying it' apart from a transient
    network/auth failure that is worth trying again."""


def _read_content_b64(path):
    """Returns the file's raw base64 content string from the Contents API, or None if it
    doesn't exist on the history branch yet. Raises on any other failure (auth, network, etc.)
    - callers treat that as best-effort and fall back to local disk, never let a GitHub hiccup
    break the app itself."""
    repo = _repo()
    r = requests.get(f"{GITHUB_API}/repos/{repo}/contents/{path}", headers=_headers(),
                      params={"ref": _branch()}, timeout=_TIMEOUT)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()["content"]


def _maybe_gunzip(raw):
    """Transparently decompresses content written by the current code, while still reading
    anything written BEFORE compression was introduced - the small history files that were
    already uploading successfully must keep loading after this change, so the magic-byte sniff
    is a compatibility requirement, not defensive padding."""
    if raw[:2] == _GZIP_MAGIC:
        return gzip.decompress(raw)
    return raw


def read_file(path):
    """Returns the file's text content, or None if it doesn't exist on the history branch yet."""
    b64 = _read_content_b64(path)
    return None if b64 is None else _maybe_gunzip(base64.b64decode(b64)).decode("utf-8")


def read_file_bytes(path):
    """Binary-safe variant of read_file, for content that isn't valid UTF-8 text (e.g. a
    pickled price-data cache blob - see data_cache.py). Returns raw bytes, or None if the file
    doesn't exist on the history branch yet."""
    b64 = _read_content_b64(path)
    return None if b64 is None else _maybe_gunzip(base64.b64decode(b64))


def _write_raw(path, content_bytes, message):
    """Compresses, size-checks, then creates/updates `path` on the dedicated history branch.

    The size check happens AFTER compression and BEFORE any network call, so content that cannot
    possibly fit fails immediately with a clear, actionable error instead of costing a doomed
    round-trip per attempt - which, at one attempt per fetched chunk, was adding real latency to
    every single backtest run."""
    payload_bytes = gzip.compress(content_bytes, 6)
    if len(payload_bytes) > MAX_CONTENT_BYTES:
        raise ContentTooLargeError(
            f"{path}: {len(payload_bytes) / 1024:.0f} KB after compression exceeds the "
            f"{MAX_CONTENT_BYTES / 1024:.0f} KB the GitHub Contents API can accept "
            f"(raw {len(content_bytes) / 1024:.0f} KB). Not retryable - this content needs the "
            f"Git Data blobs API or a smaller payload, not another attempt.")
    _write_content_b64(path, base64.b64encode(payload_bytes).decode("ascii"), message)


def _write_content_b64(path, b64_content, message):
    """Creates or updates `path` on the dedicated history branch with already-base64-encoded
    content."""
    _ensure_branch_exists()
    repo = _repo()
    # GitHub's Contents API refuses to update an existing file without its current sha (this
    # is what makes a concurrent-write conflict loud - a stale sha gets a 409/422, not a
    # silent overwrite) - so always look it up fresh right before writing.
    existing = requests.get(f"{GITHUB_API}/repos/{repo}/contents/{path}", headers=_headers(),
                             params={"ref": _branch()}, timeout=_TIMEOUT)
    sha = existing.json()["sha"] if existing.status_code == 200 else None

    payload = {"message": message, "branch": _branch(), "content": b64_content}
    if sha:
        payload["sha"] = sha
    r = requests.put(f"{GITHUB_API}/repos/{repo}/contents/{path}", headers=_headers(),
                      json=payload, timeout=_TIMEOUT)
    r.raise_for_status()


def write_file(path, content, message):
    """Creates or updates `path` on the dedicated history branch with `content` (text)."""
    _write_raw(path, content.encode("utf-8"), message)


def write_file_bytes(path, content_bytes, message):
    """Binary-safe variant of write_file, for content that isn't UTF-8 text (e.g. a pickled
    price-data cache blob - see data_cache.py)."""
    _write_raw(path, content_bytes, message)


# Last write failure seen this session, surfaced in the UI so a silently-broken sync stops looking
# identical to a working one. is_configured() only ever answered "is a token present", which is a
# strictly weaker claim than "persistence is working" - and the gap between those two is exactly
# where the production failure lived.
_last_write_error = None


def note_write_failure(exc):
    global _last_write_error
    _last_write_error = str(exc)


def last_write_error():
    return _last_write_error


def health():
    """(ok, detail) for display. ok=False means saved data is NOT actually reaching GitHub, whether
    or not a token is configured."""
    if not is_configured():
        return False, "not configured"
    if _last_write_error:
        return False, _last_write_error
    return True, "ok"
