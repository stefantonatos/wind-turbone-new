# Generic disk cache for raw dukascopy_python.fetch() calls.
#
# This is the fallback cache used for any research/*.py module that does NOT already
# have its own disk caching (e.g. day_trading_rauf_dukascopy_backtest.py already caches
# to disk itself - see its CACHE_DIR/fetch_instrument_data - so this layer is redundant
# but harmless for that one module; it's load-bearing for every other module).
#
# HOW IT WORKS: every research module in this project calls `dukascopy_python.fetch(...)`
# as a plain attribute access on the shared `dukascopy_python` module object (not
# `from dukascopy_python import fetch`), so a single process-wide monkeypatch of
# `dukascopy_python.fetch` transparently intercepts the call no matter which research
# module makes it, or how deep inside that module's own functions the call happens
# (e.g. orb_indices_dukascopy_backtest.py's backtest_index() fetches internally with no
# separate fetch function to call instead). The patch is applied only for the duration
# of a single "Run Backtest" click (see the `cached_dukascopy_fetch` context manager
# below) and always restored afterwards, so it never leaks into anything else running
# in the same process.
#
# Cache key = (instrument, interval, offer_side, start, end) - the exact same tuple
# that would otherwise hit Dukascopy again. Nothing here touches research/*.py.
#
# PERSISTENCE ACROSS RESTARTS: local disk here is the fast path for the lifetime of one
# running process, but Streamlit Community Cloud wipes it on every redeploy/sleep-wake cycle -
# the exact same problem run_history.py already solved for backtest history, via
# github_storage.py's optional, best-effort GitHub-repo-backed durability layer. This cache
# reuses that SAME layer (github_storage.read_file_bytes/write_file_bytes - binary-safe
# variants, since a pickled DataFrame isn't UTF-8 text) rather than inventing a second
# persistence mechanism: on a local cache miss, try pulling the blob from GitHub before ever
# hitting Dukascopy for real; after a genuine fetch, push the result back so the NEXT cold
# start (not this one) is fast. Every research script that chunks its own fetches into
# multi-month pieces (the "own cache" family - Rauf, Donchian, MA Cross, Bollinger, RSI, Asian
# Range, Dow Theory, Bollinger Squeeze, Climax Volume, VWAP ORB) already routes through THIS
# exact monkeypatch as a backstop (see each runner's `with cached_dukascopy_fetch():` in
# registry.py) - which conveniently means those scripts' own combined per-instrument pickles
# don't need their own separate GitHub sync: even though THAT top-level file is empty after a
# restart, every chunk it re-fetches underneath hits this GitHub-backed cache instead of real
# Dukascopy, so the net effect is the same fast warm-start without touching research/*.py at
# all.
#
# HONEST LIMIT ON THAT LAST PARAGRAPH: GitHub's Contents API tops out around 1MB per file, and a
# pickled float64 OHLC frame is near-incompressible (gzip buys ~1.2x, versus ~290x on the JSON that
# run history stores). A 3-month chunk of 5-min bars is ~1.25MB raw, i.e. genuinely too big. So on
# narrow ranges the price cache does persist across restarts, and on wide ones it does not - the
# chunks simply exceed what this storage backend can hold. That is a real, bounded limitation of
# using a git host as a blob store, not a bug to be tuned away, and the app now says so in the UI
# (github_storage.health()) instead of reporting "configured" and appearing to work. If durable
# wide-range caching becomes worth it, the fix is a real object store (S3/R2/GCS) or GitHub's Git
# Data blobs API, NOT a larger cap here.

import functools
import hashlib
import os
import pickle
from contextlib import contextmanager

import dukascopy_python
import requests

import github_storage

# dukascopy_python's own _fetch() calls requests.get(...) with no timeout= at all, and
# requests has no default timeout of its own - if Dukascopy's server accepts the connection
# but then stalls (never finishes the response), that single call blocks forever with no
# exception ever raised, so _stream()'s own retry loop (which only triggers on an exception)
# never kicks in either. From the outside this looks exactly like "stuck" with no way to
# recover short of manually rebooting the whole app - confirmed live on the deployed app,
# stuck on one strategy for 10+ minutes with a static progress bar. Patched here rather than
# in the vendored package: only fills in a default when a caller didn't already pass timeout=
# (github_storage.py's own requests calls already set one explicitly and are unaffected), so
# a genuinely stalled connection now fails after a bounded time and surfaces as a real error
# instead of hanging the whole run.
_DUKASCOPY_HTTP_TIMEOUT_S = 30
_original_requests_get = requests.get


@functools.wraps(_original_requests_get)
def _requests_get_with_default_timeout(*args, **kwargs):
    kwargs.setdefault("timeout", _DUKASCOPY_HTTP_TIMEOUT_S)
    return _original_requests_get(*args, **kwargs)


requests.get = _requests_get_with_default_timeout

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "dukascopy_raw")
_GITHUB_CACHE_PREFIX = "webapp_price_cache"
_MAX_GITHUB_BLOB_BYTES = 700 * 1024   # must track github_storage.MAX_CONTENT_BYTES - the previous
                                       # 8MB value was 8x GitHub's real Contents-API ceiling, so it
                                       # never rejected anything and every push silently 422'd

# Set once per session if GitHub rejects a price-cache blob as unfittable, to stop re-attempting a
# write that provably cannot succeed (see the push site below for why this matters for latency).
_github_push_disabled_reason = None


def _cache_key(instrument, interval, offer_side, start, end):
    raw = f"{instrument}|{interval}|{offer_side}|{start.isoformat()}|{end.isoformat()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _cache_path(instrument, interval, offer_side, start, end):
    key = _cache_key(instrument, interval, offer_side, start, end)
    # keep the instrument label in the filename too, purely so the cache dir is
    # human-browsable - the hash is what actually guarantees uniqueness.
    safe_instrument = str(instrument).replace("/", "-")
    return os.path.join(CACHE_DIR, f"{safe_instrument}_{interval}_{offer_side}_{key}.pkl")


def _github_cache_path(instrument, interval, offer_side, start, end):
    key = _cache_key(instrument, interval, offer_side, start, end)
    return f"{_GITHUB_CACHE_PREFIX}/{key}.pkl"


def _make_caching_fetch(original_fetch):
    def cached_fetch(instrument, interval, offer_side, start, end, *args, **kwargs):
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = _cache_path(instrument, interval, offer_side, start, end)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        # local cache is cold (a fresh process - e.g. right after a Streamlit Cloud restart
        # wiped local disk) - try the GitHub-backed copy before ever hitting Dukascopy for
        # real. Best-effort: any failure here just falls through to the real fetch below,
        # exactly like an unconfigured/never-populated cache would.
        github_path = _github_cache_path(instrument, interval, offer_side, start, end)
        if github_storage.is_configured():
            blob = None
            try:
                blob = github_storage.read_file_bytes(github_path)
            except Exception as exc:
                print(f"Price-cache GitHub pull failed (falling back to a real fetch): {exc}")
            if blob is not None:
                try:
                    df = pickle.loads(blob)
                except Exception:
                    df = None   # corrupt/unreadable cached blob - fall through to a real fetch
                if df is not None:
                    try:
                        with open(path, "wb") as f:
                            f.write(blob)
                    except OSError:
                        pass
                    return df

        df = original_fetch(instrument, interval, offer_side, start, end, *args, **kwargs)
        blob = pickle.dumps(df)
        try:
            with open(path, "wb") as f:
                f.write(blob)
        except OSError:
            pass  # cache write failure shouldn't break the backtest itself

        # Push to GitHub only while it's still plausibly working. A pickled float64 OHLC frame is
        # near-incompressible, so a wide chunk genuinely cannot fit through the Contents API - and
        # when that's the case it will be the case for EVERY chunk in the run. Previously each one
        # still paid a GET + a doomed PUT (2 round-trips x ~144 chunks for a 4-instrument strategy)
        # and printed an identical error, which is both pure latency and pure noise. One
        # ContentTooLargeError now disables price-cache pushes for the rest of the session with a
        # single clear message; genuinely transient failures (network, auth) are NOT latched, since
        # those are worth retrying.
        global _github_push_disabled_reason
        if (github_storage.is_configured() and _github_push_disabled_reason is None
                and len(blob) <= _MAX_GITHUB_BLOB_BYTES):
            try:
                github_storage.write_file_bytes(github_path, blob,
                                                  f"Cache {instrument} {interval} "
                                                  f"{start.date()}-{end.date()}")
            except github_storage.ContentTooLargeError as exc:
                _github_push_disabled_reason = str(exc)
                github_storage.note_write_failure(
                    f"price cache too large for GitHub storage - price data will NOT persist across "
                    f"restarts (run history still will). {exc}")
                print(f"Price-cache GitHub push disabled for this session: {exc}")
            except Exception as exc:
                github_storage.note_write_failure(f"price cache push failed: {exc}")
                print(f"Price-cache GitHub push failed (still cached locally this session): {exc}")

        return df

    return cached_fetch


@contextmanager
def cached_dukascopy_fetch():
    """Monkeypatches dukascopy_python.fetch with a disk-caching wrapper for the
    duration of the `with` block, then restores the original. Use this around any
    call into a research module that ends up calling dukascopy_python.fetch,
    directly or indirectly."""
    original = dukascopy_python.fetch
    dukascopy_python.fetch = _make_caching_fetch(original)
    try:
        yield
    finally:
        dukascopy_python.fetch = original


def cache_size_summary():
    """Small helper for the UI - how much is cached on disk right now."""
    if not os.path.isdir(CACHE_DIR):
        return 0, 0.0
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pkl")]
    total_bytes = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files)
    return len(files), total_bytes / (1024 * 1024)
