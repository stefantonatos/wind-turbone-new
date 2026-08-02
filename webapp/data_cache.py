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
# all. _MAX_GITHUB_BLOB_BYTES caps what gets pushed - a wide-range single-shot fetch (the
# non-chunked strategies, on a manually widened date range) can produce a genuinely large
# blob; rather than fail or silently skip it, that one entry just stays local-only for this
# process and gets re-fetched for real next cold start, same as before this feature existed.

import hashlib
import os
import pickle
from contextlib import contextmanager

import dukascopy_python

import github_storage

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "dukascopy_raw")
_GITHUB_CACHE_PREFIX = "webapp_price_cache"
_MAX_GITHUB_BLOB_BYTES = 8 * 1024 * 1024   # ~8MB raw pickle - comfortably under the Contents
                                             # API's practical single-PUT size before base64
                                             # inflation (~33%) pushes it toward GitHub's own
                                             # limits; a 3-month 5-min chunk for one instrument
                                             # (the common case) is a small fraction of this.


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

        if github_storage.is_configured() and len(blob) <= _MAX_GITHUB_BLOB_BYTES:
            try:
                github_storage.write_file_bytes(github_path, blob,
                                                  f"Cache {instrument} {interval} "
                                                  f"{start.date()}-{end.date()}")
            except Exception as exc:
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
