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

import hashlib
import os
import pickle
from contextlib import contextmanager

import dukascopy_python

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "dukascopy_raw")


def _cache_key(instrument, interval, offer_side, start, end):
    raw = f"{instrument}|{interval}|{offer_side}|{start.isoformat()}|{end.isoformat()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _cache_path(instrument, interval, offer_side, start, end):
    key = _cache_key(instrument, interval, offer_side, start, end)
    # keep the instrument label in the filename too, purely so the cache dir is
    # human-browsable - the hash is what actually guarantees uniqueness.
    safe_instrument = str(instrument).replace("/", "-")
    return os.path.join(CACHE_DIR, f"{safe_instrument}_{interval}_{offer_side}_{key}.pkl")


def _make_caching_fetch(original_fetch):
    def cached_fetch(instrument, interval, offer_side, start, end, *args, **kwargs):
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = _cache_path(instrument, interval, offer_side, start, end)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        df = original_fetch(instrument, interval, offer_side, start, end, *args, **kwargs)
        try:
            with open(path, "wb") as f:
                pickle.dump(df, f)
        except OSError:
            pass  # cache write failure shouldn't break the backtest itself
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
