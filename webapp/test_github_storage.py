# Tests for the GitHub-backed persistence layer, against a fake Contents API.
#
# WHY THESE EXIST: this layer was completely broken in production for weeks - every price-cache
# write returned 422 because the payload exceeded GitHub's ~1MB Contents-API ceiling while the
# app's own guard was set at 8MB - and NOTHING caught it. Not the 433 research tests, not the
# webapp tests, not the UI (which reported "configured" and therefore looked healthy), because the
# only signal was a print() to a log nobody reads. Every test below pins one of the specific
# properties whose absence made that failure invisible.
#
# Run with:  python -m pytest webapp/test_github_storage.py -v

import base64
import gzip
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import github_storage as gs


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeGitHub:
    """Minimal stand-in for the Contents API: remembers what was PUT so a test can assert on the
    bytes that actually went over the wire, and enforces the same size ceiling the real API does."""

    def __init__(self, ceiling_bytes=1024 * 1024):
        self.files = {}
        self.ceiling_bytes = ceiling_bytes
        self.put_calls = 0

    def get(self, url, **kw):
        if "/branches/" in url:
            return _FakeResponse(200)
        path = url.split("/contents/", 1)[1] if "/contents/" in url else None
        if path in self.files:
            return _FakeResponse(200, {"content": self.files[path], "sha": "deadbeef"})
        return _FakeResponse(404)

    def put(self, url, json=None, **kw):
        self.put_calls += 1
        content = json["content"]
        if len(content) > self.ceiling_bytes:
            return _FakeResponse(422)   # exactly how the real API rejected oversized content
        self.files[url.split("/contents/", 1)[1]] = content
        return _FakeResponse(200)


def _configured(fake):
    """Patches secrets + requests so github_storage talks to `fake` instead of the network."""
    return mock.patch.multiple(
        gs,
        _secret=lambda name, default=None: {"GITHUB_TOKEN": "t", "GITHUB_REPO": "o/r"}.get(name, default),
        requests=mock.Mock(get=fake.get, put=fake.put, post=lambda *a, **k: _FakeResponse(201)),
    )


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        gs._last_write_error = None

    def test_text_round_trips(self):
        fake = _FakeGitHub()
        with _configured(fake):
            gs.write_file("runs.jsonl", '{"run_id": "abc"}', "msg")
            self.assertEqual(gs.read_file("runs.jsonl"), '{"run_id": "abc"}')

    def test_bytes_round_trip_exactly(self):
        fake = _FakeGitHub()
        blob = bytes(range(256)) * 40
        with _configured(fake):
            gs.write_file_bytes("cache/x.pkl", blob, "msg")
            self.assertEqual(gs.read_file_bytes("cache/x.pkl"), blob)

    def test_missing_file_reads_as_none_not_an_error(self):
        with _configured(_FakeGitHub()):
            self.assertIsNone(gs.read_file("nope.json"))
            self.assertIsNone(gs.read_file_bytes("nope.pkl"))


class TestCompression(unittest.TestCase):
    """Compression is what makes large run histories fit at all - a 41k-trade run is ~4.6MB of
    JSON (hopeless) but ~16KB gzipped (trivial)."""

    def test_content_is_actually_stored_compressed(self):
        fake = _FakeGitHub()
        with _configured(fake):
            gs.write_file("runs.jsonl", "x" * 100_000, "msg")
        stored = base64.b64decode(fake.files["runs.jsonl"])
        self.assertEqual(stored[:2], gs._GZIP_MAGIC)
        self.assertLess(len(stored), 5_000, "highly repetitive text should compress hard")

    def test_a_realistically_large_trade_history_now_fits(self):
        # the exact case that used to 422: a run with tens of thousands of trades
        payload = '{"side":"LONG","outcome":"SL","r":-1.0,"date":"2024-06-01"},' * 41_000
        fake = _FakeGitHub()
        with _configured(fake):
            gs.write_file("trades/big.json", payload, "msg")
            self.assertEqual(gs.read_file("trades/big.json"), payload)

    def test_still_reads_uncompressed_files_written_before_this_change(self):
        # small history files uploaded successfully pre-compression and must keep loading
        fake = _FakeGitHub()
        fake.files["legacy.json"] = base64.b64encode(b'{"legacy": true}').decode("ascii")
        with _configured(fake):
            self.assertEqual(gs.read_file("legacy.json"), '{"legacy": true}')


class TestOversizedContentFailsFastAndClearly(unittest.TestCase):
    """The original failure mode: content that can never fit, retried forever, one doomed network
    round-trip at a time, reported only to stdout."""

    def setUp(self):
        gs._last_write_error = None

    def test_incompressible_oversized_content_raises_before_any_network_call(self):
        fake = _FakeGitHub()
        blob = os.urandom(2 * 1024 * 1024)   # random => gzip cannot shrink it
        with _configured(fake):
            with self.assertRaises(gs.ContentTooLargeError):
                gs.write_file_bytes("cache/huge.pkl", blob, "msg")
        self.assertEqual(fake.put_calls, 0,
                         "must fail on the size check, not by burning a doomed request")

    def test_error_names_the_sizes_so_it_is_actionable(self):
        with _configured(_FakeGitHub()):
            try:
                gs.write_file_bytes("cache/huge.pkl", os.urandom(2 * 1024 * 1024), "msg")
                self.fail("expected ContentTooLargeError")
            except gs.ContentTooLargeError as exc:
                self.assertIn("KB", str(exc))
                self.assertIn("Not retryable", str(exc))

    def test_too_large_is_distinguishable_from_a_transient_failure(self):
        # callers latch on ContentTooLargeError (stop trying) but retry generic failures
        self.assertTrue(issubclass(gs.ContentTooLargeError, ValueError))
        self.assertFalse(issubclass(RuntimeError, gs.ContentTooLargeError))


class TestHealthReporting(unittest.TestCase):
    """is_configured() answers 'is a token present', which is strictly weaker than 'persistence
    works'. Conflating them is what let a fully broken sync look fine in the UI."""

    def setUp(self):
        gs._last_write_error = None

    def test_healthy_when_configured_and_nothing_has_failed(self):
        with _configured(_FakeGitHub()):
            self.assertEqual(gs.health(), (True, "ok"))

    def test_unhealthy_after_a_write_failure_even_though_still_configured(self):
        with _configured(_FakeGitHub()):
            gs.note_write_failure("price cache too large")
            ok, detail = gs.health()
            self.assertTrue(gs.is_configured(), "token is still present...")
            self.assertFalse(ok, "...but persistence is NOT working, and health must say so")
            self.assertIn("too large", detail)

    def test_unconfigured_is_reported_as_unhealthy_too(self):
        with mock.patch.object(gs, "_secret", lambda name, default=None: default):
            self.assertEqual(gs.health(), (False, "not configured"))


if __name__ == "__main__":
    unittest.main()
