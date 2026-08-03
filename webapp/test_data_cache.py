import unittest

import requests

import data_cache


class TestDukascopyTimeoutPatch(unittest.TestCase):
    """dukascopy_python._fetch() calls requests.get() with no timeout= at all, so a stalled
    connection blocks the whole Streamlit run forever - confirmed live on the deployed app,
    stuck on one strategy for 10+ minutes. data_cache patches a default timeout into
    requests.get() at import time; these tests exercise that patch without touching the
    network by swapping out data_cache._original_requests_get for the duration of the call.
    """

    def setUp(self):
        self.captured = {}
        self._real_original = data_cache._original_requests_get

        def fake_get(*args, **kwargs):
            self.captured["args"] = args
            self.captured["kwargs"] = kwargs
            return "fake-response"

        data_cache._original_requests_get = fake_get

    def tearDown(self):
        data_cache._original_requests_get = self._real_original

    def test_default_timeout_injected_when_caller_omits_one(self):
        result = requests.get("http://example.com")
        self.assertEqual(result, "fake-response")
        self.assertEqual(self.captured["kwargs"].get("timeout"), data_cache._DUKASCOPY_HTTP_TIMEOUT_S)

    def test_explicit_caller_timeout_is_not_overridden(self):
        # github_storage.py always passes its own timeout= explicitly - the patch must leave
        # that alone rather than clobbering it with the dukascopy-specific default.
        requests.get("http://example.com", timeout=5)
        self.assertEqual(self.captured["kwargs"].get("timeout"), 5)

    def test_requests_get_is_actually_patched(self):
        self.assertIs(requests.get, data_cache._requests_get_with_default_timeout)


if __name__ == "__main__":
    unittest.main()
