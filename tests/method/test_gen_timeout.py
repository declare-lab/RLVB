"""PL_GEN_TIMEOUT: the client timeout of one /generate request and its one retry (sampler.py)."""
import os, socket, sys, unittest, urllib.error
from unittest import mock

from plearn.models import sampler as S


class TimeoutTests(unittest.TestCase):
    def sampler(self, answers):
        s = object.__new__(S.MMSampler); s.base = "http://fake:1"; s.calls = []
        def post(route, payload, timeout=600):
            s.calls.append((route, timeout)); a = answers.pop(0)
            if isinstance(a, BaseException): raise a
            return a
        s._post = post
        return s

    def test_the_env_is_read(self):
        old = os.environ.pop("PL_GEN_TIMEOUT", None)
        try:
            self.assertEqual(S.gen_timeout(), 7200.0)
            os.environ["PL_GEN_TIMEOUT"] = "43200"; self.assertEqual(S.gen_timeout(), 43200.0)
            s = self.sampler([[[{"text": "abc", "finish": "stop"}]]])
            out = s.complete(["hello"], n=1)
            self.assertEqual(out, [[{"text": "abc", "finish": "stop", "logprobs": None}]]); self.assertEqual(s.calls, [("/generate", 43200.0)])
        finally:
            os.environ.pop("PL_GEN_TIMEOUT", None)
            if old is not None: os.environ["PL_GEN_TIMEOUT"] = old

    def test_one_retry_on_a_timeout(self):
        for ex in (TimeoutError("timed out"), socket.timeout("timed out"), urllib.error.URLError(socket.timeout("timed out"))):
            s = self.sampler([ex, [[{"text": "ok", "finish": "stop"}]]])
            self.assertEqual(s.complete(["hello"])[0][0]["text"], "ok"); self.assertEqual(len(s.calls), 2)
        s = self.sampler([TimeoutError("timed out"), TimeoutError("timed out"), "never"])
        with self.assertRaises(TimeoutError): s.complete(["hello"])
        self.assertEqual(len(s.calls), 2)                                        # once more, not again

    def test_no_retry_on_another_error(self):
        s = self.sampler([urllib.error.URLError(ConnectionRefusedError("refused")), "never"])
        with self.assertRaises(urllib.error.URLError): s.complete(["hello"])
        self.assertEqual(len(s.calls), 1)
        self.assertFalse(S.is_timeout(ValueError("x"))); self.assertTrue(S.is_timeout(urllib.error.URLError(TimeoutError())))


if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__": unittest.main()
