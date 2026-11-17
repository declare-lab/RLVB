"""The context guard of VLLMSampler.complete() (without it, a belief-line request with a fixed
max_tokens 160 on a prompt near the context end fails with vLLM HTTP 400). A prompt that fills the context gives a cut, empty completion
(finish "length") with no request; a fixed max_tokens is clipped to the room that is left; a short prompt is sent as before, with no
/tokenize request. A fake _post stands in for the server, so no vLLM is needed."""
import io
import os
import unittest
import urllib.error

from plearn.models.sampler import RemoteVLLMSampler


class Fake(RemoteVLLMSampler):
    def __init__(self, L, count):
        os.environ.pop("PL_MAX_MODEL_LEN", None)
        super().__init__("http://x:1", max_model_len=L); self.count = count; self.calls = []

    def _post(self, route, payload, timeout=600):
        self.calls.append((route, payload.get("max_tokens")))
        if route == "/tokenize": return {"count": self.count}
        return {"choices": [{"text": "a", "finish_reason": "stop"}] * payload["n"]}


class GuardTests(unittest.TestCase):
    def test_short_prompt_no_tokenize(self):
        s = Fake(1000, 10); out = s.complete(["abc"], max_tokens=160)
        self.assertEqual(s.calls, [("/v1/completions", 160)]); self.assertEqual(out[0][0]["finish"], "stop")

    def test_fixed_max_tokens_clipped(self):
        s = Fake(1000, 950); s.complete(["x" * 990], max_tokens=160)
        self.assertEqual(s.calls, [("/tokenize", None), ("/v1/completions", 50)])

    def test_full_context_gives_cut_without_request(self):
        s = Fake(1000, 1000); out = s.complete(["x" * 1000], n=3, max_tokens=160, logprobs=True)
        self.assertEqual(s.calls, [("/tokenize", None)])
        self.assertEqual(out, [[{"text": "", "finish": "length", "logprobs": []}] * 3])

    def test_uncapped_thought_without_room(self):
        s = Fake(1000, 600); os.environ["PL_TAIL_RESERVE"] = "512"
        try: out = s.complete(["x"], max_tokens=-1)
        finally: os.environ.pop("PL_TAIL_RESERVE")
        self.assertEqual(s.calls, [("/tokenize", None)]); self.assertEqual(out[0][0]["finish"], "length")

    def test_uncapped_thought_with_room(self):
        s = Fake(1000, 100); s.complete(["x"], max_tokens=-1)
        self.assertEqual(s.calls, [("/tokenize", None), ("/v1/completions", 388)])

    def test_http_400_body_is_logged_and_raised(self):
        class Bad(Fake):
            def _post(self, route, payload, timeout=600):
                raise urllib.error.HTTPError("u", 400, "Bad Request", {}, io.BytesIO(b"maximum context length"))
        import contextlib
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(urllib.error.HTTPError): Bad(1000, 0).complete(["abc"], max_tokens=5)
        self.assertIn("maximum context length", err.getvalue())


if __name__ == "__main__":
    unittest.main()
