"""PL_PRESENCE_PENALTY (the model card's thinking setting, presence_penalty 1.5): the local and the remote
picture server get --presence_penalty only when the variable is set, the server's flag defaults to 0 (no change), and vLLM's
SamplingParams takes the value. A test is skipped when vLLM is not installed. (This tree has no loop cut, so the flag reaches one
sampling path, params().)"""
import ast
import importlib.util
import json
import os
import unittest
from pathlib import Path

from plearn.models.sampler import RemoteMMSampler

SERVER = Path(__file__).resolve().parents[2] / "plearn" / "models" / "mm_server.py"


class Env:
    def __init__(self, **kv): self.kv = kv
    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kv}
        for k, v in self.kv.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


class ArgvTests(unittest.TestCase):
    def argv(self, pp):
        with Env(PL_REMOTE_SERVE=json.dumps({"http://x:1": {"gpus": "0,1", "model": "m"}}), PL_PRESENCE_PENALTY=pp):
            return RemoteMMSampler("http://x:1").remote_argv(0.8)

    def test_the_remote_server_gets_the_flag_only_when_set(self):
        a = self.argv("1.5"); self.assertIn("--presence_penalty", a); self.assertEqual(a[a.index("--presence_penalty") + 1], "1.5")
        self.assertNotIn("--presence_penalty", self.argv(None))

    def test_the_local_server_and_the_text_requests_get_it(self):
        src = (SERVER.parent / "sampler.py").read_text()
        self.assertEqual(src.count('argv += ["--presence_penalty", os.environ["PL_PRESENCE_PENALTY"]]'), 2)   # local and remote picture server
        self.assertEqual(src.count('payload["presence_penalty"] = float(os.environ["PL_PRESENCE_PENALTY"])'), 1)   # the text server's requests

    def test_the_server_flag_defaults_to_zero_and_reaches_the_sampling(self):
        src = SERVER.read_text(); tree = ast.parse(src)
        flags = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
                 and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--presence_penalty"]
        self.assertEqual(len(flags), 1)
        self.assertEqual({k.arg: k.value.value for k in flags[0].keywords if k.arg == "default"}, {"default": 0.0})
        self.assertEqual(src.count("presence_penalty=args.presence_penalty"), 1)   # params()


@unittest.skipUnless(importlib.util.find_spec("vllm"), "needs vllm")
class VllmTests(unittest.TestCase):
    def test_sampling_params_take_the_value(self):
        from vllm import SamplingParams
        self.assertEqual(SamplingParams(n=1, max_tokens=8, temperature=1.0, presence_penalty=1.5).presence_penalty, 1.5)


if __name__ == "__main__":
    unittest.main()
