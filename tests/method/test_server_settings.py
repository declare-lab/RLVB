"""The server settings of the runs, no method change:
- PL_GPUS_PER_SERVER=k: k GPUs a picture server, on this machine (the trainer's ServerSampler) and on the other machine (the launcher's
  PL_VLLM_REMOTE and PL_REMOTE_SERVE: one server a group, ports server_port + i, logs mm_server-<i>.log); unset: as before;
- PL_VLLM_SEED: the engine's seed of the picture server (mm_server.py); unset: vLLM's default."""
import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from plearn import train
from plearn.method import trainer
from plearn.models import sampler

SERVER = Path(__file__).resolve().parents[2] / "plearn" / "models" / "mm_server.py"


class Fake:
    started = []
    def __init__(self, model, gpus, **kw): self.gpus = gpus; Fake.started.append(gpus)
    def wait(self): pass
    def stop(self): pass


def local(env, gpus="0,1,2,3", images=True):
    """The GPU groups of the local servers that ServerSampler starts under env."""
    Fake.started = []; s = object.__new__(trainer.ServerSampler); s._cfg = ("m", gpus, "/nonexistent", 0.5, images)
    with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(sampler, "MMSampler", Fake), mock.patch.object(sampler, "VLLMSampler", Fake):
        for k in ("PL_GPUS_PER_SERVER", "PL_VLLM_TP", "PL_VLLM_REMOTE"):
            if k not in env: os.environ.pop(k, None)
        s._start()
    return Fake.started


class GpusPerServerTests(unittest.TestCase):
    def test_the_local_servers(self):
        self.assertEqual(local({"PL_GPUS_PER_SERVER": "2"}), ["0,1", "2,3"])
        self.assertEqual(local({"PL_VLLM_TP": "1"}), ["0,1,2,3"])                 # unset: one tensor-parallel server, as before
        with self.assertRaises(AssertionError): local({})                        # an image domain needs one server or the setting

    def test_the_other_machine(self):
        with tempfile.TemporaryDirectory() as d:
            nd = {"host": "h", "ssh": "ssh", "addr": "10.0.0.2", "repo": "/r", "env_sh": "/e", "model": "/m", "gpus": "4,5,6,7", "runs": "/runs", "server_port": 18090}
            c = {"train": "x", "model": d, "train_tasks": "frozenlake_hidden_goal", "output": d, "gpus": {"actor": "0,1", "learner": "0,1"},
                 "nodes": [nd], "master": {"addr": "10.0.0.1", "port": 29500}}
            one = train.commands(c)["env"]
            self.assertEqual(one["PL_VLLM_REMOTE"], "http://10.0.0.2:18090")
            self.assertEqual(json.loads(one["PL_REMOTE_SERVE"])["http://10.0.0.2:18090"]["log"], "/runs/x/logs/mm_server.log")
            two = train.commands({**c, "server": {"PL_GPUS_PER_SERVER": 2}})["env"]
            self.assertEqual(two["PL_VLLM_REMOTE"], "http://10.0.0.2:18090,http://10.0.0.2:18091")
            srv = json.loads(two["PL_REMOTE_SERVE"])
            self.assertEqual([(v["gpus"], v["log"]) for v in srv.values()], [("4,5", "/runs/x/logs/mm_server-0.log"), ("6,7", "/runs/x/logs/mm_server-1.log")])


class SeedTests(unittest.TestCase):
    def test_the_engine_seed(self):
        src = SERVER.read_text(); ast.parse(src)
        self.assertEqual(src.count('if os.environ.get("PL_VLLM_SEED"): kw["seed"] = int(os.environ["PL_VLLM_SEED"])'), 1)
        self.assertLess(src.index('kw["seed"]'), src.index("AsyncEngineArgs(**kw)"))    # set before the engine starts


if __name__ == "__main__":
    unittest.main()
