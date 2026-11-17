"""The training launcher (plearn/train.py) on the registered runs: the watch sample of the benchmarks reaches the trainer as
--watch, every watched task is registered, no watched or evaluated benchmark task is a training task, and no watch has a RoboQuest
task, and no run file evaluates a RoboQuest task in training."""
import json
import os
import tempfile
import unittest
from unittest import mock

from plearn import train
from plearn.api.registry import REPO_ROOT, registry
from plearn.api.task import Task


def flag(argv, name):
    return argv[argv.index(name) + 1]


# The tests build their runs (a watch sample, two machines) from the 8B run file.
BASE = "suite_qwen3vl_8b_thinking"
NODE = {"host": "user@10.0.0.2", "ssh": "ssh -i ~/.ssh/id_ed25519 -o BatchMode=yes -o StrictHostKeyChecking=accept-new",
        "addr": "10.0.0.2", "repo": "/data/pro-worlds", "env_sh": "/data/env.sh",
        "model": "/data/models/Qwen3-VL-8B-Thinking", "gpus": "0,1,2,3,4,5,6,7", "runs": "/data/runs", "server_port": 18090}


class LauncherTests(unittest.TestCase):
    def setUp(self):                                                  # the model's folder (no weights are read)
        self.tmp = tempfile.TemporaryDirectory(); os.mkdir(os.path.join(self.tmp.name, "Qwen3-VL-8B-Thinking"))
        self.env = mock.patch.dict(os.environ, {"PL_MODELS": self.tmp.name}); self.env.start()

    def tearDown(self): self.env.stop(); self.tmp.cleanup()

    def test_the_watch_reaches_the_trainer(self):
        """No registered run file has a watch: the 8B run file with a watch sample of two test-only tasks."""
        reg = registry()
        c = {**train.load(BASE), "watch": {"proworlds_expansion": 4, "minesweeper_hard": 40}}; cmd = train.commands(c)
        watch = json.loads(flag(cmd["actor"], "--watch"))
        self.assertEqual(watch, c["watch"]); self.assertEqual(flag(cmd["learner"], "--watch"), flag(cmd["actor"], "--watch"))
        watched = set(reg.tasks(list(watch)))
        self.assertTrue(watched)
        self.assertFalse(watched & set(reg.tasks(c["train_tasks"])))     # the watch is never a training task
        self.assertEqual([n for n in watched if Task.load(n, reg).dataset == "roboquest"], [])
        self.assertEqual(flag(cmd["actor"], "--eval_max_think"), "8192")
        self.assertEqual(flag(train.commands(train.load(BASE))["actor"], "--watch"), "")   # no watch: an empty flag

    def test_no_run_evaluates_roboquest(self):
        reg = registry()
        for f in sorted((REPO_ROOT / "configs" / "train").glob("*.yaml")):
            with self.subTest(run=f.stem):
                c = train.load(f.stem)
                ev = c.get("eval_tasks") or []
                names = ([ev] if isinstance(ev, str) else list(ev)) + list(c.get("watch") or {})
                tasks = set(reg.tasks(names)) if names else set()
                self.assertEqual(sorted(n for n in tasks if Task.load(n, reg).dataset == "roboquest"), [])

    def test_the_run_name_has_its_start(self):
        """The W&B run name tells the runs of one config apart: NAME-MMDD-HHMM of the start, -rR when resumed;
        a name in the config is kept."""
        named = train.load(BASE)
        c = {**named, "wandb": {k: v for k, v in named["wandb"].items() if k != "name"}}
        self.assertEqual(flag(train.commands(c, stamp="1031-1816")["actor"], "--wandb_name"), f"{BASE}-1031-1816")
        self.assertEqual(flag(train.commands(c, 25, stamp="1101-0930")["learner"], "--wandb_name"), f"{BASE}-1101-0930-r25")
        self.assertRegex(flag(train.commands(c)["learner"], "--wandb_name"), rf"^{BASE}-\d{{4}}-\d{{4}}$")
        self.assertEqual(flag(train.commands(named)["actor"], "--wandb_name"), named["wandb"]["name"])
        self.assertEqual(flag(train.commands({**c, "wandb": {**c["wandb"], "name": "my run"}})["actor"], "--wandb_name"), "my run")

    def test_the_two_machine_run(self):
        """A run on two machines (the keys of a 2-node run file on the 8B run file): the learner over the 2 machines (rank 0 here; the other machine's ranks with the same trainer flags and their own
        model path), the actor's view of the other machine (its server, how this process starts it, where adapters go), and NCCL
        and gloo on eth0."""
        b = train.load(BASE)
        c = {**b, "train": f"{BASE}_2node", "output": f"runs/{BASE}_2node", "gpus": {"actor": "0,1,2,3,4,5,6,7", "learner": "0,1,2,3,4,5,6,7", "colocate": True},
             "server": {k: v for k, v in b["server"].items() if k != "PL_GPUS_PER_SERVER"},     # one server over the other machine's 8 GPUs
             "master": {"addr": "10.0.0.1", "port": 29510}, "ifname": "eth0", "nodes": [dict(NODE)]}
        cmd = train.commands(c); L = cmd["learner"]
        self.assertNotIn("--standalone", L)
        self.assertEqual((flag(L, "--nnodes"), flag(L, "--node_rank"), flag(L, "--master_addr"), flag(L, "--master_port")), ("2", "0", "10.0.0.1", "29510"))
        (nd,) = cmd["remote"]; R = nd["argv"]
        self.assertEqual((flag(R, "--nnodes"), flag(R, "--node_rank"), flag(R, "--nproc_per_node")), ("2", "1", "8"))
        self.assertEqual(flag(R, "--model"), "/data/models/Qwen3-VL-8B-Thinking")
        i, j = L.index("plearn.method.trainer"), R.index("plearn.method.trainer")
        self.assertEqual(train.with_flag(L[i:], "--model", "M"), train.with_flag(R[j:], "--model", "M"))   # every other flag is the same
        env = cmd["env"]; url = "http://10.0.0.2:18090"
        self.assertEqual(env["PL_VLLM_REMOTE"], url)
        srv = json.loads(env["PL_REMOTE_SERVE"])[url]
        self.assertEqual((srv["model"], srv["gpus"], srv["host"]), ("/data/models/Qwen3-VL-8B-Thinking", "0,1,2,3,4,5,6,7", "user@10.0.0.2"))
        self.assertEqual(env["PL_REMOTE_SYNC"], f"user@10.0.0.2:/data/runs/{BASE}_2node/adapters")
        self.assertEqual((env["NCCL_SOCKET_IFNAME"], env["GLOO_SOCKET_IFNAME"], env["NCCL_IB_DISABLE"]), ("eth0", "eth0", "1"))
        sh = train.remote_cmd(nd, {"PL_X": "a b"}, R, nd["log"])
        self.assertIn("PL_X='a b' setsid nohup python -m torch.distributed.run", sh); self.assertTrue(sh.endswith(f"echo $! > {nd['log']}.pid"))
        one = train.commands({**c, "method": {**c["method"], "steps": 2, "phase": 2}})         # the smoke test's shape (2 updates)
        self.assertEqual(flag(one["actor"], "--steps"), "2"); self.assertEqual(len(one["remote"]), 1)
        k1 = train.commands({**c, "server": b["server"]})["env"]                               # PL_GPUS_PER_SERVER 1: a server a GPU there
        self.assertEqual(k1["PL_VLLM_REMOTE"].split(","), [f"http://10.0.0.2:{18090 + i}" for i in range(8)])


if __name__ == "__main__":
    unittest.main()
