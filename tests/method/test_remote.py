"""The other machine's picture server in a 2-node run (plearn/models/sampler.py RemoteMMSampler with PL_REMOTE_SERVE):
its command there carries the local server's settings, and without PL_REMOTE_SERVE nothing is started or stopped from here."""
import json
import os
import unittest
from unittest import mock

from plearn.models.sampler import RemoteMMSampler

URL = "http://10.0.0.2:18090"
CTL = {"host": "worker@10.0.0.2", "ssh": "ssh -o BatchMode=yes", "repo": "/r", "env": "/r/env.sh", "model": "/m/Qwen3-VL-8B-Thinking",
       "gpus": "0,1,2,3,4,5,6,7", "log": "/runs/x/logs/mm_server.log"}


class RemoteServerTests(unittest.TestCase):
    def test_the_command_carries_the_local_settings(self):
        env = {"PL_REMOTE_SERVE": json.dumps({URL: CTL}), "PL_MAX_MODEL_LEN": "57344", "PL_MAX_IMAGES": "128", "PL_MAX_NUM_SEQS": "512",
               "PL_MM_SIZE": '{"shortest_edge": 3136, "longest_edge": 262144}', "PL_MAX_BATCHED": "65536"}
        with mock.patch.dict(os.environ, env):
            r = RemoteMMSampler(URL); a = r.remote_argv(0.85)
        flag = lambda k: a[a.index(k) + 1]
        self.assertEqual(a[:3], ["python", "-m", "plearn.models.mm_server"])
        self.assertEqual((flag("--model"), flag("--port"), flag("--host"), flag("--tp"), flag("--mem")), ("/m/Qwen3-VL-8B-Thinking", "18090", "0.0.0.0", "8", "0.85"))
        self.assertEqual((flag("--max_model_len"), flag("--max_images"), flag("--max_num_seqs"), flag("--max_batched")), ("57344", "128", "512", "65536"))
        self.assertEqual(flag("--mm_size"), env["PL_MM_SIZE"]); self.assertEqual(flag("--max_lora_rank"), "16")

    def test_no_control_without_the_setting(self):
        with mock.patch.dict(os.environ, {"PL_REMOTE_SERVE": ""}):
            r = RemoteMMSampler(URL)
        self.assertIsNone(r.ctl)
        with mock.patch("subprocess.run") as run:
            r.stop(); run.assert_not_called()                  # its own launcher stops a server that this process did not start


    def test_busy_gpus_stop_the_start(self):
        """A process that holds those GPUs (a server left from an earlier start, another user's job) stops the start with its list;
        CUDA contexts alone (the parked learner ranks, about 0.4 GB) do not."""
        with mock.patch.dict(os.environ, {"PL_REMOTE_SERVE": json.dumps({URL: CTL})}):
            r = RemoteMMSampler(URL)
        ok = mock.Mock(returncode=0, stdout="", stderr="")
        busy = mock.Mock(returncode=0, stdout="3865812, 10460, VLLM::Worker_TP0\n3872423, 416, python\n", stderr="")
        with mock.patch.object(r, "_ssh", side_effect=[busy]):
            with self.assertRaises(RuntimeError) as ex: r.launch(0.85)
        self.assertIn("3865812, 10460, VLLM::Worker_TP0", str(ex.exception)); self.assertNotIn("3872423", str(ex.exception))
        small = mock.Mock(returncode=0, stdout="3872423, 416, python\n", stderr="")
        with mock.patch.object(r, "_ssh", side_effect=[small, ok]) as s:
            r.launch(0.85)
        self.assertIn("setsid nohup python -m plearn.models.mm_server", s.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
