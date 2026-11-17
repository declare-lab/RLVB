"""The Qwen3.5-4B RLVB (EM) run at T 0.6 (configs/train/suite_qwen35_4b_t06.yaml): the trainer commands of suite_qwen35_4b.yaml
except the temperature (0.6 for 1.0) and the run's own names; lr 1e-5 (METHOD); the normal LoRA."""
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class T06RunTests(unittest.TestCase):
    def test_only_the_temperature_changes(self):
        from plearn.train import commands, load
        sys.path.insert(0, os.path.join(ROOT, "tests", "method"))
        from test_run_files import flags
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "Qwen3.5-4B"))
            env = {k: v for k, v in os.environ.items() if k not in ("PL_GPUS", "PL_SEQS", "PL_PRESENCE_PENALTY", "QWEN35_4B_PATH")}
            with mock.patch.dict(os.environ, {**env, "PL_MODELS": d, "PL_RUNS": d}, clear=True):
                base = commands(load(os.path.join(ROOT, "configs", "train", "suite_qwen35_4b.yaml")), 0, stamp="0000-0000")
                c = commands(load(os.path.join(ROOT, "configs", "train", "suite_qwen35_4b_t06.yaml")), 0, stamp="0000-0000")
                for role in ("actor", "learner"):
                    with self.subTest(role=role):
                        a, b = flags(c[role]), flags(base[role])
                        if "--temperature" in a or "--temperature" in b:
                            self.assertEqual(float(a.pop("--temperature")), 0.6); self.assertEqual(float(b.pop("--temperature")), 1.0)
                        if "--lr" in a: self.assertEqual(float(a["--lr"]), 1e-5)
                        for k in ("--tag", "--out", "--queue", "--dump_em", "--dump_rollouts", "--wandb_name"): a.pop(k, None); b.pop(k, None)
                        self.assertEqual(a, b)                                     # the same recipe otherwise (lr, kl_stop, LoRA, caps)
                        self.assertNotIn("--lora_all_linear", flags(c[role]))      # the normal LoRA
                self.assertEqual(float(flags(c["actor"])["--temperature"]), 0.6)
                self.assertEqual(c["env"], base["env"]); self.assertTrue(str(c["out"]).endswith("suite_qwen35_4b_t06"))


if __name__ == "__main__":
    unittest.main()
