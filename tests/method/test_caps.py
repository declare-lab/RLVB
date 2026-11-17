"""The thinking caps: 16,384 tokens in training (METHOD), 8,192 in every evaluation and test. The RLVB (EM) run files of the three
models (and their RLVB (GRPO) ablations) train at 8,192 too (method.max_think)."""
import glob
import os
import tempfile
import unittest
from unittest import mock

from plearn.api.registry import registry
from plearn.api.task import Task
from plearn.train import METHOD, commands, load

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CapTests(unittest.TestCase):
    def test_method_caps(self):
        self.assertEqual((METHOD["max_think"], METHOD["eval_max_think"]), (16384, 8192))

    # the run files whose training cap is the tests' cap: the Qwen3.5-4B RLVB (EM) runs with the presence score
    # (suite_qwen35_4b_presence_8k.yaml, suite_qwen35_4b_line_target.yaml), the 8B and 4B-VL RLVB (EM) run files, and the RLVB (GRPO)
    # ablations (the RLVB (EM) run files with --update grpo)
    TRAIN_CAP_8192 = {"suite_qwen35_4b_presence_8k.yaml", "suite_qwen35_4b_line_target.yaml", "suite_qwen3vl_8b_thinking.yaml",
                      "suite_qwen3vl_4b_thinking.yaml", "grpo_posterior_qwen3vl_8b_thinking.yaml",
                      "grpo_posterior_qwen3vl_4b_thinking.yaml", "grpo_posterior_qwen35_4b.yaml"}

    def test_every_pl_run_file_keeps_the_method_caps(self):
        files = sorted(glob.glob(os.path.join(ROOT, "configs", "train", "*.yaml")))
        self.assertTrue(files); seen = set()
        for f in files:
            m = (load(f).get("method") or {})
            if os.path.basename(f) in self.TRAIN_CAP_8192:
                seen.add(os.path.basename(f)); self.assertEqual(m.get("max_think"), 8192, f); self.assertNotIn("eval_max_think", m, f); continue
            self.assertNotIn("max_think", m, f); self.assertNotIn("eval_max_think", m, f)
        self.assertEqual(seen, self.TRAIN_CAP_8192)

    def test_the_run_files_reach_the_trainer_with_the_caps(self):
        names = {"suite_qwen3vl_8b_thinking": "8192", "suite_qwen3vl_4b_thinking": "8192", "suite_qwen35_4b": "16384"}
        with tempfile.TemporaryDirectory() as d:
            for m in ("Qwen3-VL-8B-Thinking", "Qwen3-VL-4B-Thinking", "Qwen3.5-4B"): os.mkdir(os.path.join(d, m))
            with mock.patch.dict(os.environ, {"PL_MODELS": d, "PL_RUNS": d}):
                os.environ.pop("QWEN35_4B_PATH", None)                    # patch.dict restores it
                for n in names:
                    a = commands(load(os.path.join(ROOT, "configs", "train", n + ".yaml")), 0, stamp="0000-0000")["learner"]
                    self.assertEqual((a[a.index("--max_think") + 1], a[a.index("--eval_max_think") + 1]), (names[n], "8192"), n)

    def test_every_test_task_is_capped_at_8192(self):
        reg = registry()
        for g in ("test_suite", "test_suite_long"):
            self.assertTrue(all(Task.load(n, reg).generation_kwargs["max_think"] == 8192 for n in reg.tasks([g])), g)


if __name__ == "__main__":
    unittest.main()
