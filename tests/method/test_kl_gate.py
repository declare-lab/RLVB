"""Qwen3.5-4B RLVB (EM): the KL gate tool (tools/kl_gate.py), the lr 3e-6 run file (the recipe of suite_qwen35_4b.yaml with another lr,
the normal LoRA), and scripts/pl_qwen35_4b.sh (no KL gate; lr 1e-5 and T 1.0)."""
import json, os, subprocess, sys, tempfile, unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import kl_gate as G   # noqa: E402


class Decide(unittest.TestCase):
    def test_cases(self):
        r = lambda s, kl, left=None: {"step": s, "kl_prior": kl, **({"kl_stop_left": left} if left is not None else {})}
        self.assertEqual(G.decide([r(1, .001), r(2, .002)], 5, .004), (None, None))                 # not yet
        self.assertEqual(G.decide([r(k, .001) for k in range(1, 6)], 5, .004)[0], "PASS")
        self.assertEqual(G.decide([r(k, .001 * k) for k in range(1, 6)], 5, .004)[0], "FAIL")      # 0.005 > 0.004
        self.assertEqual(G.decide([r(1, .001), r(2, .016, 23), r(26, .001)], 5, .004)[0], "FAIL")    # KL STOP at u2
        self.assertEqual(G.decide([r(1, .001), r(2, .002), r(4, .003)], 5, .004), (None, None))
        self.assertEqual(G.decide([r(1, .001), r(6, .001)], 5, .004)[0], "FAIL")                    # u5 skipped
        self.assertEqual(G.decide([r(5, .004)], 5, .004)[0], "PASS")                               # the limit itself passes

    def test_main_reads_a_growing_log_and_skips_a_cut_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.jsonl")
            with open(p, "w") as f:
                f.write(json.dumps({"phase_set": 0}) + "\n")                                       # not an update row
                for k in range(1, 6): f.write(json.dumps({"step": k, "kl_prior": 0.0005 * k, "loss_think": 1.0}) + "\n")
                f.write('{"step": 6, "kl_pr')                                                       # a line not written whole yet
            self.assertEqual(G.main(["--log", p, "--poll", "0"]), 0)
            with open(p, "w") as f: f.write(json.dumps({"step": 1, "kl_prior": 0.001}) + "\n")
            self.assertEqual(G.main(["--log", p, "--poll", "0.01", "--timeout", "0.05"]), 2)          # no decision: NONE
            dead = subprocess.Popen(["true"]); dead.wait()
            self.assertEqual(G.main(["--log", p, "--poll", "0.01", "--pid", str(dead.pid)]), 2)


class LearnerLog(unittest.TestCase):
    def test_the_trainer_names_its_log_so(self):
        """The trainer's own lines (plearn/method/trainer.py): the run's tag and the learner's log file; learner_log finds that file
        (the domain's name first) and not the actor's."""
        src = open(os.path.join(ROOT, "plearn", "method", "trainer.py")).read()
        self.assertIn('tag = f"{dom_name}{(\'-\' + args.tag) if args.tag else \'\'}-seed{args.seed}"', src)
        self.assertIn("f_log = os.path.join(args.out, f\"{tag}{'-actor' if actor else ''}.jsonl\")", src)
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(G.learner_log(d, "suite_qwen35_4b_lr3e6"))
            for f in ("mixture-suite_qwen35_4b_lr3e6-seed0-actor.jsonl", "suite_qwen35_4b_lr3e6.jsonl", "mixture-other-seed0.jsonl"): open(os.path.join(d, f), "w").close()
            self.assertIsNone(G.learner_log(d, "suite_qwen35_4b_lr3e6"))
            open(os.path.join(d, "mixture-suite_qwen35_4b_lr3e6-seed0.jsonl"), "w").close()
            self.assertEqual(G.learner_log(d, "suite_qwen35_4b_lr3e6"), os.path.join(d, "mixture-suite_qwen35_4b_lr3e6-seed0.jsonl"))
            with open(os.path.join(d, "mixture-suite_qwen35_4b_lr3e6-seed0.jsonl"), "w") as f:
                for k in range(1, 6): f.write(json.dumps({"step": k, "kl_prior": 0.0005 * k}) + "\n")
            self.assertEqual(G.main(["--run_dir", d, "--tag", "suite_qwen35_4b_lr3e6", "--poll", "0"]), 0)


class RunFiles(unittest.TestCase):
    def test_lr_run_is_the_base_recipe_with_another_lr(self):
        from plearn.train import commands, load
        sys.path.insert(0, os.path.join(ROOT, "tests", "method"))
        from test_run_files import flags
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "Qwen3.5-4B"))
            env = {k: v for k, v in os.environ.items() if k not in ("PL_GPUS", "PL_SEQS", "PL_PRESENCE_PENALTY", "QWEN35_4B_PATH")}
            with mock.patch.dict(os.environ, {**env, "PL_MODELS": d, "PL_RUNS": d}, clear=True):
                base = commands(load(os.path.join(ROOT, "configs", "train", "suite_qwen35_4b.yaml")), 0, stamp="0000-0000")
                for tag, lr in (("lr3e6", "3e-06"),):
                    with self.subTest(run=tag):
                        c = commands(load(os.path.join(ROOT, "configs", "train", f"suite_qwen35_4b_{tag}.yaml")), 0, stamp="0000-0000")
                        a, b = flags(c["actor"]), flags(base["actor"])
                        self.assertEqual(float(a.pop("--lr")), float(lr)); self.assertEqual(float(b.pop("--lr")), 1e-5)
                        for k in ("--tag", "--out", "--queue", "--dump_em", "--dump_rollouts", "--wandb_name"): a.pop(k, None); b.pop(k, None)
                        self.assertEqual(a, b)                                                     # the same recipe otherwise
                        self.assertNotIn("--lora_all_linear", flags(c["learner"]))                 # the normal LoRA (no DeltaNet in_proj_a/b)
                        self.assertEqual(c["env"], base["env"]); self.assertTrue(str(c["out"]).endswith(f"suite_qwen35_4b_{tag}"))


class Script(unittest.TestCase):
    """scripts/pl_qwen35_4b.sh: the default run file suite_qwen35_4b_presence_8k.yaml (lr 1e-5, T 1.0), no KL gate; RUN_FILE picks another."""
    def test_dry(self):
        r = subprocess.run(["bash", os.path.join(ROOT, "scripts", "pl_qwen35_4b.sh")], env={**os.environ, "DRY": "1"}, capture_output=True, text=True, cwd="/")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("+ python -m plearn train --config suite_qwen35_4b_presence_8k\n", r.stdout)   # the default run file
        self.assertNotIn("kl_gate", r.stdout); self.assertNotIn("lr3e6", r.stdout); self.assertNotIn("lr1e6", r.stdout)
        self.assertIn("--adapter runs/suite_qwen35_4b_presence_8k/queue-suite_qwen35_4b_presence_8k/adapter_50", r.stdout)
        r0 = subprocess.run(["bash", os.path.join(ROOT, "scripts", "pl_qwen35_4b.sh")], env={**os.environ, "DRY": "1", "RUN_FILE": "suite_qwen35_4b"}, capture_output=True, text=True, cwd="/")
        self.assertIn("+ python -m plearn train --config suite_qwen35_4b\n", r0.stdout)   # another run file by RUN_FILE

    def test_run_file_is_lr_1e5_t_1(self):
        from plearn.train import commands, load
        sys.path.insert(0, os.path.join(ROOT, "tests", "method"))
        from test_run_files import flags
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "Qwen3.5-4B"))
            env = {k: v for k, v in os.environ.items() if k not in ("PL_GPUS", "PL_SEQS", "PL_PRESENCE_PENALTY", "QWEN35_4B_PATH")}
            with mock.patch.dict(os.environ, {**env, "PL_MODELS": d, "PL_RUNS": d}, clear=True):
                a = flags(commands(load(os.path.join(ROOT, "configs", "train", "suite_qwen35_4b.yaml")), 0, stamp="0000-0000")["actor"])
                self.assertEqual(float(a["--lr"]), 1e-5); self.assertEqual(float(a["--temperature"]), 1.0)


if __name__ == "__main__":
    unittest.main()
