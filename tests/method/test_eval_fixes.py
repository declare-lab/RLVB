"""Evaluation and training fixes, on the CPU:
  PL_TEST_T        plearn/api/task.py test_temperature: temperature and line_temperature of every task;
  preimport        plearn/evaluator.py: the game libraries in the main thread before the game threads;
  PL_NO_CUDNN_SDP  plearn/__init__.py cudnn_sdp_off: the cuDNN SDPA backend off;
  repeat_loop      plearn/method/trainer.py: the log metric think_repeat_n / think_repeat_of;
  METHOD           the method defaults: dec_term own, max_think 16,384, eval_max_think 8,192."""
import os
import subprocess
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTemperatureTests(unittest.TestCase):
    def load(self, name, env):
        from plearn.api.task import Task
        with mock.patch.dict(os.environ, env, clear=False):
            if "PL_TEST_T" not in env: os.environ.pop("PL_TEST_T", None)
            return Task.load(name).generation_kwargs

    def test_not_set_is_the_old_behaviour(self):
        from plearn.api.task import GEN_DEFAULTS
        for v in ({}, {"PL_TEST_T": ""}):
            g = self.load("mastermind", v)
            self.assertEqual((g["temperature"], g["line_temperature"]), (GEN_DEFAULTS["temperature"], GEN_DEFAULTS["line_temperature"]))
        self.assertEqual(GEN_DEFAULTS["temperature"], 0.6)

    def test_set_is_every_tasks_temperature(self):
        for name in ("mastermind", "frozenlake_hidden_goal", "proworlds_expansion_face", "roboquest_text_odd_parcel"):
            with self.subTest(task=name):
                g = self.load(name, {"PL_TEST_T": "1.0"})
                self.assertEqual((g["temperature"], g["line_temperature"]), (1.0, 1.0))
                self.assertEqual((g["top_p"], g["top_k"], g["max_think"]), (0.95, 20, 8192))   # nothing else changes

    def test_over_a_task_files_own_value(self):
        from plearn.api.task import test_temperature
        g = {"temperature": 0.3, "line_temperature": 0.2, "top_p": 0.9}
        with mock.patch.dict(os.environ, {"PL_TEST_T": "0.8"}): self.assertEqual(test_temperature(g), {"temperature": 0.8, "line_temperature": 0.8, "top_p": 0.9})
        with mock.patch.dict(os.environ, {"PL_TEST_T": ""}): self.assertEqual(test_temperature(g), g)


class PreimportTests(unittest.TestCase):
    def tasks(self, *names):
        from plearn.api.task import Task
        return [Task.load(n) for n in names]

    def test_the_games_libraries_of_the_run(self):
        from plearn import evaluator
        seen = []
        with mock.patch.object(evaluator.importlib, "import_module", side_effect=lambda m: seen.append(m)):
            done = evaluator.preimport(self.tasks("frozenlake_hidden_goal", "minigrid_keycorridor", "mastermind"))
        self.assertEqual(seen, ["gymnasium", "gymnasium.utils", "gymnasium.envs.registration", "gymnasium.envs.toy_text.frozen_lake",
                                "minigrid", "minigrid.envs.keycorridor"])
        self.assertEqual(done, seen)

    def test_only_the_tasks_in_the_run(self):
        from plearn import evaluator
        seen = []
        with mock.patch.object(evaluator.importlib, "import_module", side_effect=lambda m: seen.append(m)):
            evaluator.preimport(self.tasks("mastermind", "proworlds_expansion_face"))
            self.assertEqual(seen, [])
            evaluator.preimport(self.tasks("minigrid_keycorridor"))
        self.assertEqual(seen, ["gymnasium", "minigrid", "minigrid.envs.keycorridor"])

    def test_a_missing_library_is_skipped(self):
        from plearn import evaluator
        def imp(m):
            if m.startswith("minigrid"): raise ModuleNotFoundError(f"No module named {m!r}")
        logs = []
        with mock.patch.object(evaluator.importlib, "import_module", side_effect=imp):
            done = evaluator.preimport(self.tasks("frozenlake_hidden_goal", "minigrid_keycorridor"), logs.append)
        self.assertEqual(done, ["gymnasium", "gymnasium.utils", "gymnasium.envs.registration", "gymnasium.envs.toy_text.frozen_lake"])
        self.assertEqual(len(logs), 2)

    def test_play_preimports_before_the_threads(self):
        """play() calls preimport in the calling thread before its thread pool (no episode is played: every spec is done)."""
        import threading
        from plearn import evaluator
        calls = []
        def pre(tasks, log=None): calls.append((threading.current_thread() is threading.main_thread(), [t.name for t in tasks]))
        ts = self.tasks("mastermind")
        with mock.patch.object(evaluator, "preimport", pre), mock.patch.object(type(ts[0]), "specs", lambda self, split=None, n=None: []):
            evaluator.play(None, None, ts, log=lambda *_: None)
        self.assertEqual(calls, [(True, ["mastermind"])])


class CudnnSdpTests(unittest.TestCase):
    def run_py(self, code, value):
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": ROOT, "CUDA_VISIBLE_DEVICES": ""}
        env.pop("PL_NO_CUDNN_SDP", None)
        if value is not None: env["PL_NO_CUDNN_SDP"] = value
        r = subprocess.run([sys.executable, "-c", code], cwd="/", env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:]); return r.stdout.split()[-1]

    def test_the_flag(self):
        self.assertEqual(self.run_py("import plearn, torch; print(torch.backends.cuda.cudnn_sdp_enabled())", "1"), "False")   # torch after plearn
        self.assertEqual(self.run_py("import torch, plearn; print(torch.backends.cuda.cudnn_sdp_enabled())", "1"), "False")   # torch before plearn
        for v in (None, "0"): self.assertEqual(self.run_py("import plearn, torch; print(torch.backends.cuda.cudnn_sdp_enabled())", v), "True")
        self.assertEqual(self.run_py("import plearn, sys; print('torch' in sys.modules)", "1"), "False")   # plearn does not import torch


class RepeatLoopTests(unittest.TestCase):
    def test_repeat_loop(self):
        from plearn.method.trainer import repeat_loop
        self.assertTrue(repeat_loop("Let me think. " + "So the answer is 3, but wait. " * 3))          # the same 30 characters 3 times at the end
        self.assertTrue(repeat_loop("start\n" + "\n".join(["Check cell 1.", "Check cell 2."] * 6)))      # half of the last lines are copies
        self.assertFalse(repeat_loop("So the answer is 3, but wait. " * 2))                             # 2 times only
        self.assertFalse(repeat_loop("\n".join(f"Step {i}: look at cell {i}." for i in range(30))))
        self.assertFalse(repeat_loop("ab" * 29))                                                         # under 3 x 20 characters: no check
        self.assertTrue(repeat_loop("ab" * 30))                                                          # a short period repeats at 20 too
        self.assertFalse(repeat_loop(""))

    def test_the_metric_and_wandb(self):
        from plearn.method.trainer import WANDB_KEEP, monitor_train_metrics
        loop = "Let me think. " + "So the answer is 3, but wait. " * 3
        rows_h = [[("p", loop), ("p", "A short thought.")], [("p", "Another thought.")]]
        out = monitor_train_metrics(None, rows_h, {}, [], None, None)
        self.assertEqual((out["think_repeat_n"], out["think_repeat_of"]), (1, 3))
        for k in ("train/think_repeat_n", "train/think_repeat_of"): self.assertTrue(WANDB_KEEP.match(k), k)


class RecipeTests(unittest.TestCase):
    def test_method_defaults(self):
        from plearn import train as T
        self.assertEqual((T.METHOD["dec_term"], T.METHOD["max_think"], T.METHOD["eval_max_think"]), ("own", 16384, 8192))


if __name__ == "__main__":
    unittest.main()
