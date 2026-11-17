"""The score's sampler settings on every rank.

A data-parallel learner starts one process a rank (torchrun). In train() a worker rank enters its job loop and never comes back,
so a setting made after that point exists on rank 0 only. When the score's two tables (SAMPLER, DECISION) are set there, a worker
rank scores its share of every batch with the defaults of the module, temperature 0.6 and no presence penalty, while the
sampler of the runs has temperature 1.0 and presence penalty 1.5. On a learner-only run of two ranks (Qwen3.5-4B, the
saved base set) the thought block's loss of update 1 was the raw model's (3,515 against 3,119 under the sampler's own
distribution), the gradient norm was 3,372 against 146 with the sampler's logged distributions in the place of the score's
normaliser, and the KL from the phase prior reached the stop value at update 7 in every Qwen3.5-4B run. These tests hold the
settings to every rank: the function that sets them, its place in train(), and the check at each data-parallel update."""
import inspect
import os
import re
import types
import unittest
from unittest import mock

import plearn.method.trainer as T


def run_args(**kw):
    return types.SimpleNamespace(**{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "score_presence": 1, **kw})


class SettingsTests(unittest.TestCase):
    def setUp(self): self.keep = (dict(T.SAMPLER), dict(T.DECISION))
    def tearDown(self): T.SAMPLER.update(self.keep[0]); T.DECISION.update(self.keep[1])

    def test_the_run_values_reach_both_tables(self):
        with mock.patch.dict(os.environ, {"PL_PRESENCE_PENALTY": "1.5"}):
            pp = T.score_settings(run_args())
        self.assertEqual(pp, 1.5)
        self.assertEqual(T.SAMPLER, {"T": 1.0, "top_p": 0.95, "top_k": 20, "presence": 1.5})
        self.assertEqual((T.DECISION["T"], T.DECISION["presence"]), (1.0, 1.5))
        self.assertEqual((T.DECISION["top_p"], T.DECISION["top_k"]), (1.0, -1))            # the decision keeps the full support

    def test_no_penalty_without_the_option_or_without_the_variable(self):
        with mock.patch.dict(os.environ, {"PL_PRESENCE_PENALTY": "1.5"}):
            self.assertEqual(T.score_settings(run_args(score_presence=0)), 0.0)
        self.assertEqual(T.SAMPLER["presence"], 0.0)
        env = {k: v for k, v in os.environ.items() if k != "PL_PRESENCE_PENALTY"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(T.score_settings(run_args()), 0.0)

    def test_the_row_of_numbers_changes_with_each_setting(self):
        T.score_settings(run_args(score_presence=0)); base = T.settings_row()
        for kw in ({"temperature": 0.6}, {"top_p": 0.8}, {"top_k": 40}):
            T.score_settings(run_args(score_presence=0, **kw)); self.assertNotEqual(T.settings_row(), base, kw)
        with mock.patch.dict(os.environ, {"PL_PRESENCE_PENALTY": "1.5"}):
            T.score_settings(run_args()); self.assertNotEqual(T.settings_row(), base)


class PlaceTests(unittest.TestCase):
    """The order of train(): the settings, then the worker rank's job loop. A source check: train() needs a model and GPUs."""
    src = inspect.getsource(T.train)

    def test_the_settings_come_before_the_worker_loop(self):
        self.assertEqual(self.src.count("score_settings(args)"), 1)
        self.assertIn("if not lead:", self.src)
        self.assertLess(self.src.index("score_settings(args)"), self.src.index("if not lead:"))

    def test_nothing_else_in_train_sets_the_tables(self):
        for name in ("SAMPLER.update", "DECISION.update"): self.assertNotIn(name, self.src, name)
        self.assertIsNone(re.search(r"(SAMPLER|DECISION)\[[^\]]+\]\s*=[^=]", self.src))            # no item of a table is assigned

    def test_every_data_parallel_update_checks_the_ranks(self):
        upd = self.src[self.src.index("def em_update("):]
        self.assertIn("if world > 1: same_settings()", upd[:upd.index("def shard_score(")])
        chk = self.src[self.src.index("def same_settings("):self.src.index("def em_update(")]
        self.assertIn("settings_row()", chk); self.assertIn("all_gather", chk); self.assertIn("assert", chk)


if __name__ == "__main__":
    unittest.main()
