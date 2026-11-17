"""The metrics that W&B records (plearn/method/trainer.py WANDB_KEEP): the useful training and evaluation metrics pass, the bookkeeping and the features that are off do not, and
PL_WANDB_ALL=1 sends every metric."""
import argparse
import os
import sys
import types
import unittest
from unittest import mock

from plearn.method.trainer import Board


def fake_wandb(last, calls):
    """A wandb module whose run's history ends at update last (None: no history; an exception: unreadable)."""
    class Api:
        default_entity = "team"
        def run(self, path):
            if isinstance(last, Exception): raise last
            calls.append(("api", path)); return types.SimpleNamespace(summary={} if last is None else {"_step": last})
    def init(**kw):
        calls.append(("init", {k: kw[k] for k in ("id", "resume", "resume_from") if k in kw}))
        return types.SimpleNamespace(id=kw.get("id", "new"), url="u", define_metric=lambda *a, **k: None)
    return types.SimpleNamespace(Api=Api, init=init)


class WandbResumeTests(unittest.TestCase):
    """A --resume_step run goes on in its own W&B run: resumed when its history ends at the resumed update,
    rewound only when it has history after it."""
    def start(self, last):
        calls = []; args = argparse.Namespace(wandb_project="p", wandb_entity=None, wandb_name="n", wandb_group=None, resume_step=50)
        with mock.patch.dict(sys.modules, {"wandb": fake_wandb(last, calls)}), mock.patch.dict(os.environ, {"PL_WANDB_ID": "k5nk0yu9"}):
            Board(args, "tag", 8)
        return [c[1] for c in calls if c[0] == "init"], [c[1] for c in calls if c[0] == "api"]

    def test_a_run_that_ends_at_the_resumed_update_goes_on(self):
        inits, paths = self.start(50)
        self.assertEqual(inits[0], {"id": "k5nk0yu9", "resume": "must"}); self.assertEqual(paths, ["team/p/k5nk0yu9"])

    def test_a_run_with_history_after_it_is_rewound(self):
        self.assertEqual(self.start(60)[0][0], {"resume_from": "k5nk0yu9?_step=50"})

    def test_an_unreadable_history_is_rewound(self):
        self.assertEqual(self.start(RuntimeError("no network"))[0][0], {"resume_from": "k5nk0yu9?_step=50"})


class WandbKeepTests(unittest.TestCase):
    def test_a_training_row(self):
        row = {"loss_dec": 0.4, "em_q_budget": 0.5, "think_tokens_q": 9000.0, "kl_q_pi": 1.6, "em_left_out": 3, "sec": 250.0,
               "queue_wait": 1.0, "credit_moved": 0.0, "m_balance": 1.1, "R_spread": 30.0, "think_cost": 0.0, "dp_param_spread": 0.0}
        self.assertEqual(sorted(Board.kept(row, "train")),
                         ["train/em_left_out", "train/em_q_budget", "train/kl_q_pi", "train/loss_dec", "train/sec", "train/think_tokens_q"])

    def test_an_evaluation_row(self):
        ev = {"update": 25, "belief_ok": 0.7, "anchor_recall": 0.6, "by_family": {"mastermind": {"exact": 1.0}},
              "groups": {"train_suite": {"success": 0.49, "episodes": 576}, "test_suite": {"success": 0.49}, "tabletop": {"read_exact": 0.6}},
              "tasks": {"mastermind": {"success": 0.94, "success_stderr": 0.03, "premature": 0.1, "turns": 3.5},
                        "tabletop_cups": {"success": 0.5, "read_misread": 0.4, "read_width": 1.3},
                        "tabletop_facts_blocks": {"fact_acc": 0.41}}}
        self.assertEqual(sorted(Board.kept(ev, "eval")),
                         ["eval/belief_ok", "eval/groups.train_suite.success", "eval/tasks.mastermind.premature", "eval/tasks.mastermind.success",
                          "eval/tasks.tabletop_cups.read_misread", "eval/tasks.tabletop_cups.success",
                          "eval/tasks.tabletop_facts_blocks.fact_acc", "eval/update"])

    def test_every_metric_on_request(self):
        with mock.patch.dict(os.environ, {"PL_WANDB_ALL": "1"}):
            self.assertEqual(len(Board.kept({"queue_wait": 1.0, "loss_dec": 0.4}, "train")), 2)


if __name__ == "__main__":
    unittest.main()
