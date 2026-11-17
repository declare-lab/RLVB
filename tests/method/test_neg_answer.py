"""The push-down of rejected answers (plearn/method/trainer.py --neg_answer): at a history where no own
turn takes an action the teacher accepts (accept_only leaves it out), each own answer is pushed down by the bounded unlikelihood
-log(1 - p) of its answer tokens only; --set_keep keeps such histories; the flag reaches the trainer from a run file; METHOD has it off."""
import os
import tempfile
import unittest
from unittest import mock

import torch

import plearn.method.trainer as T
from plearn.method.trainer import CLOSE, TERM, keep_set, loss_neg, neg_item, neg_rows, neg_thought_items
from plearn.train import METHOD, commands


def turn(b, u, acc, z="z"):
    """One own turn as the E-step writes it (prompt, thought, prefix, line 1, line 2, R, stopped, readout, accepted)."""
    return ("p", z, "p" + z + CLOSE, b, u, 0.0, True, None, acc)


class RowsTests(unittest.TestCase):
    def setUp(self):
        self.rows_h = [[turn("Possible: rows 1, 2", "ACTION: SUBMIT 1", False), turn("Possible: rows 1, 2", "ACTION: LOOK 2", True)],  # 0: one accepted
                       [turn("Possible: row 1", "ACTION: SUBMIT 1", False), turn("Possible: row 3", "ACTION: LOOK 3", False),
                        turn("Possible: row 2", "ACTION: SUBMIT 2", False)],                                                         # 1: none accepted
                       [turn("Possible: row 1", "ACTION: LOOK 1", False)],                                                           # 2: none, no answer
                       [turn("Possible: row 1", "ACTION: SUBMIT 1", True)]]                                                          # 3: accepted
        self.sizes = [2, 0, 0, 1]

    def test_only_the_answers_of_a_history_with_no_accepted_turn(self):
        out = neg_rows(self.sizes, self.rows_h)
        self.assertIsNone(out[0]); self.assertIsNone(out[2]); self.assertIsNone(out[3])
        self.assertEqual([r[4] for r in out[1]], ["ACTION: SUBMIT 1", "ACTION: SUBMIT 2"])   # the look is not pushed down

    def test_a_history_the_estep_fits_has_none(self):
        self.assertEqual(neg_rows([3, 0, 0, 1], self.rows_h)[0], None)
        self.assertIsNone(neg_rows([0, 1, 0, 1], self.rows_h)[1])       # sizes > 0: the E-step fits it, no push-down

    def test_a_repeated_turn_counts_once_and_at_most_neg_max(self):
        same = turn("Possible: row 1", "ACTION: SUBMIT 1", False)
        self.assertEqual(len(neg_rows([0], [[same, same, same]])[0]), 1)
        many = [turn("Possible: row 1", f"ACTION: SUBMIT {i}", False) for i in range(20)]
        out = neg_rows([0], [many])[0]
        self.assertEqual(len(out), T.NEG_MAX); self.assertEqual(out[0][4], "ACTION: SUBMIT 0")
        self.assertEqual(len({r[4] for r in out}), T.NEG_MAX)              # distinct, spread over the 20

    def test_the_action_line_first(self):
        rows_h = [[turn("ACTION: SUBMIT 1", "Possible: row 1", False)]]
        self.assertEqual(len(neg_rows([0], rows_h)[0]), 1)

    def test_keep_set_keeps_the_histories_to_push_down(self):
        self.assertEqual(keep_set(self.sizes, self.rows_h, 10), ([0, 3], 2))               # off: as before
        self.assertEqual(keep_set(self.sizes, self.rows_h, 10, neg=True), ([0, 1, 3], 2))  # the count stays the accepted ones
        self.assertEqual(keep_set(self.sizes, self.rows_h, 2, neg=True), ([0, 1], 2))


class ItemTests(unittest.TestCase):
    def counted(self, it):
        prefix, cont, (a, b) = it
        self.assertEqual(a, 0); return cont[b:]

    def test_only_the_answer_line_is_counted(self):
        r = turn("Possible: row 1", "ACTION: SUBMIT 1", False, z="I think")
        it = neg_item(r)
        self.assertEqual(it[0], "p"); self.assertEqual(it[1], "I think" + CLOSE + "Possible: row 1" + TERM + "ACTION: SUBMIT 1")
        self.assertEqual(self.counted(it), "ACTION: SUBMIT 1")                  # no terminator: the turn ends with <|im_end|>

    def test_the_action_line_first_ends_the_item(self):
        it = neg_item(turn("ACTION: SUBMIT 2", "Possible: row 2", False, z="t"))
        self.assertEqual(it[1], "t" + CLOSE + "ACTION: SUBMIT 2"); self.assertEqual(self.counted(it), "ACTION: SUBMIT 2")


class ThoughtTests(unittest.TestCase):
    """--neg_thought: one negative thought-block item per distinct thought of the rejected answers."""
    def test_one_item_per_thought_with_the_negative_weight(self):
        rs = [turn("Possible: row 1", "ACTION: SUBMIT 1", False, z="zA"), turn("Possible: row 2", "ACTION: SUBMIT 2", False, z="zA"),
              turn("Possible: row 3", "ACTION: SUBMIT 3", False, z="zB")]
        it, w = neg_thought_items(rs, 1.0, "blocks", 0.5)
        self.assertEqual([i[1] for i in it], ["zA" + CLOSE, "zB" + CLOSE])
        self.assertEqual(w, [-0.25, -0.25])                                  # -L / 2 thoughts x the draw weight 0.5
        self.assertTrue(all(i[2] is None and i[3] == len(i[1]) and i[4] == "blocks" for i in it))   # all thought, its own </think>

    def test_a_cut_thought_does_not_count_the_forced_close(self):
        r = ("p", "zz", "", "b", "ACTION: SUBMIT 1", 0.0, False, None, False)    # stopped = False: </think> forced by the protocol
        it, w = neg_thought_items([r], 2.0, "blocks")
        self.assertEqual(it[0][2], (2, 2 + len(CLOSE))); self.assertEqual(w, [-2.0])

    def test_none_gives_nothing(self):
        self.assertEqual(neg_thought_items(None, 1.0, "blocks"), ([], []))


class LossTests(unittest.TestCase):
    """loss_neg with a stand-in score(): one parameter per item is its answer's log-probability."""
    def run_loss(self, lps, weights, N=4):
        x = torch.nn.Parameter(torch.tensor(lps, dtype=torch.float64)); gd = [torch.zeros_like(x)]
        fake = lambda model, tok, items, chunk: x[[int(i[0]) for i in items]]
        with mock.patch.object(T, "score", fake):
            loss, ps = loss_neg(None, None, [(str(i), "", (0, 0)) for i in range(len(lps))], N, 2, weights, gd, [x])
        return loss, ps, gd[0], x

    def test_the_gradient_lowers_the_answer_and_lands_in_the_decision_buffer(self):
        loss, ps, g, x = self.run_loss([-0.1, -2.0, -8.0], [1.0, 1.0, 1.0])
        self.assertIsNone(x.grad)                                        # moved to the buffer
        self.assertTrue(all(v > 0 for v in g.tolist()[:2]))              # a step -g lowers log p
        p = torch.tensor([-0.1, -2.0, -8.0]).exp()
        coef = (p / (1 - p)).clamp(max=1.0)
        self.assertTrue(torch.allclose(g, (coef / 4).double(), atol=1e-6))   # w p / (1 - p) capped at 1, over N
        self.assertAlmostEqual(loss, float(-torch.log1p(-p).sum() / 4), places=5)
        self.assertEqual(len(ps), 3); self.assertAlmostEqual(ps[1], float(p[1]), places=6)

    def test_the_push_stops_when_the_answer_is_rare_and_is_capped_when_it_is_sure(self):
        _, _, g, _ = self.run_loss([-20.0, -1e-4], [1.0, 1.0], N=1)
        self.assertLess(float(g[0]), 1e-8)                               # p ~ 2e-9: no push
        self.assertAlmostEqual(float(g[1]), 1.0, places=6)               # p ~ 1: capped at NEG_CAP

    def test_the_weights_scale_it(self):
        _, _, g, _ = self.run_loss([-1.0, -1.0], [0.5, 0.25], N=1)
        self.assertAlmostEqual(float(g[0]) / float(g[1]), 2.0, places=6)


class FlagTests(unittest.TestCase):
    def test_the_method_has_it_off(self):
        self.assertNotIn("neg_answer", METHOD); self.assertNotIn("neg_thought", METHOD)

    def test_neg_thought_reaches_the_trainer(self):
        c = {"train": "t", "model": "qwen3-vl-8b-thinking", "train_tasks": ["minigrid_keycorridor"], "output": "/tmp/_neg_answer_test",
             "gpus": {"actor": "0", "learner": "0"}, "method": {"neg_thought": 1.0}}
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "Qwen3-VL-8B-Thinking"))
            with mock.patch.dict(os.environ, {"PL_MODELS": d}):
                r = commands(c, 0, stamp="0000-0000")
        for a in (r["learner"], r["actor"]):
            self.assertEqual(a[a.index("--neg_thought") + 1], "1.0"); self.assertNotIn("--neg_answer", a)

    def test_a_run_file_passes_it_to_the_trainer(self):
        c = {"train": "t", "model": "qwen3-vl-8b-thinking", "train_tasks": ["minigrid_keycorridor"], "output": "/tmp/_neg_answer_test",
             "gpus": {"actor": "0", "learner": "0"}, "method": {"neg_answer": 1.0}}
        with tempfile.TemporaryDirectory() as d:                         # the model's folder only (no weights are read)
            os.mkdir(os.path.join(d, "Qwen3-VL-8B-Thinking"))
            with mock.patch.dict(os.environ, {"PL_MODELS": d}):
                r = commands(c, 0, stamp="0000-0000")
        a = r["learner"]; i = a.index("--neg_answer")
        self.assertEqual(a[i + 1], "1.0")
        self.assertIn("--accept_only", a); self.assertEqual(a[a.index("--accept_only") + 1], "1")
        self.assertEqual(a[a.index("--m_step") + 1], "blocks")


if __name__ == "__main__":
    unittest.main()
