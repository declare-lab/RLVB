"""The E-step weights of one history (plearn/method/trainer.py estep_weights): the softmax of the turn scores over the history's own
turns; with --accept_only only the turns whose action the teacher accepts share the weight, and a history with no
accepted turn gets none (an unverified action is never a positive target)."""
import unittest

import numpy as np

from plearn.method.trainer import estep_weights, no_target


class EStepWeightTests(unittest.TestCase):
    def test_the_plain_weights_are_the_softmax(self):
        R = np.array([-6.1, -37.3, -50.8, -53.6])
        w = estep_weights(R, [True] * 4, 1.0)
        self.assertTrue(np.allclose(w, np.exp(R - R.max()) / np.exp(R - R.max()).sum()))
        self.assertTrue(np.allclose(estep_weights(R, [True] * 4, 2.0, lam=0.5), estep_weights(R / 4, [True] * 4, 1.0)))

    def test_a_rejected_action_gets_no_weight(self):
        # update 2, history 6 of the smoke run: the best line ('C5, D4', one of the three cells) came with RIGHT, which the teacher
        # rejects; the turns with an accepted action (LEFT) had lines with none of the cells
        R = np.array([-38.0, -50.1, -53.0]); acc = [False, True, True]
        self.assertGreater(estep_weights(R, acc, 1.0)[0], 0.99)          # without the rule the rejected action takes the weight
        w = estep_weights(R, acc, 1.0, accept_only=True)
        self.assertEqual(w[0], 0.0); self.assertAlmostEqual(w.sum(), 1.0); self.assertGreater(w[1], w[2])

    def test_no_accepted_turn_gets_no_weight(self):
        # a history where every turn submits a cup already seen empty (log pi* -200): the plain weights make that rejected
        # action a positive target
        R = np.array([-3.0, -5.0])
        w = estep_weights(R, [False, False], 1.0, accept_only=True)
        self.assertEqual(len(w), 2); self.assertEqual(w.sum(), 0.0)
        self.assertAlmostEqual(estep_weights(R, [False, False], 1.0).sum(), 1.0)   # without the rule: the plain weights

    def test_a_left_out_history_has_no_weights(self):
        self.assertEqual(len(estep_weights([], [], 1.0, accept_only=True)), 0)

    def test_a_history_without_a_positive_target(self):
        plan = {"logp": {"ACTION: UP": -0.6, "ACTION: SUBMIT 1": -200.0}}      # accepted: within log 20 of the best action
        up, submit = {"A": -0.6}, {"A": -200.0}
        self.assertEqual(no_target([{**submit, "S": 0.0}] * 3, plan, 20.0, 50.0), "no_accept")
        # the first sample set, Sokoban history 8: every accepted turn kept 'target at C6' where the evidence shows E5 (S at the floor)
        self.assertEqual(no_target([{**up, "S": -50.0}] * 3, plan, 20.0, 50.0), "wrong_line")
        self.assertIsNone(no_target([{**up, "S": -50.0}, {**up, "S": -1.1}], plan, 20.0, 50.0))   # one accepted line keeps the posterior
        self.assertIsNone(no_target([{**up, "S": -50.0}], plan, 20.0, None))                       # the line is not scored


if __name__ == "__main__":
    unittest.main()
