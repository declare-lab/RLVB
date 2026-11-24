"""The two flags of the RLVB (GRPO) ablations: --act_score 0 (the turn score R without the action's log pi*) and
--accept_rule 0 (no min(A, 0) for a turn whose action the teacher rejects). CPU only, no model."""
import os, sys, unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from plearn.method.trainer import grpo_advantages, turn_score


class ActScoreTests(unittest.TestCase):
    def test_act_one_is_the_method(self):
        for mode, Sd in (("exact", None), ("reader", None), ("decision", -1.5), ("none", None)):
            self.assertEqual(turn_score(mode, 0.1, -2.0, -0.5, Sd, -3.0), turn_score(mode, 0.1, -2.0, -0.5, Sd, -3.0, 1))

    def test_act_zero_drops_the_action(self):
        self.assertAlmostEqual(turn_score("exact", 0.1, -2.0, -0.5, None, -3.0, 0), -2.5)
        self.assertAlmostEqual(turn_score("decision", 0.1, -2.0, -0.5, -1.5, -3.0, 0), -1.5 + 0.1 * -2.5)
        self.assertAlmostEqual(turn_score("decision", 0.1, -2.0, -0.5, -1.5, -3.0, 1) - turn_score("decision", 0.1, -2.0, -0.5, -1.5, -3.0, 0), -3.0)

    def test_line_score_none_is_the_action_only(self):
        self.assertAlmostEqual(turn_score("none", 0.1, -2.0, -0.5, None, -3.0), -3.0)


class AcceptRuleTests(unittest.TestCase):
    R = np.array([-1.0, -2.0, -4.0])

    def test_rejected_turn_capped_with_the_rule(self):
        A = grpo_advantages(self.R, [False, True, True])
        self.assertLessEqual(A[0], 0.0)
        self.assertGreater(grpo_advantages(self.R, None)[0], 0.0)

    def test_no_rule_equals_all_accepted(self):
        np.testing.assert_allclose(grpo_advantages(self.R, None), grpo_advantages(self.R, [True, True, True]))


if __name__ == "__main__":
    unittest.main()
