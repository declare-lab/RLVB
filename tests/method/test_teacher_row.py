"""--rho_exact and --teacher_row (plearn/method/trainer.py, plearn/method/estimator.py;). Both are off by default, and
off they change nothing.
--rho_exact 1: the teacher's answer accuracy is the exact posterior U(b) for the whole run: the table measured from the agent's own
answers keeps its prior, so neither its mean nor its optimistic bound moves with the agent's answers.
--teacher_row 1: at a history where no own turn takes an action the teacher accepts, the teacher's own belief line and action join
the group as one more turn (the tenth field of its row is True). The history is then kept (keep_set), the own turns get a negative
advantage against it, its own advantage is limited (--teacher_row_cap), and its thought is not trained."""
import unittest

import numpy as np

from plearn.method import estimator as E
from plearn.method.trainer import (LOGP_FLOOR, accepted, add_teacher_rows, cap_teacher_adv, grpo_advantages, grpo_weights, keep_set, neg_rows,
                                   own_rows, teacher_action, teacher_row)


class B:
    """A belief with a class and an exact accuracy of its best answer."""
    def __init__(self, u, k=("t", 2)): self.u, self.k = u, k
    def U(self): return self.u
    def cls(self): return self.k


def own(accepted, R=0.0):
    """An own turn as the E-step writes it (prompt, thought, prefix, line, action, R, stopped, readout, accepted)."""
    return ("p", "z", "p z", "b", "u", R, True, None, accepted)


def teacher(R=0.0):
    """The teacher's turn of --teacher_row: the same nine fields, accepted, and the tenth field True."""
    return ("p", "z", "p z", "b*", "u*", R, True, None, True, True)


class RhoExactTests(unittest.TestCase):
    def test_a_frozen_table_is_the_exact_posterior(self):
        for u in (0.2, 0.5, 0.8, 1.0):
            rel = E.Reliability(4.0, 1.0, fidelity=True, value="log"); rel.frozen = True
            for _ in range(50): rel.update(B(u), False, None, 0.1); rel.add(("t", 2), 0.3, 3)   # wrong answers of the agent
            self.assertEqual(rel.table(), {})
            self.assertAlmostEqual(rel.mean(B(u)), u); self.assertAlmostEqual(rel.ucb(B(u)), u)  # no optimism either

    def test_the_default_table_still_learns(self):
        rel = E.Reliability(4.0, 1.0, fidelity=True, value="log")
        self.assertFalse(rel.frozen)
        m0 = rel.mean(B(0.8))
        for i in range(20): rel.update(B(0.8), False, i, 0.1)
        self.assertLess(rel.mean(B(0.8)), m0)                         # the agent's wrong answers lower the teacher's value of answering
        rel.add(("t", 3), 1.0, 2); self.assertIn("('t', 3)", rel.table())


class TeacherRowTests(unittest.TestCase):
    def test_the_flag(self):
        self.assertFalse(teacher_row(own(True))); self.assertFalse(teacher_row(own(False)))
        self.assertFalse(teacher_row(("p", "z", "", "b", "u", 0.0, True)))     # an old row without the accepted flag
        self.assertTrue(teacher_row(teacher()))

    def test_the_teacher_action_is_the_most_likely_one(self):
        self.assertEqual(teacher_action({"tgt": {"ACTION: TURN LEFT": 0.1, "ACTION: SUBMIT 2": 0.7, "ACTION: FORWARD": 0.2}}), "ACTION: SUBMIT 2")
        self.assertEqual(teacher_action({"tgt": {"ACTION: A": 0.5, "ACTION: B": 0.5}}), "ACTION: A")   # the first of equal ones

    def test_the_history_is_kept_with_a_teacher_row(self):
        lost = [own(False), own(False)]
        self.assertEqual(keep_set([2], [lost], 8), ([], 0))            # today: no accepted own turn, the history is skipped
        self.assertEqual(keep_set([3], [lost + [teacher()]], 8), ([0], 1))
        self.assertEqual(neg_rows([0], [lost + [teacher()]]), [None])   # and it is not a --neg_answer history

    def test_the_own_turns_are_pushed_down_and_the_teacher_row_is_capped(self):
        R = np.array([-200.0] * 4 + [0.0]); acc = [False] * 4 + [True]; flags = [False] * 4 + [True]   # four moves at the answer turn
        A = grpo_advantages(R, acc, 20.0)
        self.assertTrue((A[:4] < 0).all()); self.assertAlmostEqual(A[4], 20.0)
        C = cap_teacher_adv(A, flags, 3.0)
        self.assertTrue(np.allclose(C[:4], A[:4])); self.assertAlmostEqual(C[4], 3.0)
        self.assertTrue(np.allclose(cap_teacher_adv(A, flags, 0.0), A))        # 0: no limit

    def test_without_a_teacher_row_nothing_changes(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            n = int(rng.integers(1, 9)); R = rng.normal(0, 3, n); acc = rng.random(n) < 0.5
            A = grpo_advantages(R, acc, 20.0)
            out = cap_teacher_adv(A, [False] * n, 3.0)
            self.assertTrue(np.array_equal(out, A))

    def test_a_group_of_equal_own_turns_gets_a_signal_only_with_the_teacher_row(self):
        R = np.array([-0.4] * 4); self.assertTrue((grpo_advantages(R, [True] * 4, 20.0) == 0).all())   # the same harmless step four times
        R2 = np.array([-20.0] * 4 + [0.0])                                    # all four rejected: with the teacher row they go down
        A = cap_teacher_adv(grpo_advantages(R2, [False] * 4 + [True], 20.0), [False] * 4 + [True], 3.0)
        self.assertTrue((A[:4] < 0).all()); self.assertGreater(A[4], 0)


class AddTeacherRowsTests(unittest.TestCase):
    """add_teacher_rows: the block of the trainer that adds the teacher's turn, with a stand-in for score_turn."""
    PLAN = {"anchor": "Possible: 2", "tgt": {"ACTION: TURN LEFT": 0.6, "ACTION: SUBMIT 2": 0.4},      # the target's mode and the accept
            "logp": {"ACTION: TURN LEFT": -4.0, "ACTION: SUBMIT 2": 0.0}}                          # rule's table can differ

    def score(self, r0, pl, b, u):
        A = pl["logp"].get(u, LOGP_FLOOR)
        return (r0[0], r0[1], r0[2], b, u, A, r0[6], None, accepted(A, pl, 20.0)), {"A": A, "S": 0.0, "S_teacher": 0.0}

    def crit(self, *A): return [{"A": a, "S": 0.0, "S_teacher": 0.0} for a in A]

    def test_the_action_comes_from_the_table_the_accept_rule_reads(self):
        self.assertEqual(teacher_action(self.PLAN), "ACTION: SUBMIT 2")          # not the mode of "tgt"
        self.assertTrue(accepted(self.PLAN["logp"][teacher_action(self.PLAN)], self.PLAN, 20.0))

    def test_a_history_with_no_accepted_own_turn_gets_one_row(self):
        rows_h = [[own(False, -4.0), own(False, -4.0)], [own(True, 0.0), own(False, -4.0)], []]
        crit_h = [self.crit(-4.0, -4.0), self.crit(0.0, -4.0), []]
        n = add_teacher_rows(rows_h, crit_h, [self.PLAN] * 3, 20.0, self.score)
        self.assertEqual(n, 1)
        self.assertEqual([len(r) for r in rows_h], [3, 2, 0]); self.assertEqual([len(c) for c in crit_h], [3, 2, 0])   # rows and scores stay aligned
        t = rows_h[0][-1]
        self.assertTrue(teacher_row(t)); self.assertTrue(t[8])
        self.assertEqual((t[1], t[3], t[4]), ("z", "Possible: 2", "ACTION: SUBMIT 2"))   # the first own thought, the teacher's line and action
        self.assertTrue(crit_h[0][-1]["teacher"]); self.assertNotIn("teacher", crit_h[0][0])
        self.assertEqual([own_rows(r) for r in rows_h], [2, 2, 0])            # the counts of own turns do not see it
        self.assertEqual(keep_set([3, 2, 0], rows_h, 8), ([0, 1], 2))

    def test_a_teacher_turn_that_is_not_accepted_is_not_added(self):
        bad = {"anchor": "Possible: 2", "tgt": {"ACTION: X": 1.0}, "logp": {"ACTION: X": 0.0}}
        rows_h = [[own(False, -9.0)]]; crit_h = [self.crit(-9.0)]
        n = add_teacher_rows(rows_h, crit_h, [bad], 20.0, lambda r0, pl, b, u: ((r0[0], r0[1], r0[2], b, u, -9.0, True, None, False), {"A": -9.0}))
        self.assertEqual((n, len(rows_h[0]), len(crit_h[0])), (0, 1, 1))

    def test_the_group_is_balanced_without_a_cap_and_small_in_the_batch(self):
        n = 64                                                               # two rounds of 32 own turns, all rejected, and the teacher row
        A = cap_teacher_adv(grpo_advantages(np.array([-20.0] * n + [0.0]), [False] * n + [True], 20.0), [False] * n + [True], 0.0)
        self.assertAlmostEqual(-A[:n].sum(), A[n])                           # the own turns lose in total what the teacher row gains
        rng = np.random.default_rng(0)
        normal = [grpo_advantages(rng.normal(0, 3, 32), rng.random(32) < 0.7, 20.0) for _ in range(31)]
        w, _ = grpo_weights(normal + [A], 32, True)
        share = np.abs(w[-1]).sum() / sum(np.abs(x).sum() for x in w)
        self.assertLess(share, 1 / 32)                                       # less than an average history of the batch


if __name__ == "__main__":
    unittest.main()
