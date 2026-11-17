"""The thought's length in the E-step (plearn/method/trainer.py). --think_cost: the price of a thought by phase, and the length
tie-break: among the samples that the teacher scores alike the shorter thought gets the weight; every group keeps its total, so
the weight of each belief and action is the E-step's as before. --length_reward: Kimi k1.5's length term in R,
the shorter correct thought gets more, a wrong turn never gets a positive term; budget_share, the budget measure."""
import unittest

import numpy as np

from plearn.method.trainer import budget_share, estep_weights, length_reward, length_tiebreak, think_cost_at


class ThinkCostTests(unittest.TestCase):
    def test_the_price_rises_by_phase(self):
        spec = "0,0.025,0.05,0.1"
        self.assertEqual([think_cost_at(spec, s, 25) for s in (1, 25, 26, 50, 51, 76, 100, 140)], [0, 0, 0.025, 0.025, 0.05, 0.1, 0.1, 0.1])
        self.assertEqual(think_cost_at("", 7, 25), 0.0); self.assertEqual(think_cost_at("0.05", 3, 0), 0.05)

    def test_alike_samples_share_their_weight_by_length(self):
        R = np.array([0.0, 0.0, 0.0, -3.3, -3.3, -200.0])                 # three alike, two alike, one alone
        w = np.exp(R) / np.exp(R).sum(); L = [16000, 2000, 8000, 500, 9000, 100]
        v = length_tiebreak(w, R, L, 0.1)
        for r in (0.0, -3.3, -200.0):                                      # each belief and action keeps its weight
            self.assertAlmostEqual(v[R == r].sum(), w[R == r].sum())
        self.assertTrue(v[1] > v[2] > v[0]); self.assertGreater(v[3], v[4])  # the shorter thought gets more
        self.assertAlmostEqual(v[1] / v[0], np.exp(0.1 * 14))               # exp(-kappa * tokens / 1000)
        self.assertTrue(np.array_equal(length_tiebreak(w, R, L, 0.0), w))  # no price: the E-step's weights


class LengthRewardTests(unittest.TestCase):
    def test_the_two_rules(self):
        L = [2000, 9000, 16384, 2000, 16384]; ok = [True, True, True, False, False]
        r = length_reward(L, ok)
        self.assertTrue(r[0] > r[1] > r[2])                                # a correct turn: the shorter, the more
        self.assertAlmostEqual(r[0], 0.5); self.assertAlmostEqual(r[2], -0.5)
        self.assertEqual(r[3], 0.0)                                        # a short wrong turn: no reward
        self.assertAlmostEqual(r[4], -0.5)                                 # a long wrong turn: a penalty
        self.assertTrue(np.array_equal(length_reward([5000] * 3, [True, False, True]), np.zeros(3)))   # one length: no term
        self.assertEqual(len(length_reward([], [])), 0)

    def test_the_weight_moves_among_correct_turns_only(self):
        # two correct turns with the same score, a short one and one cut at the cap, and a short turn whose line misses the posterior
        R = np.array([-1.0, -1.0, -51.0]); L = [3000, 16384, 3000]; ok = [True, True, False]
        w0 = estep_weights(R, [True] * 3, 1.0, accept_only=True)
        w = estep_weights(R + 4.0 * length_reward(L, ok), [True] * 3, 1.0, accept_only=True)
        self.assertAlmostEqual(w0[0], w0[1]); self.assertGreater(w[0] / w[1], np.exp(3.9))       # C = 4: e^4 between the two
        self.assertLess(w[2], 1e-20); self.assertAlmostEqual(w[:2].sum(), 1.0)                 # the wrong line stays out

    def test_the_budget_measure(self):
        w = [0.5, 0.3, 0.15, 0.05]; ok = [True, True, False, True]; stopped = [True, False, True, True]; L = [4000, 16384, 3000, 9000]
        qb, qc = budget_share(w, ok, stopped, L, 8192)
        self.assertAlmostEqual(qb, 0.5)                                    # correct, ended by itself, within 8,192
        self.assertAlmostEqual(qc, 0.85)                                   # correct at any length
        self.assertIsNone(budget_share([0.0, 0.0], [True, True], [True, True], [10, 10], 8192))   # a history with no weight


if __name__ == "__main__":
    unittest.main()
