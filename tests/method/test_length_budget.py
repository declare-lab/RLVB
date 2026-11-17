"""--length_budget (plearn/method/trainer.py length_reward: one rule for every model, the length term only where a thought
does not fit the test's budget): 0 (the default) gives the old term exactly; B > 0: a correct turn whose thought fits gets 0, one
over B a penalty from 0 at B to -0.5 at the history's longest thought; a wrong turn keeps min(0, lam). The flag reaches the trainer
from the run file's method (plearn/train.py) and is a run setting of a set (the set's own flags may differ)."""
import unittest

import numpy as np

from plearn.method.trainer import length_reward


def old_length_reward(lengths, ok):
    """length_reward before --length_budget (8662fb5), as it was."""
    L = np.asarray(lengths, dtype=float)
    if not len(L) or L.max() == L.min(): return np.zeros(len(L))
    lam = 0.5 - (L - L.min()) / (L.max() - L.min())
    return np.where(np.asarray(ok, dtype=bool), lam, np.minimum(0.0, lam))


class LengthBudgetTests(unittest.TestCase):
    def test_no_budget_is_the_old_term(self):
        rng = np.random.default_rng(0)
        for _ in range(2000):
            n = rng.integers(1, 9); L = rng.integers(100, 20000, n); ok = rng.random(n) < 0.6
            self.assertTrue(np.allclose(length_reward(L, ok, 0), old_length_reward(L, ok)))
            self.assertTrue(np.allclose(length_reward(L, ok), old_length_reward(L, ok)))

    def test_the_budget_rule(self):
        rng = np.random.default_rng(1); B = 8192
        for _ in range(2000):
            n = rng.integers(2, 9); L = rng.integers(100, 20000, n); ok = rng.random(n) < 0.6
            t, t0 = length_reward(L, ok, B), old_length_reward(L, ok)
            self.assertTrue(np.all(t[ok & (L <= B)] == 0))                         # a correct thought that fits: no term
            over = t[ok & (L > B)]; self.assertTrue(np.all((over < 0) & (over >= -0.5)))
            if (ok & (L == L.max()) & (L > B)).any(): self.assertAlmostEqual(float(t[ok & (L == L.max())].min()), -0.5)
            self.assertTrue(np.allclose(t[~ok], t0[~ok]))                          # a wrong turn: as before
            self.assertTrue(np.all(t <= 0))                                        # never a positive term with a budget

    def test_hand_cases(self):
        B = 8192
        self.assertTrue(np.array_equal(length_reward([4000, 6000, 8000], [1, 1, 1], B), np.zeros(3)))
        self.assertTrue(np.allclose(length_reward([4000, 10000, 16000], [1, 1, 1], B), [0.0, -0.5 * 1808 / 7808, -0.5]))
        self.assertTrue(np.allclose(length_reward([4000, 10000, 16000], [1, 0, 1], B), [0.0, min(0.0, 0.5 - 6000 / 12000), -0.5]))
        self.assertTrue(np.array_equal(length_reward([5000, 5000], [1, 1], B), np.zeros(2)))

    def test_the_flag(self):
        import inspect
        from plearn.method import trainer
        src = inspect.getsource(trainer)
        self.assertIn('ap.add_argument("--length_budget", type=int, default=0', src)
        self.assertIn("length_reward(l_, o_, args.length_budget)", src)
        self.assertIn('"length_budget"', src[src.index("free = {"):src.index("free = {") + 600])   # a run setting of a set


if __name__ == "__main__":
    unittest.main()
