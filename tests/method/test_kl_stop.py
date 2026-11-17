"""The limit of a phase's step (plearn/method/trainer.py --kl_stop: a phase must not go far in a wrong direction):
the probe takes own turns of the phase prior spread over the batch's histories, the KL is exact at each position, the phase's
last update is where a stopped phase hands over its model, and the run's metrics of the stop reach W&B; --kl_log measures the KL without the limit."""
import unittest

import torch

from plearn.method.trainer import Board, budget_share, kl_measured, kl_probe, kl_states, phase_end


def rows(n_thoughts, m, h):
    """The own turns of history h: n_thoughts thoughts with m decisions each, as the E-step writes them (thought-major)."""
    return [(f"p{h}", f"z{h}.{t}", "", f"b{d}", f"a{d}", 0.0, True) for t in range(n_thoughts) for d in range(m)]


class ProbeTests(unittest.TestCase):
    def test_the_probe_spreads_over_the_histories(self):
        r = rows(3, 2, 0) + rows(2, 2, 2) + rows(1, 2, 3)                 # history 1 is left out (no rows)
        sizes = [6, 0, 4, 2]
        idx = kl_probe(r, sizes, 5)
        self.assertEqual([r[i][1] for i in idx], ["z0.0", "z2.0", "z3.0", "z0.1", "z2.1"])   # first thoughts first, one per thought
        self.assertTrue(all(r[i][3] == "b0" for i in idx))                  # each thought with its first decision
        self.assertEqual(len(kl_probe(r, sizes, 100)), 6)                   # at most every thought once
        self.assertEqual(kl_probe(r, sizes, 0), []); self.assertEqual(kl_probe([], [0, 0], 4), [])

    def test_a_repeated_thought_counts_once(self):
        r = rows(2, 2, 0); r[2] = (*r[2][:1], "z0.0", *r[2][2:])           # the second thought's first turn has the first thought's text
        self.assertEqual([r[i][1] for i in kl_probe(r, [4], 8)], ["z0.0", "z0.1"])


class KLTests(unittest.TestCase):
    """kl_states: the exact KL of the next-token distributions at each position, from the two models' final states and their shared
    output layer, at the thought's or the decision's temperature."""
    def setUp(self):
        torch.manual_seed(0); self.head = torch.nn.Linear(8, 50, bias=False)
        self.h0, self.h1 = torch.randn(300, 8), torch.randn(300, 8); self.dec = torch.arange(300) >= 280   # the last 20 are the decision

    def test_it_is_the_kl_at_each_position(self):
        s, n = kl_states(self.head, self.h0, self.h1, self.dec, 0.6, 1.0, chunk=64)
        T = torch.where(self.dec, 1.0, 0.6)[:, None]
        p, q = torch.softmax(self.head(self.h0) / T, -1), torch.softmax(self.head(self.h1) / T, -1)
        self.assertAlmostEqual(s, float((p * (p.log() - q.log())).sum()), places=2); self.assertEqual(n, 300)

    def test_it_is_zero_for_the_same_model_and_never_negative(self):
        with torch.no_grad():
            self.assertAlmostEqual(kl_states(self.head, self.h0, self.h0, self.dec, 0.6, 0.6)[0], 0.0, places=6)
            for _ in range(5):
                self.assertGreaterEqual(kl_states(self.head, self.h0, self.h0 + 0.1 * torch.randn(300, 8), self.dec, 0.6, 0.6)[0], 0.0)


class PhaseTests(unittest.TestCase):
    def test_the_last_update_of_a_phase(self):
        self.assertEqual([phase_end(s, 25, 100) for s in (1, 7, 25, 26, 74, 75, 76, 100)], [25, 25, 25, 50, 75, 75, 100, 100])
        self.assertEqual(phase_end(60, 25, 70), 70)                         # the run ends inside the phase


class MetricTests(unittest.TestCase):
    def test_the_stop_reaches_wandb(self):
        kept = Board.kept({"kl_prior": 0.03, "kl_stop_left": 12, "loss_dec": 0.4}, "train")
        self.assertEqual(sorted(kept), ["train/kl_prior", "train/kl_stop_left", "train/loss_dec"])

    def test_unlimited_reasoning_has_no_budget(self):
        w, ok, stopped, L = [0.5, 0.3, 0.2], [True, True, False], [True, False, True], [30000, 9000, 100]
        self.assertEqual(budget_share(w, ok, stopped, L, -1), (0.5, 0.8))   # ended by itself at any length
        self.assertEqual(budget_share(w, ok, stopped, L, 8192), (0.0, 0.8))


class SwitchTests(unittest.TestCase):
    def test_the_kl_is_measured_with_the_limit_or_the_log(self):
        from types import SimpleNamespace as A                             # --kl_log 1: measured and logged, no phase stopped
        self.assertEqual([kl_measured(A(kl_stop=s, kl_log=g)) for s, g in ((0.0, 0), (0.015, 0), (0.0, 1), (0.015, 1))],
                         [False, True, True, True])
        self.assertTrue(kl_measured(A(kl_stop=0.015)))                      # the settings of runs before the switch


if __name__ == "__main__":
    unittest.main()
