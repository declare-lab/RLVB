"""Dynamic sampling (plearn/method/trainer.py --set_keep): a phase's batches come only from the histories of its
sample set that give a gradient (the E-step fits them, an own turn takes an action the teacher accepts), the first L in the set's
order; any other history is skipped and not sampled again. The run files of the ablation study keep it off before its last row, and
the method's settings have it on."""
import unittest
from pathlib import Path

from plearn.api.registry import REPO_ROOT, load_yaml
from plearn.method.trainer import Board, by_task, keep_set
from plearn.train import METHOD

RUNS = Path(REPO_ROOT) / "configs" / "train"


def turns(*accepted):
    """The own turns of one history as the E-step writes them (prompt, thought, ..., R, stopped, readout, accepted)."""
    return [("p", "z", "", "b", "u", 0.0, True, None, a) for a in accepted]


class KeepTests(unittest.TestCase):
    def setUp(self):
        self.rows = [turns(False, True),        # 0: one own turn is accepted
                     turns(False, False),       # 1: none is: no gradient
                     turns(True),               # 2: accepted
                     turns(True),               # 3: accepted, but the E-step leaves it out (its lines have none of the posterior)
                     turns(True, True)]         # 4: accepted
        self.sizes = [2, 2, 1, 0, 2]

    def test_it_keeps_the_histories_with_a_gradient_in_order(self):
        self.assertEqual(keep_set(self.sizes, self.rows, 32), ([0, 2, 4], 3))

    def test_it_takes_the_first_L_and_does_not_fill_up(self):
        self.assertEqual(keep_set(self.sizes, self.rows, 2), ([0, 2], 3))
        self.assertEqual(keep_set([2], [turns(False, False)], 4), ([], 0))   # no history kept: the actor stops the run

    def test_rows_without_the_flag_count_as_accepted(self):
        self.assertEqual(keep_set([1], [[("p", "z", "", "b", "u", 0.0, True)]], 1), ([0], 1))

    def test_the_count_per_task(self):
        plans = [{"cls": "('minigrid', 2)"}, {"cls": "('tabletop-cups', 3)"}, {"cls": "('minigrid', 1)"}, {}]
        self.assertEqual(by_task(plans, range(4)), {"minigrid": 2, "tabletop-cups": 1, "?": 1})
        self.assertEqual(by_task(plans, [1]), {"tabletop-cups": 1})


class SettingTests(unittest.TestCase):
    def test_the_method_has_it_and_the_ablation_adds_it_last(self):
        self.assertEqual(METHOD["phase_set"], 1.5 * METHOD["set_keep"])
        self.assertEqual(load_yaml(RUNS / "abl_pl_4b.yaml")["method"]["set_keep"], 0)
        ds = load_yaml(RUNS / "abl_pl_4b_ds.yaml")["method"]
        self.assertEqual((ds["phase_set"], ds["set_keep"]), (48, 32))

    def test_the_share_reaches_wandb(self):
        self.assertEqual(sorted(Board.kept({"set_accept_share": 0.9, "set_drawn": 64}, "train")), ["train/set_accept_share"])
