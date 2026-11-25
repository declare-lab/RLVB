"""The test split's move limit (plearn/tasks/__init__.py TEST_MOVES, TEST_TURNS): 150 moves and 152 turns in every test task, so
that no agent reaches the move limit; the training and validation splits keep each task's own limit; PL_TEST_MOVES=0 (here:
TEST_MOVES = 0) gives each task's own limit in the test split too, as in the paper's result tables. Pro-Worlds: test_proworlds."""
import importlib.util
import unittest
from unittest import mock

import plearn.tasks as _tasks


def needs(*modules):
    return unittest.skipUnless(all(importlib.util.find_spec(m) for m in modules), f"needs {', '.join(modules)}")


# (teacher module, kwargs, own T, own turns)
CASES = [("sokoban", {}, 15, 15), ("frozenlake", {}, 25, 25), ("minigrid", {"levels": ["S4R3"]}, 64, 65),
         ("minesweeper", {"levels": ["easy"]}, 32, 33), ("tabletop", {"render": False}, 5, 7)]


class DomainLimits(unittest.TestCase):
    def domain(self, name, kw, split):
        return importlib.import_module(f"plearn.tasks.{name}.teacher").Domain(split=split, **kw)

    @needs("gymnasium", "gym_sokoban", "minigrid", "jax", "popgym_arcade")
    def test_the_test_split_has_150_moves_and_152_turns(self):
        self.assertEqual((_tasks.TEST_MOVES, _tasks.TEST_TURNS), (150, 152))
        for name, kw, T, turns in CASES:
            with self.subTest(task=name):
                D = self.domain(name, kw, "test"); self.assertEqual((D.T, D.turns), (150, 152))
                for split in ("train", "validation"):
                    D = self.domain(name, kw, split); self.assertEqual((D.T, D.turns), (T, turns))

    @needs("gymnasium", "gym_sokoban", "minigrid", "jax", "popgym_arcade")
    def test_test_moves_0_gives_each_tasks_own_limit(self):
        with mock.patch.object(_tasks, "TEST_MOVES", 0), mock.patch.object(_tasks, "TEST_TURNS", 0):
            for name, kw, T, turns in CASES:
                with self.subTest(task=name):
                    D = self.domain(name, kw, "test"); self.assertEqual((D.T, D.turns), (T, turns))


class RoboQuestLimits(unittest.TestCase):
    """A stub bridge (no simulator): the test split gets 150 evidence actions and 152 decisions; the dev split keeps T and T + 2."""
    class Bridge:
        def call(self, **r):
            if r["op"] == "open": return dict(goal="g", intro="i", help="h", T=12, images=None)
            if r["op"] == "act": return dict(status="ok", reading="", images=None)
            return dict(outcome="x")

    def play(self, split):
        from plearn.tasks.roboquest import bench_env as BE
        with mock.patch.object(BE, "_bridge", lambda n, gpu, log: (self.Bridge(), 1)):
            ep = BE.Episode([split, "x"], "odd_parcel", version="text")
        n = 0
        while not ep.e.step("ACTION: WEIGH 1 VS 2"): n += 1
        return ep.e, n + 1

    def test_the_test_split(self):
        e, n = self.play("test")
        self.assertEqual((e.T0, e.T, e.limit, n, e.t), (12, 150, 152, 152, 150))
        self.assertEqual(e.result["errors"], ["no SUBMIT within 152 decisions"])

    def test_the_dev_split_and_test_moves_0(self):
        e, n = self.play("dev"); self.assertEqual((e.T, e.limit, n, e.t), (12, None, 14, 12))   # limit None = T + 2
        with mock.patch.object(_tasks, "TEST_MOVES", 0), mock.patch.object(_tasks, "TEST_TURNS", 0):
            e, n = self.play("test"); self.assertEqual((e.T, e.limit, n, e.t), (12, None, 14, 12))   # limit None = T + 2


if __name__ == "__main__":
    unittest.main()
