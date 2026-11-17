"""The mirror term of the line's log score (plearn/method/trainer.py --line_wide): a line pays the floor m times
the share of its options that the teacher's exact line excludes, as it pays m times the posterior mass it leaves out. On every
training teacher, along episodes that follow the teacher: the exact line pays nothing; each earlier exact line, which every later
belief still keeps whole, pays its share 1 - |e| / |b|, the same width that the log score reads (-log |b| against -log |e|); a line
outside the language pays nothing more than the floor it has. MineSweeper's share, counted with binomials, agrees with a count over
every mine set of small boards. A test is skipped when its simulator is not installed (pip install -e ".[games]")."""
import importlib.util
import itertools
import math
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

from plearn.method import estimator as E
from plearn.method import trainer
from plearn.method.trainer import WANDB_KEEP, Planner, line_wide
from plearn.train import METHOD, flags


def needs(*modules):
    return unittest.skipUnless(all(importlib.util.find_spec(m) for m in modules), f"needs {', '.join(modules)}")


class ScoreTests(unittest.TestCase):
    """line_wide with a stand-in teacher that reads a line as the answers it names (the trainer's default reading)."""
    def setUp(self):
        self.D = SimpleNamespace(anchor_support=lambda t, answers, e=None: {a for a in answers if a in t.split(":", 1)[1]})
        self.plan = {"B": None, "anchor": "Possible: 1, 2", "answers": ["1", "2", "3", "4"]}

    def test_the_share_outside_the_exact_line_pays_the_floor(self):
        W = lambda line: line_wide(self.D, line, self.plan, 50.0)
        self.assertEqual(W("Possible: 1, 2"), 0.0)                      # the exact line
        self.assertEqual(W("Possible: 1"), 0.0)                         # inside it: the log score prices the mass it leaves out
        self.assertAlmostEqual(W("Possible: 1, 2, 3, 4"), -25.0)        # two times too wide: half of its options are excluded
        self.assertAlmostEqual(W("Possible: 1, 3"), -25.0)
        self.assertAlmostEqual(W("Possible: 3, 4"), -50.0)


class SettingTests(unittest.TestCase):
    def test_the_three_pl_run_files_turn_it_on(self):
        """line_wide is on in the run files of the 3 RLVB (EM) training scripts (scripts/pl_qwen3vl_4b_thinking.sh,
        pl_qwen3vl_8b_thinking.sh, pl_qwen35_4b.sh default RUN_FILE) and in the RLVB (GRPO) ablations' run files (the RLVB (EM) files
        with --update grpo); no other run file, and not in METHOD (the default stays off)."""
        from plearn.api.registry import REPO_ROOT, load_yaml
        runs = Path(REPO_ROOT) / "configs" / "train"
        on = {"suite_qwen3vl_4b_thinking", "suite_qwen3vl_8b_thinking", "suite_qwen35_4b_presence_8k",
              "grpo_posterior_qwen3vl_8b_thinking", "grpo_posterior_qwen3vl_4b_thinking", "grpo_posterior_qwen35_4b"}   # the RLVB (GRPO) ablations (the RLVB (EM) run files with --update grpo)
        got = {f.stem for f in runs.glob("*.yaml") if (load_yaml(f).get("method") or {}).get("line_wide") is True}
        self.assertEqual(got, on)
        for f in (runs / f"{n}.yaml" for n in on):
            self.assertEqual(load_yaml(f)["method"].get("line_score", METHOD["line_score"]), "decision")   # W enters R as line_mix (S + W)
        scripts = Path(REPO_ROOT) / "scripts"
        self.assertIn("NAME=suite_qwen3vl_4b_thinking", (scripts / "pl_qwen3vl_4b_thinking.sh").read_text())
        self.assertIn("NAME=suite_qwen3vl_8b_thinking", (scripts / "pl_qwen3vl_8b_thinking.sh").read_text())
        self.assertIn("RUN_FILE:-suite_qwen35_4b_presence_8k", (scripts / "pl_qwen35_4b.sh").read_text())
        self.assertNotIn("line_wide", METHOD)
        self.assertEqual(flags({"line_wide": True}), ["--line_wide"])
        self.assertIn('"--line_wide"', Path(trainer.__file__).read_text())
        self.assertTrue(WANDB_KEEP.match("train/crit_line_wide"))


class MineSetTests(unittest.TestCase):
    """MineSweeper's share against a count over every mine set: a line allows the sets of m mines on the hidden cells with its
    mines and none of its safe cells."""
    @staticmethod
    def count(B, line, other=None):
        hidden = [i for i in range(B.n * B.n) if not B.revealed >> i & 1]
        def allows(ln, S): return ln[0] & ~S == 0 and S & ln[1] == 0
        sets = [sum(1 << i for i in c) for c in itertools.combinations(hidden, B.m)]
        ins = [S for S in sets if allows(line, S)]
        return len(ins), (sum(not allows(other, S) for S in ins) if other is not None else None)

    def test_the_counted_share_is_the_share_of_mine_sets(self):
        from plearn.tasks.minesweeper.teacher import set_excess
        rng = random.Random(0); checked = 0
        for n, m in ((4, 2), (4, 3), (5, 4)):
            for _ in range(300):
                N = n * n; B = SimpleNamespace(n=n, m=m, revealed=sum(1 << i for i in rng.sample(range(N), rng.randint(0, N // 3))))
                pick = lambda most: sum(1 << i for i in rng.sample(range(N), rng.randint(0, most)))   # a few stated cells
                line, exact = (pick(1), pick(3)), (pick(2), pick(4))
                if line[0] & B.revealed:                                   # a mine on a revealed cell: the line allows no set
                    self.assertEqual(set_excess(B, line, exact), 0.0); continue
                k, out = self.count(B, line, (exact[0], exact[1] & ~B.revealed))
                self.assertAlmostEqual(set_excess(B, line, exact), out / k if k else 0.0, places=12)
                checked += k > 0 and 0 < out < k
        self.assertGreater(checked, 100)
        self.assertEqual(set_excess(B, None, exact), 0.0)


class TeacherTests(unittest.TestCase):
    @needs("gymnasium", "gym_sokoban", "minigrid", "jax", "mani_skill")
    def test_on_every_training_teacher(self):
        from plearn.tasks.mixture import teachers_of
        wider = {}
        for cls, args in teachers_of(["train_suite"]):
            name = cls.__module__.split(".")[2]
            with self.subTest(teacher=cls.__module__, args=str(args)):
                D = cls(**{**args, "split": "validation", **({"render": False} if name == "tabletop" else {})}); P = Planner(D, E.Reliability(fidelity=True, value=METHOD["value"]), METHOD["tau"], 2)
                eps = D.start(D.eval_specs(2)); key = f"{name} {args}"; wider.setdefault(key, 0)
                for e in eps:
                    lines = []
                    for _ in range(12):
                        plan = P(e); B, L = plan["B"], D.belief_note(e)
                        self.assertEqual(plan["anchor"], L)
                        self.assertEqual(D.line_excess(B, L, L), 0.0)                    # the exact line pays nothing
                        self.assertEqual(line_wide(D, L, plan, 50.0), 0.0)
                        self.assertEqual(D.line_excess(B, "I think the key is near.", L), 0.0)   # outside the language: the floor only
                        for old in lines:                                              # an earlier exact line: wider or the same
                            x = D.line_excess(B, old, L)
                            gap = D.belief_score(B, old, 50.0) - D.belief_score(B, L, 50.0)   # -log |b| + log |e|: both keep all of P*
                            self.assertAlmostEqual(x, 1.0 - math.exp(gap), places=9)
                            self.assertAlmostEqual(line_wide(D, old, plan, 50.0), -50.0 * x, places=9)
                            wider[key] += x > 1e-9
                        lines.append(L)
                        act = max(plan["logp"], key=plan["logp"].get)
                        look = next((k for k, s in plan["looks"].items() if s == act), None)
                        if look is None or e.done: break                  # the teacher answers, or the game is over
                        D.do_look(e, look); e.t += 1
                        if hasattr(D, "after_turn"): D.after_turn(e)
                        if e.done: break
                D.close(eps)
        print("wider earlier lines by teacher:", wider)
        self.assertGreaterEqual(sum(v > 0 for v in wider.values()), len(wider) - 1)   # nearly every teacher meets a wider line


if __name__ == "__main__":
    unittest.main()
