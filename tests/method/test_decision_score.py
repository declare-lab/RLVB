"""The decision score of a belief line (plearn/method/trainer.py --line_score decision): a line is scored by the
act the teacher takes when the line is its posterior. On every training teacher, along episodes that follow the teacher: the exact
line implies the teacher's own act and gets the highest score (the score is proper), a line outside the language gets the floor,
and where the answer is known a line that keeps every answer open implies another look and scores lower (MineSweeper aside: its
line_belief cannot widen the evidence's sets). The validation's decision_regret is 0 for the exact line, more
than 1 nat for that open line, and W&B records it. A test is skipped when its simulator is not installed (pip install -e ".[games]")."""
import importlib.util
import unittest
from types import SimpleNamespace

from plearn.method import estimator as E
from plearn.method.trainer import LOGP_FLOOR, WANDB_KEEP, Planner, decision_regret, decision_score
from plearn.train import METHOD


def needs(*modules):
    return unittest.skipUnless(all(importlib.util.find_spec(m) for m in modules), f"needs {', '.join(modules)}")


class ScoreTests(unittest.TestCase):
    """decision_score itself, with a stand-in teacher: the mean log pi* of the implied acts, the floor where there is none."""
    def setUp(self):
        self.plan = {"B": "b", "logp": {"look 1": -0.1, "look 2": -0.1, "submit 3": -4.0}, "waste_lp": -9.0}

    def planner(self, acts): return SimpleNamespace(implied=lambda D, plan, Bb: acts)

    def test_the_implied_acts_are_scored_by_the_teachers_policy(self):
        D = SimpleNamespace(line_belief=lambda B, text: "stated")
        self.assertAlmostEqual(decision_score(D, self.planner(["look 1", "look 2"]), self.plan, "Possible: 1, 2"), -0.1)
        self.assertAlmostEqual(decision_score(D, self.planner(["submit 3"]), self.plan, "Possible: 3"), -4.0)
        self.assertAlmostEqual(decision_score(D, self.planner(["look 9"]), self.plan, "Possible: 9"), -9.0)   # not legal: a wasted decision

    def test_no_belief_or_no_act_gets_the_floor(self):
        self.assertEqual(decision_score(SimpleNamespace(line_belief=lambda B, t: None), self.planner(["look 1"]), self.plan, "I think"), LOGP_FLOOR)
        self.assertEqual(decision_score(SimpleNamespace(line_belief=lambda B, t: "stated"), self.planner([]), self.plan, "Possible: 1"), LOGP_FLOOR)


class SettingTests(unittest.TestCase):
    def test_the_validation_reports_it_to_wandb(self):
        for k in ("decision_regret", "decision_ok", "decision_regret_sampled", "decision_ok_sampled"):
            self.assertTrue(WANDB_KEEP.match(f"eval/{k}"), k)

    def test_the_ablation_compares_it_with_the_full_method(self):
        from pathlib import Path
        from plearn.api.registry import REPO_ROOT, load_yaml
        runs = Path(REPO_ROOT) / "configs" / "train"
        dec, ds = load_yaml(runs / "abl_pl_4b_dec.yaml"), load_yaml(runs / "abl_pl_4b_ds.yaml")
        self.assertEqual((dec["method"]["line_score"], ds["method"]["line_score"] if "line_score" in ds["method"] else "exact"), ("decision", "exact"))
        drop = lambda m: {k: v for k, v in m.items() if k not in ("line_score", "line_mix")}
        self.assertEqual(drop(dec["method"]), drop(ds["method"]))

    def test_the_method_and_the_large_runs_have_it_and_the_ablation_rows_do_not(self):
        from pathlib import Path
        from plearn.api.registry import REPO_ROOT, load_yaml
        runs = Path(REPO_ROOT) / "configs" / "train"
        self.assertEqual((METHOD["line_score"], METHOD["line_mix"]), ("decision", 0.1))
        for f in sorted(runs.glob("*.yaml")):
            c = load_yaml(f)
            with self.subTest(run=f.name):
                m = {**METHOD, **(c.get("method") or {})}
                if f.name.startswith("abl_pl_4b"): self.assertEqual(m["line_score"], "decision" if "_dec" in f.name else "none" if "naive" in f.name else "exact")
                else: self.assertEqual(m["line_score"], "decision")


WIDE = {"minesweeper": None, "mastermind": "Possible: all", "minigrid": "Possible: all", "frozenlake": "Possible: any",
        "sokoban": "Possible: any", "tabletop": "Possible: all"}


class TeacherTests(unittest.TestCase):
    @needs("gymnasium", "gym_sokoban", "minigrid", "jax", "mani_skill")
    def test_on_every_training_teacher(self):
        from plearn.tasks.mixture import teachers_of
        for cls, args in teachers_of(["train_suite"]):
            name = cls.__module__.split(".")[2]
            with self.subTest(teacher=cls.__module__, args=str(args)):
                D = cls(**{**args, "split": "validation", **({"render": False} if name == "tabletop" else {})}); P = Planner(D, E.Reliability(fidelity=True, value=METHOD["value"]), METHOD["tau"], 2)
                eps = D.start(D.eval_specs(2))
                for e in eps:
                    wide_checked = False
                    for _ in range(12):
                        plan = P(e); best = max(plan["logp"].values())
                        self.assertAlmostEqual(decision_score(D, P, plan, D.belief_note(e)), best, places=6)   # proper: the exact line is best
                        self.assertAlmostEqual(decision_regret(D, P, plan, D.belief_note(e)), 0.0, places=6)
                        self.assertEqual(decision_score(D, P, plan, "I think the key is near."), LOGP_FLOOR)
                        if WIDE[name] and plan["B"].identified() and plan["answers"] and not wide_checked:
                            self.assertLess(decision_score(D, P, plan, WIDE[name]), best - 1.0)   # an open line where the answer is known
                            self.assertGreater(decision_regret(D, P, plan, WIDE[name]), 1.0)
                            wide_checked = True
                        act = max(plan["logp"], key=plan["logp"].get)
                        look = next((k for k, s in plan["looks"].items() if s == act), None)
                        if look is None or e.done: break                  # the teacher answers, or the game is over
                        D.do_look(e, look); e.t += 1
                        if hasattr(D, "after_turn"): D.after_turn(e)
                        if e.done: break
                D.close(eps)
