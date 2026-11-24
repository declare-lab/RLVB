"""--tau_look: a separate temperature for pi*'s Boltzmann among the looks (plearn/method/trainer.py, the module text).
The answer-or-look sigmoid pC, the stop rule, the Boltzmann among the answers and wrong_lp keep --tau; a domain with
look_scale = False (mastermind) keeps --tau for its looks too; --tau_look_play 1 also moves the teacher's played moves (plan tgt).
Checks: with --tau_look 0 the plans and the scores equal those of commit c6914598 (the trainer and the estimator before the flag,
loaded through git show; skipped without git); a hand-computed small case; on real histories of each training game (skipped where
the simulator is missing) the order of the looks, pC and the answers' scores, ties, mastermind, tgt, and the premature-answer gap;
the run files looks_<model>.yaml = grpo_posterior_<model>.yaml plus --tau_look 0.02."""
import importlib.util
import math
import os
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from plearn.method import estimator as E
from plearn.method import trainer as T
from plearn.train import METHOD

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = "c6914598"
TAU, TL = METHOD["tau"], 0.02


_OLD = []


def old_modules():
    """The estimator and the trainer of commit BASE as modules (the trainer reads the old estimator); None without git."""
    if _OLD: return _OLD[0]
    try:
        src = {p: subprocess.run(["git", "show", f"{BASE}:plearn/method/{p}.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
               for p in ("estimator", "trainer")}
    except Exception:
        return None
    mods = {}
    with tempfile.TemporaryDirectory() as d:
        for p in ("estimator", "trainer"):
            f = os.path.join(d, f"old_{p}.py"); open(f, "w").write(src[p])
            spec = importlib.util.spec_from_file_location(f"old_{p}_{BASE}", f); m = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = m; spec.loader.exec_module(m); mods[p] = m
    mods["trainer"].E = mods["estimator"]
    _OLD.append(mods); return mods


def rel(): return E.Reliability(fidelity=METHOD["rho"] == "fidelity", value=METHOD["value"])


GAMES = {"mastermind": ("plearn.tasks.mastermind.teacher", None),
         "minesweeper": ("plearn.tasks.minesweeper.teacher", {"levels": ["easy"]}),
         "minigrid": ("plearn.tasks.minigrid.teacher", {"levels": ["S4R3"]}),
         "frozenlake": ("plearn.tasks.frozenlake.teacher", None),
         "sokoban": ("plearn.tasks.sokoban.teacher", None),
         "tabletop": ("plearn.tasks.tabletop.teacher", {"render": False})}   # the plans need no pictures (lamps, cups, blocks)


def game(name):
    """The game's Domain, or None when its simulator is missing."""
    mod, args = GAMES[name]
    try:
        import importlib
        D = importlib.import_module(mod).Domain(split="train", **(args or {}))
        D.close(D.start([D.sample(random.Random(0))]))           # the simulator runs (TableTop: sapien)
        return D
    except Exception:
        return None


def histories(D, n=40, steps=10, seed=0):
    """Episodes on fresh training specs, a random look at each step (as collect_histories' random half), up to n histories."""
    rng = random.Random(seed); k = 0
    while k < n:
        e = D.start([D.sample(rng)])[0]
        for _ in range(steps):
            if k >= n or getattr(e, "done", False): break
            looks = D.look_strings(e) if e.t < getattr(e, "_teacher", D).T else {}
            k += 1; yield e
            if not looks: break
            D.do_look(e, sorted(looks, key=str)[rng.randrange(len(looks))]); e.t += 1
        D.close([e])


def plans_at(D, Ps, n=40, seed=0):
    """At each history: the plan of every Planner in Ps (the same history)."""
    return [[P(e) for P in Ps] for e in histories(D, n, seed=seed)]


def levels(plan):
    return sorted({round(plan["logp"][s], 9) for s in plan["looks"].values()}, reverse=True)


class FakeRel:
    """Log values with accuracy 1: util(c, r) = log c + log r."""
    def mean(self, B): return 1.0
    def util(self, c, r): return math.log(c) + math.log(r) if r > 0 else -50.0


class FakeB:
    t = 0
    def identified(self): return False


class HandTests(unittest.TestCase):
    """Three looks at q, q - 0.02, q - 0.06 and one answer worth 0 (q = -0.05): pC = sigmoid(0.5) at tau 0.1 either way."""
    Q = {"a": -0.05, "b": -0.07, "c": -0.11}

    def plan(self, tl):
        looks = {k: "look " + k for k in self.Q}
        with mock.patch.object(E, "lookahead_Q", lambda B, rel, lam, T_, depth, acts=None: dict(self.Q)):
            tgt, Q, pC, qy = E.targets(FakeB(), looks, ["y"], {"y": "submit y"}, {"y": 1.0}, FakeRel(), 1.0, 10, TAU, 2, tl)
        P = T.Planner(None, FakeRel(), TAU, 2, tl or 0.0)
        return tgt, pC, P.act_logp(looks, {"y": "submit y"}, Q, pC, qy, tl)

    def test_the_hand_computed_plan(self):
        tgt0, pC0, lp0 = self.plan(None)
        tgt, pC, lp = self.plan(TL)
        pC_ = 1 / (1 + math.exp(-0.5))
        self.assertAlmostEqual(pC0, pC_, places=12); self.assertEqual(pC, pC0)          # pC unchanged
        self.assertEqual(lp["submit y"], lp0["submit y"]); self.assertAlmostEqual(lp["submit y"], math.log(pC_), places=12)
        z = 1 + math.exp(-1) + math.exp(-3)
        for k, d in (("a", 0), ("b", 1), ("c", 3)):                                    # (q_k - qL) / 0.02 = 0, -1, -3
            self.assertAlmostEqual(lp["look " + k], math.log(1 - pC_) - d - math.log(z), places=9)
            self.assertAlmostEqual(math.log(tgt["look " + k]), lp["look " + k], places=9)
        self.assertAlmostEqual(lp["look a"] - lp["look b"], 1.0, places=9)             # one step: 1 nat (0.2 at tau 0.1)
        self.assertAlmostEqual(lp0["look a"] - lp0["look b"], 0.2, places=9)
        acc = lambda lp: {s for s, v in lp.items() if T.accepted(v, {"logp": lp}, 20)}
        self.assertEqual(acc(lp), {"submit y", "look a", "look b"})                    # c (-4.32) is out of the window (-3.47)
        self.assertEqual(acc(lp0), {"submit y", "look a", "look b", "look c"})
        st = T.look_stats([{"looks": {k: "look " + k for k in self.Q}, "logp": lp, "tl_on": True}], 20)
        self.assertAlmostEqual(st["look_gap"], 1.0, places=9); self.assertAlmostEqual(st["look_accept"], 2 / 3); self.assertEqual(st["tau_look_on"], 1.0)

    def test_the_opt_out_and_the_defaults(self):
        P = T.Planner(None, FakeRel(), TAU, 2, TL)
        self.assertEqual(P.look_tau(types.SimpleNamespace()), TL)
        self.assertEqual(P.look_tau(types.SimpleNamespace(look_scale=False)), TAU)
        self.assertEqual(T.Planner(None, FakeRel(), TAU, 2).look_tau(types.SimpleNamespace()), TAU)
        from plearn.tasks.mastermind import teacher as MM
        self.assertIs(MM.Domain.look_scale, False)
        self.assertNotIn("tau_look", METHOD); self.assertNotIn("tau_look_play", METHOD)

    def test_the_saved_set_check_and_the_logs(self):
        with open(T.__file__, encoding="utf-8") as f: src = f.read()
        self.assertIn('(meta["args"].get("tau_look", 0.0), meta["args"].get("tau_look_play", 0)) == (args.tau_look, args.tau_look_play)', src)
        free = src[src.index('free = {"role"'):src.index("mine = vars(args)")]
        self.assertNotIn("tau_look", free)
        for k in ("look_gap", "look_accept", "tau_look_on"): self.assertTrue(T.WANDB_KEEP.match(f"train/{k}"), k)


@unittest.skipIf(old_modules() is None, "needs git and commit " + BASE)
class OffIdentityTests(unittest.TestCase):
    """--tau_look 0: every plan and score as at commit c6914598."""
    KEYS = ("tgt", "logp", "values", "waste", "waste_lp", "wrong_lp", "anchor", "exact", "best", "legal", "looks", "commits")

    @classmethod
    def setUpClass(cls): cls.old = old_modules()

    def test_the_estimator_targets(self):
        looks = {k: "look " + k for k in HandTests.Q}
        for tl in (None, 0, 0.0):
            with mock.patch.object(E, "lookahead_Q", lambda *a, **k: dict(HandTests.Q)), \
                 mock.patch.object(self.old["estimator"], "lookahead_Q", lambda *a, **k: dict(HandTests.Q)):
                new = E.targets(FakeB(), looks, ["y", "z"], {"y": "s y", "z": "s z"}, {"y": 0.7, "z": 0.3}, FakeRel(), 1.0, 10, TAU, 2, tl)
                old = self.old["estimator"].targets(FakeB(), looks, ["y", "z"], {"y": "s y", "z": "s z"}, {"y": 0.7, "z": 0.3}, FakeRel(), 1.0, 10, TAU, 2)
            self.assertEqual(new, old)

    def check_game(self, name, n=30):
        D = game(name)
        if D is None: self.skipTest(f"{name}: simulator missing")
        r = rel(); OT = self.old["trainer"]
        P_new, P_old = T.Planner(D, r, TAU, 2, 0.0, 0), OT.Planner(D, r, TAU, 2)
        P_play = T.Planner(D, r, TAU, 2, 0.0, 1)                   # --tau_look_play 1 without --tau_look: nothing changes
        k = 0
        for e in histories(D, n):
            a, b, c = P_new(e), P_old(e), P_play(e)
            for key in self.KEYS:
                self.assertEqual(a[key], b[key], f"{name}: {key}"); self.assertEqual(c[key], b[key], f"{name}: {key}")
            self.assertFalse(a["tl_on"])
            Dk = T.teacher_of(D, a)
            for s in list(a["legal"]) + ["action: nonsense", "submit nowhere"]:
                self.assertEqual(T.action_logp(D, s, a), OT.action_logp(D, s, b))
                self.assertEqual(T.accepted(T.action_logp(D, s, a), a, 20), OT.accepted(OT.action_logp(D, s, b), b, 20))
            if hasattr(D, "line_belief"):
                for s in [a["anchor"]]:
                    self.assertEqual(T.decision_score(Dk, P_new, a, s), OT.decision_score(Dk, P_old, b, s))
            k += 1
        self.assertGreater(k, 5)

    def test_mastermind(self): self.check_game("mastermind")
    def test_minesweeper(self): self.check_game("minesweeper")
    def test_minigrid(self): self.check_game("minigrid")
    def test_frozenlake(self): self.check_game("frozenlake")
    def test_sokoban(self): self.check_game("sokoban")
    def test_tabletop(self): self.check_game("tabletop", 15)


class RealHistoryTests(unittest.TestCase):
    """--tau_look 0.02 against 0 at the same histories."""
    def check_game(self, name, n=40):
        D = game(name)
        if D is None: self.skipTest(f"{name}: simulator missing")
        r = rel()
        Ps = [T.Planner(D, r, TAU, 2), T.Planner(D, r, TAU, 2, TL, 0), T.Planner(D, r, TAU, 2, TL, 1)]
        rows = plans_at(D, Ps, n); self.assertGreater(len(rows), 5)
        F = T.LOGP_FLOOR; tgt_moved = multi = 0
        for p0, p1, p2 in rows:
            commits = set(p0["commits"].values()); looks = list(p0["looks"].values())
            for s in commits: self.assertEqual(p1["logp"][s], p0["logp"][s], f"{name}: answer {s}")   # pC and the answers' scores
            self.assertEqual(p1["wrong_lp"], p0["wrong_lp"]); self.assertEqual(p1["values"], p0["values"])
            self.assertEqual(p1["tgt"], p0["tgt"], f"{name}: --tau_look_play 0 moved tgt")
            for x in looks:                                                  # the order of the looks, ties kept
                for y in looks:
                    if p0["logp"][x] == p0["logp"][y]: self.assertEqual(p1["logp"][x], p1["logp"][y], f"{name}: a tie broke")
                    elif p0["logp"][x] > p0["logp"][y]: self.assertTrue(p1["logp"][x] > p1["logp"][y] or p1["logp"][y] == F, f"{name}: order")
            if name == "mastermind":
                for key in ("tgt", "logp", "waste_lp", "wrong_lp"): self.assertEqual(p1[key], p0[key]); self.assertEqual(p2[key], p0[key])
                self.assertFalse(p1["tl_on"])
                continue
            if looks: self.assertTrue(p1["tl_on"])
            if len(levels(p0)) > 1: multi += 1; tgt_moved += p2["tgt"] != p0["tgt"]
            if looks and commits and not p0["B"].identified():               # the premature-answer gap: the same or larger
                g = lambda p: max(p["logp"][s] for s in looks) - max(p["logp"][s] for s in commits)
                self.assertGreaterEqual(g(p1), g(p0) - 1e-9, f"{name}: the gap to an early answer shrank")
        if name not in ("mastermind", "tabletop") and multi:
            self.assertGreater(tgt_moved, 0, f"{name}: --tau_look_play 1 never moved tgt")

    def test_mastermind(self): self.check_game("mastermind")
    def test_minesweeper(self): self.check_game("minesweeper")
    def test_minigrid(self): self.check_game("minigrid")
    def test_frozenlake(self): self.check_game("frozenlake")
    def test_sokoban(self): self.check_game("sokoban")
    def test_tabletop(self): self.check_game("tabletop", 15)

    def test_a_mixed_run_reads_the_episode_teacher(self):
        a, b = game("mastermind"), game("minesweeper")
        if a is None or b is None: self.skipTest("simulator missing")
        from plearn.tasks.mixture import Mixed
        D = Mixed([a, b]); r = rel(); P0, P1 = T.Planner(D, r, TAU, 2), T.Planner(D, r, TAU, 2, TL)
        seen = set()
        for e in histories(D, 80, steps=4, seed=3):
            p0, p1 = P0(e), P1(e); seen.add(p0["dom"])
            if p0["dom"] == "mastermind": self.assertEqual(p1["logp"], p0["logp"]); self.assertFalse(p1["tl_on"])
            elif p0["looks"]: self.assertTrue(p1["tl_on"])
        self.assertEqual(seen, {"mastermind", "minesweeper"})


class RunFileTests(unittest.TestCase):
    PAIRS = (("abl_grpo_nobelief_qwen3vl_4b_thinking", "looks_nobelief_qwen3vl_4b_thinking"),)   # (the run file, the same + tau_look 0.02)

    def test_the_run_files_add_tau_look_only(self):
        from plearn.train import commands, load
        for base, looks in self.PAIRS:
            with self.subTest(looks), tempfile.TemporaryDirectory() as d:
                for x in ("Qwen3-VL-8B-Thinking", "Qwen3-VL-4B-Thinking", "Qwen3.5-4B"): os.mkdir(os.path.join(d, x))
                env = {k: v for k, v in os.environ.items() if k not in ("PL_GPUS", "PL_SEQS", "PL_PRESENCE_PENALTY", "QWEN35_4B_PATH")}
                with mock.patch.dict(os.environ, {**env, "PL_MODELS": d, "PL_RUNS": d}, clear=True):
                    g = commands(load(os.path.join(ROOT, "configs", "train", f"{base}.yaml")), 0, stamp="0000-0000")
                    lk = commands(load(os.path.join(ROOT, "configs", "train", f"{looks}.yaml")), 0, stamp="0000-0000")
                names = {"--tag", "--out", "--queue", "--dump_em", "--dump_rollouts", "--wandb_name"}
                for role in ("actor", "learner"):
                    a, b = g[role], lk[role]
                    fa = a[a.index("plearn.method.trainer") + 1:]; fb = b[b.index("plearn.method.trainer") + 1:]
                    pa = {fa[i]: fa[i + 1] for i in range(len(fa) - 1) if fa[i].startswith("--") and not fa[i + 1].startswith("--")}
                    pb = {fb[i]: fb[i + 1] for i in range(len(fb) - 1) if fb[i].startswith("--") and not fb[i + 1].startswith("--")}
                    self.assertEqual(pb.pop("--tau_look"), "0.02")
                    self.assertEqual({k: v for k, v in pb.items() if k not in names}, {k: v for k, v in pa.items() if k not in names})
                    self.assertEqual(sorted(x for x in fb if x.startswith("--")), sorted([x for x in fa if x.startswith("--")] + ["--tau_look"]))
                    self.assertNotIn("--tau_look_play", fb)
                self.assertEqual(g["env"], lk["env"])


if __name__ == "__main__":
    unittest.main()
