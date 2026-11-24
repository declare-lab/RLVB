"""--stall_cap (plearn/method/trainer.py cap_stalls, look_info, Planner; the five game teachers' Belief.look_info;), part
3 of the reward repair: a look that is neither one of the teacher's best looks nor one with real expected evidence, and an action that
is not legal, get log pi* <= best - C, so that a stalling step is rejected. Off by default, and off it changes nothing: the plans
equal those of commit f23d8f6d (the trainer before --stall_cap; loaded through git show, skipped without git or the commit).
Checks: a hand-made plan; the rule only lowers a value and keeps the best; on real histories of each training game (skipped where
the simulator is missing) a look with evidence and the best looks keep their values, the teacher's moves (tgt) and its most likely
action are unchanged, and the steps that show nothing (MineSweeper cursor moves, MiniGrid turning on the spot, a step into a wall)
have no evidence; --stall_play 1 (the teacher's own moves from the capped scores: capped_tgt) changes tgt only, by exp(new - old)
on the lowered looks; the run file phase5_nobelief_trow_stall_qwen35_4b = phase5_nobelief_trow_qwen35_4b plus stall_cap 3.5 and
stall_play 1."""
import importlib.util
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from plearn.method import estimator as E
from plearn.method import trainer as T
from plearn.train import METHOD

ROOT = Path(__file__).resolve().parents[2]
BASE = "f23d8f6d"                                                     # the trainer before --stall_cap
CAP, ACCEPT = 3.5, 20.0

GAMES = {"mastermind": ("plearn.tasks.mastermind.teacher", None),
         "minesweeper": ("plearn.tasks.minesweeper.teacher", {"levels": ["easy"]}),
         "minigrid": ("plearn.tasks.minigrid.teacher", {"levels": ["S4R3"]}),
         "frozenlake": ("plearn.tasks.frozenlake.teacher", None),
         "sokoban": ("plearn.tasks.sokoban.teacher", None),
         "tabletop": ("plearn.tasks.tabletop.teacher", {"render": False})}


def game(name):
    """The game's Domain, or None when its simulator is missing."""
    mod, args = GAMES[name]
    try:
        D = importlib.import_module(mod).Domain(split="train", **(args or {}))
        D.close(D.start([D.sample(random.Random(0))]))
        return D
    except Exception:
        return None


def rel():
    r = E.Reliability(fidelity=METHOD["rho"] == "fidelity", value=METHOD["value"]); r.frozen = True; return r   # --rho_exact 1


def histories(D, n=30, steps=12, seed=0):
    """Histories of fresh training specs, a random look at each step."""
    rng = random.Random(seed); k = 0
    while k < n:
        e = D.start([D.sample(rng)])[0]
        for _ in range(steps):
            if k >= n or getattr(e, "done", False): break
            looks = D.look_strings(e) if e.t < D.T else {}
            k += 1; yield e
            if not looks: break
            D.do_look(e, sorted(looks, key=str)[rng.randrange(len(looks))]); e.t += 1
        D.close([e])


_OLD = []


def old_trainer():
    """The trainer of commit BASE as a module, or None without git."""
    if _OLD: return _OLD[0]
    try:
        src = subprocess.run(["git", "show", f"{BASE}:plearn/method/trainer.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception:
        return None
    # that trainer also had an option (--action_value full, off by default) whose module this tree does not have; the line
    # that imports it is left out (the option is off in every plan compared here)
    src = src.replace("from plearn.method import fullvalue as FVM", "FVM = None")
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "old_trainer.py"); open(f, "w").write(src)
        spec = importlib.util.spec_from_file_location(f"old_trainer_{BASE}", f); m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m; spec.loader.exec_module(m)
    _OLD.append(m); return m


class HandTests(unittest.TestCase):
    """An answer, the best look, a look that shows nothing, a look with evidence, a look already far down; and waste_lp."""
    LP = {"submit y": -0.5, "look a": -1.0, "look b": -1.2, "look c": -1.3, "look d": -10.0}
    LOOKS = {"a": "look a", "b": "look b", "c": "look c", "d": "look d"}
    INFO = {"a": 0.0, "b": 0.0, "c": 0.4, "d": 0.0}

    def capped(self, cap=CAP, info=None):
        calls = []
        def f(a): calls.append(a); return (info or self.INFO)[a]
        return T.cap_stalls(dict(self.LP), -1.1, self.LOOKS, cap, f), calls

    def test_the_stalling_look_and_the_illegal_action_are_rejected(self):
        (lp, w, st), calls = self.capped()
        self.assertEqual(lp["look b"], -0.5 - CAP); self.assertEqual(w, -0.5 - CAP)
        for s in ("submit y", "look a", "look c", "look d"): self.assertEqual(lp[s], self.LP[s], s)   # the answer, the best look, evidence, far down
        self.assertEqual(st, {"logp": {"look b": -1.2}, "waste_lp": -1.1})
        self.assertEqual(sorted(calls), ["b", "c"])                     # info only where the cap would lower the value
        acc = lambda v, lp_: T.accepted(v, {"logp": lp_}, ACCEPT)
        self.assertTrue(acc(self.LP["look b"], self.LP) and acc(-1.1, self.LP))   # before: accepted (0.7 and 0.6 nats from the best)
        self.assertFalse(acc(lp["look b"], lp) or acc(w, lp))                     # after: rejected
        self.assertTrue(acc(lp["look a"], lp) and acc(lp["look c"], lp))
        self.assertTrue(T.stall_rejected(lp["look b"], {"logp": lp}, ACCEPT, self.LP["look b"]))
        self.assertFalse(T.stall_rejected(lp["look a"], {"logp": lp}, ACCEPT, self.LP["look a"]))
        plan = {"logp": lp, "waste_lp": w, "stall": st}
        self.assertEqual(T.uncapped(plan)["logp"], self.LP); self.assertEqual(T.uncapped(plan)["waste_lp"], -1.1)

    def test_off_and_unknown_evidence(self):
        (lp, w, st), calls = self.capped(cap=0.0)
        self.assertEqual((lp, w, st, calls), (self.LP, -1.1, None, []))
        (lp, w, st), _ = self.capped(info={a: None for a in self.INFO})   # a belief that cannot say: the look counts as evidence
        self.assertEqual(lp, self.LP); self.assertEqual(w, -0.5 - CAP)        # (an action that is not legal is still capped)
        self.assertEqual(T.cap_stalls({}, -2.0, {}, CAP, lambda a: 0.0), ({}, -2.0, None))
        p = {"logp": self.LP, "waste_lp": -1.1}; self.assertIs(T.uncapped(p), p)        # nothing lowered: the plan itself

    def test_the_teacher_plays_by_the_capped_scores(self):
        (lp, w, st), _ = self.capped()
        tgt = {"submit y": 0.3, "look a": 0.3, "look b": 0.25, "look c": 0.1, "look d": 0.05}
        t2 = T.capped_tgt(tgt, lp, st)
        self.assertAlmostEqual(sum(t2.values()), 1.0)
        self.assertAlmostEqual(t2["look b"] / t2["look a"], 0.25 / 0.3 * math.exp(lp["look b"] - self.LP["look b"]))
        for s in ("submit y", "look c", "look d"): self.assertAlmostEqual(t2[s] / t2["look a"], tgt[s] / tgt["look a"])   # the others keep their ratios
        self.assertIs(T.capped_tgt(tgt, self.LP, {"logp": {}, "waste_lp": None}), tgt); self.assertIs(T.capped_tgt(tgt, self.LP, None), tgt)

    def test_it_only_lowers_and_keeps_the_best(self):
        rng = np.random.default_rng(0)
        for _ in range(2000):
            n = int(rng.integers(1, 7)); looks = {i: f"look {i}" for i in range(n)}
            lp = {s: float(v) for s, v in zip(looks.values(), np.round(rng.normal(-3, 3, n), 1))}
            if rng.random() < 0.7: lp["submit"] = float(rng.normal(-1, 2))
            w = float(rng.normal(-3, 3)); info = {i: float(rng.choice([0.0, 0.0, 0.3])) for i in looks}
            out, w2, st = T.cap_stalls(lp, w, looks, CAP, info.get)
            self.assertEqual(max(out.values()), max(lp.values()))
            self.assertTrue(all(out[s] <= lp[s] for s in lp)); self.assertLessEqual(w2, w)
            top = max(lp[s] for s in looks.values()); line = max(lp.values()) - CAP
            for i, s in looks.items():
                if out[s] < lp[s]: self.assertTrue(lp[s] < top and info[i] == 0.0 and out[s] == line)
                else: self.assertTrue(lp[s] >= top or info[i] > 0 or lp[s] <= line)
            self.assertEqual(T.uncapped({"logp": out, "waste_lp": w2, "stall": st})["logp"], lp)


class GameTests(unittest.TestCase):
    """On real histories of each training game: Planner(stall_cap 3.5) against Planner(stall_cap 0) and the trainer of BASE."""

    def plans(self, name, n=30, extra=None):
        """Per history: (the plan off, the plan with the cap, the plan of BASE, look_info of each look, extra(D, e, plan)); all made
        at the history itself (the episode goes on afterwards)."""
        D = game(name)
        if D is None: self.skipTest(f"{name}: no simulator")
        r = rel(); P0, P1 = T.Planner(D, r, METHOD["tau"], 2), T.Planner(D, r, METHOD["tau"], 2, stall_cap=CAP)
        old = old_trainer(); Po = old.Planner(D, r, METHOD["tau"], 2) if old else None
        out = []
        for e in histories(D, n):
            a, b = P0(e), P1(e); o = Po(e) if Po else None
            out.append((a, b, o, {k: T.look_info(D.belief(e), k) for k in a["looks"]}, extra(D, e, a) if extra else None))
        return out

    def check(self, name, extra=None):
        rows = self.plans(name, extra=extra); n_cap = 0
        for a, b, o, info, _ in rows:
            self.assertNotIn("stall", a)
            if o is not None:                                             # off: the plans of the trainer before the switch
                self.assertEqual(set(a), set(o))
                for k in ("logp", "waste_lp", "wrong_lp", "tgt", "values", "anchor", "exact", "legal"): self.assertEqual(a[k], o[k], k)
            for k in ("tgt", "values", "wrong_lp", "anchor", "legal"): self.assertEqual(a[k], b[k], k)   # the teacher's own moves stay
            self.assertEqual(max(a["logp"].values()), max(b["logp"].values()))
            self.assertEqual(T.teacher_action(a), T.teacher_action(b))
            self.assertEqual(T.uncapped(b)["logp"], a["logp"]); self.assertEqual(T.uncapped(b)["waste_lp"], a["waste_lp"])
            top = max((a["logp"][s] for s in a["looks"].values()), default=None); line = max(a["logp"].values()) - CAP
            for k, s in a["looks"].items():
                self.assertGreaterEqual(info[k], 0.0)
                if b["logp"][s] < a["logp"][s]:
                    n_cap += 1; self.assertTrue(info[k] <= T.STALL_INFO and a["logp"][s] < top); self.assertEqual(b["logp"][s], line)
                    self.assertFalse(T.accepted(b["logp"][s], b, ACCEPT))
                else:
                    self.assertTrue(info[k] > T.STALL_INFO or a["logp"][s] >= top or a["logp"][s] <= line)
            self.assertEqual(b["waste_lp"], min(a["waste_lp"], line))
        return rows, n_cap

    def test_mastermind(self):
        h = lambda p: -sum(v * math.log(v) for v in p.values() if v > 0)
        def generic(D, e, a):                                             # its own look_info is the generic one (outcomes, post)
            B = D.belief(e); return {k: h(B.post()) - sum(p * h(b_.post()) for p, b_ in B.outcomes(k)) for k in list(a["looks"])[:20]}
        rows, _ = self.check("mastermind", generic)
        for a, b, o, info, gen in rows[:8]:
            for k, v in gen.items(): self.assertAlmostEqual(info[k], max(0.0, v), places=9)

    def test_minesweeper(self):
        from plearn.tasks.minesweeper.teacher import FIRE
        rows, n_cap = self.check("minesweeper")
        for a, b, o, info, _ in rows:
            for k in a["looks"]:
                if k != FIRE: self.assertEqual(info[k], 0.0)              # a cursor move shows nothing
        self.assertTrue(any(info.get(FIRE, 0.0) > 0 for *_, info, _ in rows))
        self.assertGreater(n_cap, 0)                                      # cursor moves off the way to the best cell

    def test_minigrid(self):
        from plearn.tasks.minigrid.teacher import LEFT
        self.check("minigrid")
        D = game("minigrid"); e = D.start([D.sample(random.Random(3))])[0]
        for _ in range(4): D.do_look(e, LEFT); e.t += 1                   # every direction seen from here
        self.assertEqual(T.look_info(D.belief(e), LEFT), 0.0)             # turning on the spot shows nothing new
        D.close([e])

    def test_frozenlake(self):
        rows, _ = self.check("frozenlake", lambda D, e, a: {k: e.move(e.pos, k) == e.pos for k in a["looks"]})
        self.assertTrue(any(any(st.values()) for *_, st in rows))
        for a, b, o, info, st in rows:
            for k in a["looks"]:
                if st[k]: self.assertEqual(info[k], 0.0)                 # a step into the edge

    def test_sokoban(self):
        rows, _ = self.check("sokoban", lambda D, e, a: {k: e.graph.step(e.state, k) == e.state for k in a["looks"]})
        self.assertTrue(any(any(st.values()) for *_, st in rows))
        for a, b, o, info, st in rows:
            for k in a["looks"]:
                if st[k]: self.assertEqual(info[k], 0.0)                 # a step into a wall

    def test_stall_play_moves_the_teacher_only(self):
        n = 0
        for name in ("minesweeper", "frozenlake", "sokoban", "mastermind"):
            D = game(name)
            if D is None: continue
            r = rel(); P1 = T.Planner(D, r, METHOD["tau"], 2, stall_cap=CAP); P2 = T.Planner(D, r, METHOD["tau"], 2, stall_cap=CAP, stall_play=1)
            for e in histories(D, 20):
                b, c = P1(e), P2(e)
                for k in ("logp", "waste_lp", "wrong_lp", "values", "stall", "anchor"): self.assertEqual(b[k], c[k], k)   # the scores stay
                self.assertEqual(c["tgt"], T.capped_tgt(b["tgt"], b["logp"], b["stall"])); self.assertAlmostEqual(sum(c["tgt"].values()), 1.0)
                rej = lambda t: sum(p for s, p in t.items() if not T.accepted(b["logp"].get(s, b["waste_lp"]), b, ACCEPT))
                self.assertLessEqual(rej(c["tgt"]), rej(b["tgt"]) + 1e-12)            # it draws fewer rejected moves
                if b["stall"]["logp"]: n += 1
                else: self.assertEqual(c["tgt"], b["tgt"])
        if n == 0: self.skipTest("no simulator, or no lowered look")

    def test_tabletop(self):
        rows, n_cap = self.check("tabletop")
        self.assertEqual(n_cap, 0)                                        # every look there shows something


class SwitchTests(unittest.TestCase):
    def test_the_flag_the_log_and_the_set_guard(self):
        src = (ROOT / "plearn" / "method" / "trainer.py").read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--stall_cap", type=float, default=0.0', src)
        self.assertIn('meta["args"].get("stall_cap", 0.0)', src); self.assertIn('meta["args"].get("stall_play", 0)', src)   # PL_SET_FROM
        self.assertIn('ap.add_argument("--stall_play", type=int, choices=(0, 1), default=0', src)
        self.assertTrue(T.WANDB_KEEP.match("train/stall_rejected"))
        self.assertNotIn("stall_cap", METHOD); self.assertNotIn("stall_play", METHOD)

    def test_the_run_file_is_the_trow_run_plus_the_cap(self):
        from plearn.train import commands, load
        new, base = "phase5_nobelief_trow_stall_qwen35_4b", "phase5_nobelief_trow_qwen35_4b"
        cn, cb = load(str(ROOT / "configs" / "train" / f"{new}.yaml")), load(str(ROOT / "configs" / "train" / f"{base}.yaml"))
        self.assertEqual(cn["method"], {**cb["method"], "stall_cap": 3.5, "stall_play": 1})
        names = {"--tag", "--out", "--queue", "--dump_em", "--dump_rollouts", "--wandb_name"}
        def flags(argv):
            a = argv[argv.index("plearn.method.trainer") + 1:]; out = {}; i = 0
            while i < len(a):
                k = a[i]; v = a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else True
                out[k] = v; i += 2 if v is not True else 1
            return out
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "Qwen3.5-4B"))
            env = {k: v for k, v in os.environ.items() if k not in ("PL_GPUS", "PL_SEQS", "PL_PRESENCE_PENALTY")}
            with mock.patch.dict(os.environ, {**env, "PL_MODELS": d, "PL_RUNS": d, "QWEN35_4B_PATH": os.path.join(d, "Qwen3.5-4B")}, clear=True):
                a, b = commands(load(str(ROOT / "configs" / "train" / f"{new}.yaml")), 0, stamp="0000-0000"), \
                       commands(load(str(ROOT / "configs" / "train" / f"{base}.yaml")), 0, stamp="0000-0000")
                for role in ("actor", "learner"):
                    fa, fb = flags(a[role]), flags(b[role])
                    self.assertEqual(float(fa.pop("--stall_cap")), 3.5); self.assertEqual(fa.pop("--stall_play"), "1")
                    self.assertEqual({k: v for k, v in fa.items() if k not in names}, {k: v for k, v in fb.items() if k not in names})
                    self.assertEqual(fa["--tag"], new)
                self.assertEqual(a["env"], b["env"])

    def test_the_script(self):
        new = "phase5_nobelief_trow_stall_qwen35_4b"
        r = subprocess.run(["bash", str(ROOT / "scripts" / "ablation_scripts" / f"{new}.sh")], env={**os.environ, "DRY": "1", "PL_RUNS": "R"}, cwd="/",
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = [l for l in r.stdout.splitlines() if l.startswith("+ ")]
        self.assertEqual(lines[0], f"+ python -m plearn train --config {new}")


if __name__ == "__main__":
    unittest.main()
