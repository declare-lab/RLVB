"""Sokoban's teacher values a move by the return it leads to (plearn/method/trainer.py --sk_guess_value; plearn.tasks.SK_GUESS_VALUE).
Before the fix heading for a target was worth the accuracy of a guess among the k candidates (log(1 / k) in the log value), and the
rest after an exploring walk again a guess, though the game has no guess: the box on another cell only shows that cell is not the
target. Here: the study's two histories now rank the moves in the order of the full-horizon reference (the teacher's own greedy play,
enumerated over the posterior; the switch gives their old values); with the target known the top moves are exactly the first steps
of the shortest solutions; a value is the exact expectimax over the posterior (checked against a plain recursion over the target);
the greedy teacher wins from start states (measured: 299 of 300 training rooms, as before); the switch reproduces the old
plans; the other games' plans do not depend on the switch. A test is skipped when its simulator is not installed.
Reference fingerprints: the code before the fix.
"""
import math
import random
import unittest
from functools import lru_cache

import plearn.tasks as TASKS
from plearn.method import estimator as E
from plearn.train import METHOD
from tests.method.test_fl_walk_value import ROOT, fingerprint, needs, planner


class guess_value:
    """plearn.tasks.SK_GUESS_VALUE set for a block (the old value), put back after it."""
    def __init__(self, on): self.on = on
    def __enter__(self): self.was = TASKS.SK_GUESS_VALUE; TASKS.SK_GUESS_VALUE = self.on
    def __exit__(self, *a): TASKS.SK_GUESS_VALUE = self.was


SPEC = ("sokoban-hidden-target", 823435504, "train")       # the study's room: box D3, the target among B3, C3, B4, B5 (it is B4)
# the actions so far, the old values (Q of each move), the reference values of the fixed teacher (its own greedy play) and their order
EXAMPLES = [([4, 4], {"UP": -0.614259, "DOWN": -1.041138, "LEFT": -0.899719, "RIGHT": -1.041138},
             {"UP": 0.860, "DOWN": 0.878, "LEFT": 0.896, "RIGHT": 0.878}, [{"LEFT"}, {"DOWN", "RIGHT"}, {"UP"}]),
            ([4, 4, 1, 3, 4, 1], {"UP": -1.101746, "DOWN": -1.121949, "LEFT": -1.081544, "RIGHT": -1.101746},
             {"UP": 0.793, "DOWN": 0.810, "LEFT": 0.778, "RIGHT": 0.793}, [{"DOWN"}, {"UP", "RIGHT"}, {"LEFT"}])]


@needs("gym_sokoban", "vagen")
class ReturnValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import plearn.tasks.sokoban.teacher as m
        cls.m = m; cls.D = m.Domain(split="train"); cls.P = planner(cls.D)

    def replay(self, spec, acts):
        e = self.D.start([spec])[0]
        for a in acts: self.D.do_look(e, a); e.t += 1; self.D.after_turn(e)
        return e

    def values(self, e):
        pl = self.P(e); return {s[8:]: pl["values"][s] for s in pl["looks"].values()}, pl

    def test_the_two_examples(self):
        for acts, old, ref, order in EXAMPLES:
            e = self.replay(SPEC, acts)
            with guess_value(True): before = self.values(e)[0]
            for k, q in old.items(): self.assertAlmostEqual(before[k], q, places=5)          # the study's values
            v, pl = self.values(e)
            levels = sorted({round(x, 9) for x in v.values()}, reverse=True)
            self.assertEqual([{k for k in v if round(v[k], 9) == q} for q in levels], order, (acts, v))   # the reference's order
            lp = {s[8:]: pl["logp"][s] for s in pl["looks"].values()}
            self.assertEqual(max(lp, key=lp.get), next(iter(order[0])))                     # and the top of log pi*
            self.D.close([e])

    def optimal_first_steps(self, B, e):
        (c,) = tuple(B.alive); d = B.ep.dist(c); L = d.get(e.state)
        if L is None: return None, set()                             # a dead box
        return L, {self.m.ACTIONS[a] for a in self.m.ACTIONS if d.get(B.ep.graph.step(e.state, a)) == L - 1}

    def test_with_the_target_known_the_top_moves_start_the_shortest_solutions(self):
        rng, n = random.Random(5), 0
        for i in range(60):
            e = self.D.start([self.D.sample(random.Random(70_000 + i))])[0]
            while not e.done:
                B = self.D.belief(e)
                if len(B.alive) == 1:
                    L, first = self.optimal_first_steps(B, e); v, _ = self.values(e); best = max(v.values())
                    if L is not None and e.t + L <= self.D.T:
                        self.assertEqual({k for k, x in v.items() if x >= best - 1e-12}, first, (e.spec, e.t, v)); n += 1
                        self.assertAlmostEqual(best, -self.D.lam * (e.t + L) / self.D.T, places=9)   # log c(t + L) (phi = 1)
                self.D.do_look(e, rng.choice(sorted(self.D.look_strings(e)))); e.t += 1; self.D.after_turn(e)
            self.D.close([e])
        self.assertGreater(n, 100)

    def test_the_value_is_the_expectimax_over_the_target(self):
        """Q(a) against a plain recursion: the belief as the set of targets the views allow (each equally likely), a target's own
        outcome after each move (won, or the cells in view hold it or not), u(c(n), 1) at a win, u(c(T), 0) at the limit."""
        m, D = self.m, self.D; rel = E.Reliability(4.0, 1.0, fidelity=True, value="log"); T = D.T
        u = lambda n: rel.util(math.exp(-D.lam * n / T), 1.0); fail = rel.util(math.exp(-D.lam), 0.0)
        rng, n = random.Random(6), 0
        for i in range(30):
            e = self.D.start([self.D.sample(random.Random(80_000 + i))])[0]
            for _ in range(rng.randrange(0, 5)):
                if e.done: break
                self.D.do_look(e, rng.choice(sorted(self.D.look_strings(e)))); e.t += 1; self.D.after_turn(e)
            B = self.D.belief(e)
            if e.done or len(B.alive) > 6: self.D.close([e]); continue
            g = e.graph; view = e.view

            @lru_cache(maxsize=None)
            def V(s, A, t):                          # the best expected return, the target uniform on A
                if t >= T: return fail
                return max(Qr(s, A, t, a) for a in m.ACTIONS)

            @lru_cache(maxsize=None)
            def Qr(s, A, t, a):
                s1 = g.step(s, a); w = m.window(s1[0], view); tot = 0.0
                for c in A:                          # the target is c: what the agent sees after the move
                    if s1[1] == c: tot += u(t + 1); continue
                    A2 = frozenset([c]) if (len(A) == 1 or c in w) else frozenset(x for x in A if x not in w and x != s1[1])
                    tot += V(s1, A2, t + 1)
                return tot / len(A)
            Q = B.lookahead(rel, D.lam, T, 2, list(m.ACTIONS))
            for a in m.ACTIONS: self.assertAlmostEqual(Q[a], Qr(e.state, frozenset(B.alive), e.t, a), places=9); n += 1
            self.D.close([e])
        self.assertGreater(n, 40)

    def test_the_greedy_teacher_wins_from_the_start(self):
        """Measured on 300 training rooms: the greedy teacher wins 99.7% (as before), in 6.1 actions (6.5), and pi* 99.7%
        (96.7%)."""
        won = []
        for i in range(40):
            e = self.D.start([self.D.sample(random.Random(700_000 + i))])[0]
            while not e.done:
                pl = self.P(e); looks = list(pl["looks"])
                a = max(looks, key=lambda k: (pl["values"][pl["looks"][k]], -looks.index(k)))
                self.D.do_look(e, a); e.t += 1; self.D.after_turn(e)
            won.append(e.won); self.D.close([e])
        self.assertGreaterEqual(sum(won), len(won) - 1)

    def test_the_switch_gives_the_old_plans(self):
        with guess_value(True): self.assertEqual(fingerprint(self.D), "2dc0c865ad3a757d")      # the code before the fix (exact)
        self.assertNotEqual(fingerprint(self.D), "2dc0c865ad3a757d")


class OtherGamesTests(unittest.TestCase):
    """The plans of the other games do not depend on the switch (byte for byte; FrozenLake's floats to 1e-9, as test_fl_walk_value)."""
    def same(self, mod, want, nd=None, **kw):
        import importlib
        D = importlib.import_module(f"plearn.tasks.{mod}.teacher").Domain(split="train", **kw)
        for on in (False, True):
            with guess_value(on): self.assertEqual(fingerprint(D, nd=nd), want, (mod, on))

    def test_mastermind(self): self.same("mastermind", "ac2a49d94581c6e5")
    def test_tabletop(self): self.same("tabletop", "10f330761921b9e0", levels=["lamps", "cups", "blocks"], render=False)
    @needs("jax")
    def test_minesweeper(self): self.same("minesweeper", "507e7026d9e0d378", levels=["easy"])
    @needs("minigrid")
    def test_minigrid(self): self.same("minigrid", "c92dfde6f3b300b3", levels=["S4R3"])
    @needs("gymnasium", "vagen")
    def test_frozenlake(self): self.same("frozenlake", "b5ae3b1071f1856e", nd=9)


class SwitchTests(unittest.TestCase):
    def test_the_fix_is_the_default_and_the_runs_do_not_name_the_flag(self):
        self.assertFalse(TASKS.SK_GUESS_VALUE)
        self.assertNotIn("sk_guess_value", METHOD)                  # the run files' command lines are unchanged (test_run_files)
        src = (ROOT / "plearn" / "method" / "trainer.py").read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--sk_guess_value", type=int, default=0', src)
        self.assertIn("_tasks.SK_GUESS_VALUE = bool(args.sk_guess_value)", src)
        self.assertIn("--sk_guess_value (default 0, the fix)", src)
        self.assertIn('meta["args"].get("sk_guess_value", 1) == args.sk_guess_value', src)   # PL_SET_FROM: a set of the other kind is refused


if __name__ == "__main__":
    unittest.main()
