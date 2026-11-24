"""FrozenLake's teacher values a move by the return it leads to (plearn/method/trainer.py --fl_view_value;
plearn.tasks.FL_VIEW_VALUE). Before the fix a move was worth the view it opened, scored as an answer, so with the goal known a walk to
any fog was worth as much as reaching the goal in as many moves. Here: the three histories of the study now rank the move towards the
goal first (and the switch gives their old values); with the goal known the top moves are exactly the first steps of the safe
shortest routes to it; the leaf is each hypothesis's own shortest walk; the teacher's greedy policy reaches the goal from start
states (300 of 300 training scenes measured); the switch reproduces the old plans; the other games' plans are those of the
code before the fix (byte-identical). The reference fingerprints are those of the code before the fix. A line's S_dec on a frozen history: tests/method/test_frozen_history.py. A test is skipped when its simulator is not installed (pip install -e ".[games]")."""
import hashlib
import importlib.util
import math
import random
import unittest
from collections import deque
from pathlib import Path

import plearn.tasks as TASKS
from plearn.method import estimator as E
from plearn.train import METHOD

ROOT = Path(__file__).resolve().parents[2]


def needs(*modules):
    return unittest.skipUnless(all(importlib.util.find_spec(m) for m in modules), f"needs {', '.join(modules)}")


class view_value:
    """plearn.tasks.FL_VIEW_VALUE set for a block (the old value), put back after it."""
    def __init__(self, on): self.on = on
    def __enter__(self): self.was = TASKS.FL_VIEW_VALUE; TASKS.FL_VIEW_VALUE = self.on
    def __exit__(self, *a): TASKS.FL_VIEW_VALUE = self.was


def planner(D, depth=2):
    from plearn.method.trainer import Planner
    return Planner(D, E.Reliability(4.0, 1.0, fidelity=True, value="log"), 0.1, depth)   # the run's settings (METHOD: log, fidelity)


def rnd(x, nd):
    if isinstance(x, float): return round(x, nd) if nd is not None else x
    if isinstance(x, (list, tuple)): return type(x)(rnd(y, nd) for y in x)
    return x


def fingerprint(D, episodes=3, steps=6, nd=None):
    """sha256 of the teacher's plans along a few training episodes with random looks (floats rounded to nd places when given)."""
    P = planner(D); rng = random.Random(7); h = hashlib.sha256()
    for i in range(episodes):
        e = D.start([D.sample(random.Random(1000 + i))])[0]
        for _ in range(steps):
            if getattr(e, "done", False): break
            pl = P(e)
            h.update(repr(rnd((pl["anchor"], pl["exact"], sorted(pl["tgt"].items()), sorted(pl["logp"].items()), sorted(pl["values"].items()),
                               pl["waste"], pl["waste_lp"], pl["wrong_lp"], pl["legal"]), nd)).encode())
            looks = D.look_strings(e)
            if not looks: break
            D.do_look(e, rng.choice(sorted(looks, key=str))); e.t += 1
            if hasattr(D, "after_turn"): D.after_turn(e)
        D.close([e])
    return h.hexdigest()[:16]


# the study's histories (training specs): the actions so far, the move that leads to the goal, and the old values (Q of each move)
EXAMPLES = [(2041220160, [], "UP", {"LEFT": -0.020203, "DOWN": -20.743469, "RIGHT": -0.020203, "UP": -0.020203}),
            (1536327547, [3, 2, 0, 2, 2, 3, 0, 3, 1, 0, 1, 1, 3, 2, 3, 3, 2, 0, 2, 0], "LEFT",
             {"LEFT": -0.484865, "DOWN": -0.464662, "RIGHT": -0.464662, "UP": -0.484865}),
            (2109885846, [0, 1, 3, 3, 1, 0, 0, 3, 0, 1, 0, 0, 1, 0, 3, 1, 2, 1, 1, 3], "RIGHT",
             {"LEFT": -0.820936, "DOWN": -0.820936, "RIGHT": -0.780531, "UP": -0.780531})]


@needs("gymnasium", "vagen")
class WalkValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import plearn.tasks.frozenlake.teacher as m
        cls.m = m; cls.D = m.Domain(split="train"); cls.P = planner(cls.D)

    def replay(self, seed, acts):
        e = self.D.start([("frozenlake-hidden-goal", seed, "train")])[0]
        for a in acts: self.D.do_look(e, a); e.t += 1; e.turns_used += 1
        return e

    def top(self, pl):
        v = {s[8:]: pl["values"][s] for s in pl["looks"].values()}; best = max(v.values())
        return {k for k, x in v.items() if x >= best - 1e-9}, v

    def test_the_three_examples(self):
        for seed, acts, goal_move, old in EXAMPLES:
            e = self.replay(seed, acts)
            with view_value(True): before = self.top(self.P(e))[1]
            for k, q in old.items(): self.assertAlmostEqual(before[k], q, places=5)   # the study's values, the old ties
            top, v = self.top(self.P(e))
            self.assertIn(goal_move, top, (seed, v))
            if seed == 1536327547: self.assertEqual(top, {"LEFT", "DOWN"})          # both start a safe shortest route to A3
            else: self.assertEqual(top, {goal_move}, (seed, v))
            self.D.close([e])

    def safe_dist(self, e, a, goal):
        B = self.D.belief(e); return B.ep.safe_reach(B.ep.move(e.pos, a)).get(goal) if not B.ep.known_hole(B.ep.move(e.pos, a)) else None

    def test_with_the_goal_known_the_top_moves_start_the_safe_shortest_routes(self):
        rng, n = random.Random(3), 0
        for i in range(60):
            e = self.D.start([self.D.sample(random.Random(50_000 + i))])[0]
            while not e.done and e.t < self.D.T:
                B = self.D.belief(e); goal = B.ep.known_goal(B)
                if goal is not None and goal in B.ep.safe_reach(e.pos):
                    d0 = B.ep.safe_reach(e.pos)[goal]; pl = self.P(e); top, v = self.top(pl)
                    first = {self.m.ACTIONS[a] for a in self.m.MOVE if self.safe_dist(e, a, goal) == d0 - 1}
                    if e.t + d0 <= self.D.T:
                        self.assertEqual(top, first, (e.spec, e.t, v)); n += 1
                        self.assertAlmostEqual(max(v.values()), -self.D.lam * (e.t + d0) / self.D.T, places=9)   # log c(t + d)
                looks = [a for a in self.m.MOVE if not B.ep.known_hole(B.ep.move(e.pos, a))]
                self.D.do_look(e, rng.choice(looks)); e.t += 1; self.D.after_turn(e)
            self.D.close([e])
        self.assertGreater(n, 100)

    def test_the_leaf_is_each_hypothesis_own_shortest_walk(self):
        """depth 1: Q(a) = E over the posterior of u(c(t + 1 + walk from the cell a leads to)), each walk found by a plain BFS."""
        m = self.m; rng, n = random.Random(4), 0
        for i in range(40):
            e = self.D.start([self.D.sample(random.Random(60_000 + i))])[0]
            for _ in range(rng.randrange(3, 9)):
                if e.done: break
                B = self.D.belief(e); self.D.do_look(e, rng.choice([a for a in m.MOVE if not B.ep.known_hole(B.ep.move(e.pos, a))])); e.t += 1
            B = self.D.belief(e); idx = [int(j) for j in B.M.nonzero()[0]]
            if e.done or B.ep.known_goal(B) is not None or len(idx) > 3000: self.D.close([e]); continue
            G, Hm, W = m.hypotheses(e.start); rel = E.Reliability(4.0, 1.0, fidelity=True, value="log")
            Q = B.lookahead(rel, self.D.lam, self.D.T, 1, list(m.MOVE))
            for a in m.MOVE:
                nxt = B.ep.move(e.pos, a); tot = val = 0.0
                for j in idx:
                    holes, g = int(Hm[j]), int(G[j]); dist = {nxt: 0} if not holes >> nxt & 1 else {}; q = deque(dist)
                    while q:
                        x = q.popleft()
                        for y in m.neighbours(x):
                            if y not in dist and not holes >> y & 1: dist[y] = dist[x] + 1; q.append(y)
                    end = e.t + 1 + dist[g] if g in dist else None
                    u = rel.util(math.exp(-self.D.lam * end / self.D.T), 1.0) if end is not None and end <= self.D.T else rel.util(math.exp(-self.D.lam), 0.0)
                    tot += W[j]; val += W[j] * u
                self.assertAlmostEqual(Q[a], val / tot, places=9); n += 1
            self.D.close([e])
        self.assertGreater(n, 20)

    def test_the_greedy_teacher_reaches_the_goal_from_the_start(self):
        """Measured on 300 training scenes: the greedy teacher wins all (3.9 actions on average; before the fix 6.7) and pi*
        98.5% (before: 85.3%, 13.2% out of time)."""
        won = []
        for i in range(40):
            e = self.D.start([self.D.sample(random.Random(500_000 + i))])[0]
            while not e.done and e.t < self.D.T:
                pl = self.P(e); looks = list(pl["looks"])
                a = max(looks, key=lambda k: (pl["values"][pl["looks"][k]], -looks.index(k)))
                self.D.do_look(e, a); e.t += 1; self.D.after_turn(e)
            won.append(e.won); self.D.close([e])
        self.assertEqual(sum(won), len(won))

    def test_the_switch_gives_the_old_plans(self):
        with view_value(True): self.assertEqual(fingerprint(self.D, nd=9), "2ddfe9bf1e088ca8")    # the code before the fix (floats to 1e-9:
        self.assertNotEqual(fingerprint(self.D, nd=9), "2ddfe9bf1e088ca8")                       # numpy versions sum differently)


class OtherGamesTests(unittest.TestCase):
    """The plans of the other games are those of the code before the fix, byte for byte (exact floats), with either setting (and
    the later switches, --sk_guess_value and --plan_past_limit, at their old values: tests/method/test_sk_return_value.py
    and test_action_limit.py)."""
    def same(self, mod, want, **kw):
        import importlib
        D = importlib.import_module(f"plearn.tasks.{mod}.teacher").Domain(split="train", **kw)
        was = TASKS.SK_GUESS_VALUE, TASKS.PLAN_PAST_LIMIT; TASKS.SK_GUESS_VALUE = TASKS.PLAN_PAST_LIMIT = True
        try:
            for on in (False, True):
                with view_value(on): self.assertEqual(fingerprint(D), want, (mod, on))
        finally: TASKS.SK_GUESS_VALUE, TASKS.PLAN_PAST_LIMIT = was

    def test_mastermind(self): self.same("mastermind", "ac2a49d94581c6e5")
    def test_tabletop(self): self.same("tabletop", "10f330761921b9e0", levels=["lamps", "cups", "blocks"], render=False)
    @needs("jax")
    def test_minesweeper(self): self.same("minesweeper", "507e7026d9e0d378", levels=["easy"])
    @needs("gym_sokoban")
    def test_sokoban(self): self.same("sokoban", "2dc0c865ad3a757d")
    @needs("minigrid")
    def test_minigrid(self): self.same("minigrid", "c92dfde6f3b300b3", levels=["S4R3"])


class SwitchTests(unittest.TestCase):
    def test_the_fix_is_the_default_and_the_runs_do_not_name_the_flag(self):
        self.assertFalse(TASKS.FL_VIEW_VALUE)
        self.assertNotIn("fl_view_value", METHOD)                   # the run files' command lines are unchanged (test_run_files)
        src = (ROOT / "plearn" / "method" / "trainer.py").read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--fl_view_value", type=int, default=0', src)
        self.assertIn("_tasks.FL_VIEW_VALUE = bool(args.fl_view_value)", src)
        self.assertIn("--fl_view_value (default 0, the fix)", src)
        self.assertIn('meta["args"].get("fl_view_value", 1) == args.fl_view_value', src)   # PL_SET_FROM: a set of the other kind is refused


if __name__ == "__main__":
    unittest.main()
