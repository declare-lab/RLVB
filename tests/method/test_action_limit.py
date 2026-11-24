"""The lookaheads respect the games' action limits (plearn/method/trainer.py --plan_past_limit; plearn.tasks.PLAN_PAST_LIMIT).
Before the fix MiniGrid valued poses that a plan reaches after look T, and MineSweeper valued a move by a FIRE after the game's last
action (and a second reveal past it), so near the end the teacher kept moving where the plan could not be finished. The end
conditions, measured on the engines and checked here: MiniGrid offers T = 64 looks, then one turn to answer (the
episode's turn limit; the engine's own max_steps 480 never binds); popgym's MineSweeper ends the game on a MOVE that is action 2 n^2
(32 on easy: no answer is possible after it), but not on a FIRE that is action 2 n^2 (popgym tests the FIRE's limit on the time
before it), after which the answer comes on the next turn. Here: the end conditions; at the histories near the limit no plan is
valued that ends after it (each value against a plain search over the plans that fit); the study's MineSweeper history, whose old
greedy play ran out of actions without an answer; TableTop and RoboQuest (the same omission, after invalid turns or with a world
limit below the domain's T); the switch gives the old plans; the other games do not depend on it. A test is skipped when its
simulator is not installed.
Reference fingerprints: the code before the fix.
"""
import hashlib
import math
import random
import unittest

import plearn.tasks as TASKS
from plearn.method import estimator as E
from plearn.train import METHOD
from tests.method.test_fl_walk_value import ROOT, fingerprint, needs, planner, rnd


class past_limit:
    """plearn.tasks.PLAN_PAST_LIMIT set for a block (the old value), put back after it."""
    def __init__(self, on): self.on = on
    def __enter__(self): self.was = TASKS.PLAN_PAST_LIMIT; TASKS.PLAN_PAST_LIMIT = self.on
    def __exit__(self, *a): TASKS.PLAN_PAST_LIMIT = self.was


REL = lambda: E.Reliability(4.0, 1.0, fidelity=True, value="log")


def late_fingerprint(D, limit, episodes=3, last=6, nd=9, seed0=3000, no_fire=None):
    """sha256 of the plans of the last `last` looks before `limit`, along a few training episodes of random looks (no_fire: a look
    left out after the first 3 looks, so that a mine does not end the game); floats to nd places."""
    P = planner(D); h = hashlib.sha256()
    for i in range(episodes):
        rng = random.Random(seed0 + i); e = D.start([D.sample(random.Random(seed0 + i))])[0]
        while not getattr(e, "done", False):
            looks = D.look_strings(e)
            if not looks: break
            if e.t >= limit - last:
                pl = P(e)
                h.update(repr(rnd((pl["anchor"], sorted(pl["tgt"].items()), sorted(pl["logp"].items()), sorted(pl["values"].items()), pl["legal"]), nd)).encode())
            D.do_look(e, rng.choice(sorted(k for k in looks if not (no_fire is not None and k == no_fire and e.t >= 3)))); e.t += 1
            if hasattr(D, "after_turn"): D.after_turn(e)
        D.close([e])
    return h.hexdigest()[:16]


@needs("minigrid")
class MiniGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import plearn.tasks.minigrid.teacher as m
        cls.m = m; cls.D = m.Domain(levels=["S4R3"], split="train"); cls.P = planner(cls.D)

    def engine_last_look(self):
        """Step the real game: the number of looks it offers, and whether an answer is possible after the last of them."""
        D = self.D; e = D.start([("keycorridor-S4R3", 1234567, "train")])[0]
        while D.look_strings(e): D.do_look(e, self.m.LEFT); e.t += 1; D.after_turn(e)
        answer = not e.done and bool(D.answer_options(e)) and e.turns_used < D.turns
        D.after_turn(e); ends = e.done; D.close([e])
        return e.t, answer, ends

    def test_the_engine_end_condition(self):
        last, answer, ends = self.engine_last_look()
        self.assertEqual(last, self.D.T)                    # 64 looks
        self.assertTrue(answer)                             # then one turn to answer
        self.assertTrue(ends)                               # and no more
        env = self.m.make_env("S4R3", 1); self.assertGreater(env.max_steps, self.D.T); env.close()   # the engine's own limit never binds

    def test_no_pose_after_the_last_look_is_valued(self):
        last = self.engine_last_look()[0]; rel = REL(); D = self.D; n = 0
        for i in range(4):
            rng = random.Random(40 + i); e = D.start([D.sample(random.Random(9000 + i))])[0]
            while D.look_strings(e):
                if e.t >= last - 8:
                    B = D.belief(e); Q = B.lookahead(rel, D.lam, D.T, 2, list(D.look_strings(e)))
                    for a, q in Q.items():                  # a plain search over the poses the plan reaches by the last look
                        nxt = B.ep.step_plan((B.ep.pos, B.ep.dir, frozenset(B.ep.opened)), a); best = rel.none
                        for st, d in B.ep.reach(nxt).items():
                            v = B.ep.pose_value(B, rel, st)
                            if v is not None and e.t + 1 + d <= last: best = max(best, rel.with_cost(math.exp(-D.lam * (e.t + 1 + d) / D.T), v))
                        self.assertEqual(q, best); n += 1
                D.do_look(e, rng.choice(sorted(D.look_strings(e)))); e.t += 1; D.after_turn(e)
            D.close([e])
        self.assertGreater(n, 60)

    def test_the_switch_gives_the_old_plans(self):
        with past_limit(True): self.assertEqual(late_fingerprint(self.D, self.D.T), "3a968c7d292cc7d4")   # the code before the fix (floats
        self.assertNotEqual(late_fingerprint(self.D, self.D.T), "3a968c7d292cc7d4")                     # to 1e-9)


@needs("jax", "popgym_arcade")
class MineSweeperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import plearn.tasks.minesweeper.teacher as m
        cls.m = m; cls.D = m.Domain(levels=["easy"], split="train"); cls.P = planner(cls.D); cls.L = 2 * 4 * 4

    def to_last_action(self, seed):
        """An easy board with the cursor on a safe cell after L - 1 moves that change nothing on it (None: a mine under it)."""
        m, D = self.m, self.D; e = D.start([("minesweeper-easy", seed, "train")])[0]
        for _ in range(self.L - 1):
            x = int(e.state.action_x); D.do_look(e, m.RIGHT if x == 0 else m.LEFT); e.t += 1; D.after_turn(e)
            assert not e.done
        return None if e.truth >> m.bit(4, int(e.state.action_x), int(e.state.action_y)) & 1 else e

    def test_the_engine_end_condition(self):
        m, D = self.m, self.D
        self.assertEqual(m.game("easy")[0].max_episode_length, self.L)                 # popgym's limit: 2 n^2 actions
        seeds = [s for s in range(5_000_000, 5_000_040) if self.to_last_action(s) is not None][:3]
        for s in seeds:
            e = self.to_last_action(s); D.do_look(e, m.UP if int(e.state.action_y) else m.DOWN); e.t += 1; D.after_turn(e)
            self.assertTrue(e.done); self.assertFalse(e.correct)                         # a move as action L: the game is over, no answer
            e = self.to_last_action(s); D.do_look(e, m.FIRE); e.t += 1; D.after_turn(e)
            self.assertFalse(e.done); self.assertEqual(D.look_strings(e), {})           # a FIRE as action L: no look left, and the
            self.assertLess(e.turns_used, D.turns)                                       # answer comes on the next turn
            self.assertTrue(D.answer_options(e))

    def late_histories(self, n_eps=6, first=24):
        m, D = self.m, self.D
        for i in range(n_eps):
            rng = random.Random(70 + i); e = D.start([D.sample(random.Random(9100 + i))])[0]
            while D.look_strings(e) and not e.done:
                if e.t >= first: yield e
                looks = sorted(k for k in D.look_strings(e) if k != m.FIRE or e.t < 3)
                D.do_look(e, rng.choice(looks)); e.t += 1; D.after_turn(e)

    def test_no_plan_ends_after_the_limit(self):
        """Each move's value is the best FIRE that a shortest cursor path starting with it reaches by action L (a plain search), with
        a reveal value whose second reveal fits in the actions left; a move as action L is worth the floor."""
        m, D = self.m, self.D; rel = REL(); n = 0
        c = lambda t: math.exp(-D.lam * t / D.T)
        for e in self.late_histories():
            B = D.belief(e); Q = B.lookahead(rel, D.lam, D.T, 2, list(D.look_strings(e)))
            if B.dead or B.k() > m.DEPTH2_MAX: continue
            # reveal values by hand: the first reveal of cell i comes at t_i (the shortest cursor path), the second by L
            _, nb = m.tables(4, 2); Ev = {}
            for i in range(16):
                if B.revealed >> i & 1: continue
                safe = B.A[((B.A >> m.np.uint64(i)) & m.np.uint64(1)) == 0]; ti = B.t + 1 + abs(i % 4 - B.x) + abs(i // 4 - B.y)
                v = (B.k() - len(safe)) / B.k() * rel.f(rel.mean(m._GB(B.N)))
                cnt = m.np.bitwise_count(safe & nb[i]) if len(safe) else []
                for cv in sorted(set(int(x) for x in cnt)):
                    g = safe[cnt == cv]; best = rel.f(rel.mean(m._GB(len(g))))
                    if len(g) > 1:
                        for j, vj in m.values_one(g, B.revealed | (1 << i), 4, 2, rel, B.N).items():
                            d = abs(j % 4 - i % 4) + abs(j // 4 - i // 4)
                            if ti + d + 1 <= self.L: best = max(best, rel.with_cost(math.exp(-D.lam * (d + 1) / D.T), vj))
                    v += len(g) / B.k() * best
                Ev[i] = v
            for a, q in Q.items():
                if a == m.FIRE:
                    here = m.bit(4, B.x, B.y); want = rel.with_cost(c(B.t + 1), Ev[here]) if here in Ev else rel.none
                elif B.t + 1 >= self.L: want = rel.util(c(B.t + 1), 0.0)
                else:
                    nx, ny = B.x + m.STEP[a][0], B.y + m.STEP[a][1]; want = rel.none
                    for i, v in Ev.items():
                        d = abs(i % 4 - nx) + abs(i // 4 - ny)
                        if abs(i % 4 - B.x) + abs(i // 4 - B.y) == d + 1 and B.t + 1 + d + 1 <= self.L: want = max(want, rel.with_cost(c(B.t + d + 2), v))
                self.assertAlmostEqual(q, want, places=9, msg=(e.spec, e.t, a)); n += 1
        self.assertGreater(n, 50)

    def test_the_study_example(self):
        """spec 970564884 at action 29: the old top move (RIGHT) was worth a FIRE then a move, and the greedy play ran out of
        actions with no answer (the reference: 0); now its plan is the FIRE at action 31 and the answer (reference 0.102)."""
        m, D = self.m, self.D
        acts = [4, 1, 2, 1, 0, 4, 3, 1, 0, 4, 0, 1, 2, 1, 1, 3, 2, 3, 0, 3, 1, 0, 2, 4, 2, 1, 2, 4, 0]   # the study's history
        def play():
            e = D.start([("minesweeper-easy", 970564884, "train")])[0]
            for a in acts: D.do_look(e, a); e.t += 1; D.after_turn(e)
            seq = []
            while not e.done:
                pl = self.P(e); pC = pl["tgt"].get(pl["commits"][pl["best"]], 0.0)
                if pC >= 0.5 or not pl["looks"]: seq.append("SUBMIT"); break
                a = max(pl["looks"], key=lambda k: (pl["values"][pl["looks"][k]], -list(pl["looks"]).index(k)))
                seq.append(m.ACTIONS[a]); D.do_look(e, a); e.t += 1; D.after_turn(e)
            return seq, e
        with past_limit(True): seq, e = play()
        self.assertEqual(seq, ["MOVE RIGHT", "FIRE", "MOVE UP"]); self.assertTrue(e.done and not e.correct)   # out of actions
        seq, e = play()
        self.assertEqual(seq, ["MOVE RIGHT", "FIRE", "SUBMIT"])

    def test_the_switch_gives_the_old_plans(self):
        with past_limit(True): self.assertEqual(late_fingerprint(self.D, self.L, no_fire=self.m.FIRE), "9cc108f4b4d69592")
        self.assertNotEqual(late_fingerprint(self.D, self.L, no_fire=self.m.FIRE), "9cc108f4b4d69592")


class SameOmissionTests(unittest.TestCase):
    def test_tabletop_after_invalid_turns(self):
        """An invalid turn counts in t (the trainer), so a look left could come after the budget of n looks: not valued now."""
        import plearn.tasks.tabletop.teacher as m
        D = m.Domain(split="train", levels=["lamps", "cups", "blocks"], render=False); rel = REL(); n = 0
        for i in range(40):
            e = D.start([D.sample(random.Random(9200 + i))])[0]
            e.t += 2; D.after_turn(e); D.after_turn(e)                                      # two invalid turns
            while D.look_strings(e):
                B = D.belief(e); Q = B.lookahead(rel, D.lam, D.T, 2, list(D.look_strings(e))) if hasattr(B, "lookahead") else \
                    E.lookahead_Q(B, rel, D.lam, D.T, 2, acts=list(D.look_strings(e)))
                for a in D.look_strings(e):                                             # a look at the last time: its value is the
                    if e.t + 1 >= e.budget:                                             # answer after it (no further look)
                        want = sum(p * rel.util(math.exp(-D.lam * B2.t / D.T), rel.mean(B2)) for p, B2 in B.outcomes(a))
                        self.assertAlmostEqual(Q[a], want, places=12); n += 1
                D.do_look(e, sorted(D.look_strings(e))[0]); e.t += 1; D.after_turn(e)
            D.close([e])
        self.assertGreater(n, 20)

    def test_roboquest_world_limit(self):
        """RoboQuest's lookahead had the domain's T (5) for worlds of 4 looks: at the last look it valued a second one."""
        try: import plearn.tasks.roboquest.roboquest as R; D = R.Domain()
        except Exception as ex: self.skipTest(f"RoboQuest's registry: {ex}")
        rel = REL(); n = 0
        for i in range(60):
            e = D.start([D.sample(random.Random(i))])[0]; rng = random.Random(i)
            while D.look_strings(e):
                if e.t == e.W.T - 1:
                    B = D.belief(e); Q = E.lookahead_Q(B, rel, D.lam, D.T, 2, acts=list(D.look_strings(e)))
                    for a in D.look_strings(e):
                        want = sum(p * rel.util(math.exp(-D.lam * B2.t / D.T), rel.mean(B2)) for p, B2 in B.outcomes(a))
                        self.assertAlmostEqual(Q[a], want, places=12); n += 1
                D.do_look(e, rng.choice(list(D.look_strings(e)))); e.t += 1
        self.assertGreater(n, 50)


class OtherGamesTests(unittest.TestCase):
    """The plans of the other games do not depend on the switch (byte for byte; FrozenLake's floats to 1e-9)."""
    def same(self, mod, want, nd=None, **kw):
        import importlib
        D = importlib.import_module(f"plearn.tasks.{mod}.teacher").Domain(split="train", **kw)
        for on in (False, True):
            with past_limit(on): self.assertEqual(fingerprint(D, nd=nd), want, (mod, on))

    def test_mastermind(self): self.same("mastermind", "ac2a49d94581c6e5")
    def test_tabletop(self): self.same("tabletop", "10f330761921b9e0", levels=["lamps", "cups", "blocks"], render=False)   # valid looks
    @needs("gym_sokoban", "vagen")
    def test_sokoban(self): self.same("sokoban", "46b5a417a7aac2e7")                    # the fixed Sokoban (test_sk_return_value)
    @needs("gymnasium", "vagen")
    def test_frozenlake(self): self.same("frozenlake", "b5ae3b1071f1856e", nd=9)


class SwitchTests(unittest.TestCase):
    def test_the_fix_is_the_default_and_the_runs_do_not_name_the_flag(self):
        self.assertFalse(TASKS.PLAN_PAST_LIMIT)
        self.assertNotIn("plan_past_limit", METHOD)                 # the run files' command lines are unchanged (test_run_files)
        src = (ROOT / "plearn" / "method" / "trainer.py").read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--plan_past_limit", type=int, default=0', src)
        self.assertIn("_tasks.PLAN_PAST_LIMIT = bool(args.plan_past_limit)", src)
        self.assertIn("--plan_past_limit (default 0, the fix)", src)
        self.assertIn('meta["args"].get("plan_past_limit", 1) == args.plan_past_limit', src)   # PL_SET_FROM: a set of the other kind is refused


if __name__ == "__main__":
    unittest.main()
