"""The score of a stated belief line by the act it implies (decision_score, --line_score decision) reads the game at the line's
history, not at the game's end. The RLVB (EM) / RLVB (GRPO) trainer plays its episodes to the end and
only then scores the own turns of each kept history (gen_update: run_episodes, then own_turns). On FrozenLake, Sokoban and MiniGrid
the teacher's belief keeps the game object, and its lookahead reads the agent's place, the seen cells and the open doors from it,
so before the fix the implied act was planned from the game's last position. Planner now stores Domain.freeze(B), a copy of the
game at the history. Each test: a plan at a history, the scores of some lines then; the game goes on to its end; the same plan
scores the same lines again (they must be equal), and a plan whose belief points at the live game (the old behaviour) differs on
some of them (the error is real in this data). A test is skipped when its simulator is not installed."""
import importlib.util
import random
import unittest


def needs(*modules):
    return unittest.skipUnless(all(importlib.util.find_spec(m) for m in modules), f"needs {', '.join(modules)}")


def planner(D):
    from plearn.method import estimator as E
    from plearn.method.trainer import Planner
    from plearn.train import METHOD
    return Planner(D, E.Reliability(fidelity=True, value=METHOD["value"]), METHOD["tau"], 2)


class FrozenHistory:
    """The check, for one domain: lines(plan) the lines to score, step(D, e, rng) one legal move (False: none), live(B, e) the
    belief B pointing at the live game e (the old behaviour)."""
    def run_check(self, D, lines, step, live, specs, at=(1, 2, 3), more=20):
        from plearn.method.trainer import decision_score
        P = planner(D); rng = random.Random(0); n = same = old_diff = 0
        for spec in specs:
            e = D.start([spec])[0]; plans = []
            for t in range(max(at) + more):
                if e.done: break
                if t in at:
                    p = P(e); plans.append((p, {s: decision_score(D, P, p, s) for s in lines(p)}))
                if not step(D, e, rng): break
                e.t += 1
            for p, ref in plans:                                       # the game is at its end (or further on) now
                for s, v in ref.items():
                    n += 1; same += decision_score(D, P, p, s) == v
                    old = dict(p, B=live(p["B"], e)); old_diff += decision_score(D, P, old, s) != v
            D.close([e])
        self.assertGreater(n, 20)
        self.assertEqual(same, n, f"{n - same} of {n} line scores changed after the history")
        self.assertGreater(old_diff, 0, "the live-game belief never differed here: the data does not show the error")


@needs("gymnasium", "pygame")
class FrozenLakeFrozenHistory(FrozenHistory, unittest.TestCase):
    def test_the_line_score_reads_the_history(self):
        from plearn.tasks.frozenlake import teacher as FL
        D = FL.Domain(split="validation")

        def step(D, e, rng):
            safe = [a for a in FL.MOVE if not e.known_hole(e.move(e.pos, a))]
            if not safe: return False
            D.do_look(e, rng.choice(safe)); D.after_turn(e); return True
        lines = lambda p: [p["anchor"]] + [f"Possible: goal at {c}" for c in sorted(p["B"].post())[:4]]
        self.run_check(D, lines, step, lambda B, e: FL.Belief(e, B.M, B.t), D.eval_specs(8))


@needs("gym_sokoban")
class SokobanFrozenHistory(FrozenHistory, unittest.TestCase):
    def test_the_line_score_reads_the_history(self):
        from plearn.tasks.sokoban import teacher as SK
        D = SK.Domain(split="validation")

        def step(D, e, rng):
            looks = list(D.look_strings(e))
            if not looks: return False
            D.do_look(e, rng.choice(looks)); D.after_turn(e); return True
        lines = lambda p: [p["anchor"]] + [f"Possible: target at {c}" for c in sorted(p["B"].post())[:4]]
        self.run_check(D, lines, step, lambda B, e: SK.Belief(e, B.alive, B.t), D.eval_specs(8))


@needs("minigrid")
class MiniGridFrozenHistory(FrozenHistory, unittest.TestCase):
    def test_the_line_score_reads_the_history(self):
        from plearn.tasks.minigrid import teacher as MG
        D = MG.Domain(levels=["S4R3"], split="validation")

        def step(D, e, rng):
            looks = list(D.look_strings(e))
            if not looks: return False
            D.do_look(e, rng.choice(looks)); D.after_turn(e); return True
        lines = lambda p: [p["anchor"]] + [f"Possible: {r}" for r in sorted(p["B"].post())]
        self.run_check(D, lines, step, lambda B, e: MG.Belief(e, B.M, B.t), D.eval_specs(8), at=(2, 6, 10), more=30)


if __name__ == "__main__":
    unittest.main()
