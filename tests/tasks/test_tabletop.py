"""The tabletop environment (plearn/tasks/tabletop): the exact belief against brute force, the teacher solves every validation
episode, the splits and their colours are apart, the belief language reads the exact line back, the actions parse, and (with
ManiSkill3 installed) the pictures: two of each observation, and lamp 1 on the left in both."""
import importlib.util
import itertools
import random
import unittest

from plearn.tasks.tabletop import teacher as T

needs_maniskill = unittest.skipUnless(importlib.util.find_spec("mani_skill"), "needs mani_skill")


def planner(D):
    from plearn.method import estimator as E
    from plearn.method.trainer import Planner
    from plearn.train import METHOD
    return Planner(D, E.Reliability(fidelity=True, value=METHOD["value"]), METHOD["tau"], 2)


class TabletopTests(unittest.TestCase):
    def test_the_belief_is_exact_and_the_teacher_solves_every_episode(self):
        for fam in T.FAMILIES:
            with self.subTest(family=fam):
                D = T.Domain(levels=[fam], split="validation", render=False); P = planner(D); won = 0
                for e in D.start(D.eval_specs(40)):
                    hyps = list(itertools.permutations(range(e.n))) if fam == "lamps" else list(range(e.n)); seen = []
                    for _ in range(e.budget + 1):
                        self.assertEqual({h for h in hyps if all(T.shown(fam, h, a) == s for a, s in seen)}, e.alive)
                        self.assertEqual(T.read_line(D.state_text(e), fam, e.n), set(D.answer_posterior(e)))   # the exact line
                        tgt = P(e)["tgt"]; best = max(tgt, key=tgt.get)
                        if best.startswith("ACTION: SUBMIT"):
                            self.assertTrue(D.belief(e).identified()); won += D.do_commit(e, best[len("ACTION: SUBMIT "):]); break
                        kind, a = D.parse_action(e, best); self.assertEqual(kind, "look")
                        D.do_look(e, a); e.t += 1; seen.append((a, T.shown(fam, e.truth, a)))
                self.assertEqual(won, 40)

    def test_splits_and_colours_are_apart(self):
        rng = random.Random(0)
        for fam in T.FAMILIES:
            D = {s: T.Domain(levels=[fam], split=s, render=False) for s in ("train", "validation", "test")}
            train = {D["train"].sample(rng)[1] for _ in range(3000)}
            val, test = ({s[1] for s in D[k].eval_specs(500)} for k in ("validation", "test"))
            self.assertFalse(train & val or train & test or val & test)
        self.assertFalse(set(T.COLOURS["train"]) & set(T.COLOURS["test"]))
        cups = T.Domain(levels=["cups"], split="test", render=False)
        for e in cups.start(cups.eval_specs(30)):
            self.assertIn(e.target[1], T.COLOURS["test"])
            self.assertTrue(all(o is None or o[1] in T.COLOURS["test"] for o in e.under.values()))

    def test_belief_lines_and_actions(self):
        self.assertEqual(T.read_line("Possible: cups 2, 4", "cups", 5), {"2", "4"})
        self.assertEqual(T.read_line("Possible: all", "blocks", 3), {"1", "2", "3"})
        self.assertEqual(T.read_line("Possible: A 2; B 1 or 3; C 1 or 3", "lamps", 3), {"A2 B1 C3", "A2 B3 C1"})
        self.assertEqual(T.read_line("Possible: A→2; B: 3", "lamps", 3), {"A2 B3 C1"})
        self.assertEqual(len(T.read_line("Possible: E 5", "lamps", 5)), 24); self.assertIsNone(T.read_line("Possible: E 5", "lamps", 4))
        impossible = "Possible: A 2 or 4; B 1, 3, 5; C 1, 3, 5; D 1, 3, 5; E 1, 3, 5"          # four buttons on three lamps
        self.assertEqual(T.read_line(impossible, "lamps", 5), set())
        B = T.Belief("lamps", 5, set(itertools.permutations(range(5))), {0}, 1)
        self.assertEqual(T.Domain(levels=["lamps"], render=False).reading(B, impossible)["kind"], "reasoning")
        self.assertEqual(T.Domain(levels=["lamps"], render=False).belief_score(B, impossible), -T.MISS)
        self.assertIsNone(T.read_line("The cup is on the left.", "cups", 3)); self.assertIsNone(T.read_line("Possible: cups 7", "cups", 3))
        D = T.Domain(levels=["lamps"], split="validation", render=False); e = D.start(D.eval_specs(1))[0]
        self.assertEqual(D.parse_action(e, "ACTION: PRESS B"), ("look", 1))
        self.assertEqual(D.parse_action(e, "action: submit " + T.perm_str(tuple(range(e.n))).lower()), ("answer", T.perm_str(tuple(range(e.n)))))
        D.do_look(e, 1); self.assertEqual(D.parse_action(e, "ACTION: PRESS B"), ("invalid", None))   # each helper once

    def test_the_reading_of_each_belief_line(self):
        """Domain.reading against the pictures so far (exact, wider, misread: the exact set of another picture, reasoning: no
        picture's set), the action against the teacher's decision, and their shares in the evaluator's episode row (validation
        10012: five cups, the yellow cylinder under cup 3, a yellow ball under cup 1, nothing under cup 2)."""
        from plearn.tasks.domain import DomainEpisode
        ep = DomainEpisode(["validation", ["tabletop-cups", 10012, "validation"]], "plearn.tasks.tabletop.teacher",
                           {"levels": ["cups"], "render": False})
        for line, action in (("Possible: all", "ACTION: LIFT 1"),             # before any action: exact, not a reading
                             ("Possible: cup 1", "ACTION: LIFT 2"),           # the ball read as the cylinder: misread
                             ("Possible: cups 3, 4", "ACTION: LIFT 4"),       # cup 5 dropped with no picture of it: reasoning
                             ("Possible: cups 3, 5", "ACTION: SUBMIT 3")):    # exact, but an answer before the evidence decides
            ep.step(line, action, {})
        row = ep.row()
        self.assertEqual([t["reading"]["kind"] for t in row["turn_log"]], ["exact", "misread", "reasoning", "exact"])
        self.assertEqual([row[f"read_{k}"] for k in ("exact", "wider", "misread", "reasoning", "unreadable")], [1 / 3, 0.0, 1 / 3, 1 / 3, 0.0])
        self.assertEqual((row["read_miss"], row["read_width"], row["read_width_exact"]), (1 / 3, 5 / 3, 3.0))   # cup 1 leaves out cup 3
        self.assertEqual(row["act_error"], 0.5)                          # of the two turns with an exact line, the early SUBMIT
        self.assertTrue(row["correct"]); self.assertEqual(row["premature"], 1.0)
        D = T.Domain(levels=["lamps"], render=False)                     # lamps: a press read as another lamp is a misread
        B = T.Belief("lamps", 4, {p for p in itertools.permutations(range(4)) if p[0] == 2}, {0}, 1)
        self.assertEqual(D.reading(B, "Possible: A 2; B 1 or 3 or 4; C 1 or 3 or 4; D 1 or 3 or 4", "A3 B1 C2 D4")["kind"], "misread")
        self.assertEqual(D.reading(B, "Possible: all", "A3 B1 C2 D4")["kind"], "wider")

    def test_two_pictures_each_observation(self):
        from plearn.tasks.tabletop.scene import SIZE
        for fam in T.FAMILIES:
            D = T.Domain(levels=[fam], split="validation"); e = D.start(D.eval_specs(1))[0]; D.do_look(e, 0)
            parts, images = D.user_content(e)
            self.assertEqual(len(images), 4); self.assertTrue(all(im.size == (SIZE, SIZE) for im in images))
            self.assertEqual(sum(p.get("type") == "image" for p in parts), 4)

    @needs_maniskill
    def test_lamp_one_is_on_the_left_in_both_pictures(self):
        """The rules number the row from the left as the cameras see them: the light of lamp 1 is in the left half of both."""
        import numpy as np
        from plearn.tasks.tabletop.scene import SIZE, render
        D = T.Domain(levels=["lamps"], split="validation", render=False); e = D.start(D.eval_specs(1))[0]
        dark = render("lamps", e.objects()); e.lit = {0}; lit = render("lamps", e.objects())
        for a, b in zip(dark, lit):
            cols = np.nonzero(np.abs(np.asarray(b, float) - np.asarray(a, float)).sum(-1) > 60)[1]
            self.assertTrue(len(cols)); self.assertLess(cols.mean(), SIZE / 2)


if __name__ == "__main__":
    unittest.main()
