"""The one-line turn of the training games (plearn/tasks/domain.py one_line, --task_args <dataset>.one_line=1): the prompt is the
two-line prompt of the same state with the action-only rules (plearn/tasks/domain.py action_only) and no carried belief line, the
evaluator asks for no belief line, the action is the first ACTION: line of the reply, and with one_line not set every prompt is the
prompt of 8662fb5 (the reference below is that code). A game whose simulator this machine does not have is checked through a stand-in
domain of the same kind (pictures or text)."""
import json
import sys
import types
import unittest

from plearn.api.episode import CARRY_MARK, action_line, carried
from plearn.api.registry import RegistryError, registry
from plearn.api.task import Task
from plearn.tasks import domain as TD

GAMES = ("mastermind", "minesweeper_easy", "minigrid_keycorridor", "frozenlake_hidden_goal", "sokoban_hidden_target",
         "tabletop_lamps", "tabletop_cups", "tabletop_blocks")


def reference_messages(ep, carry):
    """DomainEpisode.messages of 8662fb5, as it was (the default-off reference)."""
    D, e = ep.D, ep.e; note = e.note if carry else None
    with ep.lock:
        if getattr(D, "images", False):
            import inspect
            parts, images = D.user_content(e, note=note) if "note" in inspect.signature(D.user_content).parameters else D.user_content(e)
        else:
            text, images = D._history(e), []
            if note:
                head, sep, tail = text.rpartition(" Decide now.")
                text = head + carried(note) + "\nDecide now." + tail if sep else text + carried(note)
            parts = text
        return TD.system_text(D, e), parts, images


def same(a, b):
    """Two (system, parts, pictures) are the same: equal texts and parts, the same pictures."""
    return a[0] == b[0] and a[1] == b[1] and len(a[2]) == len(b[2]) and all(x is y or x == y for x, y in zip(a[2], b[2]))


class Script:
    """A model server for the evaluator: the reasoning, then the action, by the request's stop."""
    def __init__(self, actions):
        self.actions, self.calls = list(actions), []
    def complete(self, prompts, n=1, max_tokens=0, stop=None, **kw):
        kind = "think" if stop and "</think>" in stop else "line"
        self.calls.append((kind, max_tokens, prompts[0]))
        text = "I think." if kind == "think" else self.actions.pop(0)
        return [[{"text": text, "finish": "stop"}]]


class Proc:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        body = lambda c: c if isinstance(c, str) else "".join(p.get("text", "<image>") for p in c)
        return "".join(f"<{m['role']}>{body(m['content'])}" for m in msgs) + "<assistant>"


class OneLineGamesTests(unittest.TestCase):
    def runnable(self):
        reg = registry(); out = []
        for name in GAMES:
            t = Task.load(name, reg)
            try: t.check(); out.append(t)
            except RegistryError: print(f"  {name}: not on this machine ({t.requires})", file=sys.stderr)
        self.assertTrue(out, "no game of the 8 runs on this machine")
        return out

    def test_prompt_is_the_training_prompt_and_default_is_unchanged(self):
        for t in self.runnable():
            with self.subTest(task=t.name):
                spec = t.specs(None, 1)[0]
                one = TD.DomainEpisode(spec, one_line=True, **t.env_args)
                base = t.episode(spec)                                    # one_line not set
                try:
                    self.assertFalse(base.one_line)
                    for k in range(2):
                        ref = reference_messages(one, False)
                        self.assertTrue(same(one.messages(True), (TD.action_only(ref[0]), ref[1], ref[2])))
                        self.assertIn("reply with one line", one.messages(True)[0])
                        for c in (True, False): self.assertTrue(same(base.messages(c), reference_messages(base, c)))
                        a = sorted(one.D.look_strings(one.e).values())[:1] or ["ACTION: none"]
                        one.step("", a[0], {}); base.step("Possible: all", a[0], {})
                        if one.done: break
                    self.assertNotIn(CARRY_MARK, json.dumps(one.messages(True)[1], default=str))
                    self.assertIn(CARRY_MARK, json.dumps(base.messages(True)[1], default=str))
                finally:
                    for ep in (one, base): ep.close()

    def test_the_evaluator_asks_for_no_belief_line(self):
        from plearn import evaluator
        for t in self.runnable()[:1]:
            t1 = Task.load(t.name, registry()); t1.env_args.update(one_line=1)   # as plearn/cli.py applies --task_args
            srv = Script(["ACTION: none\nmore"] * 3 + ["ACTION: SUBMIT 1 2 3 4"] * 40)
            rows, _ = evaluator.play(srv, Proc(), [t1], n=1, log=lambda *_: None)
            row = rows[t.name][0]; self.assertNotIn("error", row)
            kinds = [k for k, _, _ in srv.calls]
            self.assertEqual(kinds, ["think", "line"] * row["turns"])     # no belief-line request
            for k, _, p in srv.calls:
                self.assertIn("reply with one line", p); self.assertNotIn(CARRY_MARK, p)
            self.assertEqual(row["turn_log"][0]["belief"], ""); self.assertEqual(row["turn_log"][0]["action"], "ACTION: none")
            srv2 = Script(["Possible: all"] + ["ACTION: none"] * 3 + ["Possible: all", "ACTION: SUBMIT 1 2 3 4"] * 40)
            rows2, _ = evaluator.play(srv2, Proc(), [Task.load(t.name, registry())], n=1, log=lambda *_: None)
            self.assertEqual(set(row) - {"turn_log"}, set(rows2[t.name][0]) - {"turn_log"})   # the same result fields

    def test_the_action_line_of_a_reply(self):
        """The first line that starts with ACTION:, else the first line that is not empty."""
        for c, want in (("ACTION: GUESS [1 2 3 4]", "ACTION: GUESS [1 2 3 4]"),
                        ("\n\nACTION: GUESS [1 2 3 4]\nPossible: all", "ACTION: GUESS [1 2 3 4]"),
                        ("Possible: all\nACTION: FIRE", "ACTION: FIRE"), ("  guess 1 2 3 4\nACTION: SUBMIT x", "ACTION: SUBMIT x"),
                        ("no action here", "no action here"), ("\n\n  first line  \nsecond", "first line"), ("", "")):
            self.assertEqual(action_line(c), want, c)


class StandInDomainTests(unittest.TestCase):
    """The same code path for a game this machine cannot run: a domain with pictures (as tabletop, minesweeper, minigrid,
    sokoban) and one with text (as mastermind, frozenlake)."""
    SYSTEM = "Find it. Each turn, reply with exactly two lines.\nLine 1: Possible: <the options>\nLine 2: ACTION: LOOK <n> or ACTION: SUBMIT <n>"

    def domain(self, images):
        mod = types.ModuleType(f"standin_domain_{int(images)}"); system = self.SYSTEM

        class Domain:
            T, turns = 3, 4
            def __init__(self, split, **_): self.split = split
            def eval_specs(self, n): return [("standin", i, self.split) for i in range(n)]
            def start(self, specs): return [types.SimpleNamespace(spec=s, log=[]) for s in specs]
            def system_text(self, e): return system
            def close(self, eps): pass
        if images:
            Domain.images = True
            def user_content(self, e, note=None):
                return ([{"type": "text", "text": f"Seen: {e.log}"}, {"type": "image"}, {"type": "text", "text": carried(note) + " Decide now."}],
                        [f"picture-{len(e.log)}"])
            Domain.user_content = user_content
        else:
            Domain._history = lambda self, e: f"Seen: {e.log}. Decide now."
        mod.Domain = Domain; sys.modules[mod.__name__] = mod
        return mod.__name__

    def test_one_line_and_default(self):
        for images in (True, False):
            with self.subTest(images=images):
                name = self.domain(images); spec = ["test", ["standin", 0, "test"]]
                one = TD.DomainEpisode(spec, domain=name, one_line=True); base = TD.DomainEpisode(spec, domain=name)
                one.e.note = base.e.note = "Possible: 1"
                system, parts, pics = one.messages(True)
                self.assertEqual(system, TD.action_only(self.SYSTEM)); self.assertIn("reply with one line: ", system)
                self.assertNotIn(CARRY_MARK, json.dumps(parts)); self.assertEqual(pics, reference_messages(one, False)[2])
                self.assertEqual((parts, system), (reference_messages(one, False)[1], TD.action_only(reference_messages(one, False)[0])))
                for c in (True, False): self.assertTrue(same(base.messages(c), reference_messages(base, c)))
                self.assertIn(CARRY_MARK + "Possible: 1", json.dumps(base.messages(True)[1]))


if __name__ == "__main__":
    unittest.main()
