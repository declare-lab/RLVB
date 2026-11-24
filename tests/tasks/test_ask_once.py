"""The ask-once baseline of the test (plearn/evaluator.py ask_once, PL_ASK_ONCE=1): the first SUBMIT of an episode
is checked once by the agent itself before it is carried out. Off by default."""
import os
import unittest
from unittest import mock


class Ep:
    one_line = False
    def __init__(self, parts): self.parts, self.steps, self.done = parts, [], False
    def messages(self, carry): return "SYSTEM", self.parts, []
    def step(self, b, a, rec):
        self.steps.append((b, a, rec)); self.done = a.upper().startswith("ACTION: SUBMIT") or len(self.steps) >= 4
    def row(self): return {"correct": True}
    def close(self): pass


class Proc:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        c = msgs[1]["content"]; return msgs[0]["content"] + "|" + (c if isinstance(c, str) else "".join(x.get("text", "") for x in c)) + "|"


class AskOnce(unittest.TestCase):
    def test_is_submit(self):
        from plearn.evaluator import is_submit
        self.assertTrue(is_submit("ACTION: SUBMIT 3")); self.assertTrue(is_submit("submit cup 2")); self.assertFalse(is_submit("ACTION: LIFT 2"))
        self.assertFalse(is_submit("")); self.assertFalse(is_submit(None))

    def test_the_note_follows_the_tasks_own_prompt_for_text_and_for_parts(self):
        from plearn.evaluator import prompt_of
        with mock.patch("plearn.models.sampler.tag_prompt", lambda text, images: text):
            self.assertEqual(prompt_of(Proc(), Ep("TASK"), True), "SYSTEM|TASK|")
            self.assertEqual(prompt_of(Proc(), Ep("TASK"), True, " NOTE"), "SYSTEM|TASK NOTE|")
            self.assertEqual(prompt_of(Proc(), Ep([{"type": "text", "text": "TASK"}]), True, " NOTE"), "SYSTEM|TASK NOTE|")

    def run_check(self, second):
        from plearn import evaluator as EV
        seen = []
        def fake_turn(srv, p, g, use_adapter=True, priority=0, **kw): seen.append((p, kw)); return ("z",) + second + ({"thought_chars": 1},)
        with mock.patch("plearn.models.sampler.tag_prompt", lambda text, images: text), mock.patch.object(EV, "turn", fake_turn):
            out = EV.ask_once(None, Proc(), Ep("TASK"), {"carry": True}, "Possible: 2", "ACTION: SUBMIT 2", {"thought_chars": 9}, one_line=False)
        return out, seen

    def test_the_agent_sees_its_answer_and_one_request_and_no_hint(self):
        (b, a, rec, info), seen = self.run_check(("Possible: 2", "ACTION: SUBMIT 2"))
        self.assertEqual(len(seen), 1); p = seen[0][0]
        self.assertIn("You replied:\nPossible: 2\nACTION: SUBMIT 2\nBefore this answer is accepted", p)
        self.assertIn("check once more whether the evidence identifies the answer", p); self.assertNotIn("Your reasoning", p)
        self.assertEqual(seen[0][1], {"one_line": False})
        self.assertEqual((b, a), ("Possible: 2", "ACTION: SUBMIT 2")); self.assertTrue(info["kept"]); self.assertFalse(info["went_on"])
        self.assertEqual((rec["asked"], rec["first_action"], rec["first_belief"]), (True, "ACTION: SUBMIT 2", "Possible: 2"))

    def test_the_agent_may_go_on_or_change_its_answer(self):
        (b, a, rec, info), _ = self.run_check(("Possible: 2, 3", "ACTION: LIFT 3"))
        self.assertEqual(a, "ACTION: LIFT 3"); self.assertEqual((info["kept"], info["changed_answer"], info["went_on"]), (False, False, True))
        (b, a, rec, info), _ = self.run_check(("Possible: 3", "ACTION: SUBMIT 3"))
        self.assertEqual((info["kept"], info["changed_answer"], info["went_on"]), (False, True, False))

    def test_the_thought_is_shown_only_when_asked_for(self):
        from plearn import evaluator as EV
        seen = []
        def fake_turn(srv, p, g, use_adapter=True, priority=0, **kw): seen.append(p); return "z", "", "ACTION: SUBMIT 2", {}
        with mock.patch("plearn.models.sampler.tag_prompt", lambda text, images: text), mock.patch.object(EV, "turn", fake_turn):
            EV.ask_once(None, Proc(), Ep("TASK"), {"carry": True}, "", "ACTION: SUBMIT 2", {}, thought="I counted the cups.", one_line=True)
        self.assertIn("Your reasoning for this reply was:\nI counted the cups.\n\nYou replied:\nACTION: SUBMIT 2\nBefore", seen[0])

    def test_off_by_default(self):
        self.assertNotEqual(os.environ.get("PL_ASK_ONCE"), "1")


if __name__ == "__main__":
    unittest.main()
