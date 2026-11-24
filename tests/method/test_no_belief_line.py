"""--no_belief_line (plearn/method/trainer.py, 24 Nov 2026): no belief line at all. The prompt has the action-only rules
of the one-line turn (plearn/tasks/domain.py action_only) and no carried line; an own turn is the thought, </think> and the action
line (turn_item); the action's prefix is the thought and </think> (after_line). Off (the default) nothing changes. CPU only; the
prompt tests need the tokenizer of Qwen3-VL-4B-Thinking in $PL_MODELS (default models/), else they are skipped."""
import os, random, sys, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import plearn.method.trainer as T
from plearn.api.episode import CARRY_MARK
from plearn.tasks.domain import action_only, system_text

TOK = os.path.join(os.environ.get("PL_MODELS", "models"), "Qwen3-VL-4B-Thinking")


class Flag:
    def __init__(self, on): self.on = on
    def __enter__(self): self.old = T.NO_BELIEF_LINE, T.CARRY; T.NO_BELIEF_LINE = self.on; T.CARRY = True
    def __exit__(self, *a): T.NO_BELIEF_LINE, T.CARRY = self.old


class TurnTextTests(unittest.TestCase):
    def test_off_is_the_two_line_turn(self):
        with Flag(False):
            self.assertEqual(T.after_line("P", "Possible: x"), "P" + "Possible: x" + T.TERM)
            cont, sp, d0 = T.turn_item("z", "Possible: x", "ACTION: UP", True, True)
            self.assertEqual(cont, "z" + T.CLOSE + "Possible: x" + T.TERM + "ACTION: UP"); self.assertEqual(d0, len("z") + len(T.CLOSE))
            self.assertEqual(sp, [(d0 + len("Possible: x"), d0 + len("Possible: x") + len(T.TERM))])

    def test_on_is_thought_close_action(self):
        with Flag(True):
            self.assertEqual(T.after_line("P", ""), "P")
            for own, stopped in ((True, True), (True, False), (False, True), (False, False)):
                cont, sp, d0 = T.turn_item("zz", "", "ACTION: UP", stopped, own)
                self.assertEqual(cont, "zz" + T.CLOSE + "ACTION: UP" + ("" if own else T.TERM))
                self.assertEqual(d0, len("zz") + len(T.CLOSE)); self.assertEqual(cont[d0:d0 + len("ACTION: UP")], "ACTION: UP")
                cut = None if stopped else (len("zz"), d0)
                self.assertEqual(sp, ([cut] if cut else None) if own else cut)   # only a protocol </think> is not counted; no span = None
                self.assertTrue(sp is None or len(sp) > 0)                       # 24 Nov: an empty span list crashed score() (Run A, update 1)
            cont, sp, d0, f = T.target_item("zz", "", "ACTION: UP", True, True, line_target=False)
            self.assertIsNone(sp); self.assertEqual(f, 0.0)
            with self.assertRaises(AssertionError): T.turn_item("zz", "Possible: x", "ACTION: UP", True, True)

    def test_spans_reach_the_counters(self):
        """24 Nov 2026: Run A crashed at update 1 in score() (attn_check, per_token): the span list of a stopped own turn was [] and
        score() made it [[]]. The span forms of every own turn must pass score()'s and block_counts()'s normalization."""
        class Tok:
            def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
                return {"offset_mapping": [(i, i + 1) for i in range(len(text))], "input_ids": list(range(len(text)))}
        for flag in (True, False):
            with Flag(flag):
                for own, stopped in ((True, True), (True, False), (False, True), (False, False)):
                    anc = "" if flag else "Possible: x"
                    cont, sp, d0 = T.turn_item("zz", anc, "ACTION: UP", stopped, own)
                    n = T.block_counts(Tok(), ("p", cont, sp, d0))
                    self.assertEqual(sum(n), len(cont) - sum(e - s for s, e in ([] if not sp else (sp if isinstance(sp, list) else [sp]))))
                    for lt in (True, False):
                        c2, sp2, d2, _ = T.target_item("zz", anc, "ACTION: UP", stopped, own, line_target=lt)
                        T.block_counts(Tok(), ("p", c2, sp2, d2))

    def test_the_guard(self):
        self.assertTrue(T.no_line_prompt("rules: reply with one line: 'ACTION: UP'"))
        for bad in ("reply with exactly two lines.\nLine 1: x\nLine 2: y", f"obs\n{CARRY_MARK}Possible: 1234\nrule"):
            self.assertFalse(T.no_line_prompt(bad))


@unittest.skipUnless(os.path.isdir(TOK), "no local tokenizer")
class PromptTests(unittest.TestCase):
    """The trainer's prompt of the tasks of train_suite, at the first turn and after a look with a belief line in e.note."""
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor, AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(TOK); cls.proc = AutoProcessor.from_pretrained(TOK)

    def setUp(self): self.old_proc = T.PROC; T.PROC = self.proc      # the image tasks: the trainer's processor (set in train())
    def tearDown(self):
        from plearn.models.sampler import IMAGES
        IMAGES.clear(); T.PROC = self.old_proc

    def episode(self, mod):
        import importlib
        D = importlib.import_module(mod).Domain(split="train")
        e = D.start([D.sample(random.Random(3))])[0]; e.t = getattr(e, "t", 0); e.note = None
        return D, e

    def check(self, mod):
        D, e = self.episode(mod)
        with Flag(False): two = T.turn_prompt(D, self.tok, e)
        with Flag(True): one = T.turn_prompt(D, self.tok, e)
        self.assertIn("reply with exactly two lines", two); self.assertNotIn("reply with exactly two lines", one)
        self.assertIn(action_only(system_text(D, e)), one)                 # the action-only rules of the one-line turn
        looks = D.look_strings(e)
        if looks:
            D.do_look(e, list(looks)[0]); e.t += 1; e.note = "Possible: any"
            with Flag(False): self.assertIn(CARRY_MARK, T.turn_prompt(D, self.tok, e))   # --carry: the line is carried
            with Flag(True):
                p = T.turn_prompt(D, self.tok, e); self.assertNotIn(CARRY_MARK, p); self.assertNotIn("Possible: any", p)
        D.close([e])

    def test_mastermind(self): self.check("plearn.tasks.mastermind.teacher")
    def test_minesweeper(self): self.check("plearn.tasks.minesweeper.teacher")
    def test_minigrid(self): self.check("plearn.tasks.minigrid.teacher")
    def test_frozenlake(self): self.check("plearn.tasks.frozenlake.teacher")
    def test_sokoban(self): self.check("plearn.tasks.sokoban.teacher")


if __name__ == "__main__":
    unittest.main()
