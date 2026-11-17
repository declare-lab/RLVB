"""--dec_term own (fix A: the decision target is the turn as the model wrote it; the METHOD default)."""
import math
import os
import tempfile
import types
import unittest
from unittest import mock

import torch

import plearn.method.trainer as T
from plearn.method.trainer import CLOSE, TERM, turn_item
from plearn.train import METHOD, commands


class TurnItemTests(unittest.TestCase):
    def test_term_target_is_the_old_one(self):
        self.assertEqual(turn_item("zz", "B", "A", True), ("zz" + CLOSE + "B" + TERM + "A" + TERM, None, 2 + len(CLOSE)))
        self.assertEqual(turn_item("zz", "B", "A", False)[1], (2, 2 + len(CLOSE)))

    def test_own_drops_the_last_term_and_masks_the_middle_one(self):
        cont, sp, d0 = turn_item("zz", "B", "A", True, own_term=True)
        self.assertEqual(cont, "zz" + CLOSE + "B" + TERM + "A"); self.assertEqual(d0, 2 + len(CLOSE))
        self.assertEqual(sp, [(d0 + 1, d0 + 1 + len(TERM))]); self.assertEqual(cont[sp[0][0]:sp[0][1]], TERM)

    def test_own_keeps_the_forced_close_span_of_a_cut_thought(self):
        cont, sp, d0 = turn_item("zz", "B", "A", False, own_term=True)
        self.assertEqual(sp[0], (2, d0)); self.assertEqual(cont[sp[1][0]:sp[1][1]], TERM)


class ScoreSpanTests(unittest.TestCase):
    """score() with a list of spans: the masked tokens get no term (uniform stand-in logits: each counted token = -log V)."""
    def test_a_list_of_spans_masks_each(self):
        base = os.path.join(os.environ.get("PL_MODELS", "models"), "Qwen3-VL-8B-Thinking")
        if not os.path.isdir(base): self.skipTest("no tokenizer")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base)
        V = len(tok)
        class M:
            device = torch.device("cpu")
            def __call__(self, input_ids, attention_mask, use_cache, logits_to_keep, **kw):
                return types.SimpleNamespace(logits=torch.zeros(input_ids.shape[0], logits_to_keep, V))
        cont = "Possible: row 2" + TERM + "ACTION: SUBMIT 2"; t0 = len("Possible: row 2")
        offs = tok(cont, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        n_all = len(offs); n_masked = sum(1 for a, b in offs if a < t0 + len(TERM) and b > t0)
        with mock.patch.dict(os.environ, {"PL_CHUNKED_HEAD": "0"}), mock.patch.object(T, "PROC", None):
            full = float(T.score(M(), tok, [("pre", cont, None, 0)], 1)[0])
            masked = float(T.score(M(), tok, [("pre", cont, [(t0, t0 + len(TERM))], 0)], 1)[0])
        self.assertAlmostEqual(full / -math.log(V), n_all, places=3)
        self.assertAlmostEqual(masked / -math.log(V), n_all - n_masked, places=3); self.assertGreaterEqual(n_masked, 1)


class FlagTests(unittest.TestCase):
    def test_method_uses_own_and_it_reaches_the_trainer(self):
        self.assertEqual(METHOD["dec_term"], "own")
        c = {"train": "t", "model": "qwen3-vl-8b-thinking", "train_tasks": ["minigrid_keycorridor"], "output": "/tmp/_fix_a_test",
             "gpus": {"actor": "0", "learner": "0"}}
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "Qwen3-VL-8B-Thinking"))
            with mock.patch.dict(os.environ, {"PL_MODELS": d}):
                r = commands(c, 0, stamp="0000-0000")
        for a in (r["learner"], r["actor"]):
            self.assertEqual(a[a.index("--dec_term") + 1], "own")

    def test_the_parser_default_is_term(self):
        self.assertIn('"--dec_term", choices=["term", "own"], default="term"', open(T.__file__).read())


if __name__ == "__main__":
    unittest.main()
