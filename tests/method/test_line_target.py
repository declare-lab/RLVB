"""--line_target: a belief line is a positive target of the M-step only when the teacher verifies it.

The E-step weights the own turns by the teacher's score, and with --accept_only a turn has a weight when the teacher accepts its
action. Where the model seldom writes the exact belief line, the fully open line ("Possible: any") is its best sample: a wrong line
pays the floor, an open line pays a few nats. On the first sample set of the Qwen3.5-4B run the open lines were 9 percent of the
sampled turns and had 38 percent of the E-step weight, and after 7 updates 28 percent of the turns wrote one. With --line_target
TOL a turn whose line is not within TOL nats of the teacher's best line trains its accepted ACTION only: the line's characters are
not counted and its thought has no weight. These tests check the item (target_item, line_span), the two blocks' losses and
gradients on a small model (loss_items_blocks with the thought weights), and the run file."""
import math
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

import plearn.method.trainer as T
from plearn.method.trainer import CLOSE, TERM, line_span, loss_items_blocks, target_item, turn_item

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def counted(cont, spans, d0):
    """The characters that the decision block counts and those that the thought block counts, as score() masks them (one token a
    character): not inside a span; the decision from d0 on."""
    sp = spans if spans and isinstance(spans[0], (tuple, list)) else ([spans] if spans else [])
    keep = [i for i in range(len(cont)) if not any(i < e and i + 1 > s for s, e in sp)]
    return "".join(cont[i] for i in keep if i >= d0), "".join(cont[i] for i in keep if i < d0)


class ItemTests(unittest.TestCase):
    def test_the_span_is_the_belief_line_in_either_order(self):
        z, b, a = "think\n", "Possible: any", "ACTION: LOOK 3"
        cont, _, _ = turn_item(z, b, a, True, own_term=True)             # the belief line first
        s, e = line_span(z, b, a, True); self.assertEqual(cont[s:e], b)
        cont, _, _ = turn_item(z, a, b, True, own_term=True)             # the model wrote the action line first
        s, e = line_span(z, a, b, True); self.assertEqual(cont[s:e], b)
        s, e = line_span(z, b, b, True); self.assertEqual((s, e), (len(z) + len(CLOSE), len(z) + len(CLOSE) + len(b)))   # two lines alike: the first

    def test_a_verified_turn_is_the_whole_turn(self):
        for stopped in (True, False):
            for own in (True, False):
                z, b, a = "think\n", "Possible: 2", "ACTION: SUBMIT 2"
                self.assertEqual(target_item(z, b, a, stopped, own, True), turn_item(z, b, a, stopped, own) + (1.0,))

    def test_an_unverified_turn_counts_only_its_action(self):
        z, b, a = "think\n", "Possible: any", "ACTION: LOOK 3"
        for first, second in ((b, a), (a, b)):                           # either order of the two lines
            cont, spans, d0, tf = target_item(z, first, second, True, True, False)
            self.assertEqual(cont, turn_item(z, first, second, True, True)[0])   # the text is the turn as the model wrote it
            self.assertEqual(tf, 0.0); self.assertEqual(counted(cont, spans, d0)[0], a)
        cont, spans, d0, tf = target_item(z, b, a, True, True, True)      # the verified turn: both lines, not the separator
        self.assertEqual(tf, 1.0); self.assertEqual(counted(cont, spans, d0), (b + a, z + CLOSE))

    def test_a_cut_thought_and_the_old_terminators(self):
        z, b, a = "think\n", "Possible: any", "ACTION: LOOK 3"
        cont, spans, d0, tf = target_item(z, b, a, False, True, False)    # a thought cut at the cap: its forced </think> stays out
        self.assertIn((len(z), d0), spans); self.assertEqual(counted(cont, spans, d0), (a, z))
        cont, spans, d0, tf = target_item(z, b, a, True, False, False)    # --dec_term term: the two terminators are counted
        self.assertEqual(counted(cont, spans, d0)[0], TERM + a + TERM); self.assertEqual(tf, 0.0)
        cont, spans, d0, tf = target_item(z, b, a, False, False, False)
        self.assertEqual(counted(cont, spans, d0), (TERM + a + TERM, z))


class Bigram(torch.nn.Module):
    """A model whose next-token logits are one row of a table: the row of the token before. A row has a gradient only when a
    counted token follows its token."""
    def __init__(self, V): super().__init__(); self.E = torch.nn.Parameter(torch.zeros(V, V))
    @property
    def device(self): return torch.device("cpu")
    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep, **kw):
        return types.SimpleNamespace(logits=self.E[input_ids][:, -logits_to_keep:, :])


class BlockTests(unittest.TestCase):
    """loss_items_blocks on the small model: a turn with a line that is not verified gives the thought block nothing and gives the
    decision block the action's tokens only."""
    def tok(self):
        try:
            from tokenizers import Regex, Tokenizer, models, pre_tokenizers
            from transformers import PreTrainedTokenizerFast
        except ImportError: self.skipTest("no tokenizers library")
        alphabet = sorted(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ </>:\n."))
        t = Tokenizer(models.WordLevel({c: i for i, c in enumerate(alphabet + ["<pad>", "<unk>"])}, unk_token="<unk>"))
        t.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
        return PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", unk_token="<unk>")

    def blocks(self, tok, verified, w=0.5, N=2):
        z, b, a = "abc", "DEF", "GHI"                                     # no character twice: a row of the table = one position
        cont, spans, d0, tf = target_item(z, b, a, True, True, verified)
        model = Bigram(len(tok)); params = [model.E]
        with mock.patch.dict(T.SAMPLER, {"T": 1.0, "top_p": 1.0, "top_k": -1, "presence": 0.0}), \
                mock.patch.dict(T.DECISION, {"T": 1.0, "top_p": 1.0, "top_k": -1, "presence": 0.0}), \
                mock.patch.dict(os.environ, {"PL_CHUNKED_HEAD": "0"}), mock.patch.object(T, "PROC", None):
            lt, ld, gt, gd, _ = loss_items_blocks(model, tok, [("x y ", cont, spans, d0, "blocks")], N, 1, [w], params, False, [w * tf])
        row = lambda g, c: g[0][tok.convert_tokens_to_ids(c)]
        return lt, ld, gt, gd, row, math.log(len(tok)), (z, b, a)

    def test_an_unverified_turn_trains_its_action_only(self):
        tok = self.tok(); lt, ld, gt, gd, row, logV, (z, b, a) = self.blocks(tok, False)
        self.assertEqual(lt, 0.0); self.assertEqual(float(gt[0].abs().sum()), 0.0)       # the thought and its </think>: no weight
        self.assertAlmostEqual(ld, 0.5 * len(a) * logV / 2, places=5)                    # the action's tokens, each -log V
        for c in "DE": self.assertEqual(float(row(gd, c).abs().sum()), 0.0)              # the rows that predict the belief line's tokens
        for c in "GH": self.assertGreater(float(row(gd, c).abs().sum()), 0.0)            # the rows that predict the action's tokens

    def test_a_verified_turn_trains_the_whole_turn(self):
        tok = self.tok(); lt, ld, gt, gd, row, logV, (z, b, a) = self.blocks(tok, True)
        self.assertAlmostEqual(lt, 0.5 * (len(z) + len(CLOSE)) * logV / 2, places=5)
        self.assertAlmostEqual(ld, 0.5 * (len(b) + len(a)) * logV / 2, places=5)
        for c in "DEGH": self.assertGreater(float(row(gd, c).abs().sum()), 0.0)
        for c in "ab": self.assertGreater(float(row(gt, c).abs().sum()), 0.0)

    def test_without_thought_weights_the_blocks_are_the_old_ones(self):
        tok = self.tok(); z, b, a = "abc", "DEF", "GHI"; cont, spans, d0 = turn_item(z, b, a, True, True)
        out = []
        for wt in (None, [0.5]):
            model = Bigram(len(tok))
            with mock.patch.dict(T.SAMPLER, {"T": 1.0, "top_p": 1.0, "top_k": -1, "presence": 0.0}), \
                    mock.patch.dict(T.DECISION, {"T": 1.0, "top_p": 1.0, "top_k": -1, "presence": 0.0}), \
                    mock.patch.dict(os.environ, {"PL_CHUNKED_HEAD": "0"}), mock.patch.object(T, "PROC", None):
                out.append(loss_items_blocks(model, tok, [("x y ", cont, spans, d0, "blocks")], 2, 1, [0.5], [model.E], False, wt))
        (lt0, ld0, gt0, gd0, _), (lt1, ld1, gt1, gd1, _) = out
        self.assertEqual((lt0, ld0), (lt1, ld1)); self.assertTrue(torch.equal(gt0[0], gt1[0])); self.assertTrue(torch.equal(gd0[0], gd1[0]))


class RunFileTests(unittest.TestCase):
    def test_the_run_file_changes_only_the_line_rule(self):
        from plearn.train import commands, load
        sys.path.insert(0, os.path.join(ROOT, "tests", "method"))
        from test_run_files import flags
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "Qwen3.5-4B"))
            env = {k: v for k, v in os.environ.items() if k not in ("PL_GPUS", "PL_SEQS", "PL_PRESENCE_PENALTY", "QWEN35_4B_PATH")}
            with mock.patch.dict(os.environ, {**env, "PL_MODELS": d, "PL_RUNS": d}, clear=True):
                base = commands(load(os.path.join(ROOT, "configs", "train", "suite_qwen35_4b_presence_8k.yaml")), 0, stamp="0000-0000")
                c = commands(load(os.path.join(ROOT, "configs", "train", "suite_qwen35_4b_line_target.yaml")), 0, stamp="0000-0000")
                for role in ("actor", "learner"):
                    with self.subTest(role=role):
                        a, b = flags(c[role]), flags(base[role])
                        self.assertEqual(a.pop("--line_target"), "0.1"); self.assertEqual(b.pop("--line_target"), "0.1")   # the base file has line_target 0.1 too
                        self.assertTrue(b.pop("--line_wide")); self.assertNotIn("--line_wide", a)   # line_wide: on in the base file, off in this file
                        self.assertEqual((a["--max_think"], a["--score_presence"]), ("8192", "1"))
                        for k in ("--tag", "--out", "--queue", "--dump_em", "--dump_rollouts", "--wandb_name"): a.pop(k, None); b.pop(k, None)
                        self.assertEqual(a, b)
                self.assertEqual(c["env"], base["env"])                    # the sampler and the servers are the same

    def test_the_trainers_default_is_off(self):
        src = open(os.path.join(ROOT, "plearn", "method", "trainer.py")).read()
        self.assertIn('ap.add_argument("--line_target", type=float, default=-1.0,', src)
        self.assertIn("if args.line_target < 0 or not g.get(\"crit_h\"): return None", src)   # off: no flags, em_update takes the old path
        self.assertIn('assert args.line_target < 0 or (args.objective == "critic" and args.m_step == "blocks")', src)   # the two-block M-step only

    def test_a_saved_first_set_does_not_depend_on_the_rule(self):
        """PL_SET_FROM: the rule changes the M-step only, so a run with another --line_target may take a saved first set."""
        src = open(os.path.join(ROOT, "plearn", "method", "trainer.py")).read()
        a = src.index('free = {"role", "tag", "out"'); self.assertIn('"line_target"', src[a:src.index("}", a)])

    def test_the_metric_reaches_wandb(self):
        self.assertTrue(T.WANDB_KEEP.match("train/w_line_target"))


if __name__ == "__main__":
    unittest.main()
