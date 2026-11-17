"""--update grpo (RLVB (GRPO)): the ablation of RLVB (EM)'s update rule, group-relative GRPO on RLVB (EM)'s own histories. The scores
are the critic's (turn_score), each own turn's advantage is relative to the other turns of its history (grpo_advantages: leave-one-
out, floored scores raised, a rejected action never pushed up, equal scores exactly 0), a batch's weights have RLVB (EM)'s total size
(grpo_weights), each block stops by PPO's clip rule against the phase prior (ppo_keep in loss_items_blocks, per token over the
tokens score() counts: block_counts), and the run file is the RLVB (EM) 8B run file with four flags added. The sample sets are RLVB (EM)'s (no
change to the leave-out rules or to keep_set), so an RLVB (EM) set can be reused (PL_SET_FROM)."""
import ast
import contextlib
import io
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from plearn.method import trainer as T
from plearn.method.trainer import block_counts, grpo_advantages, grpo_weights, ppo_keep, target_item, turn_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def old_turn_score(line_score, line_mix, S, W, Sd, A):
    """The critic's R as score_turn wrote it inline before turn_score."""
    return (S + W if line_score in ("reader", "exact") else Sd + line_mix * (S + W) if Sd is not None else 0.0) + A


# the worked example of the Teacher Turn Rewards note: tabletop cups, cup 1 lifted and empty (t = 1), cups 2-4 possible; RLVB (EM)'s
# scores with value log, tau 0.1, line_score decision, line_mix 0.1, length weight 4; four own turns a (LIFT 2, short thought),
# b (LIFT 3, long thought), c (premature SUBMIT 2 with the line 'cup 2'), d (an unreadable action line); a and b accepted
LIFT, SUB, WASTE, OUT, S_EXACT, S_CUP2 = -1.0987, -11.7474, -10.8505, -200.0, -1.0986, -33.3333
R_CUPS = np.array([turn_score("decision", 0.1, S_EXACT, 0.0, LIFT, LIFT) + 4 * 0.3182,
                   turn_score("decision", 0.1, S_EXACT, 0.0, LIFT, LIFT) - 4 * 0.5,
                   turn_score("decision", 0.1, S_CUP2, 0.0, SUB, SUB),
                   turn_score("decision", 0.1, S_EXACT, 0.0, LIFT, WASTE)])
ACC_CUPS = [True, True, False, False]


def char_tok():
    """The character tokenizer of tests/method/test_line_target.py: one token per character, with offsets."""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    alphabet = sorted(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ </>:\n."))
    t = Tokenizer(models.WordLevel({c: i for i, c in enumerate(alphabet + ["<pad>", "<unk>"])}, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    return PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", unk_token="<unk>")


class TurnScoreTests(unittest.TestCase):
    def test_the_same_score_as_before(self):
        rng = random.Random(0)
        for _ in range(500):
            mode = rng.choice(["decision", "exact", "reader", "none"])
            S, W, A = -rng.uniform(0, 60), -rng.uniform(0, 30), -rng.uniform(0, 200)
            Sd = None if mode == "none" else -rng.uniform(0, 200)
            mix = rng.choice([0.0, 0.1, 0.5])
            self.assertEqual(turn_score(mode, mix, S, W, Sd, A), old_turn_score(mode, mix, S, W, Sd, A))


class AdvantageTests(unittest.TestCase):
    def test_leave_one_out_on_the_cups_group(self):
        # c's score (-26.8) is raised to the best minus 20 (-21.0) before the means; then each turn against the mean of the others
        Rc = np.maximum(R_CUPS, R_CUPS.max() - 20.0)
        want = Rc - (Rc.sum() - Rc) / 3
        A = grpo_advantages(R_CUPS, ACC_CUPS, 20.0)
        self.assertTrue(np.allclose(A, want)); self.assertTrue(np.allclose(A, [11.43, 7.07, -15.23, -3.27], atol=0.01), A)
        self.assertGreater(A[0], A[1])                                # the short right turn above the long one

    def test_a_rejected_turn_is_never_pushed_up(self):
        # every turn rejected (premature SUBMIT 2, an unreadable line, SUBMIT 1 outside the posterior): without the rule the two less
        # bad turns would get about +9 and +10, so the update would raise a premature answer
        R = np.array([turn_score("decision", 0.1, S_EXACT, 0.0, LIFT, SUB), turn_score("decision", 0.1, S_EXACT, 0.0, LIFT, WASTE),
                      turn_score("decision", 0.1, S_EXACT, 0.0, LIFT, OUT)])
        free = grpo_advantages(R, None, 20.0)
        self.assertTrue(np.allclose(free, [9.10, 10.45, -19.55], atol=0.01), free)
        A = grpo_advantages(R, [False, False, False], 20.0)
        self.assertTrue(np.allclose(A, [0.0, 0.0, free[2]])); self.assertTrue((A <= 0).all())

    def test_equal_scores_give_exactly_zero(self):
        # a rounding residue (+8.9e-16 for eight scores of -7.77 before the shift) would be scaled up by grpo_weights' batch factor
        for s in (-7.77, -12.96, -0.1, -199.9):
            A = grpo_advantages([s] * 8, [True] * 8)
            self.assertTrue((A == 0).all(), (s, A))
        w, sc = grpo_weights([grpo_advantages([-7.77] * 8, [True] * 8) for _ in range(32)], 32)
        self.assertEqual(sc, 0.0); self.assertTrue(all((x == 0).all() for x in w))       # a batch with no signal: no step at all

    def test_groups_without_signal(self):
        self.assertTrue((grpo_advantages([-12.96] * 4, [False] * 4) == 0).all())   # equal scores: no relative advantage
        self.assertTrue((grpo_advantages([-4.0], [True]) == 0).all())             # one turn: no group
        self.assertEqual(len(grpo_advantages([], [])), 0)

    def test_the_floor_and_the_limit(self):
        R = np.array([-1.0, -3.0, -200.0])
        self.assertTrue(np.allclose(grpo_advantages(R, None, 0.0), [100.5, 97.5, -198.0]))    # clip 0: the floor lifts the others
        self.assertTrue(np.allclose(grpo_advantages(R, None, 20.0), [11.0, 8.0, -19.0]))      # raised to -21 first
        self.assertTrue((np.abs(grpo_advantages(np.array([0.0, 0.0, -60.0, -60.0]), None, 20.0)) <= 20.0).all())

    def test_scores_that_are_not_finite(self):
        A = grpo_advantages([-1.0, float("nan"), -3.0], None, 20.0)     # NaN counts as the lowest finite score minus clip (-23)
        self.assertTrue(np.isfinite(A).all()); self.assertTrue(np.allclose(A, grpo_advantages([-1.0, -23.0, -3.0], None, 20.0)))
        self.assertTrue(np.isfinite(grpo_advantages([-1.0, -float("inf")], None, 0.0)).all())   # clip 0: 200 nats below
        self.assertTrue((grpo_advantages([float("nan")] * 3, None) == 0).all())               # no finite score: no signal

    def test_the_batch_weights_have_pls_total(self):
        A_h = [grpo_advantages(R_CUPS, ACC_CUPS), np.zeros(4), grpo_advantages(np.array([-1.0, -9.0]), [True, True])]
        w, sc = grpo_weights(A_h, n_fit=3)
        self.assertAlmostEqual(float(sum(np.abs(x).sum() for x in w)), 3.0)                    # the E-step's: 1 per fitted history
        raw = [a / len(a) for a in A_h]
        for x, r in zip(w, raw): self.assertTrue(np.allclose(x, r * sc))                       # one factor: the relative weights kept
        w0, sc0 = grpo_weights(A_h, n_fit=3, norm=False); self.assertEqual(sc0, 1.0); self.assertTrue(np.allclose(w0[0], raw[0]))


class BlockCountTests(unittest.TestCase):
    """block_counts must count the tokens score() counts (its spans excluded), or the PPO stop's per-token ratio is diluted: the
    first count re-encoded each block's whole text, 37 decision tokens where score() counted the action's 2."""
    def setUp(self):
        try: self.tok = char_tok()
        except ImportError: self.skipTest("no tokenizers library")

    def test_the_spans_are_not_counted(self):
        z, b, a = "abc", "DEF", "GHI"
        CLOSE = len(T.CLOSE)
        cont, spans, d0, _ = target_item(z, b, a, False, True, False)     # a cut thought, --dec_term own, an unverified line
        self.assertEqual(block_counts(self.tok, ("p", cont, spans, d0, "blocks")), [3, 3])   # the forced </think>, the line, TERM out
        cont, spans, d0, _ = target_item(z, b, a, True, True, True)       # a thought that ended by itself, a verified line
        self.assertEqual(block_counts(self.tok, ("p", cont, spans, d0, "blocks")), [3 + CLOSE, 6])   # its own </think> counted; TERM out
        cont, spans, d0 = T.turn_item(z, b, a, True, False)               # --dec_term term: the TERMs are the turn's own
        self.assertEqual(block_counts(self.tok, ("p", cont, spans, d0, "blocks")), [3 + CLOSE, 8])

    def test_the_same_tokens_as_score(self):
        # the masks score() builds (its own code path, before the model) agree with the counts, token by token
        z, b, a = "abc", "DEF", "GHI"
        for stopped, own, verified in ((False, True, False), (True, True, True), (True, True, False), (False, True, True)):
            cont, spans, d0, _ = target_item(z, b, a, stopped, own, verified)
            enc = self.tok(cont, add_special_tokens=False, return_offsets_mapping=True)
            sp_ = list(spans) if spans and isinstance(spans[0], (tuple, list)) else [spans]
            m = [not any(x < e and y > s for s, e in sp_) for x, y in enc["offset_mapping"]]
            dec = [x >= d0 for x, _ in enc["offset_mapping"]]
            want = [sum(1 for k, dk in zip(m, dec) if k and not dk), sum(1 for k, dk in zip(m, dec) if k and dk)]
            self.assertEqual(block_counts(self.tok, ("p", cont, spans, d0, "blocks")), want)


class PPOStopTests(unittest.TestCase):
    def test_the_rule(self):
        w = torch.tensor([-1.0, -1.0, 1.0, 1.0, 0.0]); eps = 0.2
        r = torch.tensor([math.log(0.79), math.log(0.81), math.log(1.21), math.log(1.19), 5.0])
        self.assertEqual(ppo_keep(w, r, eps).tolist(), [False, True, False, True, True])

    def test_each_block_by_its_own_ratio_and_count(self):
        # two items whose thought and decision blocks move differently and have different counts (10 and 2 tokens): item 0 (pushed
        # down) moved -0.25 per thought token (stop) and -0.15 per decision token (keep); item 1 (pushed up) moved +0.2 and +0.25 (both
        # stop). Only item 0's decision keeps its gradient.
        old = torch.tensor([[-50.0, -3.0], [-40.0, -2.0]], dtype=torch.float64)
        delta = torch.tensor([[-2.5, -0.3], [2.0, 0.5]])
        p = torch.zeros(2, 2, requires_grad=True)
        def fake_score(model, tok, items, chunk, readout=False):
            idx = [it[0] for it in items]
            return old[idx].float() + delta[idx] + p[idx]
        items = [(k, "cont", None, 0, "blocks") for k in range(2)]
        ntok = torch.tensor([[10.0, 2.0], [10.0, 2.0]], dtype=torch.float64)
        with mock.patch.object(T, "score", fake_score):
            _, _, gt, gd, _ = T.loss_items_blocks(None, None, items, 1, 2, [-1.0, 1.0], [p], ppo=(old, ntok, 0.2))
        self.assertEqual((T.MSTEP["grpo_stop_t"], T.MSTEP["grpo_stop_d"], T.MSTEP["grpo_items"]), (2, 1, 2))
        self.assertEqual(float(gt[0].abs().sum()), 0.0)
        self.assertEqual(gd[0].tolist(), [[0.0, 1.0], [0.0, 0.0]])           # d(-w lp)/dp for item 0's decision, w = -1
        # a thought weight of 0 (--line_target, a line not verified) is no stop: item 0's thought is not counted then
        with mock.patch.object(T, "score", fake_score):
            T.loss_items_blocks(None, None, items, 1, 2, [-1.0, 1.0], [p], weights_t=[0.0, 1.0], ppo=(old, ntok, 0.2))
        self.assertEqual(T.MSTEP["grpo_stop_t"], 1)

    def fit(self, w, eps, steps):
        """loss_items_blocks with the scorer replaced by a 4-way softmax (one "turn" per item, both blocks the same), the old
        log-probabilities those of the start, `steps` gradient steps; returns the final probabilities and the stop counts."""
        logits = torch.zeros(4, requires_grad=True)
        def fake_score(model, tok, items, chunk, readout=False):
            lp = torch.log_softmax(logits, 0)[[it[0] for it in items]]
            return torch.stack([lp, lp], 1)
        items = [(k, "cont", None, 0, "blocks") for k in range(4)]
        with torch.no_grad(): old = torch.log_softmax(logits, 0).repeat(2, 1).T.double()
        ppo = (old, torch.ones(4, 2, dtype=torch.float64), eps) if eps else None
        with mock.patch.object(T, "score", fake_score):
            for _ in range(steps):
                _, _, gt, gd, _ = T.loss_items_blocks(None, None, items, 1, 4, list(w), [logits], ppo=ppo)
                with torch.no_grad(): logits -= 0.5 * gd[0]
        return torch.softmax(logits.detach(), 0), dict(T.MSTEP)

    def test_one_step_moves_the_turns_by_their_advantages(self):
        w, _ = grpo_weights([grpo_advantages(R_CUPS, ACC_CUPS)], 1)
        p, _ = self.fit(w[0], 0.0, 1)
        self.assertGreater(p[0], 0.25); self.assertLess(p[2], 0.25); self.assertLess(p[3], 0.25); self.assertEqual(int(torch.argmax(p)), 0)

    def test_the_clip_stops_the_push_down(self):
        # over reused samples, without the stop the pushed-down turn c falls to ~0 (its weight never lets go); with eps 0.2 every
        # item's gradient stops at its limit (one token per block here) and the fit freezes: 40 and 80 steps end in the same place.
        # c can end below 0.8 x its start, as in PPO: its own gradient stops, the rising turns still take probability through the
        # shared normalisation, until they stop too
        w, _ = grpo_weights([grpo_advantages(R_CUPS, ACC_CUPS)], 1)
        free, _ = self.fit(w[0], 0.0, 40)
        c40, ms = self.fit(w[0], 0.2, 40)
        c80, _ = self.fit(w[0], 0.2, 80)
        self.assertLess(float(free[2]), 1e-3)
        self.assertTrue(torch.allclose(c40, c80)); self.assertGreater(float(c40[2]), 0.1)
        self.assertEqual((ms["grpo_stop_t"], ms["grpo_stop_d"], ms["grpo_items"]), (4, 4, 4))   # every block stopped
        self.assertLessEqual(float(c40[0]), 0.25 * 1.2 * 1.1); self.assertGreaterEqual(float(c40[3]), 0.25 * 0.8 * 0.9)   # at their limits


class ClosureTests(unittest.TestCase):
    def test_em_update_reads_nothing_a_worker_rank_lacks(self):
        # a worker rank runs its job loop and returns before the rest of train() binds its names (`phase`, ...): every name em_update
        # reads must be its own, train()'s parameter, a module name, or bound in train() before that loop
        with open(T.__file__, encoding="utf-8") as f: tree = ast.parse(f.read())
        train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
        em = next(n for n in ast.walk(train) if isinstance(n, ast.FunctionDef) and n.name == "em_update")
        loop = next(n.lineno for n in ast.walk(train) if isinstance(n, ast.If) and isinstance(n.test, ast.UnaryOp)
                    and isinstance(n.test.operand, ast.Name) and n.test.operand.id == "lead")
        bound = {a.arg for a in train.args.args} | set(dir(T)) | set(dir(__builtins__) if isinstance(__builtins__, dict) is False else __builtins__)
        for n in train.body:                                          # the statements of train() before the worker loop
            if n.lineno >= loop: break
            for x in ast.walk(n):
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store): bound.add(x.id)
                if isinstance(x, (ast.FunctionDef, ast.ClassDef)): bound.add(x.name)
                if isinstance(x, (ast.Import, ast.ImportFrom)): bound |= {a.asname or a.name.split(".")[0] for a in x.names}
        local = {x.id for x in ast.walk(em) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)} | {a.arg for a in em.args.args}
        local |= {x.name for x in ast.walk(em) if isinstance(x, ast.FunctionDef)}
        reads = {x.id for x in ast.walk(em) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
        self.assertEqual(sorted(reads - local - bound), [])


class RunFileTests(unittest.TestCase):
    PAIRS = (("suite_qwen3vl_8b_thinking", "grpo_posterior_qwen3vl_8b_thinking"),       # the 3 models
             ("suite_qwen3vl_4b_thinking", "grpo_posterior_qwen3vl_4b_thinking"),
             ("suite_qwen35_4b_presence_8k", "grpo_posterior_qwen35_4b"))

    def test_the_run_file_is_the_pl_run_file_with_the_update_switched(self):
        for pl_file, gr_file in self.PAIRS:
            with self.subTest(gr_file): self.check_pair(pl_file, gr_file)

    def check_pair(self, pl_file, gr_file):
        from plearn.train import commands, load
        with tempfile.TemporaryDirectory() as d:
            for m in ("Qwen3-VL-8B-Thinking", "Qwen3-VL-4B-Thinking", "Qwen3.5-4B"): os.mkdir(os.path.join(d, m))
            env = {k: v for k, v in os.environ.items() if k not in ("PL_GPUS", "PL_SEQS", "PL_PRESENCE_PENALTY", "QWEN35_4B_PATH")}
            with mock.patch.dict(os.environ, {**env, "PL_MODELS": d, "PL_RUNS": d}, clear=True):
                pl = commands(load(os.path.join(ROOT, "configs", "train", pl_file + ".yaml")), 0, stamp="0000-0000")
                gr = commands(load(os.path.join(ROOT, "configs", "train", gr_file + ".yaml")), 0, stamp="0000-0000")
        def flags(argv):
            a = argv[argv.index("plearn.method.trainer") + 1:]; out = {}; i = 0
            while i < len(a):
                k = a[i]; v = a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else True
                out[k] = v; i += 2 if v is not True else 1
            return out
        mine = {"--update", "--grpo_clip", "--grpo_eps", "--grpo_norm", "--attn"}
        for role in ("actor", "learner"):
            f_pl, f_gr = flags(pl[role]), flags(gr[role])
            names = {"--tag", "--out", "--queue", "--dump_em", "--dump_rollouts", "--wandb_name"}
            self.assertEqual({k: v for k, v in f_gr.items() if k not in names | mine}, {k: v for k, v in f_pl.items() if k not in names})
            self.assertEqual([f_gr[k] for k in ("--update", "--grpo_clip", "--grpo_eps", "--grpo_norm", "--attn")], ["grpo", "20", "0.2", "1", "flash_attention_3"])
            self.assertNotIn("--update", f_pl)                          # the method's default: --update em
            self.assertNotIn("--attn", f_pl)                            # the RLVB (EM) runs keep the default: sdpa
        self.assertEqual(gr["env"], pl["env"])

    def test_a_pl_sample_set_can_start_the_run(self):
        # PL_SET_FROM compares the saved set's flags except the free ones; the RLVB (GRPO) flags are among them (no sample depends on them)
        with open(T.__file__, encoding="utf-8") as f: src = f.read()
        self.assertIn('"update", "grpo_clip", "grpo_eps", "grpo_norm", "attn"}', src)

    def test_the_metrics_reach_wandb(self):
        for k in ("grpo_scale", "grpo_adv_abs", "grpo_adv_neg", "grpo_adv_pos", "grpo_adv_zero_hist", "grpo_stop_t", "grpo_stop_d", "grpo_items",
                  "attn_flash", "attn_dlogp_mean", "attn_dlogp_max"):
            self.assertTrue(T.WANDB_KEEP.match(f"train/{k}"), k)


class AttnCheckTests(unittest.TestCase):
    """--attn: the flash attention is kept only when its log-probabilities match sdpa's (attn_verdict, attn_check)."""
    def test_the_verdict(self):
        a = [torch.tensor([-1.0, -2.0, -0.5]), torch.tensor([-3.0])]
        self.assertEqual(T.attn_verdict(a, [x.clone() for x in a])[:2], (True, 0.0))
        small = [x + 0.01 for x in a]; ok, mean, mx, n = T.attn_verdict(a, small, tol=0.02)
        self.assertTrue(ok); self.assertAlmostEqual(mean, 0.01, places=6); self.assertEqual(n, 4)
        self.assertFalse(T.attn_verdict(a, [x + 0.5 for x in a], tol=0.02)[0])          # e.g. attention over part of the sequence only
        self.assertFalse(T.attn_verdict(a, [torch.tensor([float("nan"), -2.0, -0.5]), a[1]])[0])
        self.assertFalse(T.attn_verdict([], [])[0])

    def check_run(self, diff):
        model = mock.Mock(spec=["set_attn_implementation"]); calls = []
        def fake_score(model_, tok, pairs, chunk, per_token=False, **kw):
            calls.append((len(pairs), model_.set_attn_implementation.call_args_list[-1][0][0] if model_.set_attn_implementation.call_args_list else "flash"))
            return [torch.tensor([-1.0, -2.0]) + (diff if model_.set_attn_implementation.call_args_list else 0.0) for _ in pairs]
        pairs = [("p", "c", None, 3, "blocks")] * 6
        with mock.patch.dict(T.ATTN, {"name": "flash_attention_3", "checked": False, "busy": False}), mock.patch.dict(T.MSTEP, {}, clear=True), \
             mock.patch.object(T, "score", side_effect=fake_score), contextlib.redirect_stdout(io.StringIO()):
            T.attn_check(model, None, pairs, 2)
            return dict(T.ATTN), dict(T.MSTEP), calls, [c[0][0] for c in model.set_attn_implementation.call_args_list]

    def test_a_match_keeps_the_flash_attention(self):
        attn, mstep, calls, sets = self.check_run(0.001)
        self.assertEqual(attn["name"], "flash_attention_3"); self.assertTrue(attn["checked"]); self.assertEqual(mstep["attn_flash"], 1.0)
        self.assertEqual([c[0] for c in calls], [4, 4])                           # the first ATTN_CHECK_N pairs, twice
        self.assertEqual(sets, ["sdpa", "flash_attention_3"])                    # sdpa for the second score, then back

    def test_a_mismatch_falls_back_to_sdpa(self):
        attn, mstep, calls, sets = self.check_run(0.5)
        self.assertEqual(attn["name"], "sdpa"); self.assertEqual(mstep["attn_flash"], 0.0); self.assertEqual(sets[-1], "sdpa")

    def test_the_pl_default_is_sdpa(self):
        with open(T.__file__, encoding="utf-8") as f: src = f.read()
        self.assertIn('ap.add_argument("--attn", choices=("sdpa", "flash_attention_3", "flash_attention_2", "auto"), default="sdpa",', src)


class CommandLineTests(unittest.TestCase):
    """The trainer's own command line (its parser is built under __main__; --help itself fails on an older help string's '%')."""
    def run_trainer(self, *argv):
        return subprocess.run([sys.executable, "-m", "plearn.method.trainer", "--model", "none", "--domain", "plearn.tasks.mixture", *argv],
                              cwd=ROOT, capture_output=True, text=True)

    def test_the_flags_take_their_values(self):
        for argv, msg in ((["--update", "bogus"], "argument --update: invalid choice: 'bogus' (choose from em, grpo)"),
                          (["--grpo_eps", "x"], "argument --grpo_eps: invalid float value: 'x'"),
                          (["--grpo_norm", "2"], "argument --grpo_norm: invalid choice: '2' (choose from 0, 1)"),
                          (["--grpo_baseline", "teacher"], "unrecognized arguments: --grpo_baseline teacher")):
            out = self.run_trainer(*argv)
            self.assertEqual(out.returncode, 2, out.stderr[-1000:]); self.assertIn(msg, out.stderr)

    def test_the_rule_of_the_update(self):
        rule = "--update grpo: the critic objective with --m_step blocks, --credit own, no --neg_answer / --neg_thought"
        e_rule = "--update grpo: no --temper and no --think_cost"
        base = ["--update", "grpo", "--objective", "critic", "--m_step", "blocks"]
        for argv, msg in ((["--update", "grpo", "--objective", "strings"], rule),
                          (["--update", "grpo", "--objective", "critic", "--m_step", "joint"], rule),
                          (base + ["--accept_only", "1", "--neg_answer", "1"], rule),
                          (base + ["--grpo_eps", "1.0"], rule),
                          (base + ["--grpo_clip", "-1"], rule),
                          (base + ["--temper", "0.1"], e_rule),
                          (base + ["--think_cost", "0,2"], e_rule)):
            out = self.run_trainer(*argv)
            self.assertNotEqual(out.returncode, 0); self.assertIn("AssertionError: " + msg, out.stderr, out.stderr[-600:])


if __name__ == "__main__":
    unittest.main()
