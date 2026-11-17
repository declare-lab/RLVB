"""--score_presence 1: the learner scores the own turns under the sampler's presence penalty too.

The RLVB (EM) sampler sends PL_PRESENCE_PENALTY (1.5 in the runs) in every request, and vLLM then lowers the logit of each token that the
same request has generated before (logits -= penalty * output_mask, before the temperature, top-k and top-p). The learner's score
had no penalty, so the own samples did not come from the scored policy: the likelihood of own samples then has a gradient even
with equal weights. These tests check the score against vLLM's rule, the requests of a turn, the score() window, the property
itself (the score function of the matched model has mean zero, that of the model with no penalty does not), and the run file."""
import math
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

import plearn.method.trainer as T
from plearn.method.trainer import CLOSE, TERM, presence_requests, presence_shift, presence_slice, turn_item

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def chars(text):
    """A stand-in tokenization: one token a character (offsets and ids)."""
    return [(i, i + 1) for i in range(len(text))], [ord(c) for c in text]


def vllm_rule(logits, out_ids, penalty):
    """vLLM's presence penalty for one request, a loop: at step t the logit of every token in out_ids[:t] is lowered once."""
    l = logits.clone().float()
    for t in range(len(out_ids)):
        for v in set(out_ids[:t]): l[t, v] -= penalty
    return l


class RequestTests(unittest.TestCase):
    def test_a_turn_is_three_requests_and_the_forced_text_is_in_none(self):
        z, b, a = "aba\n", "LINE", "ACT"
        cont, _, d0 = turn_item(z, b, a, True, own_term=True)          # z CLOSE b TERM a
        offs, ids = chars(cont); rq = presence_requests(cont, d0, offs, ids)
        self.assertEqual(len(rq), 3)
        end = cont.index("</think>") + len("</think>")
        self.assertEqual((rq[0][0], rq[0][1]), (0, end))                # the thought through its </think>
        self.assertEqual(cont[rq[1][0]:rq[1][1]], b); self.assertEqual(cont[rq[2][0]:rq[2][1]], a)
        covered = set(range(rq[0][0], rq[0][1])) | set(range(rq[1][0], rq[1][1])) | set(range(rq[2][0], rq[2][1]))
        self.assertEqual("".join(cont[j] for j in range(len(cont)) if j not in covered), "\n\n" + TERM)   # after </think>, between the lines

    def test_first_occurrences_are_per_request(self):
        cont, _, d0 = turn_item("aba\n", "aLa", "a", True, own_term=True)
        offs, ids = chars(cont); rq = presence_requests(cont, d0, offs, ids)
        f0 = dict(zip(rq[0][2].tolist(), rq[0][3].tolist())); self.assertEqual(f0[ord("a")], 0); self.assertEqual(f0[ord("b")], 1)
        f1 = dict(zip(rq[1][2].tolist(), rq[1][3].tolist())); self.assertEqual(f1[ord("a")], rq[1][0])   # the line's own first "a", not the thought's
        self.assertEqual(rq[2][3].tolist(), [rq[2][0]])

    def test_a_thought_alone_and_a_decision_alone(self):
        offs, ids = chars("abc"); rq = presence_requests("abc", None, offs, ids)
        self.assertEqual([(r[0], r[1]) for r in rq], [(0, 3)])
        text = "LINE" + TERM + "ACT" + TERM; offs, ids = chars(text); rq = presence_requests(text, 0, offs, ids)
        self.assertEqual([text[r[0]:r[1]] for r in rq], ["LINE", "ACT" + TERM])


class ShiftTests(unittest.TestCase):
    def test_the_shift_is_vllms_rule_request_by_request(self):
        torch.manual_seed(0); V = 140
        cont, _, d0 = turn_item("abab ba\n", "a b a", "ab", True, own_term=True); offs, ids = chars(cont); n = len(ids)
        rq = presence_requests(cont, d0, offs, ids); logits = torch.randn(1, n, V)
        want = logits[0].clone()
        for j0, j1, _, _ in rq: want[j0:j1] = vllm_rule(logits[0, j0:j1], ids[j0:j1], 1.5)
        got = presence_shift(logits, presence_slice([rq], [0], 0, n, torch.device("cpu")), 1.5)
        self.assertTrue(torch.allclose(got[0], want)); self.assertFalse(torch.allclose(got[0], logits[0]))
        self.assertTrue(torch.equal(logits[0], logits[0].clone()))        # the input is not changed in place

    def test_slices_and_the_right_aligned_window_agree_with_one_pass(self):
        torch.manual_seed(1); V = 140
        cont, _, d0 = turn_item("abcabcabc\n", "a a", "b", True, own_term=True); offs, ids = chars(cont); n = len(ids)
        rq = presence_requests(cont, d0, offs, ids); sh = 5; logits = torch.randn(1, n + sh, V)   # the row starts at window position 5
        one = presence_shift(logits, presence_slice([rq], [sh], 0, n + sh, torch.device("cpu")), 1.5)
        parts = [presence_shift(logits[:, a:a + 7], presence_slice([rq], [sh], a, min(a + 7, n + sh), torch.device("cpu")), 1.5) for a in range(0, n + sh, 7)]
        self.assertTrue(torch.allclose(torch.cat(parts, 1), one)); self.assertTrue(torch.equal(one[0, :sh], logits[0, :sh]))

    def test_token_logp_is_the_penalised_samplers_probability(self):
        torch.manual_seed(2); V = 60; ids = [3, 7, 3, 3, 9, 7, 11, 3]; n = len(ids); logits = torch.randn(1, n, V) * 2
        rq = [(0, n, torch.tensor(sorted(set(ids))), torch.tensor([ids.index(v) for v in sorted(set(ids))]))]
        pres = presence_slice([rq], [0], 0, n, torch.device("cpu")); tgt = torch.tensor([ids])
        with mock.patch.dict(T.SAMPLER, {"T": 0.8, "top_p": 0.9, "top_k": 10, "presence": 1.5}):
            got = T.token_logp(logits, tgt, None, pres)
            l = vllm_rule(logits[0], ids, 1.5) / 0.8                    # penalty, temperature, top-k, top-p (vLLM's order)
            keep = l >= l.topk(10, -1).values[..., -1:]
            pr = torch.softmax(l.masked_fill(~keep, float("-inf")), -1); ps, idx = pr.sort(-1, descending=True)
            keep &= ~torch.zeros_like(keep).scatter(-1, idx, (ps.cumsum(-1) - ps) >= 0.9); keep.scatter_(-1, tgt[0][:, None], True)
            want = torch.log_softmax(l.masked_fill(~keep, float("-inf")), -1).gather(-1, tgt[0][:, None]).squeeze(-1)
            self.assertTrue(torch.allclose(got[0], want, atol=1e-5))
            self.assertFalse(torch.allclose(got[0], T.token_logp(logits, tgt), atol=1e-3))   # no requests given: no penalty
        with mock.patch.dict(T.SAMPLER, {"T": 0.8, "top_p": 0.9, "top_k": 10, "presence": 0.0}):   # penalty 0: the old score, bit for bit
            self.assertTrue(torch.equal(T.token_logp(logits, tgt, None, pres), T.token_logp(logits, tgt)))


class ZeroMeanTests(unittest.TestCase):
    """Sequences from a sampler with the penalty (a one-layer model: logits theta at every step). With equal weights the gradient of
    the log-likelihood of the samples is zero in expectation under the score that has the penalty, and is not zero under the score
    that has none: it points to the rare tokens (the penalty moves mass to the tokens not yet used)."""
    def test_the_matched_score_has_no_drift_and_the_plain_score_has(self):
        g = torch.Generator().manual_seed(0); V, L, N, pen = 12, 30, 3000, 1.5
        theta = torch.linspace(2.0, -2.0, V)
        seqs = torch.zeros(N, L, dtype=torch.long); seen = torch.zeros(N, V)
        for t in range(L):
            x = torch.multinomial(torch.softmax(theta[None] - pen * seen, -1), 1, generator=g).squeeze(1)
            seqs[:, t] = x; seen[torch.arange(N), x] = 1.0
        th = theta.clone().requires_grad_(True); per = []
        for mode in ("matched", "plain"):
            lp = []
            for i in range(N):
                ids = seqs[i].tolist(); logits = th[None, None, :].expand(1, L, V)
                if mode == "matched":
                    cols = torch.tensor(sorted(set(ids))); first = torch.tensor([ids.index(v) for v in cols.tolist()])
                    logits = presence_shift(logits, presence_slice([[(0, L, cols, first)]], [0], 0, L, torch.device("cpu")), pen)
                lp.append(torch.log_softmax(logits[0], -1).gather(-1, seqs[i][:, None]).sum())
            grads = torch.stack([torch.autograd.grad(x, th, retain_graph=True)[0] for x in lp])
            per.append((grads.mean(0), grads.std(0) / math.sqrt(N)))
        (m_mean, m_se), (p_mean, p_se) = per
        self.assertLess(float((m_mean / m_se).abs().max()), 4.5)        # matched: every coordinate within its noise
        self.assertGreater(float((p_mean / p_se).abs().max()), 20.0)    # plain: a clear common direction
        self.assertGreater(float(p_mean[-1]), 0.0); self.assertLess(float(p_mean[0]), 0.0)   # toward the rare token, away from the common one


class ScoreWindowTests(unittest.TestCase):
    """score() with the penalty on stand-in logits (all zero) and a one-character tokenizer: a counted token that its request has
    generated before has log-probability -penalty - log(V - s + s exp(-penalty)), s = the request's distinct tokens so far."""
    def tok(self):
        try:
            from tokenizers import Regex, Tokenizer, models, pre_tokenizers
            from transformers import PreTrainedTokenizerFast
        except ImportError: self.skipTest("no tokenizers library")
        alphabet = sorted(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ </>:\n."))
        t = Tokenizer(models.WordLevel({c: i for i, c in enumerate(alphabet + ["<pad>", "<unk>"])}, unk_token="<unk>"))
        t.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
        return PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", unk_token="<unk>")

    def run_score(self, pairs, chunk, presence):
        tok = self.tok(); V = len(tok)
        class M:
            device = torch.device("cpu")
            def __call__(self, input_ids, attention_mask, use_cache, logits_to_keep, **kw):
                return types.SimpleNamespace(logits=torch.zeros(input_ids.shape[0], logits_to_keep, V))
        with mock.patch.dict(T.SAMPLER, {"T": 1.0, "top_p": 1.0, "top_k": -1, "presence": presence}), \
                mock.patch.dict(T.DECISION, {"T": 1.0, "top_p": 1.0, "top_k": -1, "presence": presence}), \
                mock.patch.dict(os.environ, {"PL_CHUNKED_HEAD": "0"}), mock.patch.object(T, "PROC", None):
            return T.score(M(), tok, pairs, chunk, per_token=True), V

    def want(self, text, d0, V, pen, span=None):
        offs, ids = chars(text); out = []
        req_of = {}
        for r, (j0, j1, _, _) in enumerate(presence_requests(text, d0, offs, ids)):
            for j in range(j0, j1): req_of[j] = (j0, r)
        for j in range(len(text)):
            if span and any(a < e and b > s for s, e in span for a, b in [offs[j]]): continue
            if j not in req_of: out.append(-math.log(V)); continue
            j0, _ = req_of[j]; before = set(text[j0:j]); s = len(before)
            out.append((-pen if text[j] in before else 0.0) - math.log(V - s + s * math.exp(-pen)))
        return out

    def test_two_rows_of_different_length_and_a_masked_span(self):
        a = turn_item("abab aba\n", "Possible: a a", "ACT a", True, own_term=True)
        b = turn_item("ccc\n", "cc", "c", False, own_term=True)           # a cut thought: its CLOSE is not counted
        pairs = [("x y z ", a[0], a[1], a[2], None), ("x ", b[0], b[1], b[2], None)]
        (got, V) = self.run_score(pairs, 2, 1.5)
        for (cont, span, d0), g in zip((a, b), got):
            w = self.want(cont, d0, V, 1.5, span)
            self.assertEqual(len(g), len(w)); self.assertTrue(torch.allclose(g, torch.tensor(w), atol=1e-5), (g, w))
        (plain, V) = self.run_score(pairs, 2, 0.0)                        # penalty 0: every counted token is -log V
        for g in plain: self.assertTrue(torch.allclose(g, torch.full_like(g, -math.log(V)), atol=1e-6))


class RunFileTests(unittest.TestCase):
    def test_only_the_score_changes(self):
        from plearn.train import commands, load
        sys.path.insert(0, os.path.join(ROOT, "tests", "method"))
        from test_run_files import flags
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "Qwen3.5-4B"))
            env = {k: v for k, v in os.environ.items() if k not in ("PL_GPUS", "PL_SEQS", "PL_PRESENCE_PENALTY", "QWEN35_4B_PATH")}
            with mock.patch.dict(os.environ, {**env, "PL_MODELS": d, "PL_RUNS": d}, clear=True):
                base = commands(load(os.path.join(ROOT, "configs", "train", "suite_qwen35_4b.yaml")), 0, stamp="0000-0000")
                c = commands(load(os.path.join(ROOT, "configs", "train", "suite_qwen35_4b_presence.yaml")), 0, stamp="0000-0000")
                for role in ("actor", "learner"):
                    with self.subTest(role=role):
                        a, b = flags(c[role]), flags(base[role])
                        self.assertEqual(a.pop("--score_presence"), "1"); self.assertNotIn("--score_presence", b)   # the base file: the trainer's default 0
                        for k in ("--tag", "--out", "--queue", "--dump_em", "--dump_rollouts", "--wandb_name"): a.pop(k, None); b.pop(k, None)
                        self.assertEqual(a, b)
                self.assertEqual(c["env"], base["env"]); self.assertEqual(c["env"]["PL_PRESENCE_PENALTY"], "1.5")   # the sampler is the same

    def test_the_8k_file_changes_only_the_training_cap(self):
        from plearn.train import commands, load
        sys.path.insert(0, os.path.join(ROOT, "tests", "method"))
        from test_run_files import flags
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "Qwen3.5-4B"))
            env = {k: v for k, v in os.environ.items() if k not in ("PL_GPUS", "PL_SEQS", "PL_PRESENCE_PENALTY", "QWEN35_4B_PATH")}
            with mock.patch.dict(os.environ, {**env, "PL_MODELS": d, "PL_RUNS": d}, clear=True):
                base = commands(load(os.path.join(ROOT, "configs", "train", "suite_qwen35_4b_presence.yaml")), 0, stamp="0000-0000")
                c = commands(load(os.path.join(ROOT, "configs", "train", "suite_qwen35_4b_presence_8k.yaml")), 0, stamp="0000-0000")
                for role in ("actor", "learner"):
                    with self.subTest(role=role):
                        a, b = flags(c[role]), flags(base[role])
                        self.assertEqual(a.pop("--max_think"), "8192"); self.assertEqual(b.pop("--max_think"), "16384")
                        self.assertEqual(a["--eval_max_think"], "8192"); self.assertEqual(a["--score_presence"], "1")   # training cap = the tests' cap
                        self.assertIs(a.pop("--line_wide"), True); self.assertNotIn("--line_wide", b)   # line_wide on in the 8k file only
                        self.assertEqual(a.pop("--line_target"), "0.1"); self.assertNotIn("--line_target", b)   # line_target on in the 8k file only
                        for k in ("--tag", "--out", "--queue", "--dump_em", "--dump_rollouts", "--wandb_name"): a.pop(k, None); b.pop(k, None)
                        self.assertEqual(a, b)
                env_c = dict(c["env"]); env_b = dict(base["env"])
                self.assertEqual(int(env_b.pop("PL_MAX_MODEL_LEN")) - int(env_c.pop("PL_MAX_MODEL_LEN")), 16384 - 8192)   # the context follows the cap
                self.assertEqual(env_c, env_b); self.assertEqual(env_c["PL_PRESENCE_PENALTY"], "1.5")
            with mock.patch.dict(os.environ, {**env, "PL_MODELS": d, "PL_RUNS": d, "PL_GPUS": "0,1,2,3,4,5,6,7"}, clear=True):   # 8 GPUs: 8 ranks
                c8 = commands(load(os.path.join(ROOT, "configs", "train", "suite_qwen35_4b_presence_8k.yaml")), 0, stamp="0000-0000")
                self.assertEqual(c8["n_learner"], 8); self.assertEqual(c8["actor_gpus"], "0,1,2,3,4,5,6,7"); self.assertEqual(c8["learner_gpus"], c8["actor_gpus"])
                self.assertEqual(c8["actor"][c8["actor"].index("--vllm_gpu") + 1], "0,1,2,3,4,5,6,7")

    def test_the_trainers_default_is_the_old_score(self):
        src = open(os.path.join(ROOT, "plearn", "method", "trainer.py")).read()
        self.assertIn('ap.add_argument("--score_presence", type=int, default=0,', src)
        self.assertEqual(T.SAMPLER["presence"], 0.0); self.assertEqual(T.DECISION["presence"], 0.0)

    def test_a_saved_first_set_does_not_depend_on_the_score(self):
        """PL_SET_FROM: the first sample set comes from the base model, so a run with another --score_presence may take it."""
        src = open(os.path.join(ROOT, "plearn", "method", "trainer.py")).read()
        a = src.index('free = {"role", "tag", "out"'); self.assertIn('"score_presence"', src[a:src.index("}", a)])


if __name__ == "__main__":
    unittest.main()
