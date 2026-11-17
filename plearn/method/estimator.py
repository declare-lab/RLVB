"""The lookahead estimator: exact depth-d lookahead over the belief, answer reliability per belief class, and the
two-level Boltzmann action target. Moved unchanged from the earlier research code. A domain supplies a hashable belief
    B.t, B.looks() -> [look ids], B.outcomes(look) -> [(p, B')], B.U() in [0,1], B.cls() -> tuple, B.identified()
A belief held as samples also supplies B.paired_outcomes(look) -> [B' per sample], the next belief of each of a fixed set
of samples; the root values then come from the same samples for every look, and looks whose values differ by less than
the sampling noise are ties.
"""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np
import torch


# --------------------------------------------------------------------- LoRA
class LoRALinear(torch.nn.Module):
    def __init__(self, base, r=16, alpha=32):
        super().__init__(); self.base = base
        for q in self.base.parameters(): q.requires_grad_(False)
        self.A = torch.nn.Parameter(torch.zeros(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)); self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + (torch.nn.functional.linear(torch.nn.functional.linear(x.to(self.A.dtype), self.A), self.B) * self.scale).to(x.dtype)


LORA_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj")
# all_linear (LoRA on all linear layers and the Gated DeltaNet part): also the Gated DeltaNet gate projections of the
# hybrid models (Qwen3.5 linear_attn in_proj_a, in_proj_b: [num_v_heads, hidden]), so every Linear layer of the language model has
# LoRA; lm_head, the vision tower and the MTP head stay frozen. On the 8B (no linear attention) it changes nothing.
LORA_ALL_LINEAR_EXTRA = ("in_proj_a", "in_proj_b")


def add_lora(model, names=LORA_NAMES, r=16, all_linear=False):
    """LoRA on the language model's projections: attention and MLP, plus the gated-delta-net projections of hybrid models
    (Qwen3.5). Vision towers use other module names and are left frozen. all_linear: names + LORA_ALL_LINEAR_EXTRA."""
    if all_linear: names = tuple(names) + LORA_ALL_LINEAR_EXTRA
    for q in model.parameters(): q.requires_grad_(False)
    n = 0
    for _, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if cname in names and isinstance(child, torch.nn.Linear): setattr(mod, cname, LoRALinear(child, r=r)); n += 1
    return n


# --------------------------------------------------------------------- answer reliability (Beta posterior per class)
VALUES = ("accuracy", "log")    # the teacher's value of a belief (Reliability.util); 'log' needs a domain that declares it
ACC_FLOOR = 1e-9                  # the accuracy of a wrong answer in the log value (below 1 / the answers of any domain)


class Reliability:
    """The agent's answer accuracy at a belief B. Default: a Beta estimate per belief class with the prior mean U(B).
    fidelity=True: an answer y at b is right with probability b(y), so the accuracy is U(b) * phi with the agent's fidelity
    phi = E[b(y) / U(b)] (1 when it names the most probable answer); U(b) is exact and only phi is estimated per class, from the
    exact b(y) / U(b) of each answer (prior mean 1, the start's accuracy U). The class estimate pools beliefs of very different U:
    with many answers it drops U, and the teacher then answers early where one answer leads but is not certain."""
    def __init__(self, kappa=4.0, z=1.0, fidelity=False, value="accuracy"):
        self.kappa, self.z, self.fidelity = kappa, z, fidelity; self.s, self.n, self.seen = {}, {}, set()
        assert value in VALUES, value; self.value = value
    # the teacher's value of a belief reached at time cost c (= exp(-lam t / T)) with answer accuracy r: the average over the
    # outcomes of u(c, r) = c r (value 'accuracy': the average accuracy) or log c + log r (value 'log': the average log accuracy;
    # each halving of the possible answers is worth log 2 at any time, so the values differ from the first action even when
    # the answers are many). f is the per-outcome part, with_cost adds the time cost, none is the value of a useless action.
    def f(self, r): return math.log(max(r, ACC_FLOOR)) if self.value == "log" else r
    def with_cost(self, c, v): return math.log(max(c, 1e-300)) + v if self.value == "log" else c * v
    def util(self, c, r): return self.with_cost(c, self.f(r))
    @property
    def none(self): return -1e6 if self.value == "log" else -1.0
    @property
    def margin(self): return 0.1 if self.value == "log" else 0.06        # 'clearly better' in the domains' lookaheads
    def beta(self, B):
        k = B.cls(); u = B.U(); s, n = self.s.get(k, 0.0), self.n.get(k, 0)
        if self.fidelity: return self.kappa + s, n - s
        return self.kappa * u + s, self.kappa * (1 - u) + n - s
    def mean(self, B):
        a, b = self.beta(B); m = a / (a + b)
        return B.U() * m if self.fidelity else m
    def ucb(self, B):
        a, b = self.beta(B); m = a / (a + b); v = a * b / ((a + b) ** 2 * (a + b + 1)); m = m + self.z * math.sqrt(max(v, 0.0))
        return min(1.0, B.U() * m if self.fidelity else m)
    def update(self, B, correct, key, ratio=None):
        tag = (key, B.cls())
        if key is not None and tag in self.seen: return
        x = float(ratio) if (self.fidelity and ratio is not None) else float(correct)
        self.seen.add(tag); k = B.cls(); self.s[k] = self.s.get(k, 0.0) + x; self.n[k] = self.n.get(k, 0) + 1
    def add(self, k, s, n):
        """A measurement of class k given as its sum s and its count n (n may be a fraction: --rho_rb weighs a cell by its answers)."""
        self.s[k] = self.s.get(k, 0.0) + float(s); self.n[k] = self.n.get(k, 0) + float(n)
    def table(self):
        return {str(k): [self.s[k], self.n[k]] for k in self.n}


# --------------------------------------------------------------------- exact lookahead and targets
def lookahead_Q(B, rel, lam, T, depth, acts=None):
    if hasattr(B, "lookahead"):                                    # a domain that computes the same values on arrays
        return B.lookahead(rel, lam, T, depth, acts)
    # a belief that can say what evidence it holds (GUI apps) gets no optimism at a leaf where that evidence has not grown:
    # navigation that reads nothing new must not look better than answering now
    root = B.evidence_key() if hasattr(B, "evidence_key") else None
    @lru_cache(maxsize=None)
    def V(b, d):
        c = math.exp(-lam * b.t / T)
        if b.t >= T or b.identified() or not b.looks(): return rel.util(c, rel.mean(b))
        if d == 0: return rel.util(c, rel.mean(b) if root is not None and b.evidence_key() == root else rel.ucb(b))
        best = rel.util(c, rel.mean(b))
        for a in b.looks():
            q = sum(p * V(b2, d - 1) for p, b2 in b.outcomes(a))
            best = max(best, q)
        return best
    acts = B.looks() if acts is None else acts                     # the root's legal looks (all of them, informative or not)
    if B.t >= T: return {a: rel.none for a in acts}
    if hasattr(B, "paired_outcomes"): return tie_merge({a: [V(b2, depth - 1) for b2 in B.paired_outcomes(a)] for a in acts})
    return {a: sum(p * V(b2, depth - 1) for p, b2 in B.outcomes(a)) for a in acts}


def tie_merge(per, z=2.0):
    """Root values from paired per-sample values: in order of the mean, a look joins the group of the best look before it
    when their paired difference is within z standard errors; a group's looks share its mean value (written as a tie)."""
    mean = {a: float(np.mean(v)) for a, v in per.items()}
    groups = []
    for a in sorted(per, key=lambda a: -mean[a]):
        if groups:
            d = np.asarray(per[groups[-1][0]]) - np.asarray(per[a])
            if d.mean() <= z * d.std(ddof=1) / math.sqrt(len(d)) + 1e-12: groups[-1].append(a); continue
        groups.append([a])
    return {a: float(np.mean([mean[b] for b in g])) for g in groups for a in g}


def targets(B, look_strings, answer_opts, commit_strings, post, rel, lam, T, tau, depth):
    """Two-level Boltzmann target over the legal strings; looks with equal value form one group. Also returns the looks'
    values Q, the probability of answering now pC and the answers' values qy."""
    Q = lookahead_Q(B, rel, lam, T, depth, acts=list(look_strings))
    c0 = math.exp(-lam * B.t / T); r = rel.mean(B)
    out = {}
    groups = {}
    for a, s in look_strings.items():
        groups.setdefault(round(Q[a], 3), []).append(s)
    if answer_opts:
        pbest = max(post.get(y, 0.0) for y in answer_opts) if post else 0.0
        qy = {y: rel.util(c0, r * ((post.get(y, 0.0) / pbest) if (post and pbest > 0) else 1.0 / len(answer_opts))) for y in answer_opts}
        qC = max(qy.values())
    else:
        qy, qC = {}, None
    if groups:
        qL = max(groups)
        pC = 0.0 if qC is None else 1.0 / (1.0 + math.exp(-(qC - qL) / tau))
        if B.identified() and qC is not None: pC = 1.0       # the answer is certain: every look only costs time, so the target stops
        z = sum(math.exp((q - qL) / tau) for q in groups)
        for q, mem in groups.items():
            for s in mem: out[s] = (1 - pC) * math.exp((q - qL) / tau) / z / len(mem)
    else:
        pC = 1.0
    if qy:
        z = sum(math.exp((v - qC) / tau) for v in qy.values())
        for y, v in qy.items(): out[commit_strings[y]] = out.get(commit_strings[y], 0.0) + pC * math.exp((v - qC) / tau) / z
    return out, Q, pC, qy


def note_text(post, answers):
    """Exact note: the support of the answer posterior (domain-independent)."""
    if not post: return "Possible: not determined yet"
    sup = [y for y in answers if post.get(y, 0.0) > 1e-9]
    if len(sup) == len(answers): return f"Possible: any of the {len(answers)}"
    return "Possible: " + ", ".join(sup)


def note_set(txt):
    """Canonical form of a 'Possible:' line, so that two renderings of the same support compare equal."""
    t = txt.strip().lower().replace("possible:", "").strip(" .")
    if "any option" in t or "any of the" in t: return ("any",)
    if "not determined" in t: return ("unknown",)
    if ";" in t or "(" in t:
        out = set()
        for part in t.split(";"):
            part = part.strip(" .")
            if not part: continue
            a, _, rest = part.partition("(")
            out.add((a.strip(), tuple(sorted(x.strip(" )") for x in rest.split(",") if x.strip(" )")))))
        return tuple(sorted(out))
    return tuple(sorted(x.strip(" .") for x in t.split(",") if x.strip(" .")))


def softmax_np(v):
    v = np.array(v, dtype=np.float64); v = v - v.max(); e = np.exp(v); return e / e.sum()


def kl_np(p, q):
    m = p > 0; return float((p[m] * (np.log(p[m]) - np.log(np.maximum(q[m], 1e-300)))).sum())
