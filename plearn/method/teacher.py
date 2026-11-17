"""The teacher as the environment: the generic exact filter, written derivation and anchor over the teacher contract.

A classical RL environment returns an observation and a scalar reward; a posterior environment returns the observation and
the teacher: the posterior over what the agent must find out, its derivation, the predictive distribution of every look,
and (through plearn.method.estimator) the lookahead values and the target policy. A new domain implements the contract below
and inherits everything else; plearn/tasks/proworlds/teacher.py implements the same interface by hand, on arrays.

The contract (class World):
    hypotheses   list of hashable theta (what the agent must find out), uniform prior unless `prior` is overridden
    answer(theta)            the answer the hypothesis implies (a string in `answers`)
    looks, obs(theta, look)  the information-gathering actions and their observation model {observation: probability},
                             from the generative model, from counterfactual rendering in a resettable simulator, or from
                             a calibrated predictor
    variables(theta)         the named hidden variables of a hypothesis, for the anchor
    rules(), obs_text(look, o), look_name(look)   the rendering as text
    optional hooks for the written derivation: predict (value a hypothesis predicts, one lookup), row (the row it is read
    from), explain (decoding of an observation), needs (the value to compare against), group (nuisance variable that
    groups hypotheses, e.g. a transformation), noise_text, transform_text
"""
from __future__ import annotations
import math, random
import numpy as np


class World:
    """One instance of a domain; subclasses fill the contract."""
    family = "world"; T = 8; answers = []
    def __init__(self, rng, **kw): raise NotImplementedError
    def answer(self, theta): raise NotImplementedError
    def obs(self, theta, look): raise NotImplementedError
    def variables(self, theta): raise NotImplementedError
    def rules(self): raise NotImplementedError
    def look_name(self, look): return str(look)
    def obs_text(self, look, o): return f"{self.look_name(look)}: {o}"
    def prior(self): return np.zeros(len(self.hypotheses))                          # log prior, uniform by default
    # hooks of the written derivation (one lookup per token)
    def predict(self, theta, look):
        d = self.obs(theta, look); return max(d, key=d.get)
    def row(self, theta, look): return None
    def explain(self, look, o): return []
    def needs(self, look, o, group=None): return o
    def group(self, theta): return None
    def group_name(self, g): return str(g)
    def group_header(self, look, g): return None
    def noise_text(self): return "Looks are exact."
    def transform_text(self): return "Transformations: none."


class Belief:
    """Log-weights over the world's hypotheses after the observations so far (hashable; the lookahead caches on it)."""
    __slots__ = ("W", "t", "lw", "_k")
    def __init__(self, W, t, lw):
        self.W, self.t, self.lw = W, t, lw; self._k = (t, tuple(np.round(lw, 6)))
    def __hash__(self): return hash(self._k)
    def __eq__(self, o): return self._k == o._k
    def probs(self):
        m = self.lw.max(); p = np.exp(self.lw - m); return p / p.sum()
    def support(self): return [h for h, p in zip(self.W.hypotheses, self.probs()) if p > 1e-12]
    def answer_counts(self):
        c = {}
        for h, p in zip(self.W.hypotheses, self.probs()):
            if p > 1e-12: y = self.W.answer(h); c[y] = c.get(y, 0.0) + p
        return c
    def U(self): return max(self.answer_counts().values())
    def identified(self): return self.U() > 1 - 1e-9
    def cls(self): return (self.W.family, min(sum(1 for v in self.answer_counts().values() if v > 0.01), 4))
    def looks(self): return [a for a in self.W.looks if len(self.outcome_dist(a)) > 1]
    def outcome_dist(self, a):
        d = {}
        for h, p in zip(self.W.hypotheses, self.probs()):
            if p <= 1e-12: continue
            for o, q in self.W.obs(h, a).items(): d[o] = d.get(o, 0.0) + p * q
        return {o: v for o, v in d.items() if v > 1e-12}
    def update(self, a, o):
        lw = self.lw.copy()
        for i, h in enumerate(self.W.hypotheses):
            if lw[i] == -np.inf: continue
            q = self.W.obs(h, a).get(o, 0.0); lw[i] = lw[i] + (math.log(q) if q > 0 else -np.inf)
        return Belief(self.W, self.t + 1, lw)
    def outcomes(self, a): return [(p, self.update(a, o)) for o, p in self.outcome_dist(a).items()]
    def marginals(self):
        m = {}
        for h, p in zip(self.W.hypotheses, self.probs()):
            if p <= 1e-12: continue
            for var, val in self.W.variables(h).items(): m.setdefault(var, {}); m[var][val] = m[var].get(val, 0.0) + p
        return m


def anchor_text(B, cap=8):
    """The belief in words: a uniform support of two variables as the support grouped by the first; otherwise each hidden
    variable's marginal, with probabilities when they are not all equal."""
    p = B.probs(); sup = B.support(); names = list(B.W.variables(sup[0]))
    if len(names) == 2 and B.W.group(sup[0]) is not None and p[p > 1e-12].max() - p[p > 1e-12].min() < 0.005:
        if len(sup) == len(B.W.hypotheses): return f"Possible: any {names[0]}, any {names[1]}"
        by = {}
        for h in sup:
            v = B.W.variables(h); by.setdefault(v[names[0]], []).append(v[names[1]])
        return "Possible: " + "; ".join(f"{names[0]} {k} ({', '.join(sorted(vs))})" for k, vs in sorted(by.items(), key=lambda kv: (len(kv[0]), kv[0])))
    parts = []
    for var, dist in B.marginals().items():
        items = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
        if len(items) == 1: parts.append(f"{var} = {items[0][0]}"); continue
        uniform = max(dist.values()) - min(dist.values()) < 0.005
        shown = items[:cap]; more = f" and {len(items) - cap} more" if len(items) > cap else ""
        if uniform: parts.append(f"{var} in {{{', '.join(v for v, _ in shown)}{more}}}")
        else: parts.append(f"{var}: " + ", ".join(f"{v} {q:.2f}" for v, q in shown) + more)
    return "Possible: " + "; ".join(parts)


class Ep:
    def __init__(self, W, theta, spec):
        self.W, self.theta, self.spec = W, theta, spec
        self.t, self.done, self.correct = 0, False, None; self.log = []
        self.B = Belief(W, 0, W.prior().astype(float))


class Domain:
    """The adapter the trainer speaks to. Subclasses provide `make(spec) -> (World, theta)`, `sample(rng) -> spec` and
    `eval_specs(n)`; a spec is (family, seed, split)."""
    name = "teacher"; lam = 0.3; T = 8
    value_modes = ("accuracy", "log")                                # the generic lookahead computes both (Reliability.util)
    def make(self, spec): raise NotImplementedError
    def sample(self, rng): raise NotImplementedError
    def eval_specs(self, n): raise NotImplementedError
    def start(self, specs):
        out = []
        for s in specs:
            W, theta = self.make(s); out.append(Ep(W, theta, s))
        return out
    def close(self, eps): pass
    def _history(self, e):
        lines = [f"Task: {e.W.rules()}", "Observations so far:" if e.log else "Observations so far: none yet"]
        lines += [f"[{k}] {e.W.obs_text(a, o)}" for k, (a, o) in enumerate(e.log)]
        lines.append(f"Looks used: {e.t} of {e.W.T}. Decide now.")
        return "\n".join(lines)
    def look_strings(self, e): return {a: f"ACTION: LOOK {e.W.look_name(a)}" for a in e.W.looks} if e.t < e.W.T else {}
    def answer_options(self, e): return list(e.W.answers)
    def commit_string(self, y): return f"ACTION: SUBMIT {y}"
    def do_look(self, e, a):
        d = e.W.obs(e.theta, a); o = random.Random(hash((e.spec, e.t, a)) & 0xFFFFFFFF).choices(list(d), weights=list(d.values()))[0]
        e.log.append((a, o)); e.B = e.B.update(a, o)
    def do_commit(self, e, y): return y == e.W.answer(e.theta)
    def belief(self, e): return e.B
    def label(self, e): return e.W.answer(e.theta)
    def answer_posterior(self, e):
        c = e.B.answer_counts(); return {y: c.get(y, 0.0) for y in e.W.answers}
    def belief_note(self, e): return anchor_text(e.B)
    def depth(self, B): return B.W.depth(B) if hasattr(B.W, "depth") else 2
    @staticmethod
    def anchor_equal(a, b): return " ".join(a.lower().split()) == " ".join(b.lower().split())
    def anchor_support(self, text, answers, e=None):
        """The answers a written anchor (in this renderer's grammar) still allows."""
        body = text.split(":", 1)[1] if ":" in text else text; out = set()
        if body.strip().lower().startswith("any "): return set(answers)
        if e is not None:
            fixed = {}
            for clause in body.split(";"):
                if " = " in clause: k, v = clause.split(" = ", 1); fixed[k.strip()] = v.strip()
            hits = [h for h in e.W.hypotheses if all(e.W.variables(h).get(k) == v for k, v in fixed.items())] if fixed else []
            if hits and len(fixed) == len(e.W.variables(hits[0])): return {e.W.answer(h) for h in hits}
        for clause in body.split(";"):
            clause = clause.strip()
            if " = " in clause:
                v = clause.split(" = ", 1)[1].strip()
                if v in answers: out.add(v)
            elif " in {" in clause:
                out.update(x.strip() for x in clause.split("{", 1)[1].rstrip("}").split(",") if x.strip() in answers)
            elif "(" in clause:
                v = clause.partition("(")[0].strip().split()[-1] if clause.partition("(")[0].strip() else ""
                if v in answers: out.add(v)
            else:
                items = clause.split(":", 1)[1] if ":" in clause else clause
                for it in items.split(","):
                    parts = it.strip().rsplit(" ", 1)
                    if len(parts) == 2 and parts[0].strip() in answers: out.add(parts[0].strip())
        return out if out else set(answers)
    def anchor_answer(self, text, answers, e=None):
        """The answer the written anchor implies: the hypothesis it pins down, else the answer variable's most probable value."""
        body = text.split(":", 1)[1] if ":" in text else text
        if body.strip().lower().startswith("any "): return answers[0]
        if e is not None:
            fixed = {}
            for clause in body.split(";"):
                if " = " in clause: k, v = clause.split(" = ", 1); fixed[k.strip()] = v.strip()
            hits = [h for h in e.W.hypotheses if e.W.variables(h) == fixed] if fixed else []
            if len(hits) == 1: return e.W.answer(hits[0])
        best, bestp = None, -1.0
        for clause in body.split(";"):
            clause = clause.strip()
            if " = " in clause:
                v = clause.split(" = ", 1)[1].strip()
                if v in answers and 1.0 > bestp: best, bestp = v, 1.0
            elif " in {" in clause:
                vals = [x.strip() for x in clause.split("{", 1)[1].rstrip("}").split(",")]; vals = [x for x in vals if x in answers]
                if vals and 0.5 > bestp: best, bestp = min(vals, key=answers.index), 0.5
            elif "(" in clause:
                head, _, rest = clause.partition("("); v = head.strip().split()[-1] if head.strip() else ""
                n = len([x for x in rest.rstrip(")").split(",") if x.strip()])
                if v in answers and n / 100.0 + 0.6 > bestp: best, bestp = v, n / 100.0 + 0.6
            else:
                items = clause.split(":", 1)[1] if ":" in clause else clause
                for it in items.split(","):
                    parts = it.strip().rsplit(" ", 1)
                    if len(parts) == 2 and parts[0].strip() in answers:
                        try: pr = float(parts[1])
                        except ValueError: continue
                        if pr > bestp: best, bestp = parts[0].strip(), pr
        return best if best is not None else answers[0]
    look_name_takes_episode = True
    def look_name(self, a, e): return e.W.look_name(a)
    def state_text(self, e):
        """The derivation of the posterior in the one-lookup-per-token form: for each observation, every hypothesis still standing
        is written with the value it predicts (one lookup) and the survivors are then copied; grouped by the nuisance variable
        when the world has one; the anchor closes the thought."""
        W = e.W; B = Belief(W, 0, W.prior().astype(float)); lines = [W.noise_text(), W.transform_text()]
        for a, o in e.log:
            alive = B.support(); ex = W.explain(a, o); why = f" ({'; '.join(ex)})" if ex else ""
            B = B.update(a, o); after = set(B.support())
            groups = {}
            for h in alive: groups.setdefault(W.group(h), []).append(h)
            noisy = max(W.obs(alive[0], a).values()) < 1.0 - 1e-9
            parts = []
            for key, hs in groups.items():
                need = W.needs(a, o, key)
                if key is not None and need is None: parts.append(f"under {W.group_name(key)}: no match -> none"); continue
                name = (lambda h: list(W.variables(h).values())[0]) if key is not None else (lambda h: "/".join(str(v) for v in W.variables(h).values()))
                preds = " ".join(f"{name(h)} {W.row(h, a)}:{W.predict(h, a)}" if W.row(h, a) is not None else f"{name(h)} {W.predict(h, a)}" for h in hs)
                kept = [name(h) for h in hs if (W.predict(h, a) == o if noisy else h in after)]
                hd = W.group_header(a, key) if key is not None else None
                head = (f"under {W.group_name(key)} ({hd})" if hd else f"under {W.group_name(key)}") if key is not None else ""
                parts.append(f"{head} needs {need}: ".lstrip() + preds + (" favours " if noisy else " -> ") + (", ".join(kept) if kept else "none"))
            lines.append(f"{W.obs_text(a, o)}{why}\n" + "\n".join(parts))
        lines.append(anchor_text(e.B)); return "\n".join(lines)
    def family_of(self, e): return e.spec[0]
    def extra_metrics(self, eps):
        by = {}
        for e in eps: by.setdefault(e.spec[0], []).append(bool(e.correct))
        return {f"acc_{k}": sum(v) / len(v) for k, v in by.items()}


# --------------------------------------------------------------------- the smallest instance of the contract, as documentation
class Drawers(World):
    """An item is in one of n drawers; opening a drawer shows it or not, and a sensor may misreport with probability eps."""
    family = "drawers"
    def __init__(self, rng, n=6, eps=0.0, T=6):
        self.n, self.eps, self.T = n, eps, T
        self.hypotheses = list(range(n)); self.looks = list(range(n)); self.answers = [str(k + 1) for k in range(n)]
    def answer(self, theta): return str(theta + 1)
    def obs(self, theta, look):
        truth = "item" if theta == look else "empty"
        return {truth: 1.0} if self.eps == 0 else {truth: 1 - self.eps, ("empty" if truth == "item" else "item"): self.eps}
    def variables(self, theta): return {"drawer": str(theta + 1)}
    def look_name(self, look): return f"drawer {look + 1}"
    def rules(self):
        noise = "" if self.eps == 0 else f" The sensor reports wrongly with probability {self.eps:.2f}."
        return f"One item is hidden in one of {self.n} drawers. LOOK drawer k opens drawer k.{noise} Answer with the drawer number."
    def noise_text(self): return "Looks are exact." if self.eps == 0 else f"Looks are right with probability {1 - self.eps:.2f}."


class DrawersDomain(Domain):
    name = "drawers"
    def __init__(self, n=6, eps=0.0, lam=0.3): self.n, self.eps, self.lam = n, eps, lam; self.T = 6
    def make(self, spec):
        rng = random.Random(spec[1]); W = Drawers(rng, self.n, self.eps); return W, rng.choice(W.hypotheses)
    def sample(self, rng): return ("drawers", rng.randrange(0, 10 ** 6), "train")
    def eval_specs(self, n): r = random.Random(4242); return [("drawers", 10 ** 6 + r.randrange(0, 10 ** 6), "validation") for _ in range(n)]
