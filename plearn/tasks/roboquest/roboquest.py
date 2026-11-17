"""RoboQuest at the decision level: posterior environments for Odd Parcel (four parcels) and Marked Mugs.

The target of every instance is in the generator's spec (benchmarks/roboquest/suite/v1/<task>/registry): the masses of the
parcels, the label under each vessel. Evidence actions are the task's information-gathering manipulations taken as
macro-actions -- weigh two groups of parcels on the balance, lift a vessel and read its label -- and their observation is
what the scene displays: the balance's static tilt (roboquest/balance/builder.py, tilt_for_mass_difference, clipped at the
stop) and the label colour. Splits: training worlds are drawn from the generator's prior restricted to the benchmark's
development classes (Marked Mugs on the development layouts that validation does not use); validation uses every fourth
development instance with its own target; the test uses the benchmark's eval instances (split="test"), whose structure
classes never occur in development (light odd parcels at four parcels; the opposite fixed-point class of the vessels).
"""
from __future__ import annotations
import itertools, json, os, random
import numpy as np
from plearn.method import teacher as Tc

RQ = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "benchmarks", "roboquest", "suite", "v1")
UNIT_G, DELTA = 40, {"s": 3, "b": 5}          # deviations of 120 g and 200 g, in units of 40 g
DEG_PER_UNIT = 6.744859 / 3                   # 120 g tilts the beam 6.74 degrees (the instance gate's odd_tilt_static_deg)
STOP = 6                                      # 240 g or more drives the beam to its 12 degree stop
LABEL_COLORS = ("red", "blue", "green", "yellow", "purple")
SYSTEM = ("You are a robot gathering evidence before you act. Each turn, reply with exactly one line:\n"
          "ACTION: <one of the evidence actions named in the task>  or  ACTION: SUBMIT <answer>\n"
          "SUBMIT ends the task. Every evidence action costs time, so submit as soon as the evidence determines the answer, "
          "and not before.")
ANCHOR_HINT = " Before the action, write one line starting with 'Possible:' that states what is still possible."


def registry(task): return json.load(open(os.path.join(RQ, task, "registry", "registry.json")))["instances"]
def instance(task, iid): return json.load(open(os.path.join(RQ, task, "registry", "instances", iid + ".json")))


# --------------------------------------------------------------------- Odd Parcel, four parcels (one odd parcel)
def grams(code):
    if abs(code) >= STOP: return ("+" if code > 0 else "-") + "stop"
    return f"{'+' if code > 0 else ''}{code * UNIT_G}" if code else "0"


def reading(code):
    if code == 0: return "level"
    side = "left" if code > 0 else "right"
    if abs(code) >= STOP: return f"{side} pan down to the stop"
    return f"{side} pan down {abs(code) * DEG_PER_UNIT:.1f} degrees"


class OddParcel(Tc.World):
    family = "odd_parcel"
    def __init__(self, n=4, T=4):
        self.n, self.T = n, T
        self.hypotheses = [(i, d, m) for i in range(n) for d in "HL" for m in "sb"]
        sizes = range(1, min(3, n // 2) + 1)
        self.looks = [(A, B) for k in sizes for A in itertools.combinations(range(n), k)
                      for B in itertools.combinations(range(n), k) if A < B and not set(A) & set(B)]
        self.answers = [str(i + 1) for i in range(n)]
    def code(self, theta, look):
        i, d, m = theta; A, B = look; v = DELTA[m] * (1 if d == "H" else -1)
        diff = v if i in A else -v if i in B else 0
        return max(-STOP, min(STOP, diff))
    def obs(self, theta, look): return {reading(self.code(theta, look)): 1.0}
    def predict(self, theta, look): return grams(self.code(theta, look))
    def answer(self, theta): return str(theta[0] + 1)
    def variables(self, theta): return {"odd parcel": str(theta[0] + 1)}
    @staticmethod
    def hname(theta): i, d, m = theta; return f"{i + 1}{d}{120 if m == 's' else 200}"
    def look_name(self, look): A, B = look; return f"weigh {' '.join(str(i + 1) for i in A)} vs {' '.join(str(i + 1) for i in B)}"
    def decode(self, o):
        """The observed reading as a mass difference (left minus right), from the balance's rule."""
        for c in range(-STOP, STOP + 1):
            if reading(c) == o: return grams(c)
    def rules(self):
        return (f"There are {self.n} parcels on the counter, numbered 1 to {self.n}. Most of these parcels weigh the same; fewer than "
                "half weigh differently, heavier or lighter by 120 g or 200 g. The only instrument is a two-pan balance that holds up to "
                "3 parcels per pan. It tilts toward the heavier pan by 2.2 degrees for every 40 g of difference: 40 g 2.2, 80 g 4.5, "
                "120 g 6.7, 160 g 9.0, 200 g 11.2 degrees; 240 g or more drives it to its stop at 12 degrees. Evidence action: WEIGH a b "
                "VS c d (the same number of parcels on each pan; the first group goes on the left pan). Answer with the numbers of "
                "the parcels that weigh differently, e.g. SUBMIT 3.")
    def belief_text(self, B):
        sup = B.support()
        if len(sup) == len(self.hypotheses): return "Possible: any parcel, heavier or lighter, by 120 or 200 g"
        return "Possible: " + ", ".join(self.hname(h) for h in sup)
    def trace(self, e):
        B = Tc.Belief(self, 0, self.prior().astype(float)); lines = []
        for a, o in e.log:
            alive = B.support(); need = self.decode(o); B = B.update(a, o); after = set(B.support())
            lines.append(f"{self.look_name(a)} -> {o} (= {need} g)\nneeds {need}: " + " ".join(f"{self.hname(h)} {self.predict(h, a)}" for h in alive)
                         + " -> " + (", ".join(self.hname(h) for h in alive if h in after) or "none"))
        return lines


# --------------------------------------------------------------------- Marked Mugs (a label colour under each vessel)
class MarkedMugs(Tc.World):
    family = "marked_mugs"
    def __init__(self, n, pad_colors, nearest, vessel="mug"):
        self.n, self.pads, self.nearest, self.noun, self.T = n, list(pad_colors), [int(x) for x in nearest], vessel, n
        self.hypotheses = [tuple(p) for p in itertools.permutations(self.pads)]
        hits = [sum(1 for lab, near in zip(h, self.nearest) if lab == self.pads[near]) for h in self.hypotheses]
        n_fix = sum(1 for x in hits if x > 0); n_der = len(hits) - n_fix
        # the generator draws the class (some vessel already nearest its own pad, or none) first, then a permutation within it
        self.lp = np.log(np.array([0.5 / (n_fix if x > 0 else n_der) for x in hits]))
        self.looks = list(range(n)); self.answers = [self.answer(h) for h in self.hypotheses]
    def prior(self): return self.lp.copy()
    def obs(self, theta, look): return {theta[look]: 1.0}
    def answer(self, theta): return " ".join(f"{i + 1}:{c}" for i, c in enumerate(theta))
    def variables(self, theta): return {f"{self.noun} {i + 1}": c for i, c in enumerate(theta)}
    def look_name(self, look): return f"lift {self.noun} {look + 1}"
    def rules(self):
        near = "; ".join(f"{self.noun} {i + 1} stands nearest the {self.pads[j]} pad" for i, j in enumerate(self.nearest))
        return (f"There are {self.n} {self.noun}s on the counter, numbered 1 to {self.n}, and {self.n} coloured pads: {', '.join(self.pads)}. "
                f"Each {self.noun} has a coloured label on its bottom, one of the pad colours, each colour on exactly one {self.noun}; an "
                f"upright {self.noun} hides its label. {near}. Evidence action: LIFT {self.noun.upper()} i (lift it and read its label). "
                f"Answer with the label of every {self.noun}, e.g. SUBMIT " + " ".join(f"{i + 1}:{c}" for i, c in enumerate(self.pads)) + ".")
    def known(self, log):
        return {a: o for a, o in log}
    def belief_text(self, B):
        m = B.marginals(); parts = []
        for i in range(self.n):
            vals = sorted(m[f"{self.noun} {i + 1}"], key=self.pads.index)
            parts.append(f"{self.noun} {i + 1} = {vals[0]}" if len(vals) == 1 else f"{self.noun} {i + 1} in {{{', '.join(vals)}}}")
        return "Possible: " + "; ".join(parts)
    def trace(self, e):
        lines, known = [], {}
        for a, o in e.log:
            known[a] = o
            open_v = [i for i in range(self.n) if i not in known]; open_c = [c for c in self.pads if c not in known.values()]
            line = f"{self.look_name(a)} -> {o}\n{self.noun} {a + 1} = {o}; "
            if len(open_v) == 1:
                line += f"the only colour left is {open_c[0]}, so {self.noun} {open_v[0] + 1} = {open_c[0]}"
            elif open_v:
                line += f"still open: {self.noun}s {', '.join(str(i + 1) for i in open_v)} with {', '.join(open_c)}"
            else:
                line += "every label is known"
            lines.append(line)
        return lines


# --------------------------------------------------------------------- the domain the trainer speaks to
class Domain(Tc.Domain):
    name = "roboquest"
    def __init__(self, tasks="odd_parcel,marked_mugs", lam=0.3, split="validation"):
        """split: what eval_specs gives - "validation" (every fourth development instance) or "test" (the eval instances)."""
        self.families, self.lam, self.T = tasks.split(","), lam, 5
        self.split = "test" if split == "test" else "validation"
        rows = {f: [x for x in registry(f) if f != "odd_parcel" or x["cell"]["parcel_count"] == 4] for f in self.families}
        dev = {f: [x["instance_id"] for x in r if x["split"] == "dev"] for f, r in rows.items()}
        val = {f: ids[::4] for f, ids in dev.items()}
        self.evals = {f: [x["instance_id"] for x in r if x["split"] == "eval"] for f, r in rows.items()} if self.split == "test" else val
        self.dev_layouts = [instance("marked_mugs", i)["spec"] for i in dev["marked_mugs"]
                            if i not in val["marked_mugs"]] if "marked_mugs" in self.families else []
    # worlds
    def sample(self, rng): return (rng.choice(self.families), rng.randrange(0, 10 ** 9), "train")
    def eval_specs(self, n):
        out, k = [], 0
        pools = {f: list(v) for f, v in self.evals.items()}
        while len(out) < n:
            f = self.families[k % len(self.families)]; k += 1
            out.append((f, pools[f][(k // len(self.families)) % len(pools[f])], self.split))
        return out
    def make(self, spec):
        fam, key, split = spec
        if split in ("validation", "test"):
            sp = instance(fam, key)["spec"]
            if fam == "odd_parcel":
                odd = [p for p in sp["parcels"] if p["odd"]][0]; W = OddParcel(int(sp["parcel_count"]))
                return W, (int(odd["id"].split("_")[1]), odd["direction"], odd["delta_key"])
            W = MarkedMugs(int(sp["vessel_count"]), sp["pad_colors"], sp["nearest_pad"], sp["vessel"])
            return W, tuple(sp["labels"])
        rng = random.Random(key)
        if fam == "odd_parcel":                       # development class p4-o1-H: one heavy parcel
            W = OddParcel(4); return W, (rng.randrange(4), "H", rng.choice("sb"))
        lay = rng.choice(self.dev_layouts); n = int(lay["vessel_count"])
        pads = rng.sample(LABEL_COLORS, n); W = MarkedMugs(n, pads, lay["nearest_pad"], lay["vessel"])
        want_fixed = n in (3, 5)                      # development classes v3-f1+, v4-f0, v5-f1+
        options = [h for h in W.hypotheses if (sum(1 for lab, near in zip(h, W.nearest) if lab == W.pads[near]) > 0) == want_fixed]
        return W, rng.choice(options)
    # text
    def _history(self, e):
        lines = [f"Task: {e.W.rules()}", "Evidence so far:" if e.log else "Evidence so far: none yet"]
        lines += [f"[{k}] {e.W.look_name(a)}: {o}" for k, (a, o) in enumerate(e.log)]
        lines.append(f"Evidence actions used: {e.t} of {e.W.T}. Decide now.")
        return "\n".join(lines)
    def look_strings(self, e): return {a: f"ACTION: {e.W.look_name(a).upper()}" for a in e.W.looks} if e.t < e.W.T else {}
    def belief_note(self, e): return e.W.belief_text(e.B)
    def state_text(self, e): return "\n".join(e.W.trace(e) + [e.W.belief_text(e.B)])
    def family_of(self, e): return e.spec[0]
    # reading the written belief back
    def anchor_support(self, text, answers, e=None):
        W = e.W; body = text.split(":", 1)[1] if ":" in text else text; hyps = list(W.hypotheses)
        if W.family == "odd_parcel":
            names = {OddParcel.hname(h): h for h in hyps}
            listed = [names[t.strip()] for t in body.split(",") if t.strip() in names]
            return set(answers) if (not listed or "any" in body) else {W.answer(h) for h in listed}
        allowed = {}
        for clause in body.split(";"):
            clause = clause.strip()
            for i in range(W.n):
                head = f"{W.noun} {i + 1} "
                if clause.startswith(head):
                    rest = clause[len(head):]
                    allowed[i] = {rest[2:].strip()} if rest.startswith("=") else {x.strip() for x in rest.split("{", 1)[-1].rstrip("}").split(",")}
        keep = [h for h in hyps if all(h[i] in v for i, v in allowed.items())]
        return {W.answer(h) for h in keep} if keep else set(answers)
    def anchor_answer(self, text, answers, e=None):
        sup = self.anchor_support(text, answers, e)
        lp = e.W.prior(); best = {}
        for h, w in zip(e.W.hypotheses, lp):
            y = e.W.answer(h)
            if y in sup: best[y] = max(best.get(y, -np.inf), w)
        return max(best, key=best.get) if best else answers[0]
