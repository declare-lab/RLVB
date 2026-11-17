"""The tabletop environment for the RLVB (EM) trainer: reading the state of objects in 3D pictures (evidence
that only pictures show, as a balance's tilt in RoboQuest). ManiSkill3's table scene (scene.py), no robot: the agent acts
through helpers that move objects exactly (as RoboQuest at the decision level), and it sees only two camera pictures (left and
right, 512 x 512) after each action. Three families, each with a hidden target, an
exact posterior and evidence that only the pictures show:

- lamps: N = 4 or 5 lamps in a row (1 = the left one, as the cameras see them) and N push buttons in front of them (A = the
  left one). Each button lights one lamp, a mapping drawn evenly from the N! ones. PRESS X lights the lamp of button X, and
  the lamp stays lit: the new lamp shows only against the pictures before (a weak glow on grey bulbs). Answer: the whole
  mapping, SUBMIT A2 B1 C4 D3.
- cups: N = 3 to 5 upside-down cups in a row (1 = the left one). The object named in the task is under one of them, drawn
  evenly; the others hide a look-alike (another colour or another shape) or nothing. LIFT k raises cup k and leaves it up.
  Answer: SUBMIT k.
- blocks: N = 3 or 4 blocks in a row (1 = the left one) on lines drawn across the table every 3 cm. One block, drawn evenly,
  slides on a slippery base. PUSH k moves block k away from the cameras: 3 cm, or 9 cm when it is the slippery one. Answer:
  SUBMIT k.

The teacher's belief is exact: the hypotheses (the mapping, the cup, the block) that agree with every picture so far, all equally
likely. Its policy pi*: the shared lookahead (estimator.lookahead_Q) over looks() and outcomes(a), which split the hypotheses by
what the pictures would show. The belief line names what is still possible: 'Possible: cups 2, 4', 'Possible: blocks 1, 3',
'Possible: A 2; B 1 or 3; C 1 or 3' ('Possible: all' before any evidence), scored as the set of answers it allows.

Splits: training seeds from 10^6 up, validation 10000 + i, test 20000 + i. The test split also has its own colours (the cups,
the blocks, the named objects, the look-alikes): pictures that training never shows.
"""
from __future__ import annotations
import itertools, math, random, re
import numpy as np
from plearn.api.episode import carried

MOVE_COST, MISS, VIEWS = 0.98, 50.0, 8                              # 0.02 nats an action; the newest VIEWS pairs of pictures
LETTERS = "ABCDE"
FAMILIES = ("lamps", "cups", "blocks")
NS = {"lamps": (4, 5), "cups": (3, 4, 5), "blocks": (3, 4)}
COLOURS = {   # name: rgba; the training and validation colours, and the test's own
    "train": {"red": (0.80, 0.10, 0.10, 1), "blue": (0.12, 0.25, 0.85, 1), "green": (0.10, 0.60, 0.20, 1), "yellow": (0.90, 0.80, 0.10, 1),
              "white": (0.92, 0.92, 0.92, 1)},
    "test": {"purple": (0.55, 0.20, 0.70, 1), "orange": (0.95, 0.50, 0.10, 1), "cyan": (0.10, 0.75, 0.80, 1), "pink": (0.95, 0.45, 0.65, 1),
             "black": (0.08, 0.08, 0.08, 1)},
}
SURFACE = {"train": {"cups": (0.15, 0.20, 0.45, 1), "blocks": (0.70, 0.62, 0.48, 1)},
           "test": {"cups": (0.45, 0.25, 0.15, 1), "blocks": (0.45, 0.50, 0.55, 1)}}
SHAPES = {"box": "cube", "sphere": "ball", "cylinder": "cylinder"}
GREY, DARK, GLOW = (0.85, 0.85, 0.85, 1), (0.18, 0.18, 0.20, 1), (1.0, 0.85, 0.35, 0.6)   # a lit bulb: grey with a weak glow
NORMAL, SLIPPERY, GAP = 0.03, 0.09, 0.03                            # the two push lengths and the lines' spacing (m)


def ys(n, half):
    """The row's places across the table: 1 (left in the pictures) to n."""
    return list(np.linspace(-half, half, n)) if n > 1 else [0.0]


def perm_str(p): return " ".join(f"{LETTERS[i]}{p[i] + 1}" for i in range(len(p)))


# --------------------------------------------------------------------- the belief and the lookahead
class Belief:
    """The hypotheses still possible (alive: cup or block numbers from 0, or mappings as tuples), the looks done, the time."""
    __slots__ = ("fam", "n", "alive", "done", "t", "_k")
    def __init__(self, fam, n, alive, done, t):
        self.fam, self.n, self.alive, self.done, self.t = fam, n, frozenset(alive), frozenset(done), t
        self._k = (fam, n, self.alive, self.done, t)
    def __hash__(self): return hash(self._k)
    def __eq__(self, o): return self._k == o._k
    def U(self): return 1.0 / len(self.alive)
    def identified(self): return len(self.alive) == 1
    def cls(self): return ("tabletop-" + self.fam, min(len(self.alive), 4))
    def answer_counts(self): return {answer_of(self.fam, h): 1 for h in self.alive}
    def post(self): return {answer_of(self.fam, h): 1.0 / len(self.alive) for h in sorted(self.alive)}
    def looks(self): return [a for a in range(self.n) if a not in self.done]   # a look done shows nothing new: each helper once
    def outcomes(self, a):
        """(probability, the belief after the pictures) for look a: the hypotheses split by what the pictures would show."""
        groups = {}
        for h in self.alive: groups.setdefault(shown(self.fam, h, a), []).append(h)
        k = len(self.alive)
        return [(len(hs) / k, Belief(self.fam, self.n, hs, self.done | {a}, self.t + 1)) for hs in groups.values()]


def answer_of(fam, h): return perm_str(h) if fam == "lamps" else str(h + 1)


def shown(fam, h, a):
    """What the pictures show after look a under hypothesis h: the lamp that lights (new against the pictures before); whether
    the object is under the cup; whether the block slides far."""
    if fam == "lamps": return h[a]
    return h == a


# --------------------------------------------------------------------- the belief language
def read_line(text, fam, n):
    """The set of answers a belief line allows (None: not a belief line; empty: a lamps line that no mapping fits)."""
    m = re.match(r"^\s*possible\s*:\s*(.*?)\s*\.?\s*$", str(text), re.I | re.S)
    if not m: return None
    body = m.group(1).strip(); every = all_answers(fam, n)
    if re.fullmatch(r"(?:all|any|anything|all of them)", body, re.I): return set(every)
    if fam == "lamps":
        allow = {}
        for part in re.split(r"[;\n]", body):
            q = re.fullmatch(r"\s*(?:button\s+)?([A-E])\s*(?:->|→|:|=|to|lights)?\s*(?:lamps?\s*)?([\d\s,]+(?:\s*(?:or|and)\s*\d+)*)\s*", part, re.I)
            if not q: return None
            L = LETTERS.index(q.group(1).upper()); lamps = {int(d) - 1 for d in re.findall(r"\d+", q.group(2))}
            if L >= n or not lamps or any(not 0 <= v < n for v in lamps): return None
            allow[L] = lamps
        return {perm_str(p) for p in itertools.permutations(range(n)) if all(p[L] in s for L, s in allow.items())}
    word = "cups?" if fam == "cups" else "blocks?"
    if not re.fullmatch(rf"(?:{word})?\s*[\d\s,]+(?:\s*(?:or|and)\s*\d+)*", body, re.I): return None
    s = {int(d) for d in re.findall(r"\d+", body)}
    return {str(v) for v in s} if s and all(1 <= v <= n for v in s) else None


def fact_classes(B):
    """The exact answer set under every picture sequence that the looks done could show (one set for each sequence of visible
    facts): the actual one is B's own; the others are what a misread picture leads to."""
    out = {}
    for h in (itertools.permutations(range(B.n)) if B.fam == "lamps" else range(B.n)):
        out.setdefault(tuple(shown(B.fam, h, a) for a in sorted(B.done)), set()).add(answer_of(B.fam, h))
    return list(out.values())


def all_answers(fam, n): return [perm_str(p) for p in itertools.permutations(range(n))] if fam == "lamps" else [str(k + 1) for k in range(n)]


def set_score(post, s, miss=MISS):
    pin = sum(post.get(a, 0.0) for a in s); return -pin * math.log(max(1, len(s))) - miss * (1.0 - pin)


def exact_line(B):
    if len(B.alive) == len(all_answers(B.fam, B.n)): return "Possible: all"
    if B.fam == "lamps":
        sets = [sorted({h[L] + 1 for h in B.alive}) for L in range(B.n)]
        return "Possible: " + "; ".join(f"{LETTERS[L]} " + " or ".join(map(str, s)) for L, s in enumerate(sets))
    word = "cup" if B.fam == "cups" else "block"
    k = sorted(h + 1 for h in B.alive)
    return f"Possible: {word} {k[0]}" if len(k) == 1 else f"Possible: {word}s " + ", ".join(map(str, k))


# --------------------------------------------------------------------- episodes
class Ep:
    def __init__(self, spec, render=True):
        self.spec = spec; fam = spec[0].split("-", 1)[1]; self.fam, split = fam, spec[2]
        rng = random.Random(f"tabletop:{fam}:{spec[1]}:{split}")
        self.n = n = rng.choice(NS[fam]); self.budget = n                 # n - 1 looks always suffice
        pal = "test" if split == "test" else "train"; self.colours = COLOURS[pal]; self.surface = SURFACE[pal]
        if fam == "lamps": self.truth = tuple(rng.sample(range(n), n))
        else: self.truth = rng.randrange(n)
        if fam == "cups":                                               # the named object, and what the other cups hide
            names = list(self.colours); shapes = list(SHAPES)
            self.target = (rng.choice(shapes), rng.choice(names)); self.under = {}
            for k in range(n):
                if k == self.truth: self.under[k] = self.target; continue
                r = rng.random()
                if r < 0.25: self.under[k] = None
                elif r < 0.625: self.under[k] = (self.target[0], rng.choice([c for c in names if c != self.target[1]]))
                else: self.under[k] = (rng.choice([s for s in shapes if s != self.target[0]]), self.target[1])
        self.jitter = [(rng.uniform(-0.008, 0.008), rng.uniform(-0.008, 0.008)) for _ in range(n)]
        self.alive = set(range(n)) if fam != "lamps" else set(itertools.permutations(range(n)))
        self.done_looks, self.lit, self.moved = set(), set(), {}
        self.t, self.done, self.correct, self.turns_used, self.invalid, self.note, self.answer = 0, False, None, 0, 0, None, None
        self.pictures, self.log, self.render = [], [], render
        self.see()

    # what the scene holds now
    def objects(self):
        n, out, J = self.n, {}, self.jitter
        if self.fam == "lamps":
            for i, y in enumerate(ys(n, 0.18)):
                x = -0.14 + J[i][0]
                out[f"stand{i}"] = (x, y, 0.050, DARK, None)
                out[f"bulb{i}"] = (x, y, 0.132, GREY, GLOW if i in self.lit else None)
                pressed = i in self.done_looks and self.log and self.log[-1] == f"press {LETTERS[i]}"
                out[f"button{i}"] = (0.10, y, 0.008 + (0.0 if pressed else 0.006), (0.75, 0.15, 0.15, 1), None)
            out["panel"] = (0.10, 0.0, 0.004, DARK, None)
        elif self.fam == "cups":
            for i, y in enumerate(ys(n, 0.20)):
                x, yy = J[i][0], y + J[i][1]
                up = 0.14 if i in self.done_looks else 0.0
                out[f"cup{i}"] = (x, yy, 0.048 + up, self.surface["cups"], None)
                if self.under[i] is not None:
                    shape, colour = self.under[i]; z = {"box": 0.018, "sphere": 0.020, "cylinder": 0.020}[shape]
                    out[f"obj{i}_{shape}"] = (x, yy, z, self.colours[colour], None)
        else:
            for i, y in enumerate(ys(n, 0.15)):
                out[f"block{i}"] = (0.04 - self.moved.get(i, 0.0), y + J[i][1], 0.025, self.surface["blocks"], None)
            for j in range(6): out[f"mark{j}"] = (0.065 - j * GAP, 0.0, 0.0006, DARK, None)
        return out

    def see(self):
        """The pictures now (left, right), kept for the prompt."""
        if self.render:
            from plearn.tasks.tabletop.scene import render
            self.pictures.append(render(self.fam, self.objects()))
        else:
            self.pictures.append((None, None))


# --------------------------------------------------------------------- prompts
def rules_text(e):
    n = e.n
    if e.fam == "lamps":
        return (f"On the table stand {n} lamps in a row, numbered 1 to {n} from the left as the cameras see them, and in front of them "
                f"{n} red push buttons, {', '.join(LETTERS[:n])} from the left. Each button lights one lamp; every mapping of the buttons "
                f"to the lamps is equally likely. PRESS and a button (for example PRESS A) lights the lamp of that button, and the "
                f"lamp stays lit; each button can be pressed once. You see two pictures of the table, from the left and from the right. Find the whole mapping and answer "
                f"with SUBMIT and each button with its lamp, for example SUBMIT {perm_str(tuple(range(n)))}. You may press at most {n} times.")
    if e.fam == "cups":
        shape, colour = e.target
        return (f"On the table stand {n} upside-down cups in a row, numbered 1 to {n} from the left as the cameras see them. The "
                f"{colour} {SHAPES[shape]} is under one of them; every cup is equally likely. The other cups hide something that "
                f"looks alike (another colour or another shape) or nothing. LIFT and a cup (for example LIFT 2) raises that cup and "
                f"leaves it up; each cup can be lifted once. You see two pictures of the table, from the left and from the right. Find the cup of the {colour} "
                f"{SHAPES[shape]} and answer with SUBMIT and its number, for example SUBMIT 3. You may lift at most {n} cups.")
    return (f"On the table stand {n} blocks in a row, numbered 1 to {n} from the left as the cameras see them, on lines drawn "
            f"across the table 3 cm apart. One block stands on a slippery base; every block is equally likely. PUSH and a block (for "
            f"example PUSH 2) pushes that block away from the cameras: it moves 3 cm, or 9 cm when it is the slippery one, and stays "
            f"there; each block can be pushed once. You see two pictures of the table, from the left and from the right. Find the slippery block and answer with "
            f"SUBMIT and its number, for example SUBMIT 1. You may push at most {n} times.")


SYSTEM = {
    "lamps": ("You find which button lights which lamp from what you see. Each turn, think, then reply with exactly two lines.\n"
              "Line 1: 'Possible: ' and, for each button, the lamps it may still light, for example 'Possible: A 2; B 1 or 3; C 1 or 3', "
              "or 'Possible: all' before any evidence.\n"
              "Line 2: 'ACTION: PRESS <button>', or 'ACTION: SUBMIT <mapping>' to answer, for example 'ACTION: SUBMIT A2 B1 C3'. You "
              "have one chance to answer: submit when the evidence identifies the mapping, and not before."),
    "cups": ("You find which cup hides an object from what you see. Each turn, think, then reply with exactly two lines.\n"
             "Line 1: 'Possible: ' and the cups that may still hide it, for example 'Possible: cups 2, 4', or 'Possible: all' before "
             "any evidence.\n"
             "Line 2: 'ACTION: LIFT <cup>', or 'ACTION: SUBMIT <cup>' to answer. You have one chance to answer: submit when the "
             "evidence identifies the cup, and not before."),
    "blocks": ("You find which block is slippery from what you see. Each turn, think, then reply with exactly two lines.\n"
               "Line 1: 'Possible: ' and the blocks that may still be the slippery one, for example 'Possible: blocks 1, 3', or "
               "'Possible: all' before any evidence.\n"
               "Line 2: 'ACTION: PUSH <block>', or 'ACTION: SUBMIT <block>' to answer. You have one chance to answer: submit when "
               "the evidence identifies the block, and not before."),
}
VERB = {"lamps": "PRESS", "cups": "LIFT", "blocks": "PUSH"}


# --------------------------------------------------------------------- the trainer's contract
class Domain:
    name = "tabletop"
    images = True
    value_modes = ("accuracy", "log")

    def __init__(self, levels=None, split="test", render=True, **_):
        self.levels = list(dict.fromkeys(levels or FAMILIES)); assert all(l in FAMILIES for l in self.levels), self.levels
        self.split, self.render = split, bool(render)
        self.T = max(max(NS[l]) for l in self.levels); self.turns = self.T + 2; self.lam = self.T * math.log(1 / MOVE_COST)
        self.families = [f"tabletop-{l}" for l in self.levels]

    def sample(self, rng): return (rng.choice(self.families), rng.randrange(10 ** 6, 2 ** 31), "train")
    def eval_specs(self, n):
        base = {"test": 20000}.get(self.split, 10000)
        return [(f, base + i, self.split) for f in self.families for i in range(n)]
    def start(self, specs): return [Ep(tuple(s), self.render) for s in specs]
    def close(self, eps): pass
    def family_of(self, e): return e.spec[0]

    def system_text(self, e): return SYSTEM[e.fam]
    def user_content(self, e, note=None):
        """The rules, then the pictures in order (the start and after each action; the newest VIEWS pairs), the last belief line,
        and the actions used."""
        parts, images = [{"type": "text", "text": rules_text(e)}], []
        first = max(0, len(e.pictures) - VIEWS)
        if first: parts.append({"type": "text", "text": f"\n(The first {first} pairs of pictures are no longer shown.)"})
        for k in range(first, len(e.pictures)):
            label = "\nThe start" if k == 0 else f"\nAfter action {k} ({e.log[k - 1]})"
            parts += [{"type": "text", "text": f"{label}, from the left: "}, {"type": "image"}, {"type": "text", "text": " from the right: "},
                      {"type": "image"}]
            images += list(e.pictures[k])
        parts.append({"type": "text", "text": carried(note) + f"\nActions used: {e.t} of {e.budget}. Decide now."})
        return parts, images
    def _history(self, e): raise NotImplementedError("the observations are pictures")

    def look_strings(self, e):
        if e.done or e.t >= e.budget: return {}
        names = LETTERS if e.fam == "lamps" else [str(k + 1) for k in range(e.n)]
        return {a: f"ACTION: {VERB[e.fam]} {names[a]}" for a in range(e.n) if a not in e.done_looks}   # each helper once
    def look_name(self, a, e=None): return f"{VERB[e.fam].lower()} {LETTERS[a] if e.fam == 'lamps' else a + 1}" if e is not None else str(a)
    look_name_takes_episode = True
    def answer_options(self, e): return all_answers(e.fam, e.n)
    def commit_string(self, y): return f"ACTION: SUBMIT {y}"
    def canonical_action(self, text):
        t = " ".join(str(text).strip().rstrip(".").split()); u = t.upper()
        m = re.fullmatch(r"(?:ACTION:\s*)?(PRESS|LIFT|PUSH)\s+(?:BUTTON\s+|CUP\s+|BLOCK\s+)?([A-E]|\d+)", u)
        if m: return f"ACTION: {m.group(1)} {m.group(2)}"
        m = re.fullmatch(r"(?:ACTION:\s*)?SUBMIT\s+(?:CUP\s+|BLOCK\s+)?(\d+)", u)
        if m: return f"ACTION: SUBMIT {int(m.group(1))}"
        m = re.fullmatch(r"(?:ACTION:\s*)?SUBMIT\s+((?:[A-E]\s*\d\s*,?\s*)+)", u)
        if m: return "ACTION: SUBMIT " + " ".join(f"{a}{b}" for a, b in re.findall(r"([A-E])\s*(\d)", m.group(1)))
        return t
    def parse_action(self, e, text):
        t = self.canonical_action(text)
        if t.startswith("ACTION: SUBMIT "):
            y = t[len("ACTION: SUBMIT "):]; return ("answer", y) if y in all_answers(e.fam, e.n) else ("invalid", None)
        for a, s in self.look_strings(e).items():
            if s == t: return ("look", a)
        return ("invalid", None)
    def do_look(self, e, a):
        a = int(a); e.done_looks.add(a)
        if e.fam == "lamps": e.lit.add(e.truth[a]); e.log.append(f"press {LETTERS[a]}")
        elif e.fam == "cups": e.log.append(f"lift {a + 1}")
        else: e.moved[a] = e.moved.get(a, 0.0) + (SLIPPERY if a == e.truth else NORMAL); e.log.append(f"push {a + 1}")
        seen = shown(e.fam, e.truth, a); e.alive = {h for h in e.alive if shown(e.fam, h, a) == seen}
        e.see()
    def do_commit(self, e, y): e.answer = y; return str(y) == answer_of(e.fam, e.truth)
    def after_turn(self, e):
        e.turns_used += 1
        if not e.done and e.turns_used >= self.turns: e.done, e.correct = True, False

    def belief(self, e): return Belief(e.fam, e.n, e.alive, e.done_looks, e.t)
    def label(self, e): return answer_of(e.fam, e.truth)
    def answer_posterior(self, e): return self.belief(e).post()
    def answer_prob(self, B, y): return B.post().get(str(y).strip(), 0.0)
    def state_text(self, e): return exact_line(self.belief(e))
    def belief_note(self, e): return self.state_text(e)
    def belief_score(self, B, text, miss=MISS):
        s = read_line(text, B.fam, B.n); return -miss if not s else set_score(B.post(), s, miss)
    def belief_best(self, B, miss=MISS): return set_score(B.post(), set(B.post()), miss)
    def line_excess(self, B, text, exact):   # the share of the line's answers that the exact line excludes (--line_wide), both read by this reader
        s, x = read_line(text, B.fam, B.n), read_line(exact, B.fam, B.n); return len(s - x) / len(s) if s and x is not None else 0.0
    def line_belief(self, B, text):
        """The belief a line states, as the teacher's posterior (--line_score decision): the hypotheses whose answer it allows
        that the pictures allow, and those it allows beyond the exact line (the lamps' language states a set per button, so the
        exact line can allow mappings the pictures excluded: they are the language's, not the line's); None when it is not a
        belief line or allows nothing."""
        s = read_line(text, B.fam, B.n)
        if not s: return None
        exact = read_line(exact_line(B), B.fam, B.n); post = B.post()
        alive = {h for h in (itertools.permutations(range(B.n)) if B.fam == "lamps" else range(B.n))
                 if answer_of(B.fam, h) in s and (answer_of(B.fam, h) in post or answer_of(B.fam, h) not in exact)}
        return Belief(B.fam, B.n, alive, B.done, B.t) if alive else None
    def reading(self, B, text, truth=None):
        """The belief line against the pictures so far (Image -> Visual Facts -> Belief -> Action). kind: 'exact';
        'wider' (every answer that the evidence allows, and more); 'misread' (the exact set of another picture sequence: a visual
        recognition error); 'reasoning' (it leaves out an answer that the evidence allows and fits no picture: a belief reasoning
        error); 'unreadable'. evidence: an action has shown something. miss: the line leaves out the true answer. width: the
        answers the line keeps, and those the evidence keeps."""
        s = read_line(text, B.fam, B.n); exact = set(B.post())
        if s is None: kind = "unreadable"
        elif s == exact: kind = "exact"
        elif exact <= s: kind = "wider"
        elif s in fact_classes(B): kind = "misread"
        else: kind = "reasoning"
        return {"kind": kind, "evidence": bool(B.done), "miss": None if truth is None or s is None else str(truth) not in s,
                "width": None if s is None else len(s), "width_exact": len(exact)}
    def action_ok(self, B, kind, val):
        """The teacher's decision at belief B: the answer once the evidence decides it, before that a helper not used yet."""
        if B.identified(): return kind == "answer" and str(val) == next(iter(B.post()))
        return kind == "look"
    def read_exact(self, text): return any(read_line(text, f, 5) is not None for f in ("cups", "blocks")) or read_line(text, "lamps", 5) is not None
    def anchor_support(self, text, answers, e=None):
        fam, n = (e.fam, e.n) if e is not None else ("cups", len(answers))
        s = read_line(text, fam, n); return set(s) if s else set(answers)
    def anchor_equal(self, a, b):
        for fam, n in (("cups", 5), ("blocks", 5), ("lamps", 5), ("lamps", 4)):
            x, y = read_line(a, fam, n), read_line(b, fam, n)
            if x is not None or y is not None: return x is not None and x == y
        return False
    def anchor_answer(self, anchor, answers, e=None):
        s = self.anchor_support(anchor, answers, e); return sorted(s)[0] if s else answers[0]
    def extra_metrics(self, eps):
        if not eps: return {}
        return {"looks": float(np.mean([len(e.done_looks) for e in eps]))}
