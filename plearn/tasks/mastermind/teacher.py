"""Mastermind for the RLVB (EM) trainer, with TextArena's rules (benchmarks/textarena: textarena/envs/Mastermind at commit
1e3160d, MIT; the PyPI release 0.7.4 has the same rules and text): the code is 4
different digits from 1 to 6 (360 codes, drawn as TextArena draws it: random.sample of the digits); a guess is 4 different
digits, not guessed before, and returns TextArena's feedback: black pegs (a right digit in the right place) and white pegs (a
right digit in a wrong place); 20 turns. One addition for our method: the game ends with an answer line, ACTION: SUBMIT
[a b c d], right when it is the code; a guess equal to the code only shows 4 black pegs. A line that is not an action of the
game, a repeated guess or a guess against the rules uses up its turn, as in TextArena. The rules text is TextArena's
(locales/en.json, player_prompt.intro) with two changes: our action lines in place of its answer-format line, and 'turns to find
the code' for 'turns to guess the code' (a guess does not end our game). The feedback text and the feedback function
(_evaluate_guess) are TextArena's.

The teacher's belief is exact: the codes that agree with every feedback (a mask over the 360 codes). Its policy pi*: the value
of a guess is the answer accuracy after its feedback, E[c(t+1) rho(b')], one guess ahead; with the uniform posterior this is
proportional to the number of feedback groups the guess makes (the 'most parts' rule). A guess that splits nothing has no
value (-1). The answer's value is c(t) rho(b). The belief line: 'Possible: <codes>' when at most 6 codes remain, else the
digits each position still allows ('Possible: position 1 in {1, 2}; position 2 = 3; ...'), scored as the set of codes it allows.
"""
from __future__ import annotations
import itertools, math, random, re
import numpy as np

DIGITS, LENGTH, TURNS = 6, 4, 20
CODES = list(itertools.permutations(range(1, DIGITS + 1), LENGTH))         # the 360 codes, in a fixed order
INDEX = {c: i for i, c in enumerate(CODES)}
ARR = np.array(CODES, dtype=np.int8)
MOVE_COST = 0.999                                                         # c of one turn: lam = T ln(1 / 0.999)
LIST_MAX, MISS = 6, 50.0


def evaluate(guess, code):
    """TextArena's _evaluate_guess: (black pegs, white pegs)."""
    black, white = 0, 0
    secret, g = list(code), list(guess)
    for i in range(len(g)):
        if g[i] == secret[i]: black += 1; secret[i] = None; g[i] = None
    for i in range(len(g)):
        if g[i] is not None and g[i] in secret: white += 1; secret[secret.index(g[i])] = None
    return black, white


FB = np.array([[b * 5 + w for b, w in (evaluate(g, c) for c in CODES)] for g in CODES], dtype=np.int8)   # FB[guess, code]


def code_str(i): return " ".join(str(d) for d in CODES[i])


def rules_text():
    """TextArena's player prompt (locales/en.json, player_prompt.intro), with the two changes of the module text."""
    return (f"You are playing Mastermind.\nYou need to find the code that is {LENGTH} digits long, each digit from 1 to {DIGITS}, "
            f"with no duplicates.\nAfter each guess, you will receive feedback in the form of black and white pegs.\n"
            "A black peg indicates a correct digit in the correct position, while a white peg indicates a correct digit in the "
            f"wrong position.\nYou have {TURNS} turns to find the code.")


SYSTEM = ("You find a hidden code from the feedback to your guesses. Each turn, think, then reply with exactly two lines.\n"
          "Line 1: 'Possible: ' and the codes that the feedback still allows, for example 'Possible: 1234, 2134' when at most six "
          "remain, or the digits each position still allows, for example 'Possible: position 1 in {1, 2}; position 2 = 3; position 3 "
          "in {4, 5, 6}; position 4 in {1, 2, 5}', or 'Possible: all' before any feedback.\n"
          "Line 2: 'ACTION: GUESS [a b c d]' to guess, or 'ACTION: SUBMIT [a b c d]' to answer. You have one chance to answer: "
          "submit when the feedback identifies the code, and not before.")


class _GB:
    """A group of k codes for the reliability table: its class and U."""
    __slots__ = ("k",)
    def __init__(self, k): self.k = k
    def cls(self): return ("mastermind", min(self.k, 4))
    def U(self): return 1.0 / self.k


class Belief:
    """The codes M (a bool mask) that agree with every feedback; t = the guesses made; tried = the guesses made."""
    __slots__ = ("M", "t", "T", "tried", "_k")
    def __init__(self, M, t, T, tried):
        self.M, self.t, self.T, self.tried = M, t, T, tried; self._k = (t, M.tobytes(), tried)
    def __hash__(self): return hash(self._k)
    def __eq__(self, o): return self._k == o._k
    def n(self): return int(self.M.sum())
    def looks(self): return [g for g in range(len(CODES)) if g not in self.tried] if self.t < self.T else []
    def outcomes(self, g):
        alive = np.nonzero(self.M)[0]; f = FB[g, alive]; out = []
        for v in np.unique(f):
            M2 = np.zeros_like(self.M); M2[alive[f == v]] = True
            out.append((float((f == v).sum()) / len(alive), Belief(M2, self.t + 1, self.T, self.tried | {g})))
        return out
    def post(self):
        alive = np.nonzero(self.M)[0]; return {code_str(i): 1.0 / len(alive) for i in alive}
    def look_info(self, g):
        """--stall_cap: the expected drop of the answer entropy (nats) of guess g, log N - sum_f (n_f / N) log n_f over
        its feedback groups; 0 for a guess that splits nothing."""
        alive = np.nonzero(self.M)[0]; N = len(alive)
        if N <= 1: return 0.0
        cnt = np.bincount(FB[g, alive].astype(np.int64)); cnt = cnt[cnt > 0]
        return max(0.0, float(math.log(N) - (cnt / N * np.log(cnt)).sum()))
    def answer_counts(self): return {code_str(i): 1 for i in np.nonzero(self.M)[0]}
    def U(self): return 1.0 / self.n()
    def identified(self): return self.n() == 1
    def cls(self): return ("mastermind", min(self.n(), 4))
    def lookahead(self, rel, lam, T, depth, acts):
        """Q(g) of the legal guesses: E[u(c(t+1), rho(b'))] over g's feedback groups (Reliability.util); none for a guess that splits nothing."""
        if not acts: return {}
        alive = np.nonzero(self.M)[0]; N = len(alive); c1 = math.exp(-lam * (self.t + 1) / T); rho = {}
        def r(k):
            if k not in rho: rho[k] = rel.mean(_GB(k))
            return rho[k]
        F = FB[np.asarray(acts)][:, alive].astype(np.int64); out = {}
        for i, g in enumerate(acts):
            cnt = np.bincount(F[i], minlength=25); cnt = cnt[cnt > 0]
            out[g] = rel.with_cost(c1, float(sum(k / N * rel.f(r(int(k))) for k in cnt))) if len(cnt) > 1 else rel.none
        return out


# --------------------------------------------------------------------- the belief language
def read_line(text):
    """The codes (a bool mask over CODES) that a line 'Possible: ...' allows, or None when it is not in the language."""
    m = re.match(r"^\s*possible\s*:\s*(.*?)\s*\.?\s*$", str(text), re.I | re.S)
    if not m: return None
    body = m.group(1).strip()
    if re.fullmatch(r"(?:all|any|all codes|any code)", body, re.I): return np.ones(len(CODES), dtype=bool)
    if re.search(r"position", body, re.I):
        allowed = [set(range(1, DIGITS + 1)) for _ in range(LENGTH)]
        for clause in re.split(r"[;\n]+", body):
            clause = clause.strip(" .")
            if not clause: continue
            c = re.fullmatch(r"position\s*([1-4])\s*(?:=|is|:)\s*([1-6])", clause, re.I)
            if c: allowed[int(c.group(1)) - 1] = {int(c.group(2))}; continue
            c = re.fullmatch(r"position\s*([1-4])\s*(?:in|:)\s*\{?\s*([1-6](?:\s*[, ]\s*[1-6])*)\s*\}?", clause, re.I)
            if not c: return None
            allowed[int(c.group(1)) - 1] = {int(d) for d in re.findall(r"[1-6]", c.group(2))}
        M = np.ones(len(CODES), dtype=bool)
        for p in range(LENGTH): M &= np.isin(ARR[:, p], list(allowed[p]))
        return M if M.any() else None
    M = np.zeros(len(CODES), dtype=bool)
    for tok in re.split(r"[,;]+|\s{2,}|\bor\b", body):
        tok = tok.strip(" .[]()")
        if not tok: continue
        digits = tuple(int(d) for d in re.findall(r"\d", tok))
        if digits not in INDEX: return None
        M[INDEX[digits]] = True
    return M if M.any() else None


def set_score(B, S, miss=MISS):
    """S = -b(S) log|S| - miss b(outside S): the log score of the line read as uniform on its set S (B: the exact belief)."""
    pin = float((B.M & S).sum()) / B.n(); return -pin * math.log(max(1, int(S.sum()))) - miss * (1.0 - pin)


def exact_mask(B):
    alive = np.nonzero(B.M)[0]
    if len(alive) <= LIST_MAX: return B.M.copy()
    M = np.ones(len(CODES), dtype=bool)
    for p in range(LENGTH): M &= np.isin(ARR[:, p], np.unique(ARR[alive, p]))
    return M


def line_text(B):
    """The teacher's belief line: the codes when at most LIST_MAX remain, else the digits each position allows."""
    alive = np.nonzero(B.M)[0]
    if len(alive) <= LIST_MAX: return "Possible: " + ", ".join("".join(str(d) for d in CODES[i]) for i in alive)
    parts = []
    for p in range(LENGTH):
        ds = sorted(set(ARR[alive, p].tolist()))
        if len(ds) == DIGITS: continue
        parts.append(f"position {p + 1} = {ds[0]}" if len(ds) == 1 else f"position {p + 1} in {{{', '.join(map(str, ds))}}}")
    return "Possible: " + ("; ".join(parts) if parts else "all")


# --------------------------------------------------------------------- episodes and the trainer's contract
class Ep:
    def __init__(self, spec, code):
        self.spec, self.code = spec, code
        self.M, self.tried, self.log = np.ones(len(CODES), dtype=bool), frozenset(), []
        self.t, self.done, self.correct, self.turns_used, self.invalid, self.note, self.answer = 0, False, None, 0, 0, None, None


class Domain:
    name = "mastermind"
    images = False
    value_modes = ("accuracy", "log")                                # Belief.lookahead computes both (Reliability.util)
    look_scale = False     # --tau_look is ignored here (tau): the lookahead separates guesses whose real difference is tiny (about a
                           # third of a guess), so a sharper scale would amplify a ranking that barely matters (plearn/method/trainer.py)

    def __init__(self, split="test", **_):
        self.split = split; self.T = TURNS - 1; self.turns = TURNS; self.lam = self.T * math.log(1 / MOVE_COST)
        self.families = ["mastermind"]

    # specs: training seeds from 10^6 up; validation 10000 + i; test 20000 + i
    def sample(self, rng): return ("mastermind", rng.randrange(10 ** 6, 2 ** 32), "train")
    def eval_specs(self, n):
        base = {"test": 20000}.get(self.split, 10000)
        return [("mastermind", base + i, self.split) for i in range(n)]
    def start(self, specs):
        out = []
        for s in specs:
            code = random.Random(int(s[1])).sample(range(1, DIGITS + 1), LENGTH)   # TextArena: random.sample(list(range(1, n + 1)), k)
            out.append(Ep(tuple(s), INDEX[tuple(code)]))
        return out
    def close(self, eps): pass
    def family_of(self, e): return "mastermind"

    # observations (text)
    def system_text(self, e): return SYSTEM
    def _history(self, e):
        lines = [rules_text(), "Guesses so far:" if e.log else "Guesses so far: none yet"]
        for g, b, w in e.log: lines.append(f"Submitted [{code_str(g)}]. Feedback: {b} black peg(s), {w} white peg(s).")
        lines.append(f"Turns used: {e.turns_used} of {TURNS}. Decide now.")
        return "\n".join(lines)

    # actions
    def look_strings(self, e): return {g: f"ACTION: GUESS [{code_str(g)}]" for g in range(len(CODES)) if g not in e.tried} if e.t < self.T else {}
    def look_name(self, g): return f"[{code_str(g)}]"
    def answer_options(self, e): return [code_str(i) for i in range(len(CODES))]
    def commit_string(self, y): return f"ACTION: SUBMIT [{y}]"
    def _read(self, text):
        m = re.match(r"^\s*ACTION\s*:\s*(GUESS|SUBMIT)\s*\[?\s*((?:\d[\s,]*){4})\]?\s*\.?\s*$", str(text), re.I)
        if not m: return None, None
        digits = tuple(int(d) for d in re.findall(r"\d", m.group(2)))
        return m.group(1).upper(), INDEX.get(digits)
    def canonical_action(self, text):
        what, i = self._read(text)
        return f"ACTION: {what} [{code_str(i)}]" if i is not None else str(text).strip()
    def parse_action(self, e, text):
        what, i = self._read(text)
        if i is None: return ("invalid", None)                         # not 4 different digits from 1 to 6
        if what == "SUBMIT": return ("answer", code_str(i))
        if i in e.tried or e.t >= self.T: return ("invalid", None)     # a repeated guess, or no guess left
        return ("look", i)
    def do_look(self, e, g):
        b, w = evaluate(CODES[g], CODES[e.code]); e.log.append((g, b, w))
        e.M = e.M & (FB[g] == FB[g, e.code]); e.tried = e.tried | {g}
    def do_commit(self, e, y): e.answer = y; return y == code_str(e.code)
    def after_turn(self, e):
        e.turns_used += 1
        if not e.done and e.turns_used >= TURNS: e.done, e.correct = True, False

    # belief and teacher
    def belief(self, e): return Belief(e.M.copy(), e.t, self.T, e.tried)
    def label(self, e): return code_str(e.code)
    def answer_posterior(self, e): return self.belief(e).post()
    def answer_prob(self, B, y):
        m = re.findall(r"\d", str(y)); i = INDEX.get(tuple(int(d) for d in m))
        return 1.0 / B.n() if i is not None and B.M[i] else 0.0
    def state_text(self, e): return line_text(self.belief(e))
    def belief_note(self, e): return self.state_text(e)
    def belief_score(self, B, text, miss=MISS):
        S = read_line(text); return -miss if S is None else set_score(B, S, miss)
    def belief_best(self, B, miss=MISS): return set_score(B, exact_mask(B), miss)
    def line_excess(self, B, text, exact):   # the share of the line's codes that the exact line excludes (--line_wide), both read by this reader
        S, X = read_line(text), read_line(exact); return float((S & ~X).sum()) / float(S.sum()) if S is not None and X is not None else 0.0
    def line_belief(self, B, text):
        """The belief a line states, as the teacher's posterior (--line_score decision): the codes it allows that the feedback
        allows, and those it allows beyond the exact line (exact_mask); None when it is not a belief line or allows nothing. The
        language names at most LIST_MAX codes, so above that the exact line allows codes the feedback excluded: they are the
        language's, not the line's (as the codes themselves, read as allowed, gave 34 percent of the teacher's histories an act
        other than the teacher's, by up to 28 nats)."""
        S = read_line(text)
        if S is None: return None
        M = (S & B.M) | (S & ~exact_mask(B))
        return Belief(M, B.t, B.T, B.tried) if M.any() else None
    def read_exact(self, text): return read_line(text) is not None
    def anchor_support(self, text, answers, e=None):
        S = read_line(text); return {code_str(i) for i in np.nonzero(S)[0]} if S is not None else set(answers)
    def anchor_equal(self, a, b):
        A, B_ = read_line(a), read_line(b); return A is not None and B_ is not None and bool((A == B_).all())
    def anchor_answer(self, anchor, answers, e=None):
        S = read_line(anchor); return code_str(int(np.nonzero(S)[0][0])) if S is not None else answers[0]
    def extra_metrics(self, eps):
        if not eps: return {}
        return {"guesses": float(np.mean([len(e.log) for e in eps])), "solved": float(np.mean([bool(e.correct) for e in eps]))}
