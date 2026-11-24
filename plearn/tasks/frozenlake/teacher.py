"""FrozenLake with the goal hidden (VAGEN's simulator: gymnasium's FrozenLake, not slippery, with VAGEN's own map generator from
benchmarks/vagen, vagen/envs/frozenlake/utils/utils.py, MIT; 4 x 4, each cell a hole with chance 0.2, the start and the goal two
different random cells, a path from the start to the goal required) for the RLVB (EM) trainer. Our hidden version: the agent
sees the cells it has been on and the four cells next to it, and fog covers every other cell, the goal too; the picture is the game's
own with that fog. The actions are the game's: LEFT, DOWN, RIGHT, UP. A hole ends the game; the goal wins it; VAGEN's evaluation
allows 25 actions. There is no answer line: reaching the goal is the answer. This tests information gathering, exploration and
search.

The teacher's belief is exact: the (goal cell, hole cells) pairs that the generator draws for this start (15 x 16,384, each with its
weight 0.8^frozen 0.2^holes, kept when a path exists), that agree with every cell seen. Its policy pi* (Reliability.util of the
return: reaching the goal after n actions is worth u(c(n), phi), a hole or the action limit u(c, 0)): a move is worth what it leads
to (Belief.lookahead). When the goal is known (and reachable through seen cells), the return of walking to it along a safe
shortest route from the cell the move leaves the agent on. Otherwise an exact expectimax over the posterior, `depth` moves deep
(each move splits the hypotheses by what the cells it shows hold; the best move after each outcome), whose leaf is the remaining
walk of each hypothesis: its shortest route to its goal through its own frozen cells (walks(); a hole or past the action limit:
the floor). A step into a known hole is worth the floor.
Before the fix (plearn.tasks.FL_VIEW_VALUE, --fl_view_value 1: _view_lookahead): a move was worth the best plan through the cells
known to be safe that starts with it: walking to a cell whose new neighbours the agent then sees, worth E[u] over what they show,
the answer accuracy of the best guess of the goal (U = its probability, times phi) then; or, when the goal is known, walking onto it
(accuracy phi); each with the time of its moves. That valued a view as if the game ended there, so with the goal known a walk to any
fog was worth as much as reaching the goal in as many moves (the trainer's module text).
The belief line names the cells that may hold the goal, 'Possible: goal at B3, D4' ('Possible: any' at the start), scored as a set.
"""
from __future__ import annotations
import copy, math, os, re
from functools import lru_cache
import numpy as np
from vagen.envs.frozenlake.utils.utils import generate_random_map         # VAGEN's map of a seed (benchmarks/vagen)
from plearn.api.episode import carried
import plearn.tasks as _tasks                                        # FL_VIEW_VALUE

SIZE, P_FROZEN, T_ACTIONS = 4, 0.8, 25
TEST_SEEDS = range(20000, 20128)                                     # VAGEN's evaluation seeds
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
ACTIONS = {LEFT: "LEFT", DOWN: "DOWN", RIGHT: "RIGHT", UP: "UP"}
MOVE = {LEFT: (0, -1), DOWN: (1, 0), RIGHT: (0, 1), UP: (-1, 0)}    # (row, column) steps, gymnasium's action numbers
MOVE_COST, MISS = 0.98, 50.0
FOG = np.array([168, 168, 168], dtype=np.uint8)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def next_seed(seed): return (1664525 * int(seed) + 1013904223) % (2 ** 32)


_TEST_MAPS = []
def test_maps():
    """The test maps (VAGEN's evaluation seeds): training and validation never use one."""
    if not _TEST_MAPS: _TEST_MAPS.append({tuple(generate_random_map(SIZE, P_FROZEN, s)) for s in TEST_SEEDS})
    return _TEST_MAPS[0]


def name(i): return f"{chr(65 + i % SIZE)}{i // SIZE + 1}"


def neighbours(i):
    r, c = divmod(i, SIZE)
    return [(r + dr) * SIZE + c + dc for dr, dc in MOVE.values() if 0 <= r + dr < SIZE and 0 <= c + dc < SIZE]


@lru_cache(maxsize=None)
def hypotheses(start):
    """Every (goal, hole mask, weight) the generator draws for this start: the cells other than the start and the goal are holes
    with chance 0.2, and a path from the start to the goal must exist."""
    n = SIZE * SIZE; full = (1 << n) - 1
    left = sum(1 << i for i in range(n) if i % SIZE == 0); right = sum(1 << i for i in range(n) if i % SIZE == SIZE - 1)
    G, Hm, W = [], [], []
    for g in range(n):
        if g == start: continue
        others = np.array([i for i in range(n) if i not in (start, g)], dtype=np.int64)
        pat = np.arange(1 << len(others), dtype=np.int64)
        holes = np.zeros(len(pat), dtype=np.int64)
        for k, cell in enumerate(others): holes |= ((pat >> k) & 1) << cell
        free = full & ~holes; reach = np.full(len(pat), 1 << start, dtype=np.int64)
        for _ in range(n):                                             # flood fill over the frozen cells
            grow = reach | (reach << SIZE) | (reach >> SIZE) | ((reach << 1) & ~left) | ((reach >> 1) & ~right)
            reach = grow & free & full
        ok = (reach >> g) & 1 == 1
        k_holes = np.bitwise_count(holes[ok]).astype(np.int64)
        G.append(np.full(int(ok.sum()), g)); Hm.append(holes[ok]); W.append((1 - P_FROZEN) ** k_holes * P_FROZEN ** (len(others) - k_holes))
    return np.concatenate(G), np.concatenate(Hm), np.concatenate(W)


NO_WALK = 255
@lru_cache(maxsize=None)
def _walks(start):
    G, Hm, _ = hypotheses(start); n = SIZE * SIZE; full = (1 << n) - 1
    left = sum(1 << i for i in range(n) if i % SIZE == 0); right = sum(1 << i for i in range(n) if i % SIZE == SIZE - 1)
    free = (full & ~Hm).astype(np.int32); reach = (np.int64(1) << G).astype(np.int32)   # from each goal through its frozen cells
    d = np.zeros((n, len(G)), dtype=np.uint8)
    for _ in range(n):                                                 # d[c] = the steps after which c is not yet reached
        for c in range(n): d[c] += (reach >> c) & 1 == 0
        reach = (reach | (reach << SIZE) | (reach >> SIZE) | ((reach << 1) & ~left) | ((reach >> 1) & ~right)) & free
    d[d >= n] = NO_WALK; d.setflags(write=False); return d


def walks(start, cell):
    """Per hypothesis of the start's table: the moves of the shortest route from `cell` to its goal through its frozen cells (0 at
    the goal, NO_WALK when `cell` is one of its holes), uint8. Made once per start (16 x ~2 x 10^5 bytes, ~1/10 s)."""
    return _walks(start)[cell]


def rules_text():
    return ("You are on a frozen lake of 4 x 4 cells (columns A to D from the left, rows 1 to 4 from the top). The goal is hidden in "
            "one of the cells. Holes are in the ice: each cell other than the start and the goal is a hole with chance 0.2, and a path "
            "from the start to the goal always exists. You see the cells you have been on and the four cells next to you; fog covers the "
            "rest. LEFT, DOWN, RIGHT and UP move you one cell. Stepping into a hole ends the game; reaching the goal wins it. You have "
            f"{T_ACTIONS} actions.")


SYSTEM = ("You search a frozen lake for a hidden goal. Each turn, think, then reply with exactly two lines.\n"
          "Line 1: 'Possible: goal at ' and the cells that may still hold the goal, for example 'Possible: goal at B3, D4', or "
          "'Possible: any' when no cell is excluded yet.\n"
          "Line 2: 'ACTION: LEFT', 'ACTION: DOWN', 'ACTION: RIGHT' or 'ACTION: UP'.")


class _GB:
    __slots__ = ("u", "n")
    def __init__(self, u, n): self.u, self.n = u, n
    def cls(self): return ("frozenlake", min(self.n, 4))
    def U(self): return self.u


class Belief:
    """The alive (goal, holes) hypotheses (index mask over the start's table) and the seen cells; the agent's cell and the time."""
    __slots__ = ("ep", "M", "t", "_k")
    def __init__(self, ep, M, t):
        self.ep, self.M, self.t = ep, M, t; self._k = (id(ep), t, ep.pos, ep.seen, M.tobytes())
    def __hash__(self): return hash(self._k)
    def __eq__(self, o): return self._k == o._k
    def goal_post(self, M=None):
        G, _, W = hypotheses(self.ep.start); M = self.M if M is None else M
        w = np.bincount(G[M], weights=W[M], minlength=SIZE * SIZE); return w / w.sum()
    def U(self): return float(self.goal_post().max())
    def identified(self): return self.U() > 1 - 1e-12
    def cls(self): return ("frozenlake", min(int((self.goal_post() > 1e-12).sum()), 4))
    def post(self):
        p = self.goal_post(); return {name(i): float(v) for i, v in enumerate(p) if v > 1e-12}
    def _rho(self, rel, M):
        p = self.goal_post(M); return rel.mean(_GB(float(p.max()), int((p > 1e-12).sum())))
    def look_info(self, a):
        """--stall_cap: the expected drop of the entropy of the goal's cell (nats) of move a: what the unseen cells next
        to the cell it leads to hold (frozen, a hole, the goal), as view_value groups them. 0 when they are all seen (a step into a
        wall too), for a step into a known hole, and once the goal is known."""
        ep = self.ep; nxt = ep.move(ep.pos, a)
        if ep.known_hole(nxt): return 0.0
        new = [j for j in neighbours(nxt) + [nxt] if not ep.seen >> j & 1]
        ent = lambda p: float(-(p[p > 1e-12] * np.log(p[p > 1e-12])).sum())
        h0 = ent(self.goal_post())
        if not new or h0 <= 0: return 0.0
        G, Hm, W = hypotheses(ep.start); idx = np.nonzero(self.M)[0]; key = np.zeros(len(idx), dtype=np.int64)
        for j in new: key = key * 3 + np.where(G[idx] == j, 2, (Hm[idx] >> j) & 1)   # each cell: 0 frozen, 1 hole, 2 the goal
        tot = W[idx].sum(); h1 = 0.0
        for kv in np.unique(key):
            sel = idx[key == kv]; h1 += W[sel].sum() / tot * ent(self.goal_post(self._mask(sel)))
        return max(0.0, float(h0 - h1))
    def view_value(self, rel, cell):
        """E[f(rho)] after standing on `cell`: what its unseen neighbours show (holes, the goal), grouped; None if nothing is new."""
        ep = self.ep; new = [j for j in neighbours(cell) + [cell] if not ep.seen >> j & 1]
        if not new: return None
        G, Hm, W = hypotheses(ep.start); idx = np.nonzero(self.M)[0]; w = W[idx]
        key = np.zeros(len(idx), dtype=np.int64)
        for k, j in enumerate(new):                                   # each new cell: 0 frozen, 1 hole, 2 the goal
            content = np.where(G[idx] == j, 2, (Hm[idx] >> j) & 1); key = key * 3 + content
        total = w.sum(); v = 0.0
        for kv in np.unique(key):
            sel = idx[key == kv]; v += W[sel].sum() / total * rel.f(self._rho(rel, self._mask(sel)))
        return float(v)
    def _mask(self, sel):
        M = np.zeros_like(self.M); M[sel] = True; return M
    def lookahead(self, rel, lam, T, depth, acts):
        """Q(a): the value of what move a leads to (see the module text); FL_VIEW_VALUE: the old value of a view."""
        if _tasks.FL_VIEW_VALUE: return self._view_lookahead(rel, lam, T, depth, acts)
        ep = self.ep; cost = lambda n: math.exp(-lam * n / T)
        acc = rel.mean(_GB(1.0, 1)); fail = rel.util(cost(T), 0.0)
        tab = np.array([rel.util(cost(n), acc) for n in range(T + 1)] + [fail])   # the return of reaching the goal at n; T + 1: never
        goal = ep.known_goal(self); out = {}
        idx = None
        for a in acts:
            nxt = ep.move(ep.pos, a)
            if ep.known_hole(nxt): out[a] = fail; continue
            if goal is not None:
                d = ep.safe_reach(nxt).get(goal)
                if d is not None: out[a] = float(tab[min(self.t + 1 + d, T + 1)]); continue
            if idx is None:
                G, Hm, W = hypotheses(ep.start); idx = np.nonzero(self.M)[0]
                ctx = (G, Hm, W, T, tab, {})
            out[a] = self._q(ctx, idx, ep.pos, ep.seen, self.t, a, max(1, int(depth)))
        return out
    def _leaf(self, ctx, sel, cell, t):
        """E over the hypotheses sel of the return of walking from `cell` at time t to the goal by its shortest route."""
        G, Hm, W, T, tab, by_t = ctx
        v = by_t.get(t)
        if v is None: v = by_t[t] = tab[np.minimum(t + np.arange(256), T + 1)]
        w = W[sel]; return float((w * v[walks(self.ep.start, cell)[sel]]).sum() / w.sum())
    def _q(self, ctx, sel, pos, seen, t, a, rem):
        """Move a from pos at time t, then the best of the next rem - 1 moves after each outcome, then the walk (the leaf)."""
        G, Hm, W, T, tab, _ = ctx; nxt = self.ep.move(pos, a); t1 = t + 1
        if rem <= 1: return self._leaf(ctx, sel, nxt, t1)                # (a hole at nxt: no walk, the floor; the goal: walk 0)
        cells = [nxt] + [j for j in neighbours(nxt) if not seen >> j & 1 and j != nxt]
        key = np.zeros(len(sel), dtype=np.int64)
        for j in cells: key = key * 3 + np.where(G[sel] == j, 2, (Hm[sel] >> j) & 1)   # each cell: 0 frozen, 1 hole, 2 the goal
        seen2 = seen
        for j in cells: seen2 |= 1 << j
        keys, inv = np.unique(key, return_inverse=True); w = W[sel]; tot = w.sum(); v = 0.0
        for g, kv in enumerate(keys):
            m = inv == g; s = sel[m]; p = w[m].sum() / tot; here = int(kv) // 3 ** (len(cells) - 1)   # what nxt holds
            if here == 1: v += p * tab[T + 1]
            elif here == 2: v += p * tab[min(t1, T + 1)]
            elif t1 >= T: v += p * tab[T + 1]
            else: v += p * max(self._q(ctx, s, nxt, seen2, t1, b, rem - 1) for b in MOVE)
        return float(v)
    def _view_lookahead(self, rel, lam, T, depth, acts):
        """Before the fix (FL_VIEW_VALUE): the best plan through known-safe cells that starts with move a (see the module text)."""
        ep = self.ep; cost = lambda n: math.exp(-lam * n / T); out = {}
        goal = ep.known_goal(self)
        for a in acts:
            nxt = ep.move(ep.pos, a)
            if ep.known_hole(nxt): out[a] = rel.util(cost(self.t + 1), 0.0); continue
            best = rel.none
            for cell, d in ep.safe_reach(nxt).items():
                n = self.t + 1 + d
                if n > T: continue
                if goal is not None and cell == goal: best = max(best, rel.util(cost(n), rel.mean(_GB(1.0, 1)))); continue
                v = self.view_value(rel, cell)
                if v is not None: best = max(best, rel.with_cost(cost(n), v))
            out[a] = best
        return out


# --------------------------------------------------------------------- the belief language
def read_line(text):
    m = re.match(r"^\s*possible\s*:\s*(.*?)\s*\.?\s*$", str(text), re.I | re.S)
    if not m: return None
    body = m.group(1).strip()
    if re.fullmatch(r"(?:any|all|any cell|anywhere)", body, re.I): return set(range(SIZE * SIZE))
    body = re.sub(r"^(?:the\s+)?goal\s*(?:at|in|:)?\s*", "", body, flags=re.I)
    cells = re.findall(r"\b([A-Da-d])\s*([1-4])\b", body)
    if not cells or re.sub(r"[A-Da-d]\s*[1-4]|[\s,;]|and|or", "", body, flags=re.I): return None
    return {(int(r) - 1) * SIZE + ord(c.upper()) - 65 for c, r in cells}


def set_score(post, s, miss=MISS):
    pin = sum(post.get(name(i), 0.0) for i in s); return -pin * math.log(max(1, len(s))) - miss * (1.0 - pin)


# --------------------------------------------------------------------- episodes and the trainer's contract
class Ep:
    def __init__(self, spec):
        from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
        self.spec = spec; seed = int(spec[1]); self.desc = generate_random_map(SIZE, P_FROZEN, seed)
        while (spec[2] if len(spec) > 2 else "test") != "test" and tuple(self.desc) in test_maps():
            seed = next_seed(seed); self.desc = generate_random_map(SIZE, P_FROZEN, seed)
        self.env = FrozenLakeEnv(desc=self.desc, is_slippery=False, render_mode="rgb_array"); s, _ = self.env.reset(seed=int(spec[1]))
        flat = "".join(self.desc); self.start, self.goal = flat.index("S"), flat.index("G")
        self.holes = sum(1 << i for i, ch in enumerate(flat) if ch == "H")
        self.pos = int(s); self.seen = 0; self.M = np.ones(len(hypotheses(self.start)[0]), dtype=bool)
        self.t, self.done, self.correct, self.turns_used, self.invalid, self.note, self.answer = 0, False, None, 0, 0, None, None
        self.log, self.won, self.fell = [], False, False
        self.see()
    def see(self):
        """The cell the agent is on and its four neighbours become seen; the hypotheses that disagree with them drop out."""
        G, Hm, _ = hypotheses(self.start)
        for j in [self.pos] + neighbours(self.pos):
            if self.seen >> j & 1: continue
            self.seen |= 1 << j
            if j == self.goal: self.M &= G == j
            elif self.holes >> j & 1: self.M &= (Hm >> j) & 1 == 1
            else: self.M &= (G != j) & ((Hm >> j) & 1 == 0)
    def picture(self):
        from PIL import Image
        img = np.array(self.env.render()); step = img.shape[0] // SIZE
        for i in range(SIZE * SIZE):
            if not self.seen >> i & 1:
                r, c = divmod(i, SIZE); img[r * step:(r + 1) * step, c * step:(c + 1) * step] = FOG
        return Image.fromarray(img)
    def move(self, i, a):
        r, c = divmod(i, SIZE); dr, dc = MOVE[a]; r2, c2 = r + dr, c + dc
        return r2 * SIZE + c2 if 0 <= r2 < SIZE and 0 <= c2 < SIZE else i
    def known_hole(self, i): return bool(self.seen >> i & 1) and bool(self.holes >> i & 1)
    def known_goal(self, B):
        p = B.goal_post(); i = int(p.argmax()); return i if p[i] > 1 - 1e-12 else None
    def safe_reach(self, start):
        """Moves from `start` to every cell reachable through seen cells that are not holes (the start included, at 0)."""
        from collections import deque
        dist = {start: 0}; q = deque([start])
        while q:
            i = q.popleft()
            for a in MOVE:
                j = self.move(i, a)
                if j not in dist and self.seen >> j & 1 and not self.holes >> j & 1: dist[j] = dist[i] + 1; q.append(j)
        return dist


class Domain:
    name = "frozenlake"
    images = True
    value_modes = ("accuracy", "log")

    def __init__(self, split="test", **_):
        self.split, self.T = split, T_ACTIONS; self.turns = self.T; self.lam = self.T * math.log(1 / MOVE_COST)
        self.families = ["frozenlake-hidden-goal"]

    # specs: training seeds from 10^6 up; validation 10000 + i; test 20000 + i (VAGEN's evaluation seeds are 20000 to 20127)
    def sample(self, rng): return (self.families[0], rng.randrange(10 ** 6, 2 ** 31), "train")
    def eval_specs(self, n):
        base = {"test": 20000}.get(self.split, 10000); return [(self.families[0], base + i, self.split) for i in range(n)]
    def start(self, specs): return [Ep(tuple(s)) for s in specs]
    def close(self, eps):
        for e in eps: e.env.close()
    def family_of(self, e): return e.spec[0]

    def system_text(self, e): return SYSTEM
    def user_content(self, e, note=None):
        done = ", ".join(e.log) if e.log else "none yet"
        parts = [{"type": "text", "text": f"{rules_text()}\nYour actions so far: {done}.\nThe lake now: "}, {"type": "image"},
                 {"type": "text", "text": carried(note) + f"\nActions used: {e.t} of {self.T}. Decide now."}]
        return parts, [e.picture()]
    def _history(self, e): raise NotImplementedError("the game's observations are pictures")

    def look_strings(self, e): return {a: f"ACTION: {s}" for a, s in ACTIONS.items()} if not e.done and e.t < self.T else {}
    def look_name(self, a): return ACTIONS[a].lower()
    def answer_options(self, e): return []
    def commit_string(self, y): return f"ACTION: {y}"
    def canonical_action(self, text):
        t = " ".join(str(text).strip().rstrip(".").split()).upper()
        for s in ACTIONS.values():
            if t in (f"ACTION: {s}", s, f"ACTION: MOVE {s}", f"MOVE {s}"): return f"ACTION: {s}"
        return " ".join(str(text).strip().split())
    def parse_action(self, e, text):
        t = self.canonical_action(text)
        for a, s in self.look_strings(e).items():
            if s == t: return ("look", a)
        return ("invalid", None)
    def do_look(self, e, a):
        s, r, term, trunc, _ = e.env.step(int(a)); e.pos = int(s); e.log.append(ACTIONS[int(a)].lower())
        if term:
            e.won = e.pos == e.goal; e.fell = not e.won; e.done, e.correct = True, e.won
        e.see()
    def do_commit(self, e, y): return False
    def after_turn(self, e):
        e.turns_used += 1
        if not e.done and e.turns_used >= self.turns: e.done, e.correct = True, False

    def belief(self, e): return Belief(e, e.M.copy(), e.t)
    def freeze(self, B):
        """B with its game frozen at B's history: a plan keeps B, and the own turns of a history are scored
        after the game has gone on, so the lookahead of a stated line (decision_score) must read the agent's place and what it has
        seen at the history, not at the game's end. The plan made at the history is unchanged: it ran on the live game then."""
        s = copy.copy(B.ep); s.M = B.ep.M.copy(); s.log = list(B.ep.log); return Belief(s, B.M, B.t)   # pos, seen: ints
    def label(self, e): return name(e.goal)
    def answer_posterior(self, e): return self.belief(e).post()
    def answer_prob(self, B, y): return B.post().get(str(y).strip().upper(), 0.0)
    def state_text(self, e):
        cells = sorted(self.belief(e).post(), key=lambda c: (c[1], c[0]))
        return "Possible: any" if len(cells) == SIZE * SIZE - 1 else "Possible: goal at " + ", ".join(cells)
    def belief_note(self, e): return self.state_text(e)
    def belief_score(self, B, text, miss=MISS):
        s = read_line(text); return -miss if s is None else set_score(B.post(), s, miss)
    def belief_best(self, B, miss=MISS):
        return set_score(B.post(), {(int(c[1]) - 1) * SIZE + ord(c[0]) - 65 for c in B.post()}, miss)
    def line_excess(self, B, text, exact):   # the share of the line's cells that the exact line excludes (--line_wide), both read by this reader
        s, x = read_line(text), read_line(exact); return len(s - x) / len(s) if s and x is not None else 0.0
    def line_belief(self, B, text):
        """The belief a line states, as the teacher's posterior (--line_score decision): the (goal, holes) hypotheses with a goal it
        names, those the evidence excluded too (every hypothesis of a goal cell that the evidence ruled out); None when it is not a
        belief line."""
        s = read_line(text)
        if s is None: return None
        G = hypotheses(B.ep.start)[0]; named = np.isin(G, sorted(s)); M = (B.M & named) | (named & ~np.isin(G, np.unique(G[B.M])))
        return Belief(B.ep, M, B.t) if M.any() else None
    def read_exact(self, text): return read_line(text) is not None
    def anchor_support(self, text, answers, e=None):
        s = read_line(text); return {name(i) for i in s} if s else set(answers)
    def anchor_equal(self, a, b): x, y = read_line(a), read_line(b); return x is not None and x == y
    def anchor_answer(self, anchor, answers, e=None):
        s = read_line(anchor); return name(min(s)) if s else (answers[0] if answers else "")
    def extra_metrics(self, eps):
        if not eps: return {}
        return {"won": float(np.mean([e.won for e in eps])), "fell": float(np.mean([e.fell for e in eps])),
                "actions": float(np.mean([e.t for e in eps]))}


