"""MineSweeper (POPGym Arcade, popgym_arcade.environments.minesweeper; MIT) for the RLVB (EM) trainer, with the game's own
picture, actions and ends: a board of n x n cells with m mines (easy 4 x 4 with 2, medium 6 x 6 with 6, hard 8 x 8 with 10), the
mines drawn as the game draws them (m different cells, every set equally likely); MOVE UP / DOWN / LEFT / RIGHT moves the cursor
(the orange cell) and FIRE reveals the cell under it: a mine ends the game, a safe cell shows how many of its eight neighbours are
mines. The game is won when every safe cell is revealed; it ends when a mine is revealed, after n^2 / 2 fires on revealed cells, or
after 2 n^2 actions. One addition for our method: ACTION: SUBMIT <cells> names the mine cells (columns A, B, ... from the left,
rows 1, 2, ... from the top) and ends the game, right when it names exactly the mines.

The teacher's belief is exact: the mine sets that agree with every revealed cell (easy 120 sets, medium 1,947,792, one 64-bit mask
each). Its policy pi* (with the value of Reliability.util): FIRE is worth E[u] over its outcomes, the counts it can show (a group
of mine sets has the answer accuracy U = 1 / |group| times phi) and a mine (the accuracy of knowing nothing, 1 / all sets); with
depth 2 (at most DEPTH2_MAX sets), each count's group is worth its best next step, the answer or a second reveal reached by the
shortest cursor path; a move is worth the best FIRE that a shortest cursor path starting with it reaches, with the time of its
moves; every FIRE of a plan by action 2 n^2 (popgym ends the game on a move that is action 2 n^2, not on a FIRE, and the
answer comes on the next turn; such a move is worth the floor; plearn.tasks.PLAN_PAST_LIMIT, --plan_past_limit 1: no limit, as
before the fix); the answer now is worth u(c(t), rho(b)). The belief line names the hidden cells the evidence decides, 'Possible: mines at B2, D5; safe at C3' ('Possible:
any' when it decides none), scored as the set of mine sets it allows. Hard is a test only (teacher=False: no belief).
"""
from __future__ import annotations
import itertools, math, os, re
from functools import lru_cache
import numpy as np
from plearn.api.episode import carried
import plearn.tasks as _tasks                                        # PLAN_PAST_LIMIT

LEVELS = {"easy": (4, 2), "medium": (6, 6), "hard": (8, 10)}
UP, DOWN, LEFT, RIGHT, FIRE = range(5)
ACTIONS = {UP: "MOVE UP", DOWN: "MOVE DOWN", LEFT: "MOVE LEFT", RIGHT: "MOVE RIGHT", FIRE: "FIRE"}
STEP = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
MOVE_COST, MISS, LIST_MAX = 0.95, 50.0, 64       # a move shows nothing: its cost (0.05 nats) must be visible at the policy's tau
DEPTH2_MAX = 50_000                              # two reveals ahead while at most this many mine sets remain
os.environ.setdefault("JAX_PLATFORMS", "cpu")                       # the game runs on the CPU, beside the model's GPUs


def cell_name(x, y): return f"{chr(65 + x)}{y + 1}"


def bit(n, x, y): return y * n + x                                  # cells in reading order: row by row from the top left


@lru_cache(maxsize=None)
def tables(n, m):
    """Every mine set as a 64-bit mask, and each cell's mask of its eight neighbours."""
    combos = np.array(list(itertools.combinations(range(n * n), m)), dtype=np.uint64)
    sets = np.bitwise_or.reduce(np.left_shift(np.uint64(1), combos), axis=1)
    nb = np.zeros(n * n, dtype=np.uint64)
    for y in range(n):
        for x in range(n):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if (dx or dy) and 0 <= x + dx < n and 0 <= y + dy < n: nb[bit(n, x, y)] |= np.uint64(1) << np.uint64(bit(n, x + dx, y + dy))
    return sets, nb


@lru_cache(maxsize=None)
def game(level):
    from popgym_arcade.environments.minesweeper import MineSweeperEasy, MineSweeperMedium, MineSweeperHard
    import jax
    env = {"easy": MineSweeperEasy, "medium": MineSweeperMedium, "hard": MineSweeperHard}[level](partial_obs=False, obs_size=256)
    return env, jax.jit(env.reset), jax.jit(env.step)


def rules_text(n, m):
    return (f"You are playing Minesweeper on a board of {n} x {n} cells with {m} mines; every set of {m} cells is equally likely. "
            f"The columns are A to {chr(64 + n)} from the left, the rows 1 to {n} from the top. The orange cell is the cursor. "
            "MOVE UP, MOVE DOWN, MOVE LEFT and MOVE RIGHT move the cursor by one cell; FIRE reveals the cell under it. A revealed mine "
            "ends the game. A revealed safe cell shows the number of mines among its eight neighbours. The game is won when every "
            f"safe cell is revealed; it ends when a mine is revealed, after {n * n // 2} fires on revealed cells, or after {2 * n * n} "
            "actions. You may answer at any time with the mine cells: SUBMIT and the cells, for example SUBMIT B2 D5.")


def system_text(n):
    return ("You find the mines of a Minesweeper board from what the revealed cells show. Each turn, think, then reply with exactly "
            "two lines.\nLine 1: 'Possible: ' and the hidden cells the evidence decides, for example 'Possible: mines at B2; safe at "
            "C1, C3', or 'Possible: any' when it decides none yet.\nLine 2: 'ACTION: MOVE UP', 'ACTION: MOVE DOWN', 'ACTION: MOVE "
            "LEFT', 'ACTION: MOVE RIGHT', 'ACTION: FIRE', or 'ACTION: SUBMIT' and the mine cells, for example 'ACTION: SUBMIT B2 D5'. "
            "You have one chance to answer: submit when the evidence identifies the mines, and not before.")


class _GB:
    """A group of k mine sets for the reliability table: its class and U."""
    __slots__ = ("k",)
    def __init__(self, k): self.k = k
    def cls(self): return ("minesweeper", min(self.k, 4))
    def U(self): return 1.0 / self.k


class Belief:
    """The mine sets A (masks) that agree with every revealed cell, the cursor, the actions t and the revealed cells."""
    __slots__ = ("n", "m", "A", "x", "y", "t", "revealed", "dead", "N", "_k", "_rv")
    def __init__(self, n, m, A, x, y, t, revealed, dead):
        self.n, self.m, self.A, self.x, self.y, self.t, self.revealed, self.dead = n, m, A, x, y, t, revealed, dead
        self.N = math.comb(n * n, m); self._k = (n, x, y, t, revealed, dead, A.tobytes()); self._rv = None
    def __hash__(self): return hash(self._k)
    def __eq__(self, o): return self._k == o._k
    def k(self): return int(len(self.A))
    def U(self): return 0.0 if self.dead else 1.0 / self.k()
    def identified(self): return not self.dead and self.k() == 1
    def cls(self): return ("minesweeper", min(self.k(), 4))
    def look_info(self, a):
        """--stall_cap: the expected drop of the answer entropy (nats) of action a. A cursor move, and FIRE on a revealed
        cell, show nothing (0). FIRE on a hidden cell: the counts it can show split the mine sets; a mine ends the game and counts
        as showing nothing (the entropy stays log k)."""
        if self.dead or a != FIRE: return 0.0
        i = bit(self.n, self.x, self.y); k = self.k()
        if self.revealed >> i & 1 or k <= 1: return 0.0
        _, nb = tables(self.n, self.m); safe = self.A[((self.A >> np.uint64(i)) & np.uint64(1)) == 0]
        if not len(safe): return 0.0
        cnt = np.bincount(np.bitwise_count(safe & nb[i]).astype(np.int64)); cnt = cnt[cnt > 0]
        return max(0.0, float(len(safe) / k * math.log(k) - (cnt / k * np.log(cnt)).sum()))
    def certain(self):
        """The hidden cells that are mines in every set, and those that are safe in every set."""
        allm = np.bitwise_and.reduce(self.A); anym = np.bitwise_or.reduce(self.A); hidden = (1 << self.n * self.n) - 1 & ~self.revealed
        return int(allm) & hidden, ~int(anym) & hidden
    def reveal_values(self, rel, lam=None, T=None, depth=1, limit=None):
        """{hidden cell: its value relative to the time after its reveal} (values_one, or values_two at depth 2). limit: the
        game's last action (2 n^2); the second reveal must come by then, the first coming by the shortest cursor path from now."""
        two = depth >= 2 and lam is not None and self.k() <= DEPTH2_MAX
        key = (two, lam, T, limit) if two else (two,)                     # (the values depend on the time left)
        if self._rv is None or self._rv[0] != key:
            left = None if limit is None else {i: limit - (self.t + 1 + abs(i % self.n - self.x) + abs(i // self.n - self.y)) for i in range(self.n * self.n)}
            self._rv = (key, values_two(self.A, self.revealed, self.n, self.m, rel, self.N, lam, T, left) if two
                        else values_one(self.A, self.revealed, self.n, self.m, rel, self.N))
        return self._rv[1]
    def lookahead(self, rel, lam, T, depth, acts):
        if self.dead or not acts: return {a: rel.none for a in acts}
        # the game's limit (popgym, measured): a move as action 2 n^2 ends the game (no answer), a FIRE as action 2 n^2 does
        # not, and the answer comes on the next turn; so a plan's FIRE must come by action 2 n^2 (PLAN_PAST_LIMIT: no limit, as before)
        limit = None if _tasks.PLAN_PAST_LIMIT else 2 * self.n * self.n
        c = lambda t: math.exp(-lam * t / T); E = self.reveal_values(rel, lam, T, depth, limit); here = bit(self.n, self.x, self.y); out = {}
        for a in acts:
            if a == FIRE:
                out[a] = rel.with_cost(c(self.t + 1), E[here]) if here in E else rel.none
                continue
            nx, ny = self.x + STEP[a][0], self.y + STEP[a][1]
            if not (0 <= nx < self.n and 0 <= ny < self.n): out[a] = rel.none; continue
            if limit is not None and self.t + 1 >= limit: out[a] = rel.util(c(self.t + 1), 0.0); continue   # the game ends, lost
            best = rel.none
            for i, v in E.items():                                   # the best FIRE a shortest path that starts with this move reaches
                cx, cy = i % self.n, i // self.n; d = abs(cx - nx) + abs(cy - ny)
                if limit is not None and self.t + 1 + d + 1 > limit: continue
                if abs(cx - self.x) + abs(cy - self.y) == d + 1: best = max(best, rel.with_cost(c(self.t + 1 + d + 1), v))
            out[a] = best
        return out


def values_one(A, revealed, n, m, rel, N):
    """E[f(rho)] of revealing each hidden cell: the counts it can show (groups of sets) and a mine (knowing nothing)."""
    _, nb = tables(n, m); k = len(A); dead = rel.f(rel.mean(_GB(N))); out = {}
    for i in range(n * n):
        if revealed >> i & 1: continue
        safe = A[((A >> np.uint64(i)) & np.uint64(1)) == 0]; v = (k - len(safe)) / k * dead
        if len(safe):
            cnt = np.bincount(np.bitwise_count(safe & nb[i]).astype(np.int64)); cnt = cnt[cnt > 0]
            v += float(sum(c / k * rel.f(rel.mean(_GB(int(c)))) for c in cnt))
        out[i] = v
    return out


def values_two(A, revealed, n, m, rel, N, lam, T, left=None):
    """Two reveals ahead: each count's group of sets is worth the better of the answer then and the best second reveal (values_one
    of the group), reached by the shortest cursor path from the first cell; a mine is worth knowing nothing. left: {first
    cell: the actions the game still allows after its reveal}; a second reveal must fit in them (None: no limit, as before)."""
    _, nb = tables(n, m); k = len(A); dead = rel.f(rel.mean(_GB(N))); out = {}
    for i in range(n * n):
        if revealed >> i & 1: continue
        safe = A[((A >> np.uint64(i)) & np.uint64(1)) == 0]; v = (k - len(safe)) / k * dead
        if len(safe):
            counts = np.bitwise_count(safe & nb[i]); rev2 = revealed | (1 << i); xi, yi = i % n, i // n
            for cv in np.unique(counts):
                group = safe[counts == cv]; best = rel.f(rel.mean(_GB(len(group))))        # the answer after the first reveal
                if len(group) > 1:
                    for j, vj in values_one(group, rev2, n, m, rel, N).items():
                        d = abs(j % n - xi) + abs(j // n - yi)
                        if left is None or d + 1 <= left[i]: best = max(best, rel.with_cost(math.exp(-lam * (d + 1) / T), vj))
                v += len(group) / k * best
        out[i] = v
    return out


# --------------------------------------------------------------------- the belief language
def read_line(text, n):
    """(mine cells, safe cells) as bit masks that a line 'Possible: ...' states, or None when it is not in the language."""
    mt = re.match(r"^\s*possible\s*:\s*(.*?)\s*\.?\s*$", str(text), re.I | re.S)
    if not mt: return None
    body = mt.group(1).strip()
    if re.fullmatch(r"(?:any|all|any cells?|nothing decided)", body, re.I): return 0, 0
    mines = safes = 0; seen_part = False
    for part in re.split(r";", body):
        part = part.strip(" .")
        if not part: continue
        p = re.fullmatch(r"(mines?|safe)\s*(?:at|:)?\s*(.*)", part, re.I)
        if not p: return None
        cells = re.findall(r"\b([A-Za-z])\s*([1-9])\b", p.group(2))
        if not cells and not re.fullmatch(r"\s*(?:none)?\s*", p.group(2), re.I): return None
        mask = 0
        for col, row in cells:
            x, y = ord(col.upper()) - 65, int(row) - 1
            if not (0 <= x < n and 0 <= y < n): return None
            mask |= 1 << bit(n, x, y)
        if p.group(1).lower().startswith("mine"): mines |= mask
        else: safes |= mask
        seen_part = True
    return (mines, safes) if seen_part else None


def names(n, mask): return [cell_name(i % n, i // n) for i in range(n * n) if mask >> i & 1]


def line_text(B):
    mines, safes = B.certain()
    parts = ([f"mines at {', '.join(names(B.n, mines))}"] if mines else []) + ([f"safe at {', '.join(names(B.n, safes))}"] if safes else [])
    return "Possible: " + ("; ".join(parts) if parts else "any")


def set_score(B, stated, miss=MISS):
    """S = -b(S) log|S| - miss b(outside S): S = the mine sets with the stated mines, without the stated safe cells, and without
    mines on revealed cells."""
    if stated is None: return -miss
    mines, safes = stated; hidden = (1 << B.n * B.n) - 1 & ~B.revealed
    if mines & ~hidden: return -miss                                   # a mine on a revealed cell: no set allows it
    safes &= hidden; free = bin(hidden & ~mines & ~safes).count("1"); left = B.m - bin(mines).count("1")
    size = math.comb(free, left) if 0 <= left <= free else 0
    if size == 0: return -miss
    ok = ((B.A & np.uint64(mines)) == np.uint64(mines)) & ((B.A & np.uint64(safes)) == 0)
    pin = float(ok.sum()) / B.k(); return -pin * math.log(size) - miss * (1.0 - pin)


def set_excess(B, stated, exact):
    """--line_wide: the share of the mine sets a line allows (set_score's S) that the exact line's sets exclude: both are the
    sets with the stated mines, without the stated safe cells and without mines on revealed cells, so the sets of both lines are
    those with the mines of both and the safe cells of both. 0 when a line is not read or allows no set."""
    if stated is None or exact is None: return 0.0
    hidden = (1 << B.n * B.n) - 1 & ~B.revealed
    def count(mines, safes):
        if mines & ~hidden or mines & safes: return 0
        free = bin(hidden & ~mines & ~safes).count("1"); left = B.m - bin(mines).count("1")
        return math.comb(free, left) if 0 <= left <= free else 0
    n = count(stated[0], stated[1] & hidden)
    return 1.0 - count(stated[0] | exact[0], (stated[1] | exact[1]) & hidden) / n if n else 0.0


# --------------------------------------------------------------------- episodes and the trainer's contract
class Ep:
    def __init__(self, spec, level, state, obs):
        self.spec, self.level = spec, level; self.n, self.m = LEVELS[level]
        self.state, self.pictures, self.log = state, [obs], []
        self.t, self.done, self.correct, self.turns_used, self.invalid, self.note, self.answer = 0, False, None, 0, 0, None, None
        self.revealed, self.dead, self.won = 0, False, False
        self.truth = 0
        mg = np.array(state.mine_grid)
        for x in range(self.n):
            for y in range(self.n):
                if mg[x, y] == 1: self.truth |= 1 << bit(self.n, x, y)
        self.A = None                                                  # the teacher's sets (start: all)


class Domain:
    name = "minesweeper"
    images = True
    value_modes = ("accuracy", "log")                                  # Belief.lookahead computes both (Reliability.util)

    def __init__(self, levels=None, level="medium", split="test", teacher=True, **_):
        """levels: the boards of this teacher (a list; the registry's tasks join theirs), or one level."""
        self.levels = list(dict.fromkeys(levels or [level])); assert all(l in LEVELS for l in self.levels), self.levels
        self.split = split; self.teacher = bool(teacher) and "hard" not in self.levels
        self.T = max(2 * LEVELS[l][0] ** 2 for l in self.levels); self.turns = self.T + 1
        self.lam = self.T * math.log(1 / MOVE_COST)                    # c(t) = MOVE_COST ** t: lam / T is the same at every level
        self.families = [f"minesweeper-{l}" for l in self.levels]

    # specs: training seeds from 10^6 up; validation 10000 + i; test 20000 + i
    def sample(self, rng): return (rng.choice(self.families), rng.randrange(10 ** 6, 2 ** 31), "train")
    def eval_specs(self, n):
        base = {"test": 20000}.get(self.split, 10000)
        return [(f, base + i, self.split) for f in self.families for i in range(n)]
    def start(self, specs):
        import jax
        out = []
        for s in specs:
            level = s[0].split("-", 1)[1]; env, reset, _ = game(level)
            obs, st = reset(jax.random.PRNGKey(int(s[1])), env.default_params)
            e = Ep(tuple(s), level, st, np.array(obs)); e.prng = jax.random.PRNGKey(int(s[1]) + 1)   # not e.key: the trainer's scene key
            if self.teacher: e.A = tables(e.n, e.m)[0]
            out.append(e)
        return out
    def close(self, eps): pass
    def family_of(self, e): return e.spec[0]

    # observations: the game's picture now (it shows every revealed cell), and the actions so far as text
    def system_text(self, e): return system_text(e.n)
    def user_content(self, e, note=None):
        from PIL import Image
        done = ", ".join(e.log) if e.log else "none yet"
        text = (f"{rules_text(e.n, e.m)}\nYour actions so far: {done}.\nThe board now: ")
        tail = carried(note) + f"\nActions used: {e.t} of {2 * e.n * e.n}. Decide now."
        return [{"type": "text", "text": text}, {"type": "image"}, {"type": "text", "text": tail}], [Image.fromarray(e.pictures[-1])]
    def _history(self, e): raise NotImplementedError("the game's observations are pictures")

    # actions
    def look_strings(self, e):
        if e.done or e.t >= 2 * e.n * e.n: return {}
        x, y = int(e.state.action_x), int(e.state.action_y); out = {}
        for a in (UP, DOWN, LEFT, RIGHT):
            if 0 <= x + STEP[a][0] < e.n and 0 <= y + STEP[a][1] < e.n: out[a] = f"ACTION: {ACTIONS[a]}"
        out[FIRE] = "ACTION: FIRE"
        return out
    def look_name(self, a): return ACTIONS[a].lower()
    def _cells(self, text, n):
        cells = re.findall(r"\b([A-Za-z])\s*([1-9])\b", text); mask = 0
        for col, row in cells:
            x, y = ord(col.upper()) - 65, int(row) - 1
            if not (0 <= x < n and 0 <= y < n): return None
            mask |= 1 << bit(n, x, y)
        return mask if cells else None
    def answer_text(self, mask, n): return " ".join(names(n, mask))
    def answer_options(self, e):
        if not self.teacher: return []
        return [self.answer_text(int(a), e.n) for a in e.A[:LIST_MAX]]
    def commit_string(self, y): return f"ACTION: SUBMIT {y}"
    def canonical_action(self, text, n=8):
        """An action line in the teacher's form: the move or FIRE, or SUBMIT with the cells in reading order."""
        t = " ".join(str(text).strip().rstrip(".").split()); u = t.upper()
        for a, s in ACTIONS.items():
            if u in (f"ACTION: {s}", s): return f"ACTION: {s}"
        if "SUBMIT" in u:
            mask = self._cells(t[u.index("SUBMIT") + len("SUBMIT"):], n)
            if mask is not None: return f"ACTION: SUBMIT {self.answer_text(mask, n)}"
        return t
    def parse_action(self, e, text):
        t = self.canonical_action(text, e.n)
        if t.startswith("ACTION: SUBMIT "):
            mask = self._cells(t[len("ACTION: SUBMIT "):], e.n); return ("answer", self.answer_text(mask, e.n)) if mask is not None else ("invalid", None)
        looks = self.look_strings(e)
        for a, s in looks.items():
            if s == t: return ("look", a)
        return ("invalid", None)
    def do_look(self, e, a):
        import jax
        env, _, step = game(e.level); x, y = int(e.state.action_x), int(e.state.action_y)
        e.prng, k = jax.random.split(e.prng); obs, st, r, done, info = step(k, e.state, int(a), env.default_params)
        e.state = st; e.pictures.append(np.array(obs)); e.log.append(ACTIONS[int(a)].lower())
        if a == FIRE:
            i = bit(e.n, x, y)
            if e.truth >> i & 1: e.dead = True
            elif not e.revealed >> i & 1:
                e.revealed |= 1 << i
                if e.A is not None:
                    _, nb = tables(e.n, e.m); count = int(np.array(st.neighbor_grid)[x, y])
                    e.A = e.A[((e.A >> np.uint64(i)) & np.uint64(1) == 0) & (np.bitwise_count(e.A & nb[i]) == count)]
        if bool(done):                                                  # the game's own end: a mine, every safe cell, or its limits
            e.won = not e.dead and bin(e.revealed).count("1") == e.n * e.n - e.m
            e.done, e.correct = True, e.won
    def do_commit(self, e, y):
        e.answer = y; mask = self._cells(str(y), e.n); return mask is not None and mask == e.truth
    def after_turn(self, e):
        e.turns_used += 1
        if not e.done and e.turns_used >= self.turns: e.done, e.correct = True, False

    # belief and teacher
    def belief(self, e):
        return Belief(e.n, e.m, e.A, int(e.state.action_x), int(e.state.action_y), e.t, e.revealed, e.dead)
    def label(self, e): return self.answer_text(e.truth, e.n)
    def answer_posterior(self, e):
        if e.A is None or e.dead: return {}
        k = len(e.A); return {self.answer_text(int(a), e.n): 1.0 / k for a in e.A[:LIST_MAX]}
    def answer_prob(self, B, y):
        mask = self._cells(str(y), B.n)
        return 1.0 / B.k() if mask is not None and not B.dead and bool((B.A == np.uint64(mask)).any()) else 0.0
    def state_text(self, e): return line_text(self.belief(e))
    def belief_note(self, e): return self.state_text(e)
    def belief_score(self, B, text, miss=MISS): return set_score(B, read_line(text, B.n), miss)
    def belief_best(self, B, miss=MISS):
        mines, safes = B.certain(); return set_score(B, (mines, safes), miss)
    def line_excess(self, B, text, exact): return set_excess(B, read_line(text, B.n), read_line(exact, B.n))
    def line_belief(self, B, text):
        """The belief a line states, as the teacher's posterior (--line_score decision): the mine sets the evidence allows that
        have its mines and none of its safe cells (the sets the evidence excluded are too many to list, so a line that leaves a
        decided cell open states the evidence's own sets); None when it is not a belief line or no allowed set fits it."""
        st = read_line(text, B.n)
        if st is None or B.dead: return None
        mines, safes = np.uint64(st[0]), np.uint64(st[1]); A = B.A[((B.A & mines) == mines) & ((B.A & safes) == 0)]
        return Belief(B.n, B.m, A, B.x, B.y, B.t, B.revealed, B.dead) if len(A) else None
    def read_exact(self, text): return read_line(text, 8) is not None
    def anchor_support(self, text, answers, e=None):
        n = e.n if e is not None else 8; st = read_line(text, n)
        if st is None: return set(answers)
        out = set()
        for y in answers:
            mask = self._cells(y, n)
            if mask is not None and mask & st[0] == st[0] and not mask & st[1]: out.add(y)
        return out
    def anchor_equal(self, a, b):
        x, y = read_line(a, 8), read_line(b, 8); return x is not None and x == y
    def anchor_answer(self, anchor, answers, e=None):
        s = self.anchor_support(anchor, answers, e); return sorted(s)[0] if s else (answers[0] if answers else "")
    def extra_metrics(self, eps):
        if not eps: return {}
        return {"won": float(np.mean([bool(e.won) for e in eps])), "died": float(np.mean([bool(e.dead) for e in eps])),
                "actions": float(np.mean([e.t for e in eps]))}


