"""MiniGrid KeyCorridor (Farama MiniGrid 3.2, minigrid.envs.keycorridor; Apache-2.0) for the RLVB (EM) trainer, with the
game's own layout, view and actions: three columns of rooms (the left rooms, a hallway in the middle, the right rooms) joined by
closed doors; a key lies in one of the left rooms (its row drawn uniformly, its cell uniformly inside the room, as the game draws
it); the agent sees the 7 x 7 cells in front of it, and walls and closed doors hide what is behind them; its actions are MiniGrid's
own: turn left, turn right, forward, and toggle (it opens the closed door in front). Our task: find the room of the key, and answer
with ACTION: SUBMIT <row> (the left room in that row, 1 = the top). The game's own mission (pick up the ball behind the locked door)
is not the task.

The teacher's belief is exact: the places (row, cell) of the key that no view has shown empty, all equally likely (the doors are
drawn after the key, and the number of draws before them does not depend on its place). Its policy pi*: an action is worth the best
pose (cell and direction) that a shortest plan of turns, steps and door openings starting with it reaches, with the view of that
pose: E[u] over 'the key is in view' (its room is then known) and 'it is not' (those places drop out), with the time of the plan
(Reliability.util); the answer now is worth u(c(t), rho(b)). The belief line names the rows still possible, 'Possible: rows 1, 3'
('Possible: all' before any evidence), scored as the set of rooms it allows.
"""
from __future__ import annotations
import math, re
from collections import deque
import numpy as np
from plearn.api.episode import carried

LEFT, RIGHT, FORWARD, TOGGLE = 0, 1, 2, 5                           # MiniGrid's action numbers
ACTIONS = {LEFT: "TURN LEFT", RIGHT: "TURN RIGHT", FORWARD: "FORWARD", TOGGLE: "TOGGLE"}
DIR = ((1, 0), (0, 1), (-1, 0), (0, -1))                            # MiniGrid's directions: 0 east, 1 south, 2 west, 3 north
MOVE_COST, MISS, HORIZON, VIEWS = 0.98, 50.0, 40, 32              # 0.02 nats an action; plans of at most HORIZON actions
LAYOUTS = {"S4R3": (4, 3), "S5R3": (5, 3), "S6R3": (6, 3)}


def make_env(layout, seed):
    from minigrid.envs.keycorridor import KeyCorridorEnv
    s, r = LAYOUTS[layout]
    env = KeyCorridorEnv(num_rows=r, room_size=s, render_mode=None); env.reset(seed=int(seed)); return env


def visible(env, pos, d, opened=()):
    """The cells (x, y) the agent would see from pos facing d, with the doors at `opened` open: MiniGrid's own view (gen_obs_grid)."""
    keep = (env.agent_pos, env.agent_dir); doors = []
    for p in opened:
        door = env.grid.get(*p)
        if door is not None and door.type == "door" and not door.is_open: door.is_open = True; doors.append(door)
    try:
        env.agent_pos, env.agent_dir = tuple(pos), int(d)
        _, vis = env.gen_obs_grid(); out = set()
        topX, topY, botX, botY = env.get_view_exts()
        for x in range(max(0, topX), min(env.width, botX)):
            for y in range(max(0, topY), min(env.height, botY)):
                vx, vy = env.get_view_coords(x, y)
                if 0 <= vx < env.agent_view_size and 0 <= vy < env.agent_view_size and vis[vx, vy]: out.add((x, y))
        return out
    finally:
        env.agent_pos, env.agent_dir = keep
        for door in doors: door.is_open = False


def rules_text(rows):
    return (f"You are in a MiniGrid building with three columns of rooms: {rows} rooms on the left, a hallway in the middle and "
            f"{rows} rooms on the right, joined by doors. A key lies in one of the left rooms; every left room, and every cell in it, "
            "is equally likely. You see the 7 x 7 cells in front of you; walls and closed doors hide what is behind them. TURN LEFT and "
            "TURN RIGHT turn you, FORWARD moves you one cell, TOGGLE opens the closed door in front of you. Find the room of the key and "
            f"answer with SUBMIT and its row (1 = the top, {rows} = the bottom), for example SUBMIT 2.")


SYSTEM = ("You find which room holds a key from what you see. Each turn, think, then reply with exactly two lines.\n"
          "Line 1: 'Possible: ' and the rows of the left rooms that may still hold the key, for example 'Possible: rows 1, 3', or "
          "'Possible: all' before any evidence.\n"
          "Line 2: 'ACTION: TURN LEFT', 'ACTION: TURN RIGHT', 'ACTION: FORWARD', 'ACTION: TOGGLE', or 'ACTION: SUBMIT <row>' to answer. "
          "You have one chance to answer: submit when the evidence identifies the room, and not before.")


class _GB:
    __slots__ = ("u", "n")
    def __init__(self, u, n): self.u, self.n = u, n
    def cls(self): return ("minigrid", min(self.n, 4))
    def U(self): return self.u


def answer_counts(rows, H, M):
    c = np.zeros(rows); np.add.at(c, H[M, 0], 1); return c


class Belief:
    """The key's places H (row, x, y) with the mask M of those still possible; the agent's pose, the doors it opened, the time."""
    __slots__ = ("ep", "M", "t", "_k")
    def __init__(self, ep, M, t):
        self.ep, self.M, self.t = ep, M, t; self._k = (id(ep), t, M.tobytes(), ep.pos, ep.dir, frozenset(ep.opened))
    def __hash__(self): return hash(self._k)
    def __eq__(self, o): return self._k == o._k
    def counts(self): return answer_counts(self.ep.rows, self.ep.H, self.M)
    def U(self): c = self.counts(); return float(c.max() / c.sum())
    def identified(self): return int((self.counts() > 0).sum()) == 1
    def cls(self): return ("minigrid", min(int((self.counts() > 0).sum()), 4))
    def post(self):
        c = self.counts(); return {str(j + 1): float(v / c.sum()) for j, v in enumerate(c) if v}
    def view_value(self, rel, cells):
        """E[f(rho)] after a view of `cells`: the key in view (its room known), or not (the places in view drop out)."""
        ep = self.ep; alive = np.nonzero(self.M)[0]; k = len(alive)
        inview = np.array([(int(ep.H[i, 1]), int(ep.H[i, 2])) in cells for i in alive], dtype=bool)
        if not inview.any(): return None                             # this view shows nothing new about the key
        v = inview.sum() / k * rel.f(rel.mean(_GB(1.0, 1)))
        rest = alive[~inview]
        if len(rest):
            c = np.bincount(ep.H[rest, 0], minlength=ep.rows); v += len(rest) / k * rel.f(rel.mean(_GB(c.max() / c.sum(), int((c > 0).sum()))))
        return float(v)
    def lookahead(self, rel, lam, T, depth, acts):
        """Q(a): the best pose a shortest plan starting with a reaches (turns, steps, door openings), worth its view's value."""
        ep = self.ep
        if not acts: return {}
        cost = lambda n: math.exp(-lam * n / T); out = {}
        for a in acts:
            nxt = ep.step_plan((ep.pos, ep.dir, frozenset(ep.opened)), a)
            if nxt is None: out[a] = rel.none; continue
            best = rel.none
            for state, d in ep.reach(nxt).items():
                v = ep.pose_value(self, rel, state)
                if v is not None: best = max(best, rel.with_cost(cost(self.t + 1 + d), v))
            out[a] = best
        return out


# --------------------------------------------------------------------- the belief language
def read_line(text, rows):
    m = re.match(r"^\s*possible\s*:\s*(.*?)\s*\.?\s*$", str(text), re.I | re.S)
    if not m: return None
    body = m.group(1).strip()
    if re.fullmatch(r"(?:all|any|all rows|any row)", body, re.I): return set(range(1, rows + 1))
    if not re.fullmatch(r"(?:rows?|rooms?)?\s*[\d\s,and]+", body, re.I): return None
    s = {int(d) for d in re.findall(r"\d+", body)}
    return s if s and all(1 <= v <= rows for v in s) else None


def set_score(post, s, miss=MISS):
    pin = sum(post.get(str(a), 0.0) for a in s); return -pin * math.log(max(1, len(s))) - miss * (1.0 - pin)


# --------------------------------------------------------------------- episodes and the trainer's contract
class Ep:
    def __init__(self, spec, layout, T):
        self.spec, self.layout, self.T = spec, layout, T; self.rows = LAYOUTS[layout][1]
        self.env = env = make_env(layout, spec[1]); key = None
        H = []
        for j in range(self.rows):
            room = env.get_room(0, j); (x0, y0), (w, h) = room.top, room.size
            for x in range(x0 + 1, x0 + w - 1):
                for y in range(y0 + 1, y0 + h - 1):
                    H.append((j, x, y)); cell = env.grid.get(x, y)
                    if cell is not None and cell.type == "key": key = (j, x, y)
        self.H = np.array(H, dtype=np.int64); self.truth = key; self.M = np.ones(len(H), dtype=bool)
        self.places = {(x, y) for _, x, y in H}; self.identified_key = None
        self.pos, self.dir, self.opened = tuple(int(v) for v in env.agent_pos), int(env.agent_dir), set()
        self.t, self.done, self.correct, self.turns_used, self.invalid, self.note, self.answer = 0, False, None, 0, 0, None, None
        self.pictures, self.log, self._vis, self._reach = [], [], {}, {}
        self.see()
    def picture(self):
        from PIL import Image
        return Image.fromarray(self.env.get_frame(highlight=False, tile_size=32, agent_pov=True))
    def see(self):
        """The view now: the places of the key it shows empty drop out; the picture is kept."""
        cells = visible(self.env, self.pos, self.dir)
        for i in range(len(self.H)):
            if self.M[i]:
                j, x, y = self.H[i]
                if (int(x), int(y)) in cells and (j, int(x), int(y)) != self.truth: self.M[i] = False
        if self.truth is not None and (self.truth[1], self.truth[2]) in cells:
            self.M[:] = [tuple(h) == self.truth for h in self.H.tolist()]; self.identified_key = (self.truth[1], self.truth[2])
        self.pictures.append(self.picture())
    # planning: states (pos, dir, opened doors), MiniGrid's rules for the four actions
    def passable(self, p, opened):
        """MiniGrid's rule, except that a place the key may hold counts as free until the key is seen there: the teacher plans
        with what the evidence allows, not with the true key."""
        if p in self.places: return not (self.identified_key is not None and p == self.identified_key)
        cell = self.env.grid.get(*p)
        if cell is None: return True
        if cell.type == "door": return cell.is_open or p in opened
        return cell.can_overlap()
    def step_plan(self, state, a):
        (x, y), d, opened = state
        if a == LEFT: return ((x, y), (d - 1) % 4, opened)
        if a == RIGHT: return ((x, y), (d + 1) % 4, opened)
        fx, fy = x + DIR[d][0], y + DIR[d][1]
        if not (0 <= fx < self.env.width and 0 <= fy < self.env.height): return None
        if a == FORWARD: return ((fx, fy), d, opened) if self.passable((fx, fy), opened) else None
        if a == TOGGLE:
            cell = self.env.grid.get(fx, fy)
            if cell is not None and cell.type == "door" and not cell.is_locked and not cell.is_open and (fx, fy) not in opened:
                return ((x, y), d, opened | {(fx, fy)})
            return None
        return None
    def reach(self, start):
        """Shortest numbers of actions from start to every state within HORIZON (cached for the episode's static grid)."""
        if start not in self._reach:
            dist = {start: 0}; q = deque([start])
            while q:
                s = q.popleft()
                if dist[s] >= HORIZON: continue
                for a in (LEFT, RIGHT, FORWARD, TOGGLE):
                    n = self.step_plan(s, a)
                    if n is not None and n not in dist: dist[n] = dist[s] + 1; q.append(n)
            self._reach[start] = dist
        return self._reach[start]
    def pose_value(self, B, rel, state):
        pos, d, opened = state; key = (pos, d, opened)
        if key not in self._vis: self._vis[key] = frozenset(visible(self.env, pos, d, opened))
        return B.view_value(rel, self._vis[key])


class Domain:
    name = "minigrid"
    images = True
    value_modes = ("accuracy", "log")

    def __init__(self, levels=None, split="test", T=64, **_):
        self.levels = list(dict.fromkeys(levels or ["S4R3"])); assert all(l in LAYOUTS for l in self.levels), self.levels
        self.split, self.T = split, int(T); self.turns = self.T + 1; self.lam = self.T * math.log(1 / MOVE_COST)
        self.families = [f"keycorridor-{l}" for l in self.levels]

    def sample(self, rng): return (rng.choice(self.families), rng.randrange(10 ** 6, 2 ** 31), "train")
    def eval_specs(self, n):
        base = {"test": 20000}.get(self.split, 10000)
        return [(f, base + i, self.split) for f in self.families for i in range(n)]
    def start(self, specs): return [Ep(tuple(s), s[0].split("-", 1)[1], self.T) for s in specs]
    def close(self, eps):
        for e in eps: e.env.close()
    def family_of(self, e): return e.spec[0]

    def system_text(self, e): return SYSTEM
    def user_content(self, e, note=None):
        """The rules, then the views in order (the start view and the view after each action, the newest VIEWS of them), the belief
        line of the last turn, and the actions used."""
        parts, images = [{"type": "text", "text": rules_text(e.rows)}], []
        first = max(0, len(e.pictures) - VIEWS)
        if first: parts.append({"type": "text", "text": f"\n(The first {first} views are no longer shown.)"})
        for k in range(first, len(e.pictures)):
            label = "\nThe start view: " if k == 0 else f"\nAfter action {k} ({e.log[k - 1]}): "
            parts += [{"type": "text", "text": label}, {"type": "image"}]; images.append(e.pictures[k])
        parts.append({"type": "text", "text": carried(note) + f"\nActions used: {e.t} of {self.T}. Decide now."})
        return parts, images
    def _history(self, e): raise NotImplementedError("the game's observations are pictures")

    def look_strings(self, e):
        if e.done or e.t >= self.T: return {}
        state = (e.pos, e.dir, frozenset(e.opened)); out = {}
        for a in (LEFT, RIGHT, FORWARD, TOGGLE):
            if e.step_plan(state, a) is not None: out[a] = f"ACTION: {ACTIONS[a]}"
        return out
    def look_name(self, a): return ACTIONS[a].lower()
    def answer_options(self, e): return [str(j + 1) for j in range(e.rows)]
    def commit_string(self, y): return f"ACTION: SUBMIT {y}"
    def canonical_action(self, text):
        t = " ".join(str(text).strip().rstrip(".").split()); u = t.upper()
        for a, s in ACTIONS.items():
            if u in (f"ACTION: {s}", s): return f"ACTION: {s}"
        m = re.fullmatch(r"(?:ACTION:\s*)?SUBMIT\s+(?:ROW\s+|ROOM\s+)?(\d+)", u)
        return f"ACTION: SUBMIT {int(m.group(1))}" if m else t
    def parse_action(self, e, text):
        t = self.canonical_action(text)
        if t.startswith("ACTION: SUBMIT "):
            y = t.split()[-1]; return ("answer", y) if y.isdigit() and 1 <= int(y) <= e.rows else ("invalid", None)
        for a, s in self.look_strings(e).items():
            if s == t: return ("look", a)
        return ("invalid", None)
    def do_look(self, e, a):
        e.env.step(int(a)); e.pos, e.dir = tuple(int(v) for v in e.env.agent_pos), int(e.env.agent_dir)
        if a == TOGGLE:
            fx, fy = e.pos[0] + DIR[e.dir][0], e.pos[1] + DIR[e.dir][1]; cell = e.env.grid.get(fx, fy)
            if cell is not None and cell.type == "door" and cell.is_open: e.opened.add((fx, fy))
        e.log.append(ACTIONS[int(a)].lower()); e.see()
    def do_commit(self, e, y): e.answer = y; return str(y) == str(e.truth[0] + 1)
    def after_turn(self, e):
        e.turns_used += 1
        if not e.done and e.turns_used >= self.turns: e.done, e.correct = True, False

    def belief(self, e): return Belief(e, e.M.copy(), e.t)
    def label(self, e): return str(e.truth[0] + 1)
    def answer_posterior(self, e): return self.belief(e).post()
    def answer_prob(self, B, y): return B.post().get(str(y).strip(), 0.0)
    def state_text(self, e):
        rows = sorted(int(r) for r in self.belief(e).post())
        return "Possible: all" if len(rows) == e.rows else "Possible: " + ("row " if len(rows) == 1 else "rows ") + ", ".join(map(str, rows))
    def belief_note(self, e): return self.state_text(e)
    def belief_score(self, B, text, miss=MISS):
        s = read_line(text, B.ep.rows); return -miss if s is None else set_score(B.post(), s, miss)
    def belief_best(self, B, miss=MISS): return set_score(B.post(), {int(r) for r in B.post()}, miss)
    def line_excess(self, B, text, exact):   # the share of the line's rows that the exact line excludes (--line_wide), both read by this reader
        s, x = read_line(text, B.ep.rows), read_line(exact, B.ep.rows); return len(s - x) / len(s) if s and x is not None else 0.0
    def line_belief(self, B, text):
        """The belief a line states, as the teacher's posterior (--line_score decision): the key's places in the rows it names, those the
        evidence excluded too (all places of a row that the evidence ruled out); None when it is not a belief line."""
        s = read_line(text, B.ep.rows)
        if s is None: return None
        row = B.ep.H[:, 0] + 1; named = np.isin(row, sorted(s)); M = (B.M & named) | (named & ~np.isin(row, np.unique(row[B.M])))
        return Belief(B.ep, M, B.t) if M.any() else None
    def read_exact(self, text): return read_line(text, 6) is not None
    def anchor_support(self, text, answers, e=None):
        s = read_line(text, len(answers)); return {str(v) for v in s} if s else set(answers)
    def anchor_equal(self, a, b): x, y = read_line(a, 6), read_line(b, 6); return x is not None and x == y
    def anchor_answer(self, anchor, answers, e=None):
        s = read_line(anchor, len(answers)); return str(min(s)) if s else answers[0]
    def extra_metrics(self, eps):
        if not eps: return {}
        return {"actions": float(np.mean([e.t for e in eps])), "doors_opened": float(np.mean([len(e.opened) for e in eps]))}


