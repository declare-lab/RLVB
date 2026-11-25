"""Pro-Worlds: the 15 visual games of the frozen expansion v6.1 set (benchmarks/proworlds/games/expansion-6.1), as episodes for
the evaluator (plearn/api/episode.py). Test only: no teacher.

The engine is the set's own (engine.js, through expansion_bridge.mjs beside this file, one process per episode). A game is one
board of sites; the window shows one site (one 16 x 16 tile, drawn here 16 times larger); a move goes
to the next site (up, down, left or right; at the edge the window stays and the move still counts, as the engine does). The
answer is one member name. The board is 'normal' or 'long' (env_args board: long).
The engine budget is 150 moves (plearn/tasks/__init__.py TEST_MOVES; PL_TEST_MOVES=0: the board's own, 35 normal, 65 long).

The test set: game k of TEST-ORDER.txt (ORDER below) plays the seeds
50,101,300 + 10,000 (k - 1) + j, j = 0 .. 19 (20 plays a game, 300 in all).

The agent's turn is the trainer's format (format.py): the public rules of the game (with the question, the answer set and the
budget), the answers as numbers ('1 = <name>'), the picture of the start site and of the site after every move, the agent's belief
line of its last turn, the moves used and the turns used. The belief line is 'Possible: <answer numbers>'; the action line is
'ACTION: MOVE UP|DOWN|LEFT|RIGHT' or 'ACTION: SUBMIT <answer number or name>'. A line that cannot be read uses up the turn; a move
with no moves left ends the game without an answer (wrong); the turn limit is 152 (PL_TEST_MOVES=0: the budget + 8). Score:
correct when the answer is the member of the world; premature: the engine's exact belief of that member was below 0.95 at the
answer.

one_line (the action-only turn, for example the base model's "Base" row: --task_args proworlds.one_line=1): the system text asks for the action
line alone, the evaluator samples the action line alone after the thought (plearn/api/episode.py), and no belief line is
carried.
"""
from __future__ import annotations
import json, os, subprocess

from plearn.api.episode import carried
import plearn.tasks as _tasks                                        # TEST_MOVES
from plearn.tasks.proworlds import format as F

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = os.environ.get("NODE_BIN") or "node"                         # Node.js 22 or later: NODE_BIN, else node on PATH
ORDER = ("face", "jigsaw", "kaleido", "bird", "fish", "recipe", "turtle", "balloon", "rocket", "bell", "bee", "candle", "sailboat",
         "barn", "lantern")                                          # TEST-ORDER.txt: line k = game k
SEED_BASE, SEED_STEP, PLAYS = 50101300, 10000, 20
STOP = 0.95                                                          # the engine's STOP: the exact belief that counts as sure
DIRS = {"UP": "up", "RIGHT": "right", "DOWN": "down", "LEFT": "left"}


def seeds(game, n=PLAYS):
    """The test seeds of a game: 50,101,300 + 10,000 (k - 1) + j, k its line in TEST-ORDER.txt."""
    k = ORDER.index(game) + 1
    return [SEED_BASE + SEED_STEP * (k - 1) + j for j in range(n)]


class Bridge:
    def __init__(self):
        self.p = subprocess.Popen([NODE, os.path.join(HERE, "expansion_bridge.mjs")], cwd=HERE, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)

    def __call__(self, **req):
        self.p.stdin.write(json.dumps(req) + "\n"); self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        if not r["ok"]: raise RuntimeError(r["error"])
        return r["result"]

    def close(self):
        try:
            self.p.stdin.write('{"op": "close"}\n'); self.p.stdin.flush(); self.p.wait(timeout=10)
        except Exception:
            self.p.kill(); self.p.wait()
        for f in (self.p.stdin, self.p.stdout):
            try: f.close()
            except Exception: pass


def picture(window, colors):
    """The 16 x 16 colour indices of a site as a 256 x 256 picture (each cell 16 x 16 pixels)."""
    from PIL import Image
    n = int(round(len(window) ** 0.5)); im = Image.new("RGB", (n, n))
    im.putdata([tuple(int(colors[c][k:k + 2], 16) for k in (1, 3, 5)) for c in window])
    return im.resize((n * 16, n * 16), Image.NEAREST)


def user_parts(t, o, members, pics, moves_log, sites, note, turns_used, limit):
    """The user message of one turn: its parts and the pictures in
    order. t: the game's public task; o: the current observation; pics: the start picture and one after every move, whose
    directions are moves_log and whose sites are sites; note: the agent's belief line of its last turn (None: not shown)."""
    at = lambda site: f" (site column {site['column']}, row {site['row']})"
    parts = [{"type": "text", "text": f"{t['name']}: {t['question']}\n{t['rules']}\nThe answers: {', '.join(F.listed(members))}.\n"
                                      f"You see the board, {t['cols']} x {t['rows']} sites, only through a window of one site "
                                      f"({t['window']} x {t['window']} cells). A move goes to the next site. At most {t['budget']} moves."
                                      f"\nThe pictures so far:\nStart{at(sites[0])}: "}, {"type": "image"}]
    for m, site in zip(moves_log, sites[1:]):
        parts += [{"type": "text", "text": f"\nAfter MOVE {m}{at(site)}: "}, {"type": "image"}]
    parts.append({"type": "text", "text": carried(note)
                  + f"\nMoves used: {o['steps']} of {t['budget']}. Moves allowed now: {'UP, RIGHT, DOWN, LEFT' if o['remaining'] > 0 else 'none'}. "
                  f"Turns used: {turns_used} of {limit}. Decide now."})
    return parts, list(pics)


class Episode:
    @classmethod
    def specs(cls, split, n, seed, game, **_):
        """[split, seed] of the game's first n test plays. The set has one split, test."""
        if split != "test": raise ValueError(f"Pro-Worlds {game}: only the test split (not {split!r})")
        return [[split, s] for s in seeds(game, n)]

    def __init__(self, spec, game, board="normal", max_turns_extra=None, budget=None, one_line=False, **_):    # **_: the env_args of the other proworlds tasks
        _, seed = spec
        self.game, self.board, self.seed, self.one_line = game, board, int(seed), bool(one_line)
        if budget is None: budget = _tasks.TEST_MOVES                 # 150 (plearn/tasks/__init__.py); 0: the board's own budget
        self.b = Bridge(); self.colors = self.b(op="palette")["colors"]
        r = self.b(op="new", id=game, seed=self.seed, geometry=board, budget=int(budget) if budget else None)
        self.task, self.obs = r["task"], r["observation"]
        if budget and self.task["budget"] != int(budget): raise RuntimeError(f"Pro-Worlds {game}: engine budget {self.task['budget']} != {budget}")
        self.members = self.task["members"]; self.keys = F.label_keys(self.members)
        self.pics, self.moves_log, self.sites = [picture(self.obs["window"], self.colors)], [], [self.obs["site"]]
        self.turns, self.note, self.done, self.correct, self.answer, self.metrics, self.end = [], None, False, False, None, {}, None
        if max_turns_extra is None: max_turns_extra = _tasks.TEST_TURNS - _tasks.TEST_MOVES if budget else 8   # 152 turns; the board's own: + 8
        self.limit = self.task["budget"] + max_turns_extra

    def messages(self, carry):
        parts, images = user_parts(self.task, self.obs, self.members, self.pics, self.moves_log, self.sites,
                                   self.note if carry and not self.one_line else None, len(self.turns), self.limit)
        return F.system("window", self.members, one_line=self.one_line), parts, images

    def step(self, belief, action, record):
        what, arg = F.read_action(action, self.keys, "window"); kind = "unreadable"
        self.turns.append({**record, "belief": belief, "action": action})
        self.note = belief
        if what == "move":
            if self.obs["remaining"] <= 0:                           # no moves left: the game ends without an answer
                kind, self.done, self.end = "budget", True, "budget"
            else:
                d = F.DIRS[arg]; self.obs = self.b(op="move", direction=DIRS[d])["observation"]; kind = "move"
                self.pics.append(picture(self.obs["window"], self.colors)); self.moves_log.append(d); self.sites.append(self.obs["site"])
        elif what == "answer":
            self.answer = self.members[arg]; self.metrics = self.b(op="commit", name=self.answer)["metrics"]
            self.correct, self.done, self.end, kind = bool(self.metrics.get("right")), True, "answer", "answer"
        self.turns[-1]["kind"] = kind
        if not self.done and len(self.turns) >= self.limit: self.done, self.end = True, "turns"

    def row(self):
        n = max(1, len(self.turns)); sub = self.answer is not None
        return {"game": self.game, "board": self.board, "seed": self.seed, "correct": self.correct, "success": float(self.correct),
                "submitted": float(sub), "answer": self.answer, "end": self.end,
                "premature": float(self.metrics.get("topChance", 0) < STOP) if sub else None,
                "moves": self.obs["steps"], "turns": len(self.turns),
                "unreadable_turns": sum(t["kind"] == "unreadable" for t in self.turns) / n,
                "thought_cut": sum(bool(t.get("thought_cut")) for t in self.turns) / n, "turn_log": self.turns}

    def close(self):
        self.b.close()
