"""Stamp Composition at the decision level: 3, 4 or 5 round self-inking stamps (one more than the target needs), a tan scratch
board, a green reference card that shows a 3x3 target pattern and a blue final board with a 3x3 grid. Some stamps stand in
another part of the kitchen, behind a tall object or under a cloche, or a flat cover lies on a board.

The die of a stamp is inside its round housing, so neither its pattern nor the way it points can be seen. Evidence action PRINT
presses a stamp on the scratch board in one of four slots that do not overlap, turned a chosen number of quarter turns
counterclockwise from the way it stands, and puts it back (the task's teleport_solution recipe: 4 mm above the board, 12 ticks,
back home). A stamp's yaw is continuous, so the helper snaps each press to a quarter turn relative to the board: the stamp's
offset is the quarter turn nearest to its yaw in the work-surface frame, measured once, so a stamp at a given turn prints the
same pattern on every board. Ink is permanent: a second print in a used slot adds its dots to the dots already there. The
reading of a print is the 3x3 dot pattern the slot then shows. LOOK AROUND THE KITCHEN, LOOK BEHIND an object and LIFT or MOVE
a cover find the hidden stamps and boards. The answer presses each chosen stamp at its chosen turn centred on the final board.
"""
import math
import re
import numpy as np
from roboquest.stamps import geometry, rules
from .base import Task as Base, Invalid, words, CAMERAS

SLOTS = ((-1, 1), (1, 1), (-1, -1), (1, -1))       # board-frame signs (x right, y away from the robot) of slots 1-4
SLOT_NAMES = ("back left", "back right", "front left", "front right")
EDGE = .045                     # slot centre to paper edge: the outer dots (reach .042) stay on the paper
DROP, PRESS, BACK, HOME_SETTLE = .004, 12, 3, 25   # teleport_solution's press, then the stamps settle at home
COLOURS = ("brown", "blue", "red", "green", "yellow", "purple")   # geometry.HANDLE_PALETTE, in order
NOUNS = {"cloche_large": "cloche", "cloche_small": "cloche", "board": "cutting board", "cereal": "cereal box",
         "boxed_food": "box", "wide_cup": "cup"}
BOARDS = {"scratch": "tan scratch board", "final": "blue final board", "reference": "green reference card"}
KEY = "(far row first, each row from left to right as the robot sees it; X is a dot, . is blank)"


def noun(kind):
    return NOUNS.get(kind, str(kind or "object").replace("_", " "))


def pattern(mask):
    """A 3x3 mask (row = board +y, away from the robot) as text, far row first."""
    m = np.asarray(mask, bool)
    return "/".join("".join("X" if m[r, c] else "." for c in range(3)) for r in (2, 1, 0))


class Task(Base):
    # The hardest eval50 instances have 5 stamps with up to 2 of them hidden (or 1 under a cloche): a good agent finds every
    # stamp (<= 2 actions), prints each stamp once (5 prints, the fifth in a used slot) and may need one more print in another
    # used slot to see the dots the first shared slot hid: 8 actions. 2 more for slack.
    T = 10

    def __init__(self, env, render):
        super().__init__(env, render)
        ts = env.task_spec; obs = ts.get("observability") or {}
        realised = {r["object"]: r for r in obs.get("realised", [])}
        hidden = set(obs.get("hidden_bodies") or [])
        self.num, self.known = {}, []                     # number -> stamp body; known stamp bodies in number order
        self.annex = sorted(s for s in hidden if realised.get(s, {}).get("realised") == "annex")
        kinds = {o["name"]: o.get("kind") for o in obs.get("occluders", [])}
        self.behind = {}                                  # agent's name of an occluder -> stamp body behind it
        for s in sorted(hidden):
            r = realised.get(s, {})
            if r.get("realised") == "occluder":
                self.behind[self._fresh(noun(kinds.get(r.get("occluder"))), self.behind)] = s
        self.under = {}                                   # agent's name of a cover -> stamp body or board name under it
        for c in obs.get("covers", []):
            if c.get("built"):
                self.under[self._fresh(noun(c.get("kind")), self.under)] = c.get("place") or c.get("object")
        self.lifted, self.looked = set(), set()
        self.covered = {v for v in self.under.values() if v in BOARDS}     # boards under a cover
        self.home = {s: np.asarray(env.sim.data.get_joint_qpos(f"{s}_joint"), float).copy() for s in env.stamp_names}
        # the quarter turn of each stamp relative to the work surface, rounded once (hidden: the housing is round)
        self.offset = {s: int(round(env.stamp_state(s)["frame_yaw_rad"] / (math.pi / 2))) % 4 for s in env.stamp_names}
        pal = np.asarray(geometry.HANDLE_PALETTE)[:, :3]
        self.colour = {f"stamp_{i}": COLOURS[int(np.argmin(np.abs(pal - np.asarray(c[:3])).sum(1)))]
                       for i, c in enumerate(env.spec["handle_rgba"])}
        pos = {s: self.home[s][:3] for s in env.stamp_names if s not in hidden}
        for s in sorted(pos, key=lambda s: self._image_x(pos[s])):
            self._number(s)
        d = env.boards["scratch"]["half_size"][0] - EDGE                # slot centres: .072 (5-cell board), .055 (4-cell)
        self.slots = [(sx * d, sy * d) for sx, sy in SLOTS]
        self.start = self._intro()

    def _image_x(self, p, camera=CAMERAS[0]):
        """Horizontal image coordinate of a world point in a camera (perspective; larger is further right)."""
        m, d = self.env.sim.model, self.env.sim.data; cid = m.camera_name2id(camera)
        local = d.cam_xmat[cid].reshape(3, 3).T @ (np.asarray(p, float) - d.cam_xpos[cid])
        return float(local[0] / max(-local[2], 1e-6))

    def _fresh(self, base, taken):
        name, k = base, 2
        while name.upper() in {t.upper() for t in taken}:
            name, k = f"{base} {k}", k + 1
        return name

    def _number(self, body):
        self.known.append(body); self.num[len(self.known)] = body; return len(self.known)

    def _label(self, body):
        return next(k for k, b in self.num.items() if b == body)

    def _stamp(self, k):
        return f"stamp {k} ({self.colour[self.num[k]]} handle)"

    # ---- what the agent is told ------------------------------------------------------------------------------------
    def _intro(self):
        n = len(self.known)
        lines = [(f"On the work surface stamps 1 to {n} stand face down, numbered from left to right as the left camera "
                  "sees them: " if n > 1 else "On the work surface one stamp stands face down: ")
                 + ", ".join(self._stamp(k) for k in self.num) + ". The round stamp housings are unmarked."]
        if "reference" not in self.covered:
            lines.append(f"The green reference card shows the target pattern {pattern(self.env.spec['target'])} {KEY}.")
        if "final" not in self.covered:
            lines.append("The blue final board shows an empty 3x3 grid.")
        if "scratch" not in self.covered:
            lines.append(self._scratch_text())
        for name in self.behind:
            lines.append(f"A tall {name} stands on the counter.")
        for name, what in self.under.items():
            lines.append(f"A {name} lies flat on the work surface." if what in BOARDS
                         else f"A {name} stands upside down on the work surface.")
        return " ".join(lines)

    def _scratch_text(self):
        return ("The tan scratch board is blank. It has room for four prints that do not overlap: "
                + ", ".join(f"slot {i + 1} {s}" for i, s in enumerate(SLOT_NAMES)) + " (as the robot sees it).")

    def intro(self):
        return self.start

    def help(self):
        out = ["PRINT a IN SLOT s TURN r - press stamp a on the scratch board in slot s (1 back left, 2 back right, "
               "3 front left, 4 front right), turned r quarter turns counterclockwise seen from above (r = 0 to 3) from "
               "the way it stands, and put it back where it stood. The print turns with the stamp. Ink is permanent: "
               "a used slot keeps its dots and adds the new ones",
               "LOOK AROUND THE KITCHEN - look for stamps in the other parts of the kitchen"]
        out += [f"LOOK BEHIND THE {name.upper()}" for name in self.behind]
        out += [f"{'MOVE' if what in BOARDS else 'LIFT'} THE {name.upper()}" for name, what in self.under.items()]
        out.append("SUBMIT a:r b:r ... - your answer: press each named stamp a, turned r quarter turns as in PRINT, centred "
                   f"on the final board, and put every stamp back down (e.g. SUBMIT 1:0 3:2). Patterns read {KEY}")
        return "\n".join(out)

    # ---- actions ---------------------------------------------------------------------------------------------------
    def act(self, line):
        t = words(line)
        if t.startswith("PRINT"):
            return self._print(t)
        if t in ("LOOK AROUND THE KITCHEN", "LOOK AROUND"):
            if "around" in self.looked:
                return {"status": "You have looked around the kitchen already.", "reading": "", "images": None}
            self.looked.add("around")
            found = [self._stamp(self._number(s)) for s in self.annex]
            status = (f"In another part of the kitchen you find {' and '.join(found)}." if found
                      else "You find no other stamps in the kitchen.")
            return {"status": status, "reading": "", "images": None}
        for name, body in self.behind.items():
            if t in (f"LOOK BEHIND THE {name.upper()}", f"LOOK BEHIND {name.upper()}"):
                if name in self.looked:
                    return {"status": f"You have looked behind the {name} already.", "reading": "", "images": None}
                self.looked.add(name)
                return {"status": f"Behind the {name} you find {self._stamp(self._number(body))}.", "reading": "",
                        "images": None}
        for name, what in self.under.items():
            if t in {f"{v} {name.upper()}" for v in ("LIFT THE", "LIFT", "MOVE THE", "MOVE")}:
                return self._lift(name, what)
        raise Invalid("not an action of this task")

    def _lift(self, name, what):
        if name in self.lifted:
            return {"status": f"The {name} is set aside already.", "reading": "", "images": None}
        self.env._uncover_steps(); self.lifted.update(self.under)       # the task's own uncover helper (one cover per scene)
        reading = ""
        if what in BOARDS:
            self.covered.discard(what)
            status = f"You slide the {name} aside. Under it lies the {BOARDS[what]}."
            reading = {"reference": f"The reference card shows the target pattern {pattern(self.env.spec['target'])}.",
                       "final": "The final board shows an empty 3x3 grid.", "scratch": self._scratch_text()}[what]
        else:
            status = f"You lift the {name} and set it aside. Under it you find {self._stamp(self._number(what))}."
        return {"status": status, "reading": reading, "images": self.views()}

    def _print(self, t):
        a, s, r = (re.search(p, t) for p in (r"^PRINT (?:STAMP )?(\d+)\b", r"\bSLOT (\d+)\b", r"\bTURNS? (\d+)\b"))
        if not a or not s:
            raise Invalid("a print needs a stamp and a slot: PRINT a IN SLOT s TURN r")
        k, slot, turn = int(a.group(1)), int(s.group(1)), int(r.group(1)) if r else 0
        if k not in self.num or not 1 <= slot <= 4 or not 0 <= turn <= 3:
            raise Invalid("a print needs a known stamp, a slot 1 to 4 and a turn 0 to 3")
        if "scratch" in self.covered:
            raise Invalid("you have not found the scratch board")
        made = self._press(self.num[k], "scratch", self.slots[slot - 1], turn); self.tick(HOME_SETTLE)
        status = (f"You press {self._stamp(k)}, turned {turn} quarter turn{'s' * (turn != 1)}, on slot {slot} of the "
                  f"scratch board and put it back where it stood." + ("" if made else " It left no print."))
        return {"status": status, "reading": f"Slot {slot} shows {self._slot_pattern(slot)}.", "images": self.views()}

    def _press(self, stamp, board, xy, turn):
        """Set the stamp 4 mm above the board point xy (board frame) at its snapped quarter turn, let it print, put it back."""
        env = self.env; b = env.boards[board]; c, s = math.cos(b["yaw_rad"]), math.sin(b["yaw_rad"])
        yaw = b["yaw_rad"] + (self.offset[stamp] + turn) * math.pi / 2
        x, y = b["position"][0] + c * xy[0] - s * xy[1], b["position"][1] + s * xy[0] + c * xy[1]
        before = len(env.ink_events)
        self._set(stamp, np.r_[x, y, b["position"][2] + DROP, math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])
        self.tick(PRESS)
        self._set(stamp, self.home[stamp]); self.tick(BACK)
        return len(env.ink_events) - before

    def _set(self, stamp, qpos):
        d = self.env.sim.data
        d.set_joint_qpos(f"{stamp}_joint", qpos); d.set_joint_qvel(f"{stamp}_joint", np.zeros(6)); self.env.sim.forward()

    def _slot_pattern(self, slot):
        """The 3x3 dots the scratch board's ink shows in a slot (board frame, row = +y)."""
        ink = self.env.ink["scratch"]; res = ink.shape[0]; half = self.env.boards["scratch"]["half_size"][0]
        sx, sy = self.slots[slot - 1]; m = np.zeros((3, 3), bool)
        for row in range(3):
            for col in range(3):
                x, y = sx + (col - 1) * rules.CELL_SPACING, sy + (row - 1) * rules.CELL_SPACING
                m[row, col] = ink[int((y / half + 1) / 2 * res), int((x / half + 1) / 2 * res)]
        return pattern(m)

    # ---- privileged, for the self-test only ------------------------------------------------------------------------
    def oracle_lines(self):
        lines = ["LOOK AROUND THE KITCHEN"] + [f"LOOK BEHIND THE {n.upper()}" for n in self.behind]
        lines += [f"{'MOVE' if w in BOARDS else 'LIFT'} THE {n.upper()}" for n, w in self.under.items()]
        n = len(self.env.stamp_names)                      # every stamp is known once the looks and lifts are done
        return lines + [f"PRINT {k} IN SLOT {(k - 1) % 4 + 1} TURN {(k - 1) // 4}" for k in range(1, n + 1)]

    def oracle_answer(self, wrong=False):
        pairs = [(self._label(f"stamp_{i}"), (k - self.offset[f"stamp_{i}"]) % 4) for i, k in self.env.spec["solution"]]
        if wrong:
            pairs[0] = (pairs[0][0], (pairs[0][1] + 1) % 4)
        return " ".join(f"{a}:{r}" for a, r in sorted(pairs))

    def submit(self, answer):
        t = words(answer)
        t = t[len("SUBMIT"):].strip() if t.startswith("SUBMIT") else t
        pairs = [] if t in ("NONE", "NOTHING") else [tuple(map(int, p)) for p in re.findall(r"(\d+) ?: ?(\d+)", t)]
        rest = re.sub(r"\d+ ?: ?\d+", "", t).replace("AND", "").strip()
        if t not in ("NONE", "NOTHING") and (not pairs or rest):
            raise Invalid("the answer must be pairs stamp:turn, e.g. SUBMIT 1:0 3:2")
        ks = [k for k, _ in pairs]
        if len(set(ks)) < len(ks) or any(k not in self.num for k in ks) or any(not 0 <= r <= 3 for _, r in pairs):
            raise Invalid("the answer names stamps that are not known, a stamp twice, or a turn outside 0 to 3")
        if set(self.under) - self.lifted:
            self.env._uncover_steps(); self.lifted.update(self.under)   # clear the boards first, as teleport_solution does
        for k, r in pairs:
            self._press(self.num[k], "final", (0., 0.), r)
        self.tick(HOME_SETTLE)
        return self.score()
