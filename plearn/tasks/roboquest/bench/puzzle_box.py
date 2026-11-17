"""Puzzle Box at the decision level: a wooden box whose sliding lid is locked by a chain of 4 to 6 bolts (the tip of each
bolt rests in a notch of the part it locks: the lid, or the bolt before it), plus 0 to 2 decoy sliders that lock nothing.
All sliders look the same: a metal bar with a round handle near its outer end. At cover 'partial' or 'full' a plate fixed
over the box hides the tips nearest the lid. In some instances one decoy drives a hidden rod against the bolt that holds the
lid: pulling that decoy pushes the bolt back in (the re-lock), and the decoy slides back by itself when it is let go.

Evidence actions: PULL a slider (out by its handle as far as it goes; it stops after a few millimetres when something
blocks it), PUSH a slider in, OPEN THE LID. They are also the progress: a part stays where it is left. The helpers set the
slide joints, as the task's own teleport_solution does (its 5 mm sweep for a full pull), so contacts never act: a part
stops where its bar would meet a chain part (the layout rectangles at the present joints, which reproduces blocked_now in
every state the chain can reach, with the end play of a blocked part), and the hidden re-lock rod is emulated.
The answer SUBMIT [a b ...] pulls the named sliders in that order, slides the lid open, and only if the lid is open stands
the item in the tray (teleport_solution's item step); the robot holds nothing. Sliders are numbered from left to right by
their handles as the left camera sees them; the intro states the engagements in view (none under the plate).
"""
import numpy as np
from roboquest.puzzle import plate
from roboquest.puzzle_box_layouts import overlap, rects_at
from roboquest.puzzle_box_geometry import BOLT_TRAVEL, ROD_CLEARANCE
from roboquest.tasks.puzzle_box import TELEPORT_SETTLE_TICKS
from .base import Task as Base, Invalid, words, numbers

STEP = .001                   # the slide model tests a bar's path in 1 mm steps
ANSWER_WORDS = {"NONE", "NOTHING", "THE", "LID", "OPEN", "PULL", "SLIDER", "SLIDERS", "THEN", "AND"}


def length(x):
    cm = round(abs(x) * 100, 1)
    return f"{cm:g} cm" if cm >= 2 else f"{cm * 10:.0f} mm"


class Task(Base):
    # Hardest eval50 case: a chain of 4 under a full cover, all 4 tips hidden (decoy tips are always in view, so the
    # intro tells decoys apart). A systematic agent pulls the hidden sliders in turn, again over the ones that stayed in,
    # then the lid: 4 + 3 + 2 + 1 + 1 = 11 actions in the worst order (a visible chain of 6 needs 7). Slack for one
    # re-lock (pull the decoy, pull the bolt again) and one test: 14.
    T = 14

    def __init__(self, env, render):
        super().__init__(env, render)
        self.layout = env.spec["layout"]; self.parts = self.layout["parts"]; self.geo = env.geometry["parts"]
        self.relock = self.layout.get("relock") or {}
        self.item = env.spec["item_name"]
        m, d = env.sim.model, env.sim.data
        handle = {n: d.geom_xpos[m.geom_name2id(self.geo[n]["handle_geom"])].copy() for n in self.parts if n != "lid"}
        self.num = {k + 1: n for k, n in enumerate(sorted(handle, key=lambda n: self.camera_x(handle[n])))}

    def _label(self, name):
        return next(k for k, n in self.num.items() if n == name)

    def _slider(self, name):
        return "the lid" if name == "lid" else f"slider {self._label(name)}"

    # ---- what the agent is told ------------------------------------------------------------------------------------
    def intro(self):
        n, lay = len(self.num), self.layout
        rects = plate.plate_rects(lay)
        lines = [f"On the counter stands a wooden box with a sliding lid and {n} sliders (metal bars with round handles), "
                 f"numbered 1 to {n} by their handles from left to right as the left camera sees them. "
                 "A tray and a red SUBMIT button stand on the counter beside it."]
        hidden, free = [], []
        for k, name in self.num.items():
            if rects and plate.tip_hidden(lay, name, rects):
                hidden.append(k); continue
            held = [p for p, part in self.parts.items() if part["blocked_by"] == name]
            if held:
                where = "the edge of the lid" if held[0] == "lid" else f"the side of slider {self._label(held[0])}"
                lines.append(f"The tip of slider {k} rests in a notch in {where}.")
            else:
                free.append(k)
        if free:
            lines.append(f"The tip{'s' if len(free) > 1 else ''} of slider{'s' if len(free) > 1 else ''} "
                         f"{self._list(free)} rest{'' if len(free) > 1 else 's'} in no notch.")
        for k, name in self.num.items():
            notch = self.parts[name].get("notch")
            if self.parts[name]["decoy"] and notch and not plate.hits_plate(rects, notch):
                lines.append(f"Slider {k} has an empty notch in its side.")
        if hidden:
            lines.append(f"A plate fixed over part of the box hides the tip{'s' if len(hidden) > 1 else ''} of "
                         f"slider{'s' if len(hidden) > 1 else ''} {self._list(hidden)}: you cannot see what "
                         f"{'they rest' if len(hidden) > 1 else 'it rests'} in.")
        lines.append(f"The {self.item} is inside the closed box.")
        return " ".join(lines)

    def help(self):
        return "\n".join([
            "PULL SLIDER n - pull slider n out by its handle as far as it goes (a part moves only when nothing blocks it)",
            "PUSH SLIDER n - push slider n in, away from its handle, as far as it goes",
            "OPEN THE LID - slide the lid open (it moves only when nothing blocks it)",
            f"SUBMIT a b ... - your answer: pull the named sliders in this order (name none if nothing more must move), "
            f"then slide the lid open; if it opens, stand the {self.item} in the tray, let go and press Submit"])

    # ---- the slider model ------------------------------------------------------------------------------------------
    def _x(self, name):
        return float(self.env.sim.data.get_joint_qpos(self.geo[name]["joint_name"]))

    def _set(self, name, x):
        d = self.env.sim.data; j = self.geo[name]["joint_name"]
        d.set_joint_qpos(j, x); d.set_joint_qvel(j, 0.)

    def _reach(self, name, target):
        """Where a part stops on its way to target: the last 1 mm step before its bar meets a chain part (the layout
        rectangles, at the parts' present joints). Decoys meet nothing (their sweeps are clear by construction)."""
        part, x = self.parts[name], self._x(name)
        if part["decoy"]:
            return target
        others = [r for q, p in self.parts.items() if q != name and not p["decoy"] for r in rects_at(p, self._x(q))]
        while abs(target - x) > 1e-9:
            nx = target if abs(target - x) <= STEP else x + (STEP if target > x else -STEP)
            if any(overlap(r, o) for r in rects_at(part, nx) for o in others):
                break
            x = nx
        return x

    def _pull(self, name):
        """Pull a part out as far as it goes; returns the distance it moved."""
        before, top = self._x(name), float(self.geo[name]["range"][1])
        stop = self._reach(name, top)
        if stop >= top - 1e-6:
            self.env._sweep_to_full_travel(self.geo[name])         # the task's own sweep (5 mm steps, kitchen check)
        else:
            self._set(name, stop)
        if name == self.relock.get("decoy"):                       # the hidden rod pushes the re-lock bolt back in
            bolt = self.relock["bolt"]
            if self._x(bolt) > BOLT_TRAVEL + ROD_CLEARANCE - self._x(name):
                self._set(bolt, self._reach(bolt, BOLT_TRAVEL + ROD_CLEARANCE - self._x(name)))
                self._set(name, min(self._x(name), BOLT_TRAVEL + ROD_CLEARANCE - self._x(bolt)))
        self.env.sim.forward()
        return self._x(name) - before

    def _push(self, name):
        """Push a slider in as far as it goes: a decoy to its inner limit, a bolt until it would meet a chain part."""
        before = self._x(name)
        self._set(name, self._reach(name, float(self.geo[name]["range"][0]))); self.env.sim.forward()
        return before - self._x(name)

    def _let_go(self, name):
        """The re-lock decoy's hidden spring brings it home when the robot lets go."""
        if name == self.relock.get("decoy") and abs(self._x(name)) > 1e-4:
            self._set(name, 0.); self.env.sim.forward()
            return " When you let go, it slides back by itself to where it started."
        return ""

    def _where(self, name):
        x = self._x(name)
        return "where it started" if abs(x) < 1e-3 else f"{length(x)} {'out' if x > 0 else 'in'} from where it started"

    def _felt(self, did, moved, way, room):
        """What the robot feels: the part went all the way, stopped after a short way, or did not move."""
        if moved < 5e-4:
            return f"{did}: it does not move."
        if moved >= room - 1e-3:
            return f"{did}: it slides {way} {length(moved)}."
        return f"{did}: it slides {way} {length(moved)} and stops; something blocks it."

    def _lid_open(self):
        return self._x("lid") >= float(self.geo["lid"]["open_at"])

    # ---- actions ---------------------------------------------------------------------------------------------------
    def act(self, line):
        t = words(line)
        if t in ("OPEN THE LID", "OPEN LID", "PULL THE LID", "PULL LID", "SLIDE THE LID OPEN"):
            if self._lid_open():
                return {"status": "The lid is open already.", "reading": "", "images": None}
            room = self.geo["lid"]["range"][1] - self._x("lid"); moved = self._pull("lid")
            reading = (f"The lid is open; the {self.item} stands inside the box." if self._lid_open() else
                       f"The lid has slid {length(moved)} and the box is still closed." if moved > 5e-4 else
                       "The lid has not moved.")
            return {"status": self._felt("You pull the lid", moved, "open", room), "reading": reading,
                    "images": self.views()}
        ks = numbers(t)
        verb = t.split(" ")[0] if t else ""
        if verb not in ("PULL", "PUSH") or len(ks) != 1 or t not in (f"{verb} SLIDER {ks[0]}", f"{verb} {ks[0]}"):
            raise Invalid("not an action of this task")
        if ks[0] not in self.num:
            raise Invalid(f"there is no slider {ks[0]}")
        name, k = self.num[ks[0]], ks[0]
        before = {n: self._x(n) for n in self.parts}
        if verb == "PULL":
            if before[name] >= float(self.geo[name]["range"][1]) - 1e-4:
                return {"status": f"Slider {k} is out as far as it goes already.", "reading": "", "images": None}
            status = self._felt(f"You pull slider {k}", self._pull(name), "out", self.geo[name]["range"][1] - before[name])
        else:
            status = self._felt(f"You push slider {k}", self._push(name), "in", before[name] - self.geo[name]["range"][0])
        changed = [n for n in self.parts if abs(self._x(n) - before[n]) > 5e-4]
        reading = " ".join(f"{self._slider(n).capitalize()} is now {self._where(n)}." for n in changed) or \
            f"Slider {k} has not moved."
        images = self.views()
        status += self._let_go(name)
        return {"status": status, "reading": reading, "images": images}

    def _list(self, ks):
        return " and ".join([", ".join(str(k) for k in ks[:-1]), str(ks[-1])] if len(ks) > 1 else [str(ks[0])])

    # ---- privileged, for the self-test only ------------------------------------------------------------------------
    def oracle_lines(self):
        order = self.layout["release_order"][:-1]                 # the bolts, free end first
        b1, n, r = order[-1], self._label, self.relock.get("decoy")
        decoys = [p for p in self.num.values() if self.parts[p]["decoy"]]
        lines = [f"PULL SLIDER {n(b1)}", "OPEN THE LID", f"PUSH SLIDER {n(order[0])}"]
        lines += [f"PUSH SLIDER {n(p)}" for p in decoys] + [f"PULL SLIDER {n(p)}" for p in decoys if p != r]
        lines += [f"PULL SLIDER {n(b)}" for b in order]
        lines.append(f"PUSH SLIDER {n(order[-2])}")              # stops: the bolt it held has moved
        lines.append(f"PULL SLIDER {n(r)}" if r else f"PUSH SLIDER {n(b1)}")   # bolt 1 back in, by the rod or by hand
        return lines + ["OPEN THE LID"]

    def oracle_answer(self, wrong=False):
        b1, r = self.layout["release_order"][-2], self.relock.get("decoy")
        if wrong:
            return f"{self._label(b1)} {self._label(r)}" if r else "NONE"   # re-locks bolt 1 again, or pulls nothing
        return str(self._label(b1))

    def submit(self, answer):
        t = words(answer)
        t = t[len("SUBMIT"):].strip() if t.startswith("SUBMIT") else t
        ks = numbers(t)
        if any(w not in ANSWER_WORDS and not w.isdigit() for w in t.split()) or any(k not in self.num for k in ks):
            raise Invalid("the answer is SUBMIT followed by slider numbers")
        for k in ks:
            self._pull(self.num[k]); self._let_go(self.num[k])
        if not self._lid_open():
            self._pull("lid")
        if self._lid_open():                                       # teleport_solution's item step
            env = self.env; d = env.sim.data
            tray, item = env.geometry["tray"], env.geometry["item"]
            centre = np.asarray(tray["center_world"], float)
            quat = np.asarray(d.get_joint_qpos(item["joint_name"]), float)[3:7]
            z = float(env._tray_inner["floor_top_z"] - item["extent"]["low"][2] + .003)
            d.set_joint_qpos(item["joint_name"], np.r_[centre[0], centre[1], z, quat])
            d.set_joint_qvel(item["joint_name"], np.zeros(6)); env.sim.forward()
        self.tick(TELEPORT_SETTLE_TICKS - self.SETTLE)
        return self.score()
