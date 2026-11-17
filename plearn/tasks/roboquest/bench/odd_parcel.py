"""Odd Parcel at the decision level: 4, 6 or 8 parcels, 1 to 3 of them heavier or lighter (120 g or 200 g), some hidden in
another part of the kitchen, behind a tall object or under a cover.

Evidence actions: WEIGH (0 to 3 known parcels on each pan; the helper sets them on the pans, the beam settles for 2 s, and
the helper puts every parcel back where it stood), LOOK AROUND THE KITCHEN (finds the parcels in other parts of the kitchen),
LOOK BEHIND an object, LIFT a cover. The reading of a weighing is the beam angle. A parcel gets a number when it is found;
the visible ones are numbered from left to right as the left camera sees them. The answer puts the named parcels in the box.
"""
from .base import Task as Base, Invalid, words, numbers
from .base import SIZE, CAMERAS
import math, os
import numpy as np
from PIL import Image

SLOTS = {1: (0.,), 2: (-.045, .045), 3: (-.07, 0., .07)}
LEVEL_DEG = .5                 # a smaller beam angle reads as level
BOX_SETTLE, BOX_TRIES = 60, 4


def noun(kind):
    return str(kind or "object").replace("_", " ").strip()


class Task(Base):
    T = 12

    def views(self):
        """RQ_DIAL_ZOOM=1: each block of views also gets a zoom on the balance dial, cut from the left camera; dial position from the simulator geometry. Off = the benchmark's 3 cameras."""
        v = super().views()
        if v is None or os.environ.get("RQ_DIAL_ZOOM") != "1":
            return v
        m, d = self.env.sim.model, self.env.sim.data; g = [i for i in range(m.ngeom) if (m.geom_id2name(i) or "").endswith("tick_plate")][0]; c = m.camera_name2id(CAMERAS[0])
        r = d.cam_xmat[c].reshape(3, 3).T @ (d.geom_xpos[g] - d.cam_xpos[c]); f = (SIZE / 2) / math.tan(math.radians(m.cam_fovy[c]) / 2)
        cx, cy = SIZE / 2 + f * r[0] / -r[2], SIZE / 2 - f * r[1] / -r[2]; sc = SIZE / 512
        box = tuple(int(round(x)) for x in (cx - 40 * sc, cy - 30 * sc, cx + 40 * sc, cy + 30 * sc))
        return v + [np.asarray(Image.fromarray(v[0]).crop(box).resize((640, 480), Image.LANCZOS))]

    def __init__(self, env, render):
        super().__init__(env, render)
        ts = env.task_spec; obs = ts.get("observability") or {}
        realised = {r["object"]: r for r in obs.get("realised", [])}
        hidden = set(ts.get("hidden_parcels_private") or [])
        self.num, self.known = {}, []                     # number -> parcel body; known parcel bodies in number order
        self.annex = sorted(p for p in hidden if realised.get(p, {}).get("realised") == "annex")
        kinds = {o["name"]: o.get("kind") for o in obs.get("occluders", [])}
        self.behind = {}                                  # agent's name of an occluder -> hidden parcel body
        for p in sorted(hidden):
            r = realised.get(p, {})
            if r.get("realised") == "occluder":
                self.behind[self._fresh(noun(kinds.get(r.get("occluder"))), self.behind)] = p
        covers = {c["name"]: c for c in obs.get("covers", []) if c.get("built")}
        self.under = {}                                   # agent's name of a cover -> hidden parcel body
        for name, c in covers.items():
            self.under[self._fresh(noun(c.get("kind")).replace(" small", ""), self.under)] = c["object"]
        self.lifted, self.looked = set(), set()
        pos = {p: env.sim.data.body_xpos[env.obj_body_id[p]].copy() for p in env.parcel_ids if p not in hidden}
        for p in sorted(pos, key=lambda p: (round(self.camera_x(pos[p]), 2), pos[p][2])):
            self._number(p)
        self.stacks = [(p, q) for p in pos for q in pos if p != q and pos[q][2] > pos[p][2] + .03
                       and abs(pos[q][0] - pos[p][0]) < .03 and abs(pos[q][1] - pos[p][1]) < .03]

    def _fresh(self, base, taken):
        name, k = base, 2
        while name.upper() in {t.upper() for t in taken}:
            name, k = f"{base} {k}", k + 1
        return name

    def _number(self, body):
        self.known.append(body); self.num[len(self.known)] = body; return len(self.known)

    def _label(self, body):
        return next(k for k, b in self.num.items() if b == body)

    # ---- what the agent is told ------------------------------------------------------------------------------------
    def intro(self):
        n = len(self.known)
        lines = [f"On the work counter you see parcels 1 to {n}, numbered from left to right as the left camera sees them."]
        for low, top in self.stacks:
            lines.append(f"Parcel {self._label(top)} stands on top of parcel {self._label(low)}.")
        lines.append("A two-pan balance with a pointer and a scale (level, half way, stop at 12 degrees) and an open box "
                     "stand on the counter.")
        for name in self.behind:
            lines.append(f"A tall {name} stands on the counter.")
        for name in self.under:
            lines.append(f"A {name} stands upside down on the counter.")
        return " ".join(lines)

    def help(self):
        out = ["WEIGH a b VS c d - put the named parcels on the pans (0 to 3 on each pan; the first group on the left pan), "
               "read the balance, put the parcels back",
               "LOOK AROUND THE KITCHEN - look for parcels in the other parts of the kitchen"]
        out += [f"LOOK BEHIND THE {name.upper()}" for name in self.behind]
        out += [f"LIFT THE {name.upper()}" for name in self.under]
        out.append("SUBMIT a b - put the named parcels in the box (your answer; SUBMIT NONE puts nothing in it)")
        return "\n".join(out)

    # ---- actions ---------------------------------------------------------------------------------------------------
    def act(self, line):
        t = words(line)
        if t.startswith("WEIGH"):
            return self._weigh(t)
        if t in ("LOOK AROUND THE KITCHEN", "LOOK AROUND"):
            if "around" in self.looked:
                return {"status": "You have looked around the kitchen already.", "reading": "", "images": None}
            self.looked.add("around")
            found = [self._number(p) for p in self.annex]
            status = (f"In another part of the kitchen you find {self._list(found)}." if found
                      else "You find no other parcels in the kitchen.")
            return {"status": status, "reading": "", "images": None}
        for name, body in self.behind.items():
            if t in (f"LOOK BEHIND THE {name.upper()}", f"LOOK BEHIND {name.upper()}"):
                if name in self.looked:
                    return {"status": f"You have looked behind the {name} already.", "reading": "", "images": None}
                self.looked.add(name)
                return {"status": f"Behind the {name} you find parcel {self._number(body)}.", "reading": "", "images": None}
        for name, body in self.under.items():
            if t in (f"LIFT THE {name.upper()}", f"LIFT {name.upper()}"):
                if name in self.lifted:
                    return {"status": f"The {name} is lifted already.", "reading": "", "images": None}
                self.env.clear_covers(); self.lifted.add(name)
                return {"status": f"You lift the {name} and set it aside. Under it you find parcel {self._number(body)}.",
                        "reading": "", "images": self.views()}
        raise Invalid("not an action of this task")

    def _weigh(self, t):
        if " VS " not in t:
            raise Invalid("a weighing needs VS")
        left, right = (numbers(s) for s in t[len("WEIGH"):].split(" VS ", 1))
        names = left + right
        if not names or len(left) > 3 or len(right) > 3 or len(set(names)) < len(names) or any(k not in self.num for k in names):
            raise Invalid("a weighing needs 0 to 3 known parcels on each pan, each parcel once")
        env = self.env
        env.park_all_parcels()
        for side, group in (("left", left), ("right", right)):
            for k, dx in zip(group, SLOTS.get(len(group), ())):
                env.place_parcel_on_pan(self.num[k], side, dx=dx)
        self.tick(max(1, round(2.0 / env.control_timestep)))
        angle = float(env.beam_angle_deg()); images = self.views()
        env.park_all_parcels(); self.tick(max(1, round(1.0 / env.control_timestep)))
        if abs(angle) < LEVEL_DEG:
            reading = "The balance is level."
        else:
            reading = f"The {'left' if angle < 0 else 'right'} pan is down {abs(angle):.1f} degrees."
        status = f"Left pan: {self._list(left)}; right pan: {self._list(right)}. The parcels are back in their places."
        return {"status": status, "reading": reading, "images": images}

    def _list(self, ks):
        if not ks:
            return "nothing"
        return ("parcel " if len(ks) == 1 else "parcels ") + " ".join(str(k) for k in ks)

    # ---- privileged, for the self-test only ------------------------------------------------------------------------
    def oracle_lines(self):
        lines = ["LOOK AROUND THE KITCHEN"] + [f"LOOK BEHIND THE {n.upper()}" for n in self.behind]
        lines += [f"LIFT THE {n.upper()}" for n in self.under]
        return lines + ["WEIGH 1 VS 2", "WEIGH 1 2 VS 3 4"]

    def oracle_answer(self, wrong=False):
        odd = [self._label(p) for p in self.env.task_spec["odd_parcels_private"]]
        if wrong:
            odd = odd[:-1] + [next(k for k in self.num if k not in odd)]
        return " ".join(str(k) for k in sorted(odd))

    def submit(self, answer):
        t = words(answer)
        ks = [] if t in ("NONE", "NOTHING", "0") else numbers(t)
        if len(set(ks)) < len(ks) or any(k not in self.num for k in ks):
            raise Invalid("the answer names parcels that are not known")
        env = self.env; names = [self.num[k] for k in ks]
        env.park_all_parcels(); self.tick(20)
        for level, name in enumerate(names):          # stack them one at a time and re-seat the stack, as the task's own solver does
            for _ in range(BOX_TRIES):
                for lower, earlier in enumerate(names[:level + 1]):
                    env.place_parcel_in_box(earlier, lower)
                self.tick(BOX_SETTLE)
                if all(p in env.parcels_in_box() for p in names[:level + 1]):
                    break
        return self.score()
