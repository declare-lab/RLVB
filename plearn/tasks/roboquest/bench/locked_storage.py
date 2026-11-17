"""Locked Storage at the decision level: one target behind a chain of 1 to 3 locked compartments; in half of the instances
one more locked compartment (a dead end) holds only a decoy.

Six compartments of the kitchen have handles. A locked one has a coloured lock plate and, on the counter near it, a reader
pad of the same colour. The task's own lock rule runs on every tick: the token of that colour on that pad unlocks the
compartment; a shut compartment without its token on the pad locks again. The first token lies on the counter; each later
token lies inside the compartment before it in the chain. The dead end's token lies on the counter or inside a chain drawer.

Evidence actions: LOOK AROUND THE KITCHEN (finds the compartments, their lock plates, the reader pads and what lies on the
counters, and numbers the compartments as the robot turns from its left round to its right), PUT a token ON a pad (the
helper takes the token, out of its compartment if it lies in one that opens, and sets it on the pad, as the task's own
onto_pad), OPEN a compartment (the base goes to the compartment's stand-off pose, backed off by the pull; a locked one does
not move; else the helper opens it, the robot looks inside and shuts it again, so only one compartment is ever open and
none hides another). A drawer carries its contents. The reading of OPEN names what lies inside; the images show it. The
answer takes the object out of the named compartment, puts it on the tray and takes the robot back to its start pose, as
the task's own teleport_solution does.
"""
import math
import numpy as np
from roboquest.locked.objects import COLOUR_NAMES
from roboquest.locked.rules import token_on_pad
from roboquest.search.layout import DRAWER_PULL_M, front_normal
from roboquest.search.objects import POOL
from roboquest.search.visibility import relative_bearing
from .base import Task as Base, Invalid, words, numbers

PUT_TICKS = 3                  # ticks after a token is set on a pad (the task's own onto_pad), so the lock rule runs
LOOK_TICKS = 10                # ticks after an opening, so the contents settle before the cameras look
BASE_JOINTS = ("joint_mobile_forward", "joint_mobile_side", "joint_mobile_yaw")
COUNT = ("No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight")


def wrap(angle):
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi


def a(colour):
    return ("an " if colour[0] in "aeiou" else "a ") + colour


def listing(parts):
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


class Task(Base):
    # The hardest eval50 row (chain depth 3 with a dead end) needs LOOK AROUND + 3 x (PUT, OPEN) = 7 actions when the agent
    # follows the right token at once. The dead end first costs 2 more and a pull on a locked compartment 1 more (10);
    # 2 more give room for a look into a free compartment.
    T = 12

    def __init__(self, env, render):
        super().__init__(env, render)
        self.start = env.base_pose()
        names = {env.sim.model.joint_id2name(j) for j in range(env.sim.model.njnt)}
        self.base_joints = next([p + j for j in BASE_JOINTS] for p in ("mobilebase0_", "robot0_", "")
                                if all(p + j in names for j in BASE_JOINTS))
        xy, yaw = self.start                              # compartments numbered from the robot's left round to its right
        pos = {c: env.compartments[c]["pos"] for c in env.compartments}
        self.order = sorted(pos, key=lambda c: (-round(relative_bearing(xy, yaw, pos[c][:2]), 2), -pos[c][2]))
        self.colour = {n: t["colour"] for n, t in env.tokens.items()}          # token body -> colour
        self.token = {c: n for n, c in self.colour.items()}
        self.pad = {env.pads[cid]["colour"]: cid for cid in env.locked_ids}    # pad colour -> its compartment
        spec = env.task_spec.get("open_tokens") or {}
        self.seen = {n for n, s in spec.items() if s.get("seen_at_reset")}     # tokens the agent knows of
        self.start_tokens = sorted(self.seen, key=lambda n: self.colour[n])
        self.num = {}                                                         # number -> compartment (after LOOK AROUND)

    def _kind(self, cid):
        return "drawer" if self.env.compartments[cid]["kind"] == "drawer" else "cabinet"

    def _label(self, cid):
        return f"{self._kind(cid)} {self.order.index(cid) + 1}"

    def _noun(self, body):
        if body in self.colour:
            return a(self.colour[body]) + " token"
        return "a " + POOL[body[len("obj_"):]]["description"].split(" ", 1)[1]

    # ---- what the agent is told ------------------------------------------------------------------------------------
    def intro(self):
        lines = ["The work counter with the tray and the Submit button is in front of you."]
        if self.start_tokens:
            lines.append(f"On the counter you see {listing([self._noun(n) for n in self.start_tokens])}.")
        lines.append("You have not looked around the kitchen yet: LOOK AROUND THE KITCHEN finds its compartments, their "
                     "lock plates and the reader pads, and gives the compartments their numbers.")
        return " ".join(lines)

    def help(self):
        return "\n".join([
            "LOOK AROUND THE KITCHEN - find the compartments, the reader pads and the tokens, and number the compartments",
            "PUT THE c TOKEN ON THE p PAD - set a token you know of on a reader pad (c and p are colours)",
            "OPEN k - open compartment k, look inside and shut it again",
            "SUBMIT k - your answer: take the object out of compartment k and put it on the tray"])

    # ---- actions ---------------------------------------------------------------------------------------------------
    def act(self, line):
        t = words(line)
        if t in ("LOOK AROUND THE KITCHEN", "LOOK AROUND"):
            return self._look()
        if t.startswith("PUT"):
            return self._put(t)
        if t.startswith("OPEN"):
            return self._open(self.num[self._number(t)])
        raise Invalid("not an action of this task")

    def _look(self):
        if self.num:
            return {"status": "You have looked around the kitchen already.", "reading": "", "images": None}
        env = self.env
        self.num = {k: cid for k, cid in enumerate(self.order, 1)}
        comps = [f"{self._label(cid)} ({env.pads[cid]['colour'] + ' lock plate' if cid in env.pads else 'no lock plate'})"
                 for cid in self.order]
        loose = sorted((n for n in self.colour if self._holder(n) is None), key=lambda n: self.colour[n])
        self.seen |= set(loose)
        others = sorted(env.task_spec.get("visible_objects") or {})
        status = (f"You turn round the kitchen. {COUNT[len(comps)]} compartments have handles; from your left round to "
                  f"your right they are {listing(comps)}. Near each locked compartment a reader pad of its colour lies on "
                  "the counter.")
        if loose:
            status += f" On the counters you see {listing([self._noun(n) for n in loose])}."
        if others:
            status += f" Other objects on the counters: {listing([self._noun(n) for n in others])}."
        return {"status": status, "reading": "", "images": None}

    def _put(self, t):
        cs = [w.lower() for w in t.split() if w.lower() in COLOUR_NAMES]
        if len(cs) != 2 or " ON " not in f" {t} ":
            raise Invalid("say PUT THE c TOKEN ON THE p PAD with two lock colours")
        tok, pad = cs
        if self.token.get(tok) not in self.seen:
            raise Invalid(f"you know of no {tok} token")
        if not self.num or pad not in self.pad:
            raise Invalid(f"you know of no {pad} pad" + ("" if self.num else "; look around the kitchen first"))
        env, name, p = self.env, self.token[tok], self.env.pads[self.pad[pad]]
        for other in self.colour:
            at = env.sim.data.body_xpos[env.obj_body_id[other]]
            if other != name and token_on_pad(p, at, at[2], 0.):
                raise Invalid(f"the {self.colour[other]} token lies on the {pad} pad")
        holder = self._holder(name)
        if holder is not None and self._locked(holder):
            return {"status": f"The {tok} token lies in {self._label(holder)}, which does not open: it is locked.",
                    "reading": "", "images": None}
        if holder is not None:
            self._set_open(holder, True)
        env._set_object_pose(name, p["centre"][:2], p["top_z"], 0.)      # the task's own onto_pad
        env.sim.forward()
        if holder is not None:
            self._set_open(holder, False)
        self.tick(PUT_TICKS)
        took = f"take the {tok} token out of {self._label(holder)} and put it" if holder else f"put the {tok} token"
        return {"status": f"You {took} on the {pad} pad.", "reading": "", "images": None}

    def _number(self, t):
        ks = numbers(t)
        if len(ks) != 1:
            raise Invalid("name one compartment by its number")
        if ks[0] not in self.num:
            raise Invalid("no compartment has that number" + ("" if self.num else "; look around the kitchen first"))
        return ks[0]

    def _open(self, cid):
        name = self._label(cid)
        self._front(cid)
        if self._locked(cid):
            return {"status": f"You pull the handle of {name}. It does not move: it is locked.", "reading": "",
                    "images": None}
        self._set_open(cid, True); self.tick(LOOK_TICKS)
        inside, images = self._inside(cid), self.views()
        self._set_open(cid, False); self.tick(PUT_TICKS)
        self.seen |= {n for n in inside if n in self.colour}
        what = f"holds {listing([self._noun(n) for n in inside])}" if inside else "is empty"
        return {"status": f"You open {name}, look inside and shut it again.", "reading": f"{name.capitalize()} {what}.",
                "images": images}

    # ---- helpers on the task's own state ---------------------------------------------------------------------------
    def _locked(self, cid):
        return bool(self.env._lock.locked.get(cid, False))

    def _inside(self, cid):
        return [n for n in sorted(self.env.objects) if self.env._inside_compartment(n, cid)]

    def _holder(self, body):
        return next((c for c in self.order if self.env._inside_compartment(body, c)), None)

    def _front(self, cid):
        """The base to the compartment's stand-off pose; a drawer's backed off by the pull, as after a real opening."""
        comp = self.env.compartments[cid]; s = comp["standoff"]
        back = DRAWER_PULL_M if comp["kind"] == "drawer" else 0.
        self._goto(np.asarray(s["xy"], float) + front_normal(comp["rot"]) * float(s.get("side", 1.)) * back, s["yaw"])

    def _goto(self, xy, yaw):
        """The base's own joints set until the base frame is at (xy, yaw) (the blackout_search task's _teleport_base)."""
        env = self.env; m, d = env.sim.model, env.sim.data
        adr = [m.get_joint_qpos_addr(j) for j in self.base_joints]
        for _ in range(3):
            have_xy, have_yaw = env.base_pose()
            e = np.asarray(xy, float) - have_xy
            b = have_yaw - float(d.qpos[adr[2]])
            d.qpos[adr[0]] += math.cos(b) * e[0] + math.sin(b) * e[1]
            d.qpos[adr[1]] += -math.sin(b) * e[0] + math.cos(b) * e[1]
            d.qpos[adr[2]] += wrap(yaw - have_yaw)
            for j in self.base_joints:
                d.set_joint_qvel(j, 0.)
            env.sim.forward()

    def _set_open(self, cid, opened):
        """Every joint of the compartment to its open (or closed) limit, as the task's own set_joints; a drawer carries the
        objects inside it."""
        env = self.env; m, d = env.sim.model, env.sim.data; comp = env.compartments[cid]
        inside = self._inside(cid) if comp["kind"] == "drawer" else []
        body = m.jnt_bodyid[m.joint_name2id(comp["joints"][0])]; before = d.xpos[body].copy()
        for j in comp["joints"]:          # the compiled range (a locked one's saved by the task): the recorded one is wider
            lo, hi = env._saved_ranges.get(j) or m.jnt_range[m.joint_name2id(j)]
            d.set_joint_qpos(j, (hi if abs(hi) >= abs(lo) else lo) if opened else (lo if abs(lo) <= abs(hi) else hi))
            d.set_joint_qvel(j, 0.)
        env.sim.forward()
        shift = d.xpos[body] - before
        for n in inside:
            joint = env.objects[n].joints[0]
            q = np.array(d.get_joint_qpos(joint), float); q[:3] += shift
            d.set_joint_qpos(joint, q); d.set_joint_qvel(joint, np.zeros(6))
        env.sim.forward()

    # ---- privileged, for the self-test only ------------------------------------------------------------------------
    def _k(self, cid):
        return self.order.index(cid) + 1

    def oracle_lines(self):
        env = self.env
        lines = ["LOOK AROUND THE KITCHEN"]
        for i, cid in enumerate(env.chain_ids + ([env.dead_end_id] if env.dead_end_id else [])):
            c = env.pads[cid]["colour"].upper()
            if i == 1:
                lines.append(f"OPEN {self._k(cid)}")                       # its token is not on its pad yet: blocked
            lines += [f"PUT THE {c} TOKEN ON THE {c} PAD", f"OPEN {self._kind(cid).upper()} {self._k(cid)}"]
        decoys = [p["compartment"] for p in env.places.values()
                  if p["role"] == "hidden_distractor" and not p.get("dead_end")]
        return lines + [f"OPEN {self._k(cid)}" for cid in decoys]

    def oracle_answer(self, wrong=False):
        env = self.env
        if not wrong:
            return str(self._k(env.chain_ids[-1]))
        decoys = [p["compartment"] for p in env.places.values() if p["role"] == "hidden_distractor"]
        return str(self._k(decoys[0]))                # the dead end's decoy, else a decoy in a free compartment

    def submit(self, answer):
        cid = self.num[self._number(words(answer))]
        env = self.env
        if not self._locked(cid):                         # a locked compartment does not open: nothing goes on the tray
            self._set_open(cid, True)
            things = [n for n in self._inside(cid) if n not in self.colour]
            if things:                                    # the task's own placing of the target on the tray
                env._set_object_pose(things[0], [float(v) for v in env.tray["centre_xy"]], env.tray["top_z"],
                                     float(env.work["yaw"]))
                env.sim.forward()
            self._set_open(cid, False)
        self._goto(*self.start)
        return self.score()
