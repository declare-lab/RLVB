"""Marked Mugs at the decision level: 3, 4 or 5 mugs or bowls, each with two balls inside and a colour label under its bottom,
and one colour pad per vessel at the back of the counter. Some vessels are in another part of the kitchen or behind a tall
object; at the uncover level a cloche stands over a vessel or a plate lies on a pad and hides its colour.

Evidence actions: LIFT MUG i (the helper lifts the vessel with its two balls kept in it and holds its bottom up to the wrist
camera, then puts it back exactly where it stood), LOOK AROUND THE KITCHEN (finds the vessels in other parts of the kitchen),
LOOK BEHIND an object, LIFT THE CLOCHE, LIFT THE PLATE (the task's own uncover step). The reading of a lift is the label colour.
A vessel gets a number when it is found; the visible ones are numbered from left to right as the left camera sees them. The
answer stands every named vessel upright on the pad of the named colour with its own two balls inside (the task's own
placement, with the pad taken from the answer instead of the label).
"""
import re

import mujoco
import numpy as np
from roboquest import observe as OB
from roboquest.kitchen import yaw_matrix, yaw_quat_wxyz
from roboquest.mugs.geometry import BALL_RADIUS, MUG_REST_CLEARANCE

from .base import Task as Base, Invalid, words, numbers, CAMERAS

HOLD_M = .20                   # the held vessel's label stands this far in front of the wrist camera


def noun(kind):
    return str(kind or "object").replace("_", " ").strip()


def span(word, n):
    return f"{word} 1" if n == 1 else f"{word}s 1 and 2" if n == 2 else f"{word}s 1 to {n}"


class Task(Base):
    # A good agent lifts the vessels (the last label follows from the others once every pad colour is known) and uses every
    # look/lift-cover action. The hardest eval50 instances (5 vessels and one look or cover action) need 6 if every vessel
    # is lifted; 8 leaves 2 for slack.
    T = 8

    def __init__(self, env, render):
        super().__init__(env, render)
        self.geo = geo = env.geometry; obs = env.task_spec.get("observability") or {}; d = env.sim.data
        self.noun = geo["vessel"]
        realised = {r["object"]: r for r in obs.get("realised", [])}
        hidden = {v for v in geo["vessels"] if v in realised}
        self.annex = sorted(v for v in hidden if realised[v]["realised"] == "annex")
        occ = {o["name"]: o for o in obs.get("occluders", [])}
        self.behind = self._named([(noun(occ[realised[v]["occluder"]]["kind"]), occ[realised[v]["occluder"]]["xy_world"], v)
                                   for v in sorted(hidden) if realised[v]["realised"] == "occluder"])
        covers = [c for c in obs.get("covers", []) if c.get("built")]
        self.under = {"cloche": c for c in covers if not c.get("place")}           # at most one cover per instance
        self.plate = next((c for c in covers if c.get("place")), None)
        self.covered = self.plate["place"][len("pad_"):] if self.plate else None   # the colour under the plate
        self.pads = sorted(geo["pads"], key=lambda c: self._image_x(geo["pads"][c]["center_world"]))
        self.known_pads = {c for c in self.pads if c != self.covered}
        self.num, self.known = {}, []
        vis = [v for v in geo["vessels"] if v not in hidden]
        for v in sorted(vis, key=lambda v: self._image_x(d.body_xpos[env.obj_body_id[v]])):
            self._number(v)
        self.looked, self.lifted = set(), set()

    def _named(self, items):
        """{agent's name: value} for (noun, world xy, value) items; two of a kind are told apart as left and right."""
        out, z = {}, float(self.env.work["top_z"])
        for kind in sorted({k for k, _, _ in items}):
            same = sorted(((self._image_x([*xy[:2], z]), i), v) for i, (k, xy, v) in enumerate(items) if k == kind)
            for j, (_, v) in enumerate(same):
                tag = "" if len(same) == 1 else ("left ", "right ")[j] if len(same) == 2 else f"{j + 1}. "
                out[tag + kind] = v
        return out

    def _image_x(self, pos, camera=CAMERAS[0]):
        """Horizontal image coordinate of a world point in a camera (larger is further right)."""
        m, d = self.env.sim.model, self.env.sim.data; cid = m.camera_name2id(camera)
        rel = np.asarray(pos, float) - d.cam_xpos[cid]; X = d.cam_xmat[cid].reshape(3, 3)
        return float(rel @ X[:, 0]) / max(float(-rel @ X[:, 2]), 1e-6)

    def _number(self, vessel):
        self.known.append(vessel); self.num[len(self.known)] = vessel; return len(self.known)

    def _label(self, vessel):
        return next(k for k, v in self.num.items() if v == vessel)

    def _pad_name(self, colour):
        return "the pad under the plate" if colour == self.covered and colour not in self.known_pads else f"the {colour} pad"

    def _nearest(self, vessel):
        """The pad nearest the vessel where it stands now (task-frame distance)."""
        xy = OB.world_to_frame_xy(self.env, self.env.sim.data.body_xpos[self.env.obj_body_id[vessel]][:2])
        return min(self.pads, key=lambda c: float(np.linalg.norm(np.asarray(self.geo["pads"][c]["center_xy"]) - xy)))

    # ---- what the agent is told ------------------------------------------------------------------------------------
    def intro(self):
        n, N = len(self.known), self.noun.capitalize()
        lines = [f"On the work counter you see {span(self.noun, n)}, numbered from left to right as the left camera sees "
                 f"{'it' if n == 1 else 'them'}; each holds two balls." if n else f"You see no {self.noun}s on the work counter."]
        lines.append(f"At the back of the counter lie {len(self.pads)} colour pads, from left to right: "
                     + ", ".join("a pad under a plate" if c == self.covered else c for c in self.pads) + ".")
        if n:
            lines.append("; ".join(f"{N if k == 1 else self.noun} {k} stands nearest {self._pad_name(self._nearest(v))}"
                                   for k, v in self.num.items()) + ".")
        if self.behind:
            lines.append(f"{'A tall object stands' if len(self.behind) == 1 else 'Tall objects stand'} on the counter: "
                         + ", ".join("the " + x for x in self.behind) + ".")
        if self.under:
            lines.append("A cloche stands on the counter.")
        helpers = sorted(ref["entry"]["kind"] for ref in getattr(self.env, "helpers", {}).values())
        if helpers:
            what = f"two {helpers[0]}s" if helpers == helpers[:1] * 2 else " and ".join("a " + h for h in helpers)
            lines.append(f"Also on the counter: {what}, empty and not among the numbered {self.noun}s.")
        return " ".join(lines)

    def help(self):
        V = self.noun.upper()
        out = [f"LIFT {V} i - lift {self.noun} i with its two balls kept in it, hold its bottom up to the wrist camera, put it "
               "back",
               f"LOOK AROUND THE KITCHEN - look for {self.noun}s in the other parts of the kitchen"]
        out += [f"LOOK BEHIND THE {name.upper()}" for name in self.behind]
        out += [f"LIFT THE {name.upper()} - lift it and set it aside" for name in self.under]
        if self.plate:
            out.append("LIFT THE PLATE - take the plate off the pad it lies on")
        out.append(f"SUBMIT 1:red 2:blue ... - stand every named {self.noun} upright on the pad of the named colour with its "
                   f"two balls inside (your answer; one {self.noun} per pad, colours of pads you have seen)")
        return "\n".join(out)

    # ---- actions ---------------------------------------------------------------------------------------------------
    def act(self, line):
        t = " ".join(w for w in words(line).split() if w != "THE")
        for verb in (f"LIFT {self.noun.upper()}", "LIFT VESSEL"):
            if t == verb or t.startswith(verb + " "):
                ks = numbers(t[len(verb):])
                if len(ks) != 1 or ks[0] not in self.num:
                    raise Invalid(f"name one known {self.noun} by its number")
                return self._lift(ks[0])
        if t in ("LOOK AROUND KITCHEN", "LOOK AROUND"):
            if "around" in self.looked:
                return {"status": "You have looked around the kitchen already.", "reading": "", "images": None}
            self.looked.add("around")
            found = [self._number(v) for v in self.annex]
            status = (f"In another part of the kitchen you find {self._list(found)}." if found
                      else f"You find no other {self.noun}s in the kitchen.")
            return {"status": status, "reading": "", "images": None}
        for name, vessel in self.behind.items():
            if t == f"LOOK BEHIND {name.upper()}":
                if name in self.looked:
                    return {"status": f"You have looked behind the {name} already.", "reading": "", "images": None}
                self.looked.add(name)
                return {"status": f"Behind the {name} you find {self.noun} {self._number(vessel)}.", "reading": "",
                        "images": None}
        for name, cover in self.under.items():
            if t == f"LIFT {name.upper()}":
                if name in self.lifted:
                    return {"status": f"The {name} is lifted already.", "reading": "", "images": None}
                self._uncover(name)
                k = self._number(cover["object"])
                return {"status": f"You lift the {name} and set it aside. Under it you find {self.noun} {k}.",
                        "reading": f"{self.noun.capitalize()} {k} stands nearest {self._pad_name(self._nearest(cover['object']))}.",
                        "images": self.views()}
        if self.plate and t == "LIFT PLATE":
            if "plate" in self.lifted:
                return {"status": "The plate is off the pad already.", "reading": "", "images": None}
            self._uncover("plate"); self.known_pads.add(self.covered)
            return {"status": "You take the plate off the pad it lay on and set it down beside it.",
                    "reading": f"The pad under the plate is {self.covered}.", "images": self.views()}
        raise Invalid("not an action of this task")

    def _uncover(self, name):
        self.env._uncover_steps(); self.lifted.add(name); self.tick(10)   # the task's own C3 step (one cover per instance)

    def _lift(self, k):
        """Hold vessel k (its balls kept in it) up to the wrist camera, bottom toward it, take the views, put it back exactly."""
        env, m, d = self.env, self.env.sim.model, self.env.sim.data
        vessel = self.num[k]; info = self.geo["vessels"][vessel]; balls = info["original_balls"]
        joints = [info["joint_name"]] + [self.geo["balls"][b]["joint_name"] for b in balls]
        saved = [(j, np.array(d.get_joint_qpos(j), float).copy(), np.array(d.get_joint_qvel(j), float).copy()) for j in joints]
        bid = env.obj_body_id[vessel]; p0, R0 = d.body_xpos[bid].copy(), d.body_xmat[bid].reshape(3, 3).copy()
        cid = m.camera_name2id(CAMERAS[2]); X = d.cam_xmat[cid].reshape(3, 3); r, u, t = X[:, 0], X[:, 1], X[:, 2]
        R = np.column_stack([r, -u, -t])                 # the vessel's up axis points away from the camera: its bottom faces it
        p = d.cam_xpos[cid] - HOLD_M * t - R @ np.asarray(info["label_center_local"], float)
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, R.flatten())
        d.set_joint_qpos(info["joint_name"], np.r_[p, quat]); d.set_joint_qvel(info["joint_name"], np.zeros(6))
        for b in balls:                                   # the balls move with the vessel, inside it
            j = self.geo["balls"][b]["joint_name"]; q = np.array(d.get_joint_qpos(j), float).copy()
            d.set_joint_qpos(j, np.r_[p + R @ R0.T @ (q[:3] - p0), q[3:]]); d.set_joint_qvel(j, np.zeros(6))
        env.sim.forward(); images = self.views()
        for j, q, v in saved:
            d.set_joint_qpos(j, q); d.set_joint_qvel(j, v)
        env.sim.forward()
        return {"status": f"You lift {self.noun} {k}, keeping its two balls in it, hold its bottom up to the wrist camera and "
                          "put it back where it stood.",
                "reading": f"The label under {self.noun} {k} is {info['label']}.", "images": images}

    def _list(self, ks):
        return (f"{self.noun} " if len(ks) == 1 else f"{self.noun}s ") + " ".join(str(k) for k in ks)

    # ---- privileged, for the self-test only ------------------------------------------------------------------------
    def oracle_lines(self):
        n = len(self.known) + len(self.annex) + len(self.behind) + len(self.under)
        lines = (["LOOK AROUND THE KITCHEN"] if self.annex else []) + [f"LOOK BEHIND THE {x.upper()}" for x in self.behind]
        lines += [f"LIFT THE {x.upper()}" for x in self.under] + (["LIFT THE PLATE"] if self.plate else [])
        return lines + [f"LIFT {self.noun.upper()} {k}" for k in range(1, n + 1)]

    def oracle_answer(self, wrong=False):
        labels = {k: self.geo["vessels"][v]["label"] for k, v in self.num.items()}
        if wrong:
            labels[1], labels[2] = labels[2], labels[1]
        return " ".join(f"{k}:{c}" for k, c in sorted(labels.items()))

    def submit(self, answer):
        pairs = [(int(k), c.lower()) for k, c in re.findall(r"(\d+)\s*[:=]?\s*([A-Z]+)", words(answer))]
        ks, cs = [k for k, _ in pairs], [c for _, c in pairs]
        if not pairs or len(set(ks)) < len(ks) or any(k not in self.num for k in ks):
            raise Invalid(f"the answer must give each named {self.noun} (a known number) one colour, e.g. 1:red 2:blue")
        if len(set(cs)) < len(cs) or any(c not in self.known_pads for c in cs):
            raise Invalid("each colour must be a pad you have seen, one pad per " + self.noun)
        for k, colour in pairs:
            self._place(self.num[k], colour)
        return self.score()

    def _place(self, vessel, colour):
        """The task's _place_vessel_on_pad with the pad from the answer: upright on the pad, its own two balls inside."""
        d = self.env.sim.data; info = self.geo["vessels"][vessel]; yaw = float(self.env.work["yaw"])
        pos = np.asarray(self.geo["pads"][colour]["center_world"], float) + np.array(
            [0., 0., -float(info["bottom_offset_z"]) + MUG_REST_CLEARANCE])
        d.set_joint_qpos(info["joint_name"], np.r_[pos, np.asarray(yaw_quat_wxyz(yaw))])
        d.set_joint_qvel(info["joint_name"], np.zeros(6))
        centre = np.asarray(info["interior_center"], float)
        floor_z = float(info["interior_floor_z"]) + BALL_RADIUS + float(info["ball_drop"])
        for i, ball in enumerate(info["original_balls"]):
            offset = yaw_matrix(yaw) @ np.array([(2 * i - 1) * float(info["ball_spread"]) + centre[0], centre[1], floor_z])
            j = self.geo["balls"][ball]["joint_name"]
            d.set_joint_qpos(j, np.r_[pos + offset, 1., 0., 0., 0.]); d.set_joint_qvel(j, np.zeros(6))
        self.env.sim.forward()
