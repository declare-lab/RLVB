"""Painted Cubes at the decision level: 4, 6 or 8 white cubes with marks (coloured shapes) on some faces, a public rule about
the marks, a blue bin for the cubes that satisfy the rule and a yellow bin for the rest. The marks that decide a cube's class
are on faces the start cameras cannot see (the bottom, and one or two far sides); some cubes are in another part of the kitchen,
behind a tall object or under a cover, and some carry a plate.

Evidence actions: LIFT CUBE c (the helper lifts the cube and holds it up to the wrist camera with its bottom corner turned to
the camera, so the bottom and the two far sides show; then it puts the cube back exactly where it stood), LOOK AT CUBE c (the
same with the top corner: the top and the two near sides show), LOOK AROUND THE KITCHEN (finds the cubes no camera shows),
LOOK BEHIND an object, LIFT a cover. The reading of LIFT / LOOK AT is the marks on the three faces turned to the camera. The intro
states what the start cameras show of each cube's top and near sides: a face counts as shown when one camera sees it within
about 75 degrees of square-on and rays to five points of its mark area reach it unblocked. A cube gets a number when it is
found; the visible ones are numbered from left to right as the left camera sees them. The answer drops the named cubes into
the blue bin and every other known cube into the yellow bin (the task's own drop_into_bin).
"""
import math

import mujoco
import numpy as np
from roboquest import observe as OB, scatter as S
from roboquest.cubes import hiding as CH, marks as M

from .base import Task as Base, Invalid, words, numbers, CAMERAS

HOLD_M = .16                   # the held cube's centre stands this far in front of the wrist camera
SEEN_COS = .25                 # a face is read only from a camera that sees it within about 75 degrees of square-on
MARK_PTS = .014                # ray targets: the face centre and four points this far off it along both face axes
GROUPS = np.array([0, 1, 1, 0, 0, 0], np.uint8)   # the geom groups the policy cameras render (group 0 is collision)
ASIDE_GAP, ASIDE_SALT = .015, 0x0C3E               # as the task's move_covers_aside
COVER_NOUN = {"cloche_small": "cloche", "bowl": "bowl"}


def noun(kind):
    return str(kind or "object").replace("_", " ").strip()


def mark(m):
    return f"a {m['colour']} {m['shape']}" if m else "no mark"


def span(word, n):
    return f"{word} 1" if n == 1 else f"{word}s 1 and 2" if n == 2 else f"{word}s 1 to {n}"


class Task(Base):
    # A good agent lifts every cube (the deciding faces are never in a start camera), looks at the top half of a cube whose top
    # or near sides it has not seen when the bottom half does not decide it, and uses every look/lift-cover action. The hardest
    # eval50 instance (71c91258f2896684) needs 14 of these (8 lifts, 4 look-at, 2 covers); 17 leaves 3 for slack.
    T = 17

    def __init__(self, env, render):
        super().__init__(env, render)
        ts = env.task_spec; obs = ts.get("observability") or {}; m, d = env.sim.model, env.sim.data
        self.eye = np.mean([d.cam_xpos[m.camera_name2id(c)] for c in CAMERAS], axis=0)
        self.marks = {n: dict(ts["objects"][n].get("marks") or {}) for n in env.cube_names}
        realised = {r["object"]: r for r in obs.get("realised", [])}
        hidden = {n for n in env.cube_names if n in realised}
        out = [n for n in env.cube_names if n not in hidden and not self._in_view(n)]   # on the counter, in no camera
        self.annex = sorted(out + [n for n in hidden if realised[n]["realised"] == "annex"])
        occ = {o["name"]: o for o in obs.get("occluders", [])}
        self.behind = self._named([(noun(occ[realised[n]["occluder"]]["kind"]), occ[realised[n]["occluder"]]["xy_world"], n)
                                   for n in sorted(hidden) if realised[n]["realised"] == "occluder"])
        self.covers = [c for c in obs.get("covers", []) if c.get("built")]
        self.under = self._named([(COVER_NOUN.get(c["kind"], noun(c["kind"])), c["xy_world"], c) for c in self.covers])
        self.num, self.known, self.shown = {}, [], {}  # number -> cube; cubes in number order; cube -> what a camera showed
        vis = [n for n in env.cube_names if n not in hidden and n not in out]
        for n in sorted(vis, key=lambda n: self._image_x(d.body_xpos[env.obj_body_id[n]])):
            self._number(n); self.shown[n] = self._shown(n)
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

    def _number(self, cube):
        self.known.append(cube); self.num[len(self.known)] = cube; return len(self.known)

    def _label(self, cube):
        return next(k for k, c in self.num.items() if c == cube)

    # ---- what the cameras see -------------------------------------------------------------------------------------
    def _image_x(self, pos, camera=CAMERAS[0]):
        """Horizontal image coordinate of a world point in a camera (larger is further right)."""
        m, d = self.env.sim.model, self.env.sim.data; cid = m.camera_name2id(camera)
        rel = np.asarray(pos, float) - d.cam_xpos[cid]; X = d.cam_xmat[cid].reshape(3, 3)
        return float(rel @ X[:, 0]) / max(float(-rel @ X[:, 2]), 1e-6)

    def _sees(self, camera, point, body):
        """Is ``point`` (just inside ``body``) in the camera's image, with ``body`` the first thing a ray from the camera meets?"""
        m, d = self.env.sim.model, self.env.sim.data; cid = m.camera_name2id(camera)
        c = d.cam_xpos[cid]; X = d.cam_xmat[cid].reshape(3, 3); rel = np.asarray(point, float) - c
        depth, edge = float(-rel @ X[:, 2]), .97 * math.tan(math.radians(float(m.cam_fovy[cid])) / 2)
        if depth <= 0 or max(abs(rel @ X[:, 0]), abs(rel @ X[:, 1])) > edge * depth:
            return False
        start = c + .03 * rel / np.linalg.norm(rel); gid = np.full((1, 1), -1, np.int32)
        x = mujoco.mj_ray(m._model, d._data, start, np.asarray(point, float) - start, GROUPS, 1, -1, gid)
        return bool(x >= 0 and m.geom_bodyid[int(gid[0, 0])] == body)

    def _face(self, cube, face, reach=MARK_PTS, grid=((0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))):
        """(body id, world normal, ray targets 1 mm under the face, inside the cube: the face centre plus ``reach`` along the
        face axes for each grid step)."""
        d = self.env.sim.data; bid = self.env.obj_body_id[cube]; R = d.body_xmat[bid].reshape(3, 3)
        local = M.FACE_NORMALS[face]; n = R @ np.array(local, float)
        a, b = (R[:, i] for i in range(3) if local[i] == 0)
        centre = d.body_xpos[bid] + (M.HALF_SIZE_M - .001) * n
        return bid, n, [centre + reach * (i * a + j * b) for i, j in grid]

    def _seen(self, cube, face):
        """Does one start camera show the whole mark area of this face, within about 75 degrees of square-on?"""
        d, m = self.env.sim.data, self.env.sim.model
        bid, n, pts = self._face(cube, face)
        for cam in CAMERAS:
            v = d.cam_xpos[m.camera_name2id(cam)] - pts[0]
            if n @ v >= SEEN_COS * np.linalg.norm(v) and all(self._sees(cam, p, bid) for p in pts):
                return True
        return False

    def _in_view(self, cube):
        """Does any camera show any part of the cube (a 3 x 3 grid of points on every face but the bottom)?"""
        d, m = self.env.sim.data, self.env.sim.model
        for face in M.FACE_NAMES:
            bid, n, pts = self._face(cube, face, M.HALF_SIZE_M - .004, [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)])
            if n[2] < -.5:
                continue
            for cam in CAMERAS:
                if n @ (d.cam_xpos[m.camera_name2id(cam)] - pts[4]) > 0 and any(self._sees(cam, p, bid) for p in pts):
                    return True
        return False

    def _corners(self, cube):
        """((bottom, far side, far side), (top, near side, near side)) of a cube where it stands now; far sides face away
        from the cameras."""
        d = self.env.sim.data; bid = self.env.obj_body_id[cube]; R = d.body_xmat[bid].reshape(3, 3)
        world = {f: R @ np.array(v, float) for f, v in M.FACE_NORMALS.items()}
        top = max(world, key=lambda f: world[f][2]); bottom = min(world, key=lambda f: world[f][2])
        to_eye = self.eye - d.body_xpos[bid]
        sides = sorted((f for f in world if f not in (top, bottom)), key=lambda f: float(world[f] @ to_eye))
        return (bottom, sides[0], sides[1]), (top, sides[3], sides[2])

    def _shown(self, cube):
        """What the cameras show now of the top and the near sides: per face its mark (None: no mark), or False (not seen)."""
        return [self.marks[cube].get(f) if self._seen(cube, f) else False for f in self._corners(cube)[1]]

    @staticmethod
    def _half_text(heads, marks):
        text = [mark(x) if x is not False else "not visible" for x in marks]
        return f"{heads[0]}: {text[0]}; {heads[1]}: {text[1]} and {text[2]}"

    # ---- what the agent is told ------------------------------------------------------------------------------------
    def intro(self):
        n = len(self.known)
        lines = [f"On the work counter you see white {span('cube', n)}, numbered from left to right as the left camera sees "
                 f"{'it' if n == 1 else 'them'}." if n else "You see no cubes on the work counter."]
        lines.append("Each face of a cube is blank or carries one mark (a coloured shape). From where you stand you see at "
                     "best the top and the two near sides of a cube; its bottom and its two far sides are turned away.")
        if n:
            lines.append("The cameras show: " + "; ".join(
                f"cube {k} - {self._half_text(('top', 'near sides'), self.shown[c])}" for k, c in self.num.items()) + ".")
        for o in self.env.spec.get("occluders") or ():
            if o["cube"] in self.num.values():
                verb = "lies flat on" if o.get("style") == "flat" else "leans on"
                lines.append(f"A plate {verb} cube {self._label(o['cube'])}.")
        bins = {lid: b["opening_center_world"] for lid, b in self.env.task_spec["bins"].items()}
        left = min(bins, key=lambda lid: self._image_x(bins[lid]))
        lines.append(f"The blue bin stands on the {'left' if left == 'blue' else 'right'} and the yellow bin on the "
                     f"{'left' if left == 'yellow' else 'right'} of the counter.")
        for one, many, names in (("A tall object stands", "Tall objects stand", self.behind),
                                 ("A cover stands", "Covers stand", self.under)):
            if names:
                lines.append(f"{one if len(names) == 1 else many} on the counter: {', '.join('the ' + n for n in names)}.")
        return " ".join(lines)

    def help(self):
        out = ["LIFT CUBE c - lift cube c, hold it up to the wrist camera with its bottom corner toward the camera (the bottom "
               "and the two far sides show), put it back",
               "LOOK AT CUBE c - lift cube c, hold it up to the wrist camera with its top corner toward the camera (the top "
               "and the two near sides show), put it back",
               "LOOK AROUND THE KITCHEN - look for the cubes the cameras do not show"]
        out += [f"LOOK BEHIND THE {name.upper()}" for name in self.behind]
        out += [f"LIFT THE {name.upper()} - lift it and set it aside" for name in self.under]
        out.append("SUBMIT a b - put the named cubes in the blue bin and every other cube you know in the yellow bin (your "
                   "answer; SUBMIT NONE puts every cube in the yellow bin)")
        return "\n".join(out)

    # ---- actions ---------------------------------------------------------------------------------------------------
    def act(self, line):
        t = " ".join(w for w in words(line).split() if w != "THE")
        for verb, bottom in (("LIFT CUBE", True), ("LOOK AT CUBE", False)):
            if t == verb or t.startswith(verb + " "):
                ks = numbers(t[len(verb):])
                if len(ks) != 1 or ks[0] not in self.num:
                    raise Invalid("name one known cube by its number")
                return self._inspect(ks[0], bottom)
        if t in ("LOOK AROUND KITCHEN", "LOOK AROUND"):
            if "around" in self.looked:
                return {"status": "You have looked around the kitchen already.", "reading": "", "images": None}
            self.looked.add("around")
            found = [self._number(c) for c in self.annex]
            status = (f"Looking around the kitchen you find {self._list(found)}." if found
                      else "You find no other cubes in the kitchen.")
            return {"status": status, "reading": "", "images": None}
        for name, cube in self.behind.items():
            if t == f"LOOK BEHIND {name.upper()}":
                if name in self.looked:
                    return {"status": f"You have looked behind the {name} already.", "reading": "", "images": None}
                self.looked.add(name)
                return {"status": f"Behind the {name} you find cube {self._number(cube)}.", "reading": "", "images": None}
        for name, cover in self.under.items():
            if t == f"LIFT {name.upper()}":
                if name in self.lifted:
                    return {"status": f"The {name} is lifted already.", "reading": "", "images": None}
                self._set_aside(cover); self.lifted.add(name)
                k = self._number(cover["object"]); self.shown[cover["object"]] = self._shown(cover["object"])
                return {"status": f"You lift the {name} and set it aside. Under it you find cube {k}.",
                        "reading": f"The cameras show cube {k} - "
                                   f"{self._half_text(('top', 'near sides'), self.shown[cover['object']])}.",
                        "images": self.views()}
        raise Invalid("not an action of this task")

    def _inspect(self, k, bottom):
        """Hold cube k up to the wrist camera with one corner toward it, take the views, put the cube back exactly."""
        env, d = self.env, self.env.sim.data; cube = self.num[k]; joint = env.objects[cube].joints[0]
        faces = self._corners(cube)[0 if bottom else 1]
        q, v = np.array(d.get_joint_qpos(joint), float).copy(), np.array(d.get_joint_qvel(joint), float).copy()
        self._hold(cube, faces); images = self.views()
        d.set_joint_qpos(joint, q); d.set_joint_qvel(joint, v); env.sim.forward()
        corner, heads = ("bottom", ("bottom", "far sides")) if bottom else ("top", ("top", "near sides"))
        status = (f"You lift cube {k}, hold it up to the wrist camera with its {corner} corner toward the camera and put it "
                  "back where it stood.")
        reading = f"Cube {k} - {self._half_text(heads, [self.marks[cube].get(f) for f in faces])}."
        return {"status": status, "reading": reading, "images": images}

    def _hold(self, cube, faces):
        """Pose the cube in front of the wrist camera with the corner of these three faces toward it, the first face at the top
        of the image (no physics step is taken while it is held)."""
        m, d = self.env.sim.model, self.env.sim.data; cid = m.camera_name2id(CAMERAS[2])
        X = d.cam_xmat[cid].reshape(3, 3); r, u, t = X[:, 0], X[:, 1], X[:, 2]   # image right, image up, toward the camera
        W = np.column_stack([t / math.sqrt(3) + math.sqrt(2 / 3) * (math.cos(a) * u + math.sin(a) * r)
                             for a in (0., 2 * math.pi / 3, 4 * math.pi / 3)])
        N = np.column_stack([M.FACE_NORMALS[f] for f in faces]).astype(float)
        if np.linalg.det(W @ N.T) < 0:
            N = N[:, [0, 2, 1]]
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, (W @ N.T).flatten())
        joint = self.env.objects[cube].joints[0]
        d.set_joint_qpos(joint, np.r_[d.cam_xpos[cid] - HOLD_M * t, quat]); d.set_joint_qvel(joint, np.zeros(6))
        self.env.sim.forward()

    def _set_aside(self, cover):
        """Set one cover down on a free spot of the work surface (the task's move_covers_aside, for one cover): clear of every
        cube, bin, plate, Submit and the other covers where they stand now."""
        env, d = self.env, self.env.sim.data
        here = lambda bid: tuple(float(v) for v in OB.world_to_frame_xy(env, d.body_xpos[bid][:2]))
        body = lambda c: env.objects[c["name"]].root_body if c["name"] in env.objects else c["name"]
        keep = list(env.hiding_keepouts()) + [S.Disc(here(env.obj_body_id[c]), CH.CUBE_RADIUS) for c in env.cube_names]
        keep += [S.Disc(here(env.sim.model.body_name2id(body(c))), float(c.get("radius") or .11)) for c in self.covers]
        rng = np.random.default_rng([int((getattr(env, "instance", None) or {}).get("seed", 0)), ASIDE_SALT, len(self.lifted)])
        spot = CH.cover_aside_spot(rng, OB.work_surface_rect(env), keep, float(cover.get("radius") or .11), gap=ASIDE_GAP)
        joint = env.free_joint_name(body(cover))
        if spot is None or joint is None:          # no free spot: the task's own helper moves every cover aside
            env.move_covers_aside(); return
        bid = env.sim.model.body_name2id(body(cover)); world = env.frame_to_world([spot[0], spot[1], 0.])
        d.set_joint_qpos(joint, np.r_[world[0], world[1], d.body_xpos[bid][2] + .002, d.body_xquat[bid]])
        d.set_joint_qvel(joint, np.zeros(6)); env.sim.forward(); self.tick(4)

    def _list(self, ks):
        return ("cube " if len(ks) == 1 else "cubes ") + " ".join(str(k) for k in ks)

    # ---- privileged, for the self-test only ------------------------------------------------------------------------
    def _decided(self, cube, low, high):
        """Does a good agent know the class from the bottom half (low faces) and what it saw of the top half?"""
        rule = self.env.task_spec["rule"]
        known = [self.marks[cube].get(f) for f in low] + [x for x in (high or [False] * 3) if x is not False]
        if high is not None and False not in high:
            return True
        if rule["kind"] == "exactly_one":
            return sum(x is not None for x in known) >= 2
        return any(x is not None and M.matches(x, rule) for x in known)

    def oracle_lines(self):
        order = list(self.known) + self.annex + list(self.behind.values()) + [c["object"] for c in self.under.values()]
        lines = (["LOOK AROUND THE KITCHEN"] if self.annex else []) + [f"LOOK BEHIND THE {n.upper()}" for n in self.behind]
        lines += [f"LIFT THE {n.upper()}" for n in self.under]
        for k, cube in enumerate(order, 1):
            lines.append(f"LIFT CUBE {k}")
            if not self._decided(cube, self._corners(cube)[0], self.shown.get(cube)):
                lines.append(f"LOOK AT CUBE {k}")
        return lines

    def oracle_answer(self, wrong=False):
        targets = sorted(self._label(c) for c in self.env.task_spec["targets"])
        return " ".join(str(k) for k in (targets[1:] if wrong else targets)) or "NONE"

    def submit(self, answer):
        t = words(answer)
        ks = [] if t in ("NONE", "SUBMIT NONE", "NOTHING", "0") else numbers(t)
        if len(set(ks)) < len(ks) or any(k not in self.num for k in ks):
            raise Invalid("the answer names cubes that are not known")
        for k, cube in sorted(self.num.items()):             # one at a time, as the task's own teleport_solution drops them
            self.env.drop_into_bin(cube, "blue" if k in ks else "yellow")
        return self.score()
