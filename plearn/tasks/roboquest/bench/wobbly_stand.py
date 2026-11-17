"""Wobbly Stand at the decision level: a four-legged stand with one short corner leg or two short adjacent legs (5, 10 or
15 mm short), and one, three or four shim plates (5, 10, 15 mm) on the counter.

Evidence actions: WATCH THE BALL (the helper puts the ball at the centre of the top and the scene runs for 3 s), PUT a plate
under a leg and REMOVE a plate (the helper lifts the stand, slides the plate under the foot or takes it out, sets the stand
down at the rigid resting pose of its own legs on the plates -- the model of the task's _place_configuration -- and runs
the same ball trial), and INSPECT THE LEGS (the robot sets the ball aside, holds the stand up level in front of the left
camera, seen from its front with the long feet at the camera's height, and puts everything back as it was). The reading of a
trial is where the ball is after 3 s, from the simulation; the reading of INSPECT is how much shorter the short legs are.
Legs are named as the robot sees them (front is nearer the robot); plates are numbered from left to right as the left camera
sees them. The answer names the plates under each leg; the score is the task's own check after the scene settles.

Known leak, kept as in the benchmark: the intro states each plate's thickness (the cameras show the plates), and the plate
set gives much of the hidden fact away (a single plate is the gap; with four plates the repeated thickness is the gap).
"""
import math
import re
import numpy as np
from roboquest.kitchen import yaw_quat_wxyz
from roboquest.stand import geometry as G, statics as S
from roboquest.tasks.wobbly_stand import PLACE_LIFT, _quat_wxyz
from .base import Task as Base, Invalid, CAMERAS, words

TRIAL_TICKS = 60               # 3 s: at every tilt a plate can leave, the ball reaches the rim within 3 s (physics gate)
CENTRE_M, CORNER_M, END_M = .01, .02, .03    # still at the centre; in a corner (near both rims); toward an end of an edge
HOLD_M, HOLD_YAW = .38, 30.    # INSPECT: the stand this far in front of the left camera, turned so that all four legs show
SECTORS = ("e", "ne", "n", "nw", "w", "sw", "s", "se")        # stand-frame directions at 0, 45, ... 315 degrees
SIDE_XY = {"e": (1, 0), "n": (0, 1), "w": (-1, 0), "s": (0, -1)}
ORDER = ("front left", "front right", "back left", "back right")
PUT = re.compile(r"^PUT (?:PLATE )?(\d+) UNDER (?:THE )?(FRONT|BACK) (LEFT|RIGHT)(?: LEG)?$")
REMOVE = re.compile(r"^REMOVE (?:PLATE )?(\d+)\b")


class Task(Base):
    # Hardest eval50 cell: one short leg with three plates (leg and gap unknown). WATCH gives the corner, INSPECT the gap, a
    # PUT checks it: 3 actions; without INSPECT, plate trials (PUT, REMOVE, PUT, maybe a stack) take up to 5. 8 leaves slack.
    T = 8
    SETTLE_FIRST = 50          # zero-action ticks after the answer is set down, as teleport_solution, before score() (+50)

    def __init__(self, env, render):
        super().__init__(env, render)
        m, d = env.sim.model, env.sim.data
        fwd = sum(-d.cam_xmat[m.camera_name2id(c)].reshape(3, 3)[:, 2] for c in CAMERAS[:2])
        fwd[2] = 0.; fwd /= np.linalg.norm(fwd); right = np.cross(fwd, [0., 0., 1.])
        _, rot = env._body_pose(env._stand_bid)
        self.leg = {}                                     # stand-frame leg -> agent's name, e.g. 'ne' -> 'back right'
        for leg in G.LEGS:
            v = rot @ np.array([*G.leg_foot_xy(leg), 0.])
            self.leg[leg] = ("front " if v @ fwd < 0 else "back ") + ("left" if v @ right < 0 else "right")
        assert len(set(self.leg.values())) == 4, self.leg
        self.byname = {v.upper(): k for k, v in self.leg.items()}
        self.side = {s: next(w for w in self.leg[a].split() if w in self.leg[b].split()) for s, (a, b) in G.SIDES.items()}
        pos = {n: d.body_xpos[env._shim_bids[n]].copy() for n in env.shim_geoms}
        self.num = {k + 1: n for k, n in enumerate(sorted(pos, key=lambda n: self.camera_x(pos[n])))}
        self.home = {n: d.qpos[slice(*env._shim_qadr[n])].copy() for n in env.shim_geoms}    # where each plate lies
        self.under = {}                                   # stand-frame leg -> plate bodies under it, bottom first

    def _mm(self, k):
        return int(round(self.env.shim_geoms[self.num[k]]["thickness_mm"]))

    def _plate(self, body):
        return next(k for k, b in self.num.items() if b == body)

    # ---- what the agent is told ------------------------------------------------------------------------------------
    def intro(self):
        n, colour = len(self.num), self.env.spec["ball"]["colour"]
        lines = [f"A small wooden stand with four legs stands on the counter, with {'an' if colour[0] in 'aeiou' else 'a'} "
                 f"{colour} ball at the centre of its top. Its legs are named as the robot sees them: front left, front right, "
                 "back left and back right (front is nearer the robot)."]
        if n == 1:
            lines.append(f"One shim plate lies on the counter: plate 1, {self._mm(1)} mm thick.")
        else:
            each = [f"plate {k} is {self._mm(k)} mm thick" for k in self.num]
            lines.append(f"{n} shim plates lie on the counter, numbered from left to right as the left camera sees them: "
                         + ", ".join(each[:-1]) + f" and {each[-1]}.")
        return " ".join(lines)

    def help(self):
        return "\n".join([
            "WATCH THE BALL - put the ball at the centre of the stand's top and watch it for 3 seconds",
            "PUT PLATE p UNDER THE <leg> LEG - lift the stand, put plate p under that leg (on any plates already there), "
            "set the stand down, then watch the ball as WATCH THE BALL does",
            "REMOVE PLATE p - lift the stand, take plate p from under its leg and put it back on the counter, set the stand "
            "down, then watch the ball",
            "INSPECT THE LEGS - set the ball aside, hold the stand up in front of the left camera to compare its legs, then "
            "put everything back as it was",
            "SUBMIT <leg>: p q, <leg>: r - your answer: the plates to put under each leg (two numbers stack two plates); "
            "the other plates stay on the counter, e.g. SUBMIT FRONT LEFT: 2",
            "(<leg> is FRONT LEFT, FRONT RIGHT, BACK LEFT or BACK RIGHT)"])

    # ---- actions ---------------------------------------------------------------------------------------------------
    def act(self, line):
        t = words(line).replace("-", " ")
        if t in ("WATCH THE BALL", "WATCH BALL"):
            return self._trial("You put the ball at the centre of the stand's top and watch it for 3 seconds.")
        if t in ("INSPECT THE LEGS", "INSPECT LEGS", "INSPECT THE FEET", "INSPECT FEET"):
            return self._inspect()
        put, rem = PUT.match(t), REMOVE.match(t)
        if put:
            k, leg = int(put[1]), self.byname[f"{put[2]} {put[3]}"]
            body = self._known(k); was = self._leg_of(body)
            if was == leg:
                return {"status": f"Plate {k} is under the {self.leg[leg]} leg already.", "reading": "", "images": None}
            if was:
                self.under[was].remove(body)
            self.under.setdefault(leg, []).append(body)
            did = (f"You take plate {k} from under the {self.leg[was]} leg and put it" if was else f"You put plate {k}")
            return self._trial(f"{did} under the {self.leg[leg]} leg, set the stand down, put the ball at the centre of "
                               "its top and watch it for 3 seconds.")
        if rem:
            k = int(rem[1]); body = self._known(k); was = self._leg_of(body)
            if not was:
                return {"status": f"Plate {k} lies on the counter already.", "reading": "", "images": None}
            self.under[was].remove(body)
            return self._trial(f"You take plate {k} from under the {self.leg[was]} leg and put it back on the counter, set "
                               "the stand down, put the ball at the centre of its top and watch it for 3 seconds.")
        raise Invalid("not an action of this task")

    def _known(self, k):
        if k not in self.num:
            raise Invalid(f"there is no plate {k}")
        return self.num[k]

    def _leg_of(self, body):
        return next((leg for leg, bs in self.under.items() if body in bs), None)

    def _now(self):
        rows = [f"{self.leg[leg]} leg: plate{'s' if len(bs) > 1 else ''} " + " and ".join(str(self._plate(b)) for b in bs)
                for leg, bs in self.under.items() if bs]
        return "Under the legs now: " + "; ".join(rows) + "." if rows else "No plate is under a leg now."

    def _arrange(self):
        """The stand on its plates at the rigid resting pose of its own built legs (the model of _place_configuration), each
        named plate stacked under its foot, every other plate where it lay at the start, the ball at the top's centre."""
        env = self.env; g = env.shim_geoms; sg = env.stand_geom
        lengths = {leg: -f[2] - G.TOP_THICKNESS + G.FOOT_RADIUS + sum(g[b]["thickness"] for b in self.under.get(leg, ()))
                   for leg, f in sg["feet_local"].items()}           # leg length (glide included) plus the plates under it
        pose = S.resting_pose(lengths, sg["com_local"])
        pos, rot = env._stand_world_pose(pose["rotation"], pose["z0"])
        env._set_free_joint(env._stand_qadr, env._stand_vadr, pos, _quat_wxyz(rot))
        for b, q in self.home.items():
            env._set_free_joint(env._shim_qadr[b], env._shim_vadr[b], q[:3], q[3:])
        quat = yaw_quat_wxyz(env.work["yaw"] + math.radians(env.spec["stand"]["yaw_deg"]))
        for leg, bs in self.under.items():
            foot = pos + rot @ np.asarray(sg["feet_local"][leg]); z = env.work["top_z"]
            for b in bs:                                  # a stack: each plate on the one below, a hair apart
                env._set_free_joint(env._shim_qadr[b], env._shim_vadr[b],
                                    [foot[0], foot[1], z + g[b]["thickness"] / 2 + PLACE_LIFT / 4], quat)
                z += g[b]["thickness"] + PLACE_LIFT / 4
        env._set_free_joint(env._ball_qadr, env._ball_vadr, pos + rot @ np.array([0., 0., G.BALL_RADIUS + PLACE_LIFT]),
                            [1., 0., 0., 0.])
        env.sim.forward()

    def _trial(self, did):
        self._arrange(); self.tick(TRIAL_TICKS)
        return {"status": did + " " + self._now(), "reading": self._ball(), "images": self.views()}

    def _toward(self, x, y):
        s = SECTORS[int(round(math.degrees(math.atan2(y, x)) / 45.)) % 8]
        return f"{self.leg[s]} corner" if s in self.leg else f"{self.side[s]} edge"

    def _ball(self):
        """Where the ball is after the trial, in the stand's top frame (what the cameras show)."""
        b = self.env.ball_state(); x, y = b["local_xy"]; r = math.hypot(x, y)
        travel = self.env.stand_geom["ball_travel"]; head = "After 3 seconds the ball is"
        if not b["within_top"]:
            return f"{head} off the stand, beyond its {self._toward(x, y)}."
        if r < CENTRE_M:
            return f"{head} still at the centre of the stand's top."
        if not b["at_rim"]:
            return f"{head} {100 * r:.0f} cm from the centre of the top, toward its {self._toward(x, y)}."
        if min(abs(x), abs(y)) >= travel - CORNER_M:
            return f"{head} in the {self._toward(x, y)} of the top."
        s, along = ("e" if x > 0 else "w", y) if abs(x) > abs(y) else ("n" if y > 0 else "s", x)
        corner = next(leg for leg, sg in G.LEG_SIGNS.items() if sg == (int(np.sign(x)), int(np.sign(y))))
        end = "" if abs(along) < END_M else f", toward the {self.leg[corner]} corner"
        return f"{head} against the {self.side[s]} edge of the top{end}."

    def _inspect(self):
        legs = {leg: -self.env.stand_geom["feet_local"][leg][2] for leg in G.LEGS}      # glide depth below the top
        short = {leg: round(1000 * (max(legs.values()) - v)) for leg, v in legs.items() if max(legs.values()) - v > .001}
        if not short:
            reading = "Held up level, the stand shows four legs of the same length."
        else:
            n = len(short); mm = " or ".join(str(v) for v in sorted(set(short.values())))
            names = sorted((self.leg[leg] for leg in short), key=ORDER.index)
            reading = (f"Held up level, the stand's {' and '.join(names)} leg"
                       f"{'s are each' if n > 1 else ' is'} {mm} mm shorter than its other "
                       f"{['', 'one', 'two', 'three'][4 - n]} leg{'s' if n < 3 else ''}.")
        return {"status": "You set the ball aside, hold the stand up level in front of the left camera, seen from its front "
                          "with the feet at the camera's height, then put the stand and the ball back as they were.",
                "reading": reading, "images": self._held() if self.render else None}

    def _held(self):
        """The three views while the stand is held up in front of the left camera; then everything is put back."""
        env = self.env; m, d = env.sim.model, env.sim.data; saved = env.sim.get_state()
        cid = m.camera_name2id(CAMERAS[0]); R = d.cam_xmat[cid].reshape(3, 3); c = d.cam_xpos[cid].copy()
        right, up, fwd = R[:, 0], R[:, 1], -R[:, 2]
        side = {w: s for s, w in self.side.items()}
        B = np.array([[*SIDE_XY[side["front"]], 0], [*SIDE_XY[side["right"]], 0], [0, 0, 1]], float).T   # stand frame
        A = np.column_stack([-fwd, right, up])          # the stand's front toward the camera, its right to the right
        a = math.radians(HOLD_YAW); yaw = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        rot = A @ B.T @ yaw
        pos = c + HOLD_M * fwd + G.STAND_HEIGHT * up      # the long legs' feet on the camera's horizontal mid-plane
        env._set_free_joint(env._stand_qadr, env._stand_vadr, pos, _quat_wxyz(rot))
        h = next(iter(env.helpers.values()), None)        # the ball set aside in a helper container
        if h is not None:
            hp = d.body_xpos[h["bid"]]
            env._set_free_joint(env._ball_qadr, env._ball_vadr,
                                [hp[0], hp[1], hp[2] - h["rim_local_z"] + G.BALL_RADIUS + .005], [1., 0., 0., 0.])
        env.sim.forward(); images = self.views()
        env.sim.set_state(saved); env.sim.forward()
        return images

    # ---- privileged, for the self-test only ------------------------------------------------------------------------
    def _right(self):
        return [k for k in self.num if self._mm(k) == int(self.env.spec["delta_mm"])]

    def oracle_lines(self):
        short = self.env.spec["short_legs"]; right = self._right()
        other = [k for k in self.num if k not in right]
        test, leg = (other[0], short[0]) if other else (right[0], S.opposite_legs(short)[0])     # one trial that fails
        lines = ["WATCH THE BALL", "INSPECT THE LEGS", f"PUT PLATE {test} UNDER THE {self.leg[leg].upper()} LEG",
                 f"REMOVE PLATE {test}"]
        return lines + [f"PUT PLATE {k} UNDER THE {self.leg[leg].upper()} LEG" for k, leg in zip(right, short)]

    def oracle_answer(self, wrong=False):
        short = self.env.spec["short_legs"]
        legs = S.opposite_legs(short) if wrong else short          # the right plates under the wrong legs (gate: not level)
        return ", ".join(f"{self.leg[leg].upper()}: {k}" for k, leg in zip(self._right(), legs))

    def submit(self, answer):
        t = words(answer).replace("-", " ").replace(":", " ")
        toks = [w for w in t.split() if w not in ("SUBMIT", "THE", "LEG", "PLATE", "PLATES", "AND", "UNDER")]
        under, leg, used, i = {}, None, set(), 0
        while i < len(toks):
            if toks[i] in ("FRONT", "BACK") and i + 1 < len(toks) and toks[i + 1] in ("LEFT", "RIGHT"):
                leg = self.byname[f"{toks[i]} {toks[i + 1]}"]; under.setdefault(leg, []); i += 2
            elif toks[i].isdigit() and leg is not None and int(toks[i]) in self.num and int(toks[i]) not in used:
                used.add(int(toks[i])); under[leg].append(self.num[int(toks[i])]); i += 1
            else:
                raise Invalid("the answer names legs and the plates under them, each plate once, e.g. FRONT LEFT: 2")
        if not used:
            raise Invalid("the answer puts no plate under a leg")
        self.under = under; self._arrange(); self.tick(self.SETTLE_FIRST)
        return self.score()
