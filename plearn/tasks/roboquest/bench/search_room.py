"""Search Room at the decision level: two or three named objects hidden in a lit kitchen -- in a drawer or a cabinet with
a handle, under a cloche on a counter, or standing on a counter that no start camera sees -- and one or two look-alike
decoys (same colour or same kind) hidden in other compartments.

Places. The intro states what the start cameras show. LOOK AROUND THE KITCHEN maps the furniture, which holds no hidden
fact, and numbers it from the robot's left round to its right (as locked_storage does): the compartments with handles
(drawer 3, cabinet 5) and the counter tops cut into sections of at most 1.2 m (counter 1 to counter S), each with its
direction and distance from the start pose. What stands on a counter shows only when the robot looks at that counter, so
an object in the open is found counter by counter (the design study's region-level search).
Evidence actions: OPEN k (a drawer is pulled out 0.40 m, the scripted motor's pull, and its contents ride along; a
cabinet door swings fully open), LOOK AT COUNTER k, LIFT THE CLOCHE ON COUNTER k (the task's own lift onto the cloche's
first set-down spot; it stays there). Each one moves the base to a stance in front of the place (the task's stand-off
pose, or one chosen by ray tests for a counter), poses the arm so the wrist camera looks at the place (the scripted
motor's look_inside, by inverse kinematics on the arm joints), takes the three images and puts the base, the arm and the
compartment back; no physics runs. The reading names what the place holds, then what else the cameras show. On a
counter, and for anything else, an object or a cloche counts as shown when it is at least 24 px tall in an image and a
ray from the camera reaches one of five points on it first, the same test in both versions (an object tucked into a
nook no camera reaches is not named). The answer names each object and the place it is taken from; the helpers take a
named object only if it is at that place, stand it on the tray slots of the task's own teleport_solution, settle and
score.
"""
import math
import re

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from roboquest.search.layout import (BASE_CLEARANCE_M, FLOOR_MARGIN_M, _blocked, _clearance, counter_patches,
                                     footprint_polygon, point_in_floor, point_in_polygon, world_from_local, yaw_matrix)
from roboquest.search.objects import POOL
from roboquest.search.visibility import camera_frusta, project, relative_bearing
from roboquest.tasks.search_room import TRAY_SLOTS

from .base import Task as Base, Invalid, words, CAMERAS, SIZE

PULL_M = .40               # a drawer opens this far (the scripted motor's DRAWER_PULL_M); the compiled range is wider
SECTION_M = 1.2            # a counter top is cut into sections no longer than this; one look sees one section
STAND_M = (.50, .60, .70, .85)   # base frame to the counter edge for a look (the task's SPOT_STANDOFF_M first)
REAR_M = .45               # the base reaches this far behind its frame (the agentview cameras sit 0.5 m behind it)
IK_TOL = (.03, 8.)         # a look pose missed by more than this (m, degrees) does not count as reached
# Wrist-camera poses, tried in order until the arm reaches one: (height, distance out of the front or the edge).
LOOK_DRAWER = ((.45, .05), (.55, .05), (.35, .10), (.55, .20))       # above the pulled drawer's floor
LOOK_CABINET = ((.15, .25), (.25, .35), (.35, .45), (.30, .25))      # above the cabinet's floor, in front of it
LOOK_SPOT = ((.42, .15), (.50, .25), (.35, .30), (.50, .05))         # above a counter spot, off the counter's edge
LOOK_COUNTER = ((.62, .15), (.55, .30), (.45, .30), (.40, .45), (.45, .15), (.35, .30))   # above a counter's edge
MIN_PX = 24                # an object shows in an image when it is at least this many pixels tall there
GROUPS = np.array([1, 1, 1, 0, 0, 0], np.uint8)    # geom groups the ray tests see (hidden handles are group 3)
ROBOT = ('robot0', 'mobilebase0', 'gripper0')
COUNT = ('No', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve',
         'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen')
OBJ = re.compile(r'(BLUE|RED|GREEN|YELLOW) (WATER BOTTLE|BOTTLE|MUG|CAN|TIN) (?:FROM|IN|ON|UNDER|AT) (?:CLOCHE ON )?'
                 r'(DRAWER|CABINET|DOOR|COMPARTMENT|COUNTER) (\d+)')


def wrap(a):
    return (float(a) + math.pi) % (2 * math.pi) - math.pi


def listing(names):
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]


def facing(cam, at, up):
    """Camera rotation (MuJoCo: looks along -z, image-up +y) at `cam` looking at `at`, image-up near `up`."""
    z = np.asarray(cam, float) - at; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


class Task(Base):
    # Budget: the hardest eval50 row (7d1f6f6d254c1140: 8 compartments, 16 counter sections, 1 cloche) has 25 places;
    # with LOOK AROUND THE KITCHEN an agent can inspect every place of every eval50 kitchen once in 26 actions, and a
    # good agent stops when it has found the targets (the median kitchen has 17 places). 28 leaves 2 for slips.
    T = 28

    def __init__(self, env, render):
        super().__init__(env, render)
        self.M, self.D = M, D = env.sim.model._model, env.sim.data._data
        self.base_q = np.array([M.jnt_qposadr[M.joint(f'mobilebase0_joint_mobile_{a}').id] for a in ('forward', 'side', 'yaw')])
        arm = [M.joint(f'robot0_joint{i}').id for i in range(1, 8)]
        self.arm_q, self.arm_d = np.array([M.jnt_qposadr[j] for j in arm]), np.array([M.jnt_dofadr[j] for j in arm])
        self.arm_lim = M.jnt_range[arm].T
        self.cam = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_CAMERA, CAMERAS[2])
        self.cam_body = {c: M.cam_bodyid[mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_CAMERA, c)] for c in CAMERAS}
        self.robot_geoms = np.array([g for g in range(M.ngeom) if M.body(M.geom_bodyid[g]).name.startswith(ROBOT)], int)
        self.base0, self.arm0 = D.qpos[self.base_q].copy(), D.qpos[self.arm_q].copy()
        self.start = env.base_pose()
        self.pool = sorted(n for n in env.objects if n.startswith('obj_'))
        self.lifted, self.mapped = set(), False
        self.comps = sorted(env.compartments, key=lambda c: self._order(env.compartments[c]['pos']))
        self.counters = sorted(self._sections(), key=lambda s: self._order(np.r_[s['xy'], s['top_z']]))
        self.seen = self._in_view(())                      # what the start cameras show

    # ---- geometry --------------------------------------------------------------------------------------------------
    def _order(self, pos):
        """From the robot's left round to its right (as locked_storage numbers its compartments), higher first."""
        return -round(math.degrees(relative_bearing(*self.start, pos[:2])), 2), -float(pos[2])

    def _where(self, xy):
        b = math.degrees(relative_bearing(*self.start, xy[:2])); d = float(np.linalg.norm(np.asarray(xy[:2]) - self.start[0]))
        a, side = abs(b), 'left' if b > 0 else 'right'
        w = ('in front of you' if a < 25 else f'ahead on your {side}' if a < 70 else f'on your {side}' if a < 110
             else f'behind you on your {side}' if a < 155 else 'behind you')
        return f'{w}, {d:.1f} m away'

    def _sections(self):
        """Counter-top patches (the task's own counter_patches) cut along their long side into pieces <= SECTION_M."""
        env, out = self.env, []
        for p in counter_patches(env._records):
            if min(p['size_xy']) < .30:                  # the task's spot samplers skip these too
                continue
            rec = env._records[p['counter']]
            along = 0 if p['size_xy'][0] >= p['size_xy'][1] else 1
            n = max(1, math.ceil(p['size_xy'][along] / SECTION_M - 1e-6))
            size = list(p['size_xy']); size[along] /= n
            for i in range(n):
                lc = list(p['offset_local'][:2]); lc[along] += -p['size_xy'][along] / 2 + (i + .5) * size[along]
                xy = world_from_local(rec['pos'], rec['rot'], [lc[0], lc[1], 0.])[:2]
                out.append(dict(xy=xy, top_z=float(p['top_z']), size=size, rec=rec, along=along,
                                poly=footprint_polygon(rec, lc, size)))
        return out

    def _pos(self, name):
        return self.D.xpos[self.env.obj_body_id[name]].copy()

    def _counter_of(self, name):
        """Number of the counter section an object (or a cloche) stands on, else None."""
        p = self._pos(name)
        for k, s in enumerate(self.counters, 1):
            if -.05 < p[2] - s['top_z'] < .30 and point_in_polygon(p[:2], s['poly']):
                return k
        return None

    def _comp_of(self, name):
        return next((k for k, cid in enumerate(self.comps, 1) if self.env._inside_compartment(name, cid)), None)

    def _covering(self):
        """{cloche: the object under it} for the cloches not lifted yet."""
        return {c: v['target'] for c, v in self.env.covers.items() if c not in self.lifted}

    def _clear(self, a, b, own=(), skip_body=-1):
        """Does the straight line from a to b reach b (or a geom in `own`) before anything else?"""
        gid = np.array([-1], np.int32)
        x = mujoco.mj_ray(self.M, self.D, np.asarray(a, float), np.asarray(b, float) - a, GROUPS, 1, skip_body, gid)
        return x < 0 or x > .97 or int(gid[0]) in own

    def _shows(self, f, p, h=None, own=(), skip_body=-1):
        """Does camera frustum f show point p (inside the image, h tall >= MIN_PX if given, nothing in front of it)?"""
        uv = project(f, p, SIZE, SIZE)
        if uv is None or not (0 <= uv[0] < SIZE and 0 <= uv[1] < SIZE):
            return False
        depth = -(f['rot'].T @ (np.asarray(p) - f['pos']))[2]
        if h is not None and h * SIZE / 2 / math.tan(f['fovy'] / 2) / depth < MIN_PX:
            return False
        return self._clear(f['pos'], p, own, skip_body)

    def _in_view(self, skip, place=None):
        """{object or cloche: its counter} the three cameras show now, besides `skip`: at least MIN_PX tall in an image,
        and one of five points on it (centre, top, bottom, both sides) inside the image with the line from the camera
        reaching it first -- and lit (blackout_search; less strictly on counter `place`, the one looked at)."""
        env = self.env
        cover = self._covering()
        names = [n for n in self.pool if n not in skip and n not in cover.values() and self._comp_of(n) is None]
        names += [c for c in env.covers if c not in skip]
        frusta = camera_frusta(env.sim, CAMERAS)
        out = {}
        for n in names:
            k = self._counter_of(n)
            if k is None:                                  # not on any counter top: no place to name it by
                continue
            if n in env.covers:
                h, r = env.covers[n]['height'], env.covers[n]['radius']; c = self._pos(n) + [0., 0., h / 2]
            else:
                h, r = POOL[n[4:]]['height'], max(POOL[n[4:]]['size_xyz_m'][:2]) / 2; c = self._pos(n)
            own = set(env._object_geoms.get(n, []))
            for f in frusta:
                v = c - f['pos']; depth = -(f['rot'].T @ v)[2]
                if depth <= 0 or h * SIZE / 2 / math.tan(f['fovy'] / 2) / depth < MIN_PX:
                    continue
                side = np.cross(v, [0., 0., 1.]); side *= .8 * r / max(np.linalg.norm(side), 1e-9)
                pts = (c, c + [0., 0., .35 * h], c - [0., 0., .3 * h], c + side, c - side)
                if any(self._shows(f, p, None, own, self.cam_body[f['name']]) and self._lit(p, own, f['pos'], k == place)
                       for p in pts):
                    out[n] = k
                    break
        return out

    def _lit(self, p, own, cam, here):
        return True

    # ---- what the agent is told ------------------------------------------------------------------------------------
    def _comp_name(self, k):
        return ('drawer ' if self.env.compartments[self.comps[k - 1]]['kind'] == 'drawer' else 'cabinet ') + str(k)

    def _desc(self, name):
        return POOL[name[4:]]['description'] if name.startswith('obj_') else 'a lifted cloche' if name in self.lifted else 'a cloche'

    def _also(self, seen):
        """The rest of what the cameras show, by counter: 'the green can on counter 7 and a cloche on counter 3'."""
        by = {}
        for n, k in seen.items():
            by.setdefault(k, []).append(self._desc(n))
        return listing([f"{listing(v)} on counter {k}" for k, v in sorted(by.items())]) if by else ''

    def intro(self):
        seen = [f"{self._desc(n)} ({self._where(self._pos(n))})" for n in self.seen]
        return ' '.join([self._opening(), self._start_words() + (listing(seen) if seen else 'no objects and no cloches') + '.',
                         "You have not looked around the kitchen yet: LOOK AROUND THE KITCHEN maps its compartments with "
                         "handles and its counter tops and numbers them."])

    def _opening(self):
        return "You stand at the work counter, facing the tray and the Submit button, in a lit kitchen."

    def _start_words(self):
        return 'From where you stand the cameras show '

    def help(self):
        return "\n".join([
            "LOOK AROUND THE KITCHEN - map the compartments with handles and the counter tops, and number them",
            "OPEN k - open compartment k (a drawer or a cabinet), look inside, close it again",
            "LOOK AT COUNTER k - look at what stands on counter k",
            "LIFT THE CLOCHE ON COUNTER k - lift the cloche that stands on counter k and look at what was under it "
            "(the cloche stays lifted)",
            "SUBMIT <object> FROM <place>, ... - your answer: the helpers take each named object from the named place "
            "(a drawer, a cabinet or a counter) and stand it on the tray; e.g. SUBMIT BLUE MUG FROM DRAWER 3, RED CAN FROM "
            "COUNTER 5"])

    # ---- actions ---------------------------------------------------------------------------------------------------
    def act(self, line):
        t = ' '.join(re.sub(r'\bTHE\b', ' ', words(line)).split())
        if t in ('LOOK AROUND KITCHEN', 'LOOK AROUND'):
            return self._map()
        m = (re.fullmatch(r'OPEN (?:(?:DRAWER|CABINET|DOOR|COMPARTMENT) )?(\d+)', t)
             or re.fullmatch(r'(?:LOOK AT|LOOK|INSPECT) COUNTER (\d+)', t)
             or re.fullmatch(r'LIFT (?:CLOCHE ON |CLOCHE |COVER ON )?COUNTER (\d+)', t))
        if not m:
            raise Invalid("not an action of this task")
        k = self._known(int(m.group(1)), 'COUNTER' if 'COUNTER' in t else 'OPEN')
        return self._open(k) if t.startswith('OPEN') else self._lift(k) if t.startswith('LIFT') else self._look(k)

    def _known(self, k, kind):
        if not self.mapped:
            raise Invalid("no place has a number yet; look around the kitchen first")
        if not 1 <= k <= len(self.counters if kind == 'COUNTER' else self.comps):
            raise Invalid(f"there is no {'counter' if kind == 'COUNTER' else 'compartment'} {k}")
        return k

    def _map(self):
        if self.mapped:
            return {"status": "You have looked around the kitchen already.", "reading": "", "images": None}
        self.mapped, env = True, self.env
        comps = []
        for k, cid in enumerate(self.comps, 1):
            c = env.compartments[cid]; z0, z1 = c['pos'][2] - c['size'][2] / 2, c['pos'][2] + c['size'][2] / 2
            front = np.asarray(c['pos'][:2]) + np.asarray(c['front_normal']) * c['size'][1] / 2
            comps.append(f"{self._comp_name(k)} ({self._where(front)}, its front {z0:.2f}-{z1:.2f} m above the floor)")
        tray = next(k for k, s in enumerate(self.counters, 1) if point_in_polygon(env.tray['centre_xy'], s['poly']))
        cnts = [f"counter {k} ({self._where(s['xy'])}, {max(s['size']):.1f} m long, {s['top_z']:.2f} m high"
                + ("; the tray and the Submit button stand on it)" if k == tray else ")") for k, s in enumerate(self.counters, 1)]
        status = (f"{self._round()} {COUNT[len(comps)]} compartments have handles; from your left round to your right they "
                  f"are {listing(comps)}. The counter tops, cut into {len(cnts)} sections of at most {SECTION_M} m, from "
                  f"your left round to your right: {listing(cnts)}. What stands on a counter shows when you look at it.")
        if self.seen:
            status += f" What you saw from the start: {self._also(self.seen)}."
        return {"status": status, "reading": "", "images": None}

    def _round(self):
        return "You turn round the kitchen."

    def _open(self, k):
        env, cid = self.env, self.comps[k - 1]
        c, name = env.compartments[cid], self._comp_name(k)
        inside = [n for n in self.pool if env._inside_compartment(n, cid)]
        shift = self._set_open(c, inside, True)
        n = np.r_[c['front_normal'], 0.]; reg = np.asarray(c['region']['centre_world'], float)
        if c['kind'] == 'drawer':                          # look_inside: down into the pulled drawer from 0.45 m above its floor
            poses = [(reg + shift + n * b + [0., 0., h], reg + shift + [0., 0., .03], -n) for h, b in LOOK_DRAWER]
        else:                                              # ... and level into the cabinet from 0.25 m in front of it
            front = reg + n * (c['region']['size_xy'][1] / 2)
            poses = [(front + n * b + [0., 0., h], reg + [0., 0., .05], np.array([0., 0., 1.])) for h, b in LOOK_CABINET]
        also, images = self._shoot(c['standoff']['xy'], c['standoff']['yaw'], poses, self._lamp_comp(c, shift), inside)
        self._set_open(c, inside, False)
        how = 'pull it out 40 cm' if c['kind'] == 'drawer' else 'swing its door open'
        reading = (f"{name.capitalize()} holds {listing([self._desc(n) for n in inside])}." if inside
                   else f"{name.capitalize()} is empty.")
        return self._result(self._carry(name) + f"You {how}, look inside and close it again.", reading, also, images)

    def _look(self, k):
        """What the cameras show on counter k (the same test as for everything else in view: an object tucked into a
        nook no camera reaches is not named), then the rest."""
        s = self.counters[k - 1]
        xy, yaw, pose, nrm, half = self._view(k)
        c0, c1, _, c3 = (np.asarray(p, float) for p in s['poly'])
        grid = [c0 + u * (c1 - c0) + v * (c3 - c0) for u in (.15, .5, .85) for v in (.25, .75)]
        shown, images = self._shoot(xy, yaw, [pose], self._lamp_counter(np.r_[s['xy'], s['top_z']], nrm, half, grid), (), k)
        here = [self._desc(n) for n, kk in shown.items() if kk == k]
        reading = (f"On counter {k} stand{'s' if len(here) == 1 else ''} {listing(here)}." if here
                   else f"You see nothing on counter {k}.")
        also = {n: kk for n, kk in shown.items() if kk != k}
        return self._result(self._carry(f'counter {k}') + f"You look along counter {k}.", reading, also, images)

    def _lift(self, k):
        env = self.env
        covers = [c for c in self._covering() if self._counter_of(c) == k]
        if not covers:
            done = [c for c in self.lifted if self._counter_of(env.covers[c]['target']) == k]
            msg = f"The cloche on counter {k} is lifted already." if done else f"There is no cloche on counter {k}; nothing is lifted."
            return {"status": msg, "reading": "", "images": None}
        under = [env.covers[c]['target'] for c in covers]
        for c in covers:                                   # the task's own lift: the cloche onto its first set-down spot
            self._lift_cloche(c)
        spot = env.covers[covers[0]]['spot']; st = spot['standoff']
        nrm = -np.array([math.cos(st['yaw']), math.sin(st['yaw']), 0.])
        tgt = np.r_[np.mean([env.covers[c]['spot']['xy'] for c in covers], axis=0), spot['top_z']]
        poses = [(tgt + nrm * b + [0., 0., h], tgt + [0., 0., .05], -nrm) for h, b in LOOK_SPOT]
        also, images = self._shoot(st['xy'], st['yaw'], poses, self._lamp_counter(tgt, nrm[:2], float(spot['edge_depth'])), under)
        word = 'the cloche' if len(covers) == 1 else 'the cloches'
        status = self._carry(f'counter {k}') + f"You lift {word} on counter {k}, set it down beside its place and look."
        return self._result(status, f"Under {word} on counter {k} stood {listing([self._desc(n) for n in under])}.", also, images)

    def _lift_cloche(self, c):
        cv = self.env.covers[c]; o = cv['setdown_options'][0]
        self.env._set_object_pose(c, o['xy'], o['top_z'], float(cv['yaw'])); self.env.sim.forward()
        self.lifted.add(c)

    def _result(self, status, reading, also, images):
        if also:
            reading = reading[:-1] + f"; the cameras also show {self._also(also)}."
        return {"status": status, "reading": self._lit_words(reading), "images": images}

    # ---- the light (blackout_search overrides these) -----------------------------------------------------------------
    def _carry(self, place):
        return f"You go to {place}. "

    def _lit_words(self, reading):
        return reading

    def _lamp_comp(self, c, shift):
        return None

    def _lamp_counter(self, centre, nrm, half, grid=()):
        return None

    def _set_lamp(self, spot):
        pass

    # ---- poses -------------------------------------------------------------------------------------------------------
    def _view(self, k):
        """(stance xy, yaw, wrist-camera pose, side normal, half depth) for a look at counter k, chosen once by geometry.
        On each of the counter's four sides: the first wrist-camera pose of LOOK_COUNTER (above a point off the edge,
        looking at the counter's centre) with free air around it from which the counter top shows; the stances 0.5 to
        0.85 m off that side, straight or 0.3 m aside, with the base and its rear on free floor. The pair from which the
        cameras see most of the counter top (ray tests, the robot left out) and whose pose the arm reaches wins."""
        s = self.counters[k - 1]
        if 'view' in s:
            return s['view']
        env, M, D = self.env, self.M, self.D
        c0, c1, _, c3 = (np.asarray(p, float) for p in s['poly'])
        pts = [np.r_[c0 + u * (c1 - c0) + v * (c3 - c0), s['top_z'] + .06] for u in (.12, .5, .88) for v in (.2, .5, .8)]
        R = yaw_matrix(s['rec']['rot'])[:2, :2]; size = s['size']
        sides = [(R @ [0., -1.], size[1] / 2), (R @ [0., 1.], size[1] / 2), (R @ [1., 0.], size[0] / 2), (R @ [-1., 0.], size[0] / 2)]
        if s['along'] == 1:
            sides = sides[2:] + sides[:2]
        centre = np.r_[s['xy'], s['top_z']]; at = centre + [0., 0., .05]
        group = M.geom_group[self.robot_geoms].copy(); M.geom_group[self.robot_geoms] = 5
        cands = []
        for i, (nrm, half) in enumerate(sides):
            up, yaw = -np.r_[nrm, 0.], math.atan2(-nrm[1], -nrm[0])
            wrists = []
            for h, b in LOOK_COUNTER:
                cam = centre + np.r_[nrm * (half + b), h]
                f = dict(pos=cam, rot=facing(cam, at, up), fovy=math.radians(M.cam_fovy[self.cam]))
                if all(self._clear(cam, cam + .06 * e) for e in np.vstack([np.eye(3), -np.eye(3)])):
                    wrists.append(([self._shows(f, p) for p in pts], (cam, at, up)))
            wrists.sort(key=lambda w: -sum(w[0]))
            w, pose = wrists[0] if wrists else ([False] * len(pts), (centre + np.r_[nrm * (half + .15), .62], at, up))
            for dist in STAND_M:
                for lat in (0., .3, -.3):
                    xy = s['xy'] + nrm * (half + dist) + np.array([-nrm[1], nrm[0]]) * lat
                    if not self._stance_ok(xy, xy + nrm * REAR_M, s['rec']['name']):
                        continue
                    self._base(xy, yaw)
                    fr = camera_frusta(env.sim, CAMERAS[:2])
                    seen = sum(3 * a + self._shows(fr[0], p) + self._shows(fr[1], p) for a, p in zip(w, pts))
                    cands.append((-seen, i, dist, abs(lat), xy, yaw, pose, nrm, half))
        M.geom_group[self.robot_geoms] = group
        cands.sort(key=lambda c: c[:4])
        if not cands:                                       # no free stance: the front side, 0.5 m off
            nrm, half = sides[0]
            cands = [(0, 0, 0, 0, s['xy'] + nrm * (half + STAND_M[0]), math.atan2(-nrm[1], -nrm[0]),
                      (centre + np.r_[nrm * (half + .15), .62], at, -np.r_[nrm, 0.]), nrm, half)]
        s['view'] = cands[0][4:]
        for c in cands[:6]:                                 # the best one the arm reaches
            self._base(c[4], c[5])
            if self._look_pose(*c[6]):
                s['view'] = c[4:]
                break
        D.qpos[self.arm_q], D.qpos[self.base_q] = self.arm0, self.base0
        env.sim.forward()
        return s['view']

    def _stance_ok(self, xy, rear, counter):
        env = self.env
        return all(point_in_floor(p, env._floor, FLOOR_MARGIN_M) and not _blocked(p, env._blockers)
                   and _clearance(p, env._blockers, ignore=(counter,)) >= g for p, g in ((xy, BASE_CLEARANCE_M), (rear, .20)))

    def _set_open(self, c, inside, opened):
        """Open (a drawer by PULL_M, a door fully) or close a compartment by its joints (the compiled ranges); a drawer's
        contents ride along. Returns the drawer's displacement."""
        env, M, D = self.env, self.M, self.D
        shift = np.zeros(3)
        for j in c['joints']:
            jid = M.joint(j).id; lo, hi = M.jnt_range[jid]; q = hi if abs(hi) >= abs(lo) else lo
            val = (math.copysign(min(abs(q), PULL_M), q) if c['kind'] == 'drawer' else q) if opened else 0.
            if c['kind'] == 'drawer':
                shift = yaw_matrix(c['rot']) @ np.array([0., val - D.qpos[M.jnt_qposadr[jid]], 0.])
            D.qpos[M.jnt_qposadr[jid]] = val
        for n in inside:
            adr = M.jnt_qposadr[M.joint(env.objects[n].joints[0]).id]
            D.qpos[adr:adr + 3] += shift
        env.sim.forward()
        return shift

    def _shoot(self, xy, yaw, poses, lamp, skip, place=None):
        """Base to (xy, yaw), the lantern to `lamp` (blackout), the wrist camera to the first of `poses` the arm reaches;
        what else is in view and the three images (simulator version); then the base, the arm and the lantern back."""
        D = self.D
        self._base(xy, yaw)
        self._set_lamp(lamp)
        if not any(self._look_pose(*pose) for pose in poses):
            D.qpos[self.arm_q] = self.arm0                 # out of reach: the cameras look from the reset arm pose
            self.env.sim.forward()
        also, images = self._in_view(skip, place), self.views()
        D.qpos[self.arm_q], D.qpos[self.base_q] = self.arm0, self.base0
        self._set_lamp(None)
        self.env.sim.forward()
        return also, images

    def _base(self, xy, yaw):
        env, D = self.env, self.D
        for _ in range(3):
            have_xy, have_yaw = env.base_pose()
            err = np.asarray(xy, float) - have_xy; b = have_yaw - float(D.qpos[self.base_q[2]])
            D.qpos[self.base_q[0]] += math.cos(b) * err[0] + math.sin(b) * err[1]
            D.qpos[self.base_q[1]] += -math.sin(b) * err[0] + math.cos(b) * err[1]
            D.qpos[self.base_q[2]] += wrap(yaw - env.base_pose()[1])
            env.sim.forward()

    def _look_pose(self, cam_pos, at, up):
        """Arm joints by damped least squares so the wrist camera sits at cam_pos looking at `at`; True if reached."""
        M, D = self.M, self.D
        D.qpos[self.arm_q] = self.arm0
        want, body = facing(cam_pos, at, up), M.cam_bodyid[self.cam]
        for _ in range(400):
            mujoco.mj_kinematics(M, D); mujoco.mj_comPos(M, D); mujoco.mj_camlight(M, D)
            p = D.cam_xpos[self.cam].copy()
            ep = cam_pos - p; er = Rotation.from_matrix(want @ D.cam_xmat[self.cam].reshape(3, 3).T).as_rotvec()
            if np.linalg.norm(ep) < .003 and np.linalg.norm(er) < .01:
                break
            jp, jr = np.zeros((3, M.nv)), np.zeros((3, M.nv))
            mujoco.mj_jac(M, D, jp, jr, p, body)
            J = np.vstack([jp[:, self.arm_d], .5 * jr[:, self.arm_d]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[ep, .5 * er])
            dq *= min(1., .2 / max(np.abs(dq).max(), 1e-9))
            D.qpos[self.arm_q] = np.clip(D.qpos[self.arm_q] + dq, self.arm_lim[0] + 1e-3, self.arm_lim[1] - 1e-3)
        self.env.sim.forward()
        return bool(np.linalg.norm(ep) <= IK_TOL[0] and math.degrees(np.linalg.norm(er)) <= IK_TOL[1])

    # ---- privileged, for the self-test only ------------------------------------------------------------------------
    def _place(self, name):
        """The agent-facing place of a pool object now: ('DRAWER'|'CABINET', k) or ('COUNTER', k)."""
        k = self._comp_of(name)
        if k is not None:
            return self._comp_name(k).split()[0].upper(), k
        return 'COUNTER', self._counter_of(name)

    def oracle_lines(self):
        lines = ['LOOK AROUND THE KITCHEN']
        for key, p in sorted(self.env.places.items()):
            name = 'obj_' + p['object']; kind, k = self._place(name)
            new = [f'OPEN {kind} {k}'] if kind != 'COUNTER' else [f'LOOK AT COUNTER {k}'] + (
                [f'LIFT THE CLOCHE ON COUNTER {k}'] if name in self._covering().values() else [])
            lines += [x for x in new if x not in lines]
        return lines

    def oracle_answer(self, wrong=False):
        env = self.env
        items = [(t, self._place(t)) for t in env.task_spec['target_ids']]
        if wrong:                                          # the last target swapped for a hidden decoy, at its place
            decoy = sorted('obj_' + p['object'] for p in env.places.values() if p['role'] == 'hidden_distractor')[0]
            items[-1] = (decoy, self._place(decoy))
        return ', '.join(f"{POOL[n[4:]]['description'][4:].upper()} FROM {kind} {k}" for n, (kind, k) in items)

    # ---- the answer --------------------------------------------------------------------------------------------------
    def submit(self, answer):
        t = words(answer)
        t = ' '.join(re.sub(r'\bTHE\b', ' ', t[len('SUBMIT'):] if t.startswith('SUBMIT') else t).split())
        items = list(OBJ.finditer(t))
        if not items or len(items) > 3 or [w for w in OBJ.sub(' ', t).split() if w != 'AND']:
            raise Invalid("the answer is SUBMIT <colour> <object> FROM <place>, ... with one to three objects")
        env, picks = self.env, []
        for m in items:
            colour, kind, where, k = m.group(1).lower(), m.group(2), m.group(3), int(m.group(4))
            pid = f"{'bottle' if 'BOTTLE' in kind else kind.lower()}_{colour}"
            if pid not in POOL:
                raise Invalid(f"there is no {colour} {kind.lower()} in this task")
            picks.append(('obj_' + pid, where, self._known(k, where)))
        if len({p[0] for p in picks}) < len(picks):
            raise Invalid("the answer names an object twice")
        slots = TRAY_SLOTS[len(picks)]
        for i, (name, where, k) in enumerate(picks):
            if where == 'COUNTER':
                here = self._counter_of(name) == k
                for c, under in self._covering().items():   # the helper lifts the cloche it takes the object from
                    if here and under == name:
                        self._lift_cloche(c)
            else:
                here = env._inside_compartment(name, self.comps[k - 1])
            if here:                                       # else the robot finds no such object there: nothing moves
                xy = env.frame_to_world([env.tray['centre_local'][0] + slots[i][0], env.tray['centre_local'][1] + slots[i][1], 0.])[:2]
                env._set_object_pose(name, xy, env.tray['top_z'], float(env.work['yaw']))
        env.sim.forward()
        return self.score()
