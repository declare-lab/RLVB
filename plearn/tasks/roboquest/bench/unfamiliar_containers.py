"""Unfamiliar Containers at the decision level: 4, 5 or 6 identical wooden boxes stand on the counters (in a look instance
one of them stands in another part of the kitchen), and 2 to 4 named items lie inside them, at most one per box. Every box
carries two identical knobbed panels on two of its faces. One panel opens the box in one of eleven ways (seven for a side
face, four for the top); the other is screwed to a solid wall. A decoy box carries two fixed panels.

Evidence actions: LOOK AROUND THE KITCHEN (walk round: finds the boxes out of view and shows the far sides of the boxes), and
<WAY> BOX n <FACE>, which works one knobbed panel in one named way. The ways are named as hand motions, never as mechanisms.
Each is done exactly as named, so a panel opens only when it is worked in the one way it allows; the box's own privileged
opener (_teleport_open) then opens it. A press that releases a touch latch, or a knob that turns, is felt and reported even
when the motion after it fails, and a released latch stays released. When a box opens, the hand reaches inside; an item found
there is held up to the wrist camera (the images) and put back. The status is what the hand feels (the press gave or not, the
knob turned or not, the panel moved or not, the hand found something or nothing); the reading names the item held up.
A box is numbered from left to right as the left camera sees it; a box out of view gets its number when it is found.
The answer names the opened boxes whose contents go to the bowl: the helper opens each again the way it was opened and drops
its item into the bowl, as the task's own teleport_solution does.
"""
import math
import numpy as np
from roboquest.containers import panels as P, placement as PL
from roboquest.containers.contract import SEALED
from roboquest.search.visibility import camera_frusta, camera_coords, point_in_frustum
from roboquest.tasks import unfamiliar_containers as UC
from .base import Task as Base, Invalid, words, numbers, CAMERAS

SIDE_WAYS = {"PULL": (None, "pull"), "SWING": (None, "swing"), "PRESS THEN PULL": ("press", "pull"),
             "PRESS THEN SWING": ("press", "swing"), "TURN THEN PULL": ("turn", "pull"),
             "PUSH SIDEWAYS": (None, "sideways"), "PUSH UP": (None, "up")}
TOP_WAYS = {"LIFT": (None, "lift"), "SLIDE": (None, "slide"), "TURN THEN LIFT": ("turn", "lift"), "TIP UP": (None, "tip")}
WAYS = sorted({**SIDE_WAYS, **TOP_WAYS}, key=len, reverse=True)
# What each mechanism needs (contract.RESPONSE): the first step it must get, and the motion that then opens it.
NEEDS = {"drawer": (None, "pull"), "press_drawer": ("press", "pull"), "turn_knob": ("turn", "pull"),
         "door": (None, "swing"), "press_door": ("press", "swing"), "slide_side": (None, "sideways"),
         "slide_up": (None, "up"), "lift_lid": (None, "lift"), "twist_lid": ("turn", "lift"),
         "slide_lid": (None, "slide"), "flip_lid": (None, "tip")}
WAY_OF = {m: next(w for w, v in {**SIDE_WAYS, **TOP_WAYS}.items() if v == n) for m, n in NEEDS.items()}
DO = {"pull": "pull the knob straight out", "swing": "pull the knob out along an arc", "sideways": "push the panel sideways",
      "up": "push the panel up", "lift": "lift the panel by its knob", "slide": "push the panel along the top",
      "tip": "lift the near edge of the panel"}
MOVES = {"pull": "it slides straight out with a tray behind it", "swing": "it swings open on a hinge at one upright edge",
         "sideways": "it slides sideways and uncovers an opening", "up": "it slides up and uncovers an opening",
         "lift": "it comes off, and you set it down beside the box", "slide": "it slides along the top and uncovers the inside",
         "tip": "it tips up on a hinge at its far edge"}
SPRINGS = {"press_drawer": "it gives, clicks and springs straight out a few centimetres",
           "press_door": "it gives, clicks and springs ajar on a hinge at one upright edge"}
SIDES = ("top", "front", "right", "back", "left")
SAY = {"front": "front", "back": "back", "left": "left side", "right": "right side", "top": "top"}
FACE_SEEN = .2          # a face counts as seen by a camera that looks at it at least this far from edge-on (cosine)
HOLD = .25              # m in front of the wrist camera where the hand holds a found item
ACT_TICKS = 20          # 1 s for the scene to settle after an action
PART_SEEN = 2           # a box with this many corners in a camera's view (but not its centre) shows only in part
CAM_SAY = dict(zip(CAMERAS, ("left", "right", "wrist")))
OVERLAP = .5            # two boxes whose image extents overlap this much (of the narrower) are said to stand one before the other


def listed(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


class Task(Base):
    # T: a good agent looks around once, then works box by box, the top panel before a side one. On a top panel: TURN THEN
    # LIFT, SLIDE, TIP UP. On a side panel: PRESS THEN PULL (after a click without opening: SWING), TURN THEN PULL, PRESS THEN
    # SWING, PUSH SIDEWAYS, PUSH UP (a press or a turn that the panel ignores does no harm, so these also cover PULL, LIFT and
    # SWING). It needs at most 28 actions on the hardest eval50 instance (9b10619936505523: 6 boxes with a decoy, in the
    # worst box order; 26 on average over box orders). 4 more for slack.
    T = 32

    def __init__(self, env, render):
        super().__init__(env, render)
        self.boxes = env.geometry["boxes"]
        frusta = camera_frusta(env.sim, CAMERAS)

        def seen(p, n=None):
            for f in frusta:
                v = f["pos"] - p
                if point_in_frustum(f, p) and (n is None or n @ v / np.linalg.norm(v) > FACE_SEEN):
                    return True
            return False

        self.seen, order, span = {}, {}, {}               # box -> sides seen now; sort key; (image extent, depth)
        self.edges = []                                   # (camera, edge) of each box only partly in view at the start
        for name, b in self.boxes.items():
            base = np.asarray(b["base"], float); centre = base + [0, 0, P.H / 2]
            co, si = math.cos(b["yaw"]), math.sin(b["yaw"])
            corners = [base + [co * x - si * y, si * x + co * y, z] for x in (-.13, .13) for y in (-.13, .13) for z in (0, P.H)]
            faces = {"top": (base + [0, 0, P.H], np.array([0., 0., 1.]))}
            for face, fy in P.FACE_YAW.items():
                n = np.array([math.sin(b["yaw"] + fy), -math.cos(b["yaw"] + fy), 0.])
                faces[PL.world_side(b["robot_local"]["yaw"], face)] = (centre + n * P.W / 2, n)
            pc = camera_coords(frusta[0], centre)
            if not b["annexed"] and seen(centre):
                self.seen[name] = {s for s, (p, n) in faces.items() if seen(p, n)}
                order[name] = (0, pc[0] / max(-pc[2], 1e-3))          # left to right in the left camera's image
                u = [q[0] / max(-q[2], 1e-3) for q in (camera_coords(frusta[0], x) for x in corners)]
                span[name] = (min(u), max(u), -pc[2])
            else:
                inside = [sum(point_in_frustum(f, x) for x in corners) for f in frusta]
                if not b["annexed"] and max(inside) >= PART_SEEN:     # the agent sees part of it, but cannot work it yet
                    f = frusta[int(np.argmax(inside))]
                    self.edges.append((CAM_SAY[f["name"]], "left" if camera_coords(f, centre)[0] < 0 else "right"))
                order[name] = (1 + b["annexed"], self.camera_x(centre))  # found later: work counter, then elsewhere
        # Every box gets its number now (so the oracle can name it); a box out of view is known only once it is found.
        self.num = {k + 1: name for k, name in enumerate(sorted(self.boxes, key=order.get))}
        self.found = {k for k, name in self.num.items() if name in self.seen}
        self.opened, self.looked = set(), False
        self.behind = []                                  # (near, far): the near box hides much of the far one in the image
        for i, a in self.num.items():
            for j, c in self.num.items():
                if a in span and c in span and span[a][2] < span[c][2]:
                    lap = min(span[a][1], span[c][1]) - max(span[a][0], span[c][0])
                    if lap > OVERLAP * min(span[a][1] - span[a][0], span[c][1] - span[c][0]):
                        self.behind.append((i, j))

    def _panels(self, name):
        b = self.boxes[name]
        return [s for s in SIDES if s in (b["side"], b["dummy_side"])]

    def _faces(self, sides):
        return listed([SAY[s] for s in SIDES if s in sides])

    def _boxes(self, ks):
        return ("box " if len(ks) == 1 else "boxes ") + listed([str(k) for k in ks])

    # ---- what the agent is told ------------------------------------------------------------------------------------
    def intro(self):
        n = len(self.found)
        lines = ([f"On the work counter you see {n} boxes, numbered 1 to {n} from left to right as the left camera sees them."]
                 if n > 1 else ["On the work counter you see one box, box 1."] if n else ["You see no box from where you stand."])
        lines.append("Every box is the same wooden case, about 26 cm wide and 22 cm tall.")
        for k in sorted(self.found):
            name = self.num[k]; faces = self.seen[name]; knobs = [s for s in self._panels(name) if s in faces]
            if not faces:                                  # its centre is in view, but no face is turned to a camera
                lines.append(f"Box {k}: you see only part of it."); continue
            what = (f"; a knobbed panel is on its {SAY[knobs[0]]}" if len(knobs) == 1 else
                    f"; knobbed panels are on its {self._faces(knobs)}" if knobs else
                    f", with no knobbed panel on {'it' if len(faces) == 1 else 'them'}")
            lines.append(f"Box {k}: you see its {self._faces(faces)}{what}.")
        lines += [f"Box {i} stands in front of box {j}." for i, j in sorted(self.behind)]
        lines += [f"Part of a box shows at the {e} edge of the {cam} camera's view." for cam, e in sorted(self.edges)]
        lines.append("The far sides of the boxes and parts of the kitchen are out of view. "
                     "The bowl stands at the front of the work counter.")
        return " ".join(lines)

    def help(self):
        return "\n".join([
            "LOOK AROUND THE KITCHEN - walk round the counters and the kitchen: find the boxes out of view and see every "
            "side of every box",
            "<WAY> BOX n <FACE> - work the knobbed panel on one face of box n in one way; if the box opens, reach inside. "
            "FACE is FRONT (the side toward you), BACK, LEFT, RIGHT or TOP. Example: PULL BOX 2 FRONT",
            "  ways for a panel on a side face:",
            "  PULL - pull the knob straight out",
            "  SWING - pull the knob out along an arc, about the panel's left or right edge",
            "  PRESS THEN PULL - press the panel in firmly, then pull the knob straight out",
            "  PRESS THEN SWING - press the panel in firmly, then pull the knob out along an arc",
            "  TURN THEN PULL - turn the knob a quarter turn, then pull it straight out",
            "  PUSH SIDEWAYS - push the panel sideways along the face",
            "  PUSH UP - push the panel up the face",
            "  ways for a panel on the top:",
            "  LIFT - lift the panel straight up by its knob",
            "  SLIDE - push the panel along the top, away from you",
            "  TURN THEN LIFT - turn the knob a quarter turn, then lift the panel straight up",
            "  TIP UP - lift the near edge of the panel",
            "SUBMIT a b - take what is inside the named boxes (each one you opened) and put it in the bowl (your answer)"])

    # ---- actions ---------------------------------------------------------------------------------------------------
    def act(self, line):
        t = words(line)
        t = t[len("TRY "):] if t.startswith("TRY ") else t
        if t in ("LOOK AROUND THE KITCHEN", "LOOK AROUND"):
            return self._look()
        for way in WAYS:
            if t.startswith(way + " "):
                rest = t[len(way):].split()
                ks, sides = numbers(" ".join(rest)), [s for s in SIDES if s.upper() in rest]
                if "BOX" not in rest or len(ks) != 1 or len(sides) != 1:
                    raise Invalid("name the way, the box and the face, e.g. PULL BOX 2 FRONT")
                return self._try(way, ks[0], sides[0])
        raise Invalid("not an action of this task")

    def _look(self):
        if self.looked:
            return {"status": "You have looked around the kitchen already.", "reading": "", "images": None}
        self.looked = True
        new = sorted(set(self.num) - self.found)
        annex = self.env.geometry["annex"] or {}                      # an annex may be a far part of the work counter
        away = [k for k in new if self.boxes[self.num[k]]["annexed"] and annex.get("fixture") != self.env.work["fixture"]]
        work = [k for k in new if k not in away]
        self.found = set(self.num)
        for name in self.boxes:
            self.seen[name] = set(SIDES)
        where = [f"{self._boxes(ks)} {w}" for ks, w in ((work, "on the work counter"), (away, "in another part of the kitchen"))
                 if ks]
        status = (f"You walk round the kitchen. You find {' and '.join(where)}. " if where else
                  "You walk round the kitchen and find no other box. ")
        status += f"There are {len(self.num)} boxes in all. " + " ".join(
            f"Box {k}: knobbed panels on its {self._faces(self._panels(self.num[k]))}." for k in sorted(self.num))
        return {"status": status, "reading": "", "images": None}

    def _try(self, way, k, side):
        if k not in self.found:
            raise Invalid(f"you know of no box {k}")
        name = self.num[k]; b = self.boxes[name]
        if side not in self._panels(name) or side not in self.seen[name]:
            raise Invalid(f"you know of no knobbed panel on the {SAY[side]} of box {k}")
        ways = TOP_WAYS if side == "top" else SIDE_WAYS
        if way not in ways:
            raise Invalid(f"{way} is a way for a panel on {'a side face' if side == 'top' else 'the top'}")
        if name in self.opened:
            return {"status": f"You have opened box {k} already.", "reading": "", "images": None}
        env = self.env; m = b["mechanism"]; prep, motion = ways[way]
        need, opens = NEEDS[m] if m != SEALED and side == b["side"] else (None, None)   # a fixed panel needs nothing, opens never
        said = []
        if prep == "press":
            if need == "press" and not env.latch_released[name]:      # the touch latch lets go; its spring pushes the panel out
                env.sim.data.eq_active[env._latches[name]] = 0; env.latch_released[name] = True
                self.tick(ACT_TICKS); said.append(f"You press the panel in: {SPRINGS[m]}.")
            else:
                said.append("You press the panel in: it " + ("is sprung already" if need == "press" else "does not give") + ".")
        if prep == "turn":
            said.append("You turn the knob: it " + ("turns a quarter turn" if need == "turn" else "does not turn") + ".")
        sprung = need == "press" and env.latch_released[name]
        if motion != opens or not (need is None or sprung or need == prep):
            said.append(f"You {DO[motion]}: the panel does not move{' further' if sprung else ''}.")
            return {"status": " ".join(said), "reading": "", "images": None}
        moving = env.sim.model.body_name2id(b["moving_body"]); before = env.sim.data.body_xpos[moving].copy()
        env._teleport_open(name)
        items = [it for it in env.geometry["items"] if it["holder"] == name]
        if m in P.TRAY_MECHANISMS:                                     # the item rides out on the tray
            shift = env.sim.data.body_xpos[moving] - before
            for it in items:
                q = env.sim.data.get_joint_qpos(it["joint_name"]).copy(); q[:3] += shift
                env.sim.data.set_joint_qpos(it["joint_name"], q)
            env.sim.forward()
        self.opened.add(name)
        said.append(f"You {DO[motion]}: {MOVES[motion]}. Box {k} is open.")
        if not items:
            self.tick(ACT_TICKS); said.append("You reach inside: the box is empty.")
            return {"status": " ".join(said), "reading": "", "images": None}
        said.append("You reach inside, take out what lies there, hold it up to your wrist camera and put it back.")
        return {"status": " ".join(said), "reading": f"You hold up {UC.article(items[0]['name'])}.",
                "images": self._hold_up(items[0])}

    def _hold_up(self, item):
        """The item held in front of the wrist camera for the images, then put back where it lay."""
        sim = self.env.sim; j = item["joint_name"]; rest = sim.data.get_joint_qpos(j).copy()
        cam = sim.model.camera_name2id(CAMERAS[2])
        at = sim.data.cam_xpos[cam] - HOLD * sim.data.cam_xmat[cam].reshape(3, 3)[:, 2]   # cameras look along -z
        sim.data.set_joint_qpos(j, np.r_[at, 1., 0., 0., 0.]); sim.forward()
        images = self.views()
        sim.data.set_joint_qpos(j, rest)
        dof = sim.model.jnt_dofadr[sim.model.joint_name2id(j)]; sim.data.qvel[dof:dof + 6] = 0.
        sim.forward(); self.tick(ACT_TICKS)
        return images

    # ---- privileged, for the self-test only ------------------------------------------------------------------------
    def oracle_lines(self):
        lines, shown = ["LOOK AROUND THE KITCHEN"], False
        for k, name in sorted(self.num.items()):
            b = self.boxes[name]; m = b["mechanism"]; real, fake = b["side"].upper(), b["dummy_side"].upper()
            if m == SEALED:
                lines.append(f"{'TURN THEN LIFT' if real == 'TOP' else 'PRESS THEN PULL'} BOX {k} {real}"); continue
            if not shown:                                              # a fixed panel never moves
                lines.append(f"{'TURN THEN LIFT' if fake == 'TOP' else 'PRESS THEN PULL'} BOX {k} {fake}"); shown = True
            if m == "press_door":                                      # the press springs it ajar; the straight pull fails
                lines += [f"PRESS THEN PULL BOX {k} {real}", f"SWING BOX {k} {real}"]
            else:
                lines.append(f"{WAY_OF[m]} BOX {k} {real}")
        return lines

    def oracle_answer(self, wrong=False):
        holders = set(self.env.geometry["holders_private"])
        ks = [k for k, name in self.num.items() if name in holders]
        if wrong:                                                      # one holder left out (an opened empty box instead)
            empty = [k for k, name in self.num.items() if name not in holders and self.boxes[name]["mechanism"] != SEALED]
            ks = ks[:-1] + empty[:1]
        return " ".join(str(k) for k in sorted(ks))

    def submit(self, answer):
        ks = numbers(words(answer))
        if not ks or len(set(ks)) < len(ks) or any(k not in self.found for k in ks):
            raise Invalid("the answer must name known boxes, each once")
        shut = [k for k in ks if self.num[k] not in self.opened]
        if shut:
            raise Invalid(f"{self._boxes(shut)} not opened yet: the helper can take out only what is in a box you opened")
        env = self.env; zero = np.zeros(env.action_dim); names = [self.num[k] for k in ks]
        everything = env.geometry["items"]; items = [it for it in everything if it["holder"] in names]

        def drop(it):
            env._drop_into_bowl(it, *env._bowl_spot(it["index"], len(everything), float(it["half"][2])), zero)

        for name in names:
            env._teleport_open(name)                                   # again: a closer may have shut it
        for it in items:
            drop(it)
        self.tick(UC.TELEPORT_SETTLE_TICKS)
        for _ in range(UC.TELEPORT_RESEATS):                           # re-seat what bounced out, as teleport_solution does
            facts = env._item_facts(env.item_states())
            loose = [it for it in items if not facts[it["index"]]["in_bowl"]]
            if not loose:
                break
            for it in loose:
                drop(it)
            self.tick(UC.TELEPORT_SETTLE_TICKS)
        return self.score()
