"""Blackout Search at the decision level: the search room with the lights out. Two or three named objects hide in high
drawers (interior floor >= 0.40 m), under a cloche, or on a counter at least 2 m from the lantern; look-alike decoys
hide in other high drawers. Doors and low drawers are always empty, which the agent is not told.

The grammar, the places, the stances, the look poses and the answer are the search room's (search_room.py). The light:
the lantern is the only light and lights about a metre around itself. Every evidence action carries the lantern to the
place and sets it down beside it before it opens or looks -- on the counter top 0.3 m to the side of a compartment, 7 cm
behind its front, else inside the opened drawer or cabinet beside its contents; on a counter, near the front edge 0.2 m
to the side of the place, else further in -- and stands it back on its reset spot afterwards. So each image shows the
place lit by the lantern alone (no light is added, nothing is drawn on the image) while the rest of the kitchen stays
dark. The reading names what that light shows ("In the lantern light, drawer 3 holds the yellow tin."); anything is
named only when it is shown (the search room's test) and lit as the camera sees it (_lit: the lantern's falloff times
the lit share of the side in view). The start view names only what stands in the lantern's light at the start.
"""
import numpy as np
from roboquest.search.layout import clear_by, counter_patches, counter_top_occupants, point_in_polygon, resolve_keepouts

from .search_room import Task as Search

LANTERN = 'work_lamp'
GLASS_Z = .08                                       # the glass (and both lights) above the lantern's foot (blackout/lamp.py)
LIT_SEEN, LIT_HERE = .35, .20                       # least lantern light on the side a camera sees (see _lit)
SIDE_COMP = (.30, -.30, .45, -.45, .15, -.15, 0.)   # lantern spots beside a compartment, in this order
SIDE_COUNTER = (.20, -.20, .35, -.35, .50, -.50, 0.)


class Task(Search):
    # Budget: as the search room's (the same places and grammar; the lantern goes along at no cost). The hardest eval50
    # rows have 8 compartments, 11 counter sections and 1 cloche: 21 actions inspect every place once after LOOK AROUND
    # THE KITCHEN; the median kitchen has 18. 23 leaves 2 for slips.
    T = 23

    def __init__(self, env, render):
        self.patches = counter_patches(env._records)
        super().__init__(env, render)

    def _lit(self, p, own, cam, here):
        """Lit as the camera sees it. The lantern's two lights cast no shadows in the compiled model, so what counts is
        their falloff 1/(0.5 + 2.5 d^2) times the lit share of the side the camera sees, (1 + cos a)/2 for the angle a
        between the light and the camera at the point: at least LIT_SEEN, or LIT_HERE on the counter looked at.
        Calibrated on the eval50 renders: below 0.35 a third of the objects read darker than 20/255."""
        to_lamp, to_cam = self._pos(LANTERN) + [0., 0., GLASS_Z] - p, np.asarray(cam) - p
        d = np.linalg.norm(to_lamp)
        return (1 + to_lamp @ to_cam / d / np.linalg.norm(to_cam)) / 2 / (.5 + 2.5 * d * d) >= (LIT_HERE if here else LIT_SEEN)

    def _opening(self):
        return ("The kitchen is dark: the only light is a lantern, which lights about a metre around itself. You stand at "
                f"the work counter, facing the tray and the Submit button; the lantern stands {self._where(self._pos(LANTERN))}. "
                "You carry the lantern to every place you inspect and set it down beside it.")

    def _start_words(self):
        return 'In the lantern light the start cameras show '

    def _round(self):
        return "You carry the lantern round the kitchen."

    def _carry(self, place):
        return f"You carry the lantern to {place} and set it down beside it. "

    def _lit_words(self, reading):
        return 'In the lantern light, ' + reading[0].lower() + reading[1:]

    # ---- where the lantern stands --------------------------------------------------------------------------------------
    def _lamp_comp(self, c, shift):
        n = np.asarray(c['front_normal'], float); t = np.array([-n[1], n[0]])
        front = np.asarray(c['pos'][:2], float) + n * c['size'][1] / 2
        inner = np.asarray(c['region']['centre_world'][:2], float) + shift[:2]; w = c['region']['size_xy'][0] / 2 - .07
        return self._free([(front - n * .07 + t * a, None) for a in SIDE_COMP]
                          + [(inner + t * a, c['region']['floor_z']) for a in (w, -w)])

    def _lamp_counter(self, centre, nrm, half, grid=()):
        nrm = np.asarray(nrm, float); t = np.array([-nrm[1], nrm[0]]); edge = np.asarray(centre[:2], float) + nrm * (half - .08)
        return self._free([(edge + t * a, None) for a in SIDE_COUNTER] + [(edge - nrm * .15 + t * a, None) for a in SIDE_COUNTER]
                          + [(g, None) for g in grid])

    def _free(self, candidates):
        """The first candidate (xy, z; z None = on the counter top there) clear of the objects, the decor, the tray and
        Submit; None leaves the lantern where it stands."""
        env = self.env
        others = [self._pos(n) for n in self.pool + list(env.covers)]
        button = np.asarray(env.task_spec['submit_button']['position'], float)
        for xy, z in candidates:
            gap = .13
            if z is None:
                top = [p['top_z'] for p in self.patches if point_in_polygon(xy, p['polygon'])]
                if not top:
                    continue
                z, gap = top[0], .15
                if not clear_by(xy, resolve_keepouts(env._keepouts, z) + list(counter_top_occupants(env._records, z).values()), .06):
                    continue
            if any(np.linalg.norm(o[:2] - xy) < gap and abs(o[2] - z) < .35 for o in others):
                continue
            if np.linalg.norm(np.asarray(env.tray['centre_xy']) - xy) < env.tray['half'] * 1.45 + .06 or np.linalg.norm(button[:2] - xy) < .12:
                continue
            return xy, z
        return None

    def _set_lamp(self, spot):
        env = self.env
        if spot is None:
            env._set_object_pose(*env._lamp_choice)
        else:
            env._set_object_pose(LANTERN, spot[0], spot[1], 0.)
        env.sim.forward()
