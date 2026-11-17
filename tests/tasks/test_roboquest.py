"""The RoboQuest test's picture memory (plearn/tasks/roboquest/bench_eval.py, Episode.content), as the benchmark keeps it
(roboquest/harness/agent.py): the images of the most recent two observations, the start view counted as the first; every
earlier action keeps its text. Odd Parcel's reading aid (RQ_DIAL_ZOOM=1, README.md) in the registered test."""
import importlib.util
import os
import unittest
from unittest import mock


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "needs torch and transformers")
class MemoryTests(unittest.TestCase):
    def episode(self, actions):
        """An episode with no simulator: the start view and actions with three pictures each, or none (an action the helpers
        cannot do)."""
        from PIL import Image
        from plearn.tasks.roboquest.bench_eval import Episode
        e = Episode("odd_parcel", "x", 0, None, 0, None)
        pic = lambda k: [Image.new("RGB", (8, 8), (k, c, 0)) for c in range(3)]
        e.goal, e.intro, e.help, e.T, e.start = "goal", "intro", "help", 12, pic(255)
        e.log = [{"action": f"act {k}", "status": "ok" if has else "not possible", "reading": "", "images": pic(k) if has else []}
                 for k, has in enumerate(actions)]
        return e

    def kept(self, e):
        parts, images = e.content("sim")
        self.assertEqual(sum(p.get("type") == "image" for p in parts), len(images))
        return [im.getpixel((0, 0))[0] for im in images[::3]], "".join(p.get("text", "") for p in parts)

    def test_the_two_newest_observations_keep_their_pictures(self):
        self.assertEqual(self.kept(self.episode([]))[0], [255])                      # the start view alone
        self.assertEqual(self.kept(self.episode([True]))[0], [255, 0])               # the start view and the first action
        firsts, text = self.kept(self.episode([True, True, True]))
        self.assertEqual(firsts, [1, 2]); self.assertIn("Start view: (images no longer shown)", text)
        self.assertIn("[0] act 0: ok (images no longer shown)", text)
        firsts, text = self.kept(self.episode([True, True, False]))                  # an action with no pictures is no observation
        self.assertEqual(firsts, [0, 1]); self.assertIn("[2] act 2: not possible", text)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "needs torch and transformers")
class ReadingAidTests(unittest.TestCase):
    def turns(self, task, zoom):
        """The text of two turns of the registered test (bench_env.Episode.messages), with no simulator."""
        from PIL import Image
        from plearn.tasks.roboquest import bench_env
        from plearn.tasks.roboquest.bench_eval import Episode
        e = Episode(task, "x", 0, None, 0, None)
        e.goal, e.intro, e.help, e.T, e.start, e.log = "goal", "intro", "help", 12, [Image.new("RGB", (8, 8))] * 3, []
        ep = object.__new__(bench_env.Episode); ep.task, ep.version, ep.note, ep.e = task, "sim", None, e
        with mock.patch.dict(os.environ, {"RQ_DIAL_ZOOM": "1"}):
            if not zoom: del os.environ["RQ_DIAL_ZOOM"]
            return ["".join(p.get("text", "") for p in ep.messages(False)[1]) for _ in range(2)]

    def test_the_rule_follows_odd_parcels_own_text_once(self):
        from plearn.tasks.roboquest.bench_eval import READING_RULE
        for text in self.turns("odd_parcel", True):
            self.assertTrue(text.startswith("Task: goal\nintro\nActions:\nhelp" + READING_RULE)); self.assertEqual(text.count(READING_RULE), 1)
        for task, zoom in (("odd_parcel", False), ("stamps", True)):              # the switch off; another task
            self.assertTrue(all(READING_RULE not in text for text in self.turns(task, zoom)))


    def test_dial_zoom_sets_the_aid_for_odd_parcel_only(self):
        """dial_zoom on Odd Parcel in the sim version sets RQ_DIAL_ZOOM=1 in specs(), before any bridge; the text tasks and other
        tasks do not; an episode with dial_zoom refuses to start without the variable."""
        from plearn.api.registry import registry
        from plearn.api.task import Task
        from plearn.tasks.roboquest import bench_env
        reg = registry()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RQ_DIAL_ZOOM", None)
            Task.load("roboquest_text_odd_parcel", reg).specs(n=1); self.assertNotIn("RQ_DIAL_ZOOM", os.environ)
            bench_env.Episode.specs("test", 1, 0, "stamps", version="sim", dial_zoom=True); self.assertNotIn("RQ_DIAL_ZOOM", os.environ)
            with self.assertRaises(RuntimeError): bench_env.Episode(["test", "x"], "odd_parcel", version="sim", dial_zoom=True)
            self.assertEqual(len(bench_env.Episode.specs("test", 50, 0, "odd_parcel", version="sim", dial_zoom=True)), 50)
            self.assertEqual(os.environ["RQ_DIAL_ZOOM"], "1")



class OneLineTests(unittest.TestCase):
    """The action-only one-line turn (bench_env.py one_line; plearn/api/episode.py): no ANCHOR_HINT in the system text; after the
    thought one completion, the action line alone, the belief line empty. The default is the two-line turn of RLVB (EM)."""
    class Server:
        def __init__(self, replies): self.replies, self.prompts = list(replies), []
        def complete(self, prompts, **kw):
            self.prompts.append(prompts[0]); return [[{"text": self.replies.pop(0), "finish": "stop"}]]

    def test_the_turn(self):
        from plearn.api.episode import turn
        from plearn.api.task import GEN_DEFAULTS
        g = {**GEN_DEFAULTS, "max_action": 200}
        srv = self.Server(["I weigh first.", "Possible: 1, 2\nACTION: WEIGH 1 vs 2"])
        z, b, a, _ = turn(srv, "P", g, one_line=True)
        self.assertEqual((z, b, a, len(srv.prompts)), ("I weigh first.", "", "ACTION: WEIGH 1 vs 2", 2))
        srv = self.Server(["I weigh first.", "Possible: 1, 2", "ACTION: WEIGH 1 vs 2"])
        self.assertEqual(turn(srv, "P", g)[1:3], ("Possible: 1, 2", "ACTION: WEIGH 1 vs 2")); self.assertEqual(len(srv.prompts), 3)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "needs torch and transformers")
    def test_the_system_text(self):
        from PIL import Image
        from plearn.tasks.roboquest import bench_env
        from plearn.tasks.roboquest.bench_eval import Episode
        from plearn.tasks.roboquest.roboquest import SYSTEM, ANCHOR_HINT
        for one, system in ((True, SYSTEM), (False, SYSTEM + ANCHOR_HINT)):
            e = Episode("stamps", "x", 0, None, 0, None)
            e.goal, e.intro, e.help, e.T, e.start, e.log = "goal", "intro", "help", 10, [Image.new("RGB", (8, 8))] * 3, []
            ep = object.__new__(bench_env.Episode); ep.task, ep.version, ep.note, ep.e, ep.one_line = "stamps", "sim", "", e, one
            self.assertEqual(ep.messages(True)[0], system)


if __name__ == "__main__":
    unittest.main()
