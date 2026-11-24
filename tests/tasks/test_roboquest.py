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



class FinalSubmit(unittest.TestCase):
    """final_submit: one more turn for the SUBMIT alone after T + 2 decisions; off by default."""
    class Bridge:
        def call(self, **r):
            if r["op"] == "open": return dict(goal="g", intro="i", help="h", T=2, images=None)
            if r["op"] == "act": return dict(status="done", reading="", images=None)
            if r["op"] == "submit": return dict(success=True, outcome="success")
            return dict(success=False)

    def play(self, final_submit, last):
        from plearn.tasks.roboquest.bench_eval import Episode
        e = Episode("odd_parcel", "x", 1, self.Bridge()); e.final_submit = final_submit; e.open(False)
        ends = [e.step("ACTION: WEIGH 1 VS 2") for _ in range(4)]          # T + 2 = 4 decisions without a SUBMIT
        if not ends[-1]: tail = e.content("text")[0]; ends.append(e.step(last))
        else: tail = None
        return e, ends, tail

    def test_off_by_default_the_episode_ends_without_an_answer(self):
        e, ends, tail = self.play(False, None)
        self.assertEqual(ends, [False, False, False, True]); self.assertEqual(e.result["outcome"], "no_answer")

    def test_one_more_turn_takes_the_submit(self):
        e, ends, tail = self.play(True, "ACTION: SUBMIT 1")
        self.assertEqual(ends, [False, False, False, False, True]); self.assertTrue(e.result["success"])
        self.assertIn("this is the last turn: reply with ACTION: SUBMIT <answer>.", tail); self.assertEqual(e.t, 2)

    def test_without_a_submit_in_that_turn_the_episode_ends_as_before(self):
        e, ends, tail = self.play(True, "ACTION: WEIGH 1 VS 2")
        self.assertEqual(ends[-1], True); self.assertEqual(e.result["outcome"], "no_answer"); self.assertEqual(e.decisions, 5)

    def test_an_episode_that_submits_in_time_is_unchanged(self):
        from plearn.tasks.roboquest.bench_eval import Episode
        for fs in (False, True):
            e = Episode("odd_parcel", "x", 1, self.Bridge()); e.final_submit = fs; e.open(False)
            self.assertFalse(e.step("ACTION: WEIGH 1 VS 2")); text = e.content("text")[0]
            self.assertTrue(text.endswith("Evidence actions used: 1 of 2. Decide now.")); self.assertTrue(e.step("ACTION: SUBMIT 1"))


class Only(unittest.TestCase):
    """only: the test plays the listed instances of a task alone."""
    def test_the_listed_instances_in_the_tasks_own_order(self):
        import tempfile
        from plearn.tasks.roboquest.bench_env import Episode, SUITE
        ids = open(SUITE / "eval50" / "puzzle_box.txt").read().split()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(f"# a comment\npuzzle_box {ids[7]}\nstamps {ids[1]}\npuzzle_box {ids[2]}\n")
        self.assertEqual(Episode.specs("test", 50, 0, "puzzle_box", only=f.name), [["test", ids[2]], ["test", ids[7]]])
        self.assertEqual(Episode.specs("test", 50, 0, "wobbly_stand", only=f.name), [])
        self.assertEqual(len(Episode.specs("test", 50, 0, "puzzle_box")), 50)
        os.remove(f.name)

    def test_the_episode_takes_the_same_env_args_as_specs(self):
        import inspect
        from plearn.tasks.roboquest.bench_env import Episode
        for k in ("only", "final_submit"): self.assertIn(k, inspect.signature(Episode.__init__).parameters)


class Replay(unittest.TestCase):
    """replay: the logged turns are carried out in the simulator without the model; the model plays what is left."""
    class Bridge:
        def __init__(self): self.acts = []
        def call(self, **r):
            if r["op"] == "open": return dict(goal="g", intro="i", help="h", T=2, images=None)
            if r["op"] == "act": self.acts.append(r["line"]); return dict(status="The balance is level.", reading="", images=None)
            if r["op"] == "submit": return dict(success=True, outcome="success")
            return dict(success=False)

    def episode(self, turns, final_submit=True):
        from plearn.tasks.roboquest import bench_env as BE
        from plearn.tasks.roboquest.bench_eval import Episode as Test
        ep = BE.Episode.__new__(BE.Episode); ep.task, ep.version, ep.note, ep.one_line = "odd_parcel", "text", None, False
        ep.e = Test("odd_parcel", "x", 1, self.Bridge()); ep.e.final_submit = final_submit; ep.e.open(False)
        ep.replayed, ep.replay_mismatch = 0, []; ep._replay(turns)
        return ep

    def turns(self, feedback="WEIGH 1 VS 2: The balance is level."):
        row = lambda k, a, f: {"turn": str(k), "belief": f"Possible: {k}", "action": a, "feedback": f}
        return [row(0, "ACTION: WEIGH 1 VS 2", feedback), row(1, "ACTION: WEIGH 1 VS 2", feedback),
                row(2, "ACTION: WEIGH 3 VS 4", "WEIGH 3 VS 4: No evidence actions are left; only SUBMIT is possible."),
                row(3, "ACTION: WEIGH 3 VS 4", "WEIGH 3 VS 4: No evidence actions are left; only SUBMIT is possible.")]

    def test_the_log_is_carried_out_and_one_turn_is_left_for_the_submit(self):
        ep = self.episode(self.turns())
        self.assertEqual(ep.e.bridge.acts, ["WEIGH 1 VS 2", "WEIGH 1 VS 2"])        # the two actions after the limit never reached the simulator
        self.assertEqual((ep.replayed, ep.replay_mismatch, ep.done, ep.e.last_call), (4, [], False, True))
        self.assertEqual(ep.note, "Possible: 3")                                     # the last logged belief line is carried
        self.assertIn("this is the last turn: reply with ACTION: SUBMIT <answer>.", ep.messages(True)[1])
        ep.step("Possible: 1", "ACTION: SUBMIT 1", {}); self.assertTrue(ep.done); self.assertTrue(ep.row()["correct"])
        self.assertEqual(ep.row()["replayed_turns"], 4)

    def test_a_different_reply_of_the_simulator_is_reported(self):
        ep = self.episode(self.turns("WEIGH 1 VS 2: The left pan is down 3.0 degrees."))
        self.assertEqual([m["turn"] for m in ep.replay_mismatch], [0, 1])

    def test_without_final_submit_the_replayed_episode_ends_as_the_log_did(self):
        ep = self.episode(self.turns(), final_submit=False)
        self.assertTrue(ep.done); self.assertEqual(ep.e.result["outcome"], "no_answer")


class SubmitPrefill(unittest.TestCase):
    """final_submit=prefill: the SUBMIT turn without a new thought; the reply starts as 'ACTION: SUBMIT'."""
    class Srv:
        def __init__(self, text): self.text, self.prompts, self.kw = text, [], []
        def complete(self, prompts, **kw):
            self.prompts += prompts; self.kw.append(kw); return [[{"text": self.text, "finish": "stop"}]]

    def test_the_model_writes_only_the_answer_after_its_own_belief_line(self):
        from plearn.api.episode import submit_turn, CLOSE, TERM
        srv = self.Srv(" 6 4 5 3\nmore text")
        z, b, a, rec = submit_turn(srv, "PROMPT", {"max_action": 48, "line_temperature": 0.0}, "Possible: the order is 6, 4, 5, 3\nsecond line")
        self.assertEqual(srv.prompts, ["PROMPT" + CLOSE + "Possible: the order is 6, 4, 5, 3" + TERM + "ACTION: SUBMIT"])
        self.assertEqual((z, b, a), ("", "Possible: the order is 6, 4, 5, 3", "ACTION: SUBMIT 6 4 5 3"))
        self.assertEqual(len(srv.prompts), 1); self.assertEqual(rec["thought_chars"], 0); self.assertTrue(rec["submit_prefill"])

    def test_without_a_belief_line_and_with_an_empty_reply(self):
        from plearn.api.episode import submit_turn, CLOSE
        srv = self.Srv("")
        _, b, a, _ = submit_turn(srv, "PROMPT", {"max_action": 48, "line_temperature": 0.0}, None)
        self.assertEqual(srv.prompts, ["PROMPT" + CLOSE + "ACTION: SUBMIT"]); self.assertEqual((b, a), ("", "ACTION: SUBMIT"))

    def test_the_episode_asks_for_it_only_in_the_extra_turn(self):
        from plearn.tasks.roboquest import bench_env as BE
        from plearn.tasks.roboquest.bench_eval import Episode as Test
        ep = BE.Episode.__new__(BE.Episode); ep.task, ep.version, ep.note, ep.one_line = "odd_parcel", "text", None, False
        ep.e = Test("odd_parcel", "x", 1, FinalSubmit.Bridge()); ep.e.final_submit = True; ep.prefill = True; ep.e.open(False)
        seen = []
        for _ in range(4): seen.append(ep.submit_only); ep.step("Possible: 1", "ACTION: WEIGH 1 VS 2", {})
        self.assertEqual(seen, [False] * 4); self.assertTrue(ep.submit_only)
        ep.step("Possible: 1", "ACTION: SUBMIT 1", {}); self.assertFalse(ep.submit_only); self.assertTrue(ep.row()["correct"])
        ep.prefill = False; self.assertFalse(ep.submit_only)


if __name__ == "__main__":
    unittest.main()
