"""Pro-Worlds (plearn/tasks/proworlds: expansion_env.py, format.py): the test sets of the normal and the long board, the
action-only one-line turn (--task_args proworlds.one_line=1), and, with Node.js 22 (NODE_BIN, else node on PATH), the episodes
of the frozen engine."""
import os
import shutil
import tempfile
import unittest
from unittest import mock

from plearn.api.registry import registry
from plearn.api.task import Task
from plearn.tasks.proworlds import expansion_env as X, format as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def node():
    return bool(shutil.which(os.environ.get("NODE_BIN") or "node"))


class TestSetTests(unittest.TestCase):
    def test_the_test_sets(self):
        """Game k of TEST-ORDER.txt plays the seeds 50,101,300 + 10,000 (k - 1) + j, j = 0 .. 19, on both boards."""
        reg = registry()
        self.assertEqual(X.ORDER, tuple(open(os.path.join(ROOT, "benchmarks", "proworlds", "games", "expansion-6.1", "TEST-ORDER.txt")).read().split()))
        for prefix, board in (("proworlds_expansion_", "normal"), ("proworlds_expansion_long_", "long")):
            for k, game in enumerate(X.ORDER):
                t = Task.load(prefix + game, reg)
                self.assertEqual((t.env_args["game"], t.env_args["board"], t.num_episodes), (game, board, 20))
                self.assertEqual(t.specs(), [["test", 50101300 + 10000 * k + j] for j in range(20)])
        self.assertEqual(reg.tasks(["proworlds_expansion"]), [f"proworlds_expansion_{g}" for g in X.ORDER])
        self.assertEqual(reg.tasks(["proworlds_expansion_long"]), [f"proworlds_expansion_long_{g}" for g in X.ORDER])


class OneLineTurnTests(unittest.TestCase):
    def test_the_one_line_system_text(self):
        for cond in ("window", "full-world"):
            two, one = F.system(cond, ["A", "B", "C"]), F.system(cond, ["A", "B", "C"], one_line=True)
            self.assertIn("Possible:", two); self.assertNotIn("Possible", one); self.assertIn("exactly one line", one)
            self.assertIn("ACTION: SUBMIT", one)

    def test_the_option_parses(self):
        """plearn eval --task_args proworlds.one_line=1 gives one_line to the Pro-Worlds tasks only."""
        from plearn import cli, evaluator, models
        seen = {}
        def evaluate(srv, proc, tasks, out, **kw): seen.update({t.name: t.env_args for t in tasks}); return {}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(models, "serve", lambda *a, **k: mock.Mock()), \
                mock.patch.object(models, "processor", lambda *a, **k: None), mock.patch.object(evaluator, "evaluate", evaluate), \
                mock.patch.object(evaluator, "table", lambda res: ""):
            rc = cli.main(["eval", "--model", d, "--tasks", "proworlds_expansion_face,proworlds_expansion_long_bee,mastermind",
                           "--task_args", "proworlds.one_line=1", "--output_path", os.path.join(d, "out")])
        self.assertEqual(rc, 0)
        for n in ("proworlds_expansion_face", "proworlds_expansion_long_bee"): self.assertEqual(seen[n]["one_line"], 1)
        self.assertNotIn("one_line", seen["mastermind"])


@unittest.skipUnless(node(), "needs Node.js 22 (NODE_BIN, else node on PATH)")
class EpisodeTests(unittest.TestCase):
    def test_the_episode(self):
        """face, seed 50101300: the start site is column 9, row 3; the turn limit is the budget + 8; a move and an answer end it."""
        t = Task.load("proworlds_expansion_face", registry()); ep = t.episode(t.specs()[0])
        try:
            self.assertEqual(ep.obs["site"], {"column": 9, "row": 3}); self.assertEqual(ep.limit, 35 + 8)
            system, parts, images = ep.messages(True)
            self.assertEqual(sum(p.get("type") == "image" for p in parts), len(images)); self.assertEqual(images[0].size, (256, 256))
            ep.step("Possible: all", "ACTION: MOVE LEFT", {}); self.assertEqual(ep.obs["site"], {"column": 8, "row": 3})
            ep.step("Possible: all", "ACTION: SUBMIT happy", {}); row = ep.row()
            self.assertTrue(ep.done); self.assertEqual((row["answer"], row["moves"], row["turns"], row["end"]), ("happy", 1, 2, "answer"))
        finally: ep.close()
        t = Task.load("proworlds_expansion_long_face", registry()); ep = t.episode(t.specs()[0])
        try: self.assertEqual(ep.limit, 65 + 8)
        finally: ep.close()

    def test_the_one_line_episode(self):
        """With one_line: the one-line system text, and the agent's last line is not carried."""
        t = Task.load("proworlds_expansion_face", registry()); t.env_args.update(one_line=1); ep = t.episode(t.specs()[0])
        try:
            ep.step("Possible: 1", "ACTION: MOVE LEFT", {}); system, parts, _ = ep.messages(True)
            self.assertNotIn("Possible", system); self.assertNotIn("Your belief line", "".join(p.get("text", "") for p in parts))
        finally:
            ep.close()


if __name__ == "__main__":
    unittest.main()
