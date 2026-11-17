"""plearn eval --task_args DATASET.KEY=VALUE (plearn/cli.py): the evaluator plays the Task objects that it is given, so the
arguments reach the episodes. The test guards against a loss of every --task_args value: when evaluate passes only the task
names to play, play loads the tasks again from the registry."""
import os
import tempfile
import unittest
from unittest import mock

from plearn import evaluator
from plearn.api.registry import registry
from plearn.api.task import Task


def run(tasks):
    """evaluate with play and score replaced: the tasks that play receives, and the results."""
    seen = []

    def play(srv, proc, tasks, **_):
        seen.extend(tasks)
        return {t.name: [] for t in tasks}, 0.0

    with tempfile.TemporaryDirectory() as out, mock.patch.object(evaluator, "play", play), \
            mock.patch.object(evaluator, "score", lambda rows, reg=None: {"results": {}, "groups": {}}):
        res = evaluator.evaluate(None, None, tasks, out)
        written = os.path.exists(os.path.join(out, "results.json"))
    return seen, res, written


class TaskArgsTests(unittest.TestCase):
    def test_a_task_object_keeps_its_arguments(self):
        t = Task.load("proworlds_expansion_face", registry())
        t.env_args.update(one_line=1)                                      # as plearn/cli.py applies --task_args
        seen, res, written = run([t])
        self.assertIs(seen[0], t); self.assertEqual(seen[0].env_args["one_line"], 1)
        self.assertEqual(res["config"]["tasks"], [t.name]); self.assertTrue(written)

    def test_names_are_loaded_from_the_registry(self):
        reg = registry()
        seen, res, _ = run("proworlds_expansion_face,proworlds_expansion_long_bee")
        self.assertEqual([t.name for t in seen], ["proworlds_expansion_face", "proworlds_expansion_long_bee"])
        self.assertTrue(all(isinstance(t, Task) for t in seen))
        self.assertNotIn("one_line", seen[0].env_args)                     # the registry's own arguments only
        self.assertEqual(res["config"]["tasks"], [t.name for t in seen])
        seen, _, _ = run(["proworlds_expansion_long_bee"])                 # a list of names
        self.assertEqual(seen[0].env_args, Task.load("proworlds_expansion_long_bee", reg).env_args)


if __name__ == "__main__":
    unittest.main()
