"""The carried belief line (plearn/api/episode.py carried): a prompt of the agent's turn format that carries the agent's own
line from its last turn also says that the line is the agent's own earlier estimate, not evidence, and that the evidence
overrules it (a belief can be updated and is subject to correction)."""
import unittest

from plearn.api.episode import CARRY_MARK, CARRY_RULE, carried


def text_of(parts):
    return parts if isinstance(parts, str) else "".join(p.get("text", "") for p in parts)


class CarryTests(unittest.TestCase):
    def test_the_rule_follows_the_line(self):
        self.assertEqual(carried(""), ""); self.assertEqual(carried(None), "")
        self.assertEqual(carried("Possible: cups 1"), f"\n{CARRY_MARK}Possible: cups 1\n{CARRY_RULE}")
        self.assertIn("not evidence", CARRY_RULE); self.assertIn("the evidence decides", CARRY_RULE)

    def test_every_training_task_carries_the_rule_after_its_line(self):
        """Each training task's own prompt (text or pictures), as the trainer and the evaluator build it, with a carried line."""
        from plearn.api.registry import RegistryError, registry
        from plearn.api.task import Task
        reg = registry(); seen = 0
        for name in reg.tasks(["train_suite", "minesweeper_medium"]):  # every training task
            t = Task.load(name, reg)
            try: t.check()
            except RegistryError: continue                             # a simulator this machine does not have
            with self.subTest(task=name):
                ep = t.episode(t.specs("validation", 1)[0])
                try:
                    ep.e.note = "Possible: all"
                    _, parts, _ = ep.messages(True)
                    self.assertIn(f"{CARRY_MARK}Possible: all\n{CARRY_RULE}", text_of(parts)); seen += 1
                    _, parts, _ = ep.messages(False)                   # no carry: no line and no rule
                    self.assertNotIn(CARRY_RULE, text_of(parts))
                finally:
                    ep.close()
        self.assertGreater(seen, 0)


if __name__ == "__main__":
    unittest.main()
