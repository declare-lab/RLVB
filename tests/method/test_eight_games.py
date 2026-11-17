"""Every training file of the run scripts trains on train_suite, the 8 HiddenGames: the 3 RLVB (EM) run files and the 3 RLVB (GRPO)
run files."""
import glob
import os
import unittest

import yaml

from plearn.api.registry import registry

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PL_FILES = ["configs/train/suite_qwen3vl_8b_thinking.yaml", "configs/train/suite_qwen3vl_4b_thinking.yaml",
            "configs/train/suite_qwen35_4b_presence_8k.yaml"]
EIGHT = ["mastermind", "minesweeper_easy", "minigrid_keycorridor", "frozenlake_hidden_goal", "sokoban_hidden_target", "tabletop_lamps",
         "tabletop_cups", "tabletop_blocks"]


def files():
    grpo = sorted(glob.glob(os.path.join(ROOT, "configs", "train", "grpo_posterior_*.yaml")))
    return [os.path.join(ROOT, f) for f in PL_FILES] + grpo, grpo


class EightGamesTests(unittest.TestCase):
    def test_train_suite_is_the_8_games(self):
        self.assertEqual(registry().tasks(["train_suite"]), EIGHT)

    def test_every_training_file_names_train_suite(self):
        allf, grpo = files()
        self.assertEqual(len(grpo), 3)
        for f in allf:
            with self.subTest(f=os.path.relpath(f, ROOT)):
                c = yaml.safe_load(open(f))
                t = c["train_tasks"]; t = t if isinstance(t, list) else [t]
                self.assertEqual(t, ["train_suite"])
                if "eval_tasks" in c: self.assertEqual(c["eval_tasks"], ["train_suite"])


if __name__ == "__main__":
    unittest.main()
