# The suites

Each file is one group. A group lists tasks or other groups; `python -m plearn show <group>` prints its tasks.

| Group | File | Members | Tasks | Games (test split) |
|---|---|---|---|---|
| `train_suite` | `train_suite.yaml` | 8 training tasks: the 7 HiddenGames test tasks and 1 training-only task (see `plearn/tasks/suites/train_suite.yaml`) | 8 | |
| `test_suite` | `test_suite.yaml` | `test_suite/proworlds`, `test_suite/hidden_games`, `test_suite/roboquest` | 32 | 1,556 |
| `test_suite_long` | `test_suite_long.yaml` | `test_suite/proworlds_long`, `test_suite/hidden_games`, `test_suite/roboquest` | 32 | 1,556 |
| `test_suite/proworlds` | `test_suite/proworlds.yaml` | `proworlds_expansion`: Pro-Worlds, 15 games x 20 plays, normal board (budget 35) | 15 | 300 |
| `test_suite/proworlds_long` | `test_suite/proworlds_long.yaml` | `proworlds_expansion_long`: the same games and plays, long board (budget 65) | 15 | 300 |
| `test_suite/hidden_games` | `test_suite/hidden_games.yaml` | the 7 HiddenGames tasks: `minesweeper_easy`, `minigrid_keycorridor`, `frozenlake_hidden_goal`, `sokoban_hidden_target`, `tabletop` (lamps, cups, blocks): 100 games each, `frozenlake_hidden_goal` and `sokoban_hidden_target` 128 each | 7 | 756 |
| `test_suite/roboquest` | `test_suite/roboquest.yaml` | `roboquest_text`: RoboQuest, 10 tasks x 50 eval50 instances, text version | 10 | 500 |

- Training uses `train_suite` on the train split: the 7 HiddenGames test tasks and 1 training-only task. Its validation is the
  same 8 tasks on the validation split (`python -m plearn eval --split validation`; in training, `eval_split`). A run mixes the
  teachers with an equal share of the training episodes each (`plearn/tasks/mixture.py`).
- The tests use the test split: `--tasks test_suite`, or `test_suite_long` for the Pro-Worlds long board. The test scripts
  select the long board with `EXPANSION=long`.
- Thinking is capped at 8,192 tokens in every test task.
- RLVB (EM) and RLVB (GRPO) use the two-line turn (the belief line, then the action line), the default. The action-only test of
  the base model ("Base") uses the one-line turn: `--task_args <dataset>.one_line=1` for each dataset (`scripts/eval_<model>.sh` with `TURN=one_line` sets them all).

```sh
python -m plearn eval --model qwen3-vl-8b-thinking --tasks test_suite --max_think 8192 --gpus 0 [--adapter DIR]
python -m plearn eval --model qwen3-vl-8b-thinking --tasks train_suite --split validation --gpus 0
```
