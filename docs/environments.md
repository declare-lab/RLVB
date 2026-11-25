# Environments

`test_suite` has three parts: RoboQuest, Pro-Worlds and HiddenGames. `train_suite` has 8 training tasks: the 7 HiddenGames
test tasks and 1 training-only task (see `plearn/tasks/suites/train_suite.yaml`), on their train split. The methods train on
these tasks only, so RoboQuest and Pro-Worlds test transfer to tasks that no run trained on.

| Part | Suite id | Tasks | Test games | Training |
|---|---|---|---|---|
| RoboQuest (text version) | `test_suite/roboquest` | 10 | 500 | none |
| Pro-Worlds, normal board | `test_suite/proworlds` | 15 | 300 | none |
| HiddenGames | `test_suite/hidden_games`; `train_suite` | 7 | 756 | train split |

Every task gives the agent a hidden target. The agent takes evidence actions (look, move, guess, press, lift, push), then
answers or reaches the goal. In the default two-line turn the agent writes a belief line (`Possible: ...`) and an action line
after its thought. `--task_args <dataset>.one_line=1` gives the action-only turn.

## RoboQuest

Ten mobile-manipulation tasks in RoboCasa kitchens (`benchmarks/roboquest`, MIT). Every task hides what the agent needs
(where the targets are, a label under a vessel, which parcel is odd, ...). The suites play the 50 eval50 instances of each task
at the decision level: the agent acts through helper actions with exact control, and the text version replaces each action's
pictures with one sentence. Score: `success` and the benchmark's progress `P`. The simulator runs in its own runtime
(`ROBOQUEST_RUNTIME`, [install.md](install.md)). Details: [plearn/tasks/roboquest/README.md](../plearn/tasks/roboquest/README.md).

Tasks: Odd Parcel, Painted Cubes, Marked Mugs, Unfamiliar Containers, Search Room, Blackout Search, Locked Storage, Puzzle Box,
Stamps, Wobbly Stand.

## Pro-Worlds (normal and long board)

Find the member of a hidden world from pictures of a moving window. The 15 visual games come from the frozen engine v6.1
(`benchmarks/proworlds/games/expansion-6.1`; `SHA256SUMS` lists the sha256 of every file, and no file is changed). The window
shows one site of the board; a move goes to the next site; the answer is one member name. Each game has 20 test plays (seeds
50,101,300 + 10,000 (k - 1) + j for game k, j = 0 .. 19).

| Board | Budget | Tasks | Group |
|---|---|---|---|
| normal | 150 moves (35 with `PL_TEST_MOVES=0`) | `proworlds_expansion_<game>` | `proworlds_expansion` = `test_suite/proworlds` |
| long | 150 moves (65 with `PL_TEST_MOVES=0`) | `proworlds_expansion_long_<game>` | `proworlds_expansion_long` = `test_suite/proworlds_long` |

The games (kind): face (relative), jigsaw (assemble), kaleido (symmetry), bird (assemble), fish (assemble), recipe (join),
turtle (relative), balloon (assemble), rocket (relative), bell (relative), bee (relative), candle (relative), sailboat
(assemble), barn (assemble), lantern (assemble).

The engine's README reports its own design checks. All normal-board rules pass for all 15 games. On the long board four games
(bell, candle, sailboat, barn) do not give at least 3 informative looks in every seed. Details:
[plearn/tasks/proworlds/README.md](../plearn/tasks/proworlds/README.md) and the engine's `README.md`.

## HiddenGames

Seven test tasks, each with an exact teacher (the posterior over the hidden target at every history). Test split: 100 games
each, FrozenLake and Sokoban 128 each (756). Validation seeds and training seeds are apart from the test seeds.

| Task | Source (license) | Hidden target | Our changes |
|---|---|---|---|
| `minesweeper_easy` | POPGym Arcade MineSweeper, pip `popgym-arcade==0.0.7` (MIT), full observation (`partial_obs=False`) | the mine cells (4 x 4, 2 mines) | the `SUBMIT` answer line, the splits, the teacher, the belief line. The game, the picture, the actions and the budget are upstream's |
| `minigrid_keycorridor` | Farama MiniGrid KeyCorridor, layout S4R3, pip `minigrid==3.2.0` (Apache-2.0) | the room of the key | the task (name the key's row, not the game's mission), the splits, the teacher, the belief line |
| `frozenlake_hidden_goal` | VAGEN FrozenLake, copy in `benchmarks/vagen` (MIT), on gymnasium 1.4.0 (MIT) | the goal cell and the holes that are not next to the agent | the goal and the holes are hidden (fog), one action a turn (25), the training and validation seeds, the teacher |
| `sokoban_hidden_target` | VAGEN Sokoban, copy in `benchmarks/vagen` (MIT), on gym_sokoban 0.0.6 (MIT) | the target square | the target is hidden, one action a turn (15), the training and validation seeds, the teacher |
| `tabletop_lamps`, `tabletop_cups`, `tabletop_blocks` | our own 3 tasks on the ManiSkill3 table scene, pip `mani-skill==3.0.1` (Apache-2.0) | which button lights which lamp; which cup hides the object; which block stands on a slippery base | everything except the table scene: the objects, the helpers, the tasks and the teacher |

Each task folder has a README with the full protocol (budget, observations, actions, splits, teacher):
`plearn/tasks/<dataset>/README.md`. MiniGrid's Apache-2.0 license comes from its LICENSE file.

`train_suite` also has 1 training-only task (see `plearn/tasks/suites/train_suite.yaml`). It is not in `test_suite`. It uses
the copy of TextArena in `benchmarks/textarena` (MIT).

`minesweeper_medium` and `minesweeper_hard` stay registered (larger boards), but no suite holds them.
