# minigrid

MiniGrid KeyCorridor, with MiniGrid's own views and actions: find the room that holds the key. One of the 7 HiddenGames test
tasks: `train_suite` plays its train split, `test_suite/hidden_games` its test split.

| | |
|---|---|
| Source | Farama MiniGrid (pip `minigrid==3.2.0`, Apache-2.0): `minigrid.envs.keycorridor`, layout S4R3 |
| Task | `minigrid_keycorridor`: three rows of 4 × 4 rooms; the key is in one of the three left rooms (12 places) |
| Hidden target θ | the key's place: its row drawn evenly, its cell drawn evenly in the room, as the game draws it |
| Evidence action | MiniGrid's own: `TURN LEFT`, `TURN RIGHT`, `FORWARD`, `TOGGLE` (open the closed door in front) |
| Answer | `ACTION: SUBMIT <row>`: the left room in that row (1 = the top); it ends the game. The game's own mission is not the task. |
| Observation | MiniGrid's 7 × 7 view in front of the agent (224 px; the newest 32 views); walls and closed doors hide what is behind them |
| Budget | 64 actions |
| Splits | test: seeds 20000 + i (100 episodes); validation: seeds 10000 + i; training: seeds from 10^6 |
| Teacher | the exact belief: the places that no view has shown empty. π*: the best view that a shortest plan of turns, steps and door openings reaches. A place of the key counts as free until it is seen, so a plan does not use the true place. Move cost 0.98. |
| Belief line | `Possible: rows 1, 3`, `Possible: all` before any evidence |
| Our changes | the task (name the key's room: `ACTION: SUBMIT <row>`, not the game's own mission), the splits, the exact teacher and the belief line |
| Teacher's own play | validation, 128 episodes (`tools/teacher_play.py`): most likely decision 1.000 (8.0 actions); sampled from π*, as in training, 0.938 (34.0 actions) |

Files: `_default.yaml` (the template), `minigrid_keycorridor.yaml` (the task), `teacher.py` (the episode and the teacher; the
episode is `plearn.tasks.domain:DomainEpisode`).
