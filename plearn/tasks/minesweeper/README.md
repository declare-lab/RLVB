# minesweeper

MineSweeper of POPGym Arcade, with the game's own picture and actions: find the mine cells. `minesweeper_easy` is one of
the 7 HiddenGames test tasks: `train_suite` plays its train split, `test_suite/hidden_games` its test split.

| | |
|---|---|
| Source | POPGym Arcade MineSweeper (pip `popgym-arcade==0.0.7`, MIT; JAX on the CPU): `popgym_arcade.environments.minesweeper`, `partial_obs=False`, pictures of 256 px |
| Tasks | `minesweeper_easy` (4 × 4, 2 mines: 120 mine sets; in the suites); `minesweeper_medium` (6 × 6, 6 mines: 1,947,792 sets; with a teacher) and `minesweeper_hard` (8 × 8, 10 mines; no teacher) stay registered, but no suite holds them; group `minesweeper` |
| Hidden target θ | the mine cells, drawn as the game draws them (every set equally likely) |
| Evidence action | the game's own: `MOVE UP`, `MOVE DOWN`, `MOVE LEFT`, `MOVE RIGHT` (the cursor) and `FIRE` (reveal the cell under the cursor). A revealed mine ends the game. |
| Answer | `ACTION: SUBMIT <cells>`: the mine cells; right when it names exactly the mines; it ends the game. Every safe cell revealed also wins. |
| Observation | the game's picture |
| Budget | the game's: 2 n² actions (32 easy, 72 medium, 128 hard) |
| Splits | test: seeds 20000 + i (100 episodes); validation: seeds 10000 + i; training: seeds from 10^6 |
| Teacher | the exact belief: the mine sets that agree with every revealed cell (a 64-bit mask each). π*: FIRE is worth its outcomes, two reveals ahead while at most 50,000 sets remain; a move is worth the best FIRE that its shortest cursor path reaches. Move cost 0.95. |
| Belief line | `Possible: mines at B2, D5; safe at C3` (the cells that the evidence decides), `Possible: any` when it decides none |
| Our changes | the answer line `ACTION: SUBMIT <cells>`, the splits, the exact teacher and the belief line; the game, its picture, its actions and its budget are POPGym Arcade's |
| Teacher's own play | validation, 128 episodes each (`tools/teacher_play.py`): easy: most likely decision 0.789 (10.5 actions), sampled from π* as in training 0.719 (18.6 actions); medium: 0.555 (27.1 actions) and 0.453 (43.9 actions) |

A revealed mine is the main cause of a failed game, also for the teacher: the game has no safe first reveal, so some games
are lost by a forced guess.

Files: `_default.yaml` (the template), one task file per board, `minesweeper.yaml` (the group), `teacher.py` (the episode and
the teacher; the episode is `plearn.tasks.domain:DomainEpisode`).
