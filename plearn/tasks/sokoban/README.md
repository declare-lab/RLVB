# sokoban

VAGEN's Sokoban with the target square hidden: find the target and push the box onto it. One of the 7 HiddenGames test tasks:
`train_suite` plays its train split, `test_suite/hidden_games` its test split.

| | |
|---|---|
| Source | VAGEN's Sokoban: `benchmarks/vagen` (mll-lab-nu/VAGEN @ 41df5e7, MIT): VAGEN's own room code (its seeding, its shortest-solution search, its solution band, its map partition and its retry rule), on gym_sokoban (pip `gym-sokoban==0.0.6`, with `gym==0.26.2`) |
| Task | `sokoban_hidden_target`: 6 × 6, one box |
| Hidden target θ | the target square, drawn evenly from the cells that the box can reach (6.5 cells on average). The generator's own target is not used, because the generator plays backwards from it and its posterior is not exact. |
| Evidence action | the game's moves: `UP`, `DOWN`, `LEFT`, `RIGHT`; a step into the box pushes it when the cell behind it is free |
| Answer | none: the box on the target wins |
| Observation | the game's pictures (96 × 96, 16 px a cell, as VAGEN gives it) of the room at the start and after each action, in order, so the cells already checked are in the prompt; the target shows once the player is on it or next to it (not diagonally) |
| Budget | 15 actions, one each turn (VAGEN's evaluation: 5 turns of up to 3 actions) |
| Splits | test: seeds 20000 to 20127 (VAGEN's evaluation seeds, no map partition, 128 episodes); validation: seeds 10000 + i (VAGEN's `eval` partition); training: seeds from 10^6 (VAGEN's `train` partition). All rooms have VAGEN's solution band of 1 to 5 actions. Training and validation never use a test room (without this rule, 8.9% of the training rooms are test rooms). |
| Teacher | the exact belief: the candidate cells that no view has excluded, all equally likely. π*: with the target known, an exact search over (player, box); before that, the best of pushing toward a likely target and walking to see more cells. Move cost 0.98. |
| Belief line | `Possible: target at B3, D4`, `Possible: any` at the start |
| Our changes | the hidden target (drawn evenly from the cells the box can reach), one action each turn (15 in all), the training and validation seeds, the exact teacher and the belief line |
| Teacher's own play | validation, 128 episodes (`tools/teacher_play.py`): most likely decision 0.984 (5.1 actions); sampled from π*, as in training, 0.977 (9.8 actions) |

Files: `_default.yaml` (the template), `sokoban_hidden_target.yaml` (the task), `teacher.py` (the episode and the teacher; the
episode is `plearn.tasks.domain:DomainEpisode`).
