# frozenlake

VAGEN's FrozenLake with the goal hidden: search for the goal and reach it, without a step into a hole. One of the 7
HiddenGames test tasks: `train_suite` plays its train split, `test_suite/hidden_games` its test split.

| | |
|---|---|
| Source | VAGEN's FrozenLake: `benchmarks/vagen` (mll-lab-nu/VAGEN @ 41df5e7, MIT): VAGEN's own map generator, on gymnasium's FrozenLake (pip `gymnasium==1.4.0`, `pygame-ce` for the picture), 4 × 4, not slippery |
| Task | `frozenlake_hidden_goal` |
| Hidden target θ | the goal's cell, and the holes that are not next to the agent: 15 goal cells × 16,384 hole sets, weighted as the generator draws them (each cell a hole with chance 0.2, a path from the start to the goal) |
| Evidence action | the game's moves: `LEFT`, `DOWN`, `RIGHT`, `UP`. A hole ends the game. |
| Answer | none: reaching the goal wins |
| Observation | the game's picture with fog over every cell that the agent has not been on or next to |
| Budget | 25 actions, one each turn (VAGEN's evaluation: 5 turns of up to 5 actions); the test split: 150 actions in 152 turns (`PL_TEST_MOVES=0`: the task's own) |
| Splits | test: seeds 20000 to 20127 (VAGEN's evaluation seeds, 128 episodes); validation: seeds 10000 + i; training: seeds from 10^6. Training and validation never use a test map (without this rule, about 0.3% of the training maps are test maps). |
| Teacher | the exact belief over (goal, holes). π*: a plan through the cells known to be safe, to a cell whose new neighbours the agent then sees, or onto the goal once it is known. Move cost 0.98. |
| Belief line | `Possible: goal at B3, D4`, `Possible: any` at the start |
| Our changes | the hidden goal and holes (fog over the cells not visited or next to the agent), one action each turn (25 in all), the training and validation seeds, the exact teacher and the belief line |
| Teacher's own play | validation, 128 episodes (`tools/teacher_play.py`): most likely decision 1.000 (6.4 actions); sampled from π*, as in training, 0.867 (14.1 actions) |

Files: `_default.yaml` (the template), `frozenlake_hidden_goal.yaml` (the task), `teacher.py` (the episode and the teacher; the
episode is `plearn.tasks.domain:DomainEpisode`).
