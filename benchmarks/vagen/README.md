# VAGEN: FrozenLake and Sokoban

The FrozenLake and Sokoban environments of [VAGEN](https://github.com/mll-lab-nu/VAGEN) (MIT, see `LICENSE`), at commit
[41df5e7](https://github.com/mll-lab-nu/VAGEN/tree/41df5e7dc4cd6d75b1d38244208d6931382d7bca). The plearn
datasets `frozenlake` and `sokoban` import VAGEN's own map and room code from this folder. Thus their maps, rooms and seeds
are VAGEN's.

| File | Use in plearn |
|---|---|
| `vagen/envs/frozenlake/utils/utils.py` | `generate_random_map`: the map of a seed (`plearn/tasks/frozenlake/teacher.py`) |
| `vagen/envs/sokoban/patch_sokoban_env.py`, `vagen/envs/sokoban/utils/seeding.py` | the room of a seed: VAGEN's seeding, shortest-solution search, solution band, map partition and retry rule (`plearn/tasks/sokoban/teacher.py`); the tests compare our rooms with `PatchedSokobanEnv.reset` |
| `vagen/envs/_common/response_format.py` | imported by `utils.py`; plearn does not use it |
| `vagen/envs/frozenlake/frozenlake_env.py`, `vagen/envs/sokoban/sokoban_env.py` | reference only: VAGEN's environments (they need VAGEN's framework to import) |
| `examples/evaluate/*/config.yaml`, `examples/train/*/*_vision.yaml` | reference only: VAGEN's evaluation, training and validation settings |

The files are VAGEN's files without change, except two package files. `vagen/envs/__init__.py` and
`vagen/envs/_common/__init__.py` are empty here, because upstream they import all of VAGEN's environments and its framework.
`plearn/__init__.py` puts this folder on the path (`import vagen`).

| | VAGEN (its `examples/` settings) | plearn |
|---|---|---|
| FrozenLake map | `generate_random_map(4, p=0.8, seed)`, not slippery | the same |
| Sokoban room | 6 × 6, one box, `min_solution_steps` [1, 5]; map partition `train` for training, `eval` for validation, none for the test | the same |
| Test seeds | 20000 to 20127 (128 episodes) | the same |
| Budget | FrozenLake: 5 turns of up to 5 actions; Sokoban: 5 turns of up to 3 actions | 25 and 15 actions, one action each turn |
| Hidden part | none: the picture shows the whole state | FrozenLake: the goal, and the holes that are not next to the agent; Sokoban: the target square, drawn evenly from the cells the box can reach |
| Training and validation | seeds 1 to 10000; a list of validation seeds | training seeds from 10^6, validation seeds 10000 + i; a test map or room is never used |
