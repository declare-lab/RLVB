# proworlds

Pro-Worlds: find the answer about a hidden world from pictures of a moving window. The tasks play the 15 visual games of the
frozen expansion v6.1 engine (`benchmarks/proworlds/games/expansion-6.1`; its `SHA256SUMS` lists every file). The tasks are
for the test only: Pro-Worlds has no training task and no teacher.

## The tests

| Group | Tasks | Board | Games |
|---|---|---|---|
| `proworlds_expansion` (in `test_suite/proworlds`, `test_suite`) | `expansion/<game>.yaml`: 15 games x 20 plays | normal: 150 moves (35 with `PL_TEST_MOVES=0`) | 300 |
| `proworlds_expansion_long` (in `test_suite/proworlds_long`, `test_suite_long`) | `expansion_long/<game>.yaml`: the same 15 games and 20 plays | long: 150 moves (65 with `PL_TEST_MOVES=0`) | 300 |

- Game k is line k of `TEST-ORDER.txt`. Its 20 test plays use the seeds 50,101,300 + 10,000 (k - 1) + j, j = 0 .. 19. The
  long board uses the same seeds.
- The episode is `expansion_env.py`. It runs the engine through `expansion_bridge.mjs`, one Node.js process per episode
  (Node.js 22 or later: `NODE_BIN`, else `node` on `PATH`).
- The window shows one site of the board (one 16 x 16 tile, drawn 16 times larger). A move goes to the next site. At the
  edge the window stays, and the move still counts, as the engine does. The answer is one member name.
- The engine budget is 150 moves and the turn limit 152 (`PL_TEST_MOVES=0`: the board's own budget, 35 or 65, and the turn limit the budget + 8). A line that cannot be read uses up the turn. A move with no moves left ends the game
  without an answer (wrong).
- Score: `success` (the answer is the member of the world); `premature` (the engine's exact belief of that member was below
  0.95 at the answer); `submitted`, `moves`, `turns`, `unreadable_turns`, `thought_cut`.
- Thinking is capped at 8,192 tokens in every task.

## The turn

`format.py` is the agent's turn: the public rules of the game (the question, the answer set and the budget), the answers as
numbers (`1 = <name>`), the picture of the start site and of the site after every move, the agent's belief line of its last
turn, the moves used and the turns used. The agent writes its reasoning, then two lines: `Possible: <answer numbers>` and
`ACTION: MOVE UP|DOWN|LEFT|RIGHT` or `ACTION: SUBMIT <answer number or name>`. This two-line turn is the default (RLVB).

- `--task_args proworlds.one_line=1`: the one-line turn (action only, for example the base model's "Base" row). The system
  text asks for the action line alone, the evaluator samples the action line alone after the thought (`plearn/api/episode.py`),
  and no belief line is carried to the next turn.

## Run

```sh
python -m plearn eval --model qwen3-vl-8b-thinking --tasks test_suite/proworlds --gpus 0 [--adapter DIR]
python -m plearn eval --model qwen3-vl-8b-thinking --tasks test_suite/proworlds_long --gpus 0 [--adapter DIR]
python -m plearn eval --model qwen3-vl-8b-thinking --tasks proworlds_expansion_face --task_args proworlds.one_line=1 --gpus 0
```

Files: `expansion/_expansion.yaml` and `expansion_long/_expansion_long.yaml` (the templates), one task file per game and
board, `proworlds_expansion.yaml` and `proworlds_expansion_long.yaml` (the groups), `expansion_env.py` (the episode),
`expansion_bridge.mjs` (the engine bridge), `format.py` (the turn).
