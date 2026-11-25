# RoboQuest at the decision level

The full-benchmark test of RoboQuest (`benchmarks/roboquest/`, MIT): the 500 eval50 instances of the ten tasks (50 each), at
the decision level with helpers. RoboQuest has no training task here.

- `roboquest_text` (`test_text/`, version text; in `test_suite/roboquest`, `test_suite`): each action's pictures are replaced
  by one sentence. This is the only version in the suites. Thinking is capped at 8,192 tokens.
- The code: `bench_env.py` (the episode for the evaluator), `bench_eval.py` (the test's own episode: prompt, helper actions,
  budget, scoring), `bench/` (one adapter per task: its helpers and its answer form), `bench_sim.py` (the simulator bridge
  process). The text version also needs the bridge: the helpers run in the simulator, and the sentence comes from its state.
- The bridge runs in the RoboQuest runtime interpreter. Set `ROBOQUEST_RUNTIME` to the runtime folder
  (`benchmarks/roboquest/README.md`: `scripts/bootstrap_simulator.sh`, then `source <runtime>/env.sh`). There is no default:
  with the variable not set, the task check names the missing interpreter `${ROBOQUEST_RUNTIME}/envs/robocasa/bin/python`.
- The turn: RLVB has two lines after the thought (the belief line `Possible: ...`, then the action line); this is the default.
  The action-only test (for example the base model's "Base" row) has one line: `--task_args roboquest.one_line=1` gives the
  system text without the `Possible:` hint, and after the thought the evaluator samples the action line alone (the first
  `ACTION:` line of the reply).

## Odd Parcel reading aid (off by default)

`RQ_DIAL_ZOOM=1` (or `dial_zoom: true` in a task's `env_args`) adds three things to Odd Parcel in the simulator version: a
fourth picture in each block of views (a zoom on the balance dial, cut from the left camera), the rule to read the pointer,
and a warning that the camera angle can make a level beam look slanted. The other nine tasks do not change. No registered
task sets it, and the text version does not use it.

## The test protocol

The agent works through helper actions with perfect control: the helpers move objects exactly with the task's own
privileged gates. The posterior and the score come from the evidence actions and the answer, never from motor control.

**The same as the benchmark:**

- the same 500 instances: the ids in `suite/v1/eval50/` and the frozen registry rows in `suite/v1/<task>/registry/`;
- the same scene builder (`roboquest.kitchen.make_env`) and the same task code;
- the benchmark's goal text of each instance;
- success by each task's own check (`_current_score`), frozen at the answer as at the Submit press;
- the progress P by the benchmark's own code (`roboquest/scoring/progress.py`), on the private files of the benchmark's
  runner that each episode keeps (`config.json`, `task-spec-private.json`, `physical-trace-private.jsonl`,
  `result.json`, then `task-progress.json`). An episode without an answer is scored on its end state with no success,
  as an episode without a press.

**Different because of the helpers:**

- the agent writes one helper line per decision and one SUBMIT line (its answer in a fixed form per task); a SUBMIT line
  in a wrong form ends the episode without success;
- nothing presses the Submit button: SUBMIT runs the answer helpers, the scene settles for 50 ticks, and the score is
  frozen. The episodes are decision-level episodes, not official ones (no replay, no `rollout.mp4`);
- the intro names the objects the agent can act on, so that a helper line can name them. It states only what the
  start view shows, never a hidden fact;
- the physical trace has one line per simulated tick; a helper that teleports an object shows the move at the next tick.

**Our own settings:**

| Setting | Benchmark | Here |
|---|---|---|
| Budget | 200 decisions (API agents) | test split: 150 evidence actions, then at most 152 decisions (`PL_TEST_MOVES=0`: T of each task (below), T + 2 decisions); a line the helpers cannot do also uses one |
| Agent turn | the benchmark's prompt and tools | the trainer's turn: reasoning, belief line, action; thinking cap 8,192 tokens; the task's sampling: temperature 0.6, top-p 0.95, top-k 20 (`PL_TEST_T` sets the test temperature; the run scripts use 1.0) |
| Memory | the pictures of the newest two observations; all text, with the agent's own earlier messages (`roboquest/harness/agent.py`) | each earlier action with its status and its reading (text version); not its earlier reasoning or belief lines (`carry: false` in `_bench.yaml`) |
| Text version | none | each action's pictures replaced by one sentence |

The evidence-action budget T of each task (the test split with `PL_TEST_MOVES=0`, and the dev split):

| Task | T | | Task | T |
|---|---|---|---|---|
| Odd Parcel | 12 | | Search Room | 28 |
| Painted Cubes | 17 | | Blackout Search | 23 |
| Marked Mugs | 8 | | Locked Storage | 12 |
| Unfamiliar Containers | 32 | | Puzzle Box | 14 |
| Stamp Composition | 10 | | Wobbly Stand | 8 |

## Run

```sh
# self-test of a task's helpers on eval50 instances: the oracle answer must succeed with P = 1, a wrong one must fail
$ROBOQUEST_RUNTIME/envs/robocasa/bin/python -m plearn.tasks.roboquest.bench.selftest odd_parcel 2 --out /tmp/selftest

# the test: success and P per task in results.json, one row per episode in <task>.jsonl, the private files in episodes/
python -m plearn eval --model qwen3-vl-8b-thinking --tasks test_suite/roboquest --gpus 0 [--adapter DIR] --output_path runs/eval/roboquest-text
# the action-only test (Base): the one-line turn
python -m plearn eval --model qwen3-vl-8b-thinking --tasks test_suite/roboquest --task_args roboquest.one_line=1 --gpus 0 --output_path runs/eval/roboquest-text-one-line
```
