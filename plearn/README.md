# plearn: tasks, evaluation and training for RLVB

plearn has the structure of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval): one package, every dataset a folder of
YAML task files with its code, every model and training run a YAML file in `configs/`, and one evaluator and one launcher
for all of them.

| Part | What |
|---|---|
| `api/` | `registry.py` (the YAML registry), `task.py` (a task: env, splits, metrics, teacher), `episode.py` (the episode contract and the agent's turn) |
| `tasks/<dataset>/*.yaml` | one task (`task: NAME`) or one group (`group: NAME` with `task: [...]`) per file; `_*.yaml` are templates |
| `tasks/<dataset>/*.py` | the dataset's episode (its specs per split, observations, actions and score) and, for the training tasks, its exact teacher (`teacher.py`) |
| `tasks/domain.py` | any trainer domain as an episode (the HiddenGames tasks use it: one definition for training and evaluation) |
| `tasks/mixture.py` | the training domain of a run: the teachers of the training tasks |
| `tasks/suites/` | `train_suite`, `test_suite`, `test_suite_long` and the parts `test_suite/<part>` ([README](tasks/suites/README.md)) |
| `method/` | RLVB (EM) and RLVB (GRPO): `trainer.py`, `estimator.py`, `teacher.py` (the domain contract) ([README](method/README.md)) |
| `models/` | the model registry's resolver, the picture server (`mm_server.py`), the samplers (`sampler.py`), `serve_vllm.py` |
| `evaluator.py`, `train.py`, `cli.py`, `loggers.py` | the evaluator, the training launcher, `python -m plearn`, W&B |
| `../configs/` | `models/*.yaml` (`local_dir` under `$PL_MODELS`, or `path`), `train/*.yaml` (model, `train_tasks`, `eval_tasks`, GPUs, settings, W&B) |

## Datasets

| Dataset | Tasks | Groups |
|---|---|---|
| `roboquest` | `roboquest_text_<task>`: the ten RoboQuest tasks, 50 eval50 instances each, text version | `roboquest_text` |
| `proworlds` | `proworlds_expansion_<game>` (normal board) and `proworlds_expansion_long_<game>` (long board): the 15 Pro-Worlds games, 20 plays each | `proworlds_expansion`, `proworlds_expansion_long` |
| `minesweeper` | `minesweeper_easy` (in the suites), `minesweeper_medium`, `minesweeper_hard` | `minesweeper` |
| `minigrid` | `minigrid_keycorridor` | |
| `frozenlake` | `frozenlake_hidden_goal` | |
| `sokoban` | `sokoban_hidden_target` | |
| `tabletop` | `tabletop_lamps`, `tabletop_cups`, `tabletop_blocks` | `tabletop` |
| `suites` | across datasets | `train_suite`, `test_suite`, `test_suite_long`, `test_suite/roboquest`, `test_suite/proworlds`, `test_suite/proworlds_long`, `test_suite/hidden_games` |

`train_suite` has 8 training tasks: the 7 HiddenGames test tasks and 1 training-only task (see
`plearn/tasks/suites/train_suite.yaml`). The training-only task has its own dataset folder in `tasks/`.

`python -m plearn tasks [--groups]` lists them; `python -m plearn show NAME` prints one with its templates and environment
variables resolved.

## Run

```sh
# evaluate a model (and an adapter) on tasks and groups; results.json and one <task>.jsonl per task in the output folder
python -m plearn eval --model qwen3-vl-8b-thinking --tasks test_suite --max_think 8192 --gpus 0 [--adapter DIR] [--limit 8] \
    [--split validation] [--task_args proworlds.one_line=1] [--wandb_project P --wandb_name N]

# train: the run file names the model, the training tasks and the tasks evaluated after every phase
python -m plearn train --config suite_qwen3vl_8b_thinking [--dry_run] [--resume_step 25]          # RLVB (EM)
python -m plearn train --config grpo_posterior_qwen3vl_8b_thinking [--dry_run] [--resume_step 25]  # RLVB (GRPO)
```

An evaluation saves every finished episode at once; a new start with the same output folder plays only the missing ones. An
episode that fails (the server or the environment, not the agent) is left out of the scores and counted as `errors`. A group's
scores pool the episodes of its tasks.

**Server settings** (environment variables, in a run file's `server:` block or the shell; each one unset changes nothing):

| Variable | What it does |
|---|---|
| `PL_PRESENCE_PENALTY` | vLLM's presence penalty on every sampled request of the picture servers and the text server (the runs use 1.5) |
| `PL_TEST_T` | the test temperature: when set, `temperature` and `line_temperature` of every task (`api/task.py`); not set: each task's own |
| `PL_VLLM_SEED` | the engine's seed of the picture server (`plearn eval` and training); not set: vLLM's default seed 0 |
| `PL_MAX_MODEL_LEN`, `PL_MAX_NUM_SEQS`, `PL_MAX_BATCHED` | the server's context, requests at once and batched tokens |
| `PL_GPUS_PER_SERVER` | k GPUs a picture server: the actor's GPUs in groups of k, one server a group |
| `PL_ATTN` | the attention of the learner (`flash_attention_3`, `flash_attention_2`, `sdpa`) |
| `PL_NO_CUDNN_SDP` | `1`: torch's cuDNN SDPA backend is off in every plearn process |

## Add a dataset

1. A folder `plearn/tasks/<dataset>/` with `__init__.py` and `env.py`. The episode class has `specs(split, n, seed, **env_args)`
   (the first n episode specs of a split, as JSON values), `__init__(spec, **env_args)`, `done`, `messages(carry)` (the
   system text, the user parts and the pictures of the next turn), `step(belief, action, record)`, `row()` (the episode's
   result, with `correct` and the metric fields) and `close()` (see `episode.py`).
2. A template `_default.yaml` (`dataset`, `env: env:Episode`, the splits, `num_episodes`, `generation_kwargs`,
   `metric_list`), and one file per task that includes it and sets `task:` and `env_args:`. Groups are files with `group:`.
3. For training, a teacher: a class with the domain contract of `plearn/method/trainer.py` (beliefs, looks, the lookahead), named in
   the task files as `teacher: teacher:Domain` with its `teacher_args`. A trainer domain that already exists can be the
   episode too (`env: plearn.tasks.domain:DomainEpisode`, `env_args: {domain: plearn.tasks.frozenlake.teacher}`).
4. Data goes in `$PL_DATASETS/<dataset>/` (one folder per dataset, like `$PL_MODELS`), named in the task files with
   `${PL_DATASETS}`; `requires: {paths: [...]}` makes a run stop at once when it is missing.
5. The benchmark's own code: a pip package, pinned in `pyproject.toml` (an optional group, such as `games`) and named in
   `requires: {modules: [...]}`; or a copy in `benchmarks/<name>/` at a fixed commit, with its LICENSE and a README of the files
   plearn uses. The teacher imports that code. Where it must keep a part of it (the training-only task keeps TextArena's
   feedback and rules text), a test in `tests/tasks/` checks that part against the copy in `benchmarks/`.
6. A `README.md` in the dataset folder: the source and its version, the splits and their seeds, the budget, the hidden target,
   the evidence actions, the answer, and the teacher's own play.

## W&B

Training logs every update under `train/` and every evaluation under `eval/` (`eval/tasks.<task>.<metric>`,
`eval/groups.<group>.<metric>`), with the training file's `wandb:` settings. `plearn eval` logs `eval/<name>/<metric>` and a
table when `--wandb_project` (or `WANDB_PROJECT`) is set; with no project nothing is logged. The entity comes from
`--wandb_entity` or `WANDB_ENTITY`. The key comes from the environment (`WANDB_API_KEY` or `wandb login`) and is never written
into the repository.
