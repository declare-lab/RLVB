# Evaluation

## The test protocol

Every model is tested on `test_suite` (1,556 games) at the test seeds 0, 1 and 2 (`PL_VLLM_SEED`). The settings:

| Setting | Value |
|---|---|
| Thinking cap | 8,192 tokens (`--max_think 8192`) |
| Sampling | temperature 1.0 (`PL_TEST_T=1.0`), top-p 0.95, top-k 20, presence penalty 1.5 (`PL_PRESENCE_PENALTY=1.5`) |
| Server | one GPU (tensor parallel 1), context 81,920 tokens, 192 requests at once, 16,384 batched tokens, GPU memory 0.85 |
| Turn | `two_line` (belief line + action line) for RLVB (EM), RLVB (GRPO) and Base (w/ belief); `one_line` (action line only) for Base |

## The test scripts

One script for each base model: `scripts/eval_qwen3vl_8b_thinking.sh`, `scripts/eval_qwen3vl_4b_thinking.sh`,
`scripts/eval_qwen35_4b.sh`. They run one `plearn eval` per seed.

| Variable | Values |
|---|---|
| `MODEL_REPO` | a full model: a weights folder or a Hugging Face id (for example `posterior-learning/pl-qwen3-vl-8b-thinking`). Not set: the base model |
| `ADAPTER` | a LoRA adapter folder or a Hugging Face id of an adapter (`ADAPTER_SUBDIR` when the adapter is not at the top) |
| `BENCH` | `all` (`test_suite`, the default), `roboquest`, `proworlds`, `hidden_games` |
| `EXPANSION` | `normal` (default) or `long`: the Pro-Worlds board. `long` gives `test_suite_long`, or `test_suite/proworlds_long` |
| `TURN` | `two_line` (default) or `one_line` |
| `SEEDS` | the test seeds (default `0 1 2`) |
| `EVAL_GPUS` | the GPU of the test server (default `0`) |
| `TAG` | the output name. Output: `$PL_RUNS/eval/<TAG>[-<BENCH>][-long][-one_line]-seed<S>` |
| `DRY=1` | print the commands, run nothing |

```sh
scripts/eval_qwen3vl_8b_thinking.sh                                            # Base (w/ belief): belief + action
TURN=one_line scripts/eval_qwen3vl_8b_thinking.sh                              # Base: action only
MODEL_REPO=posterior-learning/pl-qwen3-vl-8b-thinking scripts/eval_qwen3vl_8b_thinking.sh               # RLVB (EM)
MODEL_REPO=posterior-learning/grpo-posterior-qwen3-vl-8b-thinking scripts/eval_qwen3vl_8b_thinking.sh   # RLVB (GRPO)
ADAPTER=runs/suite_qwen3vl_8b_thinking/queue-suite_qwen3vl_8b_thinking/adapter_50 scripts/eval_qwen3vl_8b_thinking.sh
```

A new start with the same output folder plays only the missing episodes. Check that `errors` is 0 in each `results.json`.

## plearn eval

The scripts call the plearn evaluator. You can call it directly:

```sh
python -m plearn tasks --groups                       # the registered groups and their task counts
python -m plearn show test_suite                      # one group with its tasks expanded
python -m plearn eval --model qwen3-vl-8b-thinking --tasks test_suite --max_think 8192 --gpus 0 \
    [--adapter DIR] [--limit 8] [--split validation] [--task_args roboquest.one_line=1 proworlds.one_line=1] \
    [--output_path DIR] [--wandb_project P]
```

`--task_args DATASET.KEY=VALUE` sets an `env_args` value for every task of a dataset. The one-line turn needs it for each
dataset of the suite: `roboquest`, `proworlds`, `minesweeper`, `minigrid`, `frozenlake`, `sokoban`, `tabletop`
(the test scripts set all of them).

The output folder has one `<task>.jsonl` per task (one row per episode) and `results.json` (the metrics of each task and of
each group, pooled over its episodes).

## Result cells

`tools/results_cells.py` turns the test folders of seeds 0, 1 and 2 into the cells of a results table (the mean and the
seed values):

```sh
python tools/results_cells.py runs/eval/<tag>-seed0 runs/eval/<tag>-seed1 runs/eval/<tag>-seed2
```

