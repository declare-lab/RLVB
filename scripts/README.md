# Run scripts

One script for each learner and base model. Each training script trains once on `train_suite` (8 training tasks: the 7
HiddenGames test tasks and 1 training-only task (see `plearn/tasks/suites/train_suite.yaml`); train split), then tests the final
checkpoint on `test_suite` at the test seeds 0, 1 and 2 with the test script of its model. Run them from any folder; they change
to the repository root. `DRY=1` prints the commands and runs nothing. `SKIP_TESTS=1` trains only.

| Script | Qwen3-VL-8B-Thinking | Qwen3-VL-4B-Thinking | Qwen3.5-4B | Test turn |
|---|---|---|---|---|
| Test only (the base model, a full model or an adapter) | `eval_qwen3vl_8b_thinking.sh` | `eval_qwen3vl_4b_thinking.sh` | `eval_qwen35_4b.sh` | `TURN` |
| RLVB (EM) | `pl_qwen3vl_8b_thinking.sh` | `pl_qwen3vl_4b_thinking.sh` | `pl_qwen35_4b.sh` | two_line |
| RLVB (GRPO) | `grpo_posterior_qwen3vl_8b_thinking.sh` | `grpo_posterior_qwen3vl_4b_thinking.sh` | `grpo_posterior_qwen35_4b.sh` | two_line |

The test script without a model or adapter tests the base model: `TURN=two_line` (the default) gives the "Base (w/ belief)" row,
`TURN=one_line` gives the "Base" row (action only). The training is described in [docs/training.md](../docs/training.md), the
tests in [docs/evaluation.md](../docs/evaluation.md).

```sh
PL_GPUS=0,1,2,3 scripts/pl_qwen3vl_8b_thinking.sh                         # RLVB (EM) 8B, then its tests
PL_GPUS=0,1,2,3 scripts/grpo_posterior_qwen3vl_8b_thinking.sh             # RLVB (GRPO) 8B, then its tests
MODEL_REPO=richmondsin/pl-qwen3-vl-8b-thinking scripts/eval_qwen3vl_8b_thinking.sh   # test a released model
TURN=one_line scripts/eval_qwen3vl_8b_thinking.sh                         # the base model, action only
BENCH=proworlds EXPANSION=long scripts/eval_qwen3vl_8b_thinking.sh       # the base model, Pro-Worlds long board only
```

## Environment variables

| Variable | Default | Use |
|---|---|---|
| `PL_RUNS` | `runs` | the output folder of the runs and the tests (`$PL_RUNS/eval/<tag>-seed<k>`) |
| `PL_MODELS` | none | the folder of the model weights (`configs/models/*.yaml` `local_dir`) |
| `QWEN35_4B_PATH` | none | optional: the Qwen3.5-4B weights folder (else `$PL_MODELS/Qwen3.5-4B`) |
| `PL_ENV_SH` | none | optional: a file to source before the training |
| `PL_GPUS` | `0,1,2,3` | RLVB (EM) and RLVB (GRPO): the training GPUs |
| `EVAL_GPUS` | `0` | the GPU of the test server |
| `SEEDS` | `0 1 2` | the test seeds (`PL_VLLM_SEED`) |
| `BENCH` | `all` | the tests: `all` (`test_suite`), `roboquest`, `proworlds`, `hidden_games` |
| `EXPANSION` | `normal` | the tests: `long` tests the Pro-Worlds long board (`test_suite_long`) |
| `TURN` | `two_line` | test scripts: `two_line` or `one_line` |
| `MODEL_REPO`, `ADAPTER`, `TAG` | none | test scripts: a full model, or an adapter (folder or Hugging Face id; none = the base model); the output name |
| `SKIP_TESTS` | `0` | training scripts: `1` = no tests after the training |
| `DRY` | `0` | `1`: print the commands, run nothing |

## Settings of the tests

Thinking cap 8,192 tokens, temperature 1.0 (`PL_TEST_T`), top-p 0.95 and top-k 20 (the tasks' own), presence penalty 1.5
(`PL_PRESENCE_PENALTY`), `PL_VLLM_SEED` = the seed, one GPU (TP 1), context 81,920 tokens (`PL_MAX_MODEL_LEN`), 192 requests at
once (`PL_MAX_NUM_SEQS`), 16,384 batched tokens (`PL_MAX_BATCHED`), GPU memory 0.85. Check that `errors` is 0 in each
`results.json`. `python tools/results_cells.py <seed 0 folder> <seed 1 folder> <seed 2 folder>` gives the cells of a results
table.
