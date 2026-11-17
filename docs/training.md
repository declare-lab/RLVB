# Training

Each learner trains once on `train_suite` (8 training tasks: the 7 HiddenGames test tasks and 1 training-only task (see
`plearn/tasks/suites/train_suite.yaml`); train split) and tests the final checkpoint on `test_suite`. Three base models:
Qwen3-VL-8B-Thinking (`qwen3vl_8b_thinking`), Qwen3-VL-4B-Thinking (`qwen3vl_4b_thinking`) and Qwen3.5-4B (`qwen35_4b`). Each
run script trains, then runs the test script at the seeds 0, 1 and 2 (`SEEDS`). `DRY=1` prints the commands only. `SKIP_TESTS=1`
trains only. All runs use LoRA r 16, alpha 32.

| Learner | Reward | Run script | Run file |
|---|---|---|---|
| RLVB (EM) | posterior process reward, EM update | `scripts/pl_<model>.sh` | `configs/train/suite_<model>.yaml` (Qwen3.5-4B: `suite_qwen35_4b_presence_8k.yaml`) |
| RLVB (GRPO) | the same process reward, group-relative update | `scripts/grpo_posterior_<model>.sh` | `configs/train/grpo_posterior_<model>.yaml` |

## RLVB (EM)

The method and its code: [plearn/method/README.md](../plearn/method/README.md). Each training task has an exact teacher. At
every history the teacher knows the posterior over the hidden state. It scores the agent's belief line and its action. RLVB (EM) samples
the agent's own turns, weights them by these scores (the E-step) and fits the weighted turns (the M-step).

```sh
PL_GPUS=0,1,2,3 scripts/pl_qwen3vl_8b_thinking.sh        # train (50 updates), then test adapter_50
python -m plearn train --config suite_qwen3vl_8b_thinking --dry_run     # print the trainer command lines
python -m plearn train --config suite_qwen3vl_8b_thinking --resume_step 25
```

The settings: METHOD in `plearn/train.py`, and in each run file thinking cap 8,192, temperature 1.0, top-p 0.95, top-k 20,
presence penalty 1.5, `line_wide`, `line_target 0.1`, AdamW lr 1e-5, 50 updates (adapters 25 and 50). The output is
`$PL_RUNS/<run>/queue-<run>/adapter_<k>`. `PL_GPUS` sets the GPUs (the picture server and the learner share them).
`configs/train/abl_pl_4b*.yaml` are the ablation runs of the method's parts on Qwen3.5-4B.

## RLVB (GRPO)

The RLVB (EM) run file with `--update grpo`: the same samples and teacher scores, but group-relative (leave-one-out) advantages over
the own turns of each history in place of RLVB (EM)'s posterior weights, and a PPO clip against the phase prior
(`plearn/method/README.md`, "RLVB (GRPO): the group-relative update").

```sh
scripts/grpo_posterior_qwen3vl_8b_thinking.sh
```

## Full models

The released models are full models: the adapter merged into the base model.

```sh
python tools/full_model.py merge --base $PL_MODELS/Qwen3-VL-8B-Thinking --adapter <adapter> --out <dir> \
    --repo <org>/<name> --base_id Qwen/Qwen3-VL-8B-Thinking --about "<text>"
```

## W&B

Logging is off unless `WANDB_PROJECT` (or the run file's `wandb.project`) is set. The entity comes from `WANDB_ENTITY`. The key
comes from `WANDB_API_KEY` or `wandb login`.
