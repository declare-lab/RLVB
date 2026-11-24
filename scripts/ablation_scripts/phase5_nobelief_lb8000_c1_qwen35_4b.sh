#!/usr/bin/env bash
# RLVB (GRPO) ablation with Qwen3.5-4B: no belief score, phase 5 (phase_set 240, set_keep 160, beta_decay 0.9856), length_budget 8000, length_reward 1.
# The run file configs/train/phase5_nobelief_lb8000_c1_qwen35_4b.yaml says what differs from the RLVB (GRPO) run file
# configs/train/grpo_posterior_qwen35_4b.yaml; everything else is the same (train_suite, 50 updates, temperature 1.0, presence penalty
# 1.5, thinking cap 8,192 tokens). Then the test of adapter_50 on test_suite at each test seed, two-line turn, cap 8,192 tokens,
# temperature 1.0, as scripts/grpo_posterior_qwen35_4b.sh.
# Environment (all optional): PL_RUNS (the runs folder, default runs), PL_MODELS (the weights folders), PL_DATASETS, PL_ENV_SH (a
# file to source before training, for example the conda environment; default none), EVAL_GPUS (the GPUs of the test server, one
# tensor-parallel group, default 0), QWEN35_4B_PATH (the weights folder; not set: $PL_MODELS/Qwen3.5-4B), SEEDS (the test seeds, default "0 1 2"), EXPANSION (normal or long: the Pro-Worlds board of
# the test, default normal), SKIP_TESTS=1 (no tests), DRY=1 (print the commands, run nothing). PL_GPUS: the training GPUs (default
# 0,1,2,3; the picture server and the learner share them).
set -euo pipefail
cd "$(dirname "$0")/../.."                                   # the repository root
: "${PL_RUNS:=runs}" "${EVAL_GPUS:=0}" "${SEEDS:=0 1 2}" "${DRY:=0}" "${SKIP_TESTS:=0}"
export PL_RUNS
[ -z "${QWEN35_4B_PATH:-}" ] || export QWEN35_4B_PATH        # configs/models/qwen3.5-4b.yaml path
export PL_ENV_SH="${PL_ENV_SH:-/dev/null}"                  # no host profile unless one is given
MODEL=qwen3.5-4b SUITE=test_suite
export PL_PRESENCE_PENALTY=1.5                                 # the training and the test
export PL_GPUS="${PL_GPUS:-0,1,2,3}"
run() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }
say() { echo "[$(date '+%F %T')] phase5_nobelief_lb8000_c1_qwen35_4b: $*"; }

test_adapter() {                                             # the test of one adapter at each seed: scripts/eval_qwen35_4b.sh
  ADAPTER="$1" TAG="$2" BENCH=all TURN=two_line SEEDS="$SEEDS" EVAL_GPUS="$EVAL_GPUS" DRY="$DRY" bash scripts/eval_qwen35_4b.sh
}

NAME=phase5_nobelief_lb8000_c1_qwen35_4b
run python -m plearn train --config "$NAME"
AD="$PL_RUNS/$NAME/queue-$NAME/adapter_50"
[ "$DRY" = 1 ] || [ -f "$AD/adapter_config.json" ] || { echo "no adapter at $AD" >&2; exit 3; }
if [ "$SKIP_TESTS" = 1 ]; then say "SKIP_TESTS=1: no tests ($AD)"; exit 0; fi
test_adapter "$AD" "$NAME"
