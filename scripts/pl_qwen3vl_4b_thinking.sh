#!/usr/bin/env bash
# RLVB (EM) with Qwen3-VL-4B-Thinking, the settings of scripts/pl_qwen3vl_8b_thinking.sh: the training on
# train_suite, the 8 HiddenGames (configs/train/suite_qwen3vl_4b_thinking.yaml: 50 updates, temperature 1.0, top-p 0.95, top-k 20,
# presence penalty 1.5, presence score, line_wide, line_target 0.1, thinking cap 8,192 tokens), then the test of adapter_50 on test_suite
# (RoboQuest text, Pro-Worlds normal board, HiddenGames) at each test seed, cap 8,192 tokens, temperature 1.0.
# Environment (all optional): PL_RUNS (the runs folder, default runs), PL_MODELS (the weights folders), PL_DATASETS, PL_ENV_SH (a
# file to source before training, for example the conda environment; default none), EVAL_GPUS (the GPUs of the test server, one
# tensor-parallel group, default 0), SEEDS (the test seeds, default "0 1 2"), EXPANSION (normal or long: the Pro-Worlds board of
# the test, default normal), DRY=1 (print the commands, run nothing).
# PL_GPUS: the training GPUs (default 0,1,2,3; the picture server and the learner share them).
set -euo pipefail
cd "$(dirname "$0")/.."                                      # the repository root
: "${PL_RUNS:=runs}" "${EVAL_GPUS:=0}" "${SEEDS:=0 1 2}" "${DRY:=0}"
export PL_RUNS
export PL_ENV_SH="${PL_ENV_SH:-/dev/null}"                  # no host profile unless one is given
MODEL=qwen3-vl-4b-thinking SUITE=test_suite
export PL_PRESENCE_PENALTY=1.5                                 # the training and the test
run() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }

test_adapter() {                                             # the test of one adapter at each seed: scripts/eval_qwen3vl_4b_thinking.sh
  ADAPTER="$1" TAG="$2" BENCH=all TURN=two_line SEEDS="$SEEDS" EVAL_GPUS="$EVAL_GPUS" DRY="$DRY" bash scripts/eval_qwen3vl_4b_thinking.sh
}

NAME=suite_qwen3vl_4b_thinking
export PL_GPUS="${PL_GPUS:-0,1,2,3}"
run python -m plearn train --config "$NAME"
AD="$PL_RUNS/$NAME/queue-$NAME/adapter_50"
[ "$DRY" = 1 ] || [ -f "$AD/adapter_config.json" ] || { echo "no adapter at $AD" >&2; exit 3; }
test_adapter "$AD" "pl_qwen3vl_4b_thinking"
