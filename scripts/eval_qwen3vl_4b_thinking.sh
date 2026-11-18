#!/usr/bin/env bash
# The test of Qwen3-VL-4B-Thinking on test_suite: one plearn eval per seed and benchmark.
# The settings: thinking cap 8,192 tokens (--max_think 8192), test temperature 1.0 (PL_TEST_T=1.0), top-p 0.95 and top-k 20 (the
# tasks' own), presence penalty 1.5, PL_VLLM_SEED = the test seed.
# The server: context 81,920 tokens (PL_MAX_MODEL_LEN), 192 requests at once (PL_MAX_NUM_SEQS), 16,384 batched tokens
# (PL_MAX_BATCHED), GPU memory 0.85 (--mem), one GPU (TP 1).
# ADAPTER    a LoRA adapter folder, or a Hugging Face repository id of an adapter (downloaded with huggingface_hub; HF_TOKEN for a
#            private one; ADAPTER_SUBDIR: the adapter's folder in the download when it is not the top). Not set: the base model.
# MODEL_REPO a full model in place of the base model: a weights folder, or a Hugging Face repository id (for example
#            richmondsin/pl-qwen3-vl-4b-thinking; downloaded with huggingface_hub). Not set: the registered base model.
# BENCH      all (the default: test_suite), roboquest (test_suite/roboquest), proworlds (test_suite/proworlds; alias: expansion),
#            hidden_games (test_suite/hidden_games, the 7 HiddenGames on their test split; alias: games): one benchmark of the suite
# EXPANSION  normal (the default) or long: the Pro-Worlds board. long: test_suite_long (BENCH=all) or test_suite/proworlds_long
#            (BENCH=proworlds), and the output name gets "-long".
# TURN       two_line (the default: the RLVB turn, the belief line and the action line) or one_line (the action line alone:
#            --task_args <dataset>.one_line=1 for RoboQuest, Pro-Worlds and HiddenGames; the base model's action-only test)
# SEEDS      the test seeds (default "0 1 2");  EVAL_GPUS the GPUs of the test server, one tensor-parallel group (default 0: one GPU)
# TAG        the output name (default: untrained_<model>, the adapter folder's name or the repository id); output
#            $PL_RUNS/eval/<TAG>[-<BENCH>][-long][-one_line]-seed<S> (BENCH all: no -<BENCH>); a new start plays only the missing episodes
# DRY=1      print the commands, run nothing (no download)
set -euo pipefail
cd "$(dirname "$0")/.."                                      # the repository root
: "${PL_RUNS:=runs}" "${EVAL_GPUS:=0}" "${SEEDS:=0 1 2}" "${DRY:=0}" "${TURN:=two_line}" "${BENCH:=all}" "${ADAPTER:=}" "${ADAPTER_SUBDIR:=}"
: "${EXPANSION:=normal}" "${MODEL_REPO:=}"
export PL_RUNS
MODEL=qwen3-vl-4b-thinking
run() { echo "+ $(printf '%q ' "$@")"; [ "$DRY" = 1 ] || "$@"; }   # the command, quoted as the shell reads it
case "$EXPANSION" in
  normal) LONG="" ;;
  long) LONG="_long" ;;
  *) echo "EXPANSION: normal or long" >&2; exit 2 ;;
esac
case "$BENCH" in
  all) TASKS=test_suite$LONG; SUF="" ;;
  roboquest) TASKS=test_suite/roboquest; SUF="-roboquest" ;;
  proworlds|expansion) TASKS=test_suite/proworlds$LONG; SUF="-$BENCH" ;;
  hidden_games|games) TASKS=test_suite/hidden_games; SUF="-$BENCH" ;;
  *) echo "BENCH: all, roboquest, proworlds or hidden_games" >&2; exit 2 ;;
esac
case "$BENCH" in roboquest|hidden_games|games) ;; *) [ -n "$LONG" ] && SUF="$SUF-long" ;; esac
TA=()
case "$TURN" in
  two_line) ;;
  one_line) TA+=(roboquest.one_line=1 proworlds.one_line=1 minesweeper.one_line=1 minigrid.one_line=1 frozenlake.one_line=1
                 sokoban.one_line=1 tabletop.one_line=1); SUF="$SUF-one_line" ;;
  *) echo "TURN: two_line or one_line" >&2; exit 2 ;;
esac
download() {                                                 # a Hugging Face repository id -> its local folder in DL (DRY: a placeholder)
  if [ "$DRY" = 1 ]; then echo "# download $1 (huggingface_hub snapshot_download)"; DL="<the download of $1>"
  else DL=$(python -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))' "$1"); fi
}
is_repo() { [ ! -e "$1" ] && [[ "$1" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; }
DEFAULT_TAG=untrained_qwen3vl_4b_thinking
if [ -n "$MODEL_REPO" ]; then                                # a full model: its folder is the --model
  if is_repo "$MODEL_REPO"; then DEFAULT_TAG=${MODEL_REPO//\//_}; download "$MODEL_REPO"; MODEL=$DL
  else DEFAULT_TAG=$(basename "$MODEL_REPO"); MODEL=$MODEL_REPO; fi
fi
ARGS=()
if [ -z "$ADAPTER" ]; then TAG=${TAG:-$DEFAULT_TAG}
elif ! is_repo "$ADAPTER"; then TAG=${TAG:-$(basename "$ADAPTER")}; ARGS=(--adapter "$ADAPTER")
else                                                         # a Hugging Face repository id
  TAG=${TAG:-${ADAPTER//\//_}}
  download "$ADAPTER"; AD=$DL
  if [ "$DRY" != 1 ]; then
    [ -n "$ADAPTER_SUBDIR" ] && AD="$AD/$ADAPTER_SUBDIR"
    if [ ! -f "$AD/adapter_config.json" ]; then                # one adapter folder inside the download
      mapfile -t found < <(find "$AD" -name adapter_config.json -printf '%h\n' | sort)
      [ "${#found[@]}" = 1 ] || { echo "$ADAPTER: ${#found[@]} adapter folders (${found[*]:-none}); set ADAPTER_SUBDIR (a full model: MODEL_REPO)" >&2; exit 3; }
      AD=${found[0]}
    fi
  fi
  ARGS=(--adapter "$AD")
fi
[ ${#TA[@]} -gt 0 ] && ARGS+=(--task_args "${TA[@]}")
for s in $SEEDS; do
  run env PL_PRESENCE_PENALTY=1.5 PL_VLLM_SEED="$s" PL_TEST_T=1.0 PL_MAX_MODEL_LEN=81920 PL_MAX_NUM_SEQS=192 PL_MAX_BATCHED=16384 python -m plearn eval --model "$MODEL" \
      --tasks "$TASKS" "${ARGS[@]}" --max_think 8192 --mem 0.85 --gpus "$EVAL_GPUS" --output_path "$PL_RUNS/eval/$TAG$SUF-seed$s"
done
