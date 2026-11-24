# Ablation scripts

The scripts of the RLVB (GRPO) ablation runs. Each script trains once with its run file `configs/train/<run>.yaml` on
`train_suite`, then tests `adapter_50` on `test_suite` at the test seeds 0, 1 and 2 (two-line turn, or the one-line turn for the
runs with no belief line; thinking cap 8,192 tokens, temperature 1.0, presence penalty 1.5) with the test script of its model (`scripts/eval_<model>.sh`). Each run file says what it
changes against the RLVB (GRPO) run file of its model (`configs/train/grpo_posterior_<model>.yaml`) and why. Run the scripts from any
folder; they change to the repository root. The environment variables are those of [scripts/README.md](../README.md) (`PL_GPUS`,
`EVAL_GPUS`, `SEEDS`, `SKIP_TESTS=1`, `DRY=1`, ...).

"No belief score" = `line_score: none`, `line_wide: false`, `line_target: -1`, `em_more_joint: 0`: the turn score is the action's
log pi* only; the belief line is still written and carried. "No belief line" = `no_belief_line: 1` (with no belief score): the
prompt has the action-only rules of the one-line turn and no carried line, each own turn is the thought, `</think>` and the action
line, and the test uses the one-line turn (`TURN=one_line`). "Pooled" = `phase_set: 1200`, `set_keep: 800` (the method: 300 and
200). "Phase 5" = `phase: 5`, `phase_set: 240`, `set_keep: 160`, `beta_decay: 0.9856` (a new sample set every 5 updates; the
method: 25).

| Run (script `<run>.sh`, run file `configs/train/<run>.yaml`) | Model | W&B run | What changes against RLVB (GRPO) |
|---|---|---|---|
| `abl_grpo_nobelief_qwen3vl_4b_thinking` | Qwen3-VL-4B-Thinking | `h8lahlew` | no belief score |
| `abl_grpo_noaction_qwen3vl_4b_thinking` | Qwen3-VL-4B-Thinking | `hg836a9u` | no action reward and no acceptance rule (`act_score: 0`, `accept_rule: 0`, `accept_only: 0`, `em_more: 0`, `set_keep: 0`, `phase_set: 200`) |
| `looks_nobelief_qwen3vl_4b_thinking` | Qwen3-VL-4B-Thinking | `u7cd59qy` | no belief score, `tau_look: 0.02` |
| `onpolicy_nobelief_qwen3vl_4b_thinking` | Qwen3-VL-4B-Thinking | `jp6bykha` | no belief score, on-policy (`phase: 1`, `phase_set: 40`, `set_keep: 32`, `beta_decay: 0.9971`) |
| `pool_nobelief_qwen3vl_4b_thinking` | Qwen3-VL-4B-Thinking | `6cwidaie` | no belief score, pooled |
| `pool_nobelief_qwen3vl_8b_thinking` | Qwen3-VL-8B-Thinking | `ukmbgztv` | no belief score, pooled |
| `pool_nobelief_qwen35_4b` | Qwen3.5-4B | `14rnckde` | no belief score, pooled |
| `pool_nobelief_lb8000_qwen35_4b` | Qwen3.5-4B | `t9wwd5k7` | no belief score, pooled, `length_budget: 8000` |
| `phase5_nobelief_lb8000_c1_qwen35_4b` | Qwen3.5-4B | `1f7wxh5u` | no belief score, phase 5, `length_budget: 8000`, `length_reward: "1"` |
| `phase5_nobelief_nolen_qwen35_4b` | Qwen3.5-4B | `makkn02j` | no belief score, phase 5, no length term (`length_reward: "0"`) |
| `phase5_nobelief_trow_qwen35_4b` | Qwen3.5-4B | `79bvn29l` | as `phase5_nobelief_nolen_qwen35_4b`, plus `rho_exact: 1`, `teacher_row: 1` |
| `phase5_nobelief_trow_stall_qwen35_4b` | Qwen3.5-4B | `67165dhm` | as `phase5_nobelief_trow_qwen35_4b`, plus `stall_cap: 3.5`, `stall_play: 1` |
| `pool_nobelief_trow_stall_qwen3vl_8b_thinking` | Qwen3-VL-8B-Thinking | `p0r9aiib` (stopped at update 25) | no belief score, pooled (phases of 25 updates), no length term, `rho_exact: 1`, `teacher_row: 1`, `stall_cap: 3.5`, `stall_play: 1` |
| `pool_nobelief_trow_stall_qwen3vl_4b_thinking` | Qwen3-VL-4B-Thinking | not trained | as `pool_nobelief_trow_stall_qwen3vl_8b_thinking` |
| `pool_nobeliefline_qwen3vl_8b_thinking` | Qwen3-VL-8B-Thinking | `jcgw4bl5` | no belief line at all (`no_belief_line: 1`), no belief score, pooled; one-line test |
| `pool_nobeliefline_qwen3vl_4b_thinking` | Qwen3-VL-4B-Thinking | `5o47hlqm` | no belief line at all (`no_belief_line: 1`), no belief score, pooled; one-line test |
| `phase5_nobeliefline_nolen_qwen35_4b` | Qwen3.5-4B | training | no belief line at all (`no_belief_line: 1`), no belief score, phase 5, no length term, 25 updates; one-line test of update 25 |

The trainer options of these runs (`--act_score`, `--accept_rule`, `--tau_look`, `--rho_exact`, `--teacher_row`, `--stall_cap`,
`--stall_play`, `--phase`, `--phase_set`, `--set_keep`, `--length_budget`, `--length_reward`, `--no_belief_line`) are described in the module text of
`plearn/method/trainer.py` and in [plearn/method/README.md](../../plearn/method/README.md); each is off by default, so the
other run files are unchanged. `PL_KEEP_LAST` (default 2) keeps the full state of the last 2 updates besides the 25 and 50 saves.

```sh
DRY=1 scripts/ablation_scripts/pool_nobelief_qwen3vl_8b_thinking.sh          # print the commands
PL_GPUS=0,1,2,3 scripts/ablation_scripts/pool_nobelief_qwen3vl_8b_thinking.sh   # train, then the tests
```

## Extra tests

The three test scripts (`scripts/eval_qwen3vl_8b_thinking.sh`, `eval_qwen3vl_4b_thinking.sh`, `eval_qwen35_4b.sh`) have the same
options, all off by default (the reported tests use none of them). Each option adds a suffix to the output folder.

| Test | Option | Command (one model; the others: their own script) |
|---|---|---|
| Ask-once check (Base, action only, seed 0): the first SUBMIT of an episode is not carried out; the agent sees its reply once with a request to check whether the evidence identifies the answer, and its second reply is carried out | `ASK_ONCE=1` (`think`: it also sees its first reasoning) | `ASK_ONCE=1 TURN=one_line SEEDS=0 scripts/eval_qwen3vl_8b_thinking.sh` |
| Forced submit: the RoboQuest episodes of a test that ended with no SUBMIT are replayed from that test's logged turns, then get one more turn for the SUBMIT alone, with no new thought | `BENCH=roboquest FINAL_SUBMIT=prefill ONLY=<list> REPLAY=<turns file>` (`FINAL_SUBMIT=1`: the extra turn with a new thought) | `ADAPTER=<the adapter of that test> SEEDS=<its seed> BENCH=roboquest FINAL_SUBMIT=prefill ONLY=no_submit.txt REPLAY=<run>.turns.csv.gz scripts/eval_qwen3vl_4b_thinking.sh` |
| Thinking cap 16,384 (Base, action only, seed 0) | `MAX_THINK=16384` | `MAX_THINK=16384 TURN=one_line SEEDS=0 scripts/eval_qwen3vl_8b_thinking.sh` |

`ONLY` is a file of lines `<task> <instance id>` (the RoboQuest episodes of the test with no SUBMIT); `REPLAY` is that test's turns
file, made by `tools/test_outputs_csv.py` (one row per turn, with the simulator's feedback); the replay counts the simulator
replies that differ from the logged feedback (`replay_mismatch` in each row). `PL_SAVE_THOUGHTS=1` keeps the thought text in the
turn records.

## Tools

| Tool | Use |
|---|---|
| `tools/test_outputs_csv.py` | the test outputs as tables: one row per turn (`<row>_seed<k>.turns.csv.gz`), the thoughts (with `PL_SAVE_THOUGHTS=1`), a manifest |
| `tools/testpass_a4a5.py` | behaviour (success, premature, late looks, budget), belief and thought measures from the saved test outputs, CPU only |
| `tools/first_set.py` | save, find and check the first sample set of a phase run, so that a run with the same settings starts from it (`PL_SET_FROM`); refused for an on-policy run |
| `tools/stall_breakdown.py` | what the stalling tries of a `--stall_cap` run are made of, from its kept batches |
