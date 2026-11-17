"""The RLVB (EM) run files reach the trainer with their recipe: the trainer flags and the server environment of
configs/train/suite_qwen3vl_8b_thinking.yaml and configs/train/suite_qwen35_4b.yaml, except the paths, the GPUs and the W&B
names. configs/train/suite_qwen35_4b.yaml has the 8B recipe at the METHOD caps (only --model, a path, is not compared)."""
import json
import os
import tempfile
import unittest
from unittest import mock

from plearn.api.registry import registry
from plearn.train import commands, load

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOT_COMPARED = {"--role", "--model", "--tag", "--out", "--queue", "--dump_em", "--dump_rollouts", "--wandb_project", "--wandb_entity",
                "--wandb_group", "--wandb_name", "--vllm_gpu"}
COMMON = {"--domain": "plearn.tasks.mixture", "--colocate": "1", "--resume_step": "0", "--eval_n": "0", "--watch": "", "--objective": "critic",
          "--proposal": "base", "--phase": "25", "--alpha": "1", "--rho": "fidelity", "--line_floor": "50", "--line_score": "decision",
          "--line_mix": "0.1", "--carry": True, "--episodes": "1", "--worlds": "4", "--G": "1", "--excite": "0.5", "--rho_rb": "1", "--B": "32",
          "--em_k": "8", "--em_m": "4", "--em_j": "32", "--em_more": "1", "--m_step": "blocks", "--think_chunk": "1", "--phase_set": "300",
          "--set_draw": "passes", "--set_prio": "0.5", "--set_keep": "200", "--accept_only": "1", "--em_more_joint": "2.0", "--leave_out": "0",
          "--max_think": "16384", "--eval_max_think": "8192", "--think_cost": "0", "--length_reward": "4", "--steps": "50", "--eval_every": "25",
          "--held_n": "40", "--eval_episodes": "0", "--eval_split": "validation", "--policy_metrics": "0", "--seed": "0", "--value": "log",
          "--tau": "0.1", "--readout": "0", "--dump_em_steps": "1,25,50,100", "--lr": "1e-05", "--weight_decay": "0.01", "--kl_stop": "0.015",
          "--kl_probe": "32", "--dec_term": "own", "--temperature": "1.0", "--vllm_mem": "0.85", "--ahead": "3", "--gen_depth": "2"}
SERVERS = {"PL_VLLM_TP": "1", "PL_ASYNC_EPISODES": "1", "PL_PLAN_STREAM": "1", "PL_CHUNKED_HEAD": "1", "PL_MAX_BATCHED": "65536",
           "PL_HTTP_WORKERS": "64", "PL_MAX_IMAGES": "128", "PL_M_SPLIT": "item", "PL_ROUTE_SPILL": "2", "PL_ROUTE_SPILL_MIN": "96",
           "PL_EVAL_DEFER": "1", "PL_SKIP_EVAL0": "1", "PL_GPUS_PER_SERVER": "1"}
TASKS8 = ["mastermind", "minesweeper_easy", "minigrid_keycorridor", "frozenlake_hidden_goal", "sokoban_hidden_target", "tabletop_lamps",
          "tabletop_cups", "tabletop_blocks"]
EVAL8 = "mastermind,minesweeper_easy,minigrid_keycorridor,frozenlake_hidden_goal,sokoban_hidden_target,tabletop"
# the base recipe: the 8 tasks of train_suite, max_think 16,384 (METHOD), the server context 57,344
BASE = ({**COMMON, "--domain_args": {"tasks": TASKS8, "images": 1}, "--eval_tasks": EVAL8},
        {**SERVERS, "PL_MAX_NUM_SEQS": "96", "PL_PRESENCE_PENALTY": "1.5", "PL_MAX_MODEL_LEN": "57344"})
RUNS = {
    # the 8B file = the base recipe + line_wide, score_presence 1, line_target 0.1 and max_think 8,192 (the server context
    # follows the cap: 57,344 - 8,192)
    "suite_qwen3vl_8b_thinking": ({**BASE[0], "--line_wide": True, "--line_target": "0.1", "--score_presence": "1", "--max_think": "8192"},
                                  {**BASE[1], "PL_MAX_MODEL_LEN": str(57344 - 8192)}),
    "suite_qwen35_4b": BASE,                                         # the 4B: the base recipe and server env
}


def flags(argv):
    a = argv[argv.index("plearn.method.trainer") + 1:]; d = {}; i = 0
    while i < len(a):
        k = a[i]; v = a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else True
        d[k] = v; i += 2 if v is not True else 1
    return {k: v for k, v in d.items() if k not in NOT_COMPARED}


class RunFileTests(unittest.TestCase):
    def test_the_runs_command_lines(self):
        reg = registry()
        with tempfile.TemporaryDirectory() as d:
            for m in ("Qwen3-VL-8B-Thinking", "Qwen3.5-4B"): os.mkdir(os.path.join(d, m))
            env = {k: v for k, v in os.environ.items() if k not in ("PL_GPUS", "PL_SEQS", "PL_PRESENCE_PENALTY", "QWEN35_4B_PATH")}
            with mock.patch.dict(os.environ, {**env, "PL_MODELS": d, "PL_RUNS": d}, clear=True):
                for name, (want, want_env) in RUNS.items():
                    with self.subTest(run=name):
                        c = commands(load(os.path.join(ROOT, "configs", "train", name + ".yaml")), 0, stamp="0000-0000")
                        got = flags(c["actor"]); got["--domain_args"] = json.loads(got["--domain_args"])
                        ev_got, ev_want = got.pop("--eval_tasks"), want["--eval_tasks"]
                        self.assertEqual(reg.tasks(ev_got.split(",")), reg.tasks(ev_want.split(",")))   # the same tasks
                        self.assertEqual(got, {k: v for k, v in want.items() if k != "--eval_tasks"})
                        lg = flags(c["learner"]); lg.pop("--eval_tasks")
                        self.assertEqual(lg, {k: v for k, v in got.items() if k not in ("--vllm_mem", "--ahead", "--gen_depth")} | {"--domain_args": lg["--domain_args"]})
                        self.assertEqual(c["env"], want_env)


if __name__ == "__main__":
    unittest.main()
