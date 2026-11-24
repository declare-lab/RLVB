# RLVB (EM)

RLVB (EM) is a reinforcement learning algorithm for agents that must collect evidence before they answer.
At each step, the environment returns a teacher in place of a scalar reward:

| | One environment step |
|---|---|
| Classical reinforcement learning | $\mathrm{step}(a)\mapsto(o,\ r)$: $r$ is a sparse scalar reward |
| RLVB (EM) | $\mathrm{step}(a)\mapsto(o,\ \mathcal T_t)$: $\mathcal T_t$ is computed from the environment's own state at every step |

The teacher's posterior teaches the agent to read its evidence. A lookahead on that posterior teaches the agent to seek
evidence and to stop.

## Method

**Problem setting.** Each episode has a target $\theta\sim p_0$ that the agent cannot see. At each step $t$, the agent
takes an evidence action $a\in\mathcal A$, which returns an observation $o\sim p(o\mid\theta,a)$ shown as a rendering
$x=g(o)$ (text or an image), or it gives an answer $y\in\mathcal Y$, which ends the episode. The return is

$$R=c(T_{\text{stop}})\,\mathbf 1\big[y=y(\theta)\big],\qquad c(t)=e^{-\lambda t/T}. \tag{1}$$

The belief is the posterior over the target, $b_t(\theta)=p(\theta\mid h_t)\propto p_0(\theta)\prod_{s\le t}p(o_s\mid\theta,a_s)$ (2).
The policy $\pi_\phi$ is a language model. At each step it writes a turn $(z_t,\beta_t,u_t)$: the reasoning, the belief
line (the targets that are still possible) and the action.

**The teacher.** The posterior environment returns $\mathcal T_t=\big(z^*_t,\ \beta^*_t,\ \pi^*(\cdot\mid b_t)\big)$ (3):
the update trace, the correct belief line and the teacher policy. The domain gives only $p_0$, $p(o\mid\theta,a)$,
$y(\theta)$ and $g$.

**Teaching to read.** $\beta^*_t$ is a fixed text form of $b_t$. The update trace does the Bayes update one observation at a time:

$$z^*_t=\Big(o_s,\ \big\{\big(\theta,\ p(o_s\mid\theta,a_s)\big)\big\}_{\theta\in\operatorname{supp}b_{s-1}}\Big)_{s=1}^{t}. \tag{4}$$

The first part, $o_s$, is the reading of the rendering $x_s$. The trace is the teacher's own derivation, kept for logs and
checks; it is not a training target, because a template thought would replace the agent's own reasoning. The agent's
reasoning $z$ is a latent variable: what is trained is the probability of the correct belief line and action with $z$
summed out, $\pi_\phi(\beta^*,u^*\mid h)=\sum_z\pi_\phi(z\mid h)\,\pi_\phi(\beta^*,u^*\mid h,z)$ (5).

**Teaching to seek and to stop.** A lookahead of depth $d$ in the belief process, with the agent's measured answer accuracy $\hat\rho$:

$$V_0(b)=c(t)\,\hat\rho(b),\qquad Q_k(b,a)=\sum_o p(o\mid b,a)\,V_{k-1}(b^{a,o}),\qquad V_k(b)=\max\{V_0(b),\ \max_a Q_k(b,a)\}, \tag{6, 7}$$

$$\pi^*(u\mid b)\propto\exp\big(Q_d(b,u)/\tau\big),\qquad \hat\rho(b)=\frac{\kappa_0\,U(b)+s_{\kappa(b)}}{\kappa_0+n_{\kappa(b)}},\qquad U(b)=\max_y p(y\mid b). \tag{8, 9}$$

Here $\kappa(b)$ is the number of answers that are still possible, $n_\kappa$ counts the agent's answers from class $\kappa$
and $s_\kappa$ the correct ones.

**Objective and update.** Maximize $\mathbb E[R]$ under the reading constraint $\mathbb E[-\log\pi_\phi(\beta^*_t\mid h_t)]\le\varepsilon$ (10).
A policy mirror-descent step $\pi_{k+1}\propto\pi_k\exp(\eta A_k)$ (11) with the exact $Q_d$ in place of a sampled
advantage gives $\pi^*$. With the constraint as a penalty, the update maximizes (5) with $u^*\sim\pi^*$. Its gradient is the
EM gradient $\nabla\log\pi_\phi(\beta^*,u^*\mid h)=\mathbb E_{z\sim\pi_\phi(z\mid h,\beta^*,u^*)}\big[\nabla\log\pi_\phi(z,\beta^*,u^*\mid h)\big]$,
estimated with $K$ of the agent's own thoughts $z_k\sim\pi_\phi(z\mid h)$ and the weights
$w_k=\pi_\phi(\beta^*,u^*\mid h,z_k)/\sum_j\pi_\phi(\beta^*,u^*\mid h,z_j)$:

$$\mathcal L(\phi)=-\mathbb E_{h_t\sim d_k,\ u^*\sim\pi^*}\sum_{k=1}^{K}\Big[\big(w_k-\tfrac1K\big)\frac{\log\pi_\phi(z_k\mid h_t)}{|z_k|}+w_k\log\pi_\phi(\beta^*_t\mid h_t,z_k)+w_k\log\pi_\phi(u^*\mid h_t,z_k,\beta^*_t)\Big]. \tag{12}$$

The baseline $1/K$ leaves the expectation unchanged (the score of the model's own samples has mean zero) and removes the
large common part; the thought term is a per-token mean. Thoughts that lead to the correct belief and action gain
probability, and the thought stays in the agent's own words (`--em_k`, default 4).

**Algorithm.**

1. Warm-up: minimize (12) on histories of the teacher policy.
2. For $k=1,2,\dots$: play episodes, and at each step use the teacher's action with probability $\alpha_k=\alpha_0\gamma^k$,
   else the agent's own action. Get $\mathcal T_t$ at each history, do one gradient step on (12), and update
   $s_\kappa,n_\kappa$ with the results of the agent's answers.
3. Test: the agent writes its reasoning greedily and takes its most probable action.

Two details of the code: the warm-up histories mix the teacher policy with random evidence actions (`--mix`), and the
lookahead uses an upper confidence bound of $\hat\rho$ at its last depth (`--ucb_z`).

## Code

| File | Part of the method |
|---|---|
| `teacher.py` | The generic posterior environment over the domain contract: the Bayes filter, the update trace and the belief line |
| `estimator.py` | The lookahead $Q_d$, the teacher policy $\pi^*$, the answer accuracy $\hat\rho$, and LoRA |
| `trainer.py` | Algorithm 1: warm-up, own episodes, loss (12) over the agent's own thoughts, and the evaluation measures |
| `../models/sampler.py`, `../models/mm_server.py` | vLLM generation with the trainer's LoRA, for text prompts and for prompts with images |
| `../tasks/` | The datasets as registered tasks: their episodes, the agent's turn format and their teachers (`../tasks/<game>/teacher.py`) |

## Run

Runs start from [`plearn`](../plearn/README.md): the registered tasks, the evaluator and the training launcher.

```sh
python -m plearn train --config suite_qwen3vl_8b_thinking     # train_suite; its validation split evaluated after every phase
python -m plearn eval --model qwen3-vl-8b-thinking --adapter runs/suite_qwen3vl_8b_thinking/<run>/adapter_50 \
    --tasks test_suite --gpus 0
```

Settings of the reported runs: $d=2$, $\tau=0.02$, $\lambda=0.3$, $\kappa_0=4$, $\alpha_0=0.5$, $\gamma=0.93$, Qwen3.5-4B with LoRA rank 16.

### RLVB (GRPO): the group-relative update

`--update grpo` keeps everything of the critic (the sample sets and histories, so a history with no acceptable own turn is left
out exactly as RLVB (EM) leaves it out and both runs train on the same histories; the $K M$ own turns per history and their teacher
scores $R$; the two-block M-step) and changes only how
the scores become weights. In place of the E-step's $w_k=\mathrm{softmax}_k(R_k/\alpha)$ over the accepted turns, every own turn
of a history (its group) gets its group-relative advantage $A_k = R_k - \frac{1}{n_h-1}\sum_{j\ne k} R_j$ (leave-one-out), with
the scores raised to at least the group's best minus $C$ and $A_k\in[-C, C]$ (`--grpo_clip`, 20), and $A_k\le 0$ for a turn whose
action the teacher rejects (RLVB (EM)'s accept rule). The weight $w_k = s\,A_k/n_h$ has one factor $s$ per batch that gives every batch
the total absolute weight of the E-step's (`--grpo_norm 1`; with AdamW and the gradient-norm clip a constant factor would not
change the step, so the factor mainly evens out how much each batch counts). The loss
$-\sum_k w_k\log\pi_\phi(z_k,b_k,a_k\mid h)$ is the policy-gradient surrogate; no value model is learned. A phase's samples are
fitted about four times, so each block of an item stops by PPO's clip rule against the phase prior (`--grpo_eps`, 0.2), its
log-probability per token taken over the tokens the M-step counts. The sample sets do not depend on `--update`, so
`PL_SET_FROM` can start an RLVB (GRPO) run from an RLVB (EM) run's first set. The run file
`configs/train/grpo_posterior_qwen3vl_8b_thinking.yaml` is the RLVB (EM) 8B run file with these four flags
(`scripts/grpo_posterior_qwen3vl_8b_thinking.sh`; tests: `tests/method/test_grpo_posterior.py`). The 4B-VL and Qwen3.5-4B runs: `grpo_posterior_qwen3vl_4b_thinking.yaml` / `grpo_posterior_qwen35_4b.yaml` = `suite_qwen3vl_4b_thinking.yaml` /
`suite_qwen35_4b_presence_8k.yaml` with the same four flags (`scripts/grpo_posterior_qwen3vl_4b_thinking.sh`, `scripts/grpo_posterior_qwen35_4b.sh`).
The 3 RLVB (GRPO) run files also set `attn: flash_attention_3` (FlashAttention 3 for speed; the log-probabilities must not
change). The trainer's `--attn` (default `sdpa`, the attention of every RLVB (EM) run) loads the learner with that attention (`PL_ATTN`
overrides it, e.g. `flash_attention_2` on A100), and each learner rank checks it once, at its first `score()`: the first 4 pairs are
scored without gradient with the flash attention and again with sdpa (`attn_check`); the rank keeps the flash attention only when the
mean |difference| of the counted tokens' log-probabilities is at most 0.02 nats (`PL_ATTN_TOL`) and every value is finite, else it
uses sdpa for the rest of the run. The log row has `attn_flash` (1/0), `attn_dlogp_mean`, `attn_dlogp_max`; the run log has the line
`ATTN CHECK PASS|FAIL`. The RLVB (EM) run files set no `attn`, so the RLVB (EM) runs keep sdpa.

```sh
python -m plearn train --config grpo_posterior_qwen3vl_8b_thinking
```

`--tau_look`: the temperature of pi*'s Boltzmann among the looks only (default 0 = `--tau`). Among looks the values
differ by about one step of time cost (0.02 in log value, 0.2 nats at tau 0.1), while an answer before the evidence is enough
costs about 25 nats, so the score taught "do not answer early" far more strongly than "choose a good look". With `--tau_look 0.02`
one wasted step costs 1 nat. The answer-or-look decision (pC), the stop rule, the choice among the answers and `wrong_lp` keep
`--tau`. Mastermind keeps `--tau` (`Domain.look_scale = False`). `--tau_look_play 0` (default): the teacher's own moves (`tgt`) keep
`--tau`, so the histories are as in RLVB (GRPO); 1: they use `--tau_look` too. PL_SET_FROM refuses a saved set with other values
(the batches hold the scores). Logs: `look_gap`, `look_accept`, `tau_look_on`. Run file
`configs/train/looks_nobelief_qwen3vl_4b_thinking.yaml` = the ablation without the belief score plus `tau_look: 0.02`
(`scripts/ablation_scripts/looks_nobelief_qwen3vl_4b_thinking.sh`; tests: `tests/method/test_tau_look.py`).

### The reward repair (three options, all off by default)

Runs without the belief score drifted from about update 25: the agent kept taking harmless steps and stopped answering. Three
options repair the action reward, each from the posterior alone:

1. `--rho_exact 1`: the teacher values an answer by the exact posterior $U(b)$ for the whole run (the table of the agent's own
   answers keeps its prior).
2. `--teacher_row 1`: at a history where no own turn is accepted, the teacher's exact belief line and action join the group
   (`tests/method/test_teacher_row.py`).
3. `--stall_cap C`: a stalling step is rejected. A look that is neither one of the teacher's best looks nor one with real
   expected evidence gets $\log\pi^*\le$ best $-\,C$, and so does an action that is not legal. The evidence of a look is the
   expected drop of the answer's entropy over its outcomes (`look_info`: Mastermind, MineSweeper, MiniGrid, FrozenLake and
   Sokoban compute it from their own outcome model, Tabletop from `Belief.outcomes`). Before, such a step lost only its time
   cost (0.2 nats at $\tau=0.1$) and was accepted (the accept rule allows $\log 20\approx3.0$ nats); with $C=3.5$ it is rejected,
   so a group of stalling own turns gets the teacher's row. The rule only lowers a value: the answers, the best looks, the looks
   with evidence and the teacher's own moves are unchanged. In MineSweeper a cursor move counts only when it heads for the
   teacher's best cell; in MiniGrid turning on the spot is rejected once it shows nothing new. Log: `stall_rejected`, the share
   of own turns rejected by the cap alone (`tests/method/test_stall_cap.py`).
   `--stall_play 1`: the teacher's own moves come from the capped scores too. At $\tau=0.1$ its choice among moves that differ by
   one or two steps of time cost is soft, so without it the teacher itself takes a move the cap rejects at about a third of its
   steps in MineSweeper, MiniGrid, FrozenLake and Sokoban; with it 3 to 5% (offline check).

Run files: `configs/train/phase5_nobelief_trow_qwen35_4b.yaml` (parts 1 and 2) and `phase5_nobelief_trow_stall_qwen35_4b.yaml`
(all three, `stall_cap: 3.5`, `stall_play: 1`); with phases of 25 updates on the two Qwen3-VL models:
`pool_nobelief_trow_stall_qwen3vl_8b_thinking.yaml` and `pool_nobelief_trow_stall_qwen3vl_4b_thinking.yaml`.

No belief line (`--no_belief_line 1`, with `--line_score none`): the turn has no belief line at all. The prompt has the
action-only rules of the one-line turn (`plearn/tasks/domain.py` `action_only`) and no carried line; each own turn is the thought,
`</think>` and the action line (M actions after each thought), scored by the action's log pi* alone. A guard stops a run whose
prompt asks for or carries a belief line, and `PL_SET_FROM` refuses a sample set saved with another `--no_belief_line`. Run files:
`configs/train/pool_nobeliefline_qwen3vl_8b_thinking.yaml`, `pool_nobeliefline_qwen3vl_4b_thinking.yaml` (the pooled runs without the
belief score) and `phase5_nobeliefline_nolen_qwen35_4b.yaml` (phase 5, no length term, update 25): the RLVB (GRPO) run files of the tables plus `no_belief_line: 1`; they test with the one-line turn (tests: `tests/method/test_no_belief_line.py`).

The other ablation run files and their start scripts: `scripts/ablation_scripts/README.md`.

## Measures

| Measure | What it counts |
|---|---|
| `success` | Episodes with a correct answer |
| `premature` | Episodes with an answer before the evidence identifies the answer |
| `mean_t` | Evidence actions per episode |
| `late_looks` | Evidence actions per episode after the evidence identifies the answer |
| `anchor_exact` | Test histories in which the belief line is exactly the posterior |
| `em_q_budget` | Training rows: the E-step weight on correct turns (accepted action, line within `--em_more_joint` nats of the exact line) whose reasoning ends by itself within the test's 8,192 tokens; `em_q_correct` the same at any length (the budget measure: an offline proxy, not the success) |

## The two learners

RLVB (EM) and RLVB (GRPO) use this trainer, the same teacher scores, the same LoRA adapter and the same training tasks
(`train_suite`). Only the M-step weights are different: `--update em` (the default) or `--update grpo` with its settings. The RLVB
(GRPO) run files also set `attn: flash_attention_3`.

| Learner | Run files | Run scripts |
|---|---|---|
| RLVB (EM) | `configs/train/suite_<model>.yaml` (Qwen3.5-4B: `suite_qwen35_4b_presence_8k.yaml`) | `scripts/pl_<model>.sh` |
| RLVB (GRPO) | `configs/train/grpo_posterior_<model>.yaml` (`--update grpo`) | `scripts/grpo_posterior_<model>.sh` |

`configs/train/abl_pl_4b*.yaml` are the ablation runs of the parts of RLVB (EM) on Qwen3.5-4B.
