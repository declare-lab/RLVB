<div align="center">
<h1>RLVB: Reinforcement Learning with Verifiable Beliefs</h1>

<a href="https://declare-lab.github.io/RLVB/"><img src="assets/rlvb_cover.gif" alt="A lazy agent answers while two cups are still possible. The simulator computes the exact posterior, RLVB rewards each action from it, and after training the agent commits once the evidence identifies the answer." width="100%"></a>
</div>

<div align="center">
<h2>Lazy Agents Answer Against Their Own Beliefs: Reinforcement Learning with Verifiable Beliefs Teaches Them to Act on What They Believe</h2>
</div>
<div align="center">

[Richmond Sin Jing Xuan](https://richmondsin.vercel.app/)<sup>1*</sup>,
[Peilin Feng](https://peilin-ff.github.io/)<sup>1*</sup>,
[Wenbin Wang](https://wenbinwang12.github.io/)<sup>2</sup>,
[Soujanya Poria](https://soujanyaporia.github.io/)<sup>1</sup>

<sup>1</sup>[DeCLaRe Lab](https://declare-lab.github.io/), Nanyang Technological University &nbsp;&nbsp; <sup>2</sup>Independent &nbsp;&nbsp; <sup>*</sup>Equal contribution

[![Website](https://img.shields.io/badge/Website-RLVB-blue)](https://declare-lab.github.io/RLVB/)
![Paper](https://img.shields.io/badge/Paper-coming%20soon-lightgrey)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/richmondsin/rlvb)

</div>

This repository contains the official implementation and evaluation code of **Lazy Agents Answer Against Their Own Beliefs: Reinforcement Learning with Verifiable Beliefs Teaches Them to Act on What They Believe**.

Agents on long tasks often answer before they have enough evidence, even while their own belief still names several
answers. An end-of-episode reward cannot show which step was wrong. In many evidence-seeking tasks, the simulator knows the
rules that link the hidden answer to the evidence, so it can compute the exact posterior after every step. Reinforcement
Learning with Verifiable Beliefs (RLVB) uses this posterior to reward every action: a further step while several answers
remain, and an answer once the evidence identifies one. The agent writes its belief (the answers that it still thinks are
possible) before each action, and the update trains this line together with the action. After training, the agent acts on
its own belief. The project page is [declare-lab.github.io/RLVB](https://declare-lab.github.io/RLVB/).

## ✨ Highlights

- 🎯 **A process reward at every turn**: the simulator computes the exact posterior after each step, and RLVB rewards each
  turn with it. An end-of-episode reward cannot show which step was wrong.
- 🔍 **A verifiable belief**: the exact posterior is the belief that the evidence supports. The reward scores each action
  against a teacher that acts on this posterior. The agent also writes a belief line (the answers that are still
  possible), and it learns to act on this line.
- 🚫 **No judge, no critic, no reward model**: the reward comes from the simulator's rules. It never pays for a lucky guess.
- 🧠 **Two learners**: RLVB (EM) (expectation maximization) and RLVB (GRPO) use the same teacher scores with a different
  update.
- 🧩 **Three benchmarks**: RoboQuest, Pro-Worlds (visual puzzles: the agent sees a large world through a small window) and
  HiddenGames. No run trains on RoboQuest or Pro-Worlds.
- 📈 **Results of the paper**: on 3 vision-language models x 3 benchmarks (2 not seen in training), RLVB improves on the base
  model in all 9 settings: +17.1 points on average with GRPO as the learner, +10.4 points with EM. GRPO and PPO with the
  outcome reward lose accuracy on average. RLVB is also better than PPO with an LLM-judge process reward, with 1.6x fewer
  GPU-hours (GRPO) and 4x fewer GPU-hours (EM). On turns with insufficient evidence where the belief line still names
  several answers, Base (w/ belief) commits in 3 to 68 % of cases. After RLVB, the same models commit in 0.1 to 2.2 % of cases.

## 🧩 Benchmarks

| Benchmark | Tasks | What is hidden | Used for |
|---|---|---|---|
| HiddenGames (`train_suite`) | 8: the 7 HiddenGames test tasks and 1 training-only task (`plearn/tasks/suites/train_suite.yaml`) | see the HiddenGames row below | training (train split) |
| RoboQuest (`test_suite/roboquest`) | 10 robot tasks in a kitchen simulator, text version; 500 test games | what the agent needs: where the targets are, a label under a vessel, which parcel is odd, ... | test only |
| Pro-Worlds (`test_suite/proworlds`) | 15 visual games (engine v6.1), normal board; 300 test games | the member of a hidden world; the window shows one site of the board | test only |
| HiddenGames (`test_suite/hidden_games`) | 7: MineSweeper (easy), MiniGrid KeyCorridor, FrozenLake (hidden goal), Sokoban (hidden target), Tabletop lamps / cups / blocks; 756 test games | the mine cells; the room of the key; the goal cell and the holes that are not next to the agent; the target square; which button lights which lamp, which cup hides the object, which block stands on a slippery base | test (test split) |

`test_suite` is the three test rows together: 32 tasks, 1,556 test games. Thinking is capped at 8,192 tokens in every test
task. The Pro-Worlds long board (65 moves) is `test_suite_long`. The sources of each environment and our changes are in
[docs/environments.md](docs/environments.md).

## 🧠 Method

- **The turn**: the agent writes its reasoning, then two lines. The belief line (`Possible: <answers>`) gives the answers
  that are still possible. The action line (`ACTION: ...`) looks for more evidence or gives the answer.
- **The teacher**: each training task has an exact teacher. The teacher does the Bayes update of the hidden state after each
  observation. It gives the correct belief line and a teacher policy for the next action (look again or answer).
- **The reward**: the teacher scores the action of each turn against the teacher policy, which acts on the exact posterior.
  A variant also scores the belief line against the exact posterior (the belief term). The reported RLVB (GRPO) uses the
  action score alone.
- **The update**: the trainer samples the agent's own turns at each history. RLVB (EM) weights the turns by their scores
  (the E-step) and fits the weighted turns (the M-step). RLVB (GRPO) uses the same samples and scores. It gives each turn its
  group-relative advantage over the other turns of the same history, with a PPO clip against the phase prior.

The equations, the algorithm and the code map are in [plearn/method/README.md](plearn/method/README.md).

## 📊 Results

Success rate in percent on the test games. A cell shows the mean of the 3 test seeds (bold) and, below it, the seed values `s0 · s1 · s2`. **Overall** = all games of the benchmark in one seed together, then the mean over the seeds. Sampling: T 1.0, top-p 0.95, top-k 20, presence penalty 1.5, thought cap 8,192 tokens (`scripts/eval_<model>.sh`). Every trained model trained only on HiddenGames tasks (`train_suite`, train split); no model trained on RoboQuest or Pro-Worlds.

**RLVB (GRPO)** in these tables: no belief score (the turn score is the action term only). Qwen3-VL-4B-Thinking and Qwen3-VL-8B-Thinking: the pooled sample set (`pool_nobelief_<model>`, update 50). Qwen3.5-4B: phase 5, no length term (`phase5_nobelief_nolen_qwen35_4b`, update 25). Run files and scripts: [scripts/ablation_scripts/README.md](scripts/ablation_scripts/README.md).

| Method (column) | Reward | What is rewarded |
| --- | --- | --- |
| RLVB (EM) | **process** | Each turn gets the teacher's score R = a belief term + an action term, from the exact posterior at that history. No episode-success reward. |
| RLVB (GRPO) | **process** | Each turn gets the teacher's action score log pi* from the exact posterior at that history (these runs: no belief score), with group-relative (leave-one-out) advantages. |
| PPO (action-only) | **outcome** | The same episode reward as GRPO, through a learned critic (GAE). |
| GRPO (action-only) | **outcome** | The episode reward R = 0.999^t x 1[correct], given to every turn of the episode. |
| PPO (belief + action) | **both** | The outcome episode reward through the critic (GAE) for the other tokens, plus a process belief value from an LLM judge at the belief-state tokens (BRACE). |
| SFT (action-only) | neither | Supervised training on the teacher's target turns, no reward. |
| Base, Base (w/ belief) | - | The base model, no training. |

### 1. RoboQuest (no training on RoboQuest; tested on all 10 tasks)

#### Qwen3.5-4B

| Subtask | Base | Base (w/ belief) | SFT (action-only) | PPO (action-only) | GRPO (action-only) | PPO (belief + action) | RLVB (EM) | RLVB (GRPO) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Odd Parcel | **22.0**<br><sub>22.0 · 22.0 · 22.0</sub> | **24.0**<br><sub>22.0 · 20.0 · 30.0</sub> | **18.0**<br><sub>30.0 · 14.0 · 10.0</sub> | **8.0**<br><sub>8.0 · 8.0 · 8.0</sub> | **22.7**<br><sub>20.0 · 22.0 · 26.0</sub> | **44.0**<br><sub>44.0 · 40.0 · 48.0</sub> | **31.3**<br><sub>30.0 · 26.0 · 38.0</sub> | **39.3**<br><sub>38.0 · 36.0 · 44.0</sub> |
| Painted Cubes | **10.7**<br><sub>8.0 · 16.0 · 8.0</sub> | **18.7**<br><sub>22.0 · 14.0 · 20.0</sub> | **12.7**<br><sub>14.0 · 10.0 · 14.0</sub> | **2.0**<br><sub>2.0 · 0.0 · 4.0</sub> | **12.0**<br><sub>10.0 · 12.0 · 14.0</sub> | **42.0**<br><sub>34.0 · 48.0 · 44.0</sub> | **20.7**<br><sub>24.0 · 16.0 · 22.0</sub> | **38.0**<br><sub>42.0 · 30.0 · 42.0</sub> |
| Marked Mugs | **30.7**<br><sub>26.0 · 30.0 · 36.0</sub> | **36.0**<br><sub>34.0 · 34.0 · 40.0</sub> | **37.3**<br><sub>38.0 · 36.0 · 38.0</sub> | **25.3**<br><sub>22.0 · 30.0 · 24.0</sub> | **34.7**<br><sub>36.0 · 30.0 · 38.0</sub> | **58.7**<br><sub>54.0 · 64.0 · 58.0</sub> | **40.7**<br><sub>46.0 · 38.0 · 38.0</sub> | **47.3**<br><sub>48.0 · 48.0 · 46.0</sub> |
| Unfamiliar Containers | **8.7**<br><sub>10.0 · 6.0 · 10.0</sub> | **9.3**<br><sub>12.0 · 2.0 · 14.0</sub> | **5.3**<br><sub>10.0 · 4.0 · 2.0</sub> | **0.0**<br><sub>0.0 · 0.0 · 0.0</sub> | **4.7**<br><sub>8.0 · 2.0 · 4.0</sub> | **16.7**<br><sub>10.0 · 16.0 · 24.0</sub> | **17.3**<br><sub>16.0 · 20.0 · 16.0</sub> | **28.7**<br><sub>26.0 · 30.0 · 30.0</sub> |
| Search Room | **60.0**<br><sub>58.0 · 68.0 · 54.0</sub> | **64.7**<br><sub>62.0 · 62.0 · 70.0</sub> | **70.7**<br><sub>68.0 · 72.0 · 72.0</sub> | **4.0**<br><sub>4.0 · 4.0 · 4.0</sub> | **62.0**<br><sub>62.0 · 56.0 · 68.0</sub> | **94.7**<br><sub>96.0 · 96.0 · 92.0</sub> | **74.7**<br><sub>76.0 · 76.0 · 72.0</sub> | **80.7**<br><sub>84.0 · 80.0 · 78.0</sub> |
| Blackout Search | **50.7**<br><sub>46.0 · 42.0 · 64.0</sub> | **54.0**<br><sub>48.0 · 58.0 · 56.0</sub> | **54.7**<br><sub>48.0 · 56.0 · 60.0</sub> | **2.0**<br><sub>2.0 · 2.0 · 2.0</sub> | **47.3**<br><sub>48.0 · 54.0 · 40.0</sub> | **91.3**<br><sub>94.0 · 90.0 · 90.0</sub> | **68.0**<br><sub>72.0 · 64.0 · 68.0</sub> | **78.0**<br><sub>74.0 · 82.0 · 78.0</sub> |
| Locked Storage | **69.3**<br><sub>72.0 · 66.0 · 70.0</sub> | **66.7**<br><sub>66.0 · 72.0 · 62.0</sub> | **61.3**<br><sub>70.0 · 50.0 · 64.0</sub> | **14.7**<br><sub>14.0 · 12.0 · 18.0</sub> | **62.0**<br><sub>62.0 · 62.0 · 62.0</sub> | **75.3**<br><sub>72.0 · 82.0 · 72.0</sub> | **77.3**<br><sub>80.0 · 74.0 · 78.0</sub> | **81.3**<br><sub>78.0 · 76.0 · 90.0</sub> |
| Puzzle Box | **44.7**<br><sub>50.0 · 42.0 · 42.0</sub> | **38.7**<br><sub>44.0 · 34.0 · 38.0</sub> | **32.7**<br><sub>34.0 · 32.0 · 32.0</sub> | **6.0**<br><sub>10.0 · 6.0 · 2.0</sub> | **31.3**<br><sub>28.0 · 34.0 · 32.0</sub> | **43.3**<br><sub>44.0 · 40.0 · 46.0</sub> | **38.0**<br><sub>48.0 · 36.0 · 30.0</sub> | **41.3**<br><sub>34.0 · 46.0 · 44.0</sub> |
| Stamps | **4.0**<br><sub>2.0 · 6.0 · 4.0</sub> | **6.0**<br><sub>4.0 · 6.0 · 8.0</sub> | **0.0**<br><sub>0.0 · 0.0 · 0.0</sub> | **0.0**<br><sub>0.0 · 0.0 · 0.0</sub> | **0.0**<br><sub>0.0 · 0.0 · 0.0</sub> | **8.0**<br><sub>6.0 · 10.0 · 8.0</sub> | **7.3**<br><sub>8.0 · 8.0 · 6.0</sub> | **6.7**<br><sub>6.0 · 8.0 · 6.0</sub> |
| Wobbly Stand | **95.3**<br><sub>96.0 · 96.0 · 94.0</sub> | **94.7**<br><sub>96.0 · 92.0 · 96.0</sub> | **90.7**<br><sub>90.0 · 90.0 · 92.0</sub> | **64.0**<br><sub>66.0 · 54.0 · 72.0</sub> | **92.7**<br><sub>96.0 · 90.0 · 92.0</sub> | **96.7**<br><sub>100.0 · 98.0 · 92.0</sub> | **97.3**<br><sub>94.0 · 100.0 · 98.0</sub> | **98.7**<br><sub>96.0 · 100.0 · 100.0</sub> |
| **Overall** | **39.6**<br><sub>39.0 · 39.4 · 40.4</sub> | **41.3**<br><sub>41.0 · 39.4 · 43.4</sub> | **38.3**<br><sub>40.2 · 36.4 · 38.4</sub> | **12.6**<br><sub>12.8 · 11.6 · 13.4</sub> | **36.9**<br><sub>37.0 · 36.2 · 37.6</sub> | **57.1**<br><sub>55.4 · 58.4 · 57.4</sub> | **47.3**<br><sub>49.4 · 45.8 · 46.6</sub> | **54.0**<br><sub>52.6 · 53.6 · 55.8</sub> |

#### Qwen3-VL-4B-Thinking

| Subtask | Base | Base (w/ belief) | SFT (action-only) | PPO (action-only) | GRPO (action-only) | PPO (belief + action) | RLVB (EM) | RLVB (GRPO) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Odd Parcel | **12.0**<br><sub>12.0 · 18.0 · 6.0</sub> | **23.3**<br><sub>30.0 · 28.0 · 12.0</sub> | **28.0**<br><sub>30.0 · 26.0 · 28.0</sub> | **2.7**<br><sub>4.0 · 0.0 · 4.0</sub> | **12.7**<br><sub>12.0 · 10.0 · 16.0</sub> | **15.3**<br><sub>10.0 · 14.0 · 22.0</sub> | **36.7**<br><sub>44.0 · 32.0 · 34.0</sub> | **72.7**<br><sub>66.0 · 80.0 · 72.0</sub> |
| Painted Cubes | **4.0**<br><sub>4.0 · 6.0 · 2.0</sub> | **6.7**<br><sub>4.0 · 6.0 · 10.0</sub> | **12.0**<br><sub>12.0 · 16.0 · 8.0</sub> | **0.0**<br><sub>0.0 · 0.0 · 0.0</sub> | **4.0**<br><sub>6.0 · 2.0 · 4.0</sub> | **10.0**<br><sub>10.0 · 8.0 · 12.0</sub> | **10.7**<br><sub>14.0 · 10.0 · 8.0</sub> | **45.3**<br><sub>48.0 · 44.0 · 44.0</sub> |
| Marked Mugs | **20.0**<br><sub>20.0 · 18.0 · 22.0</sub> | **24.7**<br><sub>22.0 · 28.0 · 24.0</sub> | **38.7**<br><sub>42.0 · 38.0 · 36.0</sub> | **4.0**<br><sub>6.0 · 2.0 · 4.0</sub> | **17.3**<br><sub>18.0 · 16.0 · 18.0</sub> | **8.7**<br><sub>10.0 · 6.0 · 10.0</sub> | **40.7**<br><sub>44.0 · 38.0 · 40.0</sub> | **53.3**<br><sub>54.0 · 52.0 · 54.0</sub> |
| Unfamiliar Containers | **0.0**<br><sub>0.0 · 0.0 · 0.0</sub> | **0.7**<br><sub>2.0 · 0.0 · 0.0</sub> | **1.3**<br><sub>0.0 · 2.0 · 2.0</sub> | **0.0**<br><sub>0.0 · 0.0 · 0.0</sub> | **0.0**<br><sub>0.0 · 0.0 · 0.0</sub> | **1.3**<br><sub>0.0 · 2.0 · 2.0</sub> | **5.3**<br><sub>8.0 · 2.0 · 6.0</sub> | **40.0**<br><sub>40.0 · 40.0 · 40.0</sub> |
| Search Room | **38.0**<br><sub>40.0 · 42.0 · 32.0</sub> | **56.7**<br><sub>54.0 · 56.0 · 60.0</sub> | **51.3**<br><sub>56.0 · 54.0 · 44.0</sub> | **14.0**<br><sub>14.0 · 18.0 · 10.0</sub> | **30.7**<br><sub>36.0 · 28.0 · 28.0</sub> | **75.3**<br><sub>78.0 · 82.0 · 66.0</sub> | **76.0**<br><sub>80.0 · 72.0 · 76.0</sub> | **89.3**<br><sub>86.0 · 96.0 · 86.0</sub> |
| Blackout Search | **11.3**<br><sub>10.0 · 16.0 · 8.0</sub> | **28.0**<br><sub>28.0 · 24.0 · 32.0</sub> | **30.0**<br><sub>24.0 · 32.0 · 34.0</sub> | **1.3**<br><sub>0.0 · 4.0 · 0.0</sub> | **7.3**<br><sub>6.0 · 4.0 · 12.0</sub> | **64.7**<br><sub>66.0 · 64.0 · 64.0</sub> | **49.3**<br><sub>58.0 · 48.0 · 42.0</sub> | **82.7**<br><sub>86.0 · 80.0 · 82.0</sub> |
| Locked Storage | **69.3**<br><sub>66.0 · 70.0 · 72.0</sub> | **68.0**<br><sub>64.0 · 74.0 · 66.0</sub> | **68.7**<br><sub>64.0 · 72.0 · 70.0</sub> | **54.0**<br><sub>58.0 · 48.0 · 56.0</sub> | **65.3**<br><sub>64.0 · 64.0 · 68.0</sub> | **81.3**<br><sub>80.0 · 84.0 · 80.0</sub> | **82.7**<br><sub>84.0 · 78.0 · 86.0</sub> | **83.3**<br><sub>82.0 · 90.0 · 78.0</sub> |
| Puzzle Box | **28.7**<br><sub>30.0 · 28.0 · 28.0</sub> | **34.0**<br><sub>32.0 · 32.0 · 38.0</sub> | **19.3**<br><sub>14.0 · 24.0 · 20.0</sub> | **33.3**<br><sub>32.0 · 26.0 · 42.0</sub> | **12.7**<br><sub>20.0 · 10.0 · 8.0</sub> | **33.3**<br><sub>40.0 · 30.0 · 30.0</sub> | **24.7**<br><sub>24.0 · 30.0 · 20.0</sub> | **43.3**<br><sub>40.0 · 44.0 · 46.0</sub> |
| Stamps | **0.7**<br><sub>0.0 · 0.0 · 2.0</sub> | **0.7**<br><sub>0.0 · 2.0 · 0.0</sub> | **0.7**<br><sub>0.0 · 2.0 · 0.0</sub> | **0.7**<br><sub>0.0 · 2.0 · 0.0</sub> | **0.0**<br><sub>0.0 · 0.0 · 0.0</sub> | **1.3**<br><sub>0.0 · 0.0 · 4.0</sub> | **0.7**<br><sub>0.0 · 0.0 · 2.0</sub> | **2.0**<br><sub>2.0 · 4.0 · 0.0</sub> |
| Wobbly Stand | **100.0**<br><sub>100.0 · 100.0 · 100.0</sub> | **97.3**<br><sub>98.0 · 100.0 · 94.0</sub> | **94.7**<br><sub>92.0 · 96.0 · 96.0</sub> | **98.7**<br><sub>98.0 · 98.0 · 100.0</sub> | **93.3**<br><sub>90.0 · 94.0 · 96.0</sub> | **96.0**<br><sub>96.0 · 96.0 · 96.0</sub> | **99.3**<br><sub>100.0 · 98.0 · 100.0</sub> | **100.0**<br><sub>100.0 · 100.0 · 100.0</sub> |
| **Overall** | **28.4**<br><sub>28.2 · 29.8 · 27.2</sub> | **34.0**<br><sub>33.4 · 35.0 · 33.6</sub> | **34.5**<br><sub>33.4 · 36.2 · 33.8</sub> | **20.9**<br><sub>21.2 · 19.8 · 21.6</sub> | **24.3**<br><sub>25.2 · 22.8 · 25.0</sub> | **38.7**<br><sub>39.0 · 38.6 · 38.6</sub> | **42.6**<br><sub>45.6 · 40.8 · 41.4</sub> | **61.2**<br><sub>60.4 · 63.0 · 60.2</sub> |

#### Qwen3-VL-8B-Thinking

| Subtask | Base | Base (w/ belief) | SFT (action-only) | PPO (action-only) | GRPO (action-only) | PPO (belief + action) | RLVB (EM) | RLVB (GRPO) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Odd Parcel | **39.3**<br><sub>42.0 · 38.0 · 38.0</sub> | **43.3**<br><sub>40.0 · 40.0 · 50.0</sub> | **38.7**<br><sub>34.0 · 44.0 · 38.0</sub> | **42.7**<br><sub>46.0 · 42.0 · 40.0</sub> | **31.3**<br><sub>32.0 · 32.0 · 30.0</sub> | **46.0**<br><sub>42.0 · 48.0 · 48.0</sub> | **50.0**<br><sub>44.0 · 52.0 · 54.0</sub> | **70.0**<br><sub>64.0 · 70.0 · 76.0</sub> |
| Painted Cubes | **9.3**<br><sub>12.0 · 8.0 · 8.0</sub> | **14.0**<br><sub>12.0 · 14.0 · 16.0</sub> | **17.3**<br><sub>16.0 · 18.0 · 18.0</sub> | **13.3**<br><sub>16.0 · 10.0 · 14.0</sub> | **14.0**<br><sub>14.0 · 18.0 · 10.0</sub> | **30.7**<br><sub>38.0 · 26.0 · 28.0</sub> | **25.3**<br><sub>20.0 · 28.0 · 28.0</sub> | **53.3**<br><sub>54.0 · 54.0 · 52.0</sub> |
| Marked Mugs | **28.0**<br><sub>24.0 · 30.0 · 30.0</sub> | **27.3**<br><sub>26.0 · 28.0 · 28.0</sub> | **40.0**<br><sub>44.0 · 38.0 · 38.0</sub> | **29.3**<br><sub>28.0 · 30.0 · 30.0</sub> | **25.3**<br><sub>24.0 · 24.0 · 28.0</sub> | **48.0**<br><sub>44.0 · 44.0 · 56.0</sub> | **45.3**<br><sub>48.0 · 46.0 · 42.0</sub> | **54.7**<br><sub>52.0 · 56.0 · 56.0</sub> |
| Unfamiliar Containers | **4.7**<br><sub>4.0 · 4.0 · 6.0</sub> | **3.3**<br><sub>10.0 · 0.0 · 0.0</sub> | **6.7**<br><sub>12.0 · 4.0 · 4.0</sub> | **10.0**<br><sub>8.0 · 12.0 · 10.0</sub> | **2.7**<br><sub>0.0 · 4.0 · 4.0</sub> | **20.7**<br><sub>18.0 · 16.0 · 28.0</sub> | **6.7**<br><sub>6.0 · 12.0 · 2.0</sub> | **40.0**<br><sub>42.0 · 42.0 · 36.0</sub> |
| Search Room | **40.7**<br><sub>38.0 · 44.0 · 40.0</sub> | **51.3**<br><sub>50.0 · 52.0 · 52.0</sub> | **52.7**<br><sub>56.0 · 48.0 · 54.0</sub> | **45.3**<br><sub>44.0 · 40.0 · 52.0</sub> | **46.0**<br><sub>44.0 · 52.0 · 42.0</sub> | **83.3**<br><sub>82.0 · 86.0 · 82.0</sub> | **74.7**<br><sub>70.0 · 82.0 · 72.0</sub> | **87.3**<br><sub>88.0 · 90.0 · 84.0</sub> |
| Blackout Search | **20.7**<br><sub>12.0 · 22.0 · 28.0</sub> | **43.3**<br><sub>48.0 · 46.0 · 36.0</sub> | **48.7**<br><sub>42.0 · 54.0 · 50.0</sub> | **39.3**<br><sub>50.0 · 32.0 · 36.0</sub> | **17.3**<br><sub>22.0 · 12.0 · 18.0</sub> | **76.7**<br><sub>78.0 · 82.0 · 70.0</sub> | **59.3**<br><sub>54.0 · 64.0 · 60.0</sub> | **88.7**<br><sub>94.0 · 88.0 · 84.0</sub> |
| Locked Storage | **86.7**<br><sub>92.0 · 84.0 · 84.0</sub> | **80.0**<br><sub>84.0 · 82.0 · 74.0</sub> | **84.7**<br><sub>86.0 · 78.0 · 90.0</sub> | **87.3**<br><sub>88.0 · 82.0 · 92.0</sub> | **79.3**<br><sub>80.0 · 82.0 · 76.0</sub> | **79.3**<br><sub>80.0 · 76.0 · 82.0</sub> | **83.3**<br><sub>80.0 · 78.0 · 92.0</sub> | **86.0**<br><sub>78.0 · 88.0 · 92.0</sub> |
| Puzzle Box | **50.0**<br><sub>44.0 · 52.0 · 54.0</sub> | **51.3**<br><sub>50.0 · 52.0 · 52.0</sub> | **48.0**<br><sub>48.0 · 52.0 · 44.0</sub> | **41.3**<br><sub>38.0 · 42.0 · 44.0</sub> | **50.0**<br><sub>54.0 · 46.0 · 50.0</sub> | **40.7**<br><sub>36.0 · 38.0 · 48.0</sub> | **40.7**<br><sub>36.0 · 38.0 · 48.0</sub> | **55.3**<br><sub>52.0 · 58.0 · 56.0</sub> |
| Stamps | **0.7**<br><sub>0.0 · 2.0 · 0.0</sub> | **0.7**<br><sub>0.0 · 2.0 · 0.0</sub> | **0.7**<br><sub>0.0 · 2.0 · 0.0</sub> | **0.7**<br><sub>0.0 · 2.0 · 0.0</sub> | **1.3**<br><sub>2.0 · 0.0 · 2.0</sub> | **1.3**<br><sub>2.0 · 0.0 · 2.0</sub> | **0.0**<br><sub>0.0 · 0.0 · 0.0</sub> | **0.7**<br><sub>0.0 · 0.0 · 2.0</sub> |
| Wobbly Stand | **82.0**<br><sub>80.0 · 82.0 · 84.0</sub> | **83.3**<br><sub>86.0 · 78.0 · 86.0</sub> | **87.3**<br><sub>92.0 · 90.0 · 80.0</sub> | **82.7**<br><sub>86.0 · 84.0 · 78.0</sub> | **80.0**<br><sub>86.0 · 84.0 · 70.0</sub> | **87.3**<br><sub>94.0 · 84.0 · 84.0</sub> | **86.7**<br><sub>78.0 · 90.0 · 92.0</sub> | **91.3**<br><sub>90.0 · 92.0 · 92.0</sub> |
| **Overall** | **36.2**<br><sub>34.8 · 36.6 · 37.2</sub> | **39.8**<br><sub>40.6 · 39.4 · 39.4</sub> | **42.5**<br><sub>43.0 · 42.8 · 41.6</sub> | **39.2**<br><sub>40.4 · 37.6 · 39.6</sub> | **34.7**<br><sub>35.8 · 35.4 · 33.0</sub> | **51.4**<br><sub>51.4 · 50.0 · 52.8</sub> | **47.2**<br><sub>43.6 · 49.0 · 49.0</sub> | **62.7**<br><sub>61.4 · 63.8 · 63.0</sub> |

PPO (belief + action) = adapter-50.

### 2. Pro-Worlds (normal board; no training on Pro-Worlds; tested on all 15 games)

#### Qwen3.5-4B

| Subtask | Base | Base (w/ belief) | SFT (action-only) | PPO (action-only) | GRPO (action-only) | PPO (belief + action) | RLVB (EM) | RLVB (GRPO) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Face | **15.0**<br><sub>20.0 · 15.0 · 10.0</sub> | **18.3**<br><sub>20.0 · 20.0 · 15.0</sub> | **10.0**<br><sub>15.0 · 10.0 · 5.0</sub> | **8.3**<br><sub>5.0 · 10.0 · 10.0</sub> | **25.0**<br><sub>15.0 · 20.0 · 40.0</sub> | **20.0**<br><sub>15.0 · 25.0 · 20.0</sub> | **15.0**<br><sub>15.0 · 10.0 · 20.0</sub> | **25.0**<br><sub>30.0 · 20.0 · 25.0</sub> |
| Fish | **16.7**<br><sub>15.0 · 20.0 · 15.0</sub> | **20.0**<br><sub>15.0 · 25.0 · 20.0</sub> | **8.3**<br><sub>5.0 · 15.0 · 5.0</sub> | **1.7**<br><sub>0.0 · 0.0 · 5.0</sub> | **11.7**<br><sub>20.0 · 5.0 · 10.0</sub> | **23.3**<br><sub>30.0 · 25.0 · 15.0</sub> | **18.3**<br><sub>20.0 · 10.0 · 25.0</sub> | **23.3**<br><sub>20.0 · 25.0 · 25.0</sub> |
| Turtle | **31.7**<br><sub>35.0 · 25.0 · 35.0</sub> | **28.3**<br><sub>25.0 · 30.0 · 30.0</sub> | **5.0**<br><sub>5.0 · 5.0 · 5.0</sub> | **16.7**<br><sub>20.0 · 10.0 · 20.0</sub> | **6.7**<br><sub>10.0 · 0.0 · 10.0</sub> | **31.7**<br><sub>35.0 · 35.0 · 25.0</sub> | **35.0**<br><sub>45.0 · 30.0 · 30.0</sub> | **30.0**<br><sub>35.0 · 20.0 · 35.0</sub> |
| Balloon | **31.7**<br><sub>20.0 · 45.0 · 30.0</sub> | **33.3**<br><sub>40.0 · 25.0 · 35.0</sub> | **11.7**<br><sub>10.0 · 5.0 · 20.0</sub> | **5.0**<br><sub>0.0 · 0.0 · 15.0</sub> | **10.0**<br><sub>15.0 · 15.0 · 0.0</sub> | **51.7**<br><sub>55.0 · 50.0 · 50.0</sub> | **48.3**<br><sub>50.0 · 40.0 · 55.0</sub> | **43.3**<br><sub>30.0 · 55.0 · 45.0</sub> |
| Rocket | **10.0**<br><sub>15.0 · 5.0 · 10.0</sub> | **16.7**<br><sub>10.0 · 20.0 · 20.0</sub> | **13.3**<br><sub>5.0 · 15.0 · 20.0</sub> | **16.7**<br><sub>15.0 · 15.0 · 20.0</sub> | **13.3**<br><sub>10.0 · 20.0 · 10.0</sub> | **18.3**<br><sub>15.0 · 20.0 · 20.0</sub> | **21.7**<br><sub>15.0 · 20.0 · 30.0</sub> | **15.0**<br><sub>15.0 · 10.0 · 20.0</sub> |
| Sailboat | **30.0**<br><sub>35.0 · 25.0 · 30.0</sub> | **43.3**<br><sub>30.0 · 65.0 · 35.0</sub> | **10.0**<br><sub>15.0 · 10.0 · 5.0</sub> | **6.7**<br><sub>10.0 · 5.0 · 5.0</sub> | **18.3**<br><sub>20.0 · 15.0 · 20.0</sub> | **36.7**<br><sub>40.0 · 30.0 · 40.0</sub> | **43.3**<br><sub>40.0 · 55.0 · 35.0</sub> | **40.0**<br><sub>40.0 · 40.0 · 40.0</sub> |
| Barn | **10.0**<br><sub>10.0 · 15.0 · 5.0</sub> | **21.7**<br><sub>25.0 · 20.0 · 20.0</sub> | **16.7**<br><sub>30.0 · 5.0 · 15.0</sub> | **8.3**<br><sub>10.0 · 5.0 · 10.0</sub> | **13.3**<br><sub>10.0 · 10.0 · 20.0</sub> | **35.0**<br><sub>35.0 · 35.0 · 35.0</sub> | **30.0**<br><sub>35.0 · 40.0 · 15.0</sub> | **48.3**<br><sub>30.0 · 65.0 · 50.0</sub> |
| Lantern | **26.7**<br><sub>25.0 · 35.0 · 20.0</sub> | **21.7**<br><sub>25.0 · 20.0 · 20.0</sub> | **13.3**<br><sub>5.0 · 20.0 · 15.0</sub> | **13.3**<br><sub>10.0 · 0.0 · 30.0</sub> | **13.3**<br><sub>10.0 · 20.0 · 10.0</sub> | **21.7**<br><sub>15.0 · 20.0 · 30.0</sub> | **13.3**<br><sub>10.0 · 15.0 · 15.0</sub> | **21.7**<br><sub>30.0 · 20.0 · 15.0</sub> |
| **Seen (8 games)** | **21.5**<br><sub>21.9 · 23.1 · 19.4</sub> | **25.4**<br><sub>23.8 · 28.1 · 24.4</sub> | **11.1**<br><sub>11.3 · 10.6 · 11.3</sub> | **9.6**<br><sub>8.8 · 5.6 · 14.4</sub> | **14.0**<br><sub>13.8 · 13.1 · 15.0</sub> | **29.8**<br><sub>30.0 · 30.0 · 29.4</sub> | **28.1**<br><sub>28.8 · 27.5 · 28.1</sub> | **30.9**<br><sub>28.8 · 31.9 · 31.9</sub> |
| Jigsaw | **5.0**<br><sub>10.0 · 0.0 · 5.0</sub> | **8.3**<br><sub>5.0 · 10.0 · 10.0</sub> | **6.7**<br><sub>5.0 · 0.0 · 15.0</sub> | **5.0**<br><sub>5.0 · 0.0 · 10.0</sub> | **6.7**<br><sub>5.0 · 5.0 · 10.0</sub> | **6.7**<br><sub>5.0 · 0.0 · 15.0</sub> | **10.0**<br><sub>10.0 · 15.0 · 5.0</sub> | **23.3**<br><sub>25.0 · 30.0 · 15.0</sub> |
| Kaleidoscope | **15.0**<br><sub>20.0 · 5.0 · 20.0</sub> | **21.7**<br><sub>20.0 · 25.0 · 20.0</sub> | **20.0**<br><sub>10.0 · 15.0 · 35.0</sub> | **25.0**<br><sub>35.0 · 20.0 · 20.0</sub> | **25.0**<br><sub>30.0 · 25.0 · 20.0</sub> | **25.0**<br><sub>30.0 · 20.0 · 25.0</sub> | **25.0**<br><sub>15.0 · 30.0 · 30.0</sub> | **30.0**<br><sub>50.0 · 20.0 · 20.0</sub> |
| Bird | **15.0**<br><sub>25.0 · 15.0 · 5.0</sub> | **8.3**<br><sub>15.0 · 5.0 · 5.0</sub> | **13.3**<br><sub>20.0 · 20.0 · 0.0</sub> | **10.0**<br><sub>10.0 · 5.0 · 15.0</sub> | **10.0**<br><sub>10.0 · 10.0 · 10.0</sub> | **16.7**<br><sub>15.0 · 25.0 · 10.0</sub> | **18.3**<br><sub>20.0 · 25.0 · 10.0</sub> | **8.3**<br><sub>15.0 · 10.0 · 0.0</sub> |
| Recipe | **13.3**<br><sub>10.0 · 15.0 · 15.0</sub> | **25.0**<br><sub>20.0 · 25.0 · 30.0</sub> | **18.3**<br><sub>15.0 · 35.0 · 5.0</sub> | **15.0**<br><sub>20.0 · 10.0 · 15.0</sub> | **13.3**<br><sub>5.0 · 10.0 · 25.0</sub> | **26.7**<br><sub>25.0 · 30.0 · 25.0</sub> | **33.3**<br><sub>35.0 · 30.0 · 35.0</sub> | **35.0**<br><sub>25.0 · 30.0 · 50.0</sub> |
| Bell | **30.0**<br><sub>30.0 · 30.0 · 30.0</sub> | **35.0**<br><sub>30.0 · 35.0 · 40.0</sub> | **16.7**<br><sub>20.0 · 10.0 · 20.0</sub> | **15.0**<br><sub>20.0 · 5.0 · 20.0</sub> | **25.0**<br><sub>20.0 · 25.0 · 30.0</sub> | **48.3**<br><sub>45.0 · 55.0 · 45.0</sub> | **50.0**<br><sub>50.0 · 60.0 · 40.0</sub> | **50.0**<br><sub>55.0 · 50.0 · 45.0</sub> |
| Bee | **31.7**<br><sub>45.0 · 25.0 · 25.0</sub> | **31.7**<br><sub>40.0 · 20.0 · 35.0</sub> | **28.3**<br><sub>35.0 · 25.0 · 25.0</sub> | **16.7**<br><sub>15.0 · 25.0 · 10.0</sub> | **21.7**<br><sub>20.0 · 15.0 · 30.0</sub> | **48.3**<br><sub>50.0 · 45.0 · 50.0</sub> | **36.7**<br><sub>50.0 · 30.0 · 30.0</sub> | **46.7**<br><sub>45.0 · 40.0 · 55.0</sub> |
| Candle | **21.7**<br><sub>20.0 · 25.0 · 20.0</sub> | **25.0**<br><sub>25.0 · 15.0 · 35.0</sub> | **11.7**<br><sub>10.0 · 5.0 · 20.0</sub> | **10.0**<br><sub>5.0 · 5.0 · 20.0</sub> | **13.3**<br><sub>10.0 · 10.0 · 20.0</sub> | **26.7**<br><sub>30.0 · 25.0 · 25.0</sub> | **23.3**<br><sub>25.0 · 25.0 · 20.0</sub> | **41.7**<br><sub>45.0 · 35.0 · 45.0</sub> |
| **Unseen (7 games)** | **18.8**<br><sub>22.9 · 16.4 · 17.1</sub> | **22.1**<br><sub>22.1 · 19.3 · 25.0</sub> | **16.4**<br><sub>16.4 · 15.7 · 17.1</sub> | **13.8**<br><sub>15.7 · 10.0 · 15.7</sub> | **16.4**<br><sub>14.3 · 14.3 · 20.7</sub> | **28.4**<br><sub>28.6 · 28.6 · 27.9</sub> | **28.1**<br><sub>29.3 · 30.7 · 24.3</sub> | **33.6**<br><sub>37.1 · 30.7 · 32.9</sub> |
| **Overall** | **20.2**<br><sub>22.3 · 20.0 · 18.3</sub> | **23.9**<br><sub>23.0 · 24.0 · 24.7</sub> | **13.6**<br><sub>13.7 · 13.0 · 14.0</sub> | **11.6**<br><sub>12.0 · 7.7 · 15.0</sub> | **15.1**<br><sub>14.0 · 13.7 · 17.7</sub> | **29.1**<br><sub>29.3 · 29.3 · 28.7</sub> | **28.1**<br><sub>29.0 · 29.0 · 26.3</sub> | **32.1**<br><sub>32.7 · 31.3 · 32.3</sub> |

#### Qwen3-VL-4B-Thinking

| Subtask | Base | Base (w/ belief) | SFT (action-only) | PPO (action-only) | GRPO (action-only) | PPO (belief + action) | RLVB (EM) | RLVB (GRPO) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Face | **11.7**<br><sub>10.0 · 15.0 · 10.0</sub> | **20.0**<br><sub>20.0 · 20.0 · 20.0</sub> | **10.0**<br><sub>15.0 · 5.0 · 10.0</sub> | **10.0**<br><sub>5.0 · 10.0 · 15.0</sub> | **13.3**<br><sub>20.0 · 10.0 · 10.0</sub> | **13.3**<br><sub>15.0 · 10.0 · 15.0</sub> | **11.7**<br><sub>15.0 · 10.0 · 10.0</sub> | **16.7**<br><sub>15.0 · 10.0 · 25.0</sub> |
| Fish | **3.3**<br><sub>0.0 · 5.0 · 5.0</sub> | **11.7**<br><sub>15.0 · 10.0 · 10.0</sub> | **5.0**<br><sub>0.0 · 0.0 · 15.0</sub> | **0.0**<br><sub>0.0 · 0.0 · 0.0</sub> | **8.3**<br><sub>10.0 · 5.0 · 10.0</sub> | **3.3**<br><sub>5.0 · 5.0 · 0.0</sub> | **16.7**<br><sub>25.0 · 20.0 · 5.0</sub> | **5.0**<br><sub>5.0 · 5.0 · 5.0</sub> |
| Turtle | **21.7**<br><sub>20.0 · 25.0 · 20.0</sub> | **15.0**<br><sub>5.0 · 25.0 · 15.0</sub> | **31.7**<br><sub>30.0 · 30.0 · 35.0</sub> | **20.0**<br><sub>15.0 · 30.0 · 15.0</sub> | **18.3**<br><sub>10.0 · 15.0 · 30.0</sub> | **13.3**<br><sub>10.0 · 5.0 · 25.0</sub> | **33.3**<br><sub>35.0 · 35.0 · 30.0</sub> | **40.0**<br><sub>25.0 · 40.0 · 55.0</sub> |
| Balloon | **21.7**<br><sub>25.0 · 20.0 · 20.0</sub> | **31.7**<br><sub>45.0 · 25.0 · 25.0</sub> | **36.7**<br><sub>40.0 · 35.0 · 35.0</sub> | **16.7**<br><sub>10.0 · 25.0 · 15.0</sub> | **21.7**<br><sub>20.0 · 15.0 · 30.0</sub> | **8.3**<br><sub>5.0 · 15.0 · 5.0</sub> | **38.3**<br><sub>35.0 · 45.0 · 35.0</sub> | **40.0**<br><sub>55.0 · 45.0 · 20.0</sub> |
| Rocket | **28.3**<br><sub>35.0 · 25.0 · 25.0</sub> | **18.3**<br><sub>20.0 · 20.0 · 15.0</sub> | **21.7**<br><sub>25.0 · 25.0 · 15.0</sub> | **30.0**<br><sub>30.0 · 25.0 · 35.0</sub> | **26.7**<br><sub>30.0 · 30.0 · 20.0</sub> | **21.7**<br><sub>35.0 · 10.0 · 20.0</sub> | **35.0**<br><sub>35.0 · 30.0 · 40.0</sub> | **40.0**<br><sub>30.0 · 45.0 · 45.0</sub> |
| Sailboat | **21.7**<br><sub>25.0 · 25.0 · 15.0</sub> | **21.7**<br><sub>25.0 · 25.0 · 15.0</sub> | **23.3**<br><sub>30.0 · 25.0 · 15.0</sub> | **13.3**<br><sub>10.0 · 15.0 · 15.0</sub> | **15.0**<br><sub>20.0 · 15.0 · 10.0</sub> | **18.3**<br><sub>20.0 · 20.0 · 15.0</sub> | **20.0**<br><sub>25.0 · 15.0 · 20.0</sub> | **26.7**<br><sub>30.0 · 25.0 · 25.0</sub> |
| Barn | **16.7**<br><sub>20.0 · 15.0 · 15.0</sub> | **13.3**<br><sub>15.0 · 10.0 · 15.0</sub> | **21.7**<br><sub>25.0 · 20.0 · 20.0</sub> | **10.0**<br><sub>5.0 · 10.0 · 15.0</sub> | **8.3**<br><sub>5.0 · 20.0 · 0.0</sub> | **18.3**<br><sub>5.0 · 30.0 · 20.0</sub> | **23.3**<br><sub>25.0 · 20.0 · 25.0</sub> | **36.7**<br><sub>40.0 · 35.0 · 35.0</sub> |
| Lantern | **13.3**<br><sub>20.0 · 10.0 · 10.0</sub> | **18.3**<br><sub>20.0 · 10.0 · 25.0</sub> | **13.3**<br><sub>15.0 · 15.0 · 10.0</sub> | **11.7**<br><sub>10.0 · 10.0 · 15.0</sub> | **15.0**<br><sub>20.0 · 10.0 · 15.0</sub> | **15.0**<br><sub>25.0 · 15.0 · 5.0</sub> | **16.7**<br><sub>10.0 · 15.0 · 25.0</sub> | **30.0**<br><sub>25.0 · 40.0 · 25.0</sub> |
| **Seen (8 games)** | **17.3**<br><sub>19.4 · 17.5 · 15.0</sub> | **18.7**<br><sub>20.6 · 18.1 · 17.5</sub> | **20.4**<br><sub>22.5 · 19.4 · 19.4</sub> | **13.9**<br><sub>10.6 · 15.6 · 15.6</sub> | **15.8**<br><sub>16.9 · 15.0 · 15.6</sub> | **14.0**<br><sub>15.0 · 13.8 · 13.1</sub> | **24.4**<br><sub>25.6 · 23.8 · 23.8</sub> | **29.4**<br><sub>28.1 · 30.6 · 29.4</sub> |
| Jigsaw | **6.7**<br><sub>5.0 · 5.0 · 10.0</sub> | **6.7**<br><sub>10.0 · 5.0 · 5.0</sub> | **13.3**<br><sub>10.0 · 15.0 · 15.0</sub> | **3.3**<br><sub>5.0 · 5.0 · 0.0</sub> | **6.7**<br><sub>5.0 · 10.0 · 5.0</sub> | **5.0**<br><sub>5.0 · 5.0 · 5.0</sub> | **5.0**<br><sub>5.0 · 5.0 · 5.0</sub> | **8.3**<br><sub>10.0 · 5.0 · 10.0</sub> |
| Kaleidoscope | **21.7**<br><sub>20.0 · 25.0 · 20.0</sub> | **15.0**<br><sub>5.0 · 10.0 · 30.0</sub> | **16.7**<br><sub>10.0 · 15.0 · 25.0</sub> | **20.0**<br><sub>15.0 · 25.0 · 20.0</sub> | **10.0**<br><sub>10.0 · 10.0 · 10.0</sub> | **20.0**<br><sub>25.0 · 15.0 · 20.0</sub> | **23.3**<br><sub>20.0 · 20.0 · 30.0</sub> | **18.3**<br><sub>15.0 · 15.0 · 25.0</sub> |
| Bird | **20.0**<br><sub>20.0 · 15.0 · 25.0</sub> | **15.0**<br><sub>5.0 · 20.0 · 20.0</sub> | **20.0**<br><sub>15.0 · 15.0 · 30.0</sub> | **10.0**<br><sub>10.0 · 10.0 · 10.0</sub> | **18.3**<br><sub>10.0 · 25.0 · 20.0</sub> | **13.3**<br><sub>15.0 · 10.0 · 15.0</sub> | **13.3**<br><sub>10.0 · 10.0 · 20.0</sub> | **6.7**<br><sub>5.0 · 5.0 · 10.0</sub> |
| Recipe | **8.3**<br><sub>5.0 · 15.0 · 5.0</sub> | **11.7**<br><sub>15.0 · 15.0 · 5.0</sub> | **5.0**<br><sub>10.0 · 0.0 · 5.0</sub> | **10.0**<br><sub>10.0 · 5.0 · 15.0</sub> | **18.3**<br><sub>20.0 · 20.0 · 15.0</sub> | **20.0**<br><sub>20.0 · 15.0 · 25.0</sub> | **15.0**<br><sub>15.0 · 10.0 · 20.0</sub> | **28.3**<br><sub>30.0 · 35.0 · 20.0</sub> |
| Bell | **21.7**<br><sub>15.0 · 20.0 · 30.0</sub> | **25.0**<br><sub>25.0 · 35.0 · 15.0</sub> | **28.3**<br><sub>35.0 · 35.0 · 15.0</sub> | **25.0**<br><sub>35.0 · 25.0 · 15.0</sub> | **30.0**<br><sub>35.0 · 25.0 · 30.0</sub> | **25.0**<br><sub>25.0 · 20.0 · 30.0</sub> | **50.0**<br><sub>45.0 · 60.0 · 45.0</sub> | **68.3**<br><sub>70.0 · 65.0 · 70.0</sub> |
| Bee | **23.3**<br><sub>30.0 · 15.0 · 25.0</sub> | **30.0**<br><sub>25.0 · 35.0 · 30.0</sub> | **16.7**<br><sub>15.0 · 15.0 · 20.0</sub> | **23.3**<br><sub>20.0 · 25.0 · 25.0</sub> | **10.0**<br><sub>15.0 · 10.0 · 5.0</sub> | **16.7**<br><sub>20.0 · 5.0 · 25.0</sub> | **16.7**<br><sub>15.0 · 20.0 · 15.0</sub> | **28.3**<br><sub>30.0 · 30.0 · 25.0</sub> |
| Candle | **18.3**<br><sub>20.0 · 25.0 · 10.0</sub> | **20.0**<br><sub>15.0 · 30.0 · 15.0</sub> | **13.3**<br><sub>15.0 · 15.0 · 10.0</sub> | **13.3**<br><sub>20.0 · 10.0 · 10.0</sub> | **8.3**<br><sub>5.0 · 10.0 · 10.0</sub> | **15.0**<br><sub>20.0 · 10.0 · 15.0</sub> | **28.3**<br><sub>30.0 · 25.0 · 30.0</sub> | **36.7**<br><sub>25.0 · 50.0 · 35.0</sub> |
| **Unseen (7 games)** | **17.1**<br><sub>16.4 · 17.1 · 17.9</sub> | **17.6**<br><sub>14.3 · 21.4 · 17.1</sub> | **16.2**<br><sub>15.7 · 15.7 · 17.1</sub> | **15.0**<br><sub>16.4 · 15.0 · 13.6</sub> | **14.5**<br><sub>14.3 · 15.7 · 13.6</sub> | **16.4**<br><sub>18.6 · 11.4 · 19.3</sub> | **21.7**<br><sub>20.0 · 21.4 · 23.6</sub> | **27.9**<br><sub>26.4 · 29.3 · 27.9</sub> |
| **Overall** | **17.2**<br><sub>18.0 · 17.3 · 16.3</sub> | **18.2**<br><sub>17.7 · 19.7 · 17.3</sub> | **18.4**<br><sub>19.3 · 17.7 · 18.3</sub> | **14.4**<br><sub>13.3 · 15.3 · 14.7</sub> | **15.2**<br><sub>15.7 · 15.3 · 14.7</sub> | **15.1**<br><sub>16.7 · 12.7 · 16.0</sub> | **23.1**<br><sub>23.0 · 22.7 · 23.7</sub> | **28.7**<br><sub>27.3 · 30.0 · 28.7</sub> |

#### Qwen3-VL-8B-Thinking

| Subtask | Base | Base (w/ belief) | SFT (action-only) | PPO (action-only) | GRPO (action-only) | PPO (belief + action) | RLVB (EM) | RLVB (GRPO) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Face | **18.3**<br><sub>20.0 · 15.0 · 20.0</sub> | **13.3**<br><sub>10.0 · 5.0 · 25.0</sub> | **15.0**<br><sub>25.0 · 15.0 · 5.0</sub> | **11.7**<br><sub>15.0 · 10.0 · 10.0</sub> | **18.3**<br><sub>20.0 · 15.0 · 20.0</sub> | **16.7**<br><sub>15.0 · 15.0 · 20.0</sub> | **28.3**<br><sub>40.0 · 25.0 · 20.0</sub> | **16.7**<br><sub>25.0 · 15.0 · 10.0</sub> |
| Fish | **5.0**<br><sub>5.0 · 5.0 · 5.0</sub> | **11.7**<br><sub>15.0 · 10.0 · 10.0</sub> | **5.0**<br><sub>5.0 · 5.0 · 5.0</sub> | **1.7**<br><sub>0.0 · 0.0 · 5.0</sub> | **3.3**<br><sub>0.0 · 5.0 · 5.0</sub> | **13.3**<br><sub>10.0 · 10.0 · 20.0</sub> | **15.0**<br><sub>20.0 · 10.0 · 15.0</sub> | **5.0**<br><sub>5.0 · 5.0 · 5.0</sub> |
| Turtle | **15.0**<br><sub>25.0 · 5.0 · 15.0</sub> | **38.3**<br><sub>55.0 · 30.0 · 30.0</sub> | **21.7**<br><sub>30.0 · 10.0 · 25.0</sub> | **20.0**<br><sub>20.0 · 25.0 · 15.0</sub> | **18.3**<br><sub>20.0 · 10.0 · 25.0</sub> | **25.0**<br><sub>20.0 · 30.0 · 25.0</sub> | **45.0**<br><sub>55.0 · 30.0 · 50.0</sub> | **45.0**<br><sub>50.0 · 60.0 · 25.0</sub> |
| Balloon | **11.7**<br><sub>10.0 · 20.0 · 5.0</sub> | **26.7**<br><sub>30.0 · 25.0 · 25.0</sub> | **30.0**<br><sub>35.0 · 25.0 · 30.0</sub> | **18.3**<br><sub>25.0 · 15.0 · 15.0</sub> | **11.7**<br><sub>15.0 · 10.0 · 10.0</sub> | **21.7**<br><sub>20.0 · 25.0 · 20.0</sub> | **40.0**<br><sub>45.0 · 40.0 · 35.0</sub> | **38.3**<br><sub>35.0 · 45.0 · 35.0</sub> |
| Rocket | **28.3**<br><sub>15.0 · 25.0 · 45.0</sub> | **28.3**<br><sub>15.0 · 30.0 · 40.0</sub> | **30.0**<br><sub>20.0 · 30.0 · 40.0</sub> | **15.0**<br><sub>5.0 · 20.0 · 20.0</sub> | **13.3**<br><sub>20.0 · 10.0 · 10.0</sub> | **31.7**<br><sub>45.0 · 25.0 · 25.0</sub> | **40.0**<br><sub>55.0 · 40.0 · 25.0</sub> | **38.3**<br><sub>45.0 · 35.0 · 35.0</sub> |
| Sailboat | **8.3**<br><sub>0.0 · 15.0 · 10.0</sub> | **25.0**<br><sub>30.0 · 15.0 · 30.0</sub> | **15.0**<br><sub>15.0 · 15.0 · 15.0</sub> | **8.3**<br><sub>5.0 · 0.0 · 20.0</sub> | **13.3**<br><sub>10.0 · 10.0 · 20.0</sub> | **21.7**<br><sub>30.0 · 5.0 · 30.0</sub> | **38.3**<br><sub>35.0 · 50.0 · 30.0</sub> | **31.7**<br><sub>25.0 · 35.0 · 35.0</sub> |
| Barn | **11.7**<br><sub>10.0 · 15.0 · 10.0</sub> | **13.3**<br><sub>15.0 · 15.0 · 10.0</sub> | **18.3**<br><sub>25.0 · 10.0 · 20.0</sub> | **13.3**<br><sub>10.0 · 15.0 · 15.0</sub> | **15.0**<br><sub>20.0 · 15.0 · 10.0</sub> | **15.0**<br><sub>20.0 · 15.0 · 10.0</sub> | **38.3**<br><sub>50.0 · 40.0 · 25.0</sub> | **36.7**<br><sub>40.0 · 25.0 · 45.0</sub> |
| Lantern | **20.0**<br><sub>15.0 · 20.0 · 25.0</sub> | **18.3**<br><sub>10.0 · 30.0 · 15.0</sub> | **28.3**<br><sub>25.0 · 20.0 · 40.0</sub> | **10.0**<br><sub>10.0 · 10.0 · 10.0</sub> | **18.3**<br><sub>5.0 · 25.0 · 25.0</sub> | **16.7**<br><sub>5.0 · 25.0 · 20.0</sub> | **16.7**<br><sub>15.0 · 20.0 · 15.0</sub> | **31.7**<br><sub>10.0 · 45.0 · 40.0</sub> |
| **Seen (8 games)** | **14.8**<br><sub>12.5 · 15.0 · 16.9</sub> | **21.9**<br><sub>22.5 · 20.0 · 23.1</sub> | **20.4**<br><sub>22.5 · 16.3 · 22.5</sub> | **12.3**<br><sub>11.3 · 11.9 · 13.8</sub> | **14.0**<br><sub>13.8 · 12.5 · 15.6</sub> | **20.2**<br><sub>20.6 · 18.8 · 21.3</sub> | **32.7**<br><sub>39.4 · 31.9 · 26.9</sub> | **30.4**<br><sub>29.4 · 33.1 · 28.8</sub> |
| Jigsaw | **3.3**<br><sub>5.0 · 5.0 · 0.0</sub> | **3.3**<br><sub>0.0 · 5.0 · 5.0</sub> | **5.0**<br><sub>0.0 · 10.0 · 5.0</sub> | **1.7**<br><sub>0.0 · 5.0 · 0.0</sub> | **5.0**<br><sub>5.0 · 5.0 · 5.0</sub> | **6.7**<br><sub>10.0 · 0.0 · 10.0</sub> | **6.7**<br><sub>10.0 · 5.0 · 5.0</sub> | **26.7**<br><sub>15.0 · 40.0 · 25.0</sub> |
| Kaleidoscope | **11.7**<br><sub>15.0 · 15.0 · 5.0</sub> | **15.0**<br><sub>20.0 · 15.0 · 10.0</sub> | **16.7**<br><sub>10.0 · 20.0 · 20.0</sub> | **21.7**<br><sub>30.0 · 20.0 · 15.0</sub> | **13.3**<br><sub>10.0 · 15.0 · 15.0</sub> | **18.3**<br><sub>20.0 · 25.0 · 10.0</sub> | **33.3**<br><sub>40.0 · 35.0 · 25.0</sub> | **26.7**<br><sub>25.0 · 30.0 · 25.0</sub> |
| Bird | **11.7**<br><sub>10.0 · 10.0 · 15.0</sub> | **16.7**<br><sub>15.0 · 15.0 · 20.0</sub> | **15.0**<br><sub>5.0 · 25.0 · 15.0</sub> | **13.3**<br><sub>20.0 · 5.0 · 15.0</sub> | **16.7**<br><sub>30.0 · 15.0 · 5.0</sub> | **5.0**<br><sub>0.0 · 5.0 · 10.0</sub> | **8.3**<br><sub>5.0 · 15.0 · 5.0</sub> | **5.0**<br><sub>10.0 · 5.0 · 0.0</sub> |
| Recipe | **5.0**<br><sub>5.0 · 5.0 · 5.0</sub> | **15.0**<br><sub>15.0 · 15.0 · 15.0</sub> | **11.7**<br><sub>10.0 · 20.0 · 5.0</sub> | **8.3**<br><sub>10.0 · 10.0 · 5.0</sub> | **8.3**<br><sub>10.0 · 5.0 · 10.0</sub> | **5.0**<br><sub>5.0 · 5.0 · 5.0</sub> | **25.0**<br><sub>20.0 · 20.0 · 35.0</sub> | **18.3**<br><sub>20.0 · 30.0 · 5.0</sub> |
| Bell | **25.0**<br><sub>35.0 · 20.0 · 20.0</sub> | **31.7**<br><sub>35.0 · 30.0 · 30.0</sub> | **26.7**<br><sub>25.0 · 20.0 · 35.0</sub> | **16.7**<br><sub>15.0 · 20.0 · 15.0</sub> | **26.7**<br><sub>30.0 · 25.0 · 25.0</sub> | **26.7**<br><sub>35.0 · 35.0 · 10.0</sub> | **53.3**<br><sub>50.0 · 50.0 · 60.0</sub> | **50.0**<br><sub>40.0 · 40.0 · 70.0</sub> |
| Bee | **21.7**<br><sub>15.0 · 35.0 · 15.0</sub> | **30.0**<br><sub>25.0 · 30.0 · 35.0</sub> | **21.7**<br><sub>15.0 · 25.0 · 25.0</sub> | **11.7**<br><sub>5.0 · 15.0 · 15.0</sub> | **18.3**<br><sub>25.0 · 20.0 · 10.0</sub> | **13.3**<br><sub>15.0 · 15.0 · 10.0</sub> | **43.3**<br><sub>50.0 · 30.0 · 50.0</sub> | **43.3**<br><sub>40.0 · 45.0 · 45.0</sub> |
| Candle | **25.0**<br><sub>30.0 · 15.0 · 30.0</sub> | **20.0**<br><sub>25.0 · 25.0 · 10.0</sub> | **30.0**<br><sub>25.0 · 45.0 · 20.0</sub> | **16.7**<br><sub>20.0 · 15.0 · 15.0</sub> | **23.3**<br><sub>30.0 · 20.0 · 20.0</sub> | **16.7**<br><sub>25.0 · 15.0 · 10.0</sub> | **55.0**<br><sub>55.0 · 50.0 · 60.0</sub> | **35.0**<br><sub>35.0 · 30.0 · 40.0</sub> |
| **Unseen (7 games)** | **14.8**<br><sub>16.4 · 15.0 · 12.9</sub> | **18.8**<br><sub>19.3 · 19.3 · 17.9</sub> | **18.1**<br><sub>12.9 · 23.6 · 17.9</sub> | **12.9**<br><sub>14.3 · 12.9 · 11.4</sub> | **16.0**<br><sub>20.0 · 15.0 · 12.9</sub> | **13.1**<br><sub>15.7 · 14.3 · 9.3</sub> | **32.2**<br><sub>32.9 · 29.3 · 34.3</sub> | **29.3**<br><sub>26.4 · 31.4 · 30.0</sub> |
| **Overall** | **14.8**<br><sub>14.3 · 15.0 · 15.0</sub> | **20.5**<br><sub>21.0 · 19.7 · 20.7</sub> | **19.3**<br><sub>18.0 · 19.7 · 20.3</sub> | **12.6**<br><sub>12.7 · 12.3 · 12.7</sub> | **14.9**<br><sub>16.7 · 13.7 · 14.3</sub> | **16.9**<br><sub>18.3 · 16.7 · 15.7</sub> | **32.4**<br><sub>36.3 · 30.7 · 30.3</sub> | **29.9**<br><sub>28.0 · 32.3 · 29.3</sub> |

### 3. HiddenGames (7 tasks; test split, 756 games)

#### Qwen3.5-4B

| Subtask | Base | Base (w/ belief) | SFT (action-only) | PPO (action-only) | GRPO (action-only) | PPO (belief + action) | RLVB (EM) | RLVB (GRPO) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minesweeper_easy | **1.3**<br><sub>3.0 · 1.0 · 0.0</sub> | **0.3**<br><sub>1.0 · 0.0 · 0.0</sub> | **0.3**<br><sub>1.0 · 0.0 · 0.0</sub> | **0.7**<br><sub>1.0 · 0.0 · 1.0</sub> | **0.7**<br><sub>0.0 · 2.0 · 0.0</sub> | **4.0**<br><sub>4.0 · 6.0 · 2.0</sub> | **2.7**<br><sub>1.0 · 3.0 · 4.0</sub> | **0.0**<br><sub>0.0 · 0.0 · 0.0</sub> |
| minigrid_keycorridor | **30.0**<br><sub>28.0 · 32.0 · 30.0</sub> | **30.7**<br><sub>36.0 · 26.0 · 30.0</sub> | **29.7**<br><sub>27.0 · 30.0 · 32.0</sub> | **34.7**<br><sub>32.0 · 37.0 · 35.0</sub> | **32.7**<br><sub>30.0 · 28.0 · 40.0</sub> | **29.3**<br><sub>20.0 · 33.0 · 35.0</sub> | **28.7**<br><sub>24.0 · 29.0 · 33.0</sub> | **32.3**<br><sub>23.0 · 36.0 · 38.0</sub> |
| frozenlake_hidden_goal | **41.1**<br><sub>43.0 · 40.6 · 39.8</sub> | **46.4**<br><sub>46.1 · 43.0 · 50.0</sub> | **31.0**<br><sub>31.3 · 31.3 · 30.5</sub> | **16.6**<br><sub>14.8 · 14.8 · 20.3</sub> | **28.1**<br><sub>30.5 · 24.2 · 29.7</sub> | **45.1**<br><sub>47.7 · 40.6 · 46.9</sub> | **50.0**<br><sub>50.0 · 49.2 · 50.8</sub> | **54.4**<br><sub>54.7 · 54.7 · 53.9</sub> |
| sokoban_hidden_target | **27.1**<br><sub>31.3 · 27.3 · 22.7</sub> | **30.2**<br><sub>30.5 · 29.7 · 30.5</sub> | **23.7**<br><sub>22.7 · 24.2 · 24.2</sub> | **28.4**<br><sub>25.0 · 32.8 · 27.3</sub> | **24.0**<br><sub>25.0 · 22.7 · 24.2</sub> | **31.3**<br><sub>30.5 · 34.4 · 28.9</sub> | **30.0**<br><sub>28.9 · 31.3 · 29.7</sub> | **27.9**<br><sub>30.5 · 25.0 · 28.1</sub> |
| tabletop_lamps | **20.0**<br><sub>17.0 · 23.0 · 20.0</sub> | **67.0**<br><sub>66.0 · 59.0 · 76.0</sub> | **31.7**<br><sub>34.0 · 35.0 · 26.0</sub> | **1.0**<br><sub>1.0 · 2.0 · 0.0</sub> | **53.3**<br><sub>60.0 · 52.0 · 48.0</sub> | **78.7**<br><sub>76.0 · 85.0 · 75.0</sub> | **86.7**<br><sub>91.0 · 85.0 · 84.0</sub> | **95.7**<br><sub>94.0 · 97.0 · 96.0</sub> |
| tabletop_cups | **44.0**<br><sub>43.0 · 46.0 · 43.0</sub> | **51.3**<br><sub>44.0 · 54.0 · 56.0</sub> | **43.7**<br><sub>51.0 · 40.0 · 40.0</sub> | **49.3**<br><sub>43.0 · 48.0 · 57.0</sub> | **56.0**<br><sub>52.0 · 56.0 · 60.0</sub> | **63.0**<br><sub>67.0 · 55.0 · 67.0</sub> | **61.3**<br><sub>63.0 · 57.0 · 64.0</sub> | **77.7**<br><sub>81.0 · 73.0 · 79.0</sub> |
| tabletop_blocks | **17.7**<br><sub>19.0 · 18.0 · 16.0</sub> | **40.7**<br><sub>36.0 · 49.0 · 37.0</sub> | **28.0**<br><sub>27.0 · 30.0 · 27.0</sub> | **30.0**<br><sub>23.0 · 35.0 · 32.0</sub> | **36.7**<br><sub>40.0 · 38.0 · 32.0</sub> | **34.0**<br><sub>35.0 · 35.0 · 32.0</sub> | **42.7**<br><sub>37.0 · 47.0 · 44.0</sub> | **55.3**<br><sub>54.0 · 54.0 · 58.0</sub> |
| **Overall** | **26.5**<br><sub>27.1 · 27.4 · 25.0</sub> | **38.1**<br><sub>37.2 · 37.2 · 39.9</sub> | **26.9**<br><sub>27.6 · 27.2 · 25.8</sub> | **22.9**<br><sub>20.0 · 24.2 · 24.6</sub> | **32.5**<br><sub>33.5 · 31.2 · 32.9</sub> | **40.5**<br><sub>39.9 · 41.0 · 40.7</sub> | **42.9**<br><sub>41.9 · 42.9 · 43.9</sub> | **48.5**<br><sub>47.8 · 47.9 · 49.7</sub> |

#### Qwen3-VL-4B-Thinking

| Subtask | Base | Base (w/ belief) | SFT (action-only) | PPO (action-only) | GRPO (action-only) | PPO (belief + action) | RLVB (EM) | RLVB (GRPO) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minesweeper_easy | **1.3**<br><sub>2.0 · 1.0 · 1.0</sub> | **3.7**<br><sub>4.0 · 4.0 · 3.0</sub> | **2.3**<br><sub>3.0 · 4.0 · 0.0</sub> | **0.7**<br><sub>0.0 · 1.0 · 1.0</sub> | **0.7**<br><sub>0.0 · 0.0 · 2.0</sub> | **2.3**<br><sub>1.0 · 2.0 · 4.0</sub> | **0.0**<br><sub>0.0 · 0.0 · 0.0</sub> | **0.7**<br><sub>1.0 · 0.0 · 1.0</sub> |
| minigrid_keycorridor | **33.7**<br><sub>35.0 · 30.0 · 36.0</sub> | **32.7**<br><sub>34.0 · 33.0 · 31.0</sub> | **30.3**<br><sub>29.0 · 29.0 · 33.0</sub> | **31.7**<br><sub>29.0 · 35.0 · 31.0</sub> | **33.7**<br><sub>35.0 · 41.0 · 25.0</sub> | **36.7**<br><sub>33.0 · 35.0 · 42.0</sub> | **31.7**<br><sub>38.0 · 25.0 · 32.0</sub> | **34.0**<br><sub>32.0 · 37.0 · 33.0</sub> |
| frozenlake_hidden_goal | **41.4**<br><sub>41.4 · 41.4 · 41.4</sub> | **41.7**<br><sub>44.5 · 43.8 · 36.7</sub> | **45.3**<br><sub>45.3 · 46.1 · 44.5</sub> | **49.8**<br><sub>50.0 · 47.7 · 51.6</sub> | **42.7**<br><sub>43.8 · 44.5 · 39.8</sub> | **41.9**<br><sub>39.8 · 42.2 · 43.8</sub> | **40.4**<br><sub>39.1 · 41.4 · 40.6</sub> | **48.7**<br><sub>46.9 · 49.2 · 50.0</sub> |
| sokoban_hidden_target | **24.0**<br><sub>24.2 · 25.8 · 21.9</sub> | **24.0**<br><sub>25.0 · 22.7 · 24.2</sub> | **27.6**<br><sub>28.1 · 25.8 · 28.9</sub> | **26.0**<br><sub>27.3 · 24.2 · 26.6</sub> | **26.5**<br><sub>28.9 · 23.4 · 27.3</sub> | **24.5**<br><sub>24.2 · 24.2 · 25.0</sub> | **25.0**<br><sub>29.7 · 21.9 · 23.4</sub> | **23.7**<br><sub>24.2 · 21.9 · 25.0</sub> |
| tabletop_lamps | **1.0**<br><sub>0.0 · 1.0 · 2.0</sub> | **3.3**<br><sub>5.0 · 2.0 · 3.0</sub> | **24.3**<br><sub>22.0 · 25.0 · 26.0</sub> | **1.7**<br><sub>2.0 · 2.0 · 1.0</sub> | **2.3**<br><sub>3.0 · 2.0 · 2.0</sub> | **8.3**<br><sub>8.0 · 6.0 · 11.0</sub> | **26.0**<br><sub>29.0 · 21.0 · 28.0</sub> | **47.3**<br><sub>47.0 · 49.0 · 46.0</sub> |
| tabletop_cups | **35.3**<br><sub>33.0 · 38.0 · 35.0</sub> | **29.7**<br><sub>30.0 · 30.0 · 29.0</sub> | **63.7**<br><sub>64.0 · 65.0 · 62.0</sub> | **45.0**<br><sub>46.0 · 42.0 · 47.0</sub> | **32.7**<br><sub>33.0 · 33.0 · 32.0</sub> | **30.3**<br><sub>38.0 · 27.0 · 26.0</sub> | **64.3**<br><sub>69.0 · 54.0 · 70.0</sub> | **70.3**<br><sub>72.0 · 72.0 · 67.0</sub> |
| tabletop_blocks | **31.0**<br><sub>29.0 · 35.0 · 29.0</sub> | **32.0**<br><sub>32.0 · 35.0 · 29.0</sub> | **22.0**<br><sub>21.0 · 22.0 · 23.0</sub> | **24.0**<br><sub>23.0 · 25.0 · 24.0</sub> | **30.0**<br><sub>24.0 · 33.0 · 33.0</sub> | **33.0**<br><sub>41.0 · 29.0 · 29.0</sub> | **32.7**<br><sub>34.0 · 31.0 · 33.0</sub> | **29.0**<br><sub>33.0 · 24.0 · 30.0</sub> |
| **Overall** | **24.6**<br><sub>24.2 · 25.3 · 24.3</sub> | **24.5**<br><sub>25.7 · 25.0 · 22.9</sub> | **31.2**<br><sub>30.8 · 31.3 · 31.5</sub> | **26.5**<br><sub>26.3 · 26.1 · 27.0</sub> | **24.9**<br><sub>24.9 · 25.9 · 23.8</sub> | **25.9**<br><sub>26.9 · 24.3 · 26.5</sub> | **31.5**<br><sub>34.1 · 28.0 · 32.4</sub> | **36.2**<br><sub>36.5 · 36.1 · 36.1</sub> |

#### Qwen3-VL-8B-Thinking

| Subtask | Base | Base (w/ belief) | SFT (action-only) | PPO (action-only) | GRPO (action-only) | PPO (belief + action) | RLVB (EM) | RLVB (GRPO) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minesweeper_easy | **1.3**<br><sub>1.0 · 3.0 · 0.0</sub> | **1.0**<br><sub>2.0 · 0.0 · 1.0</sub> | **1.0**<br><sub>2.0 · 0.0 · 1.0</sub> | **1.0**<br><sub>2.0 · 1.0 · 0.0</sub> | **0.7**<br><sub>1.0 · 1.0 · 0.0</sub> | **2.0**<br><sub>1.0 · 4.0 · 1.0</sub> | **0.3**<br><sub>0.0 · 1.0 · 0.0</sub> | **0.0**<br><sub>0.0 · 0.0 · 0.0</sub> |
| minigrid_keycorridor | **30.7**<br><sub>31.0 · 32.0 · 29.0</sub> | **31.3**<br><sub>29.0 · 31.0 · 34.0</sub> | **32.3**<br><sub>31.0 · 28.0 · 38.0</sub> | **32.3**<br><sub>29.0 · 30.0 · 38.0</sub> | **30.7**<br><sub>31.0 · 31.0 · 30.0</sub> | **34.3**<br><sub>34.0 · 31.0 · 38.0</sub> | **32.3**<br><sub>39.0 · 29.0 · 29.0</sub> | **32.7**<br><sub>32.0 · 29.0 · 37.0</sub> |
| frozenlake_hidden_goal | **42.5**<br><sub>43.8 · 39.8 · 43.8</sub> | **41.1**<br><sub>45.3 · 42.2 · 35.9</sub> | **44.5**<br><sub>45.3 · 43.8 · 44.5</sub> | **39.6**<br><sub>37.5 · 46.9 · 34.4</sub> | **43.0**<br><sub>43.0 · 41.4 · 44.5</sub> | **37.8**<br><sub>38.3 · 39.1 · 35.9</sub> | **35.4**<br><sub>39.1 · 33.6 · 33.6</sub> | **40.9**<br><sub>41.4 · 38.3 · 43.0</sub> |
| sokoban_hidden_target | **22.7**<br><sub>21.9 · 21.1 · 25.0</sub> | **24.0**<br><sub>24.2 · 26.6 · 21.1</sub> | **20.9**<br><sub>21.9 · 18.0 · 22.7</sub> | **22.9**<br><sub>23.4 · 20.3 · 25.0</sub> | **22.4**<br><sub>22.7 · 21.1 · 23.4</sub> | **21.4**<br><sub>21.9 · 25.0 · 17.2</sub> | **23.4**<br><sub>23.4 · 21.9 · 25.0</sub> | **24.2**<br><sub>22.7 · 26.6 · 23.4</sub> |
| tabletop_lamps | **1.3**<br><sub>2.0 · 1.0 · 1.0</sub> | **1.7**<br><sub>0.0 · 2.0 · 3.0</sub> | **5.7**<br><sub>6.0 · 4.0 · 7.0</sub> | **1.0**<br><sub>0.0 · 1.0 · 2.0</sub> | **1.0**<br><sub>1.0 · 1.0 · 1.0</sub> | **1.3**<br><sub>2.0 · 1.0 · 1.0</sub> | **21.3**<br><sub>18.0 · 20.0 · 26.0</sub> | **32.3**<br><sub>32.0 · 28.0 · 37.0</sub> |
| tabletop_cups | **25.7**<br><sub>34.0 · 21.0 · 22.0</sub> | **38.3**<br><sub>39.0 · 37.0 · 39.0</sub> | **51.7**<br><sub>56.0 · 51.0 · 48.0</sub> | **31.3**<br><sub>30.0 · 35.0 · 29.0</sub> | **26.3**<br><sub>25.0 · 29.0 · 25.0</sub> | **56.3**<br><sub>58.0 · 61.0 · 50.0</sub> | **64.3**<br><sub>67.0 · 57.0 · 69.0</sub> | **61.3**<br><sub>63.0 · 66.0 · 55.0</sub> |
| tabletop_blocks | **31.0**<br><sub>34.0 · 31.0 · 28.0</sub> | **25.0**<br><sub>27.0 · 24.0 · 24.0</sub> | **29.7**<br><sub>30.0 · 31.0 · 28.0</sub> | **33.3**<br><sub>34.0 · 33.0 · 33.0</sub> | **32.3**<br><sub>34.0 · 34.0 · 29.0</sub> | **30.0**<br><sub>30.0 · 28.0 · 32.0</sub> | **25.7**<br><sub>19.0 · 33.0 · 25.0</sub> | **25.7**<br><sub>30.0 · 21.0 · 26.0</sub> |
| **Overall** | **22.9**<br><sub>24.6 · 22.0 · 22.2</sub> | **23.9**<br><sub>24.6 · 24.1 · 23.0</sub> | **27.0**<br><sub>27.9 · 25.5 · 27.5</sub> | **23.7**<br><sub>22.9 · 24.6 · 23.5</sub> | **23.1**<br><sub>23.3 · 23.3 · 22.8</sub> | **26.4**<br><sub>26.7 · 27.4 · 25.1</sub> | **29.0**<br><sub>29.5 · 27.9 · 29.6</sub> | **31.1**<br><sub>31.6 · 30.0 · 31.7</sub> |

## 🛠️ Installation

### Requirements

- Linux with NVIDIA GPUs. The tested environment: Python 3.12, PyTorch 2.13 (CUDA 13.0), transformers 5.17, vLLM 0.32 and
  peft 0.19.
- Attention: flash-attention 3 on Hopper GPUs. On A100 GPUs, install flash-attention 2 and set
  `PL_ATTN=flash_attention_2`.
- Node.js 22 or later for the Pro-Worlds engine (`node` on `PATH`, or `NODE_BIN`).
- A GPU with a Vulkan driver for the tabletop tasks (ManiSkill3, Vulkan / EGL).
- The RoboQuest runtime (RoboCasa simulator; `ROBOQUEST_RUNTIME`). You download the RoboCasa assets yourself.

The full steps are in [docs/install.md](docs/install.md).

### Install

```bash
git clone https://github.com/declare-lab/RLVB && cd RLVB
# install PyTorch and vLLM for your CUDA version first; then plearn from the checkout, editable (plearn reads its tasks,
# configs/ and benchmarks/ from the checkout), with the game simulators and ManiSkill3 for the tabletop tasks
pip install -e ".[games,tabletop,test]"
# RoboQuest: build its simulator runtime once (RoboCasa's kitchen assets: you download them)
(cd benchmarks/roboquest && scripts/bootstrap_simulator.sh --runtime ~/roboquest-runtime --assets download)
# in every new shell: sets ROBOQUEST_RUNTIME
source ~/roboquest-runtime/env.sh
# the base models: one folder per model under $PL_MODELS
export PL_MODELS=$HOME/models
hf download Qwen/Qwen3-VL-8B-Thinking --local-dir $PL_MODELS/Qwen3-VL-8B-Thinking
hf download Qwen/Qwen3-VL-4B-Thinking --local-dir $PL_MODELS/Qwen3-VL-4B-Thinking
hf download Qwen/Qwen3.5-4B           --local-dir $PL_MODELS/Qwen3.5-4B
```

### Check the installation

```bash
# the unit tests that need no GPU; each command ends with "OK" (some tests skip without a GPU, Node.js or PL_MODELS)
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests/method -t .
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests/tools -t .
# the registered suites; prints train_suite (8 tasks), test_suite (32 tasks) and test_suite_long (32 tasks)
python -m plearn tasks --groups
# the Pro-Worlds engine is unchanged and its rule fixtures pass
(cd benchmarks/proworlds/games/expansion-6.1 && sha256sum -c SHA256SUMS && node test.mjs --fixtures)
# a training script with DRY=1: prints the training command and the test commands, runs nothing
DRY=1 scripts/pl_qwen3vl_8b_thinking.sh
```

## 🧑‍💻 Usage

Each training script trains once on `train_suite` (50 updates, LoRA r 16), then tests `adapter_50` on `test_suite` at the
seeds 0, 1 and 2 with the test script of its model. `DRY=1` prints the commands only. `SKIP_TESTS=1` trains only. The same
scripts exist for `qwen3vl_8b_thinking` (Qwen3-VL-8B-Thinking), `qwen3vl_4b_thinking` (Qwen3-VL-4B-Thinking) and `qwen35_4b`
(Qwen3.5-4B).

### Train: RLVB (EM)

```bash
# RLVB (EM) with Qwen3-VL-8B-Thinking on GPUs 0-3 (the run file configs/train/suite_qwen3vl_8b_thinking.yaml),
# then the tests of adapter_50
PL_GPUS=0,1,2,3 scripts/pl_qwen3vl_8b_thinking.sh
# continue a stopped run from the end of update 25
python -m plearn train --config suite_qwen3vl_8b_thinking --resume_step 25
```

### Train: RLVB (GRPO)

```bash
# RLVB (GRPO) as in the tables: no belief score (the belief line stays in the turn), on GPUs 0-3, then the tests.
# Qwen3-VL-8B-Thinking and Qwen3-VL-4B-Thinking: the pooled sample set (phase_set 1200, set_keep 800), tests of adapter_50
PL_GPUS=0,1,2,3 scripts/ablation_scripts/pool_nobelief_qwen3vl_8b_thinking.sh
PL_GPUS=0,1,2,3 scripts/ablation_scripts/pool_nobelief_qwen3vl_4b_thinking.sh
# Qwen3.5-4B: phase 5 (phase_set 240, set_keep 160), no length term; the tables use the update-25 adapter
PL_GPUS=0,1,2,3 SKIP_TESTS=1 scripts/ablation_scripts/phase5_nobelief_nolen_qwen35_4b.sh
ADAPTER=runs/phase5_nobelief_nolen_qwen35_4b/queue-phase5_nobelief_nolen_qwen35_4b/adapter_25 scripts/eval_qwen35_4b.sh
# the first RLVB (GRPO) version (belief line + belief score): the RLVB (EM) run file with --update grpo
PL_GPUS=0,1,2,3 scripts/grpo_posterior_qwen3vl_8b_thinking.sh
```

### Test a model

```bash
# a released model on test_suite, seed 0, one GPU; TURN=two_line (the default): the belief line and the action line
MODEL_REPO=richmondsin/pl-qwen3-vl-8b-thinking SEEDS=0 EVAL_GPUS=0 scripts/eval_qwen3vl_8b_thinking.sh
# your own adapter, at the seeds 0, 1 and 2
ADAPTER=runs/suite_qwen3vl_8b_thinking/queue-suite_qwen3vl_8b_thinking/adapter_50 scripts/eval_qwen3vl_8b_thinking.sh
# the base model with the belief line ("Base (w/ belief)"), and with the action line only ("Base", TURN=one_line)
scripts/eval_qwen3vl_8b_thinking.sh
TURN=one_line scripts/eval_qwen3vl_8b_thinking.sh
# one benchmark only (roboquest, proworlds or hidden_games); EXPANSION=long tests the Pro-Worlds long board
BENCH=proworlds scripts/eval_qwen3vl_8b_thinking.sh
```

All settings and variables of the tests are in [docs/evaluation.md](docs/evaluation.md). The training settings are in
[docs/training.md](docs/training.md).

### What a run saves

| Path | Contents |
|---|---|
| `$PL_RUNS/<run>/queue-<run>/adapter_<k>` | the LoRA adapter after update k (25 and 50) |
| `$PL_RUNS/<run>/logs/` | the logs of the learner and the actor |
| `$PL_RUNS/<run>/case-<run>/` | the rollouts and the evaluations of the training |
| `$PL_RUNS/<run>/config-<run>.json` | the run's settings and command lines |
| `$PL_RUNS/eval/<tag>[-<bench>][-long][-one_line]-seed<k>/results.json` | the metrics of each task and of each group, pooled over its episodes |
| `$PL_RUNS/eval/<tag>...-seed<k>/<task>.jsonl` | one row per episode, with its turns |

`PL_RUNS` defaults to `runs`. A new start of a test with the same output folder plays only the missing episodes.

## 📦 Released models

Full models (the adapter merged into the base model) on Hugging Face, in the collection
[`richmondsin/rlvb`](https://huggingface.co/collections/richmondsin/rlvb):

| Learner | Qwen3-VL-8B-Thinking | Qwen3.5-4B | Qwen3-VL-4B-Thinking |
|---|---|---|---|
| RLVB (EM) | `richmondsin/pl-qwen3-vl-8b-thinking` | `richmondsin/pl-qwen3.5-4b` | `richmondsin/pl-qwen3-vl-4b-thinking` |
| RLVB (GRPO) | `richmondsin/grpo-posterior-pool-run-qwen3-vl-8b-thinking` | `richmondsin/grpo-posterior-pool-run-qwen3.5-4b` | `richmondsin/grpo-posterior-pool-run-qwen3-vl-4b-thinking` |

Test a released model with `MODEL_REPO=<repository id> scripts/eval_<model>.sh` (`TURN=two_line`, the default).

## 📏 Scoring

- **Success rate**: the share of games with a correct answer (`success` in `results.json`), in percent.
- **Overall**: for each seed, all games of a benchmark together (correct games / games), then the mean over the seeds 0, 1
  and 2.
- **Errors**: a game that fails because of the server or the environment (not the agent) is left out of the scores and
  counted as `errors`. A seed counts only when every task has 0 errors.
- **Premature answers**: `premature` counts the games with an answer before the evidence identifies the answer.

`python tools/results_cells.py <seed 0 folder> <seed 1 folder> <seed 2 folder>` gives the cells of a results table with
these rules.

## 📁 Repository layout

```text
plearn/api/              the task registry, the task, the episode contract and the agent's turn
plearn/tasks/            the tasks: their episodes, turn format and exact teachers; suites/ has train_suite and test_suite
plearn/method/           the RLVB trainer (RLVB (EM) and RLVB (GRPO)), the teacher and the estimator
plearn/models/           the model registry's resolver, the picture server and the samplers (vLLM)
plearn/evaluator.py      the evaluator (python -m plearn eval)
plearn/train.py          the training launcher (python -m plearn train)
configs/models/          the base models
configs/train/           the run files: RLVB (EM) suite_*.yaml; RLVB (GRPO) of the tables pool_nobelief_qwen3vl_*.yaml and
                         phase5_nobelief_nolen_qwen35_4b.yaml; the first RLVB (GRPO) version grpo_posterior_*.yaml; ablations
                         abl_pl_4b*.yaml and the RLVB (GRPO) ablations (scripts/ablation_scripts/README.md)
scripts/                 one training script for each learner and base model, one test script for each base model;
                         ablation_scripts/: one script for each RLVB (GRPO) ablation run
benchmarks/              the environments' own code: roboquest/, proworlds/games/expansion-6.1/, vagen/, textarena/
tools/                   merge an adapter into a full model, result cells, a KL gate, the teacher's own play
tests/                   unit tests (most run on a CPU)
docs/                    install, evaluation, training, environments
```

## 📝 Citation

```bibtex
@misc{rlvb2026,
  title  = {Lazy Agents Answer Against Their Own Beliefs: Reinforcement Learning with Verifiable Beliefs Teaches Them to Act on What They Believe},
  author = {{Richmond Sin Jing Xuan} and Peilin Feng and Wenbin Wang and Soujanya Poria},
  year   = {2026},
  howpublished = {\url{https://github.com/declare-lab/RLVB}}
}
```

## 📜 License

MIT ([LICENSE](LICENSE)). The vendored code keeps its own license: `benchmarks/roboquest` (MIT), `benchmarks/vagen` (MIT),
`benchmarks/textarena` (MIT).

## 🙏 Acknowledgements

We build on [vLLM](https://github.com/vllm-project/vllm) (serving). Some settings of the trainer follow
[verl](https://github.com/verl-project/verl). The environments use code from [VAGEN](https://github.com/mll-lab-nu/VAGEN) and
RAGEN (FrozenLake, Sokoban), [TextArena](https://github.com/TextArena/TextArena) (the training-only task),
[POPGym Arcade](https://pypi.org/project/popgym-arcade/) (MineSweeper), [MiniGrid](https://github.com/Farama-Foundation/Minigrid)
(KeyCorridor), [ManiSkill](https://github.com/haosulab/ManiSkill) (the tabletop scene) and [RoboCasa](https://github.com/robocasa/robocasa)
(the RoboQuest assets, which you download yourself).
