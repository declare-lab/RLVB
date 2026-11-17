# Install

## The Python environment

The tested environment: Python 3.12, PyTorch 2.13.0 (CUDA 13.0), transformers 5.17.0, vLLM 0.32.0, peft 0.19.1.
Install PyTorch and vLLM for your CUDA version first, then plearn from the checkout:

```sh
git clone https://github.com/declare-lab/RLVB && cd RLVB
pip install -e .                       # plearn: numpy, Pillow, PyYAML
pip install -e ".[games]"              # MineSweeper (POPGym Arcade, JAX on the CPU), MiniGrid, FrozenLake (gymnasium), Sokoban
pip install -e ".[tabletop]"           # ManiSkill3 and SAPIEN for the tabletop tasks
pip install -e ".[test]"               # pytest
```

plearn reads its tasks, `configs/` and `benchmarks/` from the checkout, so use the editable install (`-e`).

## Attention kernels

- Hopper GPUs (H100, H200): flash-attention 3. The RLVB (GRPO) run files ask for `flash_attention_3`.
- A100 GPUs: install flash-attention 2 and set `PL_ATTN=flash_attention_2`. It overrides the run files' `flash_attention_3`.
- With no flash package, `PL_ATTN=sdpa` uses PyTorch's attention. The RLVB (EM) run files use sdpa.

The flash packages are not in the requirements. Use a prebuilt wheel. For PyTorch 2.13 + CUDA 13.0 and Python 3.12, the tested
wheel is `flash_attn_3-3.0.0+cu130torch2.13gite2743ab-cp39-abi3-linux_x86_64.whl` (github.com/mjun0812/flash-attention-prebuild-wheels,
release v0.9.47). Install it with `pip install --no-deps --target <dir> <wheel>`, then put `<dir>` on `PYTHONPATH`. Each learner
rank with a flash attention compares it with sdpa once at the start and uses sdpa when they do not agree (`plearn/method/README.md`).

## Node.js (Pro-Worlds)

The Pro-Worlds engine is JavaScript (`benchmarks/proworlds/games/expansion-6.1`). Install Node.js 22 or later. plearn uses
`node` on `PATH`, or the binary in `NODE_BIN`. Check the frozen engine:

```sh
cd benchmarks/proworlds/games/expansion-6.1 && sha256sum -c SHA256SUMS && node test.mjs --fixtures
```

## ManiSkill3 (tabletop)

The three tabletop tasks render 3D pictures with ManiSkill3 off-screen (Vulkan; EGL on headless GPU hosts). The renderer needs
a GPU with a Vulkan driver.

## RoboQuest

RoboQuest runs in its own simulator runtime (MuJoCo, robosuite, RoboCasa). Build it once with the benchmark's script, then set
`ROBOQUEST_RUNTIME`:

```sh
cd benchmarks/roboquest
scripts/bootstrap_simulator.sh --runtime ~/roboquest-runtime --assets download   # RoboCasa's kitchen assets: you download them
source ~/roboquest-runtime/env.sh                                                 # sets ROBOQUEST_RUNTIME
```

See `benchmarks/roboquest/README.md` for the disk space, the asset options and a check episode.

## Models and folders

| Variable | Use |
|---|---|
| `PL_MODELS` | the folder of the model weights: one folder per model, named as `local_dir` in `configs/models/*.yaml` (for example `Qwen3-VL-8B-Thinking`) |
| `QWEN35_4B_PATH` | optional: the Qwen3.5-4B weights folder (else `$PL_MODELS/Qwen3.5-4B`) |
| `PL_RUNS` | the output folder of the runs and the tests (default `runs`) |
| `PL_DATASETS` | the folder of downloaded datasets, when a task needs one |
| `PL_ENV_SH` | optional: a file that the training sources first (for example a conda activation) |
| `HF_TOKEN` | optional: for a private Hugging Face download |
| `WANDB_PROJECT`, `WANDB_ENTITY` | optional: W&B logging. With no project, nothing is logged |

Get the base models from Hugging Face, for example:

```sh
hf download Qwen/Qwen3-VL-8B-Thinking --local-dir $PL_MODELS/Qwen3-VL-8B-Thinking
hf download Qwen/Qwen3-VL-4B-Thinking --local-dir $PL_MODELS/Qwen3-VL-4B-Thinking
hf download Qwen/Qwen3.5-4B           --local-dir $PL_MODELS/Qwen3.5-4B
```

## Tests

Most tests run on a CPU:

```sh
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests/method -t .
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests/tasks -t .
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests/tools -t .
```

Some tests skip without a GPU, Node.js or `PL_MODELS`. The tabletop tests and the tests that play the tabletop tasks need a GPU
renderer.
