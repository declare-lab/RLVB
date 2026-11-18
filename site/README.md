---
title: RLVB
emoji: 🧭
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
short_description: Step through test episodes of RLVB and seven baselines
---

# RLVB: Turning Agents from Lazy Guessers into Evidence Seekers with Verifiable Beliefs as Process Rewards

Richmond Sin Jing Xuan, Peilin Feng, Wenbin Wang, Soujanya Poria. DeCLaRe Lab, Nanyang Technological University. Code: https://github.com/declare-lab/RLVB

Project page of the paper. Each game of HiddenGames, Pro-Worlds and RoboQuest has one test episode played by the eight methods
of the main table (Qwen3-VL-4B, seed 0). The page explains how each game is played and replays every method step by step: the
frame the agent saw, its belief, the exact posterior from the simulator, its action and the environment's reply.

Static site: `index.html`, `static/` (style and the explorer), `data/` (one JSON per game, plus `games.json`), `frames/`
(the rendered steps), `boards/` (the whole board of each Pro-Worlds case). `tools/build_games.py` rebuilds `data/games.json`.

To publish it as a Hugging Face Space (static SDK), from the folder above `site/`:

```python
from huggingface_hub import HfApi   # pip install -U huggingface_hub; then log in with: hf auth login
api = HfApi()
api.create_repo("posterior-learning/RLVB", repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(folder_path="site", repo_id="posterior-learning/RLVB", repo_type="space")
```
