---
title: RLVB
emoji: 🧭
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
short_description: Lazy agents answer against their own beliefs
---

# Lazy Agents Answer Against Their Own Beliefs: Reinforcement Learning with Verifiable Beliefs Teaches Them to Act on What They Believe

Richmond Sin Jing Xuan, Peilin Feng, Wenbin Wang, Soujanya Poria. DeCLaRe Lab, Nanyang Technological University. Code: https://github.com/declare-lab/RLVB

Project page of the paper. Each game of HiddenGames, Pro-Worlds and RoboQuest has one test episode played by eight methods
(seven of the main table and RLVB trained with the belief term) (Qwen3-VL-4B, seed 0). The page explains how each game is played and replays every method step by step: the
frame the agent saw, its belief, the exact posterior from the simulator, its action and the environment's reply.

Static site: `index.html`, `static/` (the style, the explorer in `explorer.js`, the result charts and the hero animation of the idea in
`page.js`, the animation of the three rewards in `credit.js`, the figures), `data/` (one JSON per game, plus `games.json`), `frames/`
(the rendered steps), `thumbs/` (the small pictures of the games), `boards/` (the whole board of each Pro-Worlds case).

To view it on your machine, run `python3 -m http.server 8765 --directory site` and open http://localhost:8765.

To publish it as a Hugging Face Space (static SDK), from the folder above `site/`:

```python
from huggingface_hub import HfApi   # pip install -U huggingface_hub; then log in with: hf auth login
api = HfApi()
api.create_repo("posterior-learning/RLVB", repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(folder_path="site", repo_id="posterior-learning/RLVB", repo_type="space")
```
