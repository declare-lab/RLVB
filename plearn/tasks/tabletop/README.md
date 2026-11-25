# tabletop

Read the state of objects in 3D pictures: ManiSkill3's table scene with no robot. The agent acts through helpers that move
objects exactly (as RoboQuest at the decision level) and sees two camera pictures after each action. The three tasks are our
own; `tabletop_lamps`, `tabletop_cups` and `tabletop_blocks` are 3 of the 7 HiddenGames test tasks: `train_suite` plays their train
split, `test_suite/hidden_games` their test split.

| | |
|---|---|
| Source | ManiSkill3 (pip `mani-skill==3.0.1`, SAPIEN 3.0.3; rendered off-screen with Vulkan): its table scene (`TableSceneBuilder`) with no robot, and our objects as kinematic actors (`scene.py`). The tasks, the objects, the helpers and the teacher are ours. |
| Tasks | `tabletop_lamps`, `tabletop_cups` and `tabletop_blocks`; group `tabletop`. Their test split is an in-distribution test (in `test_suite/hidden_games`). |
| Hidden target θ | lamps: which of N = 4 or 5 buttons lights which of the N lamps (every mapping of the N! is equally likely); cups: which of N = 3 to 5 cups hides the object that the task names; blocks: which of N = 3 or 4 blocks stands on a slippery base |
| Evidence action | the helpers: `PRESS <button>` (the button's lamp lights and stays lit), `LIFT <cup>` (the cup goes up and stays up), `PUSH <block>` (the block moves 3 cm away from the cameras, or 9 cm when it is the slippery one). Each helper can be used once. |
| Answer | `ACTION: SUBMIT A2 B1 C4 D3` (the whole mapping) or `ACTION: SUBMIT <number>`; one chance |
| Observation | two pictures (the left and the right camera, 512 × 512) at the start and after each action, the newest 8 pairs; the text names the actions, never what they showed |
| Budget | N actions (N − 1 always suffice) in N + 2 turns; the test split: 150 actions in 152 turns (`PL_TEST_MOVES=0`: the task's own); each helper still acts once |
| Splits | test: seeds 20000 + i (100 episodes a task), with colours that training never shows (purple, orange, cyan, pink and black for the named object and the look-alikes; other colours for the cups and the blocks); validation: seeds 10000 + i; training: seeds from 10^6 |
| Teacher | the exact belief: the hypotheses that agree with every picture so far, all equally likely. π*: the shared lookahead (`estimator.lookahead_Q`) over the helpers not used yet and their outcomes (what the pictures would show). Move cost 0.98. |
| Belief line | `Possible: A 2; B 1 or 3; C 1 or 3`, `Possible: cups 2, 4`, `Possible: blocks 1, 3`; `Possible: all` before any evidence |
| Teacher's own play | validation, 128 episodes each (`tools/teacher_play.py`): lamps 1.000 (2.5 actions), cups 1.000 (2.3), blocks 1.000 (2.0); the same when sampled from π* as in training |

Only the pictures show the evidence: which lamp glows now and did not glow before (a weak glow on grey bulbs); whether the object under a lifted cup has the named colour and shape
(the look-alikes differ in one of the two); and whether a pushed block moved one line or three lines across the table.

The reading is measured apart from the decisions (Image → Visual Facts → Belief → Action):

- Every evaluation compares each belief line after an action with the exact posterior of that moment (`Domain.reading`):
  `read_exact`, `read_wider` (every answer that the evidence allows, and more), `read_misread` (the exact set of another
  picture: a visual recognition error), `read_reasoning` (no picture's set: a belief reasoning error) and `read_unreadable`,
  as shares of the lines; `read_miss` (the line leaves out the true answer); `read_width` against `read_width_exact` (the
  answers the line keeps, and those the evidence keeps); `act_error` (an exact line, and not the teacher's decision).
- In training, each E-step sample's line has its reading kind (in the E-step dumps), and every update logs the shares by
  history (`estep_read_*`) and `estep_read_no_exact`: the histories where no sample reads exactly, which give the M-step no
  exact reading to learn from.

The renderer uses the first GPU of `CUDA_VISIBLE_DEVICES`, or in a torchrun rank the rank's own GPU (about 0.7 GB);
training and `python -m plearn eval` set it. Install with `pip install -e ".[tabletop]"`; the host needs the NVIDIA
driver's Vulkan support.

Files: `_default.yaml` (the template), one task file per family, `tabletop.yaml` (the group), `teacher.py` (the episode and
the teacher; the episode is `plearn.tasks.domain:DomainEpisode`), `scene.py` (the scene and the two cameras).
