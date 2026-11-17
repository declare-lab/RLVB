# mastermind

Mastermind with TextArena's rules, in text: find a code of 4 different digits from 1 to 6 from black and white pegs. One of
the 8 HiddenGames tasks: `train_suite` plays its train split, `test_suite/hidden_games` its test split.

| | |
|---|---|
| Source | TextArena's Mastermind: `benchmarks/textarena` (TextArena/TextArena @ 1e3160d, MIT); the PyPI release 0.7.4 has the same rules and text. plearn keeps its feedback function and its rules text; `tests/tasks/test_games.py` checks both against that copy. |
| Hidden target θ | the code: 4 different digits from 1 to 6 (360 codes), drawn as TextArena draws it (`random.sample`) |
| Evidence action | `ACTION: GUESS [a b c d]`: a guess not made before; TextArena's feedback (black and white pegs) |
| Answer | `ACTION: SUBMIT [a b c d]`: right when it is the code; it ends the game. A guess equal to the code only shows 4 black pegs. |
| Observation | TextArena's rules text (two changes: `benchmarks/textarena/README.md`) and the feedback of every guess |
| Budget | 20 turns |
| Splits | test: seeds 20000 + i (100 episodes); validation: seeds 10000 + i; training: seeds from 10^6 |
| Teacher | the exact belief: the codes that agree with every feedback. π* looks one guess ahead. |
| Belief line | `Possible: <codes>` when at most 6 codes remain, else the digits that each position allows |
| Our changes | the answer line `ACTION: SUBMIT` (a guess does not end the game), the two rules-text changes, the splits, the exact teacher and the belief line |
| Teacher's own play | validation, 128 episodes (`tools/teacher_play.py`): most likely decision 1.000 (3.2 guesses); sampled from π*, as in training, 1.000 (3.5 guesses) |

Files: `_default.yaml` (the template), `mastermind.yaml` (the task), `teacher.py` (the episode and the teacher; the
episode is `plearn.tasks.domain:DomainEpisode`).
