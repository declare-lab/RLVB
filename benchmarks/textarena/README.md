# TextArena: Mastermind

The Mastermind environment of [TextArena](https://github.com/TextArena/TextArena) (MIT, see `LICENSE`), at commit
[1e3160d](https://github.com/TextArena/TextArena/tree/1e3160dea659971f120bf42708fafa8ea3001b81). The PyPI
release 0.7.4 has the same rules, feedback and text.

This copy is for reference: plearn does not import it, because `env.py` needs TextArena's core. The plearn dataset
`mastermind` (`plearn/tasks/mastermind/teacher.py`) keeps two parts of it, and `tests/tasks/test_games.py` checks both
against these files:

- the feedback: `_evaluate_guess` in `env.py` (black and white pegs);
- the rules text: `player_prompt.intro` in `locales/en.json`, with two changes. Our action lines replace its answer-format
  line, and "turns to find the code" replaces "turns to guess the code", because a guess does not end our game (the answer
  line `ACTION: SUBMIT [a b c d]` ends it).

The code is drawn as TextArena draws it: `random.sample` of the digits 1 to 6, so 4 different digits (360 codes).
