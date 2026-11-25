"""The teachers' module switches (the trainer's module text). Each False is the fixed teacher (the default); True gives
the values of every run before the fix. The trainer sets them from its flags."""
# FL_VIEW_VALUE (--fl_view_value): True FrozenLake's teacher values a move by the view it opens, as every run before the fix
# did; False (the default) by the return it leads to (plearn/tasks/frozenlake/teacher.py, the trainer's module text).
FL_VIEW_VALUE = False
# SK_GUESS_VALUE (--sk_guess_value): True Sokoban's teacher values heading for a target as a guess among the candidates,
# as every run before the fix did; False (the default) a move by the return it leads to (plearn/tasks/sokoban/teacher.py).
SK_GUESS_VALUE = False
# PLAN_PAST_LIMIT (--plan_past_limit): True the MiniGrid and MineSweeper lookaheads value plans that end after the game's
# action limit, as every run before the fix did; False (the default) such a plan is not valued (plearn/tasks/minigrid/teacher.py,
# plearn/tasks/minesweeper/teacher.py; the trainer's module text).
PLAN_PAST_LIMIT = False
# TEST_MOVES, TEST_TURNS (25 Nov 2026): every test episode has 150 moves (actions, evidence actions or engine moves) and 152
# turns, in every task (RoboQuest, Pro-Worlds, HiddenGames), so that no agent hits the move limit. The training and validation
# splits keep each task's own limit (the limits of the paper's training). The paper's test tables used each task's own limit:
# PL_TEST_MOVES=0 gives that back (each task's own moves and turns in the test split too, as in the paper).
import os as _os
TEST_MOVES = int(_os.environ.get("PL_TEST_MOVES", "150")); TEST_TURNS = TEST_MOVES + 2 if TEST_MOVES else 0
