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
