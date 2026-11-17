"""RoboQuest tasks at the decision level, for the full-benchmark test.

Each task is a scene of the benchmark (its eval50 instances) that the agent works through named helper actions. The helpers
move objects exactly with the task's own privileged gates: control is assumed perfect. An evidence action returns a status
(what the robot knows from its own action) and then what the action shows: one sentence in the text version (the reading),
or the benchmark's three policy-camera images in the simulator version. An answer (SUBMIT <answer>) runs the answer helpers,
lets the scene settle and is scored by the task's own success check (_current_score) and by the benchmark's progress P (see
base.py). Nothing presses the Submit button, so a result is a decision-level result, not an official benchmark episode. The
protocol: ../README.md.

This package runs inside the RoboQuest runtime interpreter (plearn/tasks/roboquest/bench_sim.py), never in the trainer.
"""
import importlib

TASKS = ("odd_parcel", "painted_cubes", "marked_mugs", "unfamiliar_containers", "search_room", "blackout_search",
         "locked_storage", "puzzle_box", "stamps", "wobbly_stand")


def adapter(task):
    """The adapter class of a task: plearn.tasks.roboquest.bench.<task>.Task."""
    return importlib.import_module(f"plearn.tasks.roboquest.bench.{task}").Task
