"""Pro-Worlds: the agent's turn (expansion_env.py). The module keeps the condition full-world of the earlier Pro-Worlds
tasks; the registered tasks play the condition window.

The prompt holds the task's public fields (name, question, rules, registration), the answers as numbers, the reference pictures
at their native size, and the observations: full-world, the complete world as one picture; window, the start picture and the
picture after every move, the window's position (Integration), the agent's belief line of its last turn, the moves used and
allowed, and the turns used. The agent writes its reasoning, then two lines: 'Possible: <answer numbers>' and 'ACTION: MOVE
UP|DOWN|LEFT|RIGHT' or 'ACTION: SUBMIT <answer number or label>' (full-world: only SUBMIT). The action-only turn (one_line,
for example the base model's "Base" row: --task_args proworlds.one_line=1) has the action line alone.
"""
from __future__ import annotations
import base64, io, json, re
from plearn.api.episode import carried

CONDITIONS = ("full-world", "window")
DIRS = ("UP", "RIGHT", "DOWN", "LEFT")                              # the environment's actions 0 to 3; an answer k is 4 + k
ACT = re.compile(r"^\s*ACTION:\s*(?:MOVE\s+(UP|DOWN|LEFT|RIGHT)|SUBMIT\s+(.+?))\s*\.?\s*$", re.I)
# native pictures: the largest picture of the benchmark (1152 x 1152) is not made smaller
NATIVE_SIZE = json.dumps({"shortest_edge": 32 * 32, "longest_edge": 1152 * 1152})


def picture(url):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


def norm(w): return " ".join(str(w).replace("−", "-").replace("→", "->").upper().split())


def label_keys(labels):
    """The words that name answer i: its number i + 1 (as the prompt lists it) and its label."""
    keys = {}
    for i, l in enumerate(labels):
        keys[str(i + 1)] = i; keys.setdefault(norm(l), i)
    return keys


def listed(labels):
    """The answers as the prompt lists them: '1', '2', ... for 'Option k' labels, else '1 = <label>'."""
    return [str(i + 1) if norm(l) == f"OPTION {i + 1}" else f"{i + 1} = {l}" for i, l in enumerate(labels)]


def system(condition, labels, one_line=False):
    """The system text. one_line (the action-only turn): the action line alone, no belief line."""
    if one_line:
        head = "You answer a question about a hidden world from pictures. Each turn, think, then reply with exactly one line.\n"
        if condition == "full-world":
            return head + "The line: 'ACTION: SUBMIT <answer number>'. The complete world is visible, so answer now."
        return head + ("The line: 'ACTION: MOVE UP', 'ACTION: MOVE DOWN', 'ACTION: MOVE LEFT' or 'ACTION: MOVE RIGHT' to move the window, "
                       "or 'ACTION: SUBMIT <answer number>' to answer. You have one chance to answer: submit when the evidence identifies "
                       "the answer, and not before.")
    ex = "1, 3" if len(labels) > 2 else "1, 2"
    head = ("You answer a question about a hidden world from pictures. Each turn, think, then reply with exactly two lines.\n"
            f"Line 1: 'Possible: ' and the numbers of the answers that the evidence still allows, for example "
            f"'Possible: {ex}', or 'Possible: all' when no answer is excluded yet.\n")
    if condition == "full-world":
        return head + "Line 2: 'ACTION: SUBMIT <answer number>'. The complete world is visible, so answer now."
    return head + ("Line 2: 'ACTION: MOVE UP', 'ACTION: MOVE DOWN', 'ACTION: MOVE LEFT' or 'ACTION: MOVE RIGHT' to move the window, or "
                   "'ACTION: SUBMIT <answer number>' to answer. You have one chance to answer: submit when the evidence identifies the answer, "
                   "and not before.")


def has_catalogue(task):
    """An Integration task's rules end with the answers' catalogue in text (the two motifs as rows of 1 and 0, every answer as one
    letter per block), the same information as its sixteen reference pictures."""
    return task["id"].startswith("integration/")


def user_parts(task, obs, condition, refs, pics, moves_log, note, turns_used, limit, ref_pictures=True):
    """The user message of one turn: its parts (text and image places) and the pictures in order. task: the first observation's
    'task' fields; obs: the current observation; refs: the reference pictures; pics: the start picture and one after every move,
    whose directions are moves_log; note: the agent's belief line of its last turn (None: not shown). ref_pictures=False leaves
    out the reference pictures of a task whose rules hold the answers' catalogue (has_catalogue)."""
    names = listed(task["labels"])
    parts = [{"type": "text", "text": f"{task['name']}: {task['question']}\n{task['rules']}\n{task.get('registration', '')}\n"
                                      f"The answers: {', '.join(names)}."}]
    images = []
    if not ref_pictures and has_catalogue(task): refs = []
    for i, im in enumerate(refs):
        parts += [{"type": "text", "text": f"\nAnswer {names[i]}, its reference picture: "}, {"type": "image"}]; images.append(im)
    if condition == "full-world":
        parts += [{"type": "text", "text": f"\nThe complete world, {obs['size']} x {obs.get('height', obs['size'])} cells: "}, {"type": "image"}]
        images.append(pics[0])
        tail = f"\nTurns used: {turns_used} of {limit}. Decide now."
    else:
        b = obs.get("sensingBudget") or {}
        # Stitch: the first crop is at the top left corner; Integration: a block corner, and the crop bounds are public
        start = "window that starts at the top left corner" if task["id"].startswith("stitch/") else "window; the line 'Window position' gives its place"
        rule = (f"\nYou see the world, {obs['size']} x {obs.get('height', obs['size'])} cells, only through a {obs['window']} x {obs['window']} "
                f"{start}. A move shifts it {obs['stride']} cells. At most {obs['budget']} moves"
                + (f"; the moves may show at most {b['maxUniqueCells']} of the {b['totalCells']} cells" if b.get("maxUniqueCells") else "")
                + ".\nThe pictures so far:\nStart: ")
        parts += [{"type": "text", "text": rule}, {"type": "image"}]; images.append(pics[0])
        for m, im in zip(moves_log, pics[1:]):
            parts += [{"type": "text", "text": f"\nAfter MOVE {m}: "}, {"type": "image"}]; images.append(im)
        allowed = [d for i, d in enumerate(DIRS) if obs["actionMask"][i]]
        tail = (f"\nWindow position: {obs['bounds']}." if obs.get("bounds") else "") + \
               carried(note) + \
               f"\nMoves used: {obs['moves']} of {obs['budget']}. Moves allowed now: {', '.join(allowed) if allowed else 'none'}. " \
               f"Turns used: {turns_used} of {limit}. Decide now."
    parts.append({"type": "text", "text": tail})
    return parts, images


def read_action(action, keys, condition):
    """('move', a) with the environment's action a, ('answer', k) with the answer index k, or ('unreadable', None). A move in the
    full-world condition, or an answer that no number or label names, is unreadable."""
    m = ACT.match(action)
    if m and m.group(1) and condition == "window": return "move", DIRS.index(m.group(1).upper())
    if m and m.group(2):
        k = keys.get(norm(m.group(2)).rstrip("."))
        if k is not None: return "answer", k
    return "unreadable", None
