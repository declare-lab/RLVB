"""The interface every task adapter fills in (see the package docstring).

What the agent gets, in both versions: the benchmark's own goal text, an intro that names the objects it can act on and
states only what the cameras show at the start (never a hidden fact), the action grammar, and after each action a status
line. The text version adds the reading of the action; the simulator version adds the three camera images instead.

With record(), an episode also keeps the private files of the benchmark's runner (config.json, task-spec-private.json,
physical-trace-private.jsonl, result.json), and the benchmark's own progress code (roboquest/scoring/progress.py) scores it:
P in [0, 1] beside the success. SUBMIT freezes the score as the Submit press does; an episode without SUBMIT is scored on
its end state with no success, as an episode without a press.
"""
import json, os, re
from copy import deepcopy
import numpy as np

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")      # the benchmark's policy cameras
SIZE = 512
TRACE = "physical-trace-private.jsonl"


def _plain(o):
    """numpy values in the JSON files."""
    return o.tolist() if hasattr(o, "tolist") else str(o)


class Invalid(Exception):
    """An action the helpers cannot carry out: unknown words, an object that is not there, an answer in the wrong form."""


def words(line):
    """An action line in canonical form: upper case, single spaces, no 'ACTION:' prefix, no final full stop."""
    t = " ".join(str(line).replace(",", " ").split()).strip().rstrip(".").upper()
    return t[len("ACTION:"):].strip() if t.startswith("ACTION:") else t


def numbers(text):
    return [int(x) for x in re.findall(r"\d+", text)]


class Task:
    T = 8                    # evidence actions an episode may use
    SETTLE = 50              # zero-action ticks before the score is read

    def __init__(self, env, render):
        self.env, self.render = env, render
        self._trace = None

    # ---- what the agent is told ------------------------------------------------------------------------------------
    def goal(self):
        """The benchmark's own instruction, and how the decision level replaces its motor steps."""
        return (self.env.PUBLIC_GOAL + " You act through helper actions; the helpers move objects exactly. "
                "Instead of pressing Submit, give your answer with SUBMIT.")

    def intro(self):
        """The objects the agent can name, and what the cameras show of them at the start."""
        raise NotImplementedError

    def help(self):
        """The action grammar: one line per evidence action, and the SUBMIT line."""
        raise NotImplementedError

    # ---- actions ---------------------------------------------------------------------------------------------------
    def act(self, line):
        """Carry out one evidence action. Returns dict(status=str, reading=str, images=list | None); raises Invalid."""
        raise NotImplementedError

    def submit(self, answer):
        """Carry out the answer, settle, and return the task's own score (a dict with 'success'); raises Invalid."""
        raise NotImplementedError

    # ---- helpers ---------------------------------------------------------------------------------------------------
    def tick(self, n):
        action = np.zeros(self.env.action_dim)
        for _ in range(n):
            self.env.step(action)
            if self._trace is not None:
                self._steps += 1; self._trace_line()

    def views(self):
        """The three policy-camera images now (simulator version), else None."""
        if not self.render:
            return None
        return [np.ascontiguousarray(self.env.sim.render(camera_name=c, width=SIZE, height=SIZE)[::-1]) for c in CAMERAS]

    def score(self):
        self.tick(self.SETTLE)
        s = self.env._current_score()
        out = {"success": bool(s.get("success")), "outcome": s.get("outcome"), "errors": list(s.get("errors", []))}
        if self._trace is not None:                   # frozen as at the Submit press (roboquest/base/submit_button.py, kitchen.py)
            env = self.env
            out.update(self._finish(dict(deepcopy(s), submitted=True, termination="helper_submit", submission_step=self._steps,
                                         outcome="success" if s.get("success") else "wrong_submit",
                                         progress=env._progress_value(env.task_progress()))))
        return out

    # ---- the benchmark's progress P --------------------------------------------------------------------------------
    def record(self, ep_dir, task, iid):
        """Keep the episode's private files as the benchmark's runner does. The trace has one line per tick: the evaluator
        snapshot on the lines the progress code reads (every SAMPLE_EVERY-th, and the last), the step alone on the others."""
        os.makedirs(ep_dir, exist_ok=True)
        with open(os.path.join(ep_dir, "config.json"), "w") as f:
            json.dump({"task": f"roboquest_{task}@{iid}", "protocol": "decision level with helpers"}, f)
        with open(os.path.join(ep_dir, "task-spec-private.json"), "w") as f:
            json.dump(self.env.task_spec, f, default=_plain)
        self._ep, self._lines, self._steps = ep_dir, 0, 0
        self._trace = open(os.path.join(ep_dir, TRACE), "w")

    def _trace_line(self, score=None):
        from roboquest.scoring.progress import SAMPLE_EVERY
        if score is None and self._lines % SAMPLE_EVERY:
            row = {"step": self._steps}
        else:
            snap = self.env.evaluator_snapshot(self._steps)
            if score is not None:
                snap["score"] = score
            s = deepcopy(snap["score"]); s["success"] = bool(s.get("success")) and bool(s.get("submitted"))   # FirstSubmitEvaluator
            row = {"step": self._steps, "sim_time": float(self.env.sim.data.time), "score": s, "snapshot": snap}
        self._trace.write(json.dumps(row, default=_plain) + "\n"); self._lines += 1

    def _finish(self, score):
        """The last trace line with the episode's score, result.json, then the benchmark's progress (task-progress.json)."""
        from roboquest.scoring.progress import write
        self._trace_line(score); self._trace.close(); self._trace = None
        with open(os.path.join(self._ep, "result.json"), "w") as f:
            json.dump({"score": score, "completed": True, "protocol": "decision level with helpers"}, f, default=_plain)
        p = write(self._ep)
        return {"P": p.get("P"), "stages": p.get("stages"), "deduction": p.get("deduction"), "flags": p.get("flags")}

    def finish(self):
        """An episode that ended without SUBMIT: scored on its end state, as the benchmark scores an episode without a press."""
        if self._trace is None:
            return {"success": False}
        self.tick(self.SETTLE)
        return {"success": False, **self._finish(self.env.evaluate_success())}

    def camera_x(self, pos_world, camera=CAMERAS[0]):
        """Horizontal position of a world point in a camera's view (larger is further right), to number objects."""
        m = self.env.sim.model; d = self.env.sim.data; cid = m.camera_name2id(camera)
        rel = np.asarray(pos_world, float) - d.cam_xpos[cid]
        return float(rel @ d.cam_xmat[cid].reshape(3, 3)[:, 0])
