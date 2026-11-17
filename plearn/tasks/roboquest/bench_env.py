"""The full-benchmark test of RoboQuest as registered tasks (roboquest_text_<task>, group roboquest_text; README.md beside this file): one instance of
one task per episode, at the decision level with helpers (bench/, bench_sim.py). The episode is the test's own
(bench_eval.Episode: its prompt, its helper actions, its budget); this class gives it the evaluator's contract
(plearn/api/episode.py) and a shared pool of simulator bridges. With the evaluation's output folder, each episode keeps the
benchmark runner's private files in <out_dir>/episodes/<task>-<instance>/, and its row carries the benchmark's progress P.

env_args: task (a RoboQuest task), version (sim: the three policy-camera pictures, and for Odd Parcel with RQ_DIAL_ZOOM=1 the
reading aid of README.md; text: one sentence per action, the version of test_suite), bridges (simulator processes shared by all episodes, default 4), render_gpu (a physical GPU index; default RQ_GPU, else the
first GPU of CUDA_VISIBLE_DEVICES, the process's own: the trainer's actor or plearn eval's server, never GPU 0 by default),
dial_zoom (true: the Odd Parcel reading aid of README.md in the sim version; no registered task sets it). The aid is read from
RQ_DIAL_ZOOM=1 in this process and in its simulator bridges, so dial_zoom sets that variable in specs(), which plearn eval calls for
every task before the first episode starts a bridge; the variable changes only Odd Parcel in the sim version. An episode with
dial_zoom refuses to start when the variable is not set, or when the bridges started before it was set. one_line (true: the
one-line turn, the action-only test of a model, for example the base model's "Base" row: the system text without ANCHOR_HINT, and the evaluator
samples the action line right after the thought, with no belief line; plearn/api/episode.py); default false: the two-line turn
of RLVB (EM). Set it with --task_args roboquest.one_line=1.
Splits: test = the eval50 instances in their file's order; dev = the registry's development rows.
"""
from __future__ import annotations
import json, os, threading
from pathlib import Path
from plearn.api.episode import carried

SUITE = Path(__file__).resolve().parents[3] / "benchmarks" / "roboquest" / "suite" / "v1"
_POOL, _LOCK, _SID = [], threading.Lock(), [0]
_POOL_ZOOM = []                                       # RQ_DIAL_ZOOM of the process when the pool's first bridge started


def _bridge(n, gpu, log):
    """A bridge of the shared pool (started on first use) and a new session id."""
    from plearn.tasks.roboquest.bench_eval import Bridge
    with _LOCK:
        if not _POOL: _POOL_ZOOM[:] = [os.environ.get("RQ_DIAL_ZOOM") == "1"]
        while len(_POOL) < n:
            os.makedirs(os.path.dirname(log) or ".", exist_ok=True); _POOL.append(Bridge(gpu, log))
        _SID[0] += 1
        return _POOL[_SID[0] % len(_POOL)], _SID[0]


class Episode:
    @classmethod
    def specs(cls, split, n, seed, task, version="sim", dial_zoom=False, **_):
        if dial_zoom and task == "odd_parcel" and version == "sim":
            os.environ["RQ_DIAL_ZOOM"] = "1"              # before any bridge starts (the module text)
        if split == "test":
            ids = open(SUITE / "eval50" / f"{task}.txt").read().split()
        else:
            rows = json.load(open(SUITE / task / "registry" / "registry.json"))["instances"]
            ids = [r["instance_id"] for r in rows if r.get("split") == split]
            if not ids: raise ValueError(f"RoboQuest {task}: no instances in split {split!r} (test, dev)")
        return [[split, i] for i in ids[:n]]

    def __init__(self, spec, task, version="sim", bridges=4, render_gpu=None, out_dir=None, dial_zoom=False, one_line=False):
        from plearn.tasks.roboquest.bench_eval import Episode as Test
        _, iid = spec
        if dial_zoom and task == "odd_parcel" and version == "sim":
            if os.environ.get("RQ_DIAL_ZOOM") != "1" or (_POOL_ZOOM and not _POOL_ZOOM[0]):
                raise RuntimeError("dial_zoom: RQ_DIAL_ZOOM=1 must be set before the simulator bridges start (call specs() first)")
        gpu = render_gpu if render_gpu is not None else (os.environ.get("RQ_GPU") or os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        bridge, sid = _bridge(int(bridges), gpu, os.path.join(out_dir or "runs", "roboquest-bridges.log"))
        ep_dir = os.path.join(out_dir, "episodes", f"{task}-{iid}") if out_dir else None
        self.task, self.version, self.note, self.one_line = task, version, None, bool(one_line)
        self.e = Test(task, iid, sid, bridge, 0, ep_dir)
        self.e.open(version == "sim")

    @property
    def done(self): return self.e.result is not None

    def messages(self, carry):
        from plearn.tasks.roboquest.roboquest import SYSTEM, ANCHOR_HINT
        from plearn.tasks.roboquest.bench_eval import READING_RULE
        parts, images = self.e.content(self.version)
        if self.task == "odd_parcel" and self.version == "sim" and os.environ.get("RQ_DIAL_ZOOM") == "1":
            parts[0]["text"] += READING_RULE              # the reading aid (README.md): the rule right after the task's own text, as in bench_eval.main
        if carry and self.note:                       # the agent's own last belief line, before the closing line
            if isinstance(parts, str):
                head, sep, tail = parts.rpartition("\nEvidence actions used:")
                parts = head + carried(self.note) + sep + tail
            else:
                parts.insert(len(parts) - 1, {"type": "text", "text": carried(self.note)})
        return (SYSTEM if getattr(self, "one_line", False) else SYSTEM + ANCHOR_HINT), parts, images

    def step(self, belief, action, record):
        self.e.turns.append(dict(record, belief=belief, action=action))
        self.e.step(action)
        self.note = belief

    def row(self):
        e, r = self.e, self.e.result or {}
        n = max(1, len(e.turns)); outcome = r.get("outcome")
        return {"correct": bool(r.get("success")), "success": float(bool(r.get("success"))), "P": r.get("P"),
                "submitted": float(outcome not in ("no_answer", "invalid_answer")), "outcome": outcome,
                "errors": r.get("errors", []), "stages": r.get("stages"), "deduction": r.get("deduction"),
                "evidence_actions": e.t, "T": e.T, "decisions": e.decisions,
                "thought_cut": sum(bool(t.get("thought_cut")) for t in e.turns) / n,
                "actions": [dict(action=x["action"], status=x["status"], reading=x["reading"]) for x in e.log], "turn_log": e.turns}

    def close(self):
        try: self.e.bridge.call(op="close", sid=self.e.sid)
        except Exception: pass
