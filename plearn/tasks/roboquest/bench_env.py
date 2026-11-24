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
of RLVB (EM). Set it with --task_args roboquest.one_line=1. final_submit (true; default false): an episode that has used
its T + 2 decisions without a SUBMIT gets one more turn that asks for the SUBMIT alone (--task_args roboquest.final_submit=1). only (a file of lines '<task> <instance id>'): the test plays this task's listed instances alone (--task_args roboquest.only=<file>). replay (a test's turns file,
logs/test-outputs/<run>.turns.csv.gz): each episode first carries out its logged turns in the simulator, without the
model, and the model plays only what is left; the row gets replayed_turns and replay_mismatch (simulator replies that differ from
the logged feedback). With final_submit, an episode that the log left without a SUBMIT gets its one turn for the SUBMIT. final_submit=prefill: in that
turn the model does not think again; its own last belief line is kept and its reply is started as 'ACTION: SUBMIT', so that it
writes only the answer (plearn/api/episode.py submit_turn).
Splits: test = the eval50 instances in their file's order; dev = the registry's development rows.
"""
from __future__ import annotations
import json, os, threading
from pathlib import Path
from plearn.api.episode import carried

SUITE = Path(__file__).resolve().parents[3] / "benchmarks" / "roboquest" / "suite" / "v1"
_POOL, _LOCK, _SID = [], threading.Lock(), [0]
_POOL_ZOOM = []                                       # RQ_DIAL_ZOOM of the process when the pool's first bridge started
_REPLAY = {}                                          # replay: turns file -> {(task, instance id): its logged turns in order}


def _logged(path, task, iid):
    """The logged turns of one RoboQuest episode of a test's turns file (logs/test-outputs/<run>.turns.csv.gz), in order."""
    import csv, gzip, io
    with _LOCK:
        if path not in _REPLAY:
            eps = {}
            for x in csv.DictReader(io.TextIOWrapper(gzip.open(path), encoding="utf-8")):
                if x["benchmark"] != "RoboQuest" or x.get("kind") == "error": continue
                t = x["task"].replace("roboquest_text_", "").replace("roboquest_sim_", "").replace("roboquest_", "")
                eps.setdefault((t, json.loads(x["spec"])[1]), []).append(x)
            _REPLAY[path] = {k: sorted(v, key=lambda x: int(x["turn"])) for k, v in eps.items()}
        return _REPLAY[path].get((task, iid), [])


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
    def specs(cls, split, n, seed, task, version="sim", dial_zoom=False, only=None, **_):
        if dial_zoom and task == "odd_parcel" and version == "sim":
            os.environ["RQ_DIAL_ZOOM"] = "1"              # before any bridge starts (the module text)
        if split == "test":
            ids = open(SUITE / "eval50" / f"{task}.txt").read().split()
        else:
            rows = json.load(open(SUITE / task / "registry" / "registry.json"))["instances"]
            ids = [r["instance_id"] for r in rows if r.get("split") == split]
            if not ids: raise ValueError(f"RoboQuest {task}: no instances in split {split!r} (test, dev)")
        ids = ids[:n]
        if only:                                          # only: a file of lines '<task> <instance id>'; this task's listed instances alone
            keep = {l.split()[1] for l in open(only) if l.strip() and not l.startswith("#") and l.split()[0] == task}
            ids = [i for i in ids if i in keep]
        return [[split, i] for i in ids]

    def __init__(self, spec, task, version="sim", bridges=4, render_gpu=None, out_dir=None, dial_zoom=False, one_line=False,
                 final_submit=False, only=None, replay=None):       # only: read by specs(); the episode does not use it
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
        self.e.final_submit = bool(final_submit)                    # one more turn for the SUBMIT alone (bench_eval.Episode; off by default)
        self.prefill = str(final_submit) == "prefill"               # ... without a new thought: the reply is started as 'ACTION: SUBMIT'
        self.e.open(version == "sim")
        self.replayed, self.replay_mismatch = 0, []
        if replay: self._replay(_logged(replay, task, iid))

    def _replay(self, turns):
        """replay: carry out the logged turns of this episode in the simulator, without the model, so that the
        episode stands where the logged test left it; the model then plays what is left (with final_submit: the one turn for the
        SUBMIT). Every reply of the simulator is compared with the logged feedback ('<action>: <status> <reading>'); a difference
        is kept in replay_mismatch (the episode goes on)."""
        norm = lambda t: " ".join((t or "").split())
        for x in turns:
            if self.done: break
            n = len(self.e.log)
            self.step(x.get("belief") or None, x["action"], {"replayed": True})
            self.replayed += 1
            if len(self.e.log) > n and x.get("feedback"):
                y = self.e.log[-1]; got = f"{y['action']}: " + " ".join(z for z in (y["status"], y["reading"]) if z)
                if norm(got) != norm(x["feedback"]): self.replay_mismatch.append({"turn": int(x["turn"]), "logged": x["feedback"][:300], "now": got[:300]})

    @property
    def done(self): return self.e.result is not None

    @property
    def submit_only(self):
        """final_submit=prefill, in the one extra turn: the evaluator takes plearn.api.episode.submit_turn (no new thought)."""
        return bool(getattr(self, "prefill", False) and self.e.last_call and not self.done)

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
                **({"replayed_turns": self.replayed, "replay_mismatch": self.replay_mismatch} if getattr(self, "replayed", 0) else {}),
                "thought_cut": sum(bool(t.get("thought_cut")) for t in e.turns) / n,
                "actions": [dict(action=x["action"], status=x["status"], reading=x["reading"]) for x in e.log], "turn_log": e.turns}

    def close(self):
        try: self.e.bridge.call(op="close", sid=self.e.sid)
        except Exception: pass
