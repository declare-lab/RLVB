"""Any training domain of plearn/method/trainer.py (its Domain contract) as an episode of the evaluator (plearn/episode.py): the
domain's own episodes of a split (Domain(split=...).eval_specs), its prompt as the trainer builds it (the system text, the
observations with pictures or as text, the agent's last belief line), its actions and its score. A task with this env and the
same domain as its teacher is evaluated in exactly the format it is trained in.

The open line: a line that keeps every option open although the evidence has excluded some. With the log score
of the line, a wrong line cost 50 times the posterior mass it left out and a wider one only the log of its width, so the safe line
was the open one ('Possible: any'). An episode row reports open_line, the share of its turns with evidence (the teacher's exact
line is no longer its line at the start) whose line states what the teacher's line at the start states (the domain's
anchor_equal): a check of the line by the domain's own reading, the same for every domain that has belief_note and anchor_equal.

one_line (the action-only turn, for example the base model's "Base" row: --task_args <dataset>.one_line=1, for example
mastermind.one_line=1 tabletop.one_line=1): the system text asks for the action line alone (action_only), the observation
carries no belief line, and the evaluator samples the action line alone after the thought (plearn/api/episode.py turn). Not
set: no change.

env_args: domain (a module with a Domain class, e.g. plearn.tasks.mastermind.teacher), domain_args (its keyword arguments besides
split), one_line.
"""
from __future__ import annotations
import importlib, json, re, sys, threading
from plearn.api.episode import carried

_DOMAINS, _LOCK = {}, threading.Lock()
TWO_LINES = re.compile(r"reply with exactly two lines\.\nLine 1: .*?\nLine 2: ", re.S)


def action_only(text):
    """The task's rules with the action line only: '... reply with exactly two lines.\\nLine 1: <the belief line>\\nLine 2: <the
    action>' becomes '... reply with one line: <the action>'. The task text, the actions and the answer rule do not change."""
    out, n = TWO_LINES.subn("reply with one line: ", text)
    if n != 1: raise ValueError(f"the rules do not have the two-line protocol: {text[:200]!r}")
    return out


def domain(name, args, split):
    """One shared Domain for each (module, arguments, split), and its lock: the domain's calls run one at a time."""
    key = (name, json.dumps(args or {}, sort_keys=True), split)
    with _LOCK:
        if key not in _DOMAINS: _DOMAINS[key] = (importlib.import_module(name).Domain(split=split, **(args or {})), threading.Lock())
        return _DOMAINS[key]


def system_text(D, e):
    """The trainer's system text (plearn/method/trainer.py turn_prompt, the exact-line protocol)."""
    if hasattr(D, "system_text"): return D.system_text(e)
    mod = sys.modules[type(D).__module__]
    if hasattr(mod, "SYSTEM_EXACT"): return mod.SYSTEM_EXACT
    return mod.SYSTEM + (getattr(mod, "BELIEF_HINT", "") or "")


class DomainEpisode:
    @classmethod
    def specs(cls, split, n, seed, domain, domain_args=None, **_):
        D, lock = globals()["domain"](domain, domain_args, split)
        with lock: return [[split, list(s)] for s in D.eval_specs(n)]

    def __init__(self, spec, domain, domain_args=None, one_line=False):
        split, s = spec; self.one_line = bool(one_line)
        self.D, self.lock = globals()["domain"](domain, domain_args, split)
        with self.lock: self.e = self.D.start([tuple(s)])[0]
        e = self.e; e.t, e.done, e.correct, e.invalid, e.note = getattr(e, "t", 0), False, None, 0, None
        self.turns, self.answer, self.premature, self.limit = [], None, None, getattr(self.D, "turns", self.D.T + 2)
        D = self.D                                                    # the teacher's line before any action: it excludes nothing yet
        reads = hasattr(D, "belief_note") and hasattr(D, "anchor_equal") and getattr(D, "teacher", True)
        with self.lock: self.start_line = D.belief_note(e) if reads else None

    @property
    def done(self): return bool(self.e.done) or len(self.turns) >= self.limit

    def messages(self, carry):
        if self.one_line:                                             # the action line alone, no carried line (the module text)
            system, parts, images = self._messages(False)
            return action_only(system), parts, images
        return self._messages(carry)

    def _messages(self, carry):
        D, e = self.D, self.e; note = e.note if carry else None
        with self.lock:
            if getattr(D, "images", False):
                import inspect
                parts, images = D.user_content(e, note=note) if "note" in inspect.signature(D.user_content).parameters else D.user_content(e)
            else:
                text, images = D._history(e), []
                if note:                                              # the agent's own last belief line, before the closing request
                    head, sep, tail = text.rpartition(" Decide now.")
                    text = head + carried(note) + "\nDecide now." + tail if sep else text + carried(note)
                parts = text
            return system_text(D, e), parts, images

    def step(self, belief, action, record):
        from plearn.method.trainer import parse_action
        D, e = self.D, self.e
        with self.lock:
            B = D.belief(e) if hasattr(D, "reading") else None
            looks = D.look_strings(e) if e.t < D.T else {}
            kind, val = parse_action(D, e, action, {"looks": looks})
            if B is not None:                                         # the line against the evidence so far, and the action against it
                record = {**record, "reading": D.reading(B, belief, D.label(e)), "act_ok": bool(D.action_ok(B, kind, val))}
            if self.start_line is not None and not D.anchor_equal(D.belief_note(e), self.start_line):   # the evidence excludes something:
                record = {**record, "open": bool(D.anchor_equal(belief or "", self.start_line))}         # is the line still open?
            if kind == "answer":
                self.premature = (not D.belief(e).identified()) if getattr(D, "teacher", True) else None; self.answer = val   # a test-only level has no belief
                e.correct = bool(D.do_commit(e, val)); e.done = True
            elif kind == "look":
                D.do_look(e, val); e.t += 1
            else:
                e.t += 1; e.invalid += 1                                  # an action the environment cannot carry out
            if hasattr(D, "after_turn"): D.after_turn(e)
        e.note = belief
        self.turns.append({**record, "belief": belief, "action": action, "kind": kind})

    def row(self):
        e = self.e; n = max(1, len(self.turns)); op = [t["open"] for t in self.turns if "open" in t]
        return {**(self.reading_metrics() if hasattr(self.D, "reading") else {}),
                **({"open_line": sum(op) / len(op) if op else None} if self.start_line is not None else {}),
                "correct": bool(e.correct), "success": float(bool(e.correct)), "submitted": float(self.answer is not None),
                "answer": self.answer, "premature": None if self.premature is None else float(self.premature),
                "looks": sum(t["kind"] == "look" for t in self.turns), "turns": len(self.turns), "invalid_turns": e.invalid / n,
                "thought_cut": sum(bool(t.get("thought_cut")) for t in self.turns) / n, "turn_log": self.turns}

    def reading_metrics(self):
        """A domain with reading(): its belief lines after an action by kind (shares), the share that leaves out the true answer
        (misread rate), the lines' width against the exact width, and the share of turns with an exact line and a poor action."""
        fm = lambda xs: sum(xs) / len(xs) if xs else None
        rs = [t["reading"] for t in self.turns if t.get("reading") and t["reading"]["evidence"]]
        ok = [r for r in rs if r["width"] is not None]
        out = {f"read_{k}": fm([r["kind"] == k for r in rs]) for k in ("exact", "wider", "misread", "reasoning", "unreadable")}
        out.update({"read_miss": fm([r["miss"] for r in ok]), "read_width": fm([r["width"] for r in ok]),
                    "read_width_exact": fm([r["width_exact"] for r in ok]),
                    "act_error": fm([not t["act_ok"] for t in self.turns if t.get("reading") and t["reading"]["kind"] == "exact"])})
        return out

    def close(self):
        with self.lock: self.D.close([self.e])
