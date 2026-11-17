"""The training domain of a run (plearn/method/trainer.py --domain plearn.tasks.mixture): the registered tasks of train_tasks, each
with its teacher (the task file's 'teacher' class and 'teacher_args').

Tasks whose teacher is the same class train in one instance of it: their teacher_args are joined ('pairs' and 'levels' lists are
concatenated; every other argument must agree). With one class, Domain(...) is that instance itself, so the trainer sees the task family's
own domain. With several classes, Domain(...) is a Mixed domain: a training episode comes from one teacher (each teacher with an
equal share, or the shares of domain_args 'weights' by teacher name), it carries its teacher (e._teacher, e._teacher_name), and
every call of the trainer goes to that teacher. A plan names its teacher (plan['dom']); Mixed.sub(name) gives the teacher back.
A belief goes to the teacher whose belief class it is.

domain_args: tasks (a comma list or list of tasks and groups), weights ({teacher name: share}), and any argument for the
teachers (split, ...).
"""
from __future__ import annotations
import inspect, sys

from plearn.models import NATIVE_SIZE
from plearn.api.registry import RegistryError, registry, resolve
from plearn.api.task import Task
from plearn.api.episode import carried


def teachers_of(tasks):
    """[(teacher class, joined teacher_args)] of the tasks: one entry per class, in the order of the tasks."""
    reg = registry(); classes, args, order = {}, {}, []
    for name in reg.tasks(tasks):
        t = Task.load(name, reg)
        if not t.teacher: raise RegistryError(f"task {name} has no teacher: it cannot be trained on")
        cls = resolve(t.teacher, t.source); key = f"{cls.__module__}.{cls.__qualname__}"
        if key not in classes: classes[key], args[key] = cls, {}; order.append(key)
        a = args[key]
        for k, v in t.teacher_args.items():
            if k in ("pairs", "levels"): a.setdefault(k, []).extend(v)
            elif k in a and a[k] != v: raise RegistryError(f"task {name}: teacher_args {k}={v!r} differs from {a[k]!r}")
            else: a[k] = v
    return [(classes[k], args[k]) for k in order]


def teacher_of(tasks):
    """(teacher class, joined teacher_args) of tasks that share one teacher class."""
    found = teachers_of(tasks)
    if len(found) != 1: raise RegistryError(f"the tasks need {len(found)} teacher classes; use teachers_of")
    return found[0]


class Domain:
    images = True                     # the trainer reads these before it builds the domain; the launcher also passes images
    mm_size = NATIVE_SIZE
    name = "mixture"                  # the run's files: <name>-<tag>-seed<seed>, the same in the learner and the actor

    def __new__(cls, tasks, weights=None, **kw):
        found = teachers_of(tasks); kw.pop("images", None)
        if len(found) == 1:
            teacher, args = found[0]; return teacher(**{**args, **kw})
        return Mixed([teacher(**{**args, **kw}) for teacher, args in found], weights)


class Mixed:
    """Several teachers in one run (see the module text)."""
    look_name_takes_episode = True

    def __init__(self, teachers, weights=None):
        self.subs = list(teachers); self.names = []
        for d in self.subs:
            n = getattr(d, "name", type(d).__name__)
            if n in self.names: raise RegistryError(f"two teachers are named {n!r}")
            self.names.append(n)
        self.by_name = dict(zip(self.names, self.subs)); self.name = "+".join(self.names)
        self.images = any(getattr(d, "images", False) for d in self.subs)
        self.T = max(d.T for d in self.subs); self.lam = max(d.lam for d in self.subs)
        self.turns = max(getattr(d, "turns", d.T + 2) for d in self.subs)
        self.weights = [float((weights or {}).get(n, 1.0)) for n in self.names]
        self.fam = {}
        for n, d in zip(self.names, self.subs):
            for f in getattr(d, "families", [n]): self.fam[f] = n
        self.families = list(self.fam)
        self._btype = {}

    def sub(self, name): return self.by_name[name]
    def _of_spec(self, s):
        if s[0] not in self.fam: raise RegistryError(f"no teacher of the mixture makes episodes of {s[0]!r}")
        return self.fam[s[0]]

    # specs and episodes
    def sample(self, rng): return self.by_name[rng.choices(self.names, weights=self.weights)[0]].sample(rng)
    def eval_specs(self, n): return [s for d in self.subs for s in d.eval_specs(n)]
    def start(self, specs):
        out, groups = [None] * len(specs), {}
        for i, s in enumerate(specs): groups.setdefault(self._of_spec(s), []).append(i)
        for n, idx in groups.items():
            d = self.by_name[n]
            for i, e in zip(idx, d.start([specs[i] for i in idx])): e._teacher, e._teacher_name = d, n; out[i] = e
        return out
    def close(self, eps):
        groups = {}
        for e in eps: groups.setdefault(e._teacher_name, []).append(e)
        for n, es in groups.items(): self.by_name[n].close(es)
    def family_of(self, e):
        d = e._teacher; return d.family_of(e) if hasattr(d, "family_of") else e._teacher_name

    # observations: picture teachers give their parts; text teachers their text, with the agent's last belief line
    def system_text(self, e):
        d = e._teacher
        if hasattr(d, "system_text"): return d.system_text(e)
        mod = sys.modules[type(d).__module__]
        return getattr(mod, "SYSTEM_EXACT", None) or mod.SYSTEM + (getattr(mod, "BELIEF_HINT", "") or "")
    def user_content(self, e, note=None):
        d = e._teacher
        if getattr(d, "images", False):
            return d.user_content(e, note=note) if "note" in inspect.signature(d.user_content).parameters else d.user_content(e)
        text = d._history(e)
        if note:
            head, sep, tail = text.rpartition(" Decide now.")
            text = head + carried(note) + "\nDecide now." + tail if sep else text + carried(note)
        return text, []
    def _history(self, e): return e._teacher._history(e)

    # actions
    def look_strings(self, e): return e._teacher.look_strings(e)
    def look_name(self, a, e):
        d = e._teacher; return d.look_name(a, e) if getattr(d, "look_name_takes_episode", False) else d.look_name(a)
    def answer_options(self, e): return e._teacher.answer_options(e)
    def parse_action(self, e, text):
        d = e._teacher
        if hasattr(d, "parse_action"): return d.parse_action(e, text)
        from plearn.method.trainer import parse_action
        return parse_action(d, e, text, {"looks": d.look_strings(e) if e.t < d.T else {}})
    def do_look(self, e, a): return e._teacher.do_look(e, a)
    def do_commit(self, e, y): return e._teacher.do_commit(e, y)
    def after_turn(self, e):
        d = e._teacher
        if hasattr(d, "after_turn"): d.after_turn(e)

    # belief and teacher
    def belief(self, e):
        B = e._teacher.belief(e); self._btype[type(B)] = e._teacher; return B
    def _of_belief(self, B): return self._btype[type(B)]
    def label(self, e): return e._teacher.label(e)
    def answer_posterior(self, e): return e._teacher.answer_posterior(e)
    def answer_prob(self, B, y): return self._of_belief(B).answer_prob(B, y)
    def depth(self, B):
        d = self._of_belief(B); return d.depth(B) if hasattr(d, "depth") else 2
    def state_text(self, e): return e._teacher.state_text(e)
    def belief_note(self, e):
        d = e._teacher; return d.belief_note(e) if hasattr(d, "belief_note") else d.state_text(e)
    def belief_score(self, B, text, miss=50.0): return self._of_belief(B).belief_score(B, text, miss)
    def line_excess(self, B, text, exact): return self._of_belief(B).line_excess(B, text, exact)
    def belief_best(self, B, miss=50.0):
        d = self._of_belief(B); return d.belief_best(B, miss) if "miss" in inspect.signature(d.belief_best).parameters else d.belief_best(B)
    def read_exact(self, text): return any(d.read_exact(text) for d in self.subs if hasattr(d, "read_exact"))
    def anchor_support(self, text, answers, e=None):
        d = e._teacher if e is not None else next((d for d in self.subs if hasattr(d, "read_exact") and d.read_exact(text)), self.subs[0])
        return d.anchor_support(text, answers, e)
    def anchor_equal(self, a, b):
        for d in self.subs:
            if hasattr(d, "read_exact") and d.read_exact(a) and d.read_exact(b): return d.anchor_equal(a, b)
        return " ".join(str(a).lower().split()) == " ".join(str(b).lower().split())
    def anchor_answer(self, anchor, answers, e=None):
        d = e._teacher if e is not None else self.subs[0]; return d.anchor_answer(anchor, answers, e)
    def extra_metrics(self, eps):
        out, groups = {}, {}
        for e in eps: groups.setdefault(e._teacher_name, []).append(e)
        for n, es in groups.items():
            d = self.by_name[n]
            if hasattr(d, "extra_metrics"): out.update({f"{n}_{k}": v for k, v in d.extra_metrics(es).items()})
            out[f"{n}_success"] = sum(bool(e.correct) for e in es) / len(es)
        return out
