"""One evaluator for every registered task: the episodes of all tasks at once through one model server, the agent's turn of
plearn/episode.py, each finished episode saved at once (a new start with the same output folder plays only the missing ones),
then each task's metrics and each group's (its episodes pooled).

Output (out_dir): <task>.jsonl (one row per episode), <task>.jsonl.part while it runs, results.json.

Before the game threads start, play() imports the game libraries of the run's tasks in the calling (main) thread (preimport): the
games import them inside functions, and many threads that import one module at once gave "_DeadlockError" and "partially
initialized module 'gymnasium'" (every game of a run failed).
"""
from __future__ import annotations
import importlib, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor

from plearn.api.episode import turn
from plearn.api.registry import registry
from plearn.api.task import Task


def prompt_of(proc, ep, carry):
    from plearn.models.sampler import tag_prompt
    system, parts, images = ep.messages(carry)
    text = proc.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": parts}], tokenize=False,
                                    add_generation_prompt=True)
    return tag_prompt(text, images)


PREIMPORT = {"frozenlake": ("gymnasium", "gymnasium.utils", "gymnasium.envs.registration", "gymnasium.envs.toy_text.frozen_lake"),
             "minigrid": ("gymnasium", "minigrid", "minigrid.envs.keycorridor")}


def preimport(tasks, log=None):
    """Import the game libraries of the tasks (PREIMPORT: a key in the task's name, env, env_args or teacher) in this thread. A
    library that is not installed is skipped (ImportError). Returns the modules imported."""
    done = []
    for key, mods in PREIMPORT.items():
        if not any(key in json.dumps([t.name, t.env, t.env_args, t.teacher or ""], default=str) for t in tasks): continue
        for m in mods:
            if m in done: continue
            try: importlib.import_module(m); done.append(m)
            except ImportError as e:
                if log: log(f"preimport: {m} not imported ({type(e).__name__}: {e})")
    return done


def play(srv, proc, tasks, n=None, split=None, out_dir=None, threads=256, use_adapter=True, priority=0, gen=None, log=print):
    """Every episode of the tasks (Task objects or names) through a running server srv. gen: generation settings over the tasks'
    own. Returns ({task: rows}, minutes)."""
    from plearn.models.sampler import IMAGES, prompt_key
    tasks = [t if isinstance(t, Task) else Task.load(t) for t in tasks]
    for t in tasks: t.check()
    preimport(tasks, log)                                            # the game libraries, in this thread, before the game threads
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    jobs, done, lock, t0 = [], {}, threading.Lock(), time.time()
    for t in tasks:
        part = os.path.join(out_dir, f"{t.name}.jsonl.part") if out_dir else None
        have = {}
        if part and os.path.exists(part):
            for line in open(part):
                r = json.loads(line)
                if "error" not in r: have[json.dumps(r["spec"], sort_keys=True)] = r
        done[t.name] = list(have.values())
        jobs += [(t, s, part) for s in t.specs(split, n) if json.dumps(s, sort_keys=True) not in have]
    count = [0]
    if any(done.values()): log(f"{sum(map(len, done.values()))} episodes from earlier starts")

    def one(job):
        t, spec, part = job
        g = {**t.generation_kwargs, **(gen or {})}; ep = None
        try:
            ep = t.episode(spec, out_dir=out_dir)
            while not ep.done:
                p = prompt_of(proc, ep, g["carry"])
                _, b, a, rec = turn(srv, p, g, use_adapter, priority, one_line=bool(getattr(ep, "one_line", False)))
                ep.step(b, a, rec)
                IMAGES.pop(prompt_key(p), None)
            row = {"spec": spec, **ep.row()}
        except Exception as e:                                       # one failed episode must not stop the others
            row = {"spec": spec, "correct": False, "error": repr(e)[:400]}
        finally:
            if ep is not None:
                try: ep.close()
                except Exception: pass
        with lock:
            count[0] += 1; done[t.name].append(row)
            if part:
                with open(part, "a") as f: f.write(json.dumps(row) + "\n")
            log(f"{time.strftime('%H:%M:%S')} {count[0]}/{len(jobs)} {t.name} {json.dumps(spec)}: "
                + (f"error {row['error'][:120]}" if "error" in row else ("correct" if row.get("correct") else "wrong")))
        return row

    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, min(threads, len(jobs)))) as ex: list(ex.map(one, jobs))
    return done, (time.time() - t0) / 60


def score(rows_by_task, reg=None):
    """Each task's metrics, and each group's that holds at least one of them: its member tasks' episodes pooled, for the metrics
    that all of its evaluated members have."""
    reg = reg or registry()
    tasks = {name: Task.load(name, reg) for name in rows_by_task}
    results = {name: t.score(rows_by_task[name]) for name, t in tasks.items()}
    groups = {}
    for g in reg.names("tasks", groups=True):
        members = [m for m in reg.tasks(g) if m in tasks]
        if not members: continue
        common = set.intersection(*[{m_["metric"] for m_ in tasks[m].metric_list} for m in members])
        pooled = {"tasks": len(members), "episodes": sum(len(rows_by_task[m]) for m in members)}
        for metric in sorted(common):
            xs = [tasks[m].metric_values(r).get(metric) for m in members for r in rows_by_task[m] if "error" not in r]
            xs = [float(x) for x in xs if x is not None]
            pooled[metric] = sum(xs) / len(xs) if xs else None
        groups[g] = pooled
    return {"results": results, "groups": groups}


def evaluate(srv, proc, tasks, out_dir, n=None, split=None, gen=None, use_adapter=True, config=None, wandb=None, step=None, log=print):
    """play + score; writes out_dir/<task>.jsonl and results.json; logs to W&B when a logger is given. Returns the results.
    tasks: names (a string of tasks and groups, or a list of task names) or Task objects; a Task object is played as it is given,
    with its env_args (plearn eval --task_args)."""
    reg = registry()
    tasks = [t if isinstance(t, Task) else Task.load(t, reg) for t in (reg.tasks(tasks) if isinstance(tasks, str) else tasks)]
    names = [t.name for t in tasks]
    rows, minutes = play(srv, proc, tasks, n=n, split=split, out_dir=out_dir, gen=gen, use_adapter=use_adapter, log=log)
    res = {**score(rows, reg), "config": {**(config or {}), "tasks": names, "split": split, "n": n, "gen": gen, "minutes": minutes}}
    for name, rs in rows.items():
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            for r in rs: f.write(json.dumps(r) + "\n")
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    if wandb is not None: wandb.log_results(res, step=step)
    return res


def table(res, metrics=("success",)):
    """The results as text: one line per group, then per task."""
    lines = []
    for kind in ("groups", "results"):
        for name, r in res[kind].items():
            vals = "  ".join(f"{m} {r[m]:.3f}" if isinstance(r.get(m), float) else f"{m} -" for m in metrics)
            lines.append(f"{'group' if kind == 'groups' else 'task ':5s} {name:48s} {r.get('episodes', 0):4d} episodes  {vals}")
    return "\n".join(lines)
