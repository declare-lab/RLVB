"""python -m plearn tasks | show | eval | train

    python -m plearn tasks [--groups] [--dataset proworlds]
    python -m plearn show test_suite
    python -m plearn eval --model qwen3-vl-8b-thinking [--adapter DIR] --tasks test_suite --gpus 0 \
        [--limit 8] [--output_path DIR] [--task_args proworlds.one_line=1] [--wandb_project P --wandb_name N]
    python -m plearn train --config suite_qwen3vl_8b_thinking [--resume_step 25] [--dry_run]      (RLVB (EM) or RLVB (GRPO))
"""
from __future__ import annotations
import argparse, json, os, sys, time


def _value(v):
    try: return json.loads(v)
    except ValueError: return v


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="python -m plearn"); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks", help="list the registered tasks (or groups)")
    t.add_argument("--groups", action="store_true"); t.add_argument("--dataset", default="")
    s = sub.add_parser("show", help="print one task, group, model or training run")
    s.add_argument("name")
    e = sub.add_parser("eval", help="evaluate a model on registered tasks and groups")
    e.add_argument("--model", required=True, help="a registered model or a weights folder")
    e.add_argument("--adapter", default="", help="a LoRA adapter folder (none: the base model)")
    e.add_argument("--tasks", required=True, help="a comma list of tasks and groups")
    e.add_argument("--gpus", default="0", help="one tensor-parallel server over these GPUs")
    e.add_argument("--limit", type=int, default=0, help="episodes of each task (0: the task's num_episodes)")
    e.add_argument("--split", default="", help="the split (default: each task's test_split)")
    e.add_argument("--output_path", default="", help="default: $PL_RUNS/eval/<model>-<time>")
    e.add_argument("--max_think", type=int, default=0, help="the reasoning cap of every task (0: the task's own)")
    e.add_argument("--task_args", nargs="*", default=[], help="DATASET.KEY=VALUE: env_args for the tasks of a dataset")
    e.add_argument("--mem", type=float, default=0.85)
    for k in ("project", "entity", "name", "group"): e.add_argument(f"--wandb_{k}", default="")
    r = sub.add_parser("train", help="start a training run from its file")
    r.add_argument("--config", required=True, help="a registered training run or a YAML path")
    r.add_argument("--resume_step", type=int, default=0); r.add_argument("--dry_run", action="store_true")
    a = ap.parse_args(argv)

    from plearn.api.registry import registry
    reg = registry()
    if a.cmd == "tasks":
        for n in reg.names("tasks", groups=a.groups):
            entry = reg.entries["tasks"][n]
            if a.dataset and entry.get("dataset") != a.dataset and not (a.groups and n.startswith(a.dataset)): continue
            print(f"{n:48s} " + (f"{len(reg.tasks(n))} tasks" if a.groups else entry.get("dataset", "")))
        return 0
    if a.cmd == "show":
        for kind in ("tasks", "models", "train"):
            if a.name in reg.entries[kind]:
                e_ = reg.get(kind, a.name)
                if kind == "tasks" and "group" in e_: e_["expanded"] = reg.tasks(a.name)
                print(json.dumps(e_, indent=1)); return 0
        print(f"unknown name {a.name!r}", file=sys.stderr); return 2
    if a.cmd == "train":
        from plearn.train import run
        return run(a.config, a.resume_step, a.dry_run)

    # eval
    from plearn import evaluator, loggers, models
    from plearn.api.task import Task
    path, _ = models.resolve(a.model)
    tasks = [Task.load(n, reg) for n in reg.tasks(a.tasks)]
    over = {}
    for kv in a.task_args:
        key, _, v = kv.partition("="); ds, _, k = key.partition(".")
        over.setdefault(ds, {})[k] = _value(v)
    for t_ in tasks: t_.env_args.update(over.get(t_.dataset, {}))
    out = a.output_path or os.path.join(os.environ.get("PL_RUNS", "runs"), "eval",
                                        f"{os.path.basename(path.rstrip('/'))}{'-' + os.path.basename(a.adapter.rstrip('/')) if a.adapter else ''}-{time.strftime('%m%d-%H%M')}")
    os.makedirs(out, exist_ok=True)
    caps = [t_.generation_kwargs["max_think"] for t_ in tasks]
    max_think = a.max_think or (-1 if min(caps) < 0 else max(caps))      # one task with no cap (-1): the server's context for no cap
    config = {"model": a.model, "path": path, "adapter": a.adapter or None, "tasks": a.tasks, "limit": a.limit, "task_args": over}
    wb = loggers.from_args(a, config)
    srv = models.serve(path, a.gpus, mem=a.mem, log=os.path.join(out, "vllm.log"), max_think=max_think, adapter=a.adapter or None)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpus.split(",")[0])   # the tasks' simulators and renderers: the server's first GPU
    try:
        res = evaluator.evaluate(srv, models.processor(path), tasks, out, n=a.limit or None, split=a.split or None,
                                 gen={"max_think": a.max_think} if a.max_think else None, use_adapter=bool(a.adapter), config=config, wandb=wb)
    finally:
        srv.stop()
        if wb: wb.finish()
    print(evaluator.table(res)); print(f"results: {out}/results.json")
    return 0
