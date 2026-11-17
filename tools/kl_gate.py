"""The KL gate of an RLVB (EM) run: reads the learner's log (one row a
real update: "step" and "kl_prior"; a KL STOP skips the next updates of the phase, with no rows for them) while the run goes on, and
decides at update STEP:
  PASS  the row of update STEP has kl_prior <= LIMIT, and no KL STOP came before it (rc 0);
  FAIL  kl_prior of update STEP > LIMIT, or a row up to STEP has kl_stop_left > 0 (a KL STOP before STEP), or the first row after
        STEP comes with no row of STEP (rc 1);
  NONE  the run (--pid) ended, or --timeout passed, before a decision (rc 2).
It prints one JSON line: {"gate": PASS|FAIL|NONE, "step", "kl_prior", "limit", "rows": [[step, kl_prior], ...]}.
    python tools/kl_gate.py --run_dir runs/X --tag X --step 5 --limit 0.004 [--pid PID] [--poll 30] [--timeout 0]   (or --log FILE)
The learner's log (plearn/method/trainer.py 1439 + 1824): <output>/<domain>-<tag>-seed<seed>.jsonl, for example
runs/suite_qwen35_4b_lr3e6/mixture-suite_qwen35_4b_lr3e6-seed0.jsonl; <...>-actor.jsonl is the actor's (no kl_prior rows).
The trainer never writes <output>/<tag>.jsonl.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time


def learner_log(run_dir, tag, seed=0):
    """The learner's log of a run: <run_dir>/<domain>-<tag>-seed<seed>.jsonl (the trainer's names; the domain's name first, for
    example 'mixture'), not the actor's '-actor.jsonl'. None while the trainer has not made it; ValueError for two candidates."""
    c = [f for f in glob.glob(os.path.join(glob.escape(run_dir), f"*-{glob.escape(tag)}-seed{int(seed)}.jsonl")) if not f.endswith("-actor.jsonl")]
    if len(c) > 1: raise ValueError(f"two learner logs for {tag}: {c}")
    return c[0] if c else None


def rows_of(path):
    """The update rows (with a step and a kl_prior or kl_stop_left) of a learner log; a line that is not whole JSON yet is left out."""
    out = []
    if not os.path.exists(path): return out
    with open(path) as f:
        for line in f:
            try: r = json.loads(line)
            except ValueError: continue
            if isinstance(r, dict) and isinstance(r.get("step"), int) and ("kl_prior" in r or "kl_stop_left" in r) and "eval" not in r: out.append(r)
    return out


def decide(rows, step, limit):
    """(PASS | FAIL | None, the row of update step or None) from the update rows so far."""
    for r in sorted(rows, key=lambda r: r["step"]):
        if r["step"] < step and (r.get("kl_stop_left") or 0) > 0: return "FAIL", r       # a KL STOP before update step
        if r["step"] == step:
            kl = r.get("kl_prior")
            return ("PASS" if kl is not None and float(kl) <= limit and not (r.get("kl_stop_left") or 0) > 0 else "FAIL"), r
        if r["step"] > step: return "FAIL", None                                            # update step was skipped
    return None, None


def alive(pid):
    try: os.kill(pid, 0); return True
    except OSError: return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=""); ap.add_argument("--run_dir", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=5); ap.add_argument("--limit", type=float, default=0.004)
    ap.add_argument("--pid", type=int, default=0); ap.add_argument("--poll", type=float, default=30); ap.add_argument("--timeout", type=float, default=0)
    a = ap.parse_args(argv); t0 = time.time()
    if not a.log and not (a.run_dir and a.tag): ap.error("give --log, or --run_dir and --tag")
    while True:
        log = a.log or learner_log(a.run_dir, a.tag, a.seed)
        rows = rows_of(log) if log else []; g, r = decide(rows, a.step, a.limit)
        ended = (a.pid and not alive(a.pid)) or (a.timeout and time.time() - t0 > a.timeout)
        if g is None and ended: g = "NONE"
        if g is not None:
            print(json.dumps({"gate": g, "log": log, "step": a.step, "kl_prior": None if r is None else r.get("kl_prior"), "limit": a.limit,
                              "rows": [[x["step"], x.get("kl_prior")] for x in sorted(rows, key=lambda x: x["step"])]}), flush=True)
            return {"PASS": 0, "FAIL": 1, "NONE": 2}[g]
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
