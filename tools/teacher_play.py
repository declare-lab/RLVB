"""The teacher's own play on registered tasks: each task's teacher plays its episodes with its policy pi*, read two ways. 'mode'
takes the most likely decision (the answer when its share of pi* is at least one half, else the look of the highest value);
'sample' draws from pi*, as the trainer's targets do. The value is the method's (plearn/train.py METHOD: the log value, tau 0.1;
the lookahead depth 2). No model is needed: the numbers say how well each environment's teacher guides, and the dataset
READMEs report them.

    python tools/teacher_play.py --tasks sokoban_hidden_target,frozenlake_hidden_goal [--split validation] [--n 128]
"""
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from plearn.api.registry import registry
from plearn.method import estimator as E
from plearn.method.trainer import Planner
from plearn.tasks.mixture import teacher_of
from plearn.train import METHOD


def play(D, P, spec, how, rng):
    """One episode of the teacher's policy: (success, actions)."""
    e = D.start([spec])[0]
    try:
        while True:
            pl = P(e); tgt, looks, commits = pl["tgt"], pl["looks"], pl["commits"]
            if how == "sample":
                ks = list(tgt); ps = np.array([tgt[k] for k in ks]); s = ks[int(rng.choice(len(ks), p=ps / ps.sum()))]
            elif not looks or sum(tgt.get(c, 0.0) for c in commits.values()) >= 0.5:
                s = None                                                  # the answer
            else:
                s = max(looks.values(), key=lambda x: pl["values"].get(x, -1e9))
            inv = {v: k for k, v in D.look_strings(e).items()}
            if s not in inv:
                return (bool(D.do_commit(e, pl["best"])) if pl["best"] is not None else False), e.t
            D.do_look(e, inv[s]); e.t += 1
            if hasattr(D, "after_turn"): D.after_turn(e)
            if e.done: return bool(e.correct), e.t
    finally:
        D.close([e])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="tasks or groups, comma separated")
    ap.add_argument("--split", default="validation"); ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()
    for name in registry().tasks(args.tasks.split(",")):
        teacher, targs = teacher_of([name]); D = teacher(**{**targs, "split": args.split})
        for how in ("mode", "sample"):
            P = Planner(D, E.Reliability(fidelity=True, value=METHOD["value"]), METHOD["tau"], 2)
            rng, t0 = np.random.default_rng(0), time.time()
            rows = [play(D, P, spec, how, rng) for spec in D.eval_specs(args.n)]
            print(f"{name:24s} {how:6s} {args.split} n={len(rows)}: success {np.mean([r[0] for r in rows]):.3f}  "
                  f"actions {np.mean([r[1] for r in rows]):5.1f}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
