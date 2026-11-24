"""What the stalling steps of a --stall_cap run are made of: reads the trained batches the run keeps
(<run>/case-<tag>/batches/batch_<update>.pkl, trainer.keep_batch) and splits the agent's own tries of each phase by game:
  tries        the agent's own tries (the teacher's row of --teacher_row is left out)
  stall        tries rejected only by the cap (the trainer's stall flag: the log stall_rejected)
  answer       tries that answer (SUBMIT); guess: answers the teacher rejects
  accepted     tries the teacher accepts
and lists the most frequent stalling actions of each game. Reads only; changes nothing.
Usage (repository root, the run's Python): python tools/stall_breakdown.py <run>/case-<tag>/batches [--phase 5] [--out breakdown.json]"""
import argparse, ast, collections, glob, json, os, pickle, re

ap = argparse.ArgumentParser()
ap.add_argument("folder"); ap.add_argument("--phase", type=int, default=5); ap.add_argument("--out", default="")
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.folder, "batch_*.pkl")), key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[0]))
if not files: raise SystemExit(f"no batch_*.pkl in {a.folder}")
C = collections.defaultdict(collections.Counter); acts = collections.defaultdict(collections.Counter); hists = collections.defaultdict(collections.Counter)
for f in files:
    u = int(re.findall(r"\d+", os.path.basename(f))[0]); ph = (u - 1) // a.phase + 1
    g = pickle.load(open(f, "rb"))
    if not g.get("crit_h"): continue
    for (_, pl), rows, crit, z in zip(g["hist"], g["rows_h"], g["crit_h"], g["sizes"]):
        if not z: continue
        try: game = ast.literal_eval(pl.get("cls", "('?',)"))[0]
        except Exception: game = str(pl.get("cls"))
        hists[ph][game] += 1
        for r, c in zip(rows, crit):
            if len(r) > 9 and r[9]: continue                                   # the teacher's row
            k = (ph, game); C[k]["tries"] += 1
            ans = str(r[4]).strip().upper().startswith("ACTION: SUBMIT")
            C[k]["answer"] += ans; C[k]["accepted"] += bool(r[8]); C[k]["guess"] += ans and not r[8]
            if c.get("stall"): C[k]["stall"] += 1; acts[k][str(r[4]).strip()[:40]] += 1
phases = sorted({k[0] for k in C}); games = sorted({k[1] for k in C})
pc = lambda x, n: 100.0 * x / n if n else float("nan")
print(f"{len(files)} batches, phases {phases}\n")
print(f"{'phase':>5s} {'game':16s} {'histories':>9s} {'tries':>6s} {'stall':>6s} {'answer':>6s} {'guess':>6s} {'accepted':>8s}")
out = {}
for ph in phases:
    tot = collections.Counter()
    for gm in games + ["all"]:
        c = tot if gm == "all" else C.get((ph, gm))
        if not c: continue
        if gm != "all": tot.update(c)
        n = c["tries"]; h = sum(hists[ph].values()) if gm == "all" else hists[ph][gm]
        row = {"histories": h, "tries": n, **{x: pc(c[x], n) for x in ("stall", "answer", "guess", "accepted")}}
        out[f"{ph}/{gm}"] = row
        print(f"{ph:5d} {gm:16s} {h:9d} {n:6d} {row['stall']:6.1f} {row['answer']:6.1f} {row['guess']:6.1f} {row['accepted']:8.1f}")
    print()
print("most frequent stalling actions (phase, game: action x count)")
for k in sorted(acts):
    print(f"  {k[0]} {k[1]:16s} " + "; ".join(f"{s} x{n}" for s, n in acts[k].most_common(4)))
if a.out: json.dump({"table": out, "stall_actions": {f"{k[0]}/{k[1]}": dict(v.most_common(10)) for k, v in acts.items()}}, open(a.out, "w"), indent=1)
