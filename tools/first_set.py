#!/usr/bin/env python3
"""first_set.py: save, find and check the first sample set of a phase run, so that a later run with the SAME settings can start
from it (PL_SET_FROM=<set folder>: the actor loads batches 1..phase and the answer-accuracy table, and the learner starts at update 1
with no first collection). It saves training time; the settings are logged in full, so that a wrong first set is never reused.
An on-policy run starts fresh; every other phase run may reuse a set made with the same settings.

A set is reusable only when ALL of these are equal (this tool refuses anything else; the trainer's own PL_SET_FROM check is a
second guard):
  1. every trainer argument that is not a run/speed key (the trainer's own `free` list in actor_loop, read from the code), incl.
     the model path, the tasks (--domain_args without "bridges"), phase, phase_set, set_keep, B, worlds, beta0, beta_decay,
     line_score, tau_look, the teacher flags, temperature, max_think and seed;
  2. the sampling environment (SET_ENV below, e.g. PL_PRESENCE_PENALTY, PL_MAX_MODEL_LEN);
  3. the code: a sha256 over every plearn/**/*.py file of the code folder (a teacher change with no flag, e.g. a fix of S_dec,
     changes the sets; so a different code tree is NEVER a match);
  4. the batches: their sha256 at save time (check).
Never for an on-policy run (phase 1): its first set is one small batch (run it fresh).

  save  --run <run output folder> --code <code folder> [--registry R] [--out D]   copy the run's first set + write set.json,
                                                                                      SETTINGS.txt and one REGISTRY.tsv line
  find  --config <config-<tag>.json of the NEW run (its DRY writes it)> --code <code folder of the new run> [--registry R]
        MATCH: prints the folder and the line `export PL_SET_FROM=<folder>`; NO MATCH: each registered set of the same model
        with every differing setting (old -> new)
  check --set <set folder> --config <new config json> --code <code folder>    the full check of one set (exit 0 = PASS)
Run with the code folder's Python environment (it imports the trainer to read its argument parser); CPU only.
"""
import argparse, ast, datetime, glob, hashlib, json, os, shutil, sys, textwrap

REGISTRY = os.environ.get("PL_FIRST_SETS", os.path.join("runs", "first-sets", "REGISTRY.tsv"))   # the sets' registry (--registry)
SET_ENV = ("PL_PRESENCE_PENALTY", "PL_MAX_MODEL_LEN", "PL_TAIL_RESERVE", "PL_MAX_IMAGES")   # the environment that changes the samples
SHOW = ("model", "domain_args", "phase", "phase_set", "set_keep", "B", "worlds", "episodes", "beta0", "beta_decay", "line_score",
        "tau", "tau_look", "tau_look_play", "fl_view_value", "sk_guess_value", "plan_past_limit", "temperature",
        "max_think", "accept_only", "em_k", "em_m", "seed")   # the settings SETTINGS.txt and the registry show first (all are compared)
COLS = ["fingerprint", "made_at", "model", "phase", "phase_set", "set_keep", "line_score", "tau_look", "beta_decay", "code_tree",
        "source_run", "folder"]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def code_tree(code):
    """sha256 over the sorted (path, content) of every plearn/**/*.py file: the code that makes a set."""
    h = hashlib.sha256(); root = os.path.join(code, "plearn")
    files = sorted(p for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True) if "__pycache__" not in p)
    if not files: raise SystemExit(f"{code}: no plearn/**/*.py files")
    for p in files:
        h.update(os.path.relpath(p, code).encode() + b"\0"); h.update(open(p, "rb").read()); h.update(b"\0")
    return h.hexdigest()[:16]


def trainer_reader(code):
    """(parse, free): the trainer's own argument parser of the code folder (its __main__ block up to parse_args, run in the
    module's namespace) and its PL_SET_FROM `free` set (the run/speed keys a saved set may differ in)."""
    src_path = os.path.join(code, "plearn", "method", "trainer.py"); src = open(src_path).read(); tree = ast.parse(src)
    free = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "free" for t in node.targets) and isinstance(node.value, ast.Set):
            free = set(ast.literal_eval(node.value))
    if free is None: raise SystemExit(f"{src_path}: no PL_SET_FROM `free` set (this code cannot load a saved set)")
    main = next((n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)), None)
    if main is None: raise SystemExit(f"{src_path}: no __main__ block")
    body = []
    for st in main.body:
        if isinstance(st, ast.Assign) and "parse_args" in ast.unparse(st.value): break
        body.append(st)
    sys.path.insert(0, code)
    import importlib
    T = importlib.import_module("plearn.method.trainer")
    if os.path.realpath(os.path.dirname(T.__file__)) != os.path.realpath(os.path.dirname(src_path)):
        raise SystemExit(f"imported {T.__file__}, not {src_path}: run this tool in a fresh process per code folder")
    ns = dict(vars(T)); exec(compile(ast.Module(body=body, type_ignores=[]), src_path, "exec"), ns)
    ap = ns["ap"]
    def parse(argv):
        i = argv.index("plearn.method.trainer") + 1
        try: return vars(ap.parse_args(argv[i:]))
        except SystemExit: raise SystemExit(f"NO MATCH / FAIL: the run's arguments do not parse with the trainer of {code} (another code: never reuse)")
    return parse, free


def domain_key(s):
    return {k: v for k, v in json.loads(s).items() if k != "bridges"}


def set_key(args, env, free):
    """Every setting that must be equal for a reuse: the trainer's non-free arguments (domain_args without bridges) + SET_ENV."""
    a = {k: (domain_key(v) if k == "domain_args" else v) for k, v in args.items() if not k.startswith("wandb_") and k not in free}
    return {"args": a, "env": {k: env[k] for k in SET_ENV if k in env}}


def fingerprint(key, tree):
    return hashlib.sha256(json.dumps({**key, "code_tree": tree}, sort_keys=True, default=str).encode()).hexdigest()[:16]


def diff(old, new):
    out = []
    for part in ("args", "env"):
        for k in sorted(set(old[part]) | set(new[part])):
            if old[part].get(k, "<absent>") != new[part].get(k, "<absent>"):
                out.append(f"{part}.{k}: {old[part].get(k, '<absent>')!r} -> {new[part].get(k, '<absent>')!r}")
    return out


def run_config(run):
    """The run's fresh-start config json (config-<tag>.json, not a -r<R> resume file) and its actor argv."""
    cs = [p for p in glob.glob(os.path.join(run, "config-*.json")) if "-r" not in os.path.basename(p)[len("config-"):].rsplit("_", 1)[-1]]
    cs = [p for p in cs if not os.path.basename(p)[:-5].split("-")[-1].startswith("r") or not os.path.basename(p)[:-5].split("-")[-1][1:].isdigit()]
    if len(cs) != 1: raise SystemExit(f"{run}: need exactly one fresh config-<tag>.json, found {cs}")
    return cs[0], json.load(open(cs[0]))


def readable(args, env, tree):
    lines = [f"  {k:16s} {json.dumps(args.get(k), default=str) if k != 'domain_args' else json.dumps(domain_key(args[k]))}" for k in SHOW if k in args]
    lines += [f"  env {k:12s} {env[k]}" for k in SET_ENV if k in env] + [f"  code_tree        {tree}"]
    return "\n".join(lines)


def save(a):
    run = os.path.normpath(a.run); cfg_path, cfg = run_config(run); parse, free = trainer_reader(a.code)
    args = parse(cfg["actor"]); env = cfg.get("env", {}); tree = code_tree(a.code); phase = int(args["phase"])
    if not args.get("phase_set") or phase < 2:
        raise SystemExit(f"{run}: phase {phase}, phase_set {args.get('phase_set')}: only a phase run (phase >= 2, phase_set > 0) has a reusable first set (on-policy: run fresh)")
    tag = os.path.basename(cfg_path)[len("config-"):-len(".json")]
    srcs = [os.path.join(run, f"case-{tag}", "batches"), os.path.join(run, f"queue-{tag}")]
    bdir = next((d for d in srcs if all(os.path.exists(os.path.join(d, f"batch_{u}.pkl")) for u in range(1, phase + 1))), None)
    if not bdir: raise SystemExit(f"{run}: batches 1..{phase} not all in {srcs}")
    rho = glob.glob(os.path.join(run, "*-rho-0.json"))
    if args.get("rho_rb") and len(rho) != 1: raise SystemExit(f"{run}: --rho_rb {args['rho_rb']} needs one *-rho-0.json, found {rho}")
    alog = glob.glob(os.path.join(run, "*-actor.jsonl"))
    row = next((json.loads(l) for p in alog for l in open(p) if '"phase_set": 0' in l), None)
    if row is None: raise SystemExit(f"{run}: no actor log row of the first set (phase_set 0) in {alog}")
    if row.get("set_from"): print(f"NOTE: the source run itself started from {row['set_from']} (the same data is saved again)")
    key = set_key(args, env, free); fp = fingerprint(key, tree)
    out = a.out or os.path.join(os.path.dirname(a.registry), f"{os.path.basename(run)}-{fp}")
    if os.path.exists(out): raise SystemExit(f"{out} exists: no overwrite")
    os.makedirs(out); shas = {}
    for u in range(1, phase + 1):
        src = os.path.join(bdir, f"batch_{u}.pkl"); dst = os.path.join(out, f"batch_{u}.pkl"); shutil.copy2(src, dst)
        shas[f"batch_{u}.pkl"] = sha_file(dst)
        if shas[f"batch_{u}.pkl"] != sha_file(src): raise SystemExit(f"copy of {src} differs")
    if rho: shutil.copy2(rho[0], os.path.join(out, "rho-0.json")); shas["rho-0.json"] = sha_file(os.path.join(out, "rho-0.json"))
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    meta = {"set_size": row["set_size"], "set_pool": row["set_pool"], "set_worlds": row["set_worlds"], "set_sec": row["set_sec"],
            "source": f"{run} ({cfg_path}; batches from {bdir}); saved {now} by tools/first_set.py",
            "args": args, "env": key["env"], "fingerprint": fp, "code_tree": tree, "code": os.path.realpath(a.code), "files_sha256": shas,
            "wandb_name": args.get("wandb_name")}
    json.dump(meta, open(os.path.join(out, "set.json"), "w"), indent=1, default=str)
    txt = (f"FIRST SAMPLE SET {fp}  (saved {now})\nsource run: {run}\nW&B name: {args.get('wandb_name')}\n"
           f"set: {row['set_size']} histories kept of {row['set_pool']} played ({row['set_worlds']} scenes, {row['set_sec'] / 60:.1f} min)\n"
           f"REUSE ONLY WHEN ALL SETTINGS BELOW (and every other non-free trainer argument in set.json) ARE EQUAL; check with\n"
           f"  first_set.py check --set {out} --config <new run's config json> --code <new run's code>\n" + readable(args, env, tree) + "\n")
    open(os.path.join(out, "SETTINGS.txt"), "w").write(txt)
    reg = a.registry; new = not os.path.exists(reg); os.makedirs(os.path.dirname(reg), exist_ok=True)
    with open(reg, "a") as f:
        if new: f.write("\t".join(COLS) + "\n")
        f.write("\t".join(str(x) for x in [fp, now, args["model"], phase, args["phase_set"], args.get("set_keep"), args.get("line_score"),
                                            args.get("tau_look", "-"), args.get("beta_decay"), tree, run, out]) + "\n")
    print(txt + f"SAVED {out} ({len(shas)} files); registry {reg}")


def new_run(a):
    cfg = json.load(open(a.config)); parse, free = trainer_reader(a.code)
    args = parse(cfg["actor"]); env = cfg.get("env", {}); return args, env, free, code_tree(a.code)


def check_one(folder, args, env, free, tree, quiet=False):
    """The full check of one set folder against a new run: (ok, reasons)."""
    meta = json.load(open(os.path.join(folder, "set.json"))); why = []
    if int(args["phase"]) < 2: why.append(f"the new run has phase {args['phase']} (on-policy: run fresh, no reuse)")
    old = set_key(meta["args"], meta.get("env", {}), free); new = set_key(args, env, free)
    why += diff(old, new)
    if meta.get("code_tree") != tree: why.append(f"code_tree: {meta.get('code_tree')} -> {tree} (the code that makes the set differs)")
    for f, s in (meta.get("files_sha256") or {}).items():
        p = os.path.join(folder, f)
        if not os.path.exists(p) or sha_file(p) != s: why.append(f"{f}: missing or changed since the save")
    for u in range(1, int(args["phase"]) + 1):
        if not os.path.exists(os.path.join(folder, f"batch_{u}.pkl")): why.append(f"batch_{u}.pkl missing")
    return not why, why


def find(a):
    args, env, free, tree = new_run(a)
    if not os.path.exists(a.registry): raise SystemExit(f"NO MATCH: no registry {a.registry}")
    rows = [dict(zip(COLS, l.rstrip("\n").split("\t"))) for l in open(a.registry).readlines()[1:] if l.strip()]
    print(f"new run: {args.get('wandb_name')}\n" + readable(args, env, tree))
    hits, near = [], []
    for r in rows:
        if r["model"] != args["model"] or not os.path.isdir(r["folder"]): continue
        ok, why = check_one(r["folder"], args, env, free, tree)
        (hits if ok else near).append((r, why))
    if hits:
        r = hits[-1][0]
        print(f"MATCH {r['fingerprint']} ({r['source_run']}, saved {r['made_at']}): every setting, the sampling environment and the code are equal")
        print(f"export PL_SET_FROM={r['folder']}"); return 0
    print(f"NO MATCH: run the first set fresh. Sets of the same model ({len(near)}):")
    for r, why in near: print(f"  {r['fingerprint']} {r['source_run']}: {len(why)} differences\n" + "\n".join(f"    {w}" for w in why[:20]))
    return 3


def check(a):
    args, env, free, tree = new_run(a); ok, why = check_one(a.set, args, env, free, tree)
    print(("PASS: " if ok else "FAIL: ") + a.set); [print("  " + w) for w in why]; return 0 if ok else 3


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("save"); s.add_argument("--run", required=True); s.add_argument("--code", required=True); s.add_argument("--out", default="")
    f = sp.add_parser("find"); f.add_argument("--config", required=True); f.add_argument("--code", required=True)
    c = sp.add_parser("check"); c.add_argument("--set", required=True); c.add_argument("--config", required=True); c.add_argument("--code", required=True)
    for p in (s, f, c): p.add_argument("--registry", default=REGISTRY)
    a = ap.parse_args()
    sys.exit({"save": save, "find": find, "check": check}[a.cmd](a) or 0)
