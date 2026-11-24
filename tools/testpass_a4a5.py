#!/usr/bin/env python
"""The paper measures A4 (behaviour), A5 (belief) and A6 (thoughts) from the saved TEST outputs of `plearn eval`. CPU only: no
model, no GPU. One eval output folder holds <task>.jsonl (one row per episode: {"spec": ..., **Episode.row()}) and results.json
(its "config": model, path, adapter, tasks, split, task_args).

    python tools/testpass_a4a5.py --folders F1 F2 --labels PL Untrained [--seeds 0 0] --out DIR
    python tools/testpass_a4a5.py --spec runs.tsv --out DIR      (a TSV with the columns folder, model, row, seed, benchmark)

In the TSV, model and seed may be empty (model: results.json config "model"; seed: 0). benchmark may be empty or "all" (every
benchmark of the folder), else one of HiddenGames, ProWorlds-expansion, RoboQuest (only that benchmark's tasks of the folder).
Folders with the same model, row and seed are joined (for example one folder per benchmark).

Output: DIR/<model>.csv with the columns benchmark, task, row, seed, measure, value, n, and DIR/README.md (the sources, the code
commit, the replay fidelity, the definitions). Each benchmark also has the task "overall" (its episodes pooled). The seed
"mean" is the mean of the per-seed values (n = the number of seeds with a value); the "overall" mean is "not measured" when the
seeds do not have the same tasks. A value that cannot exist is "not defined"; a
value that this tool cannot compute from the saved outputs is "not measured".

The measures (the trainer's own definitions, plearn/method/trainer.py evaluate and read_metrics; its functions where it has them):
A4, per episode, then the mean over the episodes:
  success       the share of episodes that are correct (all episodes).
  premature     over the ANSWERED episodes only (trainer answered / premature_rate): the share whose answer came while the exact
                posterior did not identify the answer (B.identified() before the answer); its n is the answered episodes that it
                uses. n_answered: the episodes that end with an answer. An episode that ends with no answer (a goal reached, a
                hole, a mine, the step or turn limit) is left out. No answered episode (or no answer line, FrozenLake and
                Sokoban): "not defined".
  late_looks    (evidence actions taken while the posterior already identified the answer, before the action) / (episodes).
  mean_t        evidence actions per episode (all episodes).
  budget        the share of all episodes that end at the step or turn limit with no answer.
A5, belief + action (two-line) rows only, per test turn (all turns pooled):
  anchor_exact  the belief line states the exact posterior support (D.anchor_equal with D.belief_note, the trainer's `same`).
  belief_ok     belief_best - belief_score <= 1e-6 (the trainer's read_metrics).
  decision_ok   decision_regret <= 1e-6 (the trainer's read_metrics: a Planner with a new reliability table, the method's tau
                0.1, depth 2, value log, rho fidelity, kappa 4, ucb_z 1; plearn/train.py METHOD).
  open_line     the share of the turns with evidence (the teacher's line excludes something) whose line is still open
                ('Possible: all / any'): the turn key `open` of plearn/tasks/domain.py.
A6, all rows, per turn: thought_chars_mean / _median (characters), thought_tokens_mean / _median (the model's tokenizer, only when
  the rows hold the thought text, PL_SAVE_THOUGHTS=1), thought_cut (the share of turns that the thinking cap cut).

The replay. The test rows of the HiddenGames games keep no per-turn posterior. For late_looks and A5 the tool rebuilds each
episode from its spec through the evaluator's own episode class (plearn/tasks/domain.py DomainEpisode, the task's domain and
domain_args, the results.json task_args applied), feeds it the RECORDED belief and action line of each turn (no model), and reads
the exact posterior D.belief(e) before each action. A domain whose __init__ takes `render` (tabletop) gets render=False: the
pictures go only into the prompt, which the replay does not build. The replay is checked against the stored row: the same kind of
every turn (and the same `open`, `reading`, `act_ok` where the row has them), the same end (success, number of turns, looks,
submitted, premature). An episode that differs is counted (replay_mismatch) and left out of the replay measures.
ProWorlds expansion: the replay is the training teacher's (plearn/tasks/proworlds/expansion_teacher.py: the engine's exact belief,
checked against the engine at every move) on the same seeds; it needs the node engine (NODE_BIN); without node it is "not
measured". Its premature is B.identified() of the replay; the test's own premature (the engine's belief of the answer below 0.95)
is the measure premature_engine095.
RoboQuest: the test episode (bench_eval) has no exact posterior, so premature, late_looks and A5 are "not defined".
"""
from __future__ import annotations
import argparse, csv, inspect, json, os, re, statistics, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""                              # CPU only, always
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_k, "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

BENCHMARKS = ("HiddenGames", "ProWorlds-expansion", "RoboQuest")
DOMAIN_ENV = "plearn.tasks.domain:DomainEpisode"
EXPANSION_TEACHER = "plearn.tasks.proworlds.expansion_teacher"
ND, NM = "not defined", "not measured"
A4 = ("success", "premature", "n_answered", "late_looks", "mean_t", "budget")
A5 = ("anchor_exact", "belief_ok", "decision_ok", "open_line")
A6 = ("thought_chars_mean", "thought_chars_median", "thought_tokens_mean", "thought_tokens_median", "thought_cut")
EXTRA = ("episodes", "errors", "replay_n", "replay_mismatch", "premature_engine095")
MEASURES = A4 + A5 + A6 + EXTRA
OK_TOL = 1e-6                                                        # the trainer's regret <= 1e-6 (read_metrics)
METHOD_PLAN = {"tau": 0.1, "depth": 2, "kappa": 4.0, "ucb_z": 1.0, "fidelity": True, "value": "log"}   # plearn/train.py METHOD


# --------------------------------------------------------------------- the tasks
def task_info(name, config, reg):
    """(benchmark, replay kind, env_args) of a task, with the run's task_args applied (plearn/cli.py)."""
    from plearn.api.task import Task
    t = Task.load(name, reg)
    env_args = {**t.env_args, **((config.get("task_args") or {}).get(t.dataset) or {})}
    if t.env == DOMAIN_ENV:
        hidden = set(reg.tasks("test_suite/hidden_games"))              # the HiddenGames test: 7 tasks
        return ("HiddenGames" if name in hidden else None), "domain", env_args
    if t.env.endswith("expansion_env:Episode"): return "ProWorlds-expansion", "expansion", env_args
    if t.env.endswith("bench_env:Episode") and t.dataset == "roboquest": return "RoboQuest", "roboquest", env_args
    return None, None, env_args


def turn_form(env_args, rows):
    """'two_line' or 'one_line': the agent's turn of the rows (one_line: an empty belief line in every turn)."""
    if env_args.get("one_line"): return "one_line"
    turns = [t for r in rows for t in r.get("turn_log", [])]
    if turns and not any((t.get("belief") or "").strip() for t in turns): return "one_line"
    return "two_line"


# --------------------------------------------------------------------- the replay (one process per worker)
class _Probe:
    """The episode's domain, with one change: it notes whether the game had already ended (an answer, a goal, a hole, a mine)
    before the domain's turn limit (after_turn) ran on the last turn. Every other attribute is the domain's own."""
    def __init__(self, D): self._D = D; self.ended_by_game = False
    def __getattr__(self, k): return getattr(self._D, k)
    def after_turn(self, e):
        self.ended_by_game = bool(e.done)
        if hasattr(self._D, "after_turn"): self._D.after_turn(e)


_PLANNERS = {}


def _planner(D):
    from plearn.method import estimator as E, trainer as TR
    if id(D) not in _PLANNERS:
        m = METHOD_PLAN
        rel = E.Reliability(m["kappa"], m["ucb_z"], fidelity=m["fidelity"], value=m["value"] if m["value"] in getattr(D, "value_modes", ("accuracy",)) else "accuracy")
        _PLANNERS[id(D)] = (D, TR.Planner(D, rel, m["tau"], m["depth"]))
    return _PLANNERS[id(D)][1]


EXP_KIND = {"move": "look", "budget": "look", "answer": "answer", "unreadable": "invalid"}   # expansion_env kinds -> the teacher's


def replay(job):
    """One episode replayed from its row: {"ok", "why", "late", "looks", "premature", "budget", "turns": [per-turn facts],
    "seconds"}. job: (kind, env_args, row, two_line, decision)."""
    kind, env_args, row, two_line, decision = job
    t0 = time.process_time()
    from plearn.tasks.domain import DomainEpisode
    from plearn.method import estimator as E, trainer as TR
    out = {"ok": False, "why": None, "late": 0, "looks": 0, "premature": None, "budget": None, "turns": []}
    if kind == "domain":
        dom = env_args["domain"]; dargs = dict(env_args.get("domain_args") or {})
        import importlib
        if "render" in inspect.signature(importlib.import_module(dom).Domain.__init__).parameters: dargs["render"] = False
        spec = row["spec"]
    else:                                                              # the ProWorlds expansion: the training teacher on the test seed
        game = env_args["game"]; dom, dargs = EXPANSION_TEACHER, {"tasks": game}
        spec = ["test", [f"expansion/{game}", int(row["spec"][1]), "test"]]
    ep = None
    try:
        ep = DomainEpisode(spec, domain=dom, domain_args=dargs)
        D = ep.D; probe = _Probe(D); ep.D = probe; out["limit"] = ep.limit
        P = _planner(D) if (two_line and decision and hasattr(D, "line_belief")) else None
        log = row.get("turn_log") or []
        for i, t in enumerate(log):
            if ep.done: out["why"] = f"the replay ended after {i} turns, the row has {len(log)}"; return out
            e, b = ep.e, t.get("belief") or ""
            with ep.lock:
                B = D.belief(e); ident = bool(B.identified()); f = {}
                if two_line:
                    anchor = D.belief_note(e)
                    f["anchor_exact"] = bool(D.anchor_equal(b, anchor)) if hasattr(D, "anchor_equal") else E.note_set(b) == E.note_set(anchor)
                    pl = P(e) if P is not None else {"anchor": anchor, "answers": D.answer_options(e), "post": D.answer_posterior(e) or {}}
                    f["belief_ok"] = bool(TR.belief_best(D, B, pl) - TR.belief_score(D, B, b, pl) <= OK_TOL)
                    if P is not None: f["decision_ok"] = bool(TR.decision_regret(D, P, pl, b) <= OK_TOL)
            ep.step(b, t.get("action") or "", {})
            r = ep.turns[-1]
            want = t.get("kind") if kind == "domain" else EXP_KIND.get(t.get("kind"))
            if r["kind"] != want: out["why"] = f"turn {i}: kind {r['kind']} in the replay, {t.get('kind')} in the row"; return out
            for k in ("open", "act_ok", "reading"):
                if k in t and t[k] != r.get(k): out["why"] = f"turn {i}: {k} {r.get(k)} in the replay, {t[k]} in the row"; return out
            if "open" in r: f["open"] = bool(r["open"])
            if r["kind"] == "look": out["looks"] += 1; out["late"] += ident
            out["turns"].append(f)
        if not ep.done: out["why"] = f"the replay has not ended after the row's {len(log)} turns"; return out
        rr = ep.row()
        if kind == "domain":
            for k in ("success", "turns", "looks", "submitted", "premature"):
                if rr.get(k) != row.get(k): out["why"] = f"end: {k} {rr.get(k)} in the replay, {row.get(k)} in the row"; return out
        else:
            e = ep.e
            got = {"success": rr["success"], "turns": rr["turns"], "submitted": rr["submitted"], "end": e.end, "moves": e.obs["steps"],
                   "answer": e.answer}
            for k, v in got.items():
                if v != row.get(k): out["why"] = f"end: {k} {v} in the replay, {row.get(k)} in the row"; return out
        out["premature"] = rr["premature"]
        out["budget"] = rr["submitted"] == 0 and not rr["correct"] and not probe.ended_by_game
        out["ok"] = True
        return out
    except Exception as ex:                                            # a replay that fails is a mismatch, with its reason
        out["why"] = f"replay error: {type(ex).__name__}: {str(ex)[:200]}"; return out
    finally:
        if ep is not None:
            try: ep.D = D; ep.close()
            except Exception: pass
        out["seconds"] = time.process_time() - t0


def node_ok():
    """The expansion engine can run: NODE_BIN (or the module's default) is an executable file."""
    from plearn.tasks.proworlds import expansion_env as X
    return os.path.isfile(X.NODE) and os.access(X.NODE, os.X_OK)


# --------------------------------------------------------------------- the episode facts
def episode_facts(kind, row, rep, two_line):
    """The facts of one episode that the measures use. rep: the replay result, or None (not run)."""
    f = {"error": "error" in row, "kind": kind, "two_line": two_line}
    if f["error"]: return f
    log = row.get("turn_log") or []
    f["success"] = float(bool(row.get("correct")))
    f["thoughts"] = [(t.get("thought_chars"), bool(t.get("thought_cut")), t.get("_tokens")) for t in log]
    f["rep"] = rep
    if kind == "domain":
        f["answered"] = row.get("premature") is not None
        f["premature"] = row.get("premature")
        f["looks"] = float(row.get("looks", 0))
        f["open"] = [bool(t["open"]) for t in log if "open" in t] if two_line else None
        if rep is not None and rep["ok"]: f["budget"] = bool(rep["budget"])
        else: f["budget"] = row.get("submitted") == 0 and not row.get("correct") and len(log) >= (rep or {}).get("limit", 10 ** 9)
    elif kind == "expansion":
        ok = rep is not None and rep["ok"]
        f["answered"] = row.get("submitted") == 1.0
        f["premature"] = rep["premature"] if ok else None
        f["premature_engine095"] = row.get("premature")
        f["looks"] = float(row.get("moves", 0))
        f["open"] = [t["open"] for t in rep["turns"] if "open" in t] if (ok and two_line) else None
        f["budget"] = row.get("end") in ("budget", "turns")
    else:                                                              # RoboQuest: the benchmark's own episode
        f["answered"] = row.get("submitted") == 1.0
        f["looks"] = float(row.get("evidence_actions", 0))
        f["budget"] = row.get("outcome") == "no_answer"
    return f


def mean(xs): return sum(xs) / len(xs) if xs else None


def aggregate(eps, status):
    """{measure: (value, n)} over the episodes eps (one task, or a benchmark's tasks pooled). status: {measure: "not defined" |
    "not measured"} for the measures that have no value in this pool."""
    ok = [e for e in eps if not e["error"]]; out = {"episodes": (len(eps), len(eps)), "errors": (len(eps) - len(ok), len(eps))}
    out["success"] = (mean([e["success"] for e in ok]), len(ok))
    out["mean_t"] = (mean([e["looks"] for e in ok]), len(ok))
    out["budget"] = (mean([float(e["budget"]) for e in ok]), len(ok))
    kinds = {e["kind"] for e in ok}
    out["n_answered"] = (sum(bool(e["answered"]) for e in ok), len(ok))   # the episodes that end with an answer
    if kinds != {"roboquest"}:                                         # premature: the trainer's answered / premature_rate
        from types import SimpleNamespace
        from plearn.method.trainer import answered, premature_rate
        pe = [SimpleNamespace(commit_ident=None if e.get("premature") is None else not e["premature"]) for e in ok if e["kind"] != "roboquest"]
        a = answered(pe)
        if a: out["premature"] = (premature_rate(pe), len(a))
        elif pe and (kinds == {"domain"} or all(e["rep"] is not None and e["rep"]["ok"] for e in ok)):
            out["premature"] = (ND, 0)                                 # no episode answered (or the game has no answer line): 0 / 0
    ex = [e for e in ok if e.get("premature_engine095") is not None]
    if ex: out["premature_engine095"] = (mean([e["premature_engine095"] for e in ex]), len(ex))
    reps = [e for e in ok if e.get("rep") is not None]
    if reps:
        good = [e for e in reps if e["rep"]["ok"]]
        out["replay_n"] = (len(reps), len(reps)); out["replay_mismatch"] = (len(reps) - len(good), len(reps))
        if good: out["late_looks"] = (sum(e["rep"]["late"] for e in good) / len(good), len(good))
        two = [e for e in good if e["two_line"]]
        for m in ("anchor_exact", "belief_ok", "decision_ok"):
            xs = [t[m] for e in two for t in e["rep"]["turns"] if m in t]
            if xs: out[m] = (mean([float(x) for x in xs]), len(xs))
    op = [x for e in ok if e["two_line"] and e.get("open") is not None for x in e["open"]]
    if op: out["open_line"] = (mean([float(x) for x in op]), len(op))
    th = [t for e in ok for t in e["thoughts"]]
    ch = [t[0] for t in th if t[0] is not None]
    if ch: out["thought_chars_mean"] = (mean(ch), len(ch)); out["thought_chars_median"] = (float(statistics.median(ch)), len(ch))
    cut = [t[1] for t in th]
    if cut: out["thought_cut"] = (mean([float(x) for x in cut]), len(cut))
    n = [t[2] for t in th if t[2] is not None]                        # the thought's tokens (the text was in the row)
    if n:
        out["thought_tokens_mean"] = (mean(n), len(n)); out["thought_tokens_median"] = (float(statistics.median(n)), len(n))
    for m in MEASURES:
        if m not in out or out[m][0] is None: out[m] = (status.get(m, NM), 0)
    return out


def statuses(kind, two_line, has_line_belief, has_text):
    """The status of each measure that has no value: 'not defined' where it cannot exist, 'not measured' where this tool cannot
    compute it from the saved outputs."""
    s = {m: NM for m in MEASURES}
    if kind == "roboquest":
        for m in ("premature", "late_looks", "replay_n", "replay_mismatch") + A5: s[m] = ND
    if kind != "expansion": s["premature_engine095"] = ND
    if kind == "domain" and not has_line_belief: s["decision_ok"] = ND
    if not two_line:
        for m in A5: s[m] = ND
    if not has_text:
        s["thought_tokens_mean"] = s["thought_tokens_median"] = NM
    return s


# --------------------------------------------------------------------- the tokenizer (A6)
def tokenizer_count(path, log):
    """A function text -> tokens of the model's tokenizer (offline, local files only), or None when it cannot load."""
    if not path or not os.path.isdir(path):                           # a local folder only: no download, no hub request
        log(f"tokenizer {path}: no local folder; thoughts in characters only"); return None
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as ex:
        log(f"tokenizer {path}: not loaded ({type(ex).__name__}: {str(ex)[:120]}); thoughts in characters only"); return None
    return lambda text: len(tok.encode(text, add_special_tokens=False))


# --------------------------------------------------------------------- the sources
def read_spec(args):
    """[(folder, model, row, seed, benchmark)] from --spec or --folders / --labels / --seeds / --models."""
    out = []
    if args.spec:
        with open(args.spec) as fh: lines = [l.rstrip("\n").split("\t") for l in fh if l.strip() and not l.startswith("#")]
        head = [h.strip().lower() for h in lines[0]]
        if "folder" in head: lines = lines[1:]
        else: head = ["folder", "model", "row", "seed", "benchmark"]
        for l in lines:
            d = {h: (l[i].strip() if i < len(l) else "") for i, h in enumerate(head)}
            out.append((d["folder"], d.get("model", ""), d.get("row", ""), d.get("seed", "") or "0", d.get("benchmark", "") or "all"))
    else:
        n = len(args.folders); labels = args.labels or [os.path.basename(f.rstrip("/")) for f in args.folders]
        seeds = args.seeds or ["0"] * n; models = args.models or [""] * n
        assert len(labels) == len(seeds) == len(models) == n, "--labels, --seeds and --models need one value per folder"
        out = [(f, m, l, s, "all") for f, l, s, m in zip(args.folders, labels, seeds, models)]
    for f, _, _, _, b in out:
        assert os.path.isfile(os.path.join(f, "results.json")), f"{f}: no results.json"
        assert b == "all" or b in BENCHMARKS, f"{f}: benchmark {b!r} is not one of {BENCHMARKS} or 'all'"
    return out


def read_config(folder):
    with open(os.path.join(folder, "results.json")) as f: return json.load(f).get("config", {})


def git_state():
    def run(*a):
        try: return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as ex: return f"(git failed: {ex})"
    return run("rev-parse", "HEAD"), run("diff", "--stat"), run("status", "--short")


def fmt(v):
    if isinstance(v, str): return v
    if isinstance(v, bool): return str(int(v))
    if isinstance(v, int): return str(v)
    return f"{v:.10g}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folders", nargs="*", default=[]); ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", default=[]); ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--spec", default="", help="a TSV: folder, model, row, seed, benchmark (a header line is optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=3, help="replay worker processes (1-6, default 3; each expansion episode adds one node process: keep 3 or less there)")
    ap.add_argument("--no_decision", action="store_true", help="skip decision_ok (the teacher's lookahead at every turn): 'not measured'")
    ap.add_argument("--limit", type=int, default=0, help="the first N episodes of each task only (a check of the tool)")
    args = ap.parse_args(argv)
    assert args.spec or args.folders, "give --spec or --folders"
    procs = max(1, min(6, args.procs))
    os.makedirs(args.out, exist_ok=True)
    say = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True); t_start = time.time()
    from plearn.api.registry import registry
    reg = registry()
    sources = read_spec(args)
    node = None
    groups = defaultdict(list)                                         # (model, row, seed) -> [(folder, benchmark filter, config)]
    for folder, model, row, seed, bench in sources:
        cfg = read_config(folder)
        groups[(model or str(cfg.get("model") or "model"), row, seed)].append((folder, bench, cfg))

    import multiprocessing as mp
    pool = mp.get_context("spawn").Pool(procs) if procs > 1 else None
    lines, notes, fidelity, cpu, work = [], [], [], {"seconds": 0.0, "episodes": 0}, []
    tok_cache = {}
    for (model, rlabel, seed), items in groups.items():
        per_task = {}                                                  # task -> [benchmark, kind, rows, two_line, replays, status, folder, form]
        for folder, bench, cfg in items:
            names = [n for n in (cfg.get("tasks") or []) if os.path.exists(os.path.join(folder, f"{n}.jsonl"))]
            names += sorted(p[:-len(".jsonl")] for p in os.listdir(folder) if p.endswith(".jsonl") and p[:-len(".jsonl")] not in names)
            path = cfg.get("path")                                    # the weights folder of the eval (plearn/cli.py config)
            if path not in tok_cache: tok_cache[path] = None if path is None else "pending"
            for name in names:
                try: b, kind, env_args = task_info(name, cfg, reg)
                except Exception as ex: notes.append(f"{folder}: task {name} skipped ({type(ex).__name__}: {ex})"); continue
                if b is None: notes.append(f"{folder}: task {name} skipped (not in {', '.join(BENCHMARKS)})"); continue
                if bench != "all" and b != bench: continue
                if name in per_task: notes.append(f"{folder}: task {name} skipped (already read from {per_task[name][6]})"); continue
                with open(os.path.join(folder, f"{name}.jsonl")) as fh: rows = [json.loads(l) for l in fh if l.strip()]
                if args.limit: rows = rows[:args.limit]
                has_text = any("thought" in t for r in rows for t in r.get("turn_log", []))
                if has_text and tok_cache.get(path) == "pending": tok_cache[path] = tokenizer_count(path, say)
                tokc = tok_cache.get(path) if has_text else None
                for r in rows:                                         # the thought's tokens now, then the text leaves the row
                    for t in r.get("turn_log", []):
                        x = t.pop("thought", None)
                        if x is not None and tokc is not None: t["_tokens"] = tokc(x)
                form = turn_form(env_args, [r for r in rows if "error" not in r]); two = form == "two_line"
                hlb = False
                do_replay = kind == "domain" or (kind == "expansion" and (node if node is not None else (node := node_ok())))
                if kind == "domain":
                    import importlib
                    hlb = hasattr(importlib.import_module(env_args["domain"]).Domain, "line_belief")
                if kind == "expansion" and not do_replay: notes.append(f"{folder}: {name}: no node engine (NODE_BIN): the replay measures are not measured")
                st = statuses(kind, two, hlb or kind == "expansion", has_text)
                if args.no_decision: st["decision_ok"] = NM if st["decision_ok"] != ND else ND
                pend = None
                if do_replay:                                          # every task's replays go to the pool at once
                    jobs = [(kind, env_args, r, two, not args.no_decision) for r in rows if "error" not in r]
                    say(f"{model} {rlabel} seed {seed}: {name}: replay of {len(jobs)} episodes ({form})")
                    pend = pool.map_async(replay, jobs, chunksize=1) if pool else list(map(replay, jobs))
                per_task[name] = [b, kind, rows, two, pend, st, folder, form]
        work.append((model, rlabel, seed, per_task))
    for model, rlabel, seed, per_task in work:
        by_bench = defaultdict(list)
        for name, (b, kind, rows, two, pend, st, folder, form) in per_task.items():
            reps = [None] * len(rows)
            if pend is not None:
                res = pend if isinstance(pend, list) else pend.get()
                it = iter(res); reps = [None if "error" in r else next(it) for r in rows]
                bad = [x for x in res if not x["ok"]]
                cpu["seconds"] += sum(x.get("seconds", 0.0) for x in res); cpu["episodes"] += len(res)
                fidelity.append((model, rlabel, seed, name, len(res), len(bad), bad[0]["why"] if bad else ""))
            facts = [episode_facts(kind, r, rep, two) for r, rep in zip(rows, reps)]
            vals = aggregate(facts, st)
            lines += [(model, b, name, rlabel, seed, m, vals[m][0], vals[m][1]) for m in MEASURES]
            by_bench[b].append((facts, st))
        for b, parts in by_bench.items():                              # the benchmark's episodes pooled
            facts = [f for fs, _ in parts for f in fs]
            st = {m: (ND if all(s[m] == ND for _, s in parts) else NM) for m in MEASURES}
            vals = aggregate(facts, st)
            lines += [(model, b, "overall", rlabel, seed, m, vals[m][0], vals[m][1]) for m in MEASURES]
    if pool: pool.close(); pool.join()
    say(f"replayed {cpu['episodes']} episodes in {time.time() - t_start:.0f} s")

    seeds_of = defaultdict(dict)                                       # the mean of the seeds
    tasks_of = defaultdict(set)                                        # the tasks of each seed (for the "overall" mean)
    for model, b, task, rlabel, seed, m, v, n in lines:
        seeds_of[(model, b, task, rlabel, m)][seed] = (v, n)
        if task != "overall": tasks_of[(model, b, rlabel, seed)].add(task)
    means = []
    for (model, b, task, rlabel, m), d in seeds_of.items():
        if len(d) < 2: continue
        if task == "overall" and len({frozenset(tasks_of[(model, b, rlabel, s)]) for s in d}) > 1:
            means.append((model, b, task, rlabel, "mean", m, NM, 0)); continue   # the seeds have different tasks: no mean
        xs = [v for v, _ in d.values() if not isinstance(v, str)]
        strs = [v for v, _ in d.values() if isinstance(v, str)]
        means.append((model, b, task, rlabel, "mean", m, sum(xs) / len(xs) if xs else (ND if ND in strs else NM), len(xs)))
    allrows = lines + means
    order = {m: i for i, m in enumerate(MEASURES)}
    files = []
    for model in sorted({x[0] for x in allrows}):
        fn = os.path.join(args.out, re.sub(r"[^A-Za-z0-9._-]+", "_", model) + ".csv"); files.append(fn)
        rs = sorted([x for x in allrows if x[0] == model], key=lambda x: (BENCHMARKS.index(x[1]), x[2] == "overall", x[2], x[3],
                                                                             x[4] == "mean", x[4], order[x[5]]))
        with open(fn, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["benchmark", "task", "row", "seed", "measure", "value", "n"])
            for _, b, task, rlabel, seed, m, v, n in rs: w.writerow([b, task, rlabel, seed, m, fmt(v), n])
    head, diff, status = git_state()
    per100 = 100 * cpu["seconds"] / cpu["episodes"] if cpu["episodes"] else None
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write("# Test measures A4 / A5 / A6 (tools/testpass_a4a5.py)\n\n")
        f.write(f"Made {time.strftime('%Y-%m-%d %H:%M:%S %Z')} on {os.uname().nodename}. Command: `{' '.join([sys.executable, *sys.argv])}`\n\n")
        f.write(f"Code: {REPO}, commit {head}.\n\n`git diff --stat`:\n```\n{diff or '(no change)'}\n```\n`git status --short`:\n```\n{status or '(clean)'}\n```\n\n")
        f.write("## Sources\n\n| folder | model | row | seed | benchmark | results.json config |\n|---|---|---|---|---|---|\n")
        for folder, model, row, seed, bench in sources:
            cfg = read_config(folder)
            c = {k: cfg.get(k) for k in ("model", "path", "adapter", "split", "task_args", "n", "minutes")}
            f.write(f"| {folder} | {model or cfg.get('model')} | {row} | {seed} | {bench} | `{json.dumps(c)}` |\n")
        f.write("\n## Files\n\n" + "\n".join(f"- {x}" for x in files) + "\n\n")
        f.write("## Replay fidelity\n\nEpisodes replayed and episodes whose replay differs from the stored row (left out of the replay "
                "measures), per task.\n\n| model | row | seed | task | replayed | mismatch | first reason |\n|---|---|---|---|---|---|---|\n")
        for x in fidelity: f.write("| " + " | ".join(map(str, x)) + " |\n")
        f.write(f"\nReplay CPU time: {cpu['seconds']:.1f} s for {cpu['episodes']} episodes"
                + (f" ({per100:.1f} s per 100 episodes, process time of the workers).\n" if per100 is not None else ".\n"))
        if notes: f.write("\n## Notes\n\n" + "\n".join(f"- {x}" for x in notes) + "\n")
        f.write("\n## Definitions\n\n```\n" + "The measures" + __doc__.split("\nThe measures", 1)[1].rstrip() + "\n```\n")
    say(f"wrote {', '.join(files)} and {os.path.join(args.out, 'README.md')}")
    return files


if __name__ == "__main__":
    main()
