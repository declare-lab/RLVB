#!/usr/bin/env python
"""The model outputs of the tests, one row per turn (what the model generated during the test). CPU only: no model, no GPU. It reads the saved TEST outputs of `plearn eval` (<task>.jsonl, one row per episode, with
turn_log; results.json with the config) and writes them as they are, in a table form. It computes nothing.

    python tools/test_outputs_csv.py --spec runs.tsv --out DIR      (the TSV of tools/testpass_a4a5.py: folder, model, row, seed, benchmark)

Output, one pair of files per (model, row, seed), name <slug> = <row>_seed<k> in a file-safe form:
  DIR/<slug>.turns.csv.gz      one line per turn. Columns: model, row, seed, benchmark, task, episode (the line number in
                               <task>.jsonl, from 0), spec (the episode spec, JSON), turn (from 0), kind (look / answer / invalid,
                               as the episode recorded it; "error" = an episode that ended with an error, one line), belief (the
                               belief line; empty in the action-only turn), action (the action line), thought_chars, thought_cut
                               (1 = the thinking cap cut the thought), has_thought (1 = the thought text is in the thoughts file),
                               ep_success, ep_submitted, ep_answer, ep_turns (the episode's values), extra (the other keys of the
                               turn record, JSON: reading, act_ok, open, line, read, reply_cut, ...), feedback (RoboQuest only: the
                               simulator's answer to this turn's evidence action, as the model saw it in its next turn,
                               "<action>: <status> <reading>", from the episode's `actions` record; empty for SUBMIT turns and for
                               the other benchmarks, whose feedback is a picture).
  DIR/thoughts/<slug>/<task>.part<N>.jsonl.gz   one line per turn whose record holds the thought text (tests with
                               PL_SAVE_THOUGHTS=1): {model, row, seed, benchmark, task, episode, turn, thought}. One file per task;
                               a new part starts at 40 MB compressed (GitHub refuses files above 100 MB). No file: no thought text.
  DIR/MANIFEST.csv             one line per output file: model, row, seed, benchmark, task (empty for the turns file), the source
                               folder, episodes, error episodes, turns, turns with thought text, file,
                               bytes, sha256.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, io, json, os, re, sys, time
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""                              # CPU only, always
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from testpass_a4a5 import read_spec, read_config, task_info      # the same sources and benchmarks as the post-pass

COLS = ("model", "row", "seed", "benchmark", "task", "episode", "spec", "turn", "kind", "belief", "action", "thought_chars",
        "thought_cut", "has_thought", "ep_success", "ep_submitted", "ep_answer", "ep_turns", "extra", "feedback")
TURN_KEYS = {"belief", "action", "kind", "thought", "thought_chars", "thought_cut"}
PART_BYTES = 40_000_000                                              # compressed bytes of one thoughts file


class Parts:
    """Gzip JSONL parts of one task: <base>.part<N>.jsonl.gz, a new part at PART_BYTES compressed. files() = the written paths."""
    def __init__(self, base): self.base, self.n, self.raw, self.gz, self.paths = base, 0, None, None, []
    def write(self, line):
        if self.gz is None or self.raw.tell() >= PART_BYTES:
            self.close(); self.n += 1; p = f"{self.base}.part{self.n}.jsonl.gz"
            assert not os.path.exists(p), f"{p}: output exists (no overwrite)"
            os.makedirs(os.path.dirname(p), exist_ok=True)
            self.raw = open(p, "wb"); self.gz = gzip.GzipFile(fileobj=self.raw, mode="wb", compresslevel=9); self.paths.append(p)
        self.gz.write(line.encode("utf-8"))
    def close(self):
        if self.gz is not None: self.gz.close(); self.raw.close(); self.gz = self.raw = None


def slug(row, seed):
    return re.sub(r"[^A-Za-z0-9.]+", "-", f"{row}").strip("-").lower() + f"_seed{seed}"


def rq_feedback(r, where):
    """RoboQuest: turn index -> "<action>: <status> <reading>". Turn k that is not SUBMIT is entry j of the episode's action log
    (bench_eval.Episode.step: every decision that is not SUBMIT appends one entry, also an invalid one). Stops on a break of this rule."""
    body = lambda a: (lambda t: t[len("ACTION:"):].strip() if t.upper().startswith("ACTION:") else t)((a or "").strip())
    acts, j, out = r.get("actions") or [], 0, {}
    for ti, t in enumerate(r.get("turn_log") or []):
        a = body(t.get("action"))
        if a.upper().startswith("SUBMIT"): continue
        assert j < len(acts) and acts[j]["action"] == a, f"{where} turn {ti}: action log differs"
        x = acts[j]; j += 1
        out[ti] = f"{x['action']}: {' '.join(s for s in (x['status'], x['reading']) if s)}"
    assert j == len(acts), f"{where}: {len(acts) - j} action log entries without a turn"
    return out


def num(v):
    if v is None: return ""
    if isinstance(v, bool): return str(int(v))
    return str(v)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folders", nargs="*", default=[]); ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", default=[]); ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--spec", default="", help="a TSV: folder, model, row, seed, benchmark (a header line is optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="the first N episodes of each task only (a check of the tool)")
    args = ap.parse_args(argv)
    assert args.spec or args.folders, "give --spec or --folders"
    os.makedirs(args.out, exist_ok=True)
    say = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
    from plearn.api.registry import registry
    reg = registry()
    groups = defaultdict(list)
    for folder, model, row, seed, bench in read_spec(args):
        cfg = read_config(folder)
        groups[(model or str(cfg.get("model") or "model"), row, seed)].append((folder, bench, cfg))
    man = []
    for (model, rlabel, seed), items in groups.items():
        name = slug(rlabel, seed)
        p_turns = os.path.join(args.out, name + ".turns.csv.gz")
        assert not os.path.exists(p_turns), f"{name}: output exists (no overwrite)"
        seen, mrows, mth = {}, [], []
        with gzip.open(p_turns, "wt", newline="", encoding="utf-8") as ft:
            w = csv.writer(ft); w.writerow(COLS)
            for folder, bench, cfg in items:
                names = [n for n in (cfg.get("tasks") or []) if os.path.exists(os.path.join(folder, f"{n}.jsonl"))]
                names += sorted(p[:-6] for p in os.listdir(folder) if p.endswith(".jsonl") and p[:-6] not in names)
                for task in names:
                    try: b, _, _ = task_info(task, cfg, reg)
                    except Exception as ex: say(f"{folder}: task {task} skipped ({type(ex).__name__}: {ex})"); continue
                    if b is None or (bench != "all" and b != bench): continue
                    if task in seen: say(f"{folder}: task {task} skipped (already read from {seen[task]})"); continue
                    seen[task] = folder
                    n_ep = n_err = n_turn = n_text = 0
                    fth = Parts(os.path.join(args.out, "thoughts", name, task))
                    with open(os.path.join(folder, f"{task}.jsonl")) as fh:
                        for ei, l in enumerate(x for x in fh if x.strip()):
                            if args.limit and ei >= args.limit: break
                            r = json.loads(l); n_ep += 1
                            base = [model, rlabel, seed, b, task, ei, json.dumps(r.get("spec"))]
                            ep = [num(r.get("success")), num(r.get("submitted")), num(r.get("answer")), num(r.get("turns"))]
                            if "error" in r:
                                n_err += 1; w.writerow(base + ["", "error", "", "", "", "", "0"] + ep + [json.dumps({"error": r["error"]}), ""]); continue
                            fb = rq_feedback(r, f"{name}: {task} episode {ei}") if b == "RoboQuest" else {}
                            for ti, t in enumerate(r.get("turn_log") or []):
                                text = t.get("thought"); n_turn += 1
                                if text is not None:
                                    n_text += 1
                                    fth.write(json.dumps({"model": model, "row": rlabel, "seed": seed, "benchmark": b, "task": task,
                                                          "episode": ei, "turn": ti, "thought": text}, ensure_ascii=False) + "\n")
                                extra = {k: v for k, v in t.items() if k not in TURN_KEYS}
                                w.writerow(base + [ti, t.get("kind", ""), t.get("belief", ""), t.get("action", ""),
                                                   num(t.get("thought_chars")), num(t.get("thought_cut")), "1" if text is not None else "0"]
                                           + ep + [json.dumps(extra, ensure_ascii=False) if extra else "", fb.get(ti, "")])
                    fth.close()
                    info = [model, rlabel, seed, b, task, folder, n_ep, n_err, n_turn, n_text]
                    mrows.append(info); mth += [info + [os.path.relpath(p, args.out), os.path.getsize(p), sha(p)] for p in fth.paths]
                    say(f"{name}: {task}: {n_ep} episodes ({n_err} errors), {n_turn} turns, {n_text} with thought text")
        tot = [sum(int(m[k]) for m in mrows) for k in (6, 7, 8, 9)]
        files = sorted({m[5] for m in mrows})
        man.append([model, rlabel, seed, ";".join(sorted({m[3] for m in mrows})), "", ";".join(files)] + tot + [os.path.basename(p_turns), os.path.getsize(p_turns), sha(p_turns)])
        man += mth
        big = max([os.path.getsize(p_turns)] + [m[-2] for m in mth])
        say(f"{name}: turns {os.path.getsize(p_turns) / 1e6:.1f} MB, thoughts {sum(m[-2] for m in mth) / 1e6:.1f} MB in {len(mth)} files, largest file {big / 1e6:.1f} MB")
    mp = os.path.join(args.out, "MANIFEST.csv"); new = not os.path.exists(mp)
    with open(mp, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["model", "row", "seed", "benchmark", "task", "folder", "episodes",
                            "error_episodes", "turns", "turns_with_thought", "file", "bytes", "sha256"])
        w.writerows(man)
    say(f"END: {len(groups)} model-seeds, {len(man)} task lines -> {args.out}")


if __name__ == "__main__":
    main()
