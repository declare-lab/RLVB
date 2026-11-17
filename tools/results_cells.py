#!/usr/bin/env python3
"""The cells of one column of a results table (a Markdown README with one table per benchmark and model) from the test folders of a
model: one folder a test seed (`python -m plearn eval`: <folder>/results.json), as scripts/eval_<model>.sh writes them
($PL_RUNS/eval/<TAG>...-seed<k>).

    python tools/results_cells.py $PL_RUNS/eval/<TAG>-games-seed0 $PL_RUNS/eval/<TAG>-games-seed1 $PL_RUNS/eval/<TAG>-games-seed2
    python tools/results_cells.py --group test_suite/hidden_games 0=<folder> 0=<folder of the tabletop units of seed 0> 1=<folder> 2=<folder>
    python tools/results_cells.py --tasks minesweeper_easy,minigrid_keycorridor <folders>

A folder's seed is the number after "seed" at the end of its name, or `<k>=<folder>`. Several folders of one seed are joined (a
seed whose tabletop units ran on another host); a task in two folders of one seed is an error. The rows are the tasks of --group
(default test_suite/hidden_games, the 7 HiddenGames test tasks, in the registry's order) or of --tasks.

The rules of a cell:
  - a value is the accuracy in percent with one decimal (the task's `success`);
  - a cell is `**mean**<br><sub>s0 · s1 · s2</sub>`: the mean of the done seeds in bold (of the exact values, then rounded), then
    the seed values; a seed that is not done is "–"; a cell with no done seed is "–";
  - a seed is done when every task of the rows is there with 0 error rows (`errors` of results.json);
  - Overall: for each seed, all games of the rows together (correct games / games), then the mean of the done seeds.
The tool prints the rows `| task | cell |` and the Overall row for the column, then the checks of each seed (games, errors).

With --readme it also writes the cells into a results table of that file (a Markdown file with such tables):
    python tools/results_cells.py <folders> --readme <file>.md --section "3." \
        --model "Qwen3-VL-8B-Thinking" --column "RLVB (GRPO), belief + action"
The table is the first one under the heading `#### <model>` in the section whose heading starts with `### <section>`. The column
is the one with that header text; when the table does not have it, it is added at the right, with "–" in the rows this call
does not give. Only that table changes. Then: the note line under the table, `git pull --rebase origin results` and the push are yours
(the results README's rules)."""
from __future__ import annotations
import argparse, json, os, re, sys

DASH = "–"


def read_seed(folders):
    """{task: {"episodes", "errors", "success"}} of one seed's folders (the module text)."""
    out = {}
    for d in folders:
        with open(os.path.join(d, "results.json")) as f: res = json.load(f)["results"]
        for t, r in res.items():
            if t in out: raise SystemExit(f"{t}: in two folders of one seed ({d})")
            out[t] = {"episodes": int(r.get("episodes", 0)), "errors": int(r.get("errors", 0)), "success": r.get("success")}
    return out


def seed_done(rows, tasks):
    """(done, reasons): every task of the rows is there with a success value and 0 error rows."""
    why = [f"{t}: no result" for t in tasks if t not in rows or rows[t]["success"] is None]
    why += [f"{t}: {rows[t]['errors']} error rows" for t in tasks if t in rows and rows[t]["errors"]]
    return not why, why


def overall(rows, tasks):
    """All games of the tasks together: correct games / games (the error rows are not games; a done seed has none)."""
    n = sum(rows[t]["episodes"] - rows[t]["errors"] for t in tasks)
    return sum(rows[t]["success"] * (rows[t]["episodes"] - rows[t]["errors"]) for t in tasks) / n if n else None


def cell(values):
    """The README's cell of the seed values (None = a seed that is not done), in percent."""
    done = [v for v in values if v is not None]
    if not done: return DASH
    return f"**{100 * sum(done) / len(done):.1f}**<br><sub>" + " · ".join(DASH if v is None else f"{100 * v:.1f}" for v in values) + "</sub>"


def column(seeds, tasks, n_seeds=3):
    """seeds: {seed number: {task: row}}. Returns ([(row name, cell)], {seed: (done, reasons, games)})."""
    state = {}
    for k in range(n_seeds):
        rows = seeds.get(k)
        if rows is None: state[k] = (False, ["no folder"], 0); continue
        ok, why = seed_done(rows, tasks)
        state[k] = (ok, why, sum(rows[t]["episodes"] for t in tasks if t in rows))
    val = lambda k, t: seeds[k][t]["success"] if state[k][0] else None
    out = [(t, cell([val(k, t) for k in range(n_seeds)])) for t in tasks]
    out.append(("**Overall**", cell([overall(seeds[k], tasks) if state[k][0] else None for k in range(n_seeds)])))
    return out, state


def _cells(line): return [c.strip() for c in line.strip().strip("|").split("|")]


def put_column(text, section, model, column, rows):
    """The README text with the cells `rows` ({row name: cell}) in the column `column` of the table under `#### model` in the
    section `### section...` (the module text). Returns (text, column added, [row names written])."""
    lines = text.split("\n")
    i = next((k for k, l in enumerate(lines) if l.startswith("### ") and l[4:].startswith(section)), None)
    if i is None: raise SystemExit(f"no section heading '### {section}...'")
    end = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("### ") or lines[k].startswith("## ")), len(lines))
    j = next((k for k in range(i + 1, end) if lines[k].startswith("#### ") and lines[k][5:].strip() == model), None)
    if j is None: raise SystemExit(f"no table heading '#### {model}' in the section '{lines[i]}'")
    h = next((k for k in range(j + 1, end) if lines[k].startswith("|")), None)
    if h is None or h + 1 >= len(lines) or not set(lines[h + 1].replace(" ", "")) <= set("|-:"): raise SystemExit(f"no table under '#### {model}'")
    head = _cells(lines[h]); added = column not in head
    if added:
        head.append(column); lines[h] = "| " + " | ".join(head) + " |"
        lines[h + 1] = lines[h + 1].rstrip().rstrip("|") + "|---:|"
    c = head.index(column); k = h + 2; written = []
    while k < len(lines) and lines[k].startswith("|"):
        cells = _cells(lines[k])
        if added: cells.append(DASH)
        if len(cells) != len(head): raise SystemExit(f"a table row with {len(cells)} cells under {len(head)} headers: {lines[k][:80]}")
        if cells[0] in rows: cells[c] = rows[cells[0]]; written.append(cells[0])
        if added or cells[0] in rows: lines[k] = "| " + " | ".join(cells) + " |"
        k += 1
    miss = [r for r in rows if r not in written]
    if miss: raise SystemExit(f"rows not in the table '#### {model}': {miss}")
    return "\n".join(lines), added, written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folders", nargs="+", help="test folders: <folder ending in seed<k>> or <k>=<folder>")
    ap.add_argument("--group", default="test_suite/hidden_games", help="the rows: the tasks of this registered group (default test_suite/hidden_games, the 7 games)")
    ap.add_argument("--tasks", default="", help="the rows as a comma list of tasks (in place of --group)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--readme", default="", help="write the cells into this results README (with --section, --model, --column)")
    ap.add_argument("--section", default="", help="the section heading's start, after '### ', for example '3.'")
    ap.add_argument("--model", default="", help="the table heading after '#### ', for example 'Qwen3-VL-8B-Thinking'")
    ap.add_argument("--column", default="", help="the column header, for example 'RLVB (GRPO), belief + action'")
    a = ap.parse_args(argv)
    if a.readme and not (a.section and a.model and a.column): raise SystemExit("--readme needs --section, --model and --column")
    by = {}
    for f in a.folders:
        m = re.fullmatch(r"(\d+)=(.+)", f)
        if m: k, d = int(m.group(1)), m.group(2)
        else:
            m = re.search(r"seed(\d+)/?$", f)
            if not m: raise SystemExit(f"{f}: no seed in the name; write <k>={f}")
            k, d = int(m.group(1)), f
        if not os.path.isfile(os.path.join(d, "results.json")): raise SystemExit(f"{d}: no results.json (the test of this seed is not complete)")
        by.setdefault(k, []).append(d)
    if a.tasks: tasks = [t for t in a.tasks.split(",") if t]
    else:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from plearn.api.registry import registry
        tasks = list(registry().tasks(a.group))
    seeds = {k: read_seed(ds) for k, ds in by.items()}
    rows, state = column(seeds, tasks, a.seeds)
    for name, c in rows: print(f"| {name} | {c} |")
    print()
    for k in range(a.seeds):
        ok, why, games = state[k]
        print(f"seed {k}: {'done' if ok else 'NOT done'}, {games} games" + ("" if ok else ": " + "; ".join(why)) + (f"  ({', '.join(by[k])})" if k in by else ""))
    if a.readme:
        with open(a.readme) as f: text = f.read()
        text, added, written = put_column(text, a.section, a.model, a.column, dict(rows))
        with open(a.readme, "w") as f: f.write(text)
        print(f"\n{a.readme}: column '{a.column}' {'added to' if added else 'updated in'} the table '{a.model}' of section '{a.section}': {len(written)} rows")
    return 0 if all(s[0] for s in state.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
