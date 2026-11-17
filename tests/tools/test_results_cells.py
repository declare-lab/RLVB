"""tools/results_cells.py (CPU, no model): the cells of a results-table column from the test folders of 3 seeds, by the rules of
the results README: percent with one decimal, the mean of the exact seed values, Overall = all games together, a seed with an
error row or a missing task is "–".

    PYTHONPATH=. python -m unittest tests.tools.test_results_cells"""
import contextlib, io, json, os, sys, tempfile, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import results_cells as RC   # noqa: E402


def folder(root, name, results):
    d = os.path.join(root, name); os.makedirs(d)
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump({"results": {t: {"episodes": n, "errors": e, "success": s} for t, (n, e, s) in results.items()}, "groups": {}}, f)
    return d


class Cells(unittest.TestCase):
    def setUp(self): self.tmp = tempfile.TemporaryDirectory()
    def tearDown(self): self.tmp.cleanup()

    def run_tool(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out): rc = RC.main(list(argv))
        return rc, out.getvalue()

    def test_the_cells_of_the_8b_untrained_column(self):
        """The 7 games of the results README's 8B table, Base (w/ belief): the counts behind its cells
        (minigrid 37 / 35 / 37 of 100, ..., frozenlake and sokoban of 128) give the cells of that table, Overall 24.9."""
        games = {"minesweeper_easy": (100, [2, 1, 2]), "minigrid_keycorridor": (100, [37, 35, 37]),
                 "frozenlake_hidden_goal": (128, [54, 55, 51]), "sokoban_hidden_target": (128, [34, 30, 30]), "tabletop_lamps": (100, [3, 2, 1]),
                 "tabletop_cups": (100, [42, 40, 40]), "tabletop_blocks": (100, [22, 21, 25])}
        ds = [folder(self.tmp.name, f"untrained-games-seed{k}", {t: (n, 0, c[k] / n) for t, (n, c) in games.items()}) for k in range(3)]
        rc, out = self.run_tool("--tasks", ",".join(games), *ds)
        self.assertEqual(rc, 0)
        rows = dict(l.strip("| \n").split(" | ") for l in out.splitlines() if l.startswith("| "))
        self.assertEqual(rows["minigrid_keycorridor"], "**36.3**<br><sub>37.0 · 35.0 · 37.0</sub>")
        self.assertEqual(rows["frozenlake_hidden_goal"], "**41.7**<br><sub>42.2 · 43.0 · 39.8</sub>")
        self.assertEqual(rows["sokoban_hidden_target"], "**24.5**<br><sub>26.6 · 23.4 · 23.4</sub>")
        self.assertEqual(rows["**Overall**"], "**24.9**<br><sub>25.7 · 24.3 · 24.6</sub>")       # 194, 184, 186 of 756: not the mean of the rows
        self.assertEqual(list(rows), list(games) + ["**Overall**"])
        self.assertIn("seed 0: done, 756 games", out); self.assertIn("seed 2: done, 756 games", out)

    def test_a_seed_with_an_error_row_or_a_missing_task_is_a_dash(self):
        d0 = folder(self.tmp.name, "a-seed0", {"minigrid_keycorridor": (100, 0, 0.5), "minesweeper_easy": (100, 0, 0.01)})
        d1 = folder(self.tmp.name, "a-seed1", {"minigrid_keycorridor": (100, 2, 0.6), "minesweeper_easy": (100, 0, 0.0)})     # 2 error rows
        d2 = folder(self.tmp.name, "a-seed2", {"minigrid_keycorridor": (100, 0, 0.7)})                                          # a task is missing
        rc, out = self.run_tool("--tasks", "minigrid_keycorridor,minesweeper_easy", d0, d1, d2)
        self.assertEqual(rc, 1)
        self.assertIn("| minigrid_keycorridor | **50.0**<br><sub>50.0 · – · –</sub> |", out)
        self.assertIn("| **Overall** | **25.5**<br><sub>25.5 · – · –</sub> |", out)
        self.assertIn("seed 1: NOT done, 200 games: minigrid_keycorridor: 2 error rows", out); self.assertIn("seed 2: NOT done, 100 games: minesweeper_easy: no result", out)
        rc, out = self.run_tool("--tasks", "minigrid_keycorridor", d0)                         # seeds with no folder
        self.assertEqual(rc, 1); self.assertIn("<sub>50.0 · – · –</sub>", out); self.assertIn("seed 1: NOT done, 0 games: no folder", out)

    def test_the_folders_of_one_seed_are_joined(self):
        """A seed whose tabletop units ran on another host: two folders, given as <k>=<folder>."""
        a = folder(self.tmp.name, "games", {"minigrid_keycorridor": (100, 0, 0.5)}); b = folder(self.tmp.name, "tabletop", {"tabletop_cups": (100, 0, 0.25)})
        s1 = folder(self.tmp.name, "x-seed1", {"minigrid_keycorridor": (100, 0, 0.6), "tabletop_cups": (100, 0, 0.35)})
        s2 = folder(self.tmp.name, "x-seed2", {"minigrid_keycorridor": (100, 0, 0.7), "tabletop_cups": (100, 0, 0.45)})
        rc, out = self.run_tool("--tasks", "minigrid_keycorridor,tabletop_cups", f"0={a}", f"0={b}", s1, s2)
        self.assertEqual(rc, 0)
        self.assertIn("| tabletop_cups | **35.0**<br><sub>25.0 · 35.0 · 45.0</sub> |", out); self.assertIn("| **Overall** | **47.5**<br><sub>37.5 · 47.5 · 57.5</sub> |", out)
        with self.assertRaises(SystemExit): self.run_tool("--tasks", "minigrid_keycorridor", f"0={a}", f"0={a}")       # a task twice in one seed
        with self.assertRaises(SystemExit): self.run_tool("--tasks", "minigrid_keycorridor", a)                        # no seed in the name
        os.makedirs(os.path.join(self.tmp.name, "empty-seed0"))
        with self.assertRaises(SystemExit): self.run_tool("--tasks", "minigrid_keycorridor", os.path.join(self.tmp.name, "empty-seed0"))   # no results.json

    README = """# PRO Worlds

## Results

### 2. Pro-Worlds, normal board

#### Qwen3-VL-8B-Thinking, normal board

| Subtask | Ours (PL Method), belief + action | Base, action only |
|---|---:|---:|
| Face | **10.0**<br><sub>10.0 · 10.0 · 10.0</sub> | – |

### 3. HiddenGames (7 tasks; trained on all)

#### Qwen3.5-4B

| Subtask | Ours (PL Method), belief + action | Base, action only |
|---|---:|---:|
| minigrid_keycorridor | **42.7**<br><sub>40.0 · 43.0 · 45.0</sub> | – |
| tabletop_cups | **54.7**<br><sub>59.0 · 46.0 · 59.0</sub> | – |
| **Overall** | **40.7**<br><sub>41.9 · 39.7 · 40.4</sub> | – |

756 games, as the 8B.

#### Qwen3-VL-8B-Thinking

| Subtask | Ours (PL Method), belief + action | Base, action only |
|---|---:|---:|
| minigrid_keycorridor | **76.0**<br><sub>75.0 · 74.0 · 79.0</sub> | **47.5**<br><sub>47.0 · 48.0 · –</sub> |
| tabletop_cups | **60.7**<br><sub>61.0 · 59.0 · 62.0</sub> | **24.7**<br><sub>24.0 · 25.0 · 25.0</sub> |
| **Overall** | **34.9**<br><sub>35.0 · 35.0 · 34.7</sub> | **24.6**<br><sub>24.9 · 24.2 · –</sub> |

756 games (100 each).

## Tasks
"""

    def test_the_cells_go_into_the_table_of_the_results_readme(self):
        """--readme: a new column at the right of the one table named (section 3, the 8B), its cells written; a second call
        updates the cells; no other line of the file changes."""
        readme = os.path.join(self.tmp.name, "README.md")
        with open(readme, "w") as f: f.write(self.README)
        ds = [folder(self.tmp.name, f"vb-seed{k}", {"minigrid_keycorridor": (100, 0, 0.5 + 0.1 * k), "tabletop_cups": (100, 0, 0.25)}) for k in range(2)]
        args = ("--tasks", "minigrid_keycorridor,tabletop_cups", "--readme", readme, "--section", "3. HiddenGames", "--model", "Qwen3-VL-8B-Thinking",
                "--column", "RLVB (GRPO), belief + action")
        rc, out = self.run_tool(*ds, *args)
        self.assertEqual(rc, 1); self.assertIn("column 'RLVB (GRPO), belief + action' added to the table 'Qwen3-VL-8B-Thinking'", out)   # seed 2 is not there yet
        new = open(readme).read().split("\n"); old = self.README.split("\n")
        changed = [k for k, (a, b) in enumerate(zip(old, new)) if a != b]
        first = old.index("#### Qwen3-VL-8B-Thinking")
        self.assertEqual(changed, [first + 2, first + 3, first + 4, first + 5, first + 6]); self.assertEqual(len(old), len(new))   # that table only
        self.assertEqual(new[first + 2], "| Subtask | Ours (PL Method), belief + action | Base, action only | RLVB (GRPO), belief + action |")
        self.assertEqual(new[first + 3], "|---|---:|---:|---:|")
        self.assertEqual(new[first + 4], "| minigrid_keycorridor | **76.0**<br><sub>75.0 · 74.0 · 79.0</sub> | **47.5**<br><sub>47.0 · 48.0 · –</sub> | **55.0**<br><sub>50.0 · 60.0 · –</sub> |")
        self.assertTrue(new[first + 6].endswith("| **40.0**<br><sub>37.5 · 42.5 · –</sub> |"))      # 75 and 85 of 200 games
        ds.append(folder(self.tmp.name, "vb-seed2", {"minigrid_keycorridor": (100, 0, 0.7), "tabletop_cups": (100, 0, 0.25)}))
        rc, out = self.run_tool(*ds, *args)                                        # the third seed: the column is updated, not added again
        self.assertEqual(rc, 0); self.assertIn("updated in the table", out)
        new = open(readme).read().split("\n")
        self.assertEqual(new[first + 2].count("RLVB (GRPO), belief + action"), 1)
        self.assertTrue(new[first + 4].endswith("| **60.0**<br><sub>50.0 · 60.0 · 70.0</sub> |"))
        self.assertIn("| minigrid_keycorridor | **42.7**<br><sub>40.0 · 43.0 · 45.0</sub> | – |", new)       # the 4B table of the section: as it was
        for bad in (("--section", "9. none"), ("--model", "Qwen9")):
            a2 = list(args); a2[a2.index(bad[0]) + 1] = bad[1]
            with self.assertRaises(SystemExit): self.run_tool(*ds, *a2)
        with self.assertRaises(SystemExit): self.run_tool(*ds, "--tasks", "minigrid_keycorridor,minesweeper_easy", *args[2:])   # a row the table does not have
        with self.assertRaises(SystemExit): self.run_tool(*ds, "--tasks", "minigrid_keycorridor", "--readme", readme)           # --readme alone

    def test_the_default_rows_are_the_7_test_games(self):
        from plearn.api.registry import registry
        tasks = list(registry().tasks("test_suite/hidden_games"))
        self.assertEqual(len(tasks), 7); self.assertNotIn("minesweeper_medium", tasks)
        self.assertEqual(set(tasks), set(registry().tasks("train_suite")) & set(registry().tasks("test_suite")))
        ds = [folder(self.tmp.name, f"g-seed{k}", {t: (100, 0, 0.5) for t in tasks}) for k in range(3)]
        rc, out = self.run_tool(*ds)
        self.assertEqual(rc, 0); self.assertEqual(sum(l.startswith("| ") for l in out.splitlines()), 8)


if __name__ == "__main__":
    unittest.main()
