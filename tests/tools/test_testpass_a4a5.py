"""tools/testpass_a4a5.py (CPU, no model): the A4 / A5 / A6 measures from saved test rows, against the trainer's own functions.

The rows come from real episodes: a scripted policy (no model) plays HiddenGames tasks through the evaluator's own episode class
(plearn/tasks/domain.py DomainEpisode) and the real domain code, and the rows are written in the evaluator's form ({"spec": ...,
**ep.row()}). While it plays, the test computes the trainer's values at every turn from the live episode (plearn/method/trainer.py:
Planner, belief_best, belief_score, decision_regret, answered, premature_rate). The tool then sees only the rows and must give the
same values through its replay.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m unittest tests.tools.test_testpass_a4a5"""
import csv, json, os, random, statistics, sys, tempfile, time, unittest
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tools"))
import testpass_a4a5 as TP   # noqa: E402
from plearn.api.registry import registry   # noqa: E402
from plearn.api.task import Task   # noqa: E402
from plearn.method import trainer as TR   # noqa: E402
from plearn.tasks.domain import DomainEpisode   # noqa: E402

REG = registry()
GAMES = ("minesweeper_easy", "minigrid_keycorridor", "frozenlake_hidden_goal", "tabletop_cups")   # HiddenGames test tasks: a board; pictures; no answer line; 3D pictures (render off)


def domain_args(task):
    """The task's domain and domain_args; a domain that renders pictures gets render=False (the play is on the CPU)."""
    t = Task.load(task, REG); dom = t.env_args["domain"]; dargs = dict(t.env_args.get("domain_args") or {})
    import importlib, inspect
    if "render" in inspect.signature(importlib.import_module(dom).Domain.__init__).parameters: dargs["render"] = False
    return t, dom, dargs


def play(task, n, seed=0, one_line=False, stall=False):
    """n test episodes of the task, played by a scripted policy. Returns (rows in the evaluator's form, the trainer's values)."""
    t, dom, dargs = domain_args(task); rng = random.Random(seed)
    specs = t.specs(None, n)
    rows, eps, turns, late = [], [], [], 0
    for spec in specs:
        ep = DomainEpisode(spec, domain=dom, domain_args=dargs); D, e = ep.D, ep.e
        P = TP._planner(D); stale = ep.start_line; commit_ident = None
        while not ep.done:
            with ep.lock:
                B = D.belief(e); ident = bool(B.identified()); pl = P(e); exact = pl["anchor"]
                b = "" if one_line else rng.choice([exact, exact, stale, ep.start_line, "Possible: no idea"])
                looks = list(pl["looks"].values()); answers = pl["answers"]
                r = rng.random()
                if stall: a = "ACTION: WAIT" if r < 0.5 else rng.choice(looks or ["ACTION: WAIT"])     # never answers: the turn limit
                elif ident and answers and r < 0.6: a = D.commit_string(pl["best"])                 # answer once identified
                elif looks and (r < 0.8 or not answers): a = rng.choice(looks)                         # a look (late once identified)
                elif answers and r < 0.92: a = D.commit_string(rng.choice(answers))                    # an early answer, or a guess
                else: a = "ACTION: WAIT"                                                                # a line the game cannot do
                if not one_line:
                    turns.append({"anchor_exact": bool(D.anchor_equal(b, exact)),
                                  "belief_ok": TR.belief_best(D, B, pl) - TR.belief_score(D, B, b, pl) <= 1e-6,
                                  "decision_ok": TR.decision_regret(D, P, pl, b) <= 1e-6})
            text = "x" * rng.randrange(5, 60)
            ep.step(b, a, {"thought_chars": len(text), "thought_cut": rng.random() < 0.2, "thought": text})
            k = ep.turns[-1]["kind"]
            if k == "look": late += ident
            if k == "answer": commit_ident = ident
            stale = exact
        rows.append({"spec": spec, **ep.row()}); ep.close()
        budget = commit_ident is None and not e.correct and not getattr(e, "fell", False) and not getattr(e, "dead", False)   # FrozenLake hole, Minesweeper mine: the game ends, not the limit
        eps.append(SimpleNamespace(correct=rows[-1]["correct"], commit_ident=commit_ident, looks=rows[-1]["looks"], budget=budget))
    want = {"success": statistics.fmean(bool(x.correct) for x in eps), "premature": TR.premature_rate(eps),
            "n_answered": len(TR.answered(eps)), "late_looks": late / len(eps), "mean_t": statistics.fmean(x.looks for x in eps),
            "budget": statistics.fmean(float(x.budget) for x in eps)}
    if not one_line:
        for m in ("anchor_exact", "belief_ok", "decision_ok"): want[m] = statistics.fmean(float(x[m]) for x in turns)
        op = [x["open"] for r in rows for x in r["turn_log"] if "open" in x]
        want["open_line"] = statistics.fmean(op) if op else None
    th = [x for r in rows for x in r["turn_log"]]
    want["thought_chars_mean"] = statistics.fmean(x["thought_chars"] for x in th)
    want["thought_cut"] = statistics.fmean(float(x["thought_cut"]) for x in th)
    return rows, want


def write_folder(root, name, tasks_rows, config=None):
    d = os.path.join(root, name); os.makedirs(d)
    for task, rows in tasks_rows.items():
        with open(os.path.join(d, f"{task}.jsonl"), "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump({"results": {}, "groups": {}, "config": {"model": "scripted", "path": None, "adapter": None, "tasks": list(tasks_rows),
                                                          "split": None, **(config or {})}}, f)
    return d


def read_csv(fn):
    out = {}
    with open(fn) as fh: rs = list(csv.DictReader(fh))
    for r in rs: out[(r["benchmark"], r["task"], r["row"], r["seed"], r["measure"])] = (r["value"], r["n"])
    return out


def num(v): return float(v[0])


class TestPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = cls.tmp.name
        cls.played = {g: play(g, 6, seed=i) for i, g in enumerate(GAMES)}
        cls.played_s1 = {g: play(g, 6, seed=10 + i) for i, g in enumerate(GAMES)}
        for s, p in (("0", cls.played), ("1", cls.played_s1)):
            for g, (_, w) in p.items(): print(f"\nseed {s} {g}: " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in w.items()))

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_values_match_the_trainer(self):
        f0 = write_folder(self.root, "s0", {g: r for g, (r, _) in self.played.items()})
        f1 = write_folder(self.root, "s1", {g: r for g, (r, _) in self.played_s1.items()})
        out = os.path.join(self.root, "out1"); t0 = time.process_time()
        TP.main(["--folders", f0, f1, "--labels", "PL", "PL", "--seeds", "0", "1", "--out", out, "--procs", "1"])
        print(f"\nCPU {time.process_time() - t0:.1f} s for {2 * 6 * len(GAMES)} episodes (the tool alone; the play is not counted)")
        v = read_csv(os.path.join(out, "scripted.csv"))
        for seed, played in (("0", self.played), ("1", self.played_s1)):
            for g, (rows, want) in played.items():
                key = lambda m: ("HiddenGames", g, "PL", seed, m)
                self.assertEqual(v[key("replay_mismatch")][0], "0", g)
                self.assertEqual(v[key("replay_n")][0], "6", g)
                for m, x in want.items():
                    if x is None: self.assertIn(v[key(m)][0], (TP.NM, TP.ND), (g, m)); continue
                    self.assertAlmostEqual(num(v[key(m)]), float(x), places=5, msg=(g, seed, m))
                self.assertEqual(v[key("thought_tokens_mean")][0], TP.NM)          # the text exists, but no tokenizer path
                if g == "frozenlake_hidden_goal": self.assertEqual(v[key("premature")][0], TP.ND)   # no answer line: 0 / 0
        # the overall line pools the episodes of the benchmark; the mean line is the mean of the two seeds
        rows0 = [r for rs, _ in self.played.values() for r in rs]
        self.assertAlmostEqual(num(v[("HiddenGames", "overall", "PL", "0", "success")]), statistics.fmean(r["success"] for r in rows0), places=5)
        self.assertEqual(v[("HiddenGames", "overall", "PL", "0", "success")][1], str(len(rows0)))
        s = [num(v[("HiddenGames", "minesweeper_easy", "PL", k, "late_looks")]) for k in ("0", "1")]
        self.assertAlmostEqual(num(v[("HiddenGames", "minesweeper_easy", "PL", "mean", "late_looks")]), sum(s) / 2, places=5)
        readme = open(os.path.join(out, "README.md")).read()
        self.assertIn("commit", readme); self.assertIn(f0, readme)

    def test_a_changed_row_is_a_mismatch_and_left_out(self):
        rows, want = self.played["minesweeper_easy"]
        bad = json.loads(json.dumps(rows))
        bad[0]["success"] = 1.0 - bad[0]["success"]; bad[0]["correct"] = not bad[0]["correct"]   # a stored end the replay cannot reach
        log = bad[1]["turn_log"]; log[0]["kind"] = "invalid" if log[0]["kind"] != "invalid" else "look"
        f = write_folder(self.root, "bad", {"minesweeper_easy": bad})
        out = os.path.join(self.root, "out2")
        TP.main(["--folders", f, "--labels", "X", "--out", out, "--procs", "1"])
        v = read_csv(os.path.join(out, "scripted.csv")); key = lambda m: ("HiddenGames", "minesweeper_easy", "X", "0", m)
        self.assertEqual(v[key("replay_mismatch")][0], "2")
        self.assertEqual(v[key("late_looks")][1], "4")                     # the 4 faithful episodes only
        self.assertEqual(v[key("success")][1], "6")                        # the stored measures keep every episode
        self.assertIn("| minesweeper_easy | 6 | 2 |", open(os.path.join(out, "README.md")).read())

    def test_the_training_only_task_is_not_a_test_task(self):
        """HiddenGames = test_suite/hidden_games (7 tasks); the one training task out of it is in no benchmark (no row)."""
        only = set(REG.tasks("train_suite")) - set(REG.tasks("test_suite/hidden_games"))
        self.assertEqual(len(only), 1)
        for n in only: self.assertIsNone(TP.task_info(n, {}, REG)[0], n)
        for n in REG.tasks("test_suite/hidden_games"): self.assertEqual(TP.task_info(n, {}, REG)[0], "HiddenGames", n)

    def test_no_overall_mean_when_the_seeds_have_other_tasks(self):
        g0, g1 = GAMES[0], GAMES[1]
        f0 = write_folder(self.root, "u0", {g0: self.played[g0][0], g1: self.played[g1][0]})
        f1 = write_folder(self.root, "u1", {g0: self.played_s1[g0][0]})                 # seed 1: one task only
        out = os.path.join(self.root, "out7")
        TP.main(["--folders", f0, f1, "--labels", "U", "U", "--seeds", "0", "1", "--out", out, "--procs", "1"])
        v = read_csv(os.path.join(out, "scripted.csv"))
        self.assertEqual(v[("HiddenGames", "overall", "U", "mean", "success")], (TP.NM, "0"))
        s = [num(v[("HiddenGames", g0, "U", k, "success")]) for k in ("0", "1")]
        self.assertAlmostEqual(num(v[("HiddenGames", g0, "U", "mean", "success")]), sum(s) / 2, places=5)   # the task mean stays

    def test_the_turn_limit_is_budget(self):
        rows, want = play("tabletop_lamps", 3, seed=4, stall=True)
        self.assertEqual(want["budget"], 1.0)
        f = write_folder(self.root, "stall", {"tabletop_lamps": rows})
        out = os.path.join(self.root, "out6")
        TP.main(["--folders", f, "--labels", "S", "--out", out, "--procs", "1"])
        v = read_csv(os.path.join(out, "scripted.csv")); key = lambda m: ("HiddenGames", "tabletop_lamps", "S", "0", m)
        self.assertEqual(v[key("budget")], ("1", "3")); self.assertEqual(v[key("premature")][0], TP.ND); self.assertEqual(v[key("n_answered")][0], "0")
        for m in ("late_looks", "mean_t", "anchor_exact", "belief_ok", "decision_ok"):
            self.assertAlmostEqual(num(v[key(m)]), want[m], places=5, msg=m)

    def test_one_line_rows_have_no_belief_measures(self):
        rows, want = play("minigrid_keycorridor", 4, seed=3, one_line=True)
        f = write_folder(self.root, "one", {"minigrid_keycorridor": rows}, {"task_args": {"minigrid": {"one_line": 1}}})
        out = os.path.join(self.root, "out3")
        TP.main(["--folders", f, "--labels", "Untrained", "--out", out, "--procs", "1"])
        v = read_csv(os.path.join(out, "scripted.csv")); key = lambda m: ("HiddenGames", "minigrid_keycorridor", "Untrained", "0", m)
        for m in TP.A5: self.assertEqual(v[key(m)][0], TP.ND, m)
        for m in ("success", "premature", "late_looks", "mean_t"):
            if want[m] is None: self.assertEqual(v[key(m)][0], TP.NM, m)
            else: self.assertAlmostEqual(num(v[key(m)]), float(want[m]), places=5, msg=m)

    def test_roboquest_rows_have_no_posterior_measures(self):
        row = lambda ok, out, t: {"spec": ["test", "x"], "correct": ok, "success": float(ok), "submitted": float(out not in ("no_answer", "invalid_answer")),
                                  "outcome": out, "evidence_actions": t, "T": 6, "decisions": t + 1, "thought_cut": 0.0,
                                  "turn_log": [{"belief": "Possible: any", "action": "ACTION: LOOK", "thought_chars": 10 * (k + 1), "thought_cut": k == 0}
                                               for k in range(t + 1)]}
        rows = [row(True, "success", 2), row(False, "no_answer", 6), row(False, "wrong_answer", 1), {"spec": ["test", "y"], "correct": False, "error": "x"}]
        f = write_folder(self.root, "rq", {"roboquest_text_odd_parcel": rows})
        out = os.path.join(self.root, "out4")
        TP.main(["--folders", f, "--labels", "PL", "--out", out, "--procs", "1"])
        v = read_csv(os.path.join(out, "scripted.csv")); key = lambda m: ("RoboQuest", "roboquest_text_odd_parcel", "PL", "0", m)
        self.assertAlmostEqual(num(v[key("success")]), 1 / 3); self.assertAlmostEqual(num(v[key("budget")]), 1 / 3)
        self.assertAlmostEqual(num(v[key("mean_t")]), 3.0); self.assertEqual(v[key("errors")][0], "1")
        self.assertEqual(v[key("n_answered")][0], "2")
        for m in ("premature", "late_looks") + TP.A5: self.assertEqual(v[key(m)][0], TP.ND, m)
        self.assertAlmostEqual(num(v[key("thought_chars_median")]), 25.0)   # turns of 10, 20, 30; 10, ..., 70; 10, 20 chars
        self.assertAlmostEqual(num(v[key("thought_cut")]), 3 / 12)

    def test_thought_tokens_with_a_local_tokenizer(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        tk = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1, "b": 2}, unk_token="[UNK]")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
        tdir = os.path.join(self.root, "tok"); os.makedirs(tdir); tk.save(os.path.join(tdir, "tokenizer.json"))
        with open(os.path.join(tdir, "tokenizer_config.json"), "w") as fh: json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"}, fh)
        texts = ["a", "a b", "a b a b", "b b b b b b"]                     # 1, 2, 4, 6 tokens; 1, 3, 7, 11 chars
        rows = [{"spec": ["test", str(i)], "correct": False, "success": 0.0, "submitted": 1.0, "outcome": "wrong_answer", "evidence_actions": 0,
                 "turn_log": [{"belief": "", "action": "ACTION: SUBMIT x", "thought_chars": len(x), "thought_cut": False, "thought": x}]}
                for i, x in enumerate(texts)]
        f = write_folder(self.root, "tok_rows", {"roboquest_text_odd_parcel": rows}, {"path": tdir})
        out = os.path.join(self.root, "out5")
        TP.main(["--folders", f, "--labels", "PL", "--out", out, "--procs", "1"])
        v = read_csv(os.path.join(out, "scripted.csv")); key = lambda m: ("RoboQuest", "roboquest_text_odd_parcel", "PL", "0", m)
        self.assertAlmostEqual(num(v[key("thought_tokens_mean")]), 13 / 4); self.assertAlmostEqual(num(v[key("thought_tokens_median")]), 3.0)
        self.assertAlmostEqual(num(v[key("thought_chars_mean")]), 22 / 4); self.assertEqual(v[key("thought_tokens_mean")][1], "4")
        for m in TP.A5: self.assertEqual(v[key(m)][0], TP.ND, m)


@unittest.skipUnless(os.environ.get("NODE_BIN") and os.path.isfile(os.environ.get("NODE_BIN", "")), "the expansion engine needs NODE_BIN")
class TestExpansion(unittest.TestCase):
    def test_expansion_replay_is_faithful(self):
        from plearn.tasks.proworlds import expansion_env as X, expansion_teacher as XT
        rng = random.Random(5); rows, late, prem = [], 0, []
        for seed in X.seeds("bell", 4):
            ep = X.Episode(["test", seed], game="bell")
            Dt = XT.Domain(split="test", tasks="bell"); te = Dt.start([("expansion/bell", seed, "test")])[0]   # the teacher beside it
            while not ep.done:
                B = Dt.belief(te); ident = B.identified()
                r = rng.random()
                if ident and r < 0.5: a = f"ACTION: SUBMIT {Dt.anchor_answer(Dt.belief_note(te), Dt.answer_options(te), te)}"
                elif ident and r < 0.97: a = f"ACTION: MOVE {rng.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])}"     # a late look
                elif r < 0.95:                                                 # the teacher's best move (or a random one)
                    pl = TP._planner(Dt)(te); mv = [s for s in pl["looks"].values()]
                    a = max(mv, key=lambda s: pl["tgt"].get(s, 0.0)) if mv and r < 0.8 else f"ACTION: MOVE {rng.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])}"
                elif r < 0.97: a = f"ACTION: SUBMIT {rng.randrange(1, len(te.members) + 1)}"
                else: a = "ACTION: JUMP"
                b = rng.choice([Dt.belief_note(te), "Possible: all"])
                ep.step(b, a, {"thought_chars": 3, "thought_cut": False})
                k = ep.turns[-1]["kind"]
                if k == "move": late += ident; Dt.do_look(te, ["UP", "RIGHT", "DOWN", "LEFT"].index(a.split()[-1]))
                if k == "answer": prem.append(float(not ident))
                if k == "budget": Dt.do_look(te, 0)
            rows.append({"spec": ["test", seed], **ep.row()}); ep.close(); Dt.close([te])
        with tempfile.TemporaryDirectory() as root:
            f = write_folder(root, "x", {"proworlds_expansion_bell": rows}); out = os.path.join(root, "out")
            TP.main(["--folders", f, "--labels", "PL", "--out", out, "--procs", "1"])
            v = read_csv(os.path.join(out, "scripted.csv"))
        key = lambda m: ("ProWorlds-expansion", "proworlds_expansion_bell", "PL", "0", m)
        print(f"\nexpansion: {len(rows)} episodes, late looks {late}, answers {len(prem)}, premature {sum(prem)}")
        self.assertGreater(late, 0)                                        # the check covers late looks
        self.assertEqual(v[key("replay_mismatch")][0], "0")
        self.assertAlmostEqual(num(v[key("late_looks")]), late / len(rows), places=5)
        if prem: self.assertAlmostEqual(num(v[key("premature")]), statistics.fmean(prem), places=5)


if __name__ == "__main__":
    unittest.main()
