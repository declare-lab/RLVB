"""The game datasets against their sources: the registry's suites; TextArena's feedback and rules text (benchmarks/textarena);
VAGEN's maps and rooms (benchmarks/vagen), with the test maps and rooms kept out of training and validation; the exact beliefs
against brute force; a won Sokoban game has the box on our target. A test is skipped when its simulator is not installed
(pip install -e ".[games]")."""
import ast
import contextlib
import importlib.util
import io
import json
import random
import unittest

import numpy as np

from plearn import BENCHMARKS


def needs(*modules):
    return unittest.skipUnless(all(importlib.util.find_spec(m) for m in modules), f"needs {', '.join(modules)}")


def planner():
    from plearn.method import estimator as E
    from plearn.method.trainer import Planner
    from plearn.train import METHOD
    return lambda D: Planner(D, E.Reliability(fidelity=True, value=METHOD["value"]), METHOD["tau"], 2)


# Every training teacher: the 8 HiddenGames tasks (train_suite) and minesweeper_medium (registered, with a teacher, in no suite).
TEACHERS = ["train_suite", "minesweeper_medium"]


class SuiteTests(unittest.TestCase):
    def test_the_suites_hold_the_game_tasks(self):
        """train_suite = 8 training tasks (856 games on their test split): the 7 HiddenGames test tasks and 1 training-only task;
        test_suite = test_suite/proworlds (Pro-Worlds, normal board, 300) + test_suite/hidden_games (the 7 tasks, 756) +
        test_suite/roboquest (RoboQuest text, 500) = 1,556 games in 32 tasks; test_suite_long = the same with the long board in
        place of the normal board; the training-only task is in no test suite; every test task is capped at 8,192 thinking
        tokens; the tasks outside the suites stay registered."""
        from plearn.api.registry import registry
        from plearn.api.task import Task
        reg = registry(); train = reg.tasks(["train_suite"]); hidden = reg.tasks(["test_suite/hidden_games"])
        eight = ["frozenlake_hidden_goal", "mastermind", "minesweeper_easy", "minigrid_keycorridor", "sokoban_hidden_target",
                 "tabletop_blocks", "tabletop_cups", "tabletop_lamps"]
        self.assertEqual(sorted(train), eight)
        self.assertEqual(hidden, [n for n in train if n != "mastermind"])            # the training tasks without the training-only one
        games = lambda g: sum(Task.load(n, reg).num_episodes for n in reg.tasks([g]))
        for g, n in (("proworlds_expansion", 300), ("proworlds_expansion_long", 300), ("roboquest_text", 500), ("train_suite", 856),
                     ("test_suite/proworlds", 300), ("test_suite/proworlds_long", 300), ("test_suite/hidden_games", 756),
                     ("test_suite/roboquest", 500), ("test_suite", 1556), ("test_suite_long", 1556)):
            self.assertEqual(games(g), n, g)
        self.assertEqual(reg.tasks(["test_suite"]), reg.tasks(["proworlds_expansion", "test_suite/hidden_games", "roboquest_text"]))
        self.assertEqual(reg.tasks(["test_suite_long"]), reg.tasks(["proworlds_expansion_long", "test_suite/hidden_games", "roboquest_text"]))
        for g, board in (("test_suite", "normal"), ("test_suite_long", "long")):
            test = reg.tasks([g]); self.assertEqual(len(test), 32, g)
            self.assertLessEqual(set(hidden), set(test)); self.assertNotIn("mastermind", test)
            self.assertEqual({Task.load(n, reg).env_args["board"] for n in test if Task.load(n, reg).dataset == "proworlds"}, {board})
            for out in ("minesweeper_medium", "minesweeper_hard"):
                self.assertNotIn(out, test); Task.load(out, reg)                         # out of the suites, still registered
            self.assertTrue(all(Task.load(n, reg).generation_kwargs["max_think"] == 8192 for n in test), g)
            self.assertTrue(all(Task.load(n, reg).test_split == "test" for n in test), g)
        self.assertEqual({Task.load(n, reg).env_args["version"] for n in reg.tasks(["roboquest_text"])}, {"text"})

    def test_no_benchmark_is_trained_on(self):
        from plearn.api.registry import registry
        from plearn.api.task import Task
        reg = registry(); test = set(reg.tasks(["test_suite"])) | set(reg.tasks(["test_suite_long"]))
        self.assertEqual([n for n in reg.tasks(TEACHERS) if Task.load(n, reg).dataset in ("proworlds", "roboquest")], [])
        self.assertTrue(all(Task.load(n, reg).teacher is None for n in test if Task.load(n, reg).dataset in ("proworlds", "roboquest")))

    def test_a_game_that_ends_without_an_answer_is_out_of_premature(self):
        from plearn.method.trainer import premature_rate
        early, late, goal = (type("Ep", (), {})() for _ in range(3))
        early.commit_ident, late.commit_ident = False, True              # goal: FrozenLake's goal reached, no SUBMIT
        self.assertEqual(premature_rate([early, late, goal]), 0.5); self.assertIsNone(premature_rate([goal]))

    @needs("gymnasium", "gym_sokoban", "minigrid", "jax", "mani_skill")
    def test_the_trainers_episode_fields_leave_the_game_alone(self):
        """The trainer writes e.key (the scene of an E-step group) and e.note (the last belief line) on every training
        episode; a teacher must keep its own state elsewhere, so a look still works after them."""
        from plearn.tasks.mixture import teachers_of
        for cls, args in teachers_of(TEACHERS):
            with self.subTest(teacher=cls.__module__):
                D = cls(**{**args, "split": "validation"}); e = D.start(D.eval_specs(1))[0]
                e.key, e.note = (1, 0), "Possible: all"
                D.do_look(e, next(iter(D.look_strings(e)))); D.close([e])

    @needs("gymnasium", "gym_sokoban", "minigrid", "jax", "mani_skill")
    def test_the_open_line_share(self):
        """open_line (plearn/tasks/domain.py): on every training task (TEACHERS: every training teacher), along the teacher's own actions, the
        exact line is never open and the line of the episode's start is always open on the turns with evidence; turns before
        any evidence do not count. W&B records it per task and for the group."""
        from plearn.api.registry import registry
        from plearn.api.task import Task
        from plearn.method.trainer import WANDB_KEEP
        from plearn.tasks.domain import DomainEpisode
        reg = registry(); P = planner(); with_evidence = 0
        for name in reg.tasks(TEACHERS):
            t = Task.load(name, reg); args = dict(t.env_args)
            if "tabletop" in args["domain"]: args["domain_args"] = {**(args.get("domain_args") or {}), "render": False}
            self.assertIn("open_line", [m["metric"] for m in t.metric_list])
            with self.subTest(task=name):
                spec = DomainEpisode.specs("validation", 1, 0, **args)[0]; rows = {}
                for kind in ("exact", "start"):
                    ep = DomainEpisode(spec, **args); Pk = P(ep.D)
                    for _ in range(6):
                        if ep.done: break
                        plan = Pk(ep.e); act = max(plan["logp"], key=plan["logp"].get)
                        ep.step(ep.D.belief_note(ep.e) if kind == "exact" else ep.start_line, act, {})
                    rows[kind] = ep.row(); ep.close()
                self.assertIn(rows["exact"]["open_line"], (0.0, None))
                self.assertIn(rows["start"]["open_line"], (1.0, None))
                self.assertEqual(rows["exact"]["open_line"] is None, rows["start"]["open_line"] is None)   # the same evidence turns
                with_evidence += rows["start"]["open_line"] is not None
        self.assertGreaterEqual(with_evidence, 6)
        for k in ("eval/tasks.mastermind.open_line", "eval/tasks.tabletop_cups.open_line", "eval/groups.train_suite.open_line"):
            self.assertTrue(WANDB_KEEP.match(k), k)

    @needs("gymnasium", "gym_sokoban", "minigrid")
    def test_no_episode_is_in_two_splits(self):
        """Every training teacher: no (task family, seed) of the training draws is a validation or a test episode, and no
        validation episode is a test episode."""
        from plearn.tasks.mixture import teachers_of
        rng = random.Random(0)
        for cls, args in teachers_of(TEACHERS):
            with self.subTest(teacher=cls.__module__):
                D = cls(**{**args, "split": "train"}); train = {D.sample(rng)[:2] for _ in range(5000)}
                val, test = ({s[:2] for s in cls(**{**args, "split": split}).eval_specs(500)} for split in ("validation", "test"))
                self.assertFalse(train & val); self.assertFalse(train & test); self.assertFalse(val & test)


class MastermindTests(unittest.TestCase):
    SRC = BENCHMARKS / "textarena" / "textarena" / "envs" / "Mastermind"

    def test_feedback_is_textarenas(self):
        from plearn.tasks.mastermind.teacher import CODES, evaluate
        tree = ast.parse((self.SRC / "env.py").read_text())             # env.py needs TextArena's core: compile the method alone
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_evaluate_guess")
        ns = {"List": list, "Tuple": tuple}; exec(compile(ast.Module(body=[fn], type_ignores=[]), "env.py", "exec"), ns)
        game = type("Game", (), {})(); game.state = type("State", (), {})(); rng = random.Random(0)
        for _ in range(3000):
            g, c = rng.choice(CODES), rng.choice(CODES)
            game.state.game_state = {"secret_code": list(c), "code_length": len(c)}
            self.assertEqual(evaluate(g, c), ns["_evaluate_guess"](game, list(g)))

    def test_rules_and_feedback_text_are_textarenas(self):
        from plearn.tasks.mastermind import teacher as MM
        text = json.loads((self.SRC / "locales" / "en.json").read_text()); p = text["player_prompt"]
        intro = p["intro"].format(code_length=MM.LENGTH, num_numbers=MM.DIGITS, repeats=p["no_duplicates"], max_turns=MM.TURNS)
        ours = (intro.replace("In your response, you can submit your guess in the following format: '[2 1 4 5]'.\n", "")
                .replace("turns to guess the code", "turns to find the code").rstrip("\n"))   # the two changes (benchmarks/textarena)
        self.assertEqual(MM.rules_text(), ours)
        D = MM.Domain(split="validation"); e = D.start(D.eval_specs(1))[0]; D.do_look(e, 0)
        b, w = MM.evaluate(MM.CODES[0], MM.CODES[e.code])
        self.assertIn(text["game_action"]["feedback"].format(guess=MM.code_str(0), black_pegs=b, white_pegs=w), D._history(e).split("\n"))


@needs("popgym_arcade", "jax")
class PlannerTests(unittest.TestCase):
    def test_an_unreadable_action_never_scores_above_the_best(self):
        """At a solved Mastermind state every guess is worth far less than a decision that changes nothing; the score of an
        unreadable line (waste_lp, a log-probability) stays at most 0 and below the teacher's best action."""
        from plearn.method.trainer import action_logp
        from plearn.tasks.mastermind import teacher as MM
        D = MM.Domain(split="validation"); e = D.start(D.eval_specs(1))[0]; P = planner()(D)
        D.do_look(e, e.code); pl = P(e)                                     # the guess is the code: the belief is identified
        self.assertTrue(pl["B"].identified())
        best = max(pl["logp"].values())
        for line in ("1.  **Digits Identification**:", "ACTION: WAIT", ""):
            a = action_logp(D, line, pl)
            self.assertLessEqual(a, 0.0); self.assertLessEqual(a, best)
        self.assertLessEqual(pl["waste_lp"], 0.0)

    @needs("gymnasium", "gym_sokoban", "minigrid", "jax", "mani_skill")
    def test_every_teacher_score_is_a_log_probability(self):
        """Every training teacher, along validation episodes of random looks: each log pi* of the plan (the looks, the answers, an
        unreadable line, a wrong answer) is at most 0."""
        from plearn.tasks.mixture import teachers_of
        rng = random.Random(0)
        for cls, args in teachers_of(TEACHERS):
            with self.subTest(teacher=cls.__module__):
                D = cls(**{**args, "split": "validation"}); P = planner()(D)
                for e in D.start(D.eval_specs(3)):
                    for _ in range(8):
                        pl = P(e)
                        for k, v in list(pl["logp"].items()) + [("waste", pl["waste_lp"]), ("wrong", pl["wrong_lp"])]:
                            self.assertLessEqual(v, 1e-9, (cls.__module__, k, pl["exact"][-300:]))
                        looks = list(D.look_strings(e))
                        if e.done or not looks: break
                        D.do_look(e, rng.choice(looks)); D.after_turn(e)
                        if e.done: break
                    D.close([e])


class MineSweeperTests(unittest.TestCase):
    def test_belief_is_exact(self):
        from plearn.tasks.minesweeper import teacher as MS
        D = MS.Domain(levels=["easy"], split="validation"); rng = random.Random(0); sets, nb = MS.tables(*MS.LEVELS["easy"]); checks = 0
        for spec in D.eval_specs(6):
            e = D.start([spec])[0]
            for _ in range(30):
                looks = D.look_strings(e)
                if not looks: break
                a = rng.choice(list(looks) + [MS.FIRE] * 3); D.do_look(e, a); e.t += 1
                if e.done: break
                if a == MS.FIRE:                                            # brute force: the sets that agree with every revealed cell
                    ok = (sets & np.uint64(e.revealed)) == 0; ng = np.array(e.state.neighbor_grid)
                    for i in range(e.n * e.n):
                        if e.revealed >> i & 1: ok &= np.bitwise_count(sets & nb[i]) == ng[i % e.n, i // e.n]
                    self.assertTrue(np.array_equal(np.sort(sets[ok]), np.sort(e.A))); self.assertTrue((e.A == np.uint64(e.truth)).any())
                    checks += 1
            D.close([e])
        self.assertGreater(checks, 5)


@needs("minigrid")
class MiniGridTests(unittest.TestCase):
    def test_belief_is_exact(self):
        from plearn.tasks.minigrid import teacher as MG
        D = MG.Domain(levels=["S4R3"], split="validation"); rng = random.Random(0); checks = 0
        for spec in D.eval_specs(5):
            e = D.start([spec])[0]; seen = set(MG.visible(e.env, e.pos, e.dir))
            for _ in range(30):
                looks = D.look_strings(e)
                if not looks: break
                D.do_look(e, rng.choice(list(looks))); e.t += 1; seen |= MG.visible(e.env, e.pos, e.dir)
                key_seen = (e.truth[1], e.truth[2]) in seen
                brute = np.array([(tuple(h) == e.truth) if key_seen else ((h[1], h[2]) not in seen) for h in e.H.tolist()])
                self.assertTrue(np.array_equal(brute, e.M)); checks += 1
            D.close([e])
        self.assertGreater(checks, 20)


@needs("gymnasium", "pygame")
class FrozenLakeTests(unittest.TestCase):
    def test_maps_are_vagens_and_the_test_maps_stay_out(self):
        from vagen.envs.frozenlake.utils.utils import generate_random_map
        from plearn.tasks.frozenlake import teacher as FL
        D = FL.Domain(split="test"); eps = D.start(D.eval_specs(16))
        for e in eps: self.assertEqual(list(e.desc), generate_random_map(4, 0.8, int(e.spec[1])))     # VAGEN's map of the seed
        D.close(eps)
        rng = random.Random(0); train, val = FL.Domain(split="train"), FL.Domain(split="validation")
        specs = [("frozenlake-hidden-goal", s, "train") for s in list(FL.TEST_SEEDS)[:8]] + [train.sample(rng) for _ in range(200)]
        for D, ss in ((train, specs), (val, val.eval_specs(128))):           # the first 8: seeds whose own map is a test map
            eps = D.start(ss); self.assertFalse(any(tuple(e.desc) in FL.test_maps() for e in eps)); D.close(eps)

    def test_belief_is_exact(self):
        from plearn.tasks.frozenlake import teacher as FL
        D = FL.Domain(split="validation"); rng = random.Random(0); checks = 0
        for spec in D.eval_specs(10):
            e = D.start([spec])[0]
            for _ in range(10):
                if e.done: break
                safe = [a for a in FL.MOVE if not e.known_hole(e.move(e.pos, a))]; D.do_look(e, rng.choice(safe)); e.t += 1
                G, Hm, _ = FL.hypotheses(e.start); flat = "".join(e.desc); ok = np.ones(len(G), dtype=bool)
                for j in range(16):
                    if e.seen >> j & 1:
                        ok &= (G == j) if flat[j] == "G" else ((Hm >> j) & 1 == 1) if flat[j] == "H" else ((G != j) & ((Hm >> j) & 1 == 0))
                self.assertTrue(np.array_equal(ok, e.M)); self.assertTrue(((G == e.goal) & (Hm == e.holes) & e.M).any()); checks += 1
            D.close([e])
        self.assertGreater(checks, 30)


@needs("gym_sokoban")
class SokobanTests(unittest.TestCase):
    def test_rooms_are_vagens(self):
        from vagen.envs.sokoban.patch_sokoban_env import PatchedSokobanEnv
        from plearn.tasks.sokoban import teacher as SK
        with contextlib.redirect_stdout(io.StringIO()): V = PatchedSokobanEnv(dim_room=(6, 6), max_steps=100, num_boxes=1)
        for split, seeds in (("test", list(SK.TEST_SEEDS)[:12]), ("validation", range(10000, 10006)), ("train", range(10 ** 6, 10 ** 6 + 6))):
            part = SK.PARTITION.get(split)
            for s in seeds:
                with contextlib.redirect_stdout(io.StringIO()): V.reset(seed=s, min_solution_steps=SK.BAND, map_partition=part)
                fixed, state = SK.generate(s, SK.BAND, part)
                self.assertTrue(np.array_equal(fixed, V.room_fixed) and np.array_equal(state, V.room_state), (split, s))

    def test_the_test_rooms_stay_out(self):
        from plearn.tasks.sokoban import teacher as SK
        T = SK.test_rooms(SK.BAND); rng = random.Random(0)
        test = SK.Domain(split="test").start(SK.Domain(split="test").eval_specs(128))
        self.assertTrue(all(e.room in T for e in test))
        specs = [("sokoban-hidden-target", s, "train") for s in list(SK.TEST_SEEDS)[:8]] + [SK.Domain().sample(rng) for _ in range(100)]
        for split, ss in (("train", specs), ("validation", SK.Domain(split="validation").eval_specs(128))):
            eps = SK.Domain(split=split).start(ss); self.assertFalse(any(e.room in T for e in eps), split)

    def test_belief_is_exact(self):
        from plearn.tasks.sokoban import teacher as SK
        D = SK.Domain(split="validation"); rng = random.Random(0); checks = 0
        for spec in D.eval_specs(10):
            e = D.start([spec])[0]; seen = set(SK.window(e.state[0], e.view))
            for _ in range(12):
                if e.done: break
                D.do_look(e, rng.choice(list(SK.ACTIONS))); e.t += 1; seen |= SK.window(e.state[0], e.view)
                brute = {e.target} if e.target in seen else {c for c in e.candidates if c not in seen}
                self.assertEqual(brute, D.belief(e).alive); checks += 1
        self.assertGreater(checks, 50)

    def test_the_prompt_shows_every_room_since_the_start(self):
        """The room at the start and after each action, in order (a move into a wall too), so the checked cells are in the prompt."""
        from plearn.tasks.sokoban import teacher as SK
        D = SK.Domain(split="validation"); e = D.start([D.eval_specs(1)[0]])[0]; rng = random.Random(1)
        for k in range(5):
            if e.done: break
            D.do_look(e, rng.choice(list(SK.ACTIONS))); e.t += 1
        parts, images = D.user_content(e, note="Possible: target at B3")
        text = "".join(p.get("text", "") for p in parts)
        self.assertEqual(len(images), len(e.log) + 1); self.assertEqual(sum(p["type"] == "image" for p in parts), len(images))
        self.assertIn("The room at the start: ", text)
        for k, a in enumerate(e.log): self.assertIn(f"After action {k + 1} ({a}): ", text)
        self.assertTrue(np.array_equal(np.asarray(images[-1]), np.asarray(e.picture())))   # the last picture is the room now
        self.assertIn("Possible: target at B3", text)

    def test_a_win_is_the_box_on_our_target(self):
        from plearn.tasks.sokoban import teacher as SK
        D = SK.Domain(split="validation"); P = planner()(D); won = 0
        for spec in D.eval_specs(8):
            e = D.start([spec])[0]; shortest = e.dist(e.target)[e.state]
            self.assertEqual(list(zip(*np.nonzero(e.env.room_fixed == 2))), [e.target])     # one target in the game: ours
            while not e.done:
                pl = P(e); s = max(pl["looks"].values(), key=lambda x: pl["values"].get(x, -1e9))
                D.do_look(e, {v: k for k, v in D.look_strings(e).items()}[s]); e.t += 1; D.after_turn(e)
            if e.won:
                won += 1; self.assertEqual(e.state[1], e.target); self.assertGreaterEqual(e.t, shortest)
        self.assertGreaterEqual(won, 6)


if __name__ == "__main__":
    unittest.main()
