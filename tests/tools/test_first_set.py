"""tools/first_set.py on a real finished phase run (a first sample set is reused only with the same settings, and the settings are
logged in full). PL_FIRST_SET_TEST = a folder with code/ (a code tree) and runs/looks_nobelief_qwen3vl_4b_thinking and
runs/pool_nobelief_qwen3vl_4b_thinking (two finished phase runs of that code); skipped without it. CPU only; each case runs the tool in its
own process (the tool imports the trainer of the --code folder)."""
import os, shutil, subprocess, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = os.path.join(ROOT, "tools", "first_set.py")
P = os.environ.get("PL_FIRST_SET_TEST", "")
LOOKS, POOL = (f"{P}/runs/looks_nobelief_qwen3vl_4b_thinking", f"{P}/runs/pool_nobelief_qwen3vl_4b_thinking") if P else ("", "")


def tool(*a):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "-I", TOOL, *a], capture_output=True, text=True, env=env, cwd=tempfile.gettempdir(), timeout=900)
    return r.returncode, r.stdout + r.stderr


@unittest.skipUnless(os.path.isdir(LOOKS) and os.path.isdir(POOL), "needs PL_FIRST_SET_TEST (code/ and two finished phase runs)")
class FirstSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(prefix="first-set-test-"); cls.reg = os.path.join(cls.d, "REGISTRY.tsv")
        rc, out = tool("save", "--run", LOOKS, "--code", f"{P}/code", "--registry", cls.reg)
        assert rc == 0, out
        cls.set = [l.split()[1] for l in out.splitlines() if l.startswith("SAVED ")][0]

    @classmethod
    def tearDownClass(cls): shutil.rmtree(cls.d, ignore_errors=True)

    def test_save_writes_the_settings_and_the_registry(self):
        txt = open(os.path.join(self.set, "SETTINGS.txt")).read()
        for k in ("phase_set        300", "set_keep         200", "tau_look         0.02", "beta_decay       0.93", "code_tree"):
            self.assertIn(k, txt)
        self.assertEqual(len(open(self.reg).read().splitlines()), 2)       # the header + one set
        self.assertTrue(all(os.path.exists(os.path.join(self.set, f"batch_{u}.pkl")) for u in range(1, 26)))

    def test_the_same_run_matches(self):
        rc, out = tool("find", "--config", f"{LOOKS}/config-looks_nobelief_qwen3vl_4b_thinking.json", "--code", f"{P}/code", "--registry", self.reg)
        self.assertEqual(rc, 0, out); self.assertIn("export PL_SET_FROM=" + self.set, out)

    def test_other_settings_do_not_match_and_are_listed(self):
        rc, out = tool("find", "--config", f"{POOL}/config-pool_nobelief_qwen3vl_4b_thinking.json", "--code", f"{P}/code", "--registry", self.reg)
        self.assertEqual(rc, 3, out); self.assertIn("NO MATCH", out)
        for k in ("args.phase_set: 300 -> 1200", "args.set_keep: 200 -> 800", "args.tau_look: 0.02 -> 0.0"): self.assertIn(k, out)

    def test_a_code_change_with_no_flag_fails(self):
        code = os.path.join(self.d, "code"); shutil.copytree(f"{P}/code/plearn", os.path.join(code, "plearn"), ignore=shutil.ignore_patterns("__pycache__"))
        with open(os.path.join(code, "plearn", "tasks", "frozenlake", "teacher.py"), "a") as f: f.write("\n# a teacher change with no flag\n")
        rc, out = tool("check", "--set", self.set, "--config", f"{LOOKS}/config-looks_nobelief_qwen3vl_4b_thinking.json", "--code", code)
        self.assertEqual(rc, 3, out); self.assertIn("code_tree", out)

    def test_a_changed_batch_fails(self):
        bad = os.path.join(self.d, "bad"); shutil.copytree(self.set, bad)
        with open(os.path.join(bad, "batch_3.pkl"), "ab") as f: f.write(b"x")
        rc, out = tool("check", "--set", bad, "--config", f"{LOOKS}/config-looks_nobelief_qwen3vl_4b_thinking.json", "--code", f"{P}/code")
        self.assertEqual(rc, 3, out); self.assertIn("batch_3.pkl: missing or changed", out)


if __name__ == "__main__":
    unittest.main()
