"""PL_KEEP_LAST: the saves at 25 and 50 stay, and only the last 2 per-update full saves stay besides them."""
import os, tempfile, unittest

from plearn.method.trainer import keep_last_saves

TAG = "mixture-run-seed0"


def kept25(n, steps=50): return n % 25 == 0 or n == steps


class KeepLastTests(unittest.TestCase):
    def make(self, d, steps, kinds=("lora", "opt")):
        for n in steps:
            for k in kinds: open(os.path.join(d, f"{TAG}-{k}-{n}.pt"), "w").close()
        open(os.path.join(d, f"{TAG}-lora.pt"), "w").close()

    def left(self, d):
        return sorted({int(f.rsplit("-", 1)[1][:-3]) for f in os.listdir(d) if f.count("-") >= 3 and f.endswith(".pt") and f.rsplit("-", 1)[1][:-3].isdigit()})

    def test_rolling_window(self):
        with tempfile.TemporaryDirectory() as d:
            for s in range(1, 51):                           # the learner: a full save after every update, then the clean-up
                self.make(d, [s])
                keep_last_saves(d, TAG, s, 2, kept25)
                if s == 43: self.assertEqual(self.left(d), [25, 42, 43])
                if s == 25: self.assertEqual(self.left(d), [24, 25])
                if s == 26: self.assertEqual(self.left(d), [25, 26])
                if s == 27: self.assertEqual(self.left(d), [25, 26, 27])
                if s >= 25: self.assertTrue(os.path.exists(os.path.join(d, f"{TAG}-opt-25.pt")))   # update 25 is never removed (28, ...)
                if s == 28: self.assertEqual(self.left(d), [25, 27, 28])
            self.assertEqual(self.left(d), [25, 49, 50])
            for n in (25, 49, 50):                            # adapter and Adam's state of each kept update
                for k in ("lora", "opt"): self.assertTrue(os.path.exists(os.path.join(d, f"{TAG}-{k}-{n}.pt")))
            self.assertTrue(os.path.exists(os.path.join(d, f"{TAG}-lora.pt")))   # the latest adapter is never touched

    def test_later_files_and_other_tags_stay(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d, [25, 30, 31, 32, 44])                # 44: a file of another attempt of the same run
            open(os.path.join(d, f"other-lora-30.pt"), "w").close()
            open(os.path.join(d, f"{TAG}-rho-30.json"), "w").close()
            self.assertEqual(keep_last_saves(d, TAG, 32, 2, kept25), [30])
            self.assertEqual(self.left(d), [25, 31, 32, 44])
            self.assertTrue(os.path.exists(os.path.join(d, "other-lora-30.pt")))
            self.assertTrue(os.path.exists(os.path.join(d, f"{TAG}-rho-30.json")))

    def test_readout_goes_with_its_update(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d, [1, 2, 3], kinds=("lora", "opt", "readout"))
            self.assertEqual(keep_last_saves(d, TAG, 3, 2, kept25), [1])
            self.assertFalse(any(f.endswith("-1.pt") for f in os.listdir(d)))

    def test_the_learner_loop_uses_it(self):
        import plearn.method.trainer as T                  # the trainer needs a model to run; so check its save lines by text
        with open(T.__file__, encoding="utf-8") as f: src = f.read()
        self.assertIn('KEEP_LAST = int(os.environ.get("PL_KEEP_LAST", "2"))', src)
        self.assertIn("if step % args.eval_every == 0 or step == args.steps or KEEP_LAST: save_lora(step)", src)
        self.assertIn("gone = keep_last_saves(args.out, tag, step, KEEP_LAST, kept_step)", src)
        self.assertIn("def kept_step(n): return n % args.eval_every == 0 or n % SAVE_STEPS == 0 or n == args.steps", src)
        self.assertIn("torch.save(obj, t); os.replace(t, f)", src)

    def test_missing_folder(self):
        self.assertEqual(keep_last_saves("/nonexistent-keep-last", TAG, 5, 2, kept25), [])


if __name__ == "__main__":
    unittest.main()
