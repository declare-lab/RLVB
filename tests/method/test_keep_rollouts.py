"""Every RLVB run keeps its training rollouts.
- The RLVB trainer (RLVB (EM) and RLVB (GRPO)): the learner MOVES the queue's batch_<u>.pkl to <out>/case-<tag>/batches after update u (before:
  it deleted it); rank 0 only; PL_KEEP_ROLLOUTS=0 deletes as before. plearn/train.py sets --dump_em <out>/case-<tag>/em, the
  folder's sibling is the keep folder, so the runs' argv does not change (tests/method/test_run_files.py)."""
import io, json, os, tempfile, unittest
from pathlib import Path
from unittest import mock

from plearn import train
from plearn.method import trainer as T


class PLTrainerKeepTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(); self.q = os.path.join(self.d, "queue-x"); os.makedirs(self.q)
        self.em = os.path.join(self.d, "case-x", "em")

    def batch(self, u, n=1000):
        p = os.path.join(self.q, f"batch_{u}.pkl"); open(p, "wb").write(b"x" * n); return p

    def test_the_batch_moves_to_the_keep_folder(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PL_KEEP_ROLLOUTS", None)
            log = io.StringIO(); src = self.batch(3)
            dst = T.keep_batch(src, self.em, 3, log)
            self.assertEqual(dst, os.path.join(self.d, "case-x", "batches", "batch_3.pkl"))
            self.assertFalse(os.path.exists(src)); self.assertEqual(open(dst, "rb").read(), b"x" * 1000)
            row = json.loads(log.getvalue().strip()); self.assertEqual((row["kept_batch"], row["kept_dir"]), (3, os.path.dirname(dst)))

    def test_opt_out_deletes_as_before(self):
        with mock.patch.dict(os.environ, {"PL_KEEP_ROLLOUTS": "0"}):
            src = self.batch(4); self.assertEqual(T.keep_batch(src, self.em, 4, io.StringIO()), "")
            self.assertFalse(os.path.exists(src)); self.assertFalse(os.path.exists(os.path.join(self.d, "case-x", "batches")))

    def test_no_dump_em_deletes_as_before(self):
        os.environ.pop("PL_KEEP_ROLLOUTS", None)
        src = self.batch(5); self.assertEqual(T.keep_batch(src, "", 5), ""); self.assertFalse(os.path.exists(src))

    def test_a_missing_batch_does_not_stop_the_run(self):
        os.environ.pop("PL_KEEP_ROLLOUTS", None)
        self.assertEqual(T.keep_batch(os.path.join(self.q, "batch_9.pkl"), self.em, 9), "")   # a KL STOP skip: no file, no error

    def test_train_py_sets_dump_em_next_to_the_keep_folder(self):
        with mock.patch("plearn.models.resolve", lambda m: (str(m), {})):
            for name in ("suite_qwen3vl_8b_thinking", "grpo_posterior_qwen3vl_4b_thinking"):
                with self.subTest(run=name):
                    r = train.commands(train.load(name)); a = r["learner"]
                    em = a[a.index("--dump_em") + 1]
                    self.assertEqual(Path(T.keep_dir(em)), Path(em).parent / "batches")
                    self.assertEqual(Path(em).parent.parent, Path(r["out"]))

    def test_the_learner_keeps_on_rank_0_only(self):
        src = open(T.__file__).read()
        self.assertIn('if lead: keep_batch(q_path(f"batch_{step}.pkl"), args.dump_em, step, log)', src)
        self.assertNotIn('os.remove(q_path(f"batch_{step}.pkl"))', src)


if __name__ == "__main__":
    unittest.main()
