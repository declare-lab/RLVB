"""The picture server's context guard (plearn/models/mm_server.py; without it, one prompt of 81,962 tokens > 81,920 fails the
whole /generate). too_long() knows vLLM's two texts; one() turns such an error into a cut, empty completion (finish "length") for each
of the n samples, and the prompt-length request into the context length (the uncapped request then gets max_tokens 1 and is cut too).
No vLLM is needed: the module is read as text for one(), and too_long() is called."""
import ast
import unittest
from pathlib import Path

SRC = (Path(__file__).resolve().parents[2] / "plearn" / "models" / "mm_server.py").read_text()


class GuardTests(unittest.TestCase):
    def test_too_long(self):
        ns = {}; tree = ast.parse(SRC)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "too_long")
        exec(compile(ast.Module([fn], []), "mm_server", "exec"), ns)
        self.assertTrue(ns["too_long"](ValueError("The decoder prompt (length 81962) is longer than the maximum model length of 81920")))
        self.assertTrue(ns["too_long"](ValueError("This model's maximum context length is 81920 tokens")))
        self.assertFalse(ns["too_long"](RuntimeError("CUDA error: out of memory")))

    def test_one_catches_only_that_error(self):
        one = SRC[SRC.index("    async def one("):SRC.index("    @app.post(\"/generate\")")]
        self.assertIn("if not too_long(ex): raise", one)
        self.assertIn('return args.max_model_len if plen else [{"text": "", "finish": "length"} for _ in range(sp.n)]', one)


if __name__ == "__main__":
    unittest.main()
