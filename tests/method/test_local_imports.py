"""No function of the package imports again a name its module imports at the top. A local import makes the name local to the
whole function, so the function's nested functions and its lines before the import fail with a NameError where the module's
name was meant (for example train's 'import copy' and collect_held: the actor stops at its start)."""
import ast
import unittest
from pathlib import Path

import plearn


def local_reimports(source):
    tree = ast.parse(source); top = set()
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)): top |= {(a.asname or a.name).split(".")[0] for a in n.names}
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
        stack = list(fn.body)
        while stack:                                                   # the function's own body (a nested function is its own scope)
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)): continue
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                out += [(fn.name, n.lineno, (a.asname or a.name).split(".")[0]) for a in n.names if (a.asname or a.name).split(".")[0] in top]
            stack.extend(ast.iter_child_nodes(n))
    return out


class LocalImportTests(unittest.TestCase):
    def test_the_check_finds_the_case_of_3_nov(self):
        src = "import copy\ndef train():\n    def held(): return copy.copy(1)\n    held()\n    import copy, json\n"
        self.assertEqual(local_reimports(src), [("train", 5, "copy")])

    def test_no_function_imports_a_module_name_again(self):
        for f in sorted(Path(plearn.__file__).parent.rglob("*.py")):
            with self.subTest(file=str(f.relative_to(Path(plearn.__file__).parent))):
                self.assertEqual(local_reimports(f.read_text()), [])


if __name__ == "__main__":
    unittest.main()
