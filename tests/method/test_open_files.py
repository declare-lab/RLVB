"""plearn raises the soft limit of open files at import (a soft limit of 1,024 gives "Too many open files" in a run)."""
import os
import resource
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def low_limit():
    """In the child, before exec: the soft limit of a typical ssh session (1,024)."""
    resource.setrlimit(resource.RLIMIT_NOFILE, (1024, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))


def run(code):
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, preexec_fn=low_limit,
                         env={**os.environ, "PYTHONPATH": ROOT})
    if out.returncode: raise AssertionError(out.stderr)
    return int(out.stdout.strip())


class OpenFilesTests(unittest.TestCase):
    def setUp(self):
        hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
        if hard != resource.RLIM_INFINITY and hard <= 1024: self.skipTest("the hard limit is 1,024 or less")

    def test_the_child_starts_at_1024_without_plearn(self):
        self.assertEqual(run("import resource; print(resource.getrlimit(resource.RLIMIT_NOFILE)[0])"), 1024)

    def test_import_raises_the_soft_limit(self):
        self.assertGreater(run("import resource, plearn; print(resource.getrlimit(resource.RLIMIT_NOFILE)[0])"), 1024)

    def test_a_process_it_starts_gets_the_raised_limit(self):
        code = "import plearn, subprocess; print(subprocess.run(['bash', '-c', 'ulimit -S -n'], capture_output=True, text=True).stdout.strip())"
        self.assertGreater(run(code), 1024)


if __name__ == "__main__":
    unittest.main()
