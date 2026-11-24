import unittest, numpy as np
from plearn.method.trainer import own_w
class T(unittest.TestCase):
    def test_teacher_row_dropped(self):
        rows = [tuple(range(9))] * 64 + [tuple(range(9)) + (True,)]
        w = np.arange(65, dtype=float)
        o = own_w(w, rows, 4); self.assertEqual(o.shape, (64,)); self.assertEqual(o[-1], 63.0); o.reshape(-1, 4)
    def test_no_teacher_row(self):
        rows = [tuple(range(9))] * 64; w = np.ones(64)
        self.assertEqual(own_w(w, rows, 4).shape, (64,))
    def test_length_mismatch_trims(self):
        self.assertEqual(own_w(np.ones(65), [tuple(range(9))] * 3, 4).shape, (64,))
if __name__ == "__main__": unittest.main()
