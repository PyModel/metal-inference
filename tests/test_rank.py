import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rank import rank


class CriticalPath(unittest.TestCase):
    def test_transfer_cost_can_reverse_a_kernel_win(self):
        arm = dict(name="ANE", critical_path_ms=20, replacement_ms=8, handoff_ms=15)
        gpu = dict(arm, name="GPU", replacement_ms=18, handoff_ms=0)
        self.assertEqual(rank(dict(request_ms=50, candidates=[arm, gpu])),
                         [("GPU", 2, 4), ("ANE", -3, -6)])

    def test_rejects_accumulated_parallel_time(self):
        arm = dict(name="GPU", critical_path_ms=60, replacement_ms=20, handoff_ms=0)
        with self.assertRaises(ValueError):
            rank(dict(request_ms=50, candidates=[arm]))
