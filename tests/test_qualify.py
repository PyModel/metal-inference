import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qualify import qualifies


class Qualification(unittest.TestCase):
    def test_cache_gain_is_separate_from_decode(self):
        for direction, samples, expected in (
            ("lower", [1, .7, 1.01], True),
            ("higher", [100, 101, 101], False),
        ):
            row = {"aba": samples, "checks_passed": True}
            self.assertEqual(qualifies([row, row], direction), expected)
            self.assertFalse(qualifies([row], direction))
            self.assertFalse(qualifies([row, dict(row, checks_passed=False)], direction))
