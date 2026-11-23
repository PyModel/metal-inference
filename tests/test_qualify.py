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
            rows = [dict(aba=samples, checks_passed=True, run_ids=list(ids))
                    for ids in ("abc", "def")]
            record = dict(metric="target", trials=rows, direction=direction, noise_pct=3)
            self.assertEqual(qualifies(record), expected)
            self.assertFalse(qualifies(dict(record, trials=rows[:1])))
            rows[1]["checks_passed"] = False
            self.assertFalse(qualifies(record))
