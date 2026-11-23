import sys
import unittest
from pathlib import Path
import subprocess
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_binary import probe
from qualify import qualifies


class Replay(unittest.TestCase):
    def test_missing_library_stops_before_help_or_inference(self):
        failure = subprocess.CalledProcessError(134, ["engine", "--version"],
                                                stderr=b"Library not loaded: @rpath/libllama.dylib")
        with patch("probe_binary.subprocess.run", side_effect=failure) as run:
            with self.assertRaises(subprocess.CalledProcessError) as result:
                probe(sys.executable)
            self.assertIn(b"@rpath/libllama", result.exception.stderr)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][-1], "--version")

    def test_ttft_win_is_not_decode_and_duplicate_trials_do_not_qualify(self):
        trials = [dict(run_ids=[f"{i}-{j}" for j in range(3)], aba=[1, 0.7, 1.01],
                       checks_passed=True) for i in range(2)]
        r = dict(metric="warm_prefix_ttft_s", direction="lower", noise_pct=3, trials=trials)
        self.assertTrue(qualifies(r))
        self.assertFalse(qualifies(dict(r, trials=trials*2)))
        for row in trials:
            row["aba"] = [100, 101, 100]
        self.assertFalse(qualifies(dict(r, metric="decode_tps", direction="higher")))
