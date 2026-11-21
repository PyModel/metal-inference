"""Regression checks for unsafe memory-fit approvals; no Apple GPU is required."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fit", ROOT / "scripts" / "fit.py")
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


class FitTests(unittest.TestCase):
    def run_fit(self, extra=(), *, gpu=0, cap=98304, ram=128 * fit.GiB,
                others=16 * fit.GiB, system="Darwin", vm_error=None):
        argv = ["fit.py", "--resident-gib", "40", "--ctx", "32768",
                "--kv-bytes-per-token", "65536", *extra]
        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", argv), \
             patch.object(fit.platform, "system", return_value=system), \
             patch.object(fit.platform, "machine", return_value="arm64"), \
             patch.object(fit, "sysctl", side_effect=lambda key: ram if key == "hw.memsize" else cap), \
             patch.object(fit, "vm_bytes", return_value=(4 * fit.GiB, others), side_effect=vm_error), \
             patch.object(fit, "gpu_alloc", return_value=gpu), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = fit.main()
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue() + err.getvalue()

    def test_valid_projected_fit(self):
        code, output = self.run_fit()
        self.assertEqual(code, 0)
        self.assertIn("verdict     FIT (projection", output)

    def test_running_gpu_usage_survives_others_override(self):
        code, output = self.run_fit(["--others-gib", "1"], gpu=60 * fit.GiB)
        self.assertEqual(code, 1)
        self.assertIn("now 60.0 GiB", output)
        self.assertIn("NO FIT", output)

    def test_unknown_gpu_usage_cannot_approve(self):
        code, output = self.run_fit(gpu=None)
        self.assertEqual(code, 2)
        self.assertIn("UNKNOWN", output)

    def test_explicit_gpu_measurement_can_resolve_unknown(self):
        code, output = self.run_fit(["--gpu-used-gib", "8"], gpu=None)
        self.assertEqual(code, 0)
        self.assertIn("explicit GPU allocation", output)

    def test_failed_wired_probe_is_not_system_default(self):
        code, output = self.run_fit(cap=None)
        self.assertEqual(code, 2)
        self.assertIn("wired limit unavailable", output)

    def test_default_wired_cap_only_provisional(self):
        code, output = self.run_fit(cap=0)
        self.assertEqual(code, 2)
        self.assertIn("PROVISIONAL", output)

    def test_explicit_gpu_budget_resolves_default(self):
        code, output = self.run_fit(["--wired-cap-gib", "96"], cap=0)
        self.assertEqual(code, 0)
        self.assertIn("explicit GPU budget", output)

    def test_ram_overflow_stays_no_fit_with_estimated_cap(self):
        code, output = self.run_fit(cap=0, ram=48 * fit.GiB)
        self.assertEqual(code, 1)
        self.assertIn("NO FIT", output)

    def test_missing_ram_is_unknown(self):
        code, output = self.run_fit(ram=None)
        self.assertEqual(code, 2)
        self.assertIn("hw.memsize unavailable", output)

    def test_broken_vm_stat_is_unknown(self):
        code, output = self.run_fit(vm_error=ValueError("missing page size"))
        self.assertEqual(code, 2)
        self.assertIn("memory measurement failed", output)

    def test_non_mac_does_not_claim_local_fit(self):
        code, output = self.run_fit(system="Linux")
        self.assertEqual(code, 2)
        self.assertIn("native arm64 Python on macOS", output)

    def test_negative_and_nonfinite_inputs_are_rejected(self):
        for flag, value in [("--runtime-gib", "-20"), ("--state-gib", "nan"),
                            ("--others-gib", "-1"), ("--headroom-gib", "inf"),
                            ("--resident-gib", "-10"), ("--kv-bytes-per-token", "-1"),
                            ("--kv-bytes", "0"), ("--gpu-used-gib", "nan"),
                            ("--wired-cap-gib", "0"), ("--seqs", "0")]:
            with self.subTest(flag=flag, value=value):
                code, output = self.run_fit([flag, value])
                self.assertEqual(code, 2)
                self.assertNotIn("verdict     FIT", output)

    def test_context_omitted_is_partial(self):
        code, output = self.run_fit(["--ctx", "0"])
        self.assertEqual(code, 2)
        self.assertIn("PROVISIONAL", output)

    def test_context_omitted_preserves_known_gpu_budget_failure(self):
        for extra, cap in [(["--ctx", "0"], 98304),
                           (["--ctx", "0", "--wired-cap-gib", "96"], 0)]:
            with self.subTest(extra=extra):
                code, output = self.run_fit(extra, gpu=60 * fit.GiB, cap=cap)
                self.assertEqual(code, 1)
                self.assertIn("verdict     NO FIT", output)
                self.assertNotIn("PROVISIONAL", output)

    def test_zero_kv_requires_fixed_state(self):
        code, _ = self.run_fit(["--kv-bytes-per-token", "0"])
        self.assertEqual(code, 2)
        code, _ = self.run_fit(["--kv-bytes-per-token", "0", "--state-gib", "1"])
        self.assertEqual(code, 0)

    def test_dense_formula_and_sequence_state(self):
        self.assertEqual(fit.kv_per_token(16, 4, 256, 2), 65536)
        kv, need, _ = fit.budget(10, 2, 100, 3, 5, 7, 0, 0, 1000)
        self.assertEqual(kv, 600)
        self.assertEqual(need, 10 + 600 + 15 + 7)

    def test_split_shards_require_exact_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for i in (0, 2, 3):
                (base / f"model-{i:05d}-of-00003.gguf").write_bytes(b"x" * i)
            with self.assertRaises(ValueError):
                fit.gguf_bytes(str(base / "model-00002-of-00003.gguf"))
            (base / "model-00001-of-00003.gguf").write_bytes(b"abcd")
            self.assertEqual(fit.gguf_bytes(str(base / "model-00002-of-00003.gguf")), 9)
            with self.assertRaises(ValueError):
                fit.gguf_bytes(str(base / "model-00000-of-00003.gguf"))

    def test_gguf_directory_is_not_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                fit.gguf_bytes(tmp)

    def test_gpu_counter_absent_is_unknown_but_actual_zero_is_valid(self):
        for text, expected in [("no stats", None), ('"Alloc system memory"=0', 0),
                               ('"Alloc system memory" = 1024', 1024)]:
            with patch.object(subprocess, "check_output", return_value=text):
                self.assertEqual(fit.gpu_alloc(), expected)

    def test_probe_timeout_is_unknown(self):
        with patch.object(subprocess, "check_output", side_effect=subprocess.TimeoutExpired("probe", 10)):
            self.assertIsNone(fit.sysctl("hw.memsize"))
            self.assertIsNone(fit.gpu_alloc())

    def test_vm_stat_uses_reported_page_size(self):
        text = ('Mach Virtual Memory Statistics: (page size of 4096 bytes)\n'
                'Pages wired down: 10.\nAnonymous pages: 20.\nPages occupied by compressor: 3.\n')
        with patch.object(subprocess, "check_output", return_value=text):
            self.assertEqual(fit.vm_bytes(), (10 * 4096, 33 * 4096))


if __name__ == "__main__":
    unittest.main()
