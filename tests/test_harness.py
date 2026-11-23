import io
import json
import os
import sys
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bench
import ceiling
import mac_guard
import workload


class CeilingTests(unittest.TestCase):
    def test_aggregate_counts_each_stream_once(self):
        m = dict(dense_gb=1, lm_head_gb=0, bandwidth_gbs=100)
        self.assertEqual(ceiling.decode_ceiling(m, 0, 1, n=4), 400)
        m["kv_bytes_per_token"] = 1000000
        self.assertEqual(ceiling.decode_ceiling(m, 0, 1, ctx=1000, n=4), 80)

    def test_routed_expert_union_and_draft_traffic(self):
        m = dict(dense_gb=1, lm_head_gb=0.25, bandwidth_gbs=100, experts_gb=4,
                 top_k=1, n_experts=4, draft_head_gb=0.1, draft_layer_gb=0.2)
        traffic = 1 + 0.25 + 4 * (1 - 0.75**2) + 0.3
        self.assertAlmostEqual(ceiling.decode_ceiling(m, 1, 1.5), 150 / traffic)

    def test_invalid_ceiling(self):
        m = json.loads((ROOT / "examples/ceiling.json").read_text())
        ceiling.validate(m)
        for key, val in (("mfu", 1.5), ("concurrency", [0]), ("spec", [{"drafts": 1, "accept": 3}])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ceiling.validate({**m, key: val})


class StreamTests(unittest.TestCase):
    cfg = {"model": "test", "base_url": "http://127.0.0.1:1"}

    def stream(self, events, done=True):
        lines = [f"data: {json.dumps(event)}\n\n" for event in events]
        if done:
            lines.append("data: [DONE]\n\n")
        return patch("workload.urllib.request.urlopen", return_value=io.BytesIO("".join(lines).encode()))

    def test_token_count_uses_usage_and_keeps_reasoning_out_of_answer(self):
        events = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning": "think "}}]},
            {"choices": [{"delta": {"content": "multi-token chunk"}}]},
            {"usage": {"prompt_tokens": 10, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 2}}},
        ]
        with self.stream(events), patch("workload.time.monotonic", side_effect=[0, 1, 3]):
            r = workload.chat(self.cfg, "hi", 8)
        self.assertEqual((r["ttft_s"], r["decode_tps"], r["cached_tokens"], r["answer"]), (1, 3, 2, "multi-token chunk"))

    def test_invalid_stream_is_never_a_speed_result(self):
        delta = {"choices": [{"delta": {"content": "answer"}}]}
        usage = {"usage": {"prompt_tokens": 1, "completion_tokens": 2}}
        for events, done in (([delta], True), ([usage], True), ([delta, usage], False), ([{"error": "fail"}], True)):
            with self.subTest(events=events, done=done), self.stream(events, done), self.assertRaises(ValueError):
                workload.chat(self.cfg, "hi", 8)

    def test_retrieval_requires_correct_answer_and_actual_context(self):
        result = {"ttft_s": 1, "cached_tokens": 0, "answer": "A B C", "prompt_tokens": 960}
        with patch("workload.document", return_value=("doc", ["A", "B", "C"])), patch("workload.chat", return_value=result):
            def check():
                return workload.context_run({"answer_tokens": 40}, 1000, 1, True)
            self.assertTrue(check()["exact"] and check()["length_ok"])
            result["answer"] = "A B C plus more"
            self.assertFalse(check()["exact"])
            result["prompt_tokens"] = 800
            self.assertFalse(check()["length_ok"])


class GuardTests(unittest.TestCase):
    def test_examples_and_guards(self):
        for name in ("attached", "mlx", "llama"):
            cfg = json.loads((ROOT / f"examples/{name}.json").read_text())
            bench.validate(cfg)
        for key, bad in (("base_url", "http://example.com:80"), ("reps", 1), ("host_cap_gib", float("nan"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate({**cfg, key: bad})

    def test_memory_caps_have_independent_scopes(self):
        initial = {"swapouts": 0, "pressure": 1, "host_used_gib": 5, "gpu_alloc_gib": 4}
        caps = {"host_used_gib": 6, "gpu_alloc_gib": 5}
        mac_guard.check(initial, initial, caps)
        for k, v in (("swapouts", 1), ("pressure", 2), ("host_used_gib", 7), ("gpu_alloc_gib", 6)):
            with self.subTest(key=k), self.assertRaises(RuntimeError):
                mac_guard.check({**initial, k: v}, initial, caps)

    def test_queue_exclusion(self):
        with mac_guard.gpu_lock(), self.assertRaises(RuntimeError), mac_guard.gpu_lock():
            self.fail("second benchmark acquired an active lock")

    def test_missing_gpu_counter_invalidates_benchmark_sample(self):
        vm = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
              "Pages wired down: 10.\nAnonymous pages: 20.\nPages occupied by compressor: 3.\n")
        with patch("mac_guard.output", return_value=vm) as probe, patch("mac_guard.gpu_alloc", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "GPU allocation probe unavailable"):
                mac_guard.sample()
            probe.assert_called_once_with("vm_stat")

    def test_identifies_native_and_interpreted_competing_jobs(self):
        ps = ("42 /env/bin/python3.12 /env/bin/mlx_lm.server --model m\n"
              "43 /bin/llama-bench -m file\n44 Python -m mlx_lm.generate --model m\n"
              "45 /bin/editor mlx_lm.server\n")
        with patch("mac_guard.output", return_value=ps):
            self.assertEqual(mac_guard.model_servers(), {42: "mlx_lm.server", 43: "llama-bench", 44: "mlx_lm.generate"})

    def test_cancel_during_startup_stops_owned_server_and_records_failure(self):
        # The fake service never binds a port. This exercises cancellation during the readiness loop.
        launcher = """
import bench, contextlib
bench.guard.gpu_lock = contextlib.nullcontext
bench.guard.model_servers = lambda: {}
bench.guard.power = lambda: {"source": "AC Power"}
bench.guard.sample = lambda pid=None: {"host_used_gib": 1, "pressure": 1, "swapouts": 0}
bench.listeners = lambda port: set()
bench.main()
"""
        child = "import os, pathlib, time; pathlib.Path('pid').write_text(str(os.getpid())); time.sleep(30)"
        cfg = {"name": "cancel", "base_url": "http://127.0.0.1:9999", "model": "test",
               "host_cap_gib": 2, "launch": [sys.executable, "-c", child]}
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "a.json").write_text(json.dumps(cfg))
            env = dict(os.environ, PYTHONPATH=str(ROOT / "scripts"), PYTHONDONTWRITEBYTECODE="1")
            with subprocess.Popen([sys.executable, "-c", launcher, "run", "a.json"], cwd=d, env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as parent:
                deadline = time.monotonic() + 5
                while not (d / "pid").exists() and parent.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.03)
                parent.terminate()
                self.assertEqual(parent.wait(timeout=5), 1)
            row = json.loads((d / "ledger.jsonl").read_text())
            self.assertFalse(row["ok"])
            self.assertIn("cancelled", row["error"])
            self.assertEqual(row["owned_pid"], int((d / "pid").read_text()))
            with self.assertRaises(ProcessLookupError):
                os.kill(row["owned_pid"], 0)

    @unittest.skipUnless(sys.platform == "darwin", "Darwin libproc")
    def test_native_pid_footprint(self):
        self.assertGreater(mac_guard.footprint(os.getpid()), 0)


if __name__ == "__main__":
    unittest.main()
