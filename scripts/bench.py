#!/usr/bin/env python3
"""Run local Mac server experiments and append results to ledger.jsonl in the working directory."""
import argparse
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import mac_guard as guard
from workload import worker


def validate(cfg):
    keys = {"name", "model", "base_url", "launch", "reps", "max_tokens", "answer_tokens", "concurrency",
            "qualify_context", "prefill_lengths", "words_per_token", "temperature", "extra_body",
            "timeout_s", "health_timeout_s", "arm_timeout_s", "host_cap_gib", "gpu_cap_gib", "server_cap_gib"}
    unknown = set(cfg) - keys
    if unknown:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")
    addr = urlsplit(cfg["base_url"])
    if (addr.scheme != "http" or addr.hostname not in {"127.0.0.1", "::1", "localhost"} or not addr.port
            or addr.username or addr.path or addr.query or addr.fragment):
        raise ValueError("base_url must be a plain loopback HTTP origin with an explicit port")
    if not isinstance(cfg["model"], str) or not cfg["model"].strip() or not cfg["name"]:
        raise ValueError("supply a run name and a model ID")
    if "launch" in cfg and not (isinstance(cfg["launch"], list) and cfg["launch"]
                                and all(isinstance(arg, str) for arg in cfg["launch"])):
        raise ValueError("launch must be a foreground executable and argv")
    for k, default, minimum in (("reps", 6, 6), ("concurrency", 1, 1), ("max_tokens", 128, 2),
                                ("answer_tokens", 128, 1), ("timeout_s", 180, 1),
                                ("health_timeout_s", 180, 1), ("arm_timeout_s", 3600, 1)):
        value = cfg.get(k, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{k} must be an integer >= {minimum}")
    lengths = cfg.get("prefill_lengths", [])
    if not isinstance(lengths, list) or any(type(v) is not int or v < 2 for v in lengths):
        raise ValueError("prefill_lengths must be a list of prompt lengths")
    if "qualify_context" in cfg and (type(cfg["qualify_context"]) is not int or
                                     cfg["qualify_context"] // 2 <= cfg.get("answer_tokens", 128)):
        raise ValueError("qualify_context must leave room for the answer at half context")
    for k in ("host_cap_gib", "gpu_cap_gib", "server_cap_gib", "words_per_token"):
        if k in cfg and (type(cfg[k]) not in (int, float) or not math.isfinite(cfg[k]) or cfg[k] <= 0):
            raise ValueError(f"{k} must be finite and positive")
    if "host_cap_gib" not in cfg:
        raise ValueError("set host_cap_gib for the whole host after inspecting its envelope")
    return addr


def listeners(port):
    r = subprocess.run(["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True, timeout=5)
    if r.returncode not in (0, 1) or r.stderr:
        raise RuntimeError("could not identify the local listener")
    return {int(pid) for pid in r.stdout.split()}


def ready(cfg):
    with urllib.request.urlopen(cfg["base_url"] + "/v1/models", timeout=2) as response:
        ids = [m["id"] for m in json.load(response)["data"]]
    if cfg["model"] not in ids:
        raise ValueError("model ID missing from /v1/models")


def run(path):
    row = {"config": Path(path).name, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ok": False}
    proc = client = receive = send = None
    try:
        raw = Path(path).read_bytes()
        cfg = json.loads(raw)
        row.update(name=cfg["name"], model=cfg["model"], config_sha256=hashlib.sha256(raw).hexdigest())
        addr = validate(cfg)
        bound = listeners(addr.port)
        if "launch" in cfg and bound:
            raise RuntimeError("launch port already occupied")
        if "launch" not in cfg and len(bound) != 1:
            raise RuntimeError("existing-server mode requires one identifiable listening PID")
        pid = next(iter(bound), None)
        state = guard.power()
        row["power_start"] = state
        if "AC Power" not in state["source"]:
            raise RuntimeError("AC power required")
        initial = guard.sample(pid)
        row["start_sample"] = initial
        caps = {"host_used_gib": cfg["host_cap_gib"]}
        for key, probe in (("gpu_cap_gib", "gpu_alloc_gib"), ("server_cap_gib", "server_footprint_gib")):
            if key in cfg:
                caps[probe] = cfg[key]

        def check():
            if proc and proc.poll() is not None:
                raise RuntimeError(f"launched server exited with status {proc.returncode}")
            others = guard.model_servers()
            others.pop(pid, None)
            if others:
                raise RuntimeError(f"other model server(s) detected: {others}")
            sample = guard.sample(pid)
            peaks = row.setdefault("sampled_peaks_gib", {})
            for key, value in sample.items():
                if key.endswith("_gib"):
                    peaks[key] = max(peaks.get(key, 0), value)
            row["last_sample"] = sample
            guard.check(sample, initial, {k: v for k, v in caps.items() if pid or k != "server_footprint_gib"})
            row["power_end"] = guard.power()
            if row["power_end"] != state:
                raise RuntimeError("power source or profile changed")

        check()
        if "launch" in cfg:
            start = time.monotonic()
            proc = subprocess.Popen(cfg["launch"], start_new_session=True)
            pid = proc.pid
            row["owned_pid"] = pid
            while time.monotonic() - start < cfg.get("health_timeout_s", 180):
                check()
                if listeners(addr.port) == {pid}:
                    try:
                        ready(cfg)
                        break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                time.sleep(0.5)
            else:
                raise RuntimeError("owned server did not become ready")
            row["startup_to_models_s"] = time.monotonic() - start
        else:
            ready(cfg)
        ctx = multiprocessing.get_context("spawn")
        receive, send = ctx.Pipe(duplex=False)
        client = ctx.Process(target=worker, args=(cfg, send))
        client.start()
        send.close()
        deadline = time.monotonic() + cfg.get("arm_timeout_s", 3600)
        while not receive.poll(1):
            check()
            if time.monotonic() > deadline or not client.is_alive():
                raise RuntimeError("workload timed out or exited without metrics")
        row.update(receive.recv())
        check()
        if any(not (r["length_ok"] and r["exact"]) for k in ("needles", "soak") for r in row.get(k, [])):
            row["error"] = "retrieval or measured context length failed"
        row["ok"] = "error" not in row
    except (Exception, KeyboardInterrupt) as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["ok"] = False
    finally:
        if client and client.pid:
            if client.is_alive():
                client.terminate()  # closes our client sockets, including concurrent requests
            client.join()
        for pipe in (receive, send):
            if pipe:
                pipe.close()
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGINT)  # request the engine's graceful shutdown
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                row["ok"] = False
                row["cleanup_error"] = f"graceful shutdown pending for owned PID {proc.pid}"
        with open("ledger.jsonl", "a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(row, indent=2))
    return row["ok"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="serialize one or more configs, e.g. A B A").add_argument("configs", nargs="+")
    commands.add_parser("show", help="summarize the local ledger")
    args = parser.parse_args()
    if args.command == "show":
        print("run\tvalid\tdecode tok/s\taggregate tok/s\terror")
        for line in Path("ledger.jsonl").read_text().splitlines():
            r = json.loads(line)
            print(r.get("name", r["config"]), r["ok"], r.get("c1_decode_tps", "-"),
                  r.get("aggregate_tps", "-"), r.get("error", r.get("cleanup_error", "-")), sep="\t")
        return
    def cancelled(signum, frame):
        raise KeyboardInterrupt("benchmark cancelled by signal")

    signal.signal(signal.SIGTERM, cancelled)
    with guard.gpu_lock():
        for config in args.configs:
            if not run(config):
                raise SystemExit(1)


if __name__ == "__main__":
    main()
