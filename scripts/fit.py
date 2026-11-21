#!/usr/bin/env python3
"""Unified-memory fit check: weights + KV/state + runtime + other processes vs RAM and the GPU wired cap.

  fit.py --gguf FILE --ctx 65536 --layers 48 --kv-heads 8 --head-dim 128
  fit.py --params 70e9 --bpw 4.5 --ctx 32768 --kv-bytes-per-token 327680
  fit.py --self-test

Dense GQA/MQA attention only for the --layers/--kv-heads/--head-dim formula.
MLA, hybrid or recurrent models: pass --kv-bytes-per-token read from the model
config or measured from the running engine. Anything that does not grow with
ctx (recurrent state, sliding-window layers' window KV) goes in --state-gib,
which is charged once per sequence.
"""
import argparse, math, os, platform, re, subprocess, sys

GiB = 1 << 30
DEFAULT_WIRED_FRACTION = 0.75  # planning estimate only; never sufficient for a FIT verdict
PROBE_TIMEOUT = 10


def kv_per_token(layers, kv_heads, head_dim, elem_bytes):
    return 2 * layers * kv_heads * head_dim * elem_bytes  # K + V


def budget(weights, kv_tok, ctx, seqs, state, runtime, others, headroom, ram):
    kv = kv_tok * ctx * seqs
    need = weights + kv + state * seqs + runtime
    room = ram - others - headroom
    return kv, need, room


def verdict(need, room, wired_now, wired_cap):
    # These are separate constraints, not additive CPU and GPU memory pools.
    if need > room or wired_now + need > wired_cap:
        return "NO FIT"
    return "FIT" if need <= room * 0.95 else "TIGHT"


def gguf_bytes(path):
    if not os.path.isfile(path):
        raise ValueError(f"GGUF path is not a regular file: {path}")
    m = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf", path)
    if not m:
        return os.path.getsize(path)
    count = int(m.group(3))
    if count < 1 or not 1 <= int(m.group(2)) <= count:
        raise ValueError("split GGUF: invalid shard index or total")
    shards = [f"{m.group(1)}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]
    if any(not os.path.isfile(s) for s in shards):
        raise ValueError(f"split GGUF: require every shard 00001 through {count:05d} next to {path}")
    return sum(os.path.getsize(s) for s in shards)


def sysctl(name):
    try:
        return int(subprocess.check_output(["sysctl", "-n", name], text=True, timeout=PROBE_TIMEOUT))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def vm_bytes():
    out = subprocess.check_output(["vm_stat"], text=True, timeout=PROBE_TIMEOUT)
    page_match = re.search(r"page size of (\d+) bytes", out)
    if not page_match or int(page_match.group(1)) <= 0:
        raise ValueError("vm_stat page size missing or invalid")
    page = int(page_match.group(1))
    def field(key):
        match = re.search(rf"^{re.escape(key)}:\s+(\d+)\.", out, re.M)
        if not match:
            raise ValueError(f"vm_stat missing {key}")
        return int(match.group(1)) * page
    wired = field("Pages wired down")
    return wired, wired + field("Anonymous pages") + field("Pages occupied by compressor")


def gpu_alloc():
    # GPU-held system memory (IOAccelerator PerformanceStatistics); this, not kernel wiring, counts against iogpu.wired_limit_mb.
    try:
        out = subprocess.check_output(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], text=True, timeout=PROBE_TIMEOUT)
        values = re.findall(r'"Alloc system memory"\s*=\s*(\d+)', out)
        return sum(int(x) for x in values) if values else None
    except (OSError, subprocess.SubprocessError):
        return None


def self_test():
    # Dense GQA anchor: 16 attention layers x (K+V) x 4 KV heads x 256 head_dim x bf16
    # = 64.0 KiB/token, the size a live MLX server reported for such a model.
    assert kv_per_token(16, 4, 256, 2) == 64 * 1024
    kv, need, room = budget(90 * GiB, 64 * 1024, 262144, 1, 0, 3 * GiB, 24 * GiB, 4 * GiB, 128 * GiB)
    assert kv == 16 * GiB and need == 109 * GiB and room == 100 * GiB
    assert verdict(need, room, 0, 128 * GiB) == "NO FIT"
    assert verdict(90 * GiB, 100 * GiB, 0, 96 * GiB) == "FIT"
    assert verdict(90 * GiB, 100 * GiB, 10 * GiB, 96 * GiB) == "NO FIT"  # wired cap already partly used
    assert verdict(97 * GiB, 100 * GiB, 0, 128 * GiB) == "TIGHT"
    print("self-test ok")


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    w = a.add_mutually_exclusive_group()
    w.add_argument("--gguf", help="weights file (first shard of a split file is enough); on-disk size is used")
    w.add_argument("--params", type=float, help="parameter count, e.g. 70e9 (resident, not active, for MoE)")
    w.add_argument("--resident-gib", type=float, help="weights the engine reports resident after load (beats file size)")
    a.add_argument("--bpw", type=float, help="effective bits/weight incl. scales (Q4_K 4.5, Q8_0 8.5, MLX 4-bit g64 4.5, g32 5.0)")
    a.add_argument("--ctx", type=int, default=0)
    a.add_argument("--seqs", type=int, default=1, help="concurrent sequences holding KV")
    a.add_argument("--layers", type=int, help="attention layers whose KV grows with ctx")
    a.add_argument("--kv-heads", type=int)
    a.add_argument("--head-dim", type=int)
    a.add_argument("--kv-bytes", type=float, default=2, help="bytes/element: f16 2, q8_0 1.0625, q4_0 0.5625")
    a.add_argument("--kv-bytes-per-token", type=float, help="overrides the dense formula (MLA/hybrid)")
    a.add_argument("--state-gib", type=float, default=0, help="fixed per-sequence cache: recurrent state + sliding-window KV")
    a.add_argument("--runtime-gib", type=float, default=3, help="prefill transient + buffers; measure and replace")
    a.add_argument("--others-gib", type=float, help="wired + anonymous + compressed of everything else; default: measured now via vm_stat")
    a.add_argument("--gpu-used-gib", type=float, help="explicit measured GPU allocation; never inferred from --others-gib")
    a.add_argument("--wired-cap-gib", type=float, help="explicit GPU budget from a verified device/runtime limit; record its source")
    a.add_argument("--headroom-gib", type=float, default=4)
    a.add_argument("--self-test", action="store_true")
    o = a.parse_args()
    if o.self_test:
        return self_test()
    if o.seqs < 1 or o.ctx < 0:
        a.error("--seqs must be >= 1 and --ctx >= 0")
    positive = ("params", "bpw", "resident_gib", "kv_bytes", "wired_cap_gib")
    nonnegative = ("state_gib", "runtime_gib", "others_gib", "headroom_gib", "gpu_used_gib", "kv_bytes_per_token", "layers", "kv_heads", "head_dim")
    for name in positive + nonnegative:
        value = getattr(o, name)
        if value is not None and (not math.isfinite(value) or value < 0 or (name in positive and value == 0)):
            a.error(f"--{name.replace('_', '-')} must be finite and {'positive' if name in positive else 'nonnegative'}")

    if o.resident_gib is not None:
        weights = o.resident_gib * GiB
    elif o.gguf:
        try:
            weights = gguf_bytes(o.gguf)  # estimate: engines may skip, stream, or repack tensors
        except (OSError, ValueError) as exc:
            a.error(str(exc))
    elif o.params and o.bpw:
        weights = o.params * o.bpw / 8
    else:
        a.error("need --resident-gib, --gguf, or --params with --bpw")
    if not math.isfinite(weights) or weights <= 0:
        a.error("weights must be positive and finite")

    if o.kv_bytes_per_token is not None:
        kv_tok = o.kv_bytes_per_token
    elif o.ctx and None in (o.layers, o.kv_heads, o.head_dim):
        a.error("--ctx needs --layers/--kv-heads/--head-dim (dense attention) or --kv-bytes-per-token")
    else:
        if o.ctx and (not o.layers or not o.kv_heads or not o.head_dim):
            a.error("dense KV dimensions must be positive; use --kv-bytes-per-token 0 with --state-gib for fixed state")
        kv_tok = kv_per_token(o.layers or 0, o.kv_heads or 0, o.head_dim or 0, o.kv_bytes)
    if o.ctx and kv_tok == 0 and o.state_gib == 0:
        a.error("zero KV growth requires a positive --state-gib for fixed-state models")
    if not math.isfinite(kv_tok):
        a.error("KV size must be finite")

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("verdict     UNKNOWN: run with native arm64 Python on macOS", file=sys.stderr)
        return 2
    ram = sysctl("hw.memsize")
    if ram is None or ram <= 0:
        print("verdict     UNKNOWN: hw.memsize unavailable", file=sys.stderr)
        return 2
    try:
        others = o.others_gib * GiB if o.others_gib is not None else vm_bytes()[1]
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"verdict     UNKNOWN: memory measurement failed: {exc}", file=sys.stderr)
        return 2
    wired_now = o.gpu_used_gib * GiB if o.gpu_used_gib is not None else gpu_alloc()
    if wired_now is None:
        print("verdict     UNKNOWN: GPU allocation unavailable; measure it or provide --gpu-used-gib", file=sys.stderr)
        return 2
    state = o.state_gib * GiB
    kv, need, room = budget(weights, kv_tok, o.ctx, o.seqs, state, o.runtime_gib * GiB, others, o.headroom_gib * GiB, ram)
    wl = sysctl("iogpu.wired_limit_mb") if o.wired_cap_gib is None else None
    if o.wired_cap_gib is None and (wl is None or wl < 0):
        print("verdict     UNKNOWN: wired limit unavailable; provide a verified --wired-cap-gib", file=sys.stderr)
        return 2
    cap_estimated = o.wired_cap_gib is None and wl == 0
    wired_cap = (o.wired_cap_gib * GiB if o.wired_cap_gib is not None else
                 int(ram * DEFAULT_WIRED_FRACTION) if cap_estimated else wl * (1 << 20))
    if not all(math.isfinite(x) for x in (need, room, wired_now, wired_cap)):
        a.error("budget arithmetic overflowed; check inputs")

    g = lambda b: f"{b / GiB:.1f} GiB"
    print(f"weights     {g(weights)}" + ("  (file-size estimate; prefer --resident-gib from the load log)" if o.gguf else ""))
    print(f"kv/state    {g(kv + state * o.seqs)}  ({kv_tok / 1024:.1f} KiB/token x {o.ctx} x {o.seqs})")
    print(f"runtime     {g(o.runtime_gib * GiB)}  (estimate)")
    print(f"need        {g(need)}   room {g(room)} = RAM {g(ram)} - others {g(others)}{'' if o.others_gib is not None else ' (measured)'} - headroom {o.headroom_gib}")
    print(f"gpu wired   now {g(wired_now)} + need {g(need)} vs cap {g(wired_cap)}"
          + (" (explicit GPU allocation)" if o.gpu_used_gib is not None else "")
          + (" (system default, 75% planning estimate only)" if cap_estimated else "")
          + (" (explicit GPU budget)" if o.wired_cap_gib is not None else ""))
    free_for_kv = min(room, wired_cap - wired_now) - weights - o.runtime_gib * GiB - state * o.seqs
    if kv_tok and free_for_kv > 0:
        print(f"mem max ctx {int(free_for_kv / kv_tok / o.seqs)} tokens/seq at this KV dtype (memory only; model max may be lower)")
    v = verdict(need, room, wired_now, wired_cap)
    if need <= room and (cap_estimated or (o.ctx == 0 and v != "NO FIT")):
        print(f"verdict     PROVISIONAL (estimated GPU budget or context omitted; projected {v})")
        return 2
    print(f"verdict     {v} (projection; validate peak memory at target load)")
    return 0 if v != "NO FIT" else 1


if __name__ == "__main__":
    sys.exit(main())
