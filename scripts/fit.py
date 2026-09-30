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
import argparse, glob, os, re, subprocess, sys

GiB = 1 << 30
DEFAULT_WIRED_FRACTION = 0.75  # estimate of macOS default GPU cap when iogpu.wired_limit_mb=0; see references/apple-silicon.md


def kv_per_token(layers, kv_heads, head_dim, elem_bytes):
    return 2 * layers * kv_heads * head_dim * elem_bytes  # K + V


def budget(weights, kv_tok, ctx, seqs, state, runtime, others, headroom, ram):
    kv = kv_tok * ctx * seqs
    need = weights + kv + state * seqs + runtime
    room = ram - others - headroom
    return kv, need, room


def verdict(need, room, wired_now, wired_cap):
    # BEHAVIOR CHANGE: the wired cap now decides the verdict. GPU buffers must be
    # wired; the cap is shared with everything already wired (other models, macOS).
    if need > room or wired_now + need > wired_cap:
        return "NO FIT"
    return "FIT" if need <= room * 0.95 else "TIGHT"


def gguf_bytes(path):
    # BEHAVIOR CHANGE: split GGUFs (name-00001-of-00003.gguf) count every shard, not the first.
    m = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", path)
    if not m:
        return os.path.getsize(path)
    shards = glob.glob(f"{glob.escape(m.group(1))}-[0-9][0-9][0-9][0-9][0-9]-of-{m.group(2)}.gguf")
    if len(shards) != int(m.group(2)):
        sys.exit(f"split GGUF: found {len(shards)} of {int(m.group(2))} shards next to {path}")
    return sum(os.path.getsize(s) for s in shards)


def sysctl(name):
    try:
        return int(subprocess.check_output(["sysctl", "-n", name], text=True))
    except (OSError, subprocess.CalledProcessError, ValueError):
        return 0


def vm_bytes():
    out = subprocess.check_output(["vm_stat"], text=True)
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    field = lambda k: int(re.search(rf"^{re.escape(k)}:\s+(\d+)\.", out, re.M).group(1)) * page
    wired = field("Pages wired down")
    return wired, wired + field("Anonymous pages") + field("Pages occupied by compressor")


def gpu_alloc():
    # GPU-held system memory (IOAccelerator PerformanceStatistics); this, not kernel wiring, counts against iogpu.wired_limit_mb.
    try:
        out = subprocess.check_output(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], text=True)
        return sum(int(x) for x in re.findall(r'"Alloc system memory"=(\d+)', out))
    except (OSError, subprocess.CalledProcessError):
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
    a.add_argument("--headroom-gib", type=float, default=4)
    a.add_argument("--self-test", action="store_true")
    o = a.parse_args()
    if o.self_test:
        return self_test()
    if o.seqs < 1 or o.ctx < 0:
        a.error("--seqs must be >= 1 and --ctx >= 0")

    if o.resident_gib:
        weights = o.resident_gib * GiB
    elif o.gguf:
        weights = gguf_bytes(o.gguf)  # upper bound: engines may skip or stream tensors
    elif o.params and o.bpw:
        weights = o.params * o.bpw / 8
    else:
        a.error("need --resident-gib, --gguf, or --params with --bpw")

    if o.kv_bytes_per_token is not None:
        kv_tok = o.kv_bytes_per_token
    elif o.ctx and None in (o.layers, o.kv_heads, o.head_dim):
        a.error("--ctx needs --layers/--kv-heads/--head-dim (dense attention) or --kv-bytes-per-token")
    else:
        kv_tok = kv_per_token(o.layers or 0, o.kv_heads or 0, o.head_dim or 0, o.kv_bytes)

    ram = sysctl("hw.memsize")
    # BEHAVIOR CHANGE: others is measured, not a silent 24 GiB default.
    wired_vm, others_measured = vm_bytes()
    others = o.others_gib * GiB if o.others_gib is not None else others_measured
    gpu = gpu_alloc()
    wired_now = wired_vm if gpu is None else gpu
    if o.others_gib is not None:
        wired_now = min(wired_now, others)  # an explicit --others-gib excludes a model the caller will unload first
    state = o.state_gib * GiB
    kv, need, room = budget(weights, kv_tok, o.ctx, o.seqs, state, o.runtime_gib * GiB, others, o.headroom_gib * GiB, ram)
    wl = sysctl("iogpu.wired_limit_mb") * (1 << 20)
    wired_cap = wl or int(ram * DEFAULT_WIRED_FRACTION)

    g = lambda b: f"{b / GiB:.1f} GiB"
    print(f"weights     {g(weights)}" + ("  (file size = upper bound; pass --resident-gib from the load log)" if o.gguf else ""))
    print(f"kv/state    {g(kv + state * o.seqs)}  ({kv_tok / 1024:.1f} KiB/token x {o.ctx} x {o.seqs})")
    print(f"runtime     {g(o.runtime_gib * GiB)}  (estimate)")
    print(f"need        {g(need)}   room {g(room)} = RAM {g(ram)} - others {g(others)}{'' if o.others_gib is not None else ' (measured)'} - headroom {o.headroom_gib}")
    print(f"gpu wired   now {g(wired_now)} + need {g(need)} vs cap {g(wired_cap)}"
          + (" (clamped to --others-gib; upper bound)" if o.others_gib is not None else "")
          + ("" if wl else " (system default, estimated)"))
    free_for_kv = min(room, wired_cap - wired_now) - weights - o.runtime_gib * GiB - state * o.seqs
    if kv_tok and free_for_kv > 0:
        print(f"mem max ctx {int(free_for_kv / kv_tok / o.seqs)} tokens/seq at this KV dtype (memory only; model max may be lower)")
    v = verdict(need, room, wired_now, wired_cap)
    print(f"verdict     {v}")
    sys.exit(0 if v != "NO FIT" else 1)


if __name__ == "__main__":
    main()
