#!/usr/bin/env python3
"""Unified-memory fit check: weights + KV/state + runtime + other processes vs RAM.

  fit.py --gguf FILE --ctx 65536 --layers 48 --kv-heads 8 --head-dim 128
  fit.py --params 70e9 --bpw 4.5 --ctx 32768 --kv-bytes-per-token 327680
  fit.py --self-test

Dense GQA/MQA attention only for the --layers/--kv-heads/--head-dim formula.
MLA, sliding-window, hybrid or recurrent models: pass --kv-bytes-per-token
(and --state-gib for fixed per-sequence recurrent state) read from the model
config or measured from the running engine.
"""
import argparse, os, subprocess, sys

GiB = 1 << 30


def kv_per_token(layers, kv_heads, head_dim, elem_bytes):
    return 2 * layers * kv_heads * head_dim * elem_bytes  # K + V


def budget(weights, kv_tok, ctx, seqs, state, runtime, others, headroom, ram):
    kv = kv_tok * ctx * seqs
    need = weights + kv + state * seqs + runtime
    room = ram - others - headroom
    return kv, need, room


def sysctl(name):
    try:
        return int(subprocess.check_output(["sysctl", "-n", name], text=True))
    except Exception:
        return 0


def self_test():
    # Measured anchor (apple-inference-perf/references/memory-budget.md, 2026-09-27):
    # Qwen3.8-27B, 16 attention layers x 2 x 4 KV heads x 256 x bf16 = 64.0 KiB/token.
    assert kv_per_token(16, 4, 256, 2) == 64 * 1024
    kv, need, room = budget(90 * GiB, 64 * 1024, 262144, 1, 0, 3 * GiB, 24 * GiB, 4 * GiB, 128 * GiB)
    assert kv == 16 * GiB and need == 109 * GiB and room == 100 * GiB
    print("self-test ok")


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    w = a.add_mutually_exclusive_group()
    w.add_argument("--gguf", help="weights file; its on-disk size is used")
    w.add_argument("--params", type=float, help="parameter count, e.g. 70e9 (resident, not active, for MoE)")
    w.add_argument("--resident-gib", type=float, help="weights the engine reports resident after load (beats file size)")
    a.add_argument("--bpw", type=float, help="effective bits/weight incl. scales (Q4_K 4.5, Q8_0 8.5, MLX 4-bit g64 4.5, g32 5.0)")
    a.add_argument("--ctx", type=int, default=0)
    a.add_argument("--seqs", type=int, default=1, help="concurrent sequences holding KV")
    a.add_argument("--layers", type=int, help="attention layers holding KV")
    a.add_argument("--kv-heads", type=int)
    a.add_argument("--head-dim", type=int)
    a.add_argument("--kv-bytes", type=float, default=2, help="bytes/element: f16 2, q8_0 1.0625, q4_0 0.5625")
    a.add_argument("--kv-bytes-per-token", type=float, help="overrides the dense formula (MLA/hybrid/SWA)")
    a.add_argument("--state-gib", type=float, default=0, help="fixed recurrent state per sequence")
    a.add_argument("--runtime-gib", type=float, default=3, help="prefill transient + buffers; measure and replace")
    a.add_argument("--others-gib", type=float, default=24, help="wired + anonymous of everything else; envelope.sh")
    a.add_argument("--headroom-gib", type=float, default=4)
    a.add_argument("--self-test", action="store_true")
    o = a.parse_args()
    if o.self_test:
        return self_test()

    if o.resident_gib:
        weights = o.resident_gib * GiB
    elif o.gguf:
        weights = os.path.getsize(o.gguf)  # upper bound: engines may skip or stream tensors
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
    kv, need, room = budget(weights, kv_tok, o.ctx, o.seqs, o.state_gib * GiB,
                            o.runtime_gib * GiB, o.others_gib * GiB, o.headroom_gib * GiB, ram)
    g = lambda b: f"{b / GiB:.1f} GiB"
    print(f"weights     {g(weights)}" + ("  (file size = upper bound; pass --resident-gib from the load log)" if o.gguf else ""))
    print(f"kv/state    {g(kv + o.state_gib * GiB * o.seqs)}  ({kv_tok / 1024:.1f} KiB/token x {o.ctx} x {o.seqs})")
    print(f"runtime     {g(o.runtime_gib * GiB)}  (estimate)")
    print(f"need        {g(need)}   room {g(room)} = RAM {g(ram)} - others {o.others_gib} - headroom {o.headroom_gib}")
    wl = sysctl("iogpu.wired_limit_mb") * (1 << 20)
    if wl and weights + kv > wl:
        print(f"WIRED LIMIT weights+kv {g(weights + kv)} > iogpu.wired_limit_mb {g(wl)}")
    if kv_tok and room > weights + o.runtime_gib * GiB:
        print(f"max ctx     {int((room - weights - o.runtime_gib * GiB - o.state_gib * GiB * o.seqs) / kv_tok / o.seqs)} tokens/seq at this KV dtype")
    verdict = "FIT" if need <= room * 0.95 else "TIGHT" if need <= room else "NO FIT"
    print(f"verdict     {verdict}")
    sys.exit(0 if verdict != "NO FIT" else 1)


if __name__ == "__main__":
    main()
