#!/usr/bin/env python3
"""Estimate bandwidth-limited aggregate decode and practical compute-limited prefill.
Input is a measured hardware/model JSON. GB is decimal; GiB is binary.
See references/ceilings.md for assumptions and measurement instructions.
"""
import argparse
import json
import math


def validate(m):
    def number(key, minimum=0, *, positive=False, integer=False):
        value = m[key]
        if (type(value) not in (int, float) or not math.isfinite(value) or value < minimum
                or (positive and value == 0) or (integer and type(value) is not int)):
            raise ValueError(f"invalid {key}: {value}")
        return value

    for key in ("bandwidth_gbs", "peak_tflops", "active_params_b", "mfu"):
        number(key, positive=True)
    if m["mfu"] > 1:
        raise ValueError("mfu must be <= 1")
    for key in ("dense_gb", "lm_head_gb", "kv_bytes_per_token", "attn_layers", "attn_heads", "head_dim"):
        number(key)
    for key in ("experts_gb", "draft_layer_gb", "draft_head_gb", "state_traffic_gb", "extra_flops_per_token"):
        if key in m:
            number(key)
    if m.get("experts_gb") and not (number("n_experts", 1, integer=True) >= number("top_k", 1, integer=True)):
        raise ValueError("top_k must be <= n_experts")
    if not m["dense_gb"] + m["lm_head_gb"] + m.get("experts_gb", 0):
        raise ValueError("target must read weights")
    for key, floor in (("contexts", 0), ("concurrency", 1)):
        if key in m and (not m[key] or any(type(x) is not int or x < floor for x in m[key])):
            raise ValueError(f"invalid {key}")
    for s in m.get("spec", []):
        k, a = s["drafts"], s["accept"]
        if type(k) is not int or k < 0 or type(a) not in (int, float) or not 1 <= a <= k + 1:
            raise ValueError("accept is emitted tokens per verification, in [1, drafts + 1]")
        if k and ("draft_layer_gb" not in m or "draft_head_gb" not in m):
            raise ValueError("speculation requires explicit draft_layer_gb and draft_head_gb")


def decode_ceiling(m, drafts, accept, ctx=0, n=1):
    union = 1 - (1 - m["top_k"] / m["n_experts"]) ** ((drafts + 1) * n) if m.get("experts_gb") else 0
    verify = m["dense_gb"] + m["lm_head_gb"] + m.get("experts_gb", 0) * union
    verify += n * m.get("kv_bytes_per_token", 0) * ctx / 1e9 * (drafts + 1 if m.get("kv_per_verify_row") else 1)
    draft = drafts * n * (m.get("draft_layer_gb", 0) + m.get("draft_head_gb", m["lm_head_gb"]))
    verify += n * m.get("state_traffic_gb", 0)
    return m["bandwidth_gbs"] * accept * n / (verify + draft)


def prefill_ceiling(m, L):
    linear = 2 * m["active_params_b"] * 1e9
    attn = 4 * m.get("attn_layers", 0) * m.get("attn_heads", 0) * m.get("head_dim", 0) * (L / 2)
    return m["mfu"] * m["peak_tflops"] * 1e12 / (linear + attn + m.get("extra_flops_per_token", 0))


def main(path):
    with open(path) as f:
        m = json.load(f)
    validate(m)
    print(f"bandwidth {m['bandwidth_gbs']} GB/s, peak {m['peak_tflops']} TFLOPS, MFU {m['mfu']}")
    for n in m.get("concurrency", [1]):
        print(f"\nDecode ceiling, concurrency {n} (aggregate tok/s):")
        print("  drafts accept" + "".join(f"{c // 1024:>8}K" for c in m.get("contexts", [0])))
        for s in m.get("spec", [{"drafts": 0, "accept": 1.0}]):
            row = "".join(f"{decode_ceiling(m, s['drafts'], s['accept'], c, n):9.0f}" for c in m.get("contexts", [0]))
            print(f"  {s['drafts']:>6} {s['accept']:>6}{row}")
    print("\nPrefill ceiling (tok/s, practical):")
    for L in m.get("contexts", [4096]):
        print(f"  {L // 1024:>4}K {prefill_ceiling(m, L):8.0f}")
    if m.get("kv_bytes_per_token"):
        print("\nKV: {:.2f} GB per 256K-token stream".format(m["kv_bytes_per_token"] * 262144 / 1e9))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_json")
    args = parser.parse_args()
    try:
        main(args.model_json)
    except (KeyError, OSError, TypeError, ValueError) as e:
        parser.exit(2, f"invalid ceiling inputs: {e}\n")
