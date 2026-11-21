# Unified-memory budget

## The equation

A model stays resident, without swapping, only while

```
RAM ≥ macOS wired baseline (~5–8 GiB idle)
    + other processes' anonymous + compressed memory (browsers, editors: often 10–30 GiB)
    + weights actually resident (engine load log; not always the file size)
    + KV / recurrent state (per token × context × sequences, + stored prompt caches)
    + engine buffer pool (freed buffers the engine keeps for reuse)
    + in-flight transients (prefill activations, attention scores, logits)
```

Write each term with a number before loading. Treat 4 GiB of headroom and the script's 3 GiB runtime allowance as starting estimates; increase them when measured prefill/concurrency peaks require it. Record unknowns. Memory pressure, swapouts, stalls, or allocation errors all fail the residency gate.

CPU and GPU share one physical memory pool. GPU allocation is a separate residency constraint, not an extra copy to add on top of the same shared buffers. The script's system snapshot includes a model already running: stop it and remeasure before budgeting its replacement, or supply documented independent RAM and GPU assumptions. `--others-gib` never lowers measured GPU usage.

## Predict transients

- Per-chunk activations = rows × hidden × dtype bytes per live tensor (MoE: rows × top_k for expert inputs).
- Unfused attention scores = heads × query rows × keys × 4 B — grows with context; fused/flash attention avoids it.
- Logits = rows × vocab × dtype bytes; engines that compute only the last row in prefill avoid the rest.
- A measured peak above prediction can reflect queued GPU work, retained references, unfused attention, packing buffers, or an incomplete model of transients. Explain the gap before tuning.
- In the checked MLX LM server, quantized KV uses unfused attention and disables batching. Its prefill score matrix can outweigh KV savings; decrease prefill step only after measuring the latency/memory trade-off. See [architectures.md](architectures.md) and [sources.md](sources.md).

## Measure each term

| Term | Command |
|---|---|
| Free, wired, anonymous, compressed | `vm_stat` — parse its reported page size; do not hardcode a byte multiplier |
| Swap used, swapouts | `sysctl vm.swapusage`; `vm_stat` "Swapouts" — diff two samples across the run |
| Pressure | `sysctl kern.memorystatus_vm_pressure_level` (1 normal, 2 warn, 4 critical) or Activity Monitor |
| Process footprint | `footprint <pid>` or `vmmap --summary <pid>` |
| MLX | `mx.get_active_memory()`, `mx.get_peak_memory()` (call `mx.reset_peak_memory()` at phase start), `mx.get_cache_memory()` |
| llama.cpp | load log lines for model buffer, KV buffer, and compute buffer sizes (Metal) |

## MLX limits (verify against the installed version's source/docs)

| Limit | Controls | Does not |
|---|---|---|
| `mx.set_memory_limit` | when `eval` stops enqueueing new work | cap an individual allocation |
| `mx.set_cache_limit` | size of MLX's freed-buffer pool | affect active memory |
| `mx.set_wired_limit` | wired residency set size (must be ≤ system cap) | guarantee pages stay wired under pressure |

Evaluate a lower cache limit against both peak memory and allocation/latency cost; it does not free live tensors. Synchronize completed work when comparing phase memory, using the installed MLX API.

## Fit verdicts

`fit.py` exits 0 for a projected `FIT`/`TIGHT`, 1 for `NO FIT`, and 2 for invalid input, `UNKNOWN`, or `PROVISIONAL`. None certifies a live workload. Missing GPU statistics are unknown, not zero. A failed wired-limit probe is unknown, not the system default. When the reported limit is zero, the 75% fallback is only a planning estimate and cannot yield `FIT`; supply a sourced conservative GPU budget with `--wired-cap-gib` or resolve the device/runtime limit. Apple describes `recommendedMaxWorkingSetSize` as an approximate performance threshold, not a hard allocation guarantee.

The read-only snapshot and arithmetic cannot predict every framework allocator, reserved cache, or unified-memory residency transition. Validate a candidate gradually at the target input/output lengths and concurrency, with an owned-process stop path and memory-pressure monitoring.

## Traps

- **Leak vs retained cache:** sample idle memory between equivalent requests. A rising baseline warrants checking retained prefixes and allocator caches before calling it a leak; compare after documented cache cleanup.
- **Prompt caches count too:** stored prefixes plus live KV must stay under the KV room at all times, not only at admission. Copying a cache while its stream still decodes counts it twice.
- **mmap'd GGUF weights** are file-backed and can be evicted and re-read from SSD under pressure; the first run after a cold start is slower. Warm up before measuring.
- **Swap changes the workload.** A swap-dependent model fails a memory-resident latency objective; report it as a different operating mode with measured latency, not a fit success.
