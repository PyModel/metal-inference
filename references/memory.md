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

Write each term with a number before loading. Leave ≥4 GiB of headroom. The failure shows first as swapouts and memory pressure, then as stalls — rarely as a clean error.

## Predict transients

- Per-chunk activations = rows × hidden × dtype bytes per live tensor (MoE: rows × top_k for expert inputs).
- Unfused attention scores = heads × query rows × keys × 4 B — grows with context; fused/flash attention avoids it.
- Logits = rows × vocab × dtype bytes; engines that compute only the last row in prefill avoid the rest.
- A measured peak several times the prediction means queued GPU work or a retained reference, not the tensor you modelled. Explain the gap before tuning.

## Measure each term

| Term | Command |
|---|---|
| Free, wired, anonymous, compressed | `vm_stat` — pages are **16 KiB** on Apple Silicon: multiply by 16384 |
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

Setting a cache limit well below the default returns memory to the OS sooner at a small allocation cost.

## Traps

- **Leak vs transient:** sample idle memory between requests. A rising idle baseline is a leak; a flat one with high peaks is transient sizing.
- **Prompt caches count too:** stored prefixes plus live KV must stay under the KV room at all times, not only at admission. Copying a cache while its stream still decodes counts it twice.
- **mmap'd GGUF weights** are file-backed and can be evicted and re-read from SSD under pressure; the first run after a cold start is slower. Warm up before measuring.
- **Swap is not VRAM.** A model that only fits with swap is operationally unusable for decode.
