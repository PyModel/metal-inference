# Checked sources and refresh rules

Checked 2026-11-21. The installed executable's version and behavior remain authoritative for a local recommendation. Follow these primary sources for feature changes; pin the exact release/commit used in an experiment. A checked main-branch capability is not a claim about every released wheel or binary.

| Source | Checked evidence | Applies to |
|---|---|---|
| [MLX LM server guide](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/SERVER.md) | KV quant flags; serial serving with quantized KV; unfused prefill score matrix; server's stated security limitations | Long-context/cache/concurrency and local serving |
| [MLX LM server implementation](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/server.py) | Parser flags and `_is_batchable` KV condition | Capability checking on a matching version |
| [MLX LM generation implementation](https://github.com/ml-explore/mlx-lm/blob/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0/mlx_lm/generate.py) | Cache conversion is conditional on cache support; working-set warning is a heuristic | Hybrid state and memory-budget assumptions |
| [llama.cpp benchmark guide](https://github.com/ggml-org/llama.cpp/blob/e6d624b9a4b95c6b52f5ca384d70b75581f9821a/tools/llama-bench/README.md) | Separate prompt/generation benchmark modes and configuration reporting | Reproducible microbenchmarks; not a replacement for request latency |
| [llama.cpp server speed benchmark](https://github.com/ggml-org/llama.cpp/blob/e6d624b9a4b95c6b52f5ca384d70b75581f9821a/tools/server/bench/speed-bench/README.md) | Server prompt/decode timings, client latency and baseline/speculation comparison | Metric definitions and speculative A/B checks |
| [llama.cpp speculation guide](https://github.com/ggml-org/llama.cpp/blob/e6d624b9a4b95c6b52f5ca384d70b75581f9821a/docs/speculative.md) | Draft/target requirements and algorithm-specific options | Verify installed implementation rather than assuming shared semantics |
| [Apple working-set property](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize.md) | Approximate performance threshold | Distinguish a planning budget from an allocation guarantee |
| [Apple shared-memory storage modes](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus.md) | CPU/GPU share physical memory; access depends on storage mode | Avoid double-counting the same shared allocation |
| [Apple M5 MLX study](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) | Specific prefill/decode workloads and GPU Neural Accelerator path | Workload-specific evidence; no universal speed promise |

## Comparison attribution

The review of [Sybil's inference-optimisation-skill at fda968e](https://github.com/sybil-solutions/inference-optimisation-skill/tree/fda968e7fc1996aa23c67139c755fe541c9d4be0) motivated the explicit bottleneck hypothesis, experiment record and stop/quality gates. These procedures are adapted in original wording and checked against the sources above. No upstream executable code was copied. Its deployment-specific memory/watchdog assumptions and unverified numerical results are not local defaults.

## Refresh a recommendation

1. Save `--version`, `--help`, model metadata and the installed source revision where available.
2. Verify every recommended flag, model/quant path and interaction in that version. Check cache dtype vs batching/attention, context vs memory, and speculation vs sampling in particular.
3. Re-run the small functional/quality test, then the bounded target workload. Keep prior results scoped to their original runtime and hardware.
4. Update the source pin only after reading the relevant change. Record unsupported or untested features explicitly.
