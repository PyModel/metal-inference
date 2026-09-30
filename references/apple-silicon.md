# Apple Silicon facts for inference

Spec figures are Apple's published peaks — verify a chip on its tech-specs page (support.apple.com) and with `system_profiler SPDisplaysDataType`. Effective bandwidth under a real decode is lower; measure it (a large streaming copy, or a known bandwidth-bound matvec) before using it as a bound.

## Memory bandwidth (spec peak, GB/s)

| Gen | Base | Pro | Max | Ultra |
|---|---|---|---|---|
| M1 | 68 | 200 | 400 | 800 |
| M2 | 100 | 200 | 400 | 800 |
| M3 | 100 | 150 | 300 (30-core GPU) / 400 (40-core) | 819 |
| M4 | 120 | 273 | 410 (32-core GPU) / 546 (40-core) | — |
| M5 | 153 | 307 | 460 (32-core GPU) / 614 (40-core) | 1200 |

Max-tier chips come in two GPU bins with different bandwidth; the largest memory configurations usually require the higher bin (M5 Max: 32-core is 36 GB only, 40-core goes to 128 GB). Check the exact bin before using a number. M5 figures: support.apple.com/en-us/126319 and /126320, apple.com/mac-studio/specs (checked 2026-09-30).

**Decode bound (batch 1):** tok/s ≤ BW_eff ÷ bytes read per token (active weights + KV read at the current context + small tensors). A claimed rate above this is wrong until explained (cache reuse, skipped work, or a measurement error).

## Prefill vs decode

- Prefill is compute-bound; decode is bandwidth-bound. Apple's MLX measurement on base M5 vs M4 (4096-token prompt): TTFT 3.3–4.1× faster, generation 1.19–1.27× — tracking the bandwidth change ([machinelearning.apple.com/research/exploring-llms-mlx-m5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)).
- M5-family GPU cores include Neural Accelerators (Metal 4 tensor ops). MLX uses them from 0.30.0 on macOS ≥ 26.2 (matmul, attention, quantized matmul); llama.cpp on M5+ since PR #16634; vllm-metal for prefill. They raise prefill and batched throughput, not single-stream decode.
- The ANE has no established advantage for LLM decode on Macs (no primary 2026 evidence; Apple points LLM prefill at the GPU accelerators). Treat it as an experiment against a GPU baseline.

## Wired memory

- GPU-resident buffers must be wired. The system cap is `sysctl iogpu.wired_limit_mb` (0 = default, about 75% of RAM on large configs, ~96 GiB on 128 GB — secondary sources; confirm with `max_recommended_working_set_size`). It bounds GPU-held memory, which `ioreg -c IOAccelerator` reports as "Alloc system memory" (envelope.sh `gpu_alloc`). Raising it (`sudo sysctl iogpu.wired_limit_mb=<MB>`) takes effect immediately, resets on reboot, and set too high can panic the machine — leave several GiB for the OS.
- Metal reports `recommendedMaxWorkingSetSize` (MLX: `mx.device_info()["max_recommended_working_set_size"]`); it is advice, not enforcement.
- A model that exceeds the wired budget still "loads" — then pages, stalls, and produces bimodal timings.

## Thermals

- Sustained runs on laptops lose a few percent over tens of minutes; 14" chassis sustain less power than 16". Separate cold, warm, and sustained numbers, and record power mode.
- `sudo powermetrics --samplers gpu_power` shows GPU power and frequency; third-party monitors exist too.
