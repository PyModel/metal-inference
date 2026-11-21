# Apple Silicon facts for inference

## Identify the actual machine

Read the chip, GPU core count, RAM, macOS build and native/Rosetta status from `scripts/envelope.sh`. Match the exact SKU against Apple's current tech-specs page; chips sharing a family name can have different GPU/memory configurations. Record the primary URL and date with any hardware figure. Do not turn a published peak into a measured sustained bandwidth value.

## Diagnose prefill and decode separately

Single-stream decode often moves weights/cache faster than it can reuse them; long or batched prefill often spends more time on matrix computation. These are starting hypotheses. Attention length, quantization kernels, dispatch overhead, SSD streaming, batching and speculative verification can change the bottleneck.

For an assumed bandwidth-limited configuration, use `BW_eff / bytes_read_per_emitted_token` as a diagnostic estimate. State the traffic model, KV reads at the tested context, active vs resident expert bytes, reuse across a batch, and accepted draft tokens. Avoid multiplying by concurrency after those savings are already in the byte denominator. An unexplained rate above the estimate means the assumptions or measurement need revisiting.

Apple's [M5 MLX study](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) compared 4096-token prompts and 128 generated tokens on specific M4/M5 systems. It found larger TTFT gains than decode gains and documents the GPU Neural Accelerators' use through Metal tensor operations. Treat its numbers as results for that workload, not promises for other chips, models or context lengths. GPU Neural Accelerators and the separate Apple Neural Engine are different execution paths; verify which backend the installed runtime actually uses.

## GPU residency and working-set budgets

CPU and GPU use the same physical memory pool, with access governed by Metal resource storage mode. Count shared buffers once in the RAM budget; check GPU residency separately. See Apple's [storage-mode documentation](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus.md).

`iogpu.wired_limit_mb`, when available, exposes a system setting. A zero value means a default is selected, not that a fixed percentage is guaranteed. A missing setting is unknown. IOAccelerator's `Alloc system memory` is a diagnostic allocation counter whose availability can vary; absence must never become zero usage. Compare counters with the runtime's load log and observed pressure.

Apple defines [`recommendedMaxWorkingSetSize`](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize.md) as an approximate threshold for avoiding performance impact. MLX exposes it through device information in supported versions. It is a planning input, not proof that an allocation will succeed or remain resident. Record explicit `fit.py --wired-cap-gib` budget sources and treat every verdict as a projection.

Use workload, cache and concurrency reductions before considering a machine-wide wired-limit change. Any system-setting experiment needs its original value, a bounded test, observed pressure, and a restoration command; never silently change a global limit to make a fit check pass.

## Thermals and repeatability

Record power source/mode, thermal state, foreground GPU users and sustained run duration. Interleave comparable configurations and repeat after warm-up; distinguish cold-start gains from steady-state gains. Use supported Instruments/Metal tooling or `powermetrics` when available and authorized. Do not assume every Mac model exposes identical samplers or privileged telemetry.
