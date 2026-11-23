# Profile Metal kernels and test ANE placement

Use two controls. The first is the shipping model server at the requested workload. The second is the existing implementation of the operation under identical shapes, weights, state and precision. Preserve the fallback on unsupported devices. Inspect the actual engine source and build from its pinned revision when changing kernels.

## Metal GPU

1. Capture one prefill and several steady decode steps. In Instruments, use Metal System Trace to identify the longest GPU encoder/dispatch and CPU waits. In Xcode GPU capture, inspect shader, bindings, pipeline, dispatch geometry, counters and command buffer boundaries. Sample clocks, thermals and memory separately. Collect final latency on an uninstrumented build.
2. Write down (M,N,K), dtype, quant block/scales, tensor strides, GQA grouping and context for the hot dispatch. Single-token GEMV is often bandwidth-bound. Prompt/batched GEMM may be compute-bound. At long context attention traffic can dominate either. Compare measured bytes/s and FLOP/s to calibration. If the GPU is idle, inspect CPU tokenization, sampler, lazy scheduling or synchronization first.
3. For **MLX**, test the built-in operation (`mx.quantized_matmul` or `mx.fast.scaled_dot_product_attention`) first. A prototype in `mx.fast.metal_kernel` must use the same layout, shape tails, streams and accumulation precision. Force `mx.eval` on outputs before stopping timers. The [calibration script](../scripts/calibrate.py) contains a runnable Metal kernel and synchronized timing loop. Confirm the changed call site runs in the server.
4. For **llama.cpp**, identify the GGML op, actual GGUF block type, host dispatch selection and Metal source in the installed revision. Rebuild with identical Release options. Verify the load log, kernel capture and forced fallback A/B. Use `llama-bench --help` and separate `-p` (prompt), `-n` (generation) and `-d` (context depth) where supported. Recheck winner under llama-server at the target N.
5. Test one of tile size, simdgroups/threadgroup, split-K, memory coalescing, dequantization fusion, occupancy or temporary-buffer reuse. Use the pipeline's execution width and maximum threads. Compare direct device access against threadgroup staging. Inspect spills, alignment/tail masking, barriers, races across threadgroups and redundant reads. Stop when a faster microbenchmark leaves end-to-end time unchanged.
6. On supported GPUs, evaluate Metal Performance Primitives / Metal TensorOps for matrix work. Gate by actual OS, SDK, GPU family, layout and dtype. Compare against the installed MLX/Metal or SIMD-group path. GPU-core Neural Accelerators are part of the GPU. Their use provides no evidence of ANE placement.
7. Prove correctness before reporting throughput. Test M=1, all served batch/verify depths, odd/tail dimensions, quant zero/extreme blocks, noncontiguous inputs and representative contexts. For attention test causal mask, RoPE offsets, grouped heads and KV append/read across calls. For MoE test routing ties, repeated experts and padding. Measure NaN/Inf, max absolute/relative error, model logits and greedy sequences. Repeat tail/race cases. Include copies, unpack, synchronization and sampling in server measurements.

## Apple Neural Engine via Core ML

MLX arrays and GGUF files have no general ANE launch switch. Use a compatible Core ML package or convert a supported subgraph from the original checkpoint with the installed coremltools. GGUF block layouts may require unpacking and a separate Core ML compression pass. Record the original weights, conversion script, deployment target, compression, dtype, layout and resulting resident bytes. Treat a precision change as a quality experiment.

Start with a substantial compute-heavy partition such as an encoder or a fixed-size prefill block. Define inputs, outputs, mask and position semantics, KV/state ownership, supported shapes and fallback. Bucket sequence lengths only after measuring padding. Compile once, reuse the loaded model, warm it, and reset state per independent sequence. On supported deployment targets, test Core ML stateful KV against explicit tensor I/O. Verify sequential predictions against a known reference, including two interleaved sequences. A split that transfers full KV or logits across runtimes on every token can lose despite a fast ANE op.

```sh
xcrun coremlcompiler compile YourModel.mlpackage /tmp
swift "$S/scripts/compute-plan.swift" /tmp/YourModel.mlmodelc ane
xcrun xctrace list templates
xcrun xctrace record --template 'Core ML' --output run.trace --launch -- /absolute/path/to/model-runner
```

The helper prints MLComputePlan's preferred and supported devices for ML Program operations, or preferred placement for neural-network layers. It needs macOS 14.4+ and an appropriate SDK. A plan is a prediction. Record actual predictions in the Core ML Instruments template and inspect observed compute-unit events and fallback, where exposed. If runtime placement is unavailable, report “ANE eligible, placement unverified.”

Use one model package, real tensors and fresh MLState per sequence. Set `MLModelConfiguration.computeUnits` **before loading** each arm. Compare `.cpuOnly`, `.cpuAndGPU`, `.cpuAndNeuralEngine` (macOS 13+) and `.all`; the latter two allow fallback. Compare outputs against CPU and the original MLX/GGUF model. Measure compile/load separately from steady prefill at multiple lengths, decode with growing KV, end-to-end TTFT, peak footprint, transfers and energy. Compare the best Core ML configuration to the shipping Metal server. Keep the ANE path only for the phase, task and device where the user metric improves and quality passes.

For iPhone/iPad or a different Mac, compile for its target OS, install the benchmark runner on the physical device and profile the runner there using Xcode Instruments. Match shapes, state and precision, record the device, thermal state and power conditions, and measure sustained performance and memory pressure on device. Mac host probes apply only to the Mac. Simulation, a Mac compute plan or the advertised ANE TOPS cannot establish on-device placement or latency.

Primary API references: [MLComputePlan](https://developer.apple.com/documentation/coreml/mlcomputeplan), [MLComputeUnits](https://developer.apple.com/documentation/coreml/mlcomputeunits), [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html), [Apple's M5 MLX measurements](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).
