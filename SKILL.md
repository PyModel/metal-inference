---
name: metal-inference
description: Fit, select, configure, serve, and triage local LLMs on Apple Silicon (Metal, MLX/mlx-lm, llama.cpp/GGUF, vLLM-Metal). Use when deciding whether a model/quant/context fits unified memory, choosing a quant or engine, picking context length or concurrency, writing launch/serve commands, or diagnosing a slow, swapping, or OOM-ing local model on a Mac.
---

# Metal inference

Treat every request as a **unified-memory budget** problem: weights, KV/state cache, runtime transients, other processes, and macOS share one pool. "The file fits" answers nothing; the budget equation answers it.

## Route first

If a more specific skill is installed for the runtime in use (a particular server or engine) or for kernel-level optimization and paired A/B benchmarking, hand off to it. This skill owns fit, quant/engine/context choice, launch commands, and triage. Custom Metal/MSL or TensorOps kernels, roofline-driven kernel work and speedup claims are out of scope here.

## References

Read the file for any area the task touches, before step 4:

| Task touches | Read |
|---|---|
| Chip bandwidth, GPU cores, Neural Accelerators, wired memory, thermals | [references/apple-silicon.md](references/apple-silicon.md) |
| Budget terms, MLX memory limits, measuring each term, swap/pressure traps | [references/memory.md](references/memory.md) |
| KV/state size by architecture, MoE resident vs active, GGUF quant families | [references/architectures.md](references/architectures.md) |
| Slow requests, benchmark design, cache/speculation trade-offs, quality gates | [references/measurement.md](references/measurement.md) |
| A version-sensitive feature, flag, hardware claim, or external recommendation | [references/sources.md](references/sources.md), then the installed binary's help/source |

## Procedure

1. **Envelope.** Run `scripts/envelope.sh` (path relative to this skill's folder). Done when you have arch, RAM, wired limit, current wired/anonymous/swap, installed engine versions, and which ports are taken. Rosetta, a taken port, or an engine marked absent changes the answer — resolve it before step 4.
2. **Workload.** Pin the model revision/file hash, engine version/commit, architecture, weight/cache precision, chat template, input/output lengths, target context, concurrent sequences, cache state, sampler, and power mode. Agree on the primary metric and an acceptable quality floor before tuning. Read `config.json` / GGUF metadata for layers, KV heads, head dim, attention type, expert count; never infer them from the model name.
3. **Budget.** Run `scripts/fit.py` with real numbers; it checks projected RAM use and GPU budget separately. Prefer resident bytes from an engine load log over file size or params×bits; engines can skip, stream, or repack tensors. Dense attention uses the formula; MLA, sliding-window, hybrid, and recurrent models take `--kv-bytes-per-token` / `--state-gib` from config or measurement. Record the source of explicit overrides. `UNKNOWN` or `PROVISIONAL` is not a fit approval: resolve missing measurements/default GPU budget, then validate peak use at target context and concurrency. A lower `--others-gib` never subtracts a running model's GPU allocation; stop the intended model and remeasure.
4. **Dominant constraint.** State a testable leading hypothesis: capacity, cache growth, decode bandwidth, prefill compute, kernel/dispatch overhead, queueing/concurrency, or thermals. Use phase timings and a context/concurrency sweep to distinguish them. Change one factor and retain a rollback configuration. If several constraints are plausible, measure before choosing a fix.
5. **Recommendation.** One configuration, chosen with the decision table below, plus the runner-up and why it lost.
6. **Commands.** Run `<binary> --help` (or `python -m mlx_lm.server --help` inside the env) before writing any flag; verify important feature interactions in that version's source/docs. Give: pinned model acquisition, localhost launch, readiness check, one test request, one benchmark, and owned-process cleanup. Bound requests, output tokens, concurrency, and the test duration. Nothing untested is presented as tested.
7. **Falsifier.** Give the measurement that would prove the recommendation wrong (peak memory vs prediction, swapouts delta, TTFT and decode tok/s at the target context, quality delta vs a higher-precision reference). If you can run it here, run it.

## Decision table

| Goal | Reach for first | Then |
|---|---|---|
| Quality | Highest precision whose budget leaves room for target ctx + runtime + others | Selective boosts validated by held-out NLL and representative task/tool/JSON tests |
| Decode tok/s | Fewer bytes read per token: smaller quant of the same model, MoE with small active set | Speculative/MTP only with measured acceptance and net time saved |
| Long context | Supported KV quant, architecture with small/fixed state, prefix caching, chunked prefill | Current MLX LM server KV quant disables batching and uses unfused attention; check peak prefill memory and installed support |
| TTFT on repeated prefixes | Prompt/prefix cache, measured cold vs warm | Larger prefill chunk if memory allows |
| Concurrency | Supported continuous batching; budget live KV/state and stored prefixes, then measure per-user latency and total throughput | Check feature exclusions (including MLX KV quant vs batching) and model support before changing engines |
| GGUF / custom quant / portability | llama.cpp (`llama-server`, `llama-bench`) | Check Metal has a kernel for every tensor type used |
| MLX-native model, Python control | `mlx_lm` (`mlx_lm.generate`, `mlx_lm.server`) | Confirm architecture support in a pinned release/commit; trial newer builds separately with a rollback |

## Triage: loads but slow, or OOM

Walk in order: arm64 native → intended GPU/offload configuration actually active → intended file and quant loaded → queue, prefill and decode separated → swapouts delta during the run → KV/state and transient peaks at actual context → decode vs a qualified bandwidth estimate → quant/kernel path → cache/speculation interactions. Fix a failed check and rerun the original workload before proceeding. For OOM, name which budget term overflowed before changing anything.

Quality regressions: compare against a higher-precision reference with identical prompt, chat template (from tokenizer metadata, never hand-written), sampler, and stop tokens; check template, EOS, tokenizer, and rope config before blaming the quant.

## Measurement rules

- Use the metric definitions and pass/fail gates in measurement.md. Missing usage, truncated streams, invalid output, and timeouts are failed/inconclusive runs, never zero-cost successes.
- Warm up and repeat balanced/interleaved arms; report raw samples, median and spread. Include tail latency only with enough samples and disclose the count.
- Label every number **measured**, **sourced**, or **estimate**. A bandwidth estimate is a diagnostic bound under stated traffic/reuse assumptions; batching/speculation changes bytes per emitted token.
- Keep cold load, cold prefix, warm prefix, sustained load, single-request latency, and aggregate throughput separate. Promote only a reproducible gain that passes the quality and memory gates.

## Report

Machine/workload contract · budget and unknowns · bottleneck evidence · hypothesis/change/rollback · recommendation + runner-up · raw measurement and quality results · source/version record · falsifier and its result if run.
