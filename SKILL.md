---
name: metal-inference
description: Fit, select, configure, serve, and triage local LLMs on Apple Silicon (Metal, MLX/mlx-lm, llama.cpp/GGUF, vLLM-Metal). Use when deciding whether a model/quant/context fits unified memory, choosing a quant or engine, picking context length or concurrency, writing launch/serve commands, or diagnosing a slow, swapping, or OOM-ing local model on a Mac.
---

# Metal inference

Treat every request as a **unified-memory budget** problem: weights, KV/state cache, runtime transients, other processes, and macOS share one pool. "The file fits" answers nothing; the budget equation answers it.

## Route first

If a more specific skill is installed for the runtime in use (a particular server or engine) or for kernel-level optimization and paired A/B benchmarking, hand off to it. This skill owns fit, quant/engine/context choice, launch commands, and triage.

## References

Read the file for any area the task touches, before step 4:

| Task touches | Read |
|---|---|
| Chip bandwidth, GPU cores, Neural Accelerators, wired memory, thermals | [references/apple-silicon.md](references/apple-silicon.md) |
| Budget terms, MLX memory limits, measuring each term, swap/pressure traps | [references/memory.md](references/memory.md) |
| KV/state size by architecture, MoE resident vs active, GGUF quant families | [references/architectures.md](references/architectures.md) |

## Procedure

1. **Envelope.** Run `scripts/envelope.sh` (path relative to this skill's folder). Done when you have arch, RAM, wired limit, current wired/anonymous/swap, installed engine versions, and which ports are taken. Rosetta, a taken port, or an engine marked absent changes the answer — resolve it before step 4.
2. **Workload.** Pin: model + exact file/repo, architecture family, quant, target context, concurrent sequences, goal metric (quality, decode tok/s, TTFT, long context, aggregate throughput). Read `config.json` / GGUF metadata for layers, KV heads, head dim, attention type, expert count; never infer them from the model name.
3. **Budget.** Run `scripts/fit.py` with real numbers (resident size from the engine's load log beats file size, which beats params×bits — engines that skip unused tensors (MTP/draft heads, vision towers) or stream experts from SSD can hold far less than the file; `--others-gib` from step 1). Dense attention uses the formula; MLA, sliding-window, hybrid, and recurrent models take `--kv-bytes-per-token` / `--state-gib` from the config or a measurement — see architectures.md. Done when every term has a number and the verdict is FIT, or you name which term to cut.
4. **Dominant constraint.** Name exactly one: capacity, KV/state growth, decode bandwidth (bytes read/token ÷ ~BW_eff), prefill compute, kernel support for the quant on Metal, concurrency, thermals. Cut the term that dominates — a KV problem is never solved by dropping weight precision first.
5. **Recommendation.** One configuration, chosen with the decision table below, plus the runner-up and why it lost.
6. **Commands.** Run `<binary> --help` (or `python -m mlx_lm.server --help` inside the env) before writing any flag. Give: model acquisition, launch, readiness check, one test request, one benchmark. Nothing untested is presented as tested.
7. **Falsifier.** Give the measurement that would prove the recommendation wrong (peak memory vs prediction, swapouts delta, TTFT and decode tok/s at the target context, quality delta vs a higher-precision reference). If you can run it here, run it.

## Decision table

| Goal | Reach for first | Then |
|---|---|---|
| Quality | Highest precision whose budget leaves room for target ctx + runtime + others | Selective boosts (embeddings/output/router/attention to Q8+) proven by NLL, not by theory |
| Decode tok/s | Fewer bytes read per token: smaller quant of the same model, MoE with small active set | Speculative/MTP only with measured acceptance and net time saved |
| Long context | KV quant, architecture with small/fixed state, prefix caching, chunked prefill | Shorter operating ctx; lower weight precision last |
| TTFT on repeated prefixes | Prompt/prefix cache, measured cold vs warm | Larger prefill chunk if memory allows |
| Concurrency | Engine with continuous batching; budget KV × seqs | vLLM-Metal only after verifying it supports this model and version |
| GGUF / custom quant / portability | llama.cpp (`llama-server`, `llama-bench`) | Check Metal has a kernel for every tensor type used |
| MLX-native model, Python control | `mlx_lm` (`mlx_lm.generate`, `mlx_lm.server`) | Confirm the installed mlx-lm has the model's architecture; newer archs often need git main |

## Triage: loads but slow, or OOM

Walk in order, stopping at the first failing check: arm64 native → GPU actually used (llama.cpp log shows Metal offload of all layers; MLX device is gpu) → intended file and quant loaded → prefill and decode reported separately → swapouts delta during the run → KV/state size at the actual context → decode vs the bandwidth bound → quant kernel path → engine features (cache, spec) → only then a smaller model. For OOM, name which budget term overflowed before changing anything.

Quality regressions: compare against a higher-precision reference with identical prompt, chat template (from tokenizer metadata, never hand-written), sampler, and stop tokens; check template, EOS, tokenizer, and rope config before blaming the quant.

## Measurement rules

- Report TTFT, prefill tok/s, decode tok/s, peak memory, and swapouts separately, with prompt/generation lengths, context, concurrency, sampler, and engine version.
- Warm up once, repeat, report the median; hold model, quant, prompt, and sampling constant across arms, or label the comparison non-equivalent.
- Label every number **measured**, **sourced**, or **estimate**. A claimed decode rate above bandwidth ÷ bytes-per-token is wrong until explained.
- Machine drift within a session is real: for a close call, interleave arms (A B A B) and discard a warm-up run.

## Report

Machine line · workload line · budget (fit.py output) · dominant constraint · recommendation + runner-up · commands · falsifier and its result if run.
