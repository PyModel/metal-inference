---
name: metal-inference
description: Fit, select, configure, serve, and triage local LLMs on Apple Silicon (Metal, MLX/mlx-lm, llama.cpp/GGUF, vLLM-Metal). Use when deciding whether a model/quant/context fits unified memory, choosing a quant or engine, picking context length or concurrency, writing launch/serve commands, or diagnosing a slow, swapping, or OOM-ing local model. Kernel, roofline, or paired-A/B optimization work → apple-inference-perf. Models served by DS4 → ds4-local-llm-optimizer.
---

# Metal inference

Treat every request as a **unified-memory budget** problem: weights, KV/state cache, runtime transients, other processes, and macOS share one pool. "The file fits" answers nothing; the budget equation answers it.

## Route first

| Task | Owner |
|---|---|
| Model served by DS4 (DeepSeek V4 Flash, GLM-5.3 Flash, Qwen3.8 Flash Next on :8000) | `ds4-local-llm-optimizer` — stop here |
| Changing kernels, graphs, engine code; proving a speedup; roofline derivations | `apple-inference-perf` |
| Fit, quant/engine/context choice, launch commands, triage | this skill |

## This machine

Facts live in their sources; read them, don't restate them:
- Hardware, bandwidth (M5 Max 40-core, 614 GB/s spec), macOS 27 residency traps: `~/.claude/skills/apple-inference-perf/references/apple-silicon.md`, `gotchas.md`.
- Budget equation, MLX limits, how to measure each term: `…/apple-inference-perf/references/memory-budget.md`.
- Quant byte costs (Q4_K 4.5 bpw, MLX 4-bit g32 5.0): `…/apple-inference-perf/references/quantization.md`.
- Installed runtimes and removed ones (Ollama, oMLX, MTPLX, mlx-serve are gone — never recommend reinstalling): `~/CLAUDE.md` § Local LLM Inference.
- Weights live only under `~/models/{gguf,mlx,hf}`. `mlx` and `mlx-native` conda envs stay separate.

## Procedure

1. **Envelope.** Run `~/.claude/skills/metal-inference/scripts/envelope.sh`. Done when you have arch, RAM, wired limit, current wired/anonymous/swap, installed engine versions, and which ports are taken. Rosetta, a taken port, or an engine marked absent changes the answer — resolve it before step 4.
2. **Workload.** Pin: model + exact file/repo, architecture family, quant, target context, concurrent sequences, goal metric (quality, decode tok/s, TTFT, long context, aggregate throughput). Read `config.json` / GGUF metadata for layers, KV heads, head dim, attention type, expert count; never infer them from the model name.
3. **Budget.** Run `scripts/fit.py` with real numbers (resident size from the engine's load log beats file size, which beats params×bits — a 185 GiB Qwen3.8 GGUF runs 89.5 GiB resident under ds4 because unused tensors are never loaded; `--others-gib` from step 1). Dense attention uses the formula; MLA, sliding-window, hybrid, and recurrent models take `--kv-bytes-per-token` / `--state-gib` from the config or a measurement — [references/architectures.md](references/architectures.md). Done when every term has a number and the verdict is FIT, or you name which term to cut.
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
| Concurrency | Engine with continuous batching; budget KV × seqs | vLLM-Metal only after verifying it supports this model and is installed (it is not by default here) |
| GGUF / custom quant / portability | llama.cpp (`/opt/homebrew/bin/llama-server`, `llama-bench`) | Check Metal has a kernel for every tensor type used |
| MLX-native model, Python control | `mlx_lm` in the `mlx` (Hy3 fork) or `mlx-native` (stock main) env | Match env to architecture support |

## Triage: loads but slow, or OOM

Walk in order, stopping at the first failing check: arm64 native → GPU actually used (llama.cpp log shows Metal offload of all layers; MLX device is gpu) → intended file and quant loaded → prefill and decode reported separately → swapouts delta during the run → KV/state size at the actual context → decode vs the bandwidth bound → quant kernel path → engine features (cache, spec) → only then a smaller model. For OOM, name which budget term overflowed before changing anything.

Quality regressions: compare against a higher-precision reference with identical prompt, chat template (from tokenizer metadata, never hand-written), sampler, and stop tokens; check template, EOS, tokenizer, and rope config before blaming the quant.

## Measurement rules

- Report TTFT, prefill tok/s, decode tok/s, peak memory, and swapouts separately, with prompt/generation lengths, context, concurrency, sampler, and engine version.
- Warm up once, repeat, report the median; hold model, quant, prompt, and sampling constant across arms, or label the comparison non-equivalent.
- Label every number **measured**, **sourced**, or **estimate**. A claimed decode rate above bandwidth ÷ bytes-per-token is wrong until explained.
- Anything beyond this (paired A/B, CIs, drift control) → `apple-inference-perf`.

## Report

Machine line · workload line · budget (fit.py output) · dominant constraint · recommendation + runner-up · commands · falsifier and its result if run.
