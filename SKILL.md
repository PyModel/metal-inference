---
name: metal-inference
description: >
  Fit, serve, profile and optimize local AI models on Apple Silicon. Use for Mac unified-memory
  budgets, MLX/mlx-lm, GGUF/llama.cpp, speculative decoding, KV cache, prefill, decode tok/s,
  TTFT, concurrency, speed ceilings, Metal/MSL kernels, M5 TensorOps and Core ML Neural Engine
  (ANE) placement on Mac or target Apple devices. Audit performance opportunities, benchmark
  alternatives under memory and quality constraints, and qualify a reproducible configuration.
---

# Metal inference

## Choose the route

- **Fit, select, launch or triage.** Use the short procedure below. Treat weights, KV/state, transients, other processes and macOS as one unified-memory budget.
- **Audit, tune or compare.** Run the fit check, then follow the optimization loop below. Own both prefill and decode through the engine, Metal dispatch and end-to-end measurement. Use an installed runtime-specific skill for its engine controls where helpful.
- **Metal kernels or ANE.** Start from a shipping GPU baseline. Identify the phase, operation, shape and limiting resource. For ANE candidates, compare Core ML compute policies and distinguish planned from observed device placement.
- **Another agent or harness.** Give it the [standalone optimization prompt](PROMPT.md). The [design map](references/adoption-map.md) traces the upstream workflow to this skill.

## References

Load the references for the selected route.

| Task touches | Read |
|---|---|
| Chip bandwidth, GPU cores, Neural Accelerators, wired memory, thermals | [references/apple-silicon.md](references/apple-silicon.md) |
| Budget terms, MLX memory limits, measuring each term, swap/pressure traps | [references/memory.md](references/memory.md) |
| KV/state size by architecture, MoE resident vs active, GGUF quant families | [references/architectures.md](references/architectures.md) |
| Slow requests, benchmark design, cache/speculation trade-offs, quality gates | [references/measurement.md](references/measurement.md) |
| A version-sensitive feature, flag, hardware claim, or external recommendation | [references/sources.md](references/sources.md), then the installed binary's help/source |
| Measured bandwidth, compute, aggregate decode math, MoE/speculative ceilings | [references/ceilings.md](references/ceilings.md) |
| Opportunity discovery by phase and engine, historical lever evidence | [Apple opportunity inventory](references/apple-opportunities.md), [field notes](references/field-notes.md) |
| A/B harness, Mac caps, controls, plateau, qualification and publishing | [references/experiments.md](references/experiments.md) |
| MLX or GGML Metal kernels, TensorOps, Core ML partitioning or ANE placement | [references/metal-ane.md](references/metal-ane.md) |

## Autonomous tuning and resumption

Use the named engine and installed pack, one request at a time, existing sampler and representative token lengths from local history. Record lengths without copying private prompts. When history is unavailable use a short and a long synthetic fixture within the supported context. Preserve capability. Report cold prefill, warm-prefix TTFT and decode independently. Ask only when the decision materially affects the outcome or capability.

In the experiment notes record each question, its answer and the decision it settles. Reuse that answer on resume. “Proceed” resumes authorized steps. Reopen a question only when new evidence changes the next action. Apply the target repository's build/load gate before any guarded command, including calibration. For a block save the action, clearing evidence, independent work completed, working directory and exact next command. Finish dependency inspection, preserve baseline files, prepare fixtures and validate CPU-only launch paths once. Claim queued work only with an actual runner ID.

From the experiment directory with the isolated launch environment, verify binary and shard hashes against the baseline manifest, every shard referenced by the index, and the engine's effective settings path and values. Inspect Mach-O imports and rpaths, then execute both version and help there. Stop at the first loader or settings failure. Report files present, executable starts, and model serves as separate milestones. Only claim serving after a bounded successful inference.

For PROVISIONAL fit record the estimated default wired cap, the device's recommended working set and observed peak allocation separately. After repository clearance, permit a single short-context smoke request only when projected weights, state and transients fit both conservative host and GPU caps with reserve, initial pressure is normal, and working probes can abort the owned process. Set explicit time, context and output limits. Stop on any new swapout, pressure above normal, missing probe or cap breach. Record sampled peaks and repeat at increasing lengths before qualifying target context. Otherwise name the missing measurement or exceeded cap. Never change the wired limit automatically.

Inspect the checkout and its history for existing fixes first. For Inkling compare preserved and rebuilt binaries on identical 32K greedy inputs using the native harness. Check decode tok/s, draft engagement, accepted tokens per round and greedy parity. On a fixed binary separately compare the 128K profile for prefix hits and repeat-turn TTFT. Treat upstream M5 Ultra measurements as leads for the M5 Max.

Automate each comparison with the native harness in isolated prefixes and settings. Hash both executables, use matched warmups and interleaved controls, save raw rows, actual feature counters, output checks, peak memory and swapout deltas. Apply the qualification rule in references/experiments.md to each metric independently. Keep the original executable and settings selected throughout trials. A failed, inconclusive or interrupted comparison thus rolls back on exit. Promote only a replicated beyond-noise gain with all protected checks passing.

## Procedure

1. **Envelope.** Run `scripts/envelope.sh` (path relative to this skill's folder). Done when you have arch, RAM, wired limit, current wired/anonymous/swap, installed engine versions, and which ports are taken. Rosetta, a taken port, or an engine marked absent changes the answer — resolve it before step 4.
2. **Workload.** Pin the model revision/file hash, engine version/commit, architecture, weight/cache precision, chat template, input/output lengths, target context, concurrent sequences, cache state, sampler, and power mode. State the primary metric and preserve the existing quality floor. Read `config.json` / GGUF metadata for layers, KV heads, head dim, attention type, expert count; never infer them from the model name.
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

## Optimization loop

1. Record the objective, quality tolerance, target context and concurrency, memory cap and scope. Preserve the initial environment and launch. Use a writable experiment directory for PLAN.md, THEORY.md, configs/ and ledger.jsonl.
2. Measure the baseline and calculate practical ceilings. Use the installed MLX Python with `scripts/calibrate.py` for synchronized GPU copy/GEMM, then `python3 scripts/ceiling.py model.json`. Attribute each ceiling input to a measurement, model tensor or stated assumption.
3. Inventory opportunities for startup, prefill, decode at short/long context, KV/state, concurrency, Metal and ANE. Trace real dispatch and shape before proposing custom code. For each candidate log evidence, time share, predicted bound, cost, risk and falsifier. Rank candidates against the user's target.
4. Change one lever in the current best config. Run `python3 "$S/scripts/bench.py" run configs/a.json configs/b.json configs/a.json` from the experiment directory with S set to the absolute skill directory. Follow the experiment guide for native launch, attached mode and watchdog semantics. Keep raw samples, failures and reasons. A faster kernel alone does not establish a faster model.
5. Require a candidate to beat both controls beyond measured noise while protecting secondary metrics (default 3% regression limit). Repeat surprising results. Gate the provisional winner on exact half/full-context needles, concurrent soak at N, output parity or agreed quality tests and peak memory. Check actual scheduler occupancy.
6. After three consecutive levers inside noise, record a plateau with remaining opportunities and their cost. Store best-config.json only for a qualified winner. For a distributable recipe, rebuild from pins and re-run the feature and performance gates on that build.

## Measurement rules

- Use the metric definitions and pass/fail gates in measurement.md. Missing usage, truncated streams, invalid output, and timeouts are failed/inconclusive runs, never zero-cost successes.
- Warm up and repeat balanced/interleaved arms; report raw samples, median and spread. Include tail latency only with enough samples and disclose the count.
- Label every number **measured**, **sourced**, or **estimate**. A bandwidth estimate is a diagnostic bound under stated traffic/reuse assumptions; batching/speculation changes bytes per emitted token.
- Keep cold load, cold prefix, warm prefix, sustained load, single-request latency, and aggregate throughput separate. Promote only a reproducible gain that passes the quality and memory gates.

## Report

For tuning, lead with the metric, baseline, qualified winner, practical ceiling and measurement method. Include the limiting evidence, raw results, remaining ranked opportunities, rejected changes, rollback, exact configs, verification gaps and final machine state.

For fit or triage, give the machine and workload, budget and unknowns, dominant constraint, recommendation and runner-up, reproducible commands, source versions and the measurement that would change the result.
