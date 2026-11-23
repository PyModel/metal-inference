# Run and qualify a tuning experiment

## Setup

In a writable experiment directory create PLAN.md, THEORY.md and configs/. Save the initial envelope, exact OS/chip/bin, power and thermal state, model revision/file digest, quant, tokenizer, template, decoder settings, command, engine build and kernel commit. Read earlier rejections. Select the priority metric, allowed quality delta, maximum context/answer reserve, full-length users and memory cap. Record caps with their scope.

Set S to the absolute skill directory. Choose [an existing server](../examples/attached.json), [native llama.cpp](../examples/llama.json) or [native MLX-LM](../examples/mlx.json). Copy the JSON to configs/a.json. Replace every path, ID, port and budget. Confirm flags against the actual executable. Attached mode requires a single direct loopback listener and an exact ID listed by /v1/models. Verify the active model from the launch args or load log before using attached mode. MLX can list every cached model and replace its resident model on request. Use a foreground executable for managed mode. Avoid a proxy, wrapper or launcher that forks the serving process. Only use launch arrays you have inspected. Use the engine's resolved local model path for MLX where applicable.

```sh
python3 "$S/scripts/bench.py" run configs/a.json
# After deriving one changed engine setting in b.json:
python3 "$S/scripts/bench.py" run configs/a.json configs/b.json configs/a.json
python3 "$S/scripts/bench.py" show
```

Managed arms start and gracefully stop their own server for each config. Attached arms issue requests to the existing service. Applying different engine configurations to the same attached URL requires reconfiguring it between invocations and recording the launch evidence. Changing the JSON name has no effect on the engine. Use a dedicated endpoint with no other clients. Keep your experiment directory on disk. For unattended runs use a persistent terminal session, record the PID and resume from the last complete ledger row after interruption.

## Controls and counters

- The per-user lock covers an entire queue. The harness checks AC, visible power profile, known competing model-server processes, swapout delta and host pressure. Inspect GPU activity, thermals and other apps separately. The name scan cannot detect all GPU clients. Discard arms with contention, throttling or a changed performance mode.
- Set `host_cap_gib` for **total host wired + anonymous + compressor** pages. Also set `server_cap_gib` for a per-server physical-footprint budget and `gpu_cap_gib` for a system-wide GPU allocation limit when relevant. The three series overlap. Keep each separate. Sampling starts before managed launch, continues through requests and may miss short peaks. Use the engine allocator and Instruments for transient peaks. A sampled guard is not an allocation limit. Leave headroom in fit.py.
- On breach the guard stops the workload client, closes its requests and requests graceful shutdown of the owned server. An attached server belongs to its caller. If it continues work after disconnect, use the engine's documented abort/drain operation. Check in-flight requests and memory have settled before the next run. Never force-kill a server doing prefill. A shutdown timeout reports the owned PID and stops the queue.
- Each run warms up and takes at least six samples on a fixed panel. Output throughput uses server completion tokens divided by total request time; it includes prompt service. Measure GPU decode with engine timings, never SSE chunk spacing. First output and first visible text have separate timings. Align thinking mode, sampler, token limit and cache state. Output hashes support greedy panel comparisons; token-ID parity needs engine token IDs.
- Concurrent aggregate = total completions / time from earliest start to last finish, including prompt service. The concurrent panel is distinct from the six C1 prompts. Per-stream counts, TTFT, cache indicators and total wall time are in the ledger. Check for prefix reuse across attached runs. Interpret missing cache counters as unknown. Confirm N simultaneous *engine* slots from the scheduler or trace. Parallel HTTP connections alone can queue.
- `prefill_lengths` generates unique synthetic documents of approximately each requested length. Compare **actual** prompt_tokens. The client reports prompt_tokens / TTFT, which includes scheduling and first decode. Record engine pp tok/s separately. Check cached_tokens, accounting for an unavoidable shared chat template. For warm-prefix tests reuse the exact prefix, prove a hit in server counters, and report turn-2 TTFT separately.
- Additional JSON keys include `temperature`, `extra_body` (engine-specific request settings), `max_tokens`, `reps`, `concurrency`, `timeout_s` (socket), `health_timeout_s` and `arm_timeout_s`. Compare rows by exact config SHA-256. Ledger rows record successful and failed runs. Preserve stdout/stderr in a local log. Keep secrets out of configs, prompts and exports.

## Accept or reject

### Control ledger and decision record

In a fresh experiment directory collect ten consecutive successful runs of the unchanged
control. Keep that ledger separately. Run `python3 "$S/scripts/aa.py" ledger.jsonl --metric c1_output_tps`
for end-to-end throughput, or use `--metric ttft_short_s --lower` for first-output latency.
Use the corresponding metric's noise floor; neither number measures native decode.
For a native metric, collect the same control series with the engine's own timings.

Record each candidate comparison as shown in [the qualifier's input example](../scripts/qualify.py).
Use distinct retained run IDs for two independent A/B/A triplets. Set `checks_passed` from
parity, engagement, protected metrics, resource and cleanup evidence. Run
`python3 "$S/scripts/qualify.py" trials.json`. A pass applies only to the named metric;
on failure leave the baseline selected. Older anonymous trial arrays lack the evidence
needed for this check.

Log one hypothesis and one changed lever per config. Estimate its maximum effect from the measured stage share. Accept provisionally only when B beats both neighbouring controls beyond the observed spread, no protected metric regresses by more than the agreed limit (default 3%), quality passes, and the host stayed comparable. Repeat surprising wins in reversed order, or ABBA. If comparing engines, use paired runs across the same prompts and a bootstrap 95% interval for the throughput ratio. Repeat close calls. Keep failed configs and their reason.

## Qualify the provisional winner

1. Set `qualify_context` to the supported per-sequence window and `answer_tokens` to the response reserve. The harness tests half and full context, then N simultaneous near-full requests when concurrency > 1. Each document has three independent codes at different depths. A pass requires the exact ordered answer and actual prompt_tokens between 95% and 100% of the available prompt budget. Calibrate `words_per_token` on a short prefill using this tokenizer, account for template overhead, and retry length mismatches. Check engine logs for truncation and simultaneous residency. Run multiple waves for a sustained soak.
2. For lossless claims compare deterministic outputs/token IDs on the same prompts, including reasoning and multi-turn state, against the pinned baseline. Use identical tokenization and sampler. Locate the first divergent token, compare logits, tolerances and top-two gap. Label approximate paths separately. For a quant or conversion change, run NLL/perplexity or task accuracy plus long-context retrieval under the agreed tolerance.
3. Save the qualified config as best-config.json. After three consecutive levers inside noise, record baseline, best, percent of practical ceiling, limiting trace, remaining ranked tests and costs in PLAN.md. Restore prior power settings and agreed server state. When freeing the GPU, drain, stop only owned services and verify PID, listener and GPU activity.
4. To distribute a recipe, pin model, tokenizer, conversion, launch args, engine, dependencies, OS/Xcode/SDK and kernels. Rebuild from a clean commit, compare the rendered launch/hash, rerun load, chat, reasoning, tool result round-trip, vision (where advertised), retrieval, soak and speed gates. Compare clean build and winning build between controls. Keep raw samples, build logs, failures and revisions. Redact identifiers, paths and credentials before approved sharing. Optional result cards use medians and separate clean acceptance-gate measurements from tuning runs.
