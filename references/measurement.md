# Measurement and adoption gates

## 1. Write the contract

Record the exact model/tokenizer/template revision and file hash; runtime release/commit and build flags; weight/KV precision; input and output token lengths; context limit; concurrent requests; sampler/seed/stop policy; cold/warm prefix state; machine, power mode and background load. Keep output caps, prompts and quality tasks fixed across arms. Pin a primary objective such as latency to a useful answer, with memory and quality limits that an improvement must satisfy.

Use a compact experiment record:

| Field | Required evidence |
|---|---|
| Observation | Raw phase timings, memory or profile sample |
| Hypothesis | A specific bottleneck and predicted change |
| Intervention | One factor; exact command/config difference |
| Falsifier | Result that would reject the explanation |
| Quality/memory gates | Fixed before the run |
| Decision | Keep, reject, or inconclusive, with raw results and rollback |

## 2. Separate the clocks

| Metric | Definition and constraint |
|---|---|
| Client TTFT | Monotonic time from sending the request to first nonempty token-bearing delta; record reasoning, visible text and tool-call onset separately when relevant |
| Queue time | Server admission wait, if instrumented; otherwise mark unknown rather than subtracting a guessed value |
| Prefill speed | Newly evaluated input tokens / server prefill time; cached tokens are reported separately |
| Decode speed | Target/emitted tokens / documented server decode duration, with the engine's counting convention |
| Stream cadence | Time between observed deltas; call it inter-token latency only when deltas are proven one token each |
| Request latency | Request start through complete terminal response, including queueing and transport |
| Aggregate throughput | Sum of actual output tokens across completed requests / one shared wall-time interval |
| Peak memory | Measured allocation/footprint at target context and concurrency; include swapout/pressure deltas |

Prompt tokens divided by client TTFT includes queueing, transport and other work; it is not pure prefill speed. Streaming chunks may contain multiple tokens, especially with speculation. `(tokens - 1) / (end - first_chunk)` is only a client approximation under stated assumptions. Never infer token counts from characters or substitute zero for missing usage.

Use server-native timings where available alongside client measurements. A benchmark needs a complete successful stream/finish, valid finite counters and expected output. Treat protocol errors, timeout, unexpected truncation, failed retrieval, invalid tool/JSON output or repetitive loops as failed/inconclusive runs. Preserve failures and partial timings in the report; exclude them from successful-throughput claims without hiding their frequency.

## 3. Find the constraint before tuning

Start with the actual request shape. Sweep short and target-long inputs, then one and target concurrency, within the measured memory budget. Separate cold model load, cold prefix, warm-prefix reuse and sustained runs. Profile only a representative bounded interval after reproducing the slowdown.

| Evidence | Next discriminating experiment |
|---|---|
| TTFT rises while decode stays stable | Separate queue time from prefill; compare cold/warm exact prefixes |
| Latency grows mainly with accumulated context | Inspect KV traffic/state and attention path at that context |
| Throughput improves but each user waits longer | Inspect batching/admission/queueing and per-request tail latency |
| First request slow, later requests stable | Separate load, compilation, prefix reuse and file-page warming |
| High memory peak only in prefill | Measure transient matrices/chunk size, including quantized attention workspace |
| Repeated stalls and swapout growth | Reduce resident demand before kernel tuning |
| Bandwidth estimate misses badly | Recheck traffic/reuse model, dispatch overhead, kernel support and telemetry |

## 4. Gate cache and speculation changes

Prefix reuse requires the same token sequence and compatible model revision, tokenizer/template, adapter, positional state and cache format/dtype. Check cache hit/evaluated-token counters, and compare a reused-prefix continuation with a fresh computation. Never reuse across a tenant or conversation boundary without the server's documented isolation. Stored prefix caches consume memory in addition to live request state.

For speculation, record the draft/verification configuration, proposed/accepted counts, target emissions, draft and verifier time, and extra memory. Acceptance alone is not a speedup. Compare end-to-end request latency and actual emitted-token throughput at the same context and concurrency; high acceptance can still lose to draft/verification overhead. Verify the implementation's sampling semantics and quality; exact speculative algorithms preserve the target distribution, while heuristic acceptance policies may not. Route kernel changes and strong speedup proofs to the specialized benchmarking skill.

## 5. Promote only after quality and repeatability pass

Run held-out task checks appropriate to the workload: coding tests, structured-output/schema validation, tool selection/arguments, long-context retrieval and normal conversation. Supplement with NLL/perplexity against the chosen precision reference where meaningful. A finite loss does not by itself prove unchanged task quality. Keep prompt/template and sampler fixed; compare a higher-precision reference when changing weights or KV precision.

Warm up each arm, then use balanced interleaving such as repeated ABBA/BAAB blocks; record sample order and raw times. Report sample count, median, spread and failures. Predeclare the minimum worthwhile gain; if noise overlaps it, gather more evidence or label the result inconclusive. p95/p99 from a handful of requests is not a reliable tail estimate.

Stop a bounded run on sustained pressure, swap growth, runaway output, deadline expiry or malformed telemetry. Target only the process/request started for the experiment; verify cancellation and remeasure idle memory. Keep localhost binding for local experiments and do not expose a development server as a production endpoint without a separate security review.
