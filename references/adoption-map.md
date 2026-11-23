# Upstream design mapped to metal-inference

The source is the [Sybil Solutions skill at fda968e](https://github.com/sybil-solutions/inference-optimisation-skill/tree/fda968e7fc1996aa23c67139c755fe541c9d4be0). The original MIT notice ships in [LICENSE-inference-optimisation](../LICENSE-inference-optimisation).

The follow-up at `d1f4e1a` contributes the [startup probe](../scripts/probe_binary.py),
[A/A noise analysis](../scripts/aa.py), listener and run identity, separate first-visible
latency, and [metric-specific qualification](../scripts/qualify.py). All run locally
from this skill; no second checkout is required.

The upstream design links each decision to evidence. A workload target determines the baseline and ceiling. The gap between them determines which phase to profile. A candidate advances only after controlled measurements and a quality check. The Mac version inserts two decisions into that sequence: whether the workload fits unified memory at the target concurrency, and whether Metal or a Core ML partition actually executes the expensive operation. A qualified configuration, rejected experiments and ranked open opportunities are the outputs.

| Upstream component | Adopted behavior | Apple implementation |
|---|---|---|
| SKILL.md and PROMPT.md | Targets, inventory, ceiling, one change, flanking controls, failure ledger, plateau, qualify, report | [Skill routes](../SKILL.md) preserve fit/triage and add tuning. [Standalone prompt](../PROMPT.md) requires a phase-by-phase opportunity inventory. |
| ceilings.md, ceiling.py and worked hardware example | Dense/MoE weight traffic, speculative verification/draft, context, N streams, compute ceiling and KV capacity | [Ceiling guide](ceilings.md), [calculator](../scripts/ceiling.py), [worked inputs](../examples/ceiling.json) and [MLX calibration](../scripts/calibrate.py). Aggregate throughput counts each emitted token once. Existing fit.py checks resident memory separately. |
| methodology.md and bench.py | Warm-up, fixed panel, medians, true token counts, client TTFT, concurrent wall time, unique prompts, long needles, watchdog, ledger | [Experiment procedure](experiments.md), [benchmark entry point](../scripts/bench.py) and [workload](../scripts/workload.py). Managed native launch and existing-server modes. Actual prompt length and exact answer both gate qualification. |
| queue.sh, mem.sh and example configs | Serialize jobs, monitor the cap, abort invalid runs, repeat a configuration reproducibly | A single queue across run configs uses macOS flock via Python. [Mac probes](../scripts/mac_guard.py) sample power, host pages, GPU allocation, server physical footprint and swap. [MLX](../examples/mlx.json), [GGUF](../examples/llama.json) and [attached](../examples/attached.json) replace the Docker/XPU example. |
| levers.md and gotchas.md | Startup caching, decode scheduling/speculation, acceptance vs cost, KV/state, chunk size, mixed workloads, kernel tuning, output sanity | [Apple opportunity table](apple-opportunities.md), [backend field notes](field-notes.md) and [Apple kernel/ANE guide](metal-ane.md). Backend-specific failures inform hypotheses. Each Apple recommendation needs its own capability check and measurement. |
| publishing.md and README | Pinned clean rebuild, load/chat/reasoning/tools/vision, retrieval, soak, clean-vs-best A/B, evidence scrub and results card | [Qualification and release procedure](experiments.md). Identify the harness and cold/warm conditions for each figure. |
