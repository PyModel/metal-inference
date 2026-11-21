# metal-inference

An agent skill (Claude Code, and any harness that reads `SKILL.md`) for running local LLMs on Apple Silicon: whether a model/quant/context fits unified memory, which engine and quant to use, launch commands, and triage of slow or OOM-ing models.

## Install

```bash
git clone https://github.com/PyModel/metal-inference.git ~/.claude/skills/metal-inference
```

## Scripts

- `scripts/envelope.sh` — read-only snapshot: chip, RAM, wired limit, memory in use, GPU-held memory and its processes, swap, installed engines, busy ports. Safe while a server holds the GPU.
- `scripts/fit.py` — projected memory budget against RAM and GPU residency; measures other processes itself. Missing telemetry returns `UNKNOWN`; an estimated default GPU budget returns `PROVISIONAL`. `fit.py --help`; `fit.py --self-test`.

```bash
scripts/fit.py --gguf model.gguf --ctx 32768 --layers 48 --kv-heads 8 --head-dim 128
```

The workflow records exact runtime/model versions, separates request latency from prefill/decode timings, and checks quality, peak memory and batching/cache interactions before adopting a faster configuration. Read [checked sources](references/sources.md) for version-specific evidence, including MLX server KV quantization's batching and prefill-memory trade-offs.

Validate script regressions with `python3 -m unittest discover -s tests -v`. Hardware measurements still run on the target Mac.
