# metal-inference

An agent skill (Claude Code, and any harness that reads `SKILL.md`) for running local LLMs on Apple Silicon: whether a model/quant/context fits unified memory, which engine and quant to use, launch commands, and triage of slow or OOM-ing models.

## Install

```bash
git clone https://github.com/PyModel/metal-inference.git ~/.claude/skills/metal-inference
```

## Scripts

- `scripts/envelope.sh` — read-only snapshot: chip, RAM, wired limit, memory in use, swap, installed engines, busy ports. Safe while a server holds the GPU.
- `scripts/fit.py` — memory budget and verdict. `fit.py --help`; `fit.py --self-test`.

```bash
scripts/fit.py --gguf model.gguf --ctx 32768 --layers 48 --kv-heads 8 --head-dim 128 --others-gib 14
```
