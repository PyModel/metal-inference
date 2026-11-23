![Metal Inference](assets/banner.svg)

[![Agent Skill](assets/badge-skill.svg)](SKILL.md) [![Apple Silicon](assets/badge-platform.svg)](references/apple-silicon.md) [![Python 3](assets/badge-python.svg)](scripts/)

An agent skill for fitting and accelerating local AI models on Apple Silicon. It covers MLX-LM, llama.cpp/GGUF, Metal kernels, and Core ML experiments on the Apple Neural Engine. Use it to answer a fit question, identify what limits prefill or decode, or qualify a faster configuration for a particular device and workload.

## Install

From the project where you use your coding agent, run the skills.sh CLI with Node.js and npm installed.

```sh
npx skills add PyModel/metal-inference --skill metal-inference
```

Select your agent when prompted. Add `--agent codex` or `--agent claude-code` to select it explicitly, or `--global` to make the skill available across projects. The CLI discovers the repository's root `SKILL.md` and installs its companion scripts and references. [PyModel on skills.sh](https://skills.sh/pymodel) · [CLI documentation](https://skills.sh/docs)

## Use with any agent

Ask your agent to “Use metal-inference to find what limits prefill and decode for my local model on this Mac.” Include the model path, engine, target context and most important metric if you know them. For an agent that reads files directly, use the [skill procedure](SKILL.md) or the [full tuning prompt](PROMPT.md).

## Run locally

From the skill directory:

```sh
scripts/envelope.sh
scripts/fit.py --gguf model.gguf --ctx 32768 --layers 48 --kv-heads 8 --head-dim 128
python3 scripts/ceiling.py examples/ceiling.json
```

The envelope reports host memory, GPU allocation, active engines and ports. Use actual model metadata for fit. Missing telemetry produces `UNKNOWN`, and an estimated GPU budget produces `PROVISIONAL`. Replace illustrative ceiling inputs with measurements. The benchmark harness requires macOS, Python 3.9+ and a local OpenAI-compatible streaming server. MLX calibration uses the Python environment containing MLX. The Core ML plan inspector needs Swift and a macOS 14.4+ SDK/runtime.

| Goal | Start here |
|---|---|
| Reproduce a controlled MLX or GGUF benchmark | [Experiment procedure and configuration](references/experiments.md) |
| Find missed opportunities in prefill, decode, scheduling or memory | [Apple opportunity inventory](references/apple-opportunities.md) |
| Test a kernel, TensorOps or an ANE partition | [Metal and ANE playbook](references/metal-ane.md) |
| Verify version-specific engine behavior | [Checked sources and runtime versions](references/sources.md) |
| Check the fit and benchmark tools | `scripts/fit.py --self-test` and `python3 -B -m unittest discover -s tests` |

The [design map](references/adoption-map.md) records how the upstream inference-optimisation workflow was adopted. Adapted materials retain the [upstream MIT notice](LICENSE-inference-optimisation).
