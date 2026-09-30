# Cache and footprint by architecture

Read the model's `config.json` or GGUF metadata; these rules say which fields matter. Recurrent formulas follow mlx-lm 0.31.3 (`models/gated_delta.py`, `models/mamba2.py`); engines that keep state in another dtype differ.

| Architecture | KV/state per token | fit.py input |
|---|---|---|
| Dense MHA/GQA/MQA | 2 × attn_layers × kv_heads × head_dim × elem_bytes | `--layers --kv-heads --head-dim --kv-bytes` |
| MLA (DeepSeek-style) | compressed latent: attn_layers × (kv_lora_rank + qk_rope_head_dim) × elem_bytes, if the engine caches the latent; some engines cache decompressed K/V (use the dense formula then) | `--kv-bytes-per-token` |
| Sliding-window layers | window × per-token size for those layers, independent of ctx; only global layers grow with ctx | `--layers` = global layers only; window cost goes in `--state-gib` (charged per sequence) |
| Hybrid Gated DeltaNet / DeltaNet (Qwen3-Next family) | only attention layers grow with ctx; per recurrent layer, per sequence: Hv × d_v × d_k × 4 B (fp32 state) + (conv_kernel − 1) × (2·key_dim + value_dim) × elem_bytes (conv state) | `--layers` = attention layers only, plus `--state-gib` |
| Hybrid Mamba-2 (Nemotron-H, Granite) | per Mamba layer, per sequence: d_inner × d_state × 4 B (SSM state) + (d_conv − 1) × (d_inner + 2·n_groups·d_state) × elem_bytes (conv state) | `--layers` = attention layers only, plus `--state-gib` |
| Indexer / sparse attention | extra indexer keys per token per layer | add to `--kv-bytes-per-token` |

KV element bytes: f16/bf16 2, q8_0 1.0625, q4_0 0.5625 (block scales included).

## MoE

- Memory ≈ all experts resident (total params); decode bytes/token ≈ active params (router-selected experts + shared experts + attention + embeddings row + lm_head).
- `--params` in fit.py is the resident count. Active count predicts decode speed, not fit.
- Expert tensors dominate bytes: they are the first place a hybrid quant saves memory, and the router/shared expert are the last place to cut precision.

## GGUF quant families

- K-quants (Q2_K…Q6_K) and legacy Q4_0/Q8_0: block-scaled; effective bpw ≈ Q2_K 2.6, Q3_K 3.4, Q4_K 4.5, Q5_K 5.5, Q6_K 6.6, Q8_0 8.5.
- IQ family (IQ2_XXS…IQ4_XS): codebook, better quality per bit at ≤3 bpw, heavier dequant; check the engine's Metal support per type.
- Mixed files: the `_S/_M/_L` suffix changes which tensors are boosted. Read the per-tensor types (`gguf-dump` or llama.cpp load log) rather than trusting the label.
- An importance matrix (imatrix) matters most at ≤3 bpw.

## Engine notes that go stale — verify with `--help` / release notes

Checked 2026-09-30 against llama.cpp build 11146, mlx-lm 0.31.3, vllm-metal v0.30.0.

- llama.cpp: `-ngl` (default `auto`), `-fa` (`auto`) and `-np` (`auto`) are auto; continuous batching is on, and `-kvu` is on when slots are auto (PR #15434). Omit `-ngl`: an explicit value disables `-fit` auto-fitting. A quantized V cache (`-ctv q8_0`) forces flash attention on and errors with `-fa off`. Split files are `name-00001-of-0000N.gguf`; size is the sum of shards. On M5 the Metal 4 tensor path is on (PR #16634); `GGML_METAL_TENSOR_DISABLE=1` turns it off for an A/B.
- mlx-lm: `mlx_lm.server` batches concurrent requests (`--decode-concurrency`, `--prompt-concurrency`, `--prefill-step-size`, `--prompt-cache-size`/`--prompt-cache-bytes`) but has **no** KV quantization or `--max-kv-size`; those (`--kv-bits`, `--kv-group-size`, `--quantized-kv-start`, `--max-kv-size`) exist only in `mlx_lm.generate` and the Python API. New architectures often need git main.
- vLLM-Metal: vLLM plugin on an MLX backend; continuous batching and a paged-attention Metal kernel by default. v0.30.0 lists Qwen3/3.5+/Next, Gemma 3/4, Llama 3, GPT-OSS and hybrid Mamba-2 (some experimental); no DeepSeek V3/V4 or Qwen3-235B, GGUF for dense models only. Check `docs/supported_models.md` before recommending.
