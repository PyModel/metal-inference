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

## Engine capability checks

Recheck the installed executable/environment before using a flag. Record the version and consult [sources.md](sources.md) for the checked upstream snapshot; upstream main is evidence of capability, not proof that a local release includes it.

- **llama.cpp:** inspect model offload, auto-fitting, flash attention, KV dtype, context allocation and slot/concurrency options together. Defaults and interactions change. Budget the actual buffer sizes from the load log. Split files must contain every index from `00001` through their declared total; count and sum all shards.
- **MLX LM:** the checked server source supports `--kv-bits`, `--kv-group-size`, and `--quantized-kv-start`. Setting KV bits disables batching; requests are served serially. Quantized attention is unfused in this snapshot and materializes a score matrix proportional to prefill step × context, so lower KV bytes can still increase peak memory. Check `--prefill-step-size`, peak allocations and queueing latency together. Cache conversion applies only to cache objects implementing `to_quantized`; measure hybrid/recurrent state separately. `--decode-concurrency` and `--prompt-concurrency` matter only on a compatible batching path. Verify generate, server, and Python API options separately.
- **vLLM-Metal or another engine:** verify the exact model architecture, quant format, attention implementation, batching and platform support in that version's primary docs and a smoke test. A successful install or a similar model name does not prove compatibility.
