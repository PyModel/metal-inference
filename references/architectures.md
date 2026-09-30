# Cache and footprint by architecture

Read the model's `config.json` or GGUF metadata; these rules say which fields matter.

| Architecture | KV/state per token | fit.py input |
|---|---|---|
| Dense MHA/GQA/MQA | 2 × attn_layers × kv_heads × head_dim × elem_bytes | `--layers --kv-heads --head-dim --kv-bytes` |
| MLA (DeepSeek-style) | compressed latent: attn_layers × (kv_lora_rank + qk_rope_head_dim) × elem_bytes, if the engine caches the latent; some engines cache decompressed K/V (use the dense formula then) | `--kv-bytes-per-token` |
| Sliding-window layers | window × per-token size for those layers, independent of ctx; only global layers grow with ctx | split: global layers via formula, add window cost to `--runtime-gib` |
| Hybrid linear/recurrent (GDN, DeltaNet, Mamba) | only the attention layers grow with ctx; recurrent layers hold a fixed state per sequence: heads × d_k × d_v × 4 B (fp32) per layer | `--layers` = attention layers only, plus `--state-gib` |
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

- llama.cpp: full offload is the default on Metal in current builds; `-ngl` may be unnecessary. KV quant via `-ctk/-ctv` needs flash attention for V.
- mlx-lm: server and generate flags change between minor versions; prompt-cache and KV-quant options live in the installed version's `--help`.
- vLLM-Metal: separate project from CUDA vLLM; supported models and install method change often — check its repo before recommending.
